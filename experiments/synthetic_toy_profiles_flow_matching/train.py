"""Training script for Synthetic Toy Profiles Flow Matching models.

This module implements unguided flow matching for training neural networks
to generate synthetic travel time profiles (computed from 2-layer toy Vs profiles)
using ODE-based generative modeling.
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

try:
    from . import config as cfg_mod
    from . import models as models_mod
    from . import utils as utils_mod
    from .data import create_dataloader
except ImportError:  # fallback when running as script
    import config as cfg_mod
    import models as models_mod
    import utils as utils_mod

    from data import create_dataloader  # type: ignore

# Global wandb variable
wandb = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_ffm_step(
    model, optimizer, batch, config, discriminator=None, discriminator_optimizer=None
):
    """
    Single FFM training step for breakpoints.

    Args:
        model: Neural field model (MLP)
        optimizer: Adam optimizer for generator
        batch: Real data breakpoints (Batch, 3) - [depth1, tts1, tts_end]
        config: Configuration object with loss_type ("l1" or "l2")
        discriminator: Optional discriminator model for GAN loss
        discriminator_optimizer: Optional optimizer for discriminator

    Returns:
        loss: Training loss value (L1 or L2 based on config.loss_type)
    """
    device = config.device

    # u1: Real data breakpoints (t=1) - shape (Batch, 3)
    if batch.dim() == 3:
        u1 = batch.squeeze(1).to(device)  # (Batch, 3)
    else:
        u1 = batch.to(device)  # (Batch, 3)

    # u0: Noise breakpoints (t=0)
    u0 = torch.randn_like(u1).to(device)

    # t: Random time from [0, 1]
    t = torch.rand(u1.shape[0], 1).to(device)

    # Interpolate breakpoints
    ut = (1 - t) * u0 + t * u1  # (Batch, 3)

    # target_v: The target vector field using proper flow matching
    target_v = torch.where(
        t < 0.999,  # Avoid division by zero
        (u1 - ut) / torch.clamp(1 - t, min=1e-6),
        torch.zeros_like(u1),
    )

    # Forward pass - model returns (Batch, 3)
    predicted_v = model(ut, t)

    # Loss calculation - L2 (MSE) - always use L2 for breakpoints
    reconstruction_loss = torch.mean((predicted_v - target_v) ** 2)

    # Apply reconstruction loss weight
    reconstruction_loss = reconstruction_loss * config.reconstruction_loss_weight

    total_loss = reconstruction_loss

    # Get predicted breakpoints at t=1 (used for both monotonicity and GAN)
    u1_predicted = ut + predicted_v * torch.clamp(1 - t, min=1e-6)

    # Add monotonicity constraint (positive slopes)
    # For breakpoints [depth1, tts1, tts_end]:
    # - Slope 1: from (0, 0) to (depth1, tts1) -> requires tts1 > 0
    # - Slope 2: from (depth1, tts1) to (500, tts_end) -> requires tts_end > tts1
    monotonicity_loss = None
    if config.monotonicity_weight > 0:
        # Extract components: [depth1, tts1, tts_end]
        # Note: tts values are in log1p space, so we need to check monotonicity constraints
        # In log1p space: tts1 > 0 (log1p(0) = 0) and tts_end > tts1
        tts1_pred = u1_predicted[:, 1]  # (Batch,)
        tts_end_pred = u1_predicted[:, 2]  # (Batch,)

        # Penalty for tts1 <= 0 (should be positive in log1p space)
        # Use ReLU to penalize negative values
        penalty_tts1 = torch.relu(-tts1_pred + 1e-6)  # Penalize if tts1 <= 0

        # Penalty for tts_end <= tts1 (should be monotonically increasing)
        # Use ReLU to penalize when tts_end <= tts1
        penalty_tts_end = torch.relu(
            tts1_pred - tts_end_pred + 1e-6
        )  # Penalize if tts_end <= tts1

        # Total monotonicity loss
        monotonicity_loss = torch.mean(penalty_tts1 + penalty_tts_end)
        total_loss = total_loss + config.monotonicity_weight * monotonicity_loss

    # Add GAN loss if discriminator is enabled
    gan_loss = None
    if discriminator is not None and config.use_gan_loss:
        # Discriminator loss (generator wants to fool discriminator)
        fake_scores = discriminator(u1_predicted)
        gan_loss = torch.mean(
            torch.nn.functional.binary_cross_entropy_with_logits(
                fake_scores, torch.ones_like(fake_scores)
            )
        )

        total_loss = total_loss + config.gan_weight * gan_loss

    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return (
        total_loss.item(),
        gan_loss.item() if gan_loss is not None else 0.0,
        monotonicity_loss.item() if monotonicity_loss is not None else 0.0,
    )


def train_discriminator_step(discriminator, optimizer, model, batch, config):
    """
    Single discriminator training step for breakpoints.

    Args:
        discriminator: Discriminator model
        optimizer: Optimizer for discriminator
        model: Generator model (FFM model)
        batch: Real data breakpoints (Batch, 3)
        config: Configuration object

    Returns:
        loss: Discriminator loss value
    """
    device = config.device
    discriminator.train()
    model.eval()  # Set generator to eval mode for discriminator training

    # Real data breakpoints
    if batch.dim() == 3:
        u1_real = batch.squeeze(1).to(device)  # (Batch, 3)
    else:
        u1_real = batch.to(device)  # (Batch, 3)

    # Generate fake samples
    with torch.no_grad():
        # Sample noise and time
        u0 = torch.randn_like(u1_real).to(device)
        t = torch.rand(u1_real.shape[0], 1).to(device)
        ut = (1 - t) * u0 + t * u1_real

        # Predict vector field and generate sample
        predicted_v = model(ut, t)
        u1_fake = ut + predicted_v * torch.clamp(1 - t, min=1e-6)

    model.train()  # Set generator back to train mode

    # Discriminator forward pass
    real_scores = discriminator(u1_real)
    fake_scores = discriminator(u1_fake.detach())

    # Discriminator loss: maximize log(D(real)) + log(1 - D(fake))
    loss_real = torch.nn.functional.binary_cross_entropy_with_logits(
        real_scores, torch.ones_like(real_scores)
    )
    loss_fake = torch.nn.functional.binary_cross_entropy_with_logits(
        fake_scores, torch.zeros_like(fake_scores)
    )

    total_loss = (loss_real + loss_fake) / 2.0

    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss.item()


def main() -> None:
    global wandb

    cfg = cfg_mod.cfg
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(cfg.plots_dir, exist_ok=True)
    os.makedirs(cfg.results_dir, exist_ok=True)

    # Initialize wandb if available
    try:
        import wandb as wandb_module

        wandb = wandb_module
        wandb_name = (
            cfg.wandb_name or f"ffm_travel_time_{cfg.model_type}_{cfg.num_steps}steps"
        )
        wandb.init(
            project=cfg.wandb_project,
            config=vars(cfg),
            name=wandb_name,
        )
        print("[info] wandb initialized")

    except ImportError:
        print("[info] wandb not available, continuing without it")
        wandb = None

    device = torch.device(cfg.device)
    loader, max_length, dataset = create_dataloader(cfg.batch_size, cfg.num_workers)

    # Create model
    model = models_mod.create_model(cfg.model_type, cfg).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
    )

    # Create discriminator if GAN loss is enabled
    discriminator = None
    discriminator_optimizer = None
    if cfg.use_gan_loss:
        discriminator = models_mod.create_discriminator(cfg).to(device)
        discriminator_optimizer = optim.AdamW(
            discriminator.parameters(),
            lr=cfg.discriminator_lr,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay,
        )
        print(
            f"Created Discriminator with {sum(p.numel() for p in discriminator.parameters()):,} parameters"
        )

    # Create LR scheduler
    scheduler = None
    if cfg.use_scheduler:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=cfg.scheduler_mode,  # type: ignore
            factor=cfg.scheduler_factor,
            patience=cfg.scheduler_patience,
            min_lr=cfg.scheduler_min_lr,
        )
        print(
            f"Using ReduceLROnPlateau scheduler with patience={cfg.scheduler_patience}, "
            f"factor={cfg.scheduler_factor}, min_lr={cfg.scheduler_min_lr}"
        )

    print(
        f"Created {cfg.model_type.upper()} model with {sum(p.numel() for p in model.parameters()):,} parameters"
    )
    print(f"Training on {len(dataset)} profiles with max_length={max_length}")
    print(f"Initial learning rate: {cfg.learning_rate}")

    step = 0
    loss_history = []
    gan_loss_history = []
    discriminator_loss_history = []

    # For LR scheduler: keep track of recent losses for averaging
    recent_losses = []

    # Fixed samples for evaluation (3 breakpoints)
    z_fixed = torch.randn(cfg.num_samples, 3).to(device)

    print(f"Starting training for {cfg.num_steps} steps...")
    print(f"Dataset size: {len(dataset)} profiles")
    print(f"Batches per epoch: {len(loader)}")

    # Create a fresh dataloader iterator
    dataloader_iter = iter(loader)

    pbar = tqdm(total=cfg.num_steps, desc="Training FFM")

    while step < cfg.num_steps:
        try:
            # Get next batch, restart dataloader if needed
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                print(f"Restarting dataloader at step {step}")
                dataloader_iter = iter(loader)
                batch = next(dataloader_iter)

            try:
                # Train discriminator if GAN loss is enabled
                disc_loss = None
                gan_loss = 0.0

                if discriminator is not None and cfg.use_gan_loss:
                    # Train discriminator multiple times per generator step
                    for _ in range(cfg.d_steps_per_g):
                        disc_loss = train_discriminator_step(
                            discriminator, discriminator_optimizer, model, batch, cfg
                        )
                        discriminator_loss_history.append(disc_loss)

                # Training step (generator)
                result = train_ffm_step(
                    model, optimizer, batch, cfg, discriminator, discriminator_optimizer
                )
                loss, gan_loss, mono_loss = result
                loss_history.append(loss)
                if gan_loss > 0:
                    gan_loss_history.append(gan_loss)
            except Exception as e:
                print(f"Error in training step {step}: {e}")
                import traceback

                traceback.print_exc()
                break

            # Update LR scheduler - step more frequently based on recent loss average
            if scheduler is not None:
                recent_losses.append(loss)
                # Keep only recent losses (e.g., last 100 steps)
                if len(recent_losses) > 100:
                    recent_losses.pop(0)

                # Step scheduler every log_every steps (more frequent than checkpoint_every)
                if step % cfg.log_every == 0 and step > 0 and len(recent_losses) >= 10:
                    avg_recent_loss = sum(recent_losses[-50:]) / min(
                        50, len(recent_losses)
                    )
                    old_lr = optimizer.param_groups[0]["lr"]
                    scheduler.step(avg_recent_loss)
                    new_lr = optimizer.param_groups[0]["lr"]
                    if old_lr != new_lr:
                        print(
                            f"LR scheduler updated at step {step}: {old_lr:.2e} -> {new_lr:.2e}"
                        )

            # Log to wandb
            if wandb is not None:
                log_dict = {
                    "step": step,
                    "train/loss": loss,
                    "train/lr": optimizer.param_groups[0]["lr"],
                }
                if gan_loss > 0:
                    log_dict["train/gan_loss"] = gan_loss
                if disc_loss is not None:
                    log_dict["train/discriminator_loss"] = disc_loss
                if mono_loss > 0:
                    log_dict["train/monotonicity_loss"] = mono_loss
                wandb.log(log_dict)

            if step % cfg.log_every == 0:
                log_str = f"step={step} loss={loss:.6f}"
                if gan_loss > 0:
                    log_str += f" gan_loss={gan_loss:.6f}"
                if disc_loss is not None:
                    log_str += f" disc_loss={disc_loss:.6f}"
                if mono_loss > 0:
                    log_str += f" mono_loss={mono_loss:.6f}"
                print(log_str)

            if step % cfg.checkpoint_every == 0 and step > 0:
                # Save checkpoint
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": vars(cfg),
                    "step": step,
                    "loss_history": loss_history,
                }
                if discriminator is not None and discriminator_optimizer is not None:
                    checkpoint["discriminator"] = discriminator.state_dict()
                    checkpoint["discriminator_optimizer"] = (
                        discriminator_optimizer.state_dict()
                    )
                if scheduler is not None:
                    checkpoint["scheduler"] = scheduler.state_dict()
                torch.save(
                    checkpoint, os.path.join(cfg.out_dir, f"checkpoint_{step}.pt")
                )

                # Generate and save samples (breakpoints)
                with torch.no_grad():
                    model.eval()
                    samples = utils_mod.sample_ffm(
                        model, z_fixed, cfg.ode_steps, device, cfg.ode_solver
                    )
                    model.train()

                    # Denormalize samples (breakpoints)
                    samples_denorm = dataset.denormalize_batch(samples)

                    # Save breakpoints
                    np.save(
                        os.path.join(cfg.out_dir, f"samples_{step}.npy"),
                        samples_denorm.cpu().numpy(),
                    )

                    # Log sample statistics to wandb
                    if wandb is not None:
                        samples_np = samples_denorm.cpu().numpy()
                        wandb.log(
                            {
                                "step": step,
                                "samples/depth1_mean": float(samples_np[:, 0].mean()),
                                "samples/tts1_mean": float(samples_np[:, 1].mean()),
                                "samples/tts_end_mean": float(samples_np[:, 2].mean()),
                            }
                        )

                # Log metrics and plot
                utils_mod.log_profiles_metrics(
                    model, cfg, device, step, cfg.out_dir, dataset, wandb
                )

                # Plot loss curves
                utils_mod.plot_loss_curves(loss_history, cfg.out_dir, step)

                # Plot TTS/Vs comparison for generated samples
                samples_np = samples_denorm.cpu().numpy()
                if samples_np.shape[0] >= 100:
                    plot_path = os.path.join(
                        cfg.plots_dir, f"tts_vs_samples_{step}.png"
                    )
                    utils_mod.plot_tts_vs_samples(
                        samples_np,
                        plot_path,
                        max_depth=dataset.max_depth,
                        dz=cfg.depth_increment,
                        num_samples=100,
                        title=f"Generated Travel Time and Vs Profiles @ Step {step}",
                    )

                print(f"Saved checkpoint and samples at step {step}")

            step += 1
            pbar.update(1)

        except Exception as e:
            print(f"Error at step {step}: {e}")
            import traceback

            traceback.print_exc()
            break

    pbar.close()

    # Save final checkpoint
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": vars(cfg),
        "step": step,
        "loss_history": loss_history,
    }
    if discriminator is not None and discriminator_optimizer is not None:
        checkpoint["discriminator"] = discriminator.state_dict()
        checkpoint["discriminator_optimizer"] = discriminator_optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
    torch.save(checkpoint, os.path.join(cfg.out_dir, "checkpoint_final.pt"))

    # Generate final samples (breakpoints)
    with torch.no_grad():
        model.eval()
        samples = utils_mod.sample_ffm(
            model, z_fixed, cfg.ode_steps, device, cfg.ode_solver
        )
        samples_denorm = dataset.denormalize_batch(samples)
        np.save(
            os.path.join(cfg.out_dir, "samples_final.npy"), samples_denorm.cpu().numpy()
        )

    print(f"Training completed! Final loss: {loss_history[-1]:.6f}")
    print(f"Checkpoints saved to: {cfg.out_dir}")

    # Finish wandb run
    if wandb is not None:
        wandb.finish()
        print("[info] wandb run finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Flow Matching Model on Synthetic Toy Profiles"
    )
    parser.add_argument(
        "--num_steps", type=int, default=None, help="Number of training steps"
    )
    parser.add_argument(
        "--betas",
        type=float,
        nargs=2,
        default=None,
        help="Adam optimizer betas (e.g., --betas 0.9 0.999)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=None, help="Adam optimizer weight decay"
    )
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb run name")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        choices=["unet", "fno"],
        help="Model architecture",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default=None,
        choices=["l1", "l2"],
        help="Reconstruction loss type: 'l1' (MAE) or 'l2' (MSE)",
    )

    args = parser.parse_args()

    # Override config with command line arguments
    if args.num_steps is not None:
        cfg_mod.cfg.num_steps = args.num_steps
    if args.betas is not None:
        cfg_mod.cfg.betas = tuple(args.betas)
    if args.weight_decay is not None:
        cfg_mod.cfg.weight_decay = args.weight_decay
    if args.wandb_name is not None:
        cfg_mod.cfg.wandb_name = args.wandb_name
    if args.model_type is not None:
        cfg_mod.cfg.model_type = args.model_type
    if args.loss_type is not None:
        cfg_mod.cfg.loss_type = args.loss_type

    main()
