"""Training script for Vs Pairs Flow Matching models.

This module implements unguided flow matching for training neural networks
to generate vs pairs using ODE-based generative modeling.
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


def train_ffm_step(model, optimizer, batch, config):
    """
    Single FFM training step.

    Args:
        model: Neural field model
        optimizer: Adam optimizer
        batch: Real data pairs (Batch, 1, 4)
        config: Configuration object

    Returns:
        loss: MSE loss value
    """
    device = config.device

    # u1: Real data pairs (t=1)
    u1 = batch.to(device)

    # u0: Noise pairs (t=0)
    u0 = torch.randn_like(u1).to(device)

    # t: Random time from [0, 1]
    t = torch.rand(u1.shape[0], 1).to(device)

    # Broadcast t for interpolation
    t_broadcast = t.view(-1, 1, 1).expand(-1, 1, u1.shape[-1])

    # ut: Interpolated pairs at time t
    ut = (1 - t_broadcast) * u0 + t_broadcast * u1

    # target_v: The target vector field using proper flow matching
    target_v = torch.where(
        t_broadcast < 0.999,  # Avoid division by zero
        (u1 - ut) / torch.clamp(1 - t_broadcast, min=1e-6),
        torch.zeros_like(u1),
    )

    # --- Forward pass ---
    predicted_v = model(ut, t)

    # --- Loss calculation ---
    # L2 (MSE) reconstruction loss - unconstrained flow matching
    reconstruction_loss = torch.mean((predicted_v - target_v) ** 2)

    # --- Backward pass ---
    optimizer.zero_grad()
    reconstruction_loss.backward()
    optimizer.step()

    return reconstruction_loss.item()


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
            cfg.wandb_name or f"ffm_vspairs_{cfg.model_type}_{cfg.num_steps}steps"
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

    # Create LR scheduler
    scheduler = None
    if cfg.use_scheduler:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=cfg.scheduler_mode,  # type: ignore[arg-type]
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
    print(f"Training on {len(dataset)} vs pairs")
    print(f"Initial learning rate: {cfg.learning_rate}")

    step = 0
    loss_history = []
    eval_loss_history = []

    # Fixed samples for evaluation
    z_fixed = torch.randn(cfg.num_samples, 1, max_length).to(device)

    print(f"Starting training for {cfg.num_steps} steps...")
    print(f"Dataset size: {len(dataset)} pairs")
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
                # Training step
                loss = train_ffm_step(model, optimizer, batch, cfg)
                loss_history.append(loss)
            except Exception as e:
                print(f"Error in training step {step}: {e}")
                import traceback

                traceback.print_exc()
                break

            # Update LR scheduler
            if scheduler is not None:
                eval_loss_history.append(loss)
                if step % cfg.checkpoint_every == 0 and step > 0:
                    recent_loss = sum(eval_loss_history[-cfg.checkpoint_every :]) / len(
                        eval_loss_history[-cfg.checkpoint_every :]
                    )
                    scheduler.step(recent_loss)
                    print(
                        f"LR scheduler updated. New LR: {optimizer.param_groups[0]['lr']:.2e}"
                    )

            # Log to wandb
            if wandb is not None:
                wandb.log(
                    {
                        "step": step,
                        "train/loss": loss,
                        "train/lr": optimizer.param_groups[0]["lr"],
                    }
                )

            if step % cfg.log_every == 0:
                print(f"step={step} loss={loss:.6f}")

            if step % cfg.checkpoint_every == 0 and step > 0:
                # Save checkpoint
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": vars(cfg),
                    "step": step,
                    "loss_history": loss_history,
                }
                if scheduler is not None:
                    checkpoint["scheduler"] = scheduler.state_dict()
                torch.save(
                    checkpoint, os.path.join(cfg.out_dir, f"checkpoint_{step}.pt")
                )

                # Generate and save samples
                with torch.no_grad():
                    model.eval()
                    samples = utils_mod.sample_ffm(
                        model, z_fixed, cfg.ode_steps, device
                    )
                    model.train()

                    # Denormalize samples
                    samples_denorm = dataset.denormalize_batch(samples)

                    # Save samples
                    np.save(
                        os.path.join(cfg.out_dir, f"samples_{step}.npy"),
                        samples_denorm.cpu().numpy(),
                    )

                    # Log sample statistics to wandb
                    if wandb is not None:
                        samples_np = samples_denorm.squeeze(1).cpu().numpy()
                        wandb.log(
                            {
                                "step": step,
                                "samples/mean": samples_np.mean().item(),
                                "samples/std": samples_np.std().item(),
                                "samples/min": samples_np.min().item(),
                                "samples/max": samples_np.max().item(),
                            }
                        )

                # Log metrics and plot
                utils_mod.log_metrics(
                    model, cfg, device, step, cfg.out_dir, dataset, wandb
                )

                # Plot loss curves
                utils_mod.plot_loss_curves(loss_history, cfg.out_dir, step)

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
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
    torch.save(checkpoint, os.path.join(cfg.out_dir, "checkpoint_final.pt"))

    # Generate final samples
    with torch.no_grad():
        model.eval()
        samples = utils_mod.sample_ffm(model, z_fixed, cfg.ode_steps, device)
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
        description="Train Flow Matching Model on Vs Pairs"
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
        choices=["mlp", "fno"],
        help="Model architecture",
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

    main()
