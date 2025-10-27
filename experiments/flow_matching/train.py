from __future__ import annotations

import os
import random
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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


def compute_tvd_loss(x):
    """
    Compute Total Variation Diminishing (TVD) loss to encourage smoothness.

    Args:
        x: Tensor of shape (batch_size, channels, length)

    Returns:
        TVD loss scalar
    """
    # Compute differences between adjacent elements along the length dimension
    diff = torch.abs(x[:, :, 1:] - x[:, :, :-1])

    # Sum over all dimensions except batch
    tvd_loss = torch.mean(diff)

    return tvd_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_ffm_step(model, optimizer, batch, config):
    """
    Single FFM training step.

    Args:
        model: Neural field model (UNet or FNO)
        optimizer: Adam optimizer
        batch: Real data profiles (Batch, 1, Length)
        config: Configuration object

    Returns:
        loss: MSE loss value
    """
    device = config.device

    # u1: Real data profile (t=1)
    u1 = batch.to(device)

    # u0: Noise profile (t=0)
    u0 = torch.randn_like(u1).to(device)

    # t: Random time from [0, 1]
    t = torch.rand(u1.shape[0], 1).to(device)

    # Broadcast t for interpolation
    # (Batch, 1) -> (Batch, 1, 1)
    t_broadcast = t.view(-1, 1, 1).expand(-1, 1, u1.shape[-1])

    # ut: Interpolated profile at time t
    # ut = (1-t)*u0 + t*u1
    ut = (1 - t_broadcast) * u0 + t_broadcast * u1

    # target_v: The target vector field using proper flow matching
    # For flow matching, we use the conditional flow: v_t = u1 - u0
    # But we can also use a more sophisticated path like:
    # v_t = (u1 - ut) / (1 - t) when t < 1, else 0
    # This ensures the flow points towards the target at all times
    target_v = torch.where(
        t_broadcast < 0.999,  # Avoid division by zero
        (u1 - ut) / torch.clamp(1 - t_broadcast, min=1e-6),
        torch.zeros_like(u1),
    )

    # --- Forward pass ---
    predicted_v = model(ut, t)

    # --- Loss calculation ---
    mse_loss = nn.MSELoss()(predicted_v, target_v)

    # Add TVD regularization to encourage smoothness
    tvd_loss = compute_tvd_loss(predicted_v)
    total_loss = mse_loss + config.tvd_weight * tvd_loss

    # --- Backward pass ---
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss.item(), mse_loss.item(), tvd_loss.item()


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
        wandb_name = cfg.wandb_name or f"ffm_{cfg.model_type}_{cfg.num_steps}steps"
        wandb.init(
            project=cfg.wandb_project,
            config=asdict(cfg),
            name=wandb_name,
        )
        print("[info] wandb initialized")
    except ImportError:
        print("[info] wandb not available, continuing without it")
        wandb = None

    device = torch.device(cfg.device)
    loader, max_length, dataset = create_dataloader(cfg.batch_size, cfg.num_workers)

    # --- Metrics prep: real Vs30 distribution and samples-per-meter estimate ---
    real_vs30, avg_samples_per_meter = utils_mod.compute_real_vs30_and_density(
        cfg.parquet_path
    )

    # Create model
    model = models_mod.create_model(cfg.model_type, cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    print(
        f"Created {cfg.model_type.upper()} model with {sum(p.numel() for p in model.parameters()):,} parameters"
    )
    print(f"Training on {len(dataset)} profiles with max_length={max_length}")

    step = 0
    loss_history = []

    # Fixed samples for evaluation
    z_fixed = torch.randn(cfg.num_samples, 1, max_length).to(device)

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
                total_loss, mse_loss, tvd_loss = train_ffm_step(
                    model, optimizer, batch, cfg
                )
                loss_history.append(total_loss)
            except Exception as e:
                print(f"Error in training step {step}: {e}")
                import traceback

                traceback.print_exc()
                break

            # Log to wandb
            if wandb is not None:
                wandb.log(
                    {
                        "step": step,
                        "train/total_loss": total_loss,
                        "train/mse_loss": mse_loss,
                        "train/tvd_loss": tvd_loss,
                        "train/lr": optimizer.param_groups[0]["lr"],
                    }
                )

            if step % cfg.log_every == 0:
                print(
                    f"step={step} total_loss={total_loss:.6f} mse_loss={mse_loss:.6f} tvd_loss={tvd_loss:.6f}"
                )

            if step % cfg.checkpoint_every == 0 and step > 0:
                # Save checkpoint
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": asdict(cfg),
                    "step": step,
                    "loss_history": loss_history,
                    "dataset_min": dataset.min_val,
                    "dataset_max": dataset.max_val,
                }
                torch.save(
                    checkpoint, os.path.join(cfg.out_dir, f"checkpoint_{step}.pt")
                )

                # Generate and save samples
                with torch.no_grad():
                    model.eval()

                    # Choose sampler based on configuration
                    if cfg.use_pcfm:
                        samples = utils_mod.sample_ffm_pcfm(
                            model,
                            z_fixed,
                            cfg.ode_steps,
                            device,
                            dataset,
                            guidance_strength=cfg.pcfm_guidance_strength,
                            monotonic_weight=cfg.pcfm_monotonic_weight,
                            positivity_weight=cfg.pcfm_positivity_weight,
                        )
                    else:
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
                        wandb.log(
                            {
                                "step": step,
                                "samples/mean": samples_denorm.mean().item(),
                                "samples/std": samples_denorm.std().item(),
                                "samples/min": samples_denorm.min().item(),
                                "samples/max": samples_denorm.max().item(),
                            }
                        )

                # Log Vs30 distribution metrics and plot
                utils_mod.log_vs30_metrics(
                    model,
                    cfg,
                    device,
                    step,
                    cfg.out_dir,
                    real_vs30,
                    avg_samples_per_meter,
                    max_length,
                    dataset,
                    wandb,
                )

                # Only plot loss curves for final checkpoint or every 50 steps
                if step % 50 == 0 or step == cfg.num_steps - 1:
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
        "config": asdict(cfg),
        "step": step,
        "loss_history": loss_history,
        "dataset_min": dataset.min_val,
        "dataset_max": dataset.max_val,
    }
    torch.save(checkpoint, os.path.join(cfg.out_dir, "checkpoint_final.pt"))

    # Generate final samples
    with torch.no_grad():
        model.eval()

        # Choose sampler based on configuration
        if cfg.use_pcfm:
            samples = utils_mod.sample_ffm_pcfm(
                model,
                z_fixed,
                cfg.ode_steps,
                device,
                dataset,
                guidance_strength=cfg.pcfm_guidance_strength,
                monotonic_weight=cfg.pcfm_monotonic_weight,
                positivity_weight=cfg.pcfm_positivity_weight,
            )
        else:
            samples = utils_mod.sample_ffm(model, z_fixed, cfg.ode_steps, device)

        samples_denorm = dataset.denormalize_batch(samples)
        np.save(
            os.path.join(cfg.out_dir, "samples_final.npy"), samples_denorm.cpu().numpy()
        )

    print(f"Training completed! Final loss: {loss_history[-1]:.6f}")
    print(f"Checkpoints saved to: {cfg.out_dir}")

    # Generate comprehensive comparison plot for final epoch and log to wandb
    if wandb is not None:
        print("Generating comprehensive comparison plot for final epoch...")
        try:
            # Load some real data for comparison (same number as generated samples)
            loader_real, _, _ = create_dataloader(
                cfg.batch_size, cfg.num_workers, shuffle=False
            )
            real_profiles = []
            for i, batch in enumerate(loader_real):
                real_denorm = dataset.denormalize_batch(batch)
                real_profiles.append(real_denorm.numpy())
                if (
                    len(real_profiles) * cfg.batch_size >= cfg.num_samples
                ):  # Match generated samples
                    break
            real_profiles = np.concatenate(real_profiles, axis=0)
            # Ensure we have exactly the same number as generated samples
            real_profiles = real_profiles[: cfg.num_samples]

            # Generate final samples for comparison
            with torch.no_grad():
                model.eval()
                if cfg.use_pcfm:
                    final_samples = utils_mod.sample_ffm_pcfm(
                        model,
                        z_fixed,
                        cfg.ode_steps,
                        device,
                        dataset,
                        guidance_strength=cfg.pcfm_guidance_strength,
                        monotonic_weight=cfg.pcfm_monotonic_weight,
                        positivity_weight=cfg.pcfm_positivity_weight,
                    )
                else:
                    final_samples = utils_mod.sample_ffm(
                        model, z_fixed, cfg.ode_steps, device
                    )
                final_samples_denorm = dataset.denormalize_batch(final_samples)
                generated_profiles = final_samples_denorm.cpu().numpy()

            # Create comprehensive comparison plot
            utils_mod.plot_comprehensive_comparison(
                real_profiles,
                generated_profiles,
                cfg.out_dir,
                step,
                max_profiles=20,
                avg_samples_per_meter=avg_samples_per_meter,
            )

            # Log the plot to wandb
            plot_path = os.path.join(
                cfg.out_dir, f"comprehensive_comparison_step_{step}.png"
            )
            if os.path.exists(plot_path):
                wandb.log(
                    {
                        "step": step,
                        "final_comprehensive_comparison": wandb.Image(plot_path),
                    }
                )
                print(f"Comprehensive comparison plot logged to wandb: {plot_path}")
            else:
                print(
                    f"Warning: Comprehensive comparison plot not found at {plot_path}"
                )

        except Exception as e:
            print(f"Error generating comprehensive comparison plot: {e}")
            import traceback

            traceback.print_exc()

    # Finish wandb run
    if wandb is not None:
        wandb.finish()
        print("[info] wandb run finished")


if __name__ == "__main__":
    main()
