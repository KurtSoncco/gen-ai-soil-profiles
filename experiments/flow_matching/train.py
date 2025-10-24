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
    
    # target_v: The target vector field (u1 - u0)
    target_v = u1 - u0
    
    # --- Forward pass ---
    predicted_v = model(ut, t)
    
    # --- Loss calculation ---
    loss = nn.MSELoss()(predicted_v, target_v)
    
    # --- Backward pass ---
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()


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
    real_vs30, avg_samples_per_meter = utils_mod.compute_real_vs30_and_density(cfg.parquet_path)

    # Create model
    model = models_mod.create_model(cfg.model_type, cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    print(f"Created {cfg.model_type.upper()} model with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Training on {len(dataset)} profiles with max_length={max_length}")

    step = 0
    loss_history = []

    # Fixed samples for evaluation
    z_fixed = torch.randn(cfg.num_samples, 1, max_length).to(device)

    while step < cfg.num_steps:
        pbar = tqdm(total=cfg.num_steps - step, desc="Training FFM", leave=False)
        for batch in loader:
            loss = train_ffm_step(model, optimizer, batch, cfg)
            loss_history.append(loss)

            # Log to wandb
            if wandb is not None:
                wandb.log({
                    "step": step,
                    "train/loss": loss,
                    "train/lr": optimizer.param_groups[0]['lr'],
                })

            if step % cfg.log_every == 0:
                print(f"step={step} loss={loss:.6f}")

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
                torch.save(checkpoint, os.path.join(cfg.out_dir, f"checkpoint_{step}.pt"))

                # Generate and save samples
                with torch.no_grad():
                    model.eval()
                    samples = utils_mod.sample_ffm(model, z_fixed, cfg.ode_steps, device)
                    model.train()
                    
                    # Denormalize samples
                    samples_denorm = dataset.denormalize_batch(samples)
                    
                    # Save samples
                    np.save(os.path.join(cfg.out_dir, f"samples_{step}.npy"), samples_denorm.cpu().numpy())
                    
                    # Log sample statistics to wandb
                    if wandb is not None:
                        wandb.log({
                            "step": step,
                            "samples/mean": samples_denorm.mean().item(),
                            "samples/std": samples_denorm.std().item(),
                            "samples/min": samples_denorm.min().item(),
                            "samples/max": samples_denorm.max().item(),
                        })

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
                    wandb,
                )

                # Only plot loss curves for final checkpoint or every 50 steps
                if step % 50 == 0 or step == cfg.num_steps - 1:
                    utils_mod.plot_loss_curves(loss_history, cfg.out_dir, step)

                print(f"Saved checkpoint and samples at step {step}")

            step += 1
            pbar.update(1)
            if step >= cfg.num_steps:
                break
        pbar.close()

        # Break out of the outer loop when we reach the step limit
        if step >= cfg.num_steps:
            break

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
        samples = utils_mod.sample_ffm(model, z_fixed, cfg.ode_steps, device)
        samples_denorm = dataset.denormalize_batch(samples)
        np.save(os.path.join(cfg.out_dir, "samples_final.npy"), samples_denorm.cpu().numpy())

    print(f"Training completed! Final loss: {loss_history[-1]:.6f}")
    print(f"Checkpoints saved to: {cfg.out_dir}")

    # Finish wandb run
    if wandb is not None:
        wandb.finish()
        print("[info] wandb run finished")


if __name__ == "__main__":
    main()
