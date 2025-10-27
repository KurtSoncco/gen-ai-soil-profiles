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


"""Training script for Conv1D GAN with Vs30 metrics and tqdm."""

# Expose commonly used symbols for linter/static analysis
cfg = cfg_mod.cfg
Generator1D = models_mod.Generator1D
Discriminator1D = models_mod.Discriminator1D
compute_real_vs30_and_density = utils_mod.compute_real_vs30_and_density
compute_generated_vs30 = utils_mod.compute_generated_vs30
ks_statistic = utils_mod.ks_statistic
plot_loss_curves = utils_mod.plot_loss_curves


def main() -> None:
    global wandb

    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(cfg.plots_dir, exist_ok=True)
    os.makedirs(cfg.results_dir, exist_ok=True)

    # Initialize wandb if available
    try:
        import wandb as wandb_module

        wandb = wandb_module
        wandb.init(
            project="conv1d-gan-soil-profiles",
            config=asdict(cfg),
            name=f"conv1d_gan_{cfg.num_steps}steps",
        )
        print("[info] wandb initialized")
    except ImportError:
        print("[info] wandb not available, continuing without it")
        wandb = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader, max_length = create_dataloader(cfg.batch_size, cfg.num_workers)

    G = Generator1D(
        latent_dim=cfg.latent_dim, base_ch=cfg.base_channels, out_length=max_length
    ).to(device)
    D = Discriminator1D(base_ch=cfg.base_channels).to(device)

    opt_G = optim.Adam(G.parameters(), lr=cfg.lr, betas=cfg.betas)
    opt_D = optim.Adam(D.parameters(), lr=cfg.lr, betas=cfg.betas)

    bce = nn.BCEWithLogitsLoss()

    step = 0
    z_fixed = torch.randn(16, cfg.latent_dim, device=device)
    loss_D = torch.tensor(0.0)
    loss_G = torch.tensor(0.0)

    # Loss tracking
    loss_history = {"steps": [], "loss_D": [], "loss_G": []}

    # --- Metrics prep: real Vs30 distribution and samples-per-meter estimate ---
    real_vs30, avg_samples_per_meter = compute_real_vs30_and_density(cfg.parquet_path)

    # metrics are handled in utils; real_vs30 and density are precomputed above

    while step < cfg.num_steps:
        pbar = tqdm(total=cfg.num_steps - step, desc="Training", leave=False)
        for real, mask in loader:
            real = real.to(device)  # (b, 1, L)
            b = real.size(0)

            # Train D
            for _ in range(cfg.d_steps_per_g):
                z = torch.randn(b, cfg.latent_dim, device=device)
                with torch.no_grad():
                    fake = G(z)

                D.zero_grad(set_to_none=True)
                pred_real = D(real)
                pred_fake = D(fake.detach())
                loss_D = bce(pred_real, torch.ones_like(pred_real)) + bce(
                    pred_fake, torch.zeros_like(pred_fake)
                )
                loss_D.backward()
                opt_D.step()

            # Train G
            G.zero_grad(set_to_none=True)
            z = torch.randn(b, cfg.latent_dim, device=device)
            fake = G(z)
            pred_fake = D(fake)
            loss_G = bce(pred_fake, torch.ones_like(pred_fake))
            loss_G.backward()
            opt_G.step()

            # Track losses
            loss_history["steps"].append(step)
            loss_history["loss_D"].append(loss_D.item())
            loss_history["loss_G"].append(loss_G.item())

            # Log to wandb
            if wandb is not None:
                wandb.log(
                    {
                        "step": step,
                        "train/loss_D": loss_D.item(),
                        "train/loss_G": loss_G.item(),
                    }
                )

            if step % cfg.log_every == 0:
                print(
                    f"step={step} loss_D={loss_D.item():.4f} loss_G={loss_G.item():.4f}"
                )

            if step % cfg.checkpoint_every == 0 and step > 0:
                torch.save(
                    {
                        "G": G.state_dict(),
                        "D": D.state_dict(),
                        "cfg": asdict(cfg),
                        "step": step,
                        "loss_history": loss_history,
                    },
                    os.path.join(cfg.out_dir, f"checkpoint_{step}.pt"),
                )

                with torch.no_grad():
                    samples = G(z_fixed).cpu().numpy()
                    np.save(os.path.join(cfg.out_dir, f"samples_{step}.npy"), samples)

                # Log Vs30 distribution metrics and plot
                _log_vs30 = utils_mod.log_vs30_metrics

                _log_vs30(
                    G,
                    cfg,
                    device,
                    step,
                    cfg.out_dir,
                    real_vs30,
                    avg_samples_per_meter,
                    wandb,
                )

                # Only plot loss curves for final checkpoint or every 50 steps
                if step % 50 == 0 or step == cfg.num_steps - 1:
                    plot_loss_curves(loss_history, cfg.out_dir, step)

            step += 1
            pbar.update(1)
            if step >= cfg.num_steps:
                break
        pbar.close()

        # Break out of the outer loop when we reach the step limit
        if step >= cfg.num_steps:
            break

    # Finish wandb run at the very end
    if wandb is not None:
        wandb.finish()
        print("[info] wandb run finished")


if __name__ == "__main__":
    main()
