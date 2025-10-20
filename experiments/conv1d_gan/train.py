from __future__ import annotations

import os
import random
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

try:
    from . import config as cfg_mod
    from .data import create_dataloader
    from . import models as models_mod
    from . import utils as utils_mod
except Exception:  # fallback when running as script
    import config as cfg_mod
    from data import create_dataloader
    import models as models_mod
    import utils as utils_mod


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


def main() -> None:
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader, max_length = create_dataloader(cfg.batch_size, cfg.num_workers)

    G = Generator1D(latent_dim=cfg.latent_dim, base_ch=cfg.base_channels, out_length=max_length).to(device)
    D = Discriminator1D(base_ch=cfg.base_channels).to(device)

    opt_G = optim.Adam(G.parameters(), lr=cfg.lr, betas=cfg.betas)
    opt_D = optim.Adam(D.parameters(), lr=cfg.lr, betas=cfg.betas)

    bce = nn.BCEWithLogitsLoss()

    step = 0
    z_fixed = torch.randn(16, cfg.latent_dim, device=device)
    loss_D = torch.tensor(0.0)
    loss_G = torch.tensor(0.0)

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
                loss_D = bce(pred_real, torch.ones_like(pred_real)) + bce(pred_fake, torch.zeros_like(pred_fake))
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

            if step % cfg.log_every == 0:
                print(f"step={step} loss_D={loss_D.item():.4f} loss_G={loss_G.item():.4f}")

            if step % cfg.checkpoint_every == 0 and step > 0:
                torch.save({
                    "G": G.state_dict(),
                    "D": D.state_dict(),
                    "cfg": asdict(cfg),
                    "step": step,
                }, os.path.join(cfg.out_dir, f"checkpoint_{step}.pt"))

                with torch.no_grad():
                    samples = G(z_fixed).cpu().numpy()
                    np.save(os.path.join(cfg.out_dir, f"samples_{step}.npy"), samples)

                # Log Vs30 distribution metrics and plot
                from utils import log_vs30_metrics as _log_vs30
                _log_vs30(G, cfg, device, step, cfg.out_dir, real_vs30, avg_samples_per_meter)

            step += 1
            pbar.update(1)
            if step >= cfg.num_steps:
                break
        pbar.close()


if __name__ == "__main__":
    main()


