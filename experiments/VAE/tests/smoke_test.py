import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

import wandb

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vae_model import VAE
from datasets import TTSDataset
from dae_trainer import TrainConfig, train_dae, train_vae


def main() -> None:
    # Config
    num_layers = 100
    latent_dim = 16
    num_samples = 256
    batch_size = 32
    lr = 5e-4

    # Synthetic non-negative d_tts-like data
    rng = np.random.default_rng(42)
    base = rng.exponential(scale=0.1, size=(num_samples, num_layers))
    data = np.log1p(base).astype(np.float32)

    train_np, val_np = data[:200], data[200:]
    train_ds = TTSDataset(train_np, corruption_noise_std=0.05)
    val_ds = TTSDataset(val_np, corruption_noise_std=0.05)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(input_dim=num_layers, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2
    )

    os.environ.setdefault("WANDB_MODE", "offline")
    run_name = f"smoke-MLP-layers{num_layers}-lat{latent_dim}-bs{batch_size}"
    wandb.init(
        project=os.environ.get("W_B_PROJECT", "soilgen-vae"),
        name=run_name,
        reinit=True,
        config={
            "num_layers": num_layers,
            "latent_dim": latent_dim,
            "batch_size": batch_size,
            "lr": lr,
            "run_name": run_name,
        },
    )
    wandb.watch(model, log="gradients", log_freq=10)

    cfg = TrainConfig(
        epochs_dae=2,
        epochs_vae=2,
        beta_start=0.0,
        beta_end=1.0,
        beta_warmup_epochs=2,
        grad_clip_norm=1.0,
        amp=False,
        early_stop_patience=5,
    )
    ckpt_path = str(Path(__file__).parent / "smoke_checkpoint.pt")
    train_dae(
        model,
        optimizer,
        train_loader,
        val_loader,
        device,
        cfg,
        checkpoint_path=ckpt_path,
    )
    train_vae(
        model,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        device,
        cfg,
        checkpoint_path=ckpt_path,
    )

    out_path = Path(__file__).parent / "smoke_model.pth"
    torch.save({"model": model.state_dict()}, out_path)
    print("Smoke test completed. Model saved to", out_path)
    wandb.finish()


if __name__ == "__main__":
    main()
