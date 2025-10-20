import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path

import wandb


@dataclass
class TrainConfig:
    epochs_dae: int = 50
    epochs_vae: int = 200
    beta_start: float = 0.0
    beta_end: float = 1.0
    beta_warmup_epochs: int = 50
    grad_clip_norm: Optional[float] = 1.0
    amp: bool = True
    early_stop_patience: int = 20


def compute_beta(epoch: int, cfg: TrainConfig) -> float:
    if epoch >= cfg.beta_warmup_epochs:
        return cfg.beta_end
    fraction = max(0.0, float(epoch) / max(1, cfg.beta_warmup_epochs))
    return cfg.beta_start + (cfg.beta_end - cfg.beta_start) * fraction


def mse_loss(x_recon: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(x_recon, x_target, reduction="mean")


def train_dae(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    cfg: TrainConfig,
    checkpoint_path: Optional[str] = None,
):
    model.to(device)
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = torch.amp.GradScaler(device_type, enabled=cfg.amp)
    best_val = float("inf")
    bad_epochs = 0

    for epoch in range(1, cfg.epochs_dae + 1):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"DAE pretrain {epoch}/{cfg.epochs_dae}")
        for noisy, clean in pbar:
            noisy = noisy.to(device)
            clean = clean.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type, enabled=cfg.amp):
                recon, mu, log_var = model(noisy)
                loss = mse_loss(recon, clean)
            scaler.scale(loss).backward()
            if cfg.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * noisy.size(0)
            pbar.set_postfix({"loss": loss.item()})

        train_loss = running / len(train_loader.dataset)
        # validation
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy = noisy.to(device)
                clean = clean.to(device)
                with torch.amp.autocast(device_type, enabled=cfg.amp):
                    recon, _, _ = model(noisy)
                    vloss = mse_loss(recon, clean)
                val_running += vloss.item() * noisy.size(0)
        val_loss = val_running / len(val_loader.dataset)

        logging.info(f"DAE Epoch {epoch} train={train_loss:.4f} val={val_loss:.4f}")
        wandb.log({"dae/train_loss": train_loss, "dae/val_loss": val_loss, "epoch": epoch})

        if val_loss + 1e-7 < best_val:
            best_val = val_loss
            bad_epochs = 0
            if checkpoint_path:
                torch.save({
                    "stage": "dae",
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                }, checkpoint_path)
                wandb.save(checkpoint_path, base_path=str(Path(checkpoint_path).parent))
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.early_stop_patience:
                logging.info("Early stopping DAE pretraining")
                break


def train_vae(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    cfg: TrainConfig,
    checkpoint_path: Optional[str] = None,
):
    model.to(device)
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = torch.amp.GradScaler(device_type, enabled=cfg.amp)
    best_val = float("inf")
    bad_epochs = 0

    for epoch in range(1, cfg.epochs_vae + 1):
        beta = compute_beta(epoch, cfg)
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"VAE finetune {epoch}/{cfg.epochs_vae} (beta={beta:.3f})")
        for x, _ in pbar:  # dataset returns (input, target), but for VAE we use x
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type, enabled=cfg.amp):
                recon, mu, log_var = model(x)
                recon_loss = mse_loss(recon, x)
                kld = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                loss = recon_loss + beta * kld
            scaler.scale(loss).backward()
            if cfg.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * x.size(0)
            pbar.set_postfix({"loss": loss.item(), "beta": beta})

        train_loss = running / len(train_loader.dataset)

        # validation
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                with torch.amp.autocast(device_type, enabled=cfg.amp):
                    recon, mu, log_var = model(x)
                    recon_loss = mse_loss(recon, x)
                    kld = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                    vloss = recon_loss + beta * kld
                val_running += vloss.item() * x.size(0)
        val_loss = val_running / len(val_loader.dataset)
        scheduler.step(val_loss)

        logging.info(
            f"VAE Epoch {epoch} train={train_loss:.4f} val={val_loss:.4f} beta={beta:.3f}"
        )
        wandb.log(
            {
                "vae/train_loss": train_loss,
                "vae/val_loss": val_loss,
                "vae/beta": beta,
                "epoch": epoch,
            }
        )

        if val_loss + 1e-7 < best_val:
            best_val = val_loss
            bad_epochs = 0
            if checkpoint_path:
                torch.save({
                    "stage": "vae",
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                }, checkpoint_path)
                wandb.save(checkpoint_path, base_path=str(Path(checkpoint_path).parent))
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.early_stop_patience:
                logging.info("Early stopping VAE fine-tuning")
                break


