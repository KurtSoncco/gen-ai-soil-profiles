"""Training script for Flow Matching Variable-Length Paired Token Breakpoints.

This module implements training for flow matching models with variable-length paired token breakpoints.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


def convert_to_json_serializable(obj):
    """Convert numpy types and other non-JSON-serializable objects to native Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    else:
        return obj


try:
    import wandb
except ImportError:
    wandb = None

try:
    from experiments.flow_matching_simplified import config as cfg_mod
    from experiments.flow_matching_simplified.data import (
        FlowMatchingDataLoader,
        FlowMatchingDataset,
    )
    from experiments.flow_matching_simplified.model import TransformerModel
    from experiments.flow_matching_simplified.utils import compute_vs_penalty
except ImportError:  # fallback when running as script
    import config as cfg_mod
    from model import TransformerModel  # type: ignore
    from utils import compute_vs_penalty  # type: ignore

    from data import FlowMatchingDataLoader, FlowMatchingDataset  # type: ignore

# Global wandb run variable
wandb_run = None


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_flow_matching_loss(
    model: TransformerModel,
    u1: torch.Tensor,
    attention_mask: torch.Tensor,
    sequence_stats: torch.Tensor,
    device: torch.device,
    config: cfg_mod.Config | None = None,
) -> torch.Tensor:
    """
    Compute flow matching loss for variable-length sequences.

    Args:
        model: Transformer model
        u1: Real data sequences (batch_size, max_length, 2)
        attention_mask: Boolean mask (batch_size, max_length) - True for real tokens
        sequence_stats: Sequence statistics (batch_size, 4)
        device: PyTorch device
        config: Optional config object for Vs regularization (if None, uses cfg_mod.cfg)

    Returns:
        Masked MSE loss (with optional Vs penalty)
    """
    if config is None:
        config = cfg_mod.cfg

    batch_size, max_length, _ = u1.shape

    # u0: Noise sequences - only generate noise where attention_mask is True
    u0 = torch.randn_like(u1).to(device)
    # Mask noise: keep padding as zeros
    u0 = u0 * attention_mask.unsqueeze(-1).float()

    # t: Random time from [0, 1]
    t = torch.rand(batch_size, 1).to(device)

    # Broadcast t for interpolation: (batch_size, 1) -> (batch_size, max_length, 2)
    t_broadcast = t.unsqueeze(-1).expand(
        -1, max_length, -1
    )  # (batch_size, max_length, 1)
    t_broadcast = t_broadcast.expand(-1, -1, 2)  # (batch_size, max_length, 2)

    # ut: Interpolated profile at time t
    # ut = (1-t)*u0 + t*u1
    ut = (1 - t_broadcast) * u0 + t_broadcast * u1

    # target_v: The target vector field
    # v_t = (u1 - ut) / (1 - t) when t < 0.999, else 0
    target_v = torch.where(
        t_broadcast < 0.999,
        (u1 - ut) / torch.clamp(1 - t_broadcast, min=1e-6),
        torch.zeros_like(u1),
    )

    # Forward pass
    predicted_v = model(ut, attention_mask, t, sequence_stats)

    # Compute MSE loss, masked by attention_mask
    # Only compute loss on real tokens (where attention_mask == True)
    loss_per_token = (predicted_v - target_v) ** 2  # (batch_size, max_length, 2)
    loss_per_token = loss_per_token.mean(dim=-1)  # (batch_size, max_length)

    # Apply attention mask
    mask_float = attention_mask.float()  # (batch_size, max_length)
    masked_loss = loss_per_token * mask_float
    total_loss = masked_loss.sum() / mask_float.sum()

    # Add Vs regularization penalty if enabled
    # This denormalizes ut before computing Vs, ensuring gradients flow correctly
    if config.use_vs_regularization:
        vs_penalty = compute_vs_penalty(
            ut,
            attention_mask,
            sequence_stats,
            config.vs_min,
            config.vs_max,
            config.min_dt,
            normalize=config.normalize,
        )
        total_loss = total_loss + config.vs_penalty_weight * vs_penalty

    return total_loss


def evaluate_model(
    model: TransformerModel,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    dataset: FlowMatchingDataset,
    wandb_run=None,
    config: cfg_mod.Config | None = None,
) -> dict[str, float]:
    """
    Evaluate model on validation set.

    Args:
        model: Transformer model
        val_loader: Validation data loader
        device: PyTorch device
        dataset: Dataset for denormalization
        wandb_run: Optional wandb run object

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    total_ts_mse = 0.0
    total_depth_mse = 0.0
    total_sequences = 0
    total_tokens = 0

    all_real_ts = []
    all_real_depth = []
    all_pred_ts = []
    all_pred_depth = []
    sequence_lengths = []

    with torch.no_grad():
        for batch in val_loader:
            tokens = batch["tokens"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            sequence_stats = batch["sequence_stats"].to(device)
            lengths = batch["length"].cpu().numpy()

            batch_size = tokens.shape[0]

            # Compute flow matching loss
            loss = compute_flow_matching_loss(
                model, tokens, attention_mask, sequence_stats, device, config=config
            )
            total_loss += loss.item() * batch_size

            # Compute reconstruction error
            # Sample random time and predict
            t = torch.rand(batch_size, 1).to(device)
            t_broadcast = (
                t.unsqueeze(-1).expand(-1, tokens.shape[1], -1).expand(-1, -1, 2)
            )
            u0 = (
                torch.randn_like(tokens).to(device)
                * attention_mask.unsqueeze(-1).float()
            )
            ut = (1 - t_broadcast) * u0 + t_broadcast * tokens
            predicted_v = model(ut, attention_mask, t, sequence_stats)

            # Estimate u at t=1
            remaining = torch.clamp(1 - t_broadcast, min=1e-6)
            u_pred = ut + predicted_v * remaining

            # Compute per-token MSE (masked)
            mask_float = attention_mask.float().unsqueeze(-1)
            ts_error = ((u_pred[:, :, 0] - tokens[:, :, 0]) ** 2) * mask_float[:, :, 0]
            depth_error = ((u_pred[:, :, 1] - tokens[:, :, 1]) ** 2) * mask_float[
                :, :, 0
            ]

            total_ts_mse += ts_error.sum().item()
            total_depth_mse += depth_error.sum().item()
            total_tokens += mask_float[:, :, 0].sum().item()

            # Collect statistics for distribution metrics
            for i in range(batch_size):
                seq_len = int(lengths[i])
                if seq_len > 0:
                    all_real_ts.extend(tokens[i, :seq_len, 0].cpu().numpy())
                    all_real_depth.extend(tokens[i, :seq_len, 1].cpu().numpy())
                    all_pred_ts.extend(u_pred[i, :seq_len, 0].cpu().numpy())
                    all_pred_depth.extend(u_pred[i, :seq_len, 1].cpu().numpy())
                    sequence_lengths.append(seq_len)

            total_sequences += batch_size

    # Compute metrics
    avg_loss = total_loss / total_sequences
    avg_ts_mse = total_ts_mse / total_tokens if total_tokens > 0 else 0.0
    avg_depth_mse = total_depth_mse / total_tokens if total_tokens > 0 else 0.0

    # Distribution metrics
    real_ts_mean = np.mean(all_real_ts) if all_real_ts else 0.0
    real_ts_std = np.std(all_real_ts) if all_real_ts else 0.0
    pred_ts_mean = np.mean(all_pred_ts) if all_pred_ts else 0.0
    pred_ts_std = np.std(all_pred_ts) if all_pred_ts else 0.0

    real_depth_mean = np.mean(all_real_depth) if all_real_depth else 0.0
    real_depth_std = np.std(all_real_depth) if all_real_depth else 0.0
    pred_depth_mean = np.mean(all_pred_depth) if all_pred_depth else 0.0
    pred_depth_std = np.std(all_pred_depth) if all_pred_depth else 0.0

    metrics = {
        "val/loss": avg_loss,
        "val/ts_mse": avg_ts_mse,
        "val/depth_mse": avg_depth_mse,
        "val/real_ts_mean": real_ts_mean,
        "val/real_ts_std": real_ts_std,
        "val/pred_ts_mean": pred_ts_mean,
        "val/pred_ts_std": pred_ts_std,
        "val/real_depth_mean": real_depth_mean,
        "val/real_depth_std": real_depth_std,
        "val/pred_depth_mean": pred_depth_mean,
        "val/pred_depth_std": pred_depth_std,
        "val/mean_sequence_length": np.mean(sequence_lengths)
        if sequence_lengths
        else 0.0,
    }

    # Log to wandb if run is provided
    if wandb_run is not None:
        wandb_run.log(metrics)

    model.train()
    return metrics


def sample_sequences(
    model: TransformerModel,
    num_samples: int,
    max_length: int,
    device: torch.device,
    dataset: FlowMatchingDataset,
    ode_steps: int = 50,
    wandb_run=None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate new sequences using ODE integration.

    Args:
        model: Trained transformer model
        num_samples: Number of samples to generate
        max_length: Maximum sequence length
        device: PyTorch device
        dataset: Dataset for sequence statistics and denormalization
        ode_steps: Number of ODE integration steps
        wandb_run: Optional wandb run object

    Returns:
        Generated sequences (numpy array), attention masks (numpy array)
    """
    model.eval()

    # Sample sequence lengths from training distribution
    train_lengths = [len(seq) for seq in dataset.sequences]
    sampled_lengths = np.random.choice(train_lengths, size=num_samples, replace=True)

    # Create initial noise and attention masks
    u = torch.randn(num_samples, max_length, 2).to(device)
    attention_masks = torch.zeros(num_samples, max_length, dtype=torch.bool).to(device)
    sequence_stats_list = []

    for i, seq_len in enumerate(sampled_lengths):
        attention_masks[i, :seq_len] = True
        # Sample random sequence stats from training data
        if dataset.normalize and len(dataset.sequence_stats) > 0:
            stats_idx = np.random.randint(0, len(dataset.sequence_stats))
            sequence_stats_list.append(dataset.sequence_stats[stats_idx])
        else:
            sequence_stats_list.append(
                {
                    "ts_mean": 0.0,
                    "ts_std": 1.0,
                    "depth_mean": 0.0,
                    "depth_std": 1.0,
                }
            )

    # Convert sequence stats to tensor
    if dataset.normalize and len(dataset.sequence_stats) > 0:
        stats_array = np.array(
            [
                [
                    stats["ts_mean"],
                    stats["ts_std"],
                    stats["depth_mean"],
                    stats["depth_std"],
                ]
                for stats in sequence_stats_list
            ]
        )
        sequence_stats = torch.FloatTensor(stats_array).to(device)
    else:
        sequence_stats = torch.zeros(num_samples, 4).to(device)

    # Mask noise
    u = u * attention_masks.unsqueeze(-1).float()

    # ODE integration using Euler method (simpler than RK4 for variable-length)
    dt = 1.0 / ode_steps

    with torch.no_grad():
        for step in range(ode_steps):
            t_val = step * dt
            t = torch.full((num_samples, 1), t_val).to(device)

            # Predict vector field
            v = model(u, attention_masks, t, sequence_stats)

            # Euler step
            u = u + v * dt

            # Keep padding masked
            u = u * attention_masks.unsqueeze(-1).float()

    # Denormalize sequences
    generated_sequences = []
    for i in range(num_samples):
        seq_len = sampled_lengths[i]
        seq_normalized = u[i, :seq_len, :].cpu().numpy()
        if dataset.normalize and len(sequence_stats_list) > i:
            # Use the stats we sampled for this sequence
            stats = sequence_stats_list[i]
            ts_denorm = seq_normalized[:, 0] * stats["ts_std"] + stats["ts_mean"]
            depth_denorm = (
                seq_normalized[:, 1] * stats["depth_std"] + stats["depth_mean"]
            )
            seq_denorm = np.column_stack([ts_denorm, depth_denorm])
        else:
            seq_denorm = seq_normalized
        generated_sequences.append(seq_denorm)

    # Log to wandb if run is provided
    if wandb_run is not None:
        # Log sample statistics
        all_ts = np.concatenate([seq[:, 0] for seq in generated_sequences])
        all_depth = np.concatenate([seq[:, 1] for seq in generated_sequences])
        wandb_run.log(
            {
                "samples/mean_ts": np.mean(all_ts),
                "samples/std_ts": np.std(all_ts),
                "samples/mean_depth": np.mean(all_depth),
                "samples/std_depth": np.std(all_depth),
                "samples/mean_length": np.mean(sampled_lengths),
            }
        )

    model.train()
    return np.array(generated_sequences, dtype=object), attention_masks.cpu().numpy()


def train_epoch(
    model: TransformerModel,
    train_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    config: cfg_mod.Config,
    dataset: FlowMatchingDataset,
    wandb_run=None,
    epoch: int = 0,
) -> float:
    """
    Train for one epoch.

    Args:
        model: Transformer model
        train_loader: Training data loader
        optimizer: Optimizer
        device: PyTorch device
        config: Configuration object
        dataset: Dataset
        wandb_run: Optional wandb run object
        epoch: Current epoch number

    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
    for batch_idx, batch in enumerate(pbar):
        tokens = batch["tokens"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        sequence_stats = batch["sequence_stats"].to(device)

        # Compute loss
        loss = compute_flow_matching_loss(
            model, tokens, attention_mask, sequence_stats, device, config=config
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({"loss": loss.item()})

        # Log to wandb
        if wandb_run is not None and batch_idx % config.log_every == 0:
            wandb_run.log(
                {
                    "train/loss": loss.item(),
                    "train/epoch": epoch,
                    "train/batch": batch_idx,
                    "train/lr": optimizer.param_groups[0]["lr"],
                }
            )

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def save_checkpoint(
    model: TransformerModel,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.ReduceLROnPlateau | None,
    epoch: int,
    best_val_loss: float,
    config: cfg_mod.Config,
    checkpoint_dir: str,
    is_best: bool = False,
) -> None:
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "config": vars(config),
    }

    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()

    # Save latest checkpoint
    latest_path = os.path.join(checkpoint_dir, "latest.pt")
    torch.save(checkpoint, latest_path)

    # Save best checkpoint
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best.pt")
        torch.save(checkpoint, best_path)

    # Save config as JSON
    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(config), f, indent=2)


def main() -> None:
    """Main training function."""
    global wandb_run

    cfg = cfg_mod.cfg
    set_seed(cfg.seed)

    # Create output directories
    os.makedirs(cfg.checkpoints_dir, exist_ok=True)
    os.makedirs(cfg.plots_dir, exist_ok=True)
    os.makedirs(cfg.results_dir, exist_ok=True)
    os.makedirs(cfg.samples_dir, exist_ok=True)

    # Initialize wandb
    try:
        if wandb is not None:
            wandb_name = cfg.wandb_name or f"flow_matching_breakpoints_{cfg.seed}"
            wandb_run = wandb.init(
                project=cfg.wandb_project,
                config=vars(cfg),
                name=wandb_name,
            )
            print("[info] wandb initialized")
        else:
            print("[info] wandb not available, continuing without it")
    except Exception as e:
        print(f"[warning] wandb initialization failed: {e}")
        wandb_run = None

    device = torch.device(cfg.device)
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    data_loader = FlowMatchingDataLoader(data_path=Path(cfg.data_path))
    data_loader.load_data()

    assert data_loader.sequences is not None, "Sequences must be loaded"
    n_total = len(data_loader.sequences)

    # Create train/val/test splits
    all_indices = torch.randperm(n_total)
    n_train = int(cfg.train_val_test_split[0] * n_total)
    n_val = int(cfg.train_val_test_split[1] * n_total)

    train_indices = all_indices[:n_train].tolist()
    val_indices = all_indices[n_train : n_train + n_val].tolist()
    test_indices = all_indices[n_train + n_val :].tolist()

    print(
        f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}"
    )

    # Get datasets
    datasets = data_loader.get_dataset(
        max_length=cfg.max_length,
        normalize=cfg.normalize,
        train_indices=train_indices,
        val_indices=val_indices,
    )
    assert isinstance(datasets, tuple), (
        "Expected tuple when train_indices and val_indices are provided"
    )
    train_dataset, val_dataset = datasets

    # Create test dataset (with per-profile normalization)
    assert data_loader.sequences is not None
    test_sequences = [data_loader.sequences[i] for i in test_indices]
    test_dataset = FlowMatchingDataset(
        test_sequences,
        max_length=cfg.max_length,
        pad_token=cfg.pad_token,
        normalize=cfg.normalize,  # Each profile normalized independently
    )

    # Create data loaders
    train_loader = data_loader.get_dataloader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    val_loader = data_loader.get_dataloader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    test_loader = data_loader.get_dataloader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    # Create model
    print("Creating model...")
    model = TransformerModel(
        input_dim=cfg.input_dim,
        output_dim=cfg.output_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
        max_length=cfg.max_length,
        time_emb_dim=cfg.time_emb_dim,
        use_sequence_stats=cfg.use_sequence_stats,
        stats_dim=cfg.stats_dim,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
    )

    # Create scheduler
    scheduler = None
    if cfg.use_scheduler:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=cfg.scheduler_mode,  # type: ignore[arg-type]
            factor=cfg.scheduler_factor,
            patience=cfg.scheduler_patience,
            min_lr=cfg.scheduler_min_lr,
        )

    # Training loop
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    print("Starting training...")
    for epoch in range(cfg.num_epochs):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, device, cfg, train_dataset, wandb_run, epoch
        )
        train_losses.append(train_loss)

        # Evaluate
        if (epoch + 1) % cfg.eval_every == 0 or epoch == 0:
            val_metrics = evaluate_model(
                model, val_loader, device, val_dataset, wandb_run, config=cfg
            )
            val_loss = val_metrics["val/loss"]
            val_losses.append(val_loss)

            print(
                f"Epoch {epoch + 1}/{cfg.num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
            )

            # Update scheduler
            if scheduler is not None:
                scheduler.step(val_loss)

            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            if (epoch + 1) % cfg.checkpoint_every == 0 or is_best:
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_val_loss,
                    cfg,
                    cfg.checkpoints_dir,
                    is_best=is_best,
                )

        # Log epoch metrics
        if wandb_run is not None:
            wandb_run.log({"train/epoch_loss": train_loss, "epoch": epoch})

    # Final evaluation on test set
    print("Evaluating on test set...")
    test_metrics = evaluate_model(
        model, test_loader, device, test_dataset, wandb_run, config=cfg
    )
    print(f"Test Loss: {test_metrics['val/loss']:.6f}")

    # Save final results
    results = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "test_metrics": test_metrics,
        "best_val_loss": best_val_loss,
    }
    results_path = os.path.join(cfg.results_dir, "final_results.json")
    with open(results_path, "w") as f:
        json.dump(convert_to_json_serializable(results), f, indent=2)

    # Generate samples
    print("Generating samples...")
    generated_sequences, attention_masks = sample_sequences(
        model,
        cfg.num_eval_samples,
        cfg.max_length,
        device,
        train_dataset,
        cfg.ode_steps,
        wandb_run,
    )

    # Save samples
    samples_path = os.path.join(cfg.samples_dir, "generated_samples.npy")
    np.save(samples_path, generated_sequences, allow_pickle=True)
    print(f"Saved samples to {samples_path}")

    if wandb_run is not None:
        wandb_run.finish()

    print("Training complete!")


if __name__ == "__main__":
    main()
