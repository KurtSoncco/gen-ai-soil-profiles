"""Extended training script with depth-dependent statistics and vertical correlation losses.

This script extends the base training script to include depth-aware regularization
that matches depth-dependent μ(ln Vs) and σ(ln Vs) trends, as well as vertical
correlation structure.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

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
    from experiments.flow_matching_simplified.train import (
        convert_to_json_serializable,
        evaluate_model,
        sample_sequences,
        save_checkpoint,
        set_seed,
        train_epoch,
    )
    from experiments.flow_matching_simplified.utils import compute_vs_penalty
    from experiments.flow_matching_simplified.depth_dependent_stats import (
        compute_depth_statistics_loss,
        compute_vertical_correlation_loss,
        load_target_statistics,
    )
    from experiments.flow_matching_simplified.depth_dependent_stats.config_extensions import (
        extend_config,
    )
except ImportError:
    # Fallback: try relative imports when running from the directory
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import config as cfg_mod  # type: ignore
    from model import TransformerModel  # type: ignore
    from train import (
        convert_to_json_serializable,
        evaluate_model,
        sample_sequences,
        save_checkpoint,
        set_seed,
        train_epoch,
    )  # type: ignore
    from utils import compute_vs_penalty  # type: ignore
    from depth_dependent_stats import (
        compute_depth_statistics_loss,
        compute_vertical_correlation_loss,
        load_target_statistics,
    )  # type: ignore
    from depth_dependent_stats.config_extensions import extend_config  # type: ignore
    from data import FlowMatchingDataLoader, FlowMatchingDataset  # type: ignore

# Global wandb run variable
wandb_run = None


def compute_flow_matching_loss_with_depth(
    model: TransformerModel,
    u1: torch.Tensor,
    attention_mask: torch.Tensor,
    sequence_stats: torch.Tensor,
    device: torch.device,
    config: cfg_mod.Config,
    target_mean_ln_vs: torch.Tensor = None,
    target_std_ln_vs: torch.Tensor = None,
    target_correlations: dict = None,
    bin_edges: np.ndarray = None,
) -> tuple[torch.Tensor, dict]:
    """
    Compute flow matching loss with optional depth-aware regularization.
    
    Args:
        model: Transformer model
        u1: Real data sequences (batch_size, max_length, 2)
        attention_mask: Boolean mask (batch_size, max_length)
        sequence_stats: Sequence statistics (batch_size, 4)
        device: PyTorch device
        config: Config object
        target_mean_ln_vs: Optional target mean ln(Vs) per depth bin
        target_std_ln_vs: Optional target std ln(Vs) per depth bin
        target_correlations: Optional target correlations {lag: expected_corr}
        bin_edges: Optional depth bin edges
    
    Returns:
        Tuple of (total_loss, loss_dict)
    """
    batch_size, max_length, _ = u1.shape

    # u0: Noise sequences
    u0 = torch.randn_like(u1).to(device)
    u0 = u0 * attention_mask.unsqueeze(-1).float()

    # t: Random time from [0, 1]
    t = torch.rand(batch_size, 1).to(device)

    # Broadcast t for interpolation
    t_broadcast = t.unsqueeze(-1).expand(-1, max_length, -1).expand(-1, -1, 2)

    # ut: Interpolated profile at time t
    ut = (1 - t_broadcast) * u0 + t_broadcast * u1

    # target_v: The target vector field
    target_v = torch.where(
        t_broadcast < 0.999,
        (u1 - ut) / torch.clamp(1 - t_broadcast, min=1e-6),
        torch.zeros_like(u1),
    )

    # Forward pass
    predicted_v = model(ut, attention_mask, t, sequence_stats)

    # Compute MSE loss, masked by attention_mask
    loss_per_token = (predicted_v - target_v) ** 2
    loss_per_token = loss_per_token.mean(dim=-1)
    mask_float = attention_mask.float()
    masked_loss = loss_per_token * mask_float
    reconstruction_loss = masked_loss.sum() / mask_float.sum()

    # Add Vs regularization penalty if enabled
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
        reconstruction_loss = reconstruction_loss + config.vs_penalty_weight * vs_penalty

    # Initialize loss dict
    loss_dict = {
        "reconstruction_loss": reconstruction_loss.item(),
        "depth_stats_loss": 0.0,
        "vertical_corr_loss": 0.0,
    }

    total_loss = reconstruction_loss

    # Add depth-dependent statistics loss if enabled and targets provided
    if (
        config.use_depth_stats_loss
        and target_mean_ln_vs is not None
        and target_std_ln_vs is not None
        and bin_edges is not None
    ):
        # Use u1 (real data at t=1) for statistics computation
        # In practice, we could also use ut or predicted profiles
        min_bin_count = getattr(config, "min_bin_count", 50)
        mean_weight_ratio = getattr(config, "mean_weight_ratio", 2.0)
        depth_stats_loss = compute_depth_statistics_loss(
            u1,
            attention_mask,
            sequence_stats,
            target_mean_ln_vs,
            target_std_ln_vs,
            bin_edges,
            normalize=config.normalize,
            weight=config.depth_stats_weight,
            min_bin_count=min_bin_count,
            mean_weight_ratio=mean_weight_ratio,
            device=device,
        )
        total_loss = total_loss + depth_stats_loss
        loss_dict["depth_stats_loss"] = depth_stats_loss.item()

    # Add vertical correlation loss if enabled and targets provided
    if config.use_vertical_corr_loss and target_correlations is not None:
        short_lag_weight = getattr(config, "short_lag_weight", 2.0)
        vertical_corr_loss = compute_vertical_correlation_loss(
            u1,
            attention_mask,
            sequence_stats,
            target_correlations,
            lags=config.correlation_lags,
            normalize=config.normalize,
            weight=config.vertical_corr_weight,
            short_lag_weight=short_lag_weight,
            device=device,
        )
        total_loss = total_loss + vertical_corr_loss
        loss_dict["vertical_corr_loss"] = vertical_corr_loss.item()

    loss_dict["total_loss"] = total_loss.item()

    return total_loss, loss_dict


def train_epoch_with_depth_losses(
    model: TransformerModel,
    train_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    config: cfg_mod.Config,
    dataset: FlowMatchingDataset,
    target_mean_ln_vs: torch.Tensor = None,
    target_std_ln_vs: torch.Tensor = None,
    target_correlations: dict = None,
    bin_edges: np.ndarray = None,
    wandb_run=None,
    epoch: int = 0,
) -> float:
    """
    Train for one epoch with depth-aware losses.
    
    Args:
        model: Transformer model
        train_loader: Training data loader
        optimizer: Optimizer
        device: PyTorch device
        config: Configuration object
        dataset: Dataset
        target_mean_ln_vs: Target mean ln(Vs) per depth bin
        target_std_ln_vs: Target std ln(Vs) per depth bin
        target_correlations: Target correlations {lag: expected_corr}
        bin_edges: Depth bin edges
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

        # Compute loss with depth-aware regularization
        loss, loss_dict = compute_flow_matching_loss_with_depth(
            model,
            tokens,
            attention_mask,
            sequence_stats,
            device,
            config,
            target_mean_ln_vs=target_mean_ln_vs,
            target_std_ln_vs=target_std_ln_vs,
            target_correlations=target_correlations,
            bin_edges=bin_edges,
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
        pbar.set_postfix(
            {
                "loss": loss.item(),
                "depth": loss_dict["depth_stats_loss"],
                "corr": loss_dict["vertical_corr_loss"],
            }
        )

        # Log to wandb
        if wandb_run is not None and batch_idx % config.log_every == 0:
            log_dict = {
                "train/loss": loss.item(),
                "train/reconstruction_loss": loss_dict["reconstruction_loss"],
                "train/depth_stats_loss": loss_dict["depth_stats_loss"],
                "train/vertical_corr_loss": loss_dict["vertical_corr_loss"],
                "train/epoch": epoch,
                "train/batch": batch_idx,
                "train/lr": optimizer.param_groups[0]["lr"],
            }
            wandb_run.log(log_dict)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def main() -> None:
    """Main training function with depth-aware losses."""
    global wandb_run

    parser = argparse.ArgumentParser(
        description="Train flow matching model with depth-dependent statistics losses"
    )
    parser.add_argument(
        "--target-stats",
        type=str,
        default=None,
        help="Path to pre-computed depth statistics .pkl file (or .npy for backward compatibility)",
    )
    parser.add_argument(
        "--target-corr",
        type=str,
        default=None,
        help="Path to pre-computed correlations .pkl file (or .npy for backward compatibility)",
    )
    parser.add_argument(
        "--depth-stats-weight",
        type=float,
        default=None,
        help="Weight for depth statistics loss (overrides config)",
    )
    parser.add_argument(
        "--vertical-corr-weight",
        type=float,
        default=None,
        help="Weight for vertical correlation loss (overrides config)",
    )

    args = parser.parse_args()

    cfg = cfg_mod.cfg
    set_seed(cfg.seed)

    # Extend config with depth statistics hyperparameters
    extend_config(cfg)

    # Override weights if provided
    if args.depth_stats_weight is not None:
        cfg.depth_stats_weight = args.depth_stats_weight
    if args.vertical_corr_weight is not None:
        cfg.vertical_corr_weight = args.vertical_corr_weight

    # Load target statistics if provided
    target_mean_ln_vs = None
    target_std_ln_vs = None
    target_correlations = None
    bin_edges = None

    if args.target_stats is not None:
        print(f"Loading target statistics from {args.target_stats}...")
        depth_stats, correlations = load_target_statistics(
            args.target_stats, args.target_corr
        )

        # Convert to tensors
        bin_edges = depth_stats["bin_edges"]
        target_mean_ln_vs = torch.from_numpy(depth_stats["mean_ln_vs"]).float()
        target_std_ln_vs = torch.from_numpy(depth_stats["std_ln_vs"]).float()

        if args.target_corr is not None:
            target_correlations = correlations["mean_correlations"]

        print("Target statistics loaded successfully")
        print(f"  Bin edges: {len(bin_edges)-1} bins")
        print(f"  Correlation lags: {list(target_correlations.keys()) if target_correlations else 'None'}")
    else:
        print("Warning: No target statistics provided. Depth-aware losses will be disabled.")
        cfg.use_depth_stats_loss = False
        cfg.use_vertical_corr_loss = False

    # Create output directories
    os.makedirs(cfg.checkpoints_dir, exist_ok=True)
    os.makedirs(cfg.plots_dir, exist_ok=True)
    os.makedirs(cfg.results_dir, exist_ok=True)
    os.makedirs(cfg.samples_dir, exist_ok=True)

    # Initialize wandb
    try:
        if wandb is not None:
            wandb_name = cfg.wandb_name or f"flow_matching_depth_stats_{cfg.seed}"
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

    # Move target statistics to device if available
    if target_mean_ln_vs is not None:
        target_mean_ln_vs = target_mean_ln_vs.to(device)
        target_std_ln_vs = target_std_ln_vs.to(device)

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

    # Create test dataset
    assert data_loader.sequences is not None
    test_sequences = [data_loader.sequences[i] for i in test_indices]
    test_dataset = FlowMatchingDataset(
        test_sequences,
        max_length=cfg.max_length,
        pad_token=cfg.pad_token,
        normalize=cfg.normalize,
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
        use_depth_conv=getattr(cfg, "use_depth_conv", True),
        depth_conv_kernel_size=getattr(cfg, "depth_conv_kernel_size", 3),
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
    print(f"  Depth stats loss: {'enabled' if cfg.use_depth_stats_loss else 'disabled'}")
    print(f"  Vertical corr loss: {'enabled' if cfg.use_vertical_corr_loss else 'disabled'}")

    for epoch in range(cfg.num_epochs):
        # Train with depth-aware losses
        train_loss = train_epoch_with_depth_losses(
            model,
            train_loader,
            optimizer,
            device,
            cfg,
            train_dataset,
            target_mean_ln_vs=target_mean_ln_vs,
            target_std_ln_vs=target_std_ln_vs,
            target_correlations=target_correlations,
            bin_edges=bin_edges,
            wandb_run=wandb_run,
            epoch=epoch,
        )
        train_losses.append(train_loss)

        # Evaluate (using standard evaluation, depth losses only in training)
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

