"""Evaluation script for Flow Matching with Variable-Length Breakpoints.

This script provides comprehensive evaluation of trained models, including
metrics computation and comparison with real data.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

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
    from experiments.flow_matching_simplified.train import evaluate_model
    from experiments.flow_matching_simplified.utils import compute_sequence_statistics
except ImportError:  # fallback when running as script
    import config as cfg_mod  # type: ignore
    from model import TransformerModel  # type: ignore
    from train import evaluate_model  # type: ignore
    from utils import compute_sequence_statistics  # type: ignore

    from data import FlowMatchingDataLoader, FlowMatchingDataset  # type: ignore


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


def load_checkpoint(
    checkpoint_path: str, device: torch.device, config: cfg_mod.Config
) -> tuple[TransformerModel, int]:
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: PyTorch device
        config: Configuration object

    Returns:
        Tuple of (model, epoch)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model
    model = TransformerModel(
        input_dim=config.input_dim,
        output_dim=config.output_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
        max_length=config.max_length,
        time_emb_dim=config.time_emb_dim,
        use_sequence_stats=config.use_sequence_stats,
        stats_dim=config.stats_dim,
    ).to(device)

    model.load_state_dict(checkpoint["model"])
    model.eval()

    epoch = checkpoint.get("epoch", 0)
    print(f"Loaded model from epoch {epoch}")

    return model, epoch


def main() -> None:
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate Flow Matching model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file (default: latest.pt)",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="Wandb project name (if None, no wandb logging)",
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="Wandb run name (optional)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results",
    )

    args = parser.parse_args()

    cfg = cfg_mod.cfg
    device = torch.device(cfg.device)

    # Determine checkpoint path
    if args.checkpoint is None:
        checkpoint_path = os.path.join(cfg.checkpoints_dir, "latest.pt")
    else:
        checkpoint_path = args.checkpoint

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load model first to get epoch
    print(f"Loading model from {checkpoint_path}")
    model, epoch = load_checkpoint(checkpoint_path, device, cfg)

    # Initialize wandb if project provided
    wandb_run = None
    if args.wandb_project and wandb is not None:
        wandb_name = args.wandb_name or f"evaluation_epoch_{epoch}"
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=wandb_name,
            config=vars(cfg),
        )
        print("[info] wandb initialized for evaluation")

    # Load data
    print("Loading data...")
    data_loader = FlowMatchingDataLoader(data_path=Path(cfg.data_path))
    data_loader.load_data()

    assert data_loader.sequences is not None
    n_total = len(data_loader.sequences)

    # Create splits (same as training)
    all_indices = torch.randperm(n_total)
    n_train = int(cfg.train_val_test_split[0] * n_total)
    n_val = int(cfg.train_val_test_split[1] * n_total)

    train_indices = all_indices[:n_train].tolist()
    val_indices = all_indices[n_train : n_train + n_val].tolist()
    test_indices = all_indices[n_train + n_val :].tolist()

    # Get datasets
    datasets = data_loader.get_dataset(
        max_length=cfg.max_length,
        normalize=cfg.normalize,
        train_indices=train_indices,
        val_indices=val_indices,
    )
    assert isinstance(datasets, tuple)
    train_dataset, val_dataset = datasets

    # Create test dataset
    test_sequences = [data_loader.sequences[i] for i in test_indices]
    test_dataset = FlowMatchingDataset(
        test_sequences,
        max_length=cfg.max_length,
        pad_token=cfg.pad_token,
        normalize=cfg.normalize,
    )

    # Create data loaders
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

    # Evaluate on validation set
    print("Evaluating on validation set...")
    val_metrics = evaluate_model(model, val_loader, device, val_dataset, wandb_run)
    print(f"Validation Loss: {val_metrics['val/loss']:.6f}")

    # Evaluate on test set
    print("Evaluating on test set...")
    test_metrics = evaluate_model(model, test_loader, device, test_dataset, wandb_run)
    print(f"Test Loss: {test_metrics['val/loss']:.6f}")

    # Compute sequence statistics
    real_stats = compute_sequence_statistics(test_dataset.sequences)
    print("\nReal data statistics:")
    print(
        f"  Mean length: {real_stats['mean_length']:.2f} ± {real_stats['std_length']:.2f}"
    )
    print(f"  Mean TS: {real_stats['mean_ts']:.4f} ± {real_stats['std_ts']:.4f}")
    print(
        f"  Mean depth: {real_stats['mean_depth']:.4f} ± {real_stats['std_depth']:.4f}"
    )

    # Save results
    output_dir = args.output_dir or cfg.results_dir
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "epoch": epoch,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "real_statistics": real_stats,
    }

    results_path = os.path.join(output_dir, f"evaluation_epoch_{epoch}.json")
    with open(results_path, "w") as f:
        json.dump(convert_to_json_serializable(results), f, indent=2)

    print(f"\nResults saved to {results_path}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
