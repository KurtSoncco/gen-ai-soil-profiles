"""Sampling script for Flow Matching with Variable-Length Breakpoints.

Generate sequences using a trained flow matching model.
Uses ODE integration to sample from the learned vector field.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

try:
    import wandb
except ImportError:
    wandb = None

from experiments.flow_matching_simplified import config as cfg_mod
from experiments.flow_matching_simplified.data import FlowMatchingDataLoader, FlowMatchingDataset
from experiments.flow_matching_simplified.model import TransformerModel
from experiments.flow_matching_simplified.train import sample_sequences
from experiments.flow_matching_simplified.utils import compute_sequence_statistics


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
    """Main sampling function."""
    parser = argparse.ArgumentParser(
        description="Generate sequences using trained Flow Matching model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file (default: latest.pt)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--ode_steps",
        type=int,
        default=None,
        help="Number of ODE integration steps (default: from config)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output path for samples (default: samples_dir/generated_samples.npy)",
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

    # Initialize wandb if project provided
    wandb_run = None
    if args.wandb_project and wandb is not None:
        wandb_name = args.wandb_name or f"sampling_epoch_{epoch}"
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=wandb_name,
            config=vars(cfg),
        )
        print(f"[info] wandb initialized for sampling")

    # Load model
    print(f"Loading model from {checkpoint_path}")
    model, epoch = load_checkpoint(checkpoint_path, device, cfg)

    # Load training dataset for sequence statistics
    print("Loading training data for sequence statistics...")
    data_loader = FlowMatchingDataLoader(data_path=Path(cfg.data_path))
    data_loader.load_data()

    assert data_loader.sequences is not None
    n_total = len(data_loader.sequences)

    # Create train split (same as training)
    all_indices = torch.randperm(n_total)
    n_train = int(cfg.train_val_test_split[0] * n_total)
    train_indices = all_indices[:n_train].tolist()

    # Get training dataset
    train_dataset, _ = data_loader.get_dataset(
        max_length=cfg.max_length,
        normalize=cfg.normalize,
        train_indices=train_indices,
        val_indices=all_indices[n_train : n_train + 1].tolist(),  # Dummy val
    )

    # Generate samples
    print(f"Generating {args.num_samples} samples...")
    ode_steps = args.ode_steps or cfg.ode_steps
    generated_sequences, attention_masks = sample_sequences(
        model,
        args.num_samples,
        cfg.max_length,
        device,
        train_dataset,
        ode_steps,
        wandb_run,
    )

    # Compute statistics
    gen_stats = compute_sequence_statistics(list(generated_sequences))
    print("\nGenerated sequences statistics:")
    print(f"  Mean length: {gen_stats['mean_length']:.2f} ± {gen_stats['std_length']:.2f}")
    print(f"  Mean TS: {gen_stats['mean_ts']:.4f} ± {gen_stats['std_ts']:.4f}")
    print(f"  Mean depth: {gen_stats['mean_depth']:.4f} ± {gen_stats['std_depth']:.4f}")

    # Save samples
    if args.output_path is None:
        os.makedirs(cfg.samples_dir, exist_ok=True)
        output_path = os.path.join(cfg.samples_dir, f"generated_samples_epoch_{epoch}.npy")
    else:
        output_path = args.output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    np.save(output_path, generated_sequences, allow_pickle=True)
    print(f"\nSaved {args.num_samples} samples to {output_path}")

    # Save statistics
    stats_path = output_path.replace(".npy", "_stats.json")

    with open(stats_path, "w") as f:
        json.dump(convert_to_json_serializable(gen_stats), f, indent=2)
    print(f"Saved statistics to {stats_path}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()

