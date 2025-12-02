"""Generate and filter soil profiles with monotonicity and thickness constraints.

This script generates a large number of profiles and filters them based on:
1. Monotonic increase: depth values must be non-decreasing
2. Minimum thickness: each layer must have thickness >= threshold
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

try:
    from experiments.flow_matching_simplified import config as cfg_mod
    from experiments.flow_matching_simplified.data import (
        FlowMatchingDataLoader,
        FlowMatchingDataset,
    )
    from experiments.flow_matching_simplified.model import TransformerModel
    from experiments.flow_matching_simplified.train import sample_sequences
    from experiments.flow_matching_simplified.utils import (
        check_min_dt,
        check_vs_bounds,
    )
except ImportError:
    import config as cfg_mod
    from model import TransformerModel
    from train import sample_sequences
    from utils import check_min_dt, check_vs_bounds  # type: ignore

    from data import FlowMatchingDataLoader, FlowMatchingDataset  # type: ignore


def check_monotonic_increase(sequence: np.ndarray) -> bool:
    """Check if depth values are monotonically increasing (non-decreasing).

    Args:
        sequence: Array of shape (n, 2) with [TTS, depth] pairs

    Returns:
        True if tts values are non-decreasing, False otherwise
    """
    if len(sequence) < 2:
        return True

    tts = sequence[:, 0]
    return bool(np.all(tts[1:] >= tts[:-1]))


def check_min_thickness(sequence: np.ndarray, min_thickness: float) -> bool:
    """Check if all layers meet minimum thickness requirement.

    Args:
        sequence: Array of shape (n, 2) with [TTS, depth] pairs
        min_thickness: Minimum thickness threshold (in same units as depth)

    Returns:
        True if all layers have thickness >= min_thickness, False otherwise
    """
    if len(sequence) < 2:
        return True

    depths = sequence[:, 1]
    thicknesses = np.diff(depths)

    # All thicknesses must be >= min_thickness
    return bool(np.all(thicknesses >= min_thickness))


def check_non_negative_values(sequence: np.ndarray) -> bool:
    """Check if all TTS and depth values are non-negative.

    Args:
        sequence: Array of shape (n, 2) with [TTS, depth] pairs

    Returns:
        True if all TTS and depth values are >= 0, False otherwise
    """
    if len(sequence) == 0:
        return True

    tts = sequence[:, 0]
    depths = sequence[:, 1]

    # All TTS and depth values must be non-negative
    return bool(np.all(tts >= 0) and np.all(depths >= 0))


def filter_profiles(
    sequences: list[np.ndarray],
    min_thickness: float | None = None,
    require_monotonic: bool = True,
    vs_min: float | None = None,
    vs_max: float | None = None,
    min_dt: float | None = None,
    require_non_negative: bool = True,
) -> tuple[list[np.ndarray], dict[str, int]]:
    """Filter profiles based on monotonicity, thickness, Vs constraints, and non-negative values.

    Args:
        sequences: List of sequences, each of shape (n, 2) with [TTS, depth]
        min_thickness: Minimum thickness threshold (None to skip thickness check)
        require_monotonic: Whether to require monotonic increase in depth
        vs_min: Minimum allowed Vs value (None to skip Vs bounds check)
        vs_max: Maximum allowed Vs value (None to skip Vs bounds check)
        min_dt: Minimum Δt threshold (None to skip min_dt check)
        require_non_negative: Whether to require all TTS and depth values to be non-negative

    Returns:
        Tuple of (filtered_sequences, stats_dict) where stats_dict contains:
        - total: total number of sequences
        - passed_non_negative: number that passed non-negative check
        - passed_monotonic: number that passed monotonic check
        - passed_thickness: number that passed thickness check
        - passed_vs_bounds: number that passed Vs bounds check
        - passed_min_dt: number that passed min_dt check
        - passed_all: number that passed all checks
    """
    filtered = []
    stats = {
        "total": len(sequences),
        "passed_non_negative": 0,
        "passed_monotonic": 0,
        "passed_thickness": 0,
        "passed_vs_bounds": 0,
        "passed_min_dt": 0,
        "passed_all": 0,
    }

    for seq in sequences:
        # Check non-negative values (TTS and depth must be >= 0)
        if require_non_negative:
            if not check_non_negative_values(seq):
                continue
            stats["passed_non_negative"] += 1

        # Check monotonicity
        if require_monotonic:
            if not check_monotonic_increase(seq):
                continue
            stats["passed_monotonic"] += 1

        # Check thickness
        if min_thickness is not None:
            if not check_min_thickness(seq, min_thickness):
                continue
            stats["passed_thickness"] += 1

        # Check Vs bounds
        if vs_min is not None and vs_max is not None:
            if not check_vs_bounds(seq, vs_min, vs_max):
                continue
            stats["passed_vs_bounds"] += 1

        # Check min_dt
        if min_dt is not None:
            if not check_min_dt(seq, min_dt):
                continue
            stats["passed_min_dt"] += 1

        # If we get here, profile passed all checks
        filtered.append(seq)
        stats["passed_all"] += 1

    return filtered, stats


def load_model(
    checkpoint_path: str, device: torch.device, config: cfg_mod.Config
) -> TransformerModel:
    """Load model from checkpoint."""
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

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        raise KeyError(
            f"Checkpoint missing model state dict. Keys: {checkpoint.keys()}"
        )

    return model


def generate_filtered_profiles(
    model: TransformerModel,
    dataset: FlowMatchingDataset,
    device: torch.device,
    target_count: int,
    batch_size: int = 1000,
    min_thickness: float | None = None,
    require_monotonic: bool = True,
    vs_min: float | None = None,
    vs_max: float | None = None,
    min_dt: float | None = None,
    ode_steps: int = 50,
    max_attempts: int | None = None,
) -> tuple[list[np.ndarray], dict[str, int]]:
    """Generate profiles in batches and filter them until we have enough valid ones.

    Args:
        model: Trained transformer model
        dataset: Dataset for sequence statistics and denormalization
        device: PyTorch device
        target_count: Target number of valid profiles to generate
        batch_size: Number of profiles to generate per batch
        min_thickness: Minimum thickness threshold (None to skip)
        require_monotonic: Whether to require monotonic increase
        ode_steps: Number of ODE integration steps
        max_attempts: Maximum number of batches to generate (None for unlimited)

    Returns:
        Tuple of (valid_profiles, cumulative_stats) where stats contains:
        - total_generated: total profiles generated
        - total_valid: total valid profiles collected
        - batches_generated: number of batches generated
    """
    valid_profiles = []
    cumulative_stats = {
        "total_generated": 0,
        "total_valid": 0,
        "batches_generated": 0,
    }

    model.eval()

    pbar = tqdm(total=target_count, desc="Generating valid profiles")

    attempt = 0
    while len(valid_profiles) < target_count:
        if max_attempts is not None and attempt >= max_attempts:
            print(f"\nReached max_attempts ({max_attempts}). Stopping generation.")
            break

        # Generate a batch
        batch_size_actual = min(batch_size, target_count - len(valid_profiles))
        generated_sequences_array, _ = sample_sequences(
            model,
            batch_size_actual,
            dataset.max_length,
            device,
            dataset,
            ode_steps,
            wandb_run=None,
        )

        # Convert to list
        batch_sequences = []
        for seq in generated_sequences_array:
            if isinstance(seq, np.ndarray):
                batch_sequences.append(seq)
            else:
                batch_sequences.append(np.array(seq))

        # Filter the batch
        filtered_batch, batch_stats = filter_profiles(
            batch_sequences,
            min_thickness=min_thickness,
            require_monotonic=require_monotonic,
            vs_min=vs_min,
            vs_max=vs_max,
            min_dt=min_dt,
            require_non_negative=True,  # Always filter negative values
        )

        # Add valid profiles
        valid_profiles.extend(filtered_batch)

        # Update stats
        cumulative_stats["total_generated"] += batch_stats["total"]
        cumulative_stats["total_valid"] += batch_stats["passed_all"]
        cumulative_stats["batches_generated"] += 1

        # Update progress bar
        acceptance_rate = (
            batch_stats["passed_all"] / batch_stats["total"]
            if batch_stats["total"] > 0
            else 0.0
        )
        pbar.update(len(filtered_batch))
        pbar.set_postfix(
            {
                "valid": len(valid_profiles),
                "accept_rate": f"{acceptance_rate:.2%}",
                "batches": cumulative_stats["batches_generated"],
            }
        )

        attempt += 1

    pbar.close()

    # Trim to exact target count if we exceeded it
    if len(valid_profiles) > target_count:
        valid_profiles = valid_profiles[:target_count]

    return valid_profiles, cumulative_stats


def save_profiles(profiles: list[np.ndarray], output_path: str):
    """Save profiles to disk as numpy array.

    Args:
        profiles: List of sequences, each of shape (n, 2)
        output_path: Path to save the profiles
    """
    # Save as numpy array (object dtype to handle variable lengths)
    profiles_array = np.array(profiles, dtype=object)
    np.save(output_path, profiles_array)
    print(f"Saved {len(profiles)} profiles to {output_path}")


def main():
    """Main function for generating and filtering profiles."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate and filter soil profiles with constraints"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (default: uses latest.pt from config)",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=10000,
        help="Target number of valid profiles to generate (default: 10000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of profiles to generate per batch (default: 1000)",
    )
    parser.add_argument(
        "--min-thickness",
        type=float,
        default=2,
        help="Minimum layer thickness threshold (default: None, no filtering)",
    )
    parser.add_argument(
        "--no-monotonic-check",
        action="store_true",
        help="Disable monotonic increase check (default: enabled)",
    )
    parser.add_argument(
        "--min-vs",
        type=float,
        default=None,
        help="Minimum Vs value (m/s) (default: uses config.vs_min)",
    )
    parser.add_argument(
        "--max-vs",
        type=float,
        default=None,
        help="Maximum Vs value (m/s) (default: uses config.vs_max)",
    )
    parser.add_argument(
        "--min-dt",
        type=float,
        default=None,
        help="Minimum Δt threshold (default: uses config.min_dt)",
    )
    parser.add_argument(
        "--ode-steps",
        type=int,
        default=50,
        help="Number of ODE integration steps (default: 50)",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Maximum number of batches to generate (default: unlimited)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: auto-generated in samples_dir)",
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

    # Load model
    print(f"Loading model from {checkpoint_path}")
    model = load_model(checkpoint_path, device, cfg)

    # Load data
    print("Loading data...")
    data_loader = FlowMatchingDataLoader(data_path=Path(cfg.data_path))
    data_loader.load_data()

    assert data_loader.sequences is not None, "Sequences must be loaded"
    n_total = len(data_loader.sequences)

    # Create train/val splits (same as train.py)
    all_indices = torch.randperm(n_total)
    n_train = int(cfg.train_val_test_split[0] * n_total)
    n_val = int(cfg.train_val_test_split[1] * n_total)

    train_indices = all_indices[:n_train].tolist()
    val_indices = all_indices[n_train : n_train + n_val].tolist()

    # Get training dataset
    datasets = data_loader.get_dataset(
        max_length=cfg.max_length,
        normalize=cfg.normalize,
        train_indices=train_indices,
        val_indices=val_indices,
    )
    assert isinstance(datasets, tuple)
    train_dataset, _ = datasets

    # Determine Vs constraints (use args if provided, else config defaults)
    vs_min = args.min_vs if args.min_vs is not None else cfg.vs_min
    vs_max = args.max_vs if args.max_vs is not None else cfg.vs_max
    min_dt = args.min_dt if args.min_dt is not None else cfg.min_dt

    # Generate and filter profiles
    print(f"\nGenerating {args.target_count} valid profiles...")
    print("Constraints:")
    print(
        f"  - Monotonic increase: {'required' if not args.no_monotonic_check else 'disabled'}"
    )
    print(
        f"  - Minimum thickness: {args.min_thickness if args.min_thickness else 'none'}"
    )
    print(f"  - Vs range: [{vs_min}, {vs_max}] m/s")
    print(f"  - Minimum Δt: {min_dt}")
    print("  - Non-negative values: required (TTS >= 0, depth >= 0)")
    print()

    valid_profiles, stats = generate_filtered_profiles(
        model=model,
        dataset=train_dataset,
        device=device,
        target_count=args.target_count,
        batch_size=args.batch_size,
        min_thickness=args.min_thickness,
        require_monotonic=not args.no_monotonic_check,
        vs_min=vs_min,
        vs_max=vs_max,
        min_dt=min_dt,
        ode_steps=args.ode_steps,
        max_attempts=args.max_attempts,
    )

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Generation Summary")
    print("=" * 60)
    print(f"Total profiles generated: {stats['total_generated']}")
    print(f"Valid profiles collected: {len(valid_profiles)}")
    print(f"Batches generated: {stats['batches_generated']}")
    if stats["total_generated"] > 0:
        overall_acceptance = len(valid_profiles) / stats["total_generated"]
        print(f"Overall acceptance rate: {overall_acceptance:.2%}")
    print("=" * 60)

    # Save profiles
    if args.output is None:
        os.makedirs(cfg.samples_dir, exist_ok=True)
        output_path = os.path.join(
            cfg.samples_dir,
            f"filtered_profiles_n{len(valid_profiles)}.npy",
        )
    else:
        output_path = args.output

    save_profiles(valid_profiles, output_path)

    print(f"\nDone! Valid profiles saved to: {output_path}")


if __name__ == "__main__":
    main()
