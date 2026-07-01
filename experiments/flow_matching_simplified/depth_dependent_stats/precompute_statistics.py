"""Precompute real data statistics for depth-dependent training.

This script computes depth-dependent statistics and vertical correlations
from the training dataset and saves them for use during training.
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from experiments.flow_matching_simplified import config as cfg_mod
    from experiments.flow_matching_simplified.data import (
        FlowMatchingDataLoader,
        FlowMatchingDataset,
    )
    from experiments.flow_matching_simplified.depth_dependent_stats import (
        compute_profile_statistics,
        compute_real_data_statistics,
        plot_depth_statistics,
        plot_vertical_correlations,
    )
except ImportError:
    # Fallback: try relative imports when running from the directory
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import config as cfg_mod  # type: ignore
    from depth_dependent_stats import (
        compute_profile_statistics,
        compute_real_data_statistics,
        plot_depth_statistics,
        plot_vertical_correlations,
    )

    from data import FlowMatchingDataLoader, FlowMatchingDataset  # type: ignore


def main():
    """Main function to compute and save real data statistics."""
    parser = argparse.ArgumentParser(
        description="Precompute real data statistics for depth-dependent training"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for statistics files (default: depth_dependent_stats/)",
    )
    parser.add_argument(
        "--bin-width",
        type=float,
        default=10.0,
        help="Depth bin width in meters (default: 10.0)",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=500.0,
        help="Maximum depth to consider in meters (default: 500.0)",
    )
    parser.add_argument(
        "--lags",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 5.0, 10.0],
        help="Depth lags for correlation computation (default: 1.0 2.0 5.0 10.0)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to data parquet file (default: from config)",
    )

    args = parser.parse_args()

    cfg = cfg_mod.cfg

    # Determine output directory
    if args.output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine data path
    data_path = Path(args.data_path) if args.data_path else Path(cfg.data_path)

    print("=" * 80)
    print("Precomputing Real Data Statistics")
    print("=" * 80)
    print(f"Data path: {data_path}")
    print(f"Output directory: {output_dir}")
    print(f"Bin width: {args.bin_width} m")
    print(f"Max depth: {args.max_depth} m")
    print(f"Correlation lags: {args.lags} m")
    print()

    # Load dataset
    print("Loading dataset...")
    data_loader = FlowMatchingDataLoader(data_path=data_path)
    data_loader.load_data()

    assert data_loader.sequences is not None, "Sequences must be loaded"
    n_total = len(data_loader.sequences)
    print(f"Loaded {n_total} profiles")

    # Create train/val/test splits (same as training)
    print("Creating train/val/test splits...")
    all_indices = torch.randperm(n_total)
    n_train = int(cfg.train_val_test_split[0] * n_total)

    train_indices = all_indices[:n_train].tolist()

    train_sequences = [data_loader.sequences[i] for i in train_indices]
    print(f"Training set: {len(train_sequences)} profiles")

    # Create training dataset
    train_dataset = FlowMatchingDataset(
        train_sequences,
        max_length=cfg.max_length,
        pad_token=cfg.pad_token,
        normalize=cfg.normalize,
    )

    # Compute depth-dependent statistics
    print("\n" + "=" * 80)
    print("Computing depth-dependent statistics...")
    print("=" * 80)
    real_stats = compute_real_data_statistics(
        train_dataset, bin_width=args.bin_width, max_depth=args.max_depth
    )

    print("\nDepth Statistics Summary:")
    valid_bins = ~np.isnan(real_stats["mean_ln_vs"])
    if valid_bins.any():
        print(f"  Valid bins: {valid_bins.sum()}/{len(valid_bins)}")
        print(
            f"  Mean ln(Vs) range: [{real_stats['mean_ln_vs'][valid_bins].min():.2f}, {real_stats['mean_ln_vs'][valid_bins].max():.2f}]"
        )
        print(
            f"  Std ln(Vs) range: [{real_stats['std_ln_vs'][valid_bins].min():.2f}, {real_stats['std_ln_vs'][valid_bins].max():.2f}]"
        )
        print(f"  Total Vs samples: {len(real_stats['all_vs'])}")

    # Compute vertical correlations
    print("\n" + "=" * 80)
    print("Computing vertical correlations...")
    print("=" * 80)
    real_corrs = compute_profile_statistics(train_dataset, lags=args.lags)

    print("\nVertical Correlation Summary:")
    for lag in sorted(real_corrs["mean_correlations"].keys()):
        mean_corr = real_corrs["mean_correlations"][lag]
        std_corr = real_corrs["std_correlations"][lag]
        n_profiles = len(real_corrs["lag_correlations"][lag])
        print(
            f"  Lag {lag:4.1f}m: mean={mean_corr:6.4f}, std={std_corr:6.4f}, "
            f"n_profiles={n_profiles}"
        )

    # Save statistics
    stats_path = output_dir / "real_depth_stats.pkl"
    corr_path = output_dir / "real_correlations.pkl"

    print("\n" + "=" * 80)
    print("Saving statistics...")
    print("=" * 80)
    with open(stats_path, "wb") as f:
        pickle.dump(real_stats, f)
    print(f"Saved depth statistics to: {stats_path}")

    with open(corr_path, "wb") as f:
        pickle.dump(real_corrs, f)
    print(f"Saved correlations to: {corr_path}")

    # Create initial visualization plots
    print("\n" + "=" * 80)
    print("Creating visualization plots...")
    print("=" * 80)

    # Plot depth statistics (baseline: real data only, no comparison)
    # Passing same data twice triggers baseline mode, showing only "Real Data" label
    depth_plot_path = output_dir / "depth_stats_initial.png"
    plot_depth_statistics(real_stats, real_stats, str(depth_plot_path))
    print(f"Saved depth statistics plot to: {depth_plot_path}")

    # Plot correlations (baseline: real data only, no comparison)
    # Passing same data twice triggers baseline mode, showing only "Real Data" label
    corr_plot_path = output_dir / "correlations_initial.png"
    plot_vertical_correlations(real_corrs, real_corrs, str(corr_plot_path))
    print(f"Saved correlation plot to: {corr_plot_path}")

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)
    print("\nStatistics files saved to:")
    print(f"  - {stats_path}")
    print(f"  - {corr_path}")
    print("\nUse these files with train_with_depth_losses.py:")
    print(f"  --target-stats {stats_path}")
    print(f"  --target-corr {corr_path}")
    print("\nNote: Files are saved as .pkl (pickle) format for better type safety.")


if __name__ == "__main__":
    main()
