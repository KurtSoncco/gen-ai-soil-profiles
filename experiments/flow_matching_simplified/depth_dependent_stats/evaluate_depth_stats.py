"""Evaluate depth-dependent statistics for generated profiles.

This script compares generated profiles to real data statistics,
computing depth-dependent μ(ln Vs) and σ(ln Vs) matching, as well
as vertical correlation structure.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from experiments.flow_matching_simplified import config as cfg_mod
    from experiments.flow_matching_simplified.depth_dependent_stats import (
        bin_by_depth,
        compute_profile_statistics,
        load_target_statistics,
        plot_depth_statistics,
        plot_vertical_correlations,
        reconstruct_vs_profile,
    )
except ImportError:
    # Fallback: try relative imports when running from the directory
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import config as cfg_mod  # type: ignore
    from depth_dependent_stats import (
        bin_by_depth,
        compute_profile_statistics,
        load_target_statistics,
        plot_depth_statistics,
        plot_vertical_correlations,
        reconstruct_vs_profile,
    )  # type: ignore


def evaluate_depth_statistics(
    generated_sequences: list,
    target_stats: dict,
    target_corrs: dict = None,
    bin_width: float = 10.0,
    max_depth: float = 500.0,
    correlation_lags: list = None,
    output_dir: str = None,
) -> dict:
    """Evaluate depth-dependent statistics match.

    Args:
        generated_sequences: List of generated [TTS, depth] sequences
        target_stats: Target depth statistics dictionary
        target_corrs: Optional target correlations dictionary
        bin_width: Depth bin width in meters
        max_depth: Maximum depth to consider
        correlation_lags: List of depth lags for correlation computation
        output_dir: Output directory for plots

    Returns:
        Dictionary with evaluation metrics
    """
    if correlation_lags is None:
        correlation_lags = [1.0, 2.0, 5.0, 10.0]

    print("=" * 80)
    print("Evaluating Depth-Dependent Statistics")
    print("=" * 80)
    print(f"Generated profiles: {len(generated_sequences)}")
    print(f"Bin width: {bin_width} m")
    print(f"Max depth: {max_depth} m")
    print()

    # Compute generated depth statistics
    print("Computing generated depth statistics...")
    bin_edges = np.arange(0, max_depth + bin_width, bin_width)

    all_depths_gen = []
    all_vs_gen = []

    for i, seq in enumerate(generated_sequences):
        if i % 1000 == 0:
            print(f"  Processing profile {i}/{len(generated_sequences)}...")

        try:
            # Reconstruct Vs profile
            depths, vs_values = reconstruct_vs_profile(seq, apply_expm1=False)

            if len(vs_values) > 0:
                # Map Vs values to layer center depths
                layer_depths = (depths[:-1] + depths[1:]) / 2.0
                all_depths_gen.extend(layer_depths)
                all_vs_gen.extend(vs_values)
        except Exception as e:
            if i < 10:  # Only print first few errors
                print(f"  Warning: Could not process profile {i}: {e}")
            continue

    all_depths_gen = np.array(all_depths_gen)
    all_vs_gen = np.array(all_vs_gen)

    print(
        f"Extracted {len(all_vs_gen)} Vs values from {len(generated_sequences)} profiles"
    )

    # Bin by depth
    bin_centers_gen, mean_ln_vs_gen, std_ln_vs_gen, bin_counts_gen = bin_by_depth(
        all_depths_gen, all_vs_gen, bin_edges, apply_log=True
    )

    gen_stats = {
        "bin_centers": bin_centers_gen,
        "mean_ln_vs": mean_ln_vs_gen,
        "std_ln_vs": std_ln_vs_gen,
        "bin_counts": bin_counts_gen,
        "bin_edges": bin_edges,
    }

    # Compute mismatch metrics
    valid_bins = (
        ~np.isnan(target_stats["mean_ln_vs"])
        & ~np.isnan(gen_stats["mean_ln_vs"])
        & ~np.isnan(target_stats["std_ln_vs"])
        & ~np.isnan(gen_stats["std_ln_vs"])
    )

    if valid_bins.sum() > 0:
        mean_mismatch = np.sqrt(
            (
                (
                    target_stats["mean_ln_vs"][valid_bins]
                    - gen_stats["mean_ln_vs"][valid_bins]
                )
                ** 2
            ).mean()
        )
        std_mismatch = np.sqrt(
            (
                (
                    target_stats["std_ln_vs"][valid_bins]
                    - gen_stats["std_ln_vs"][valid_bins]
                )
                ** 2
            ).mean()
        )
    else:
        mean_mismatch = np.nan
        std_mismatch = np.nan

    print("\nDepth Statistics Comparison:")
    print(f"  Mean ln(Vs) RMSE: {mean_mismatch:.6f}")
    print(f"  Std ln(Vs) RMSE:  {std_mismatch:.6f}")

    # Compute vertical correlations if target provided
    gen_corrs = None
    if target_corrs is not None:
        print("\nComputing generated vertical correlations...")

        # Create a simple dataset-like object for compatibility
        class SimpleDataset:
            def __init__(self, sequences):
                self.sequences = sequences
                self.normalize = False

            def __len__(self):
                return len(self.sequences)

            def denormalize_sequence(self, seq, idx):
                return seq

        simple_dataset = SimpleDataset(generated_sequences)
        gen_corrs = compute_profile_statistics(simple_dataset, lags=correlation_lags)

        print("\nVertical Correlation Comparison:")
        for lag in sorted(target_corrs["mean_correlations"].keys()):
            target_mean = target_corrs["mean_correlations"][lag]
            gen_mean = gen_corrs["mean_correlations"][lag]
            diff = abs(target_mean - gen_mean)
            print(
                f"  Lag {lag:4.1f}m: target={target_mean:6.4f}, gen={gen_mean:6.4f}, diff={diff:6.4f}"
            )

    # Create comparison plots
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

        # Plot depth statistics comparison
        depth_plot_path = os.path.join(output_dir, "depth_statistics_comparison.png")
        print(f"\nSaving depth statistics plot to {depth_plot_path}...")
        plot_depth_statistics(target_stats, gen_stats, depth_plot_path)

        # Plot correlation comparison if available
        if gen_corrs is not None:
            corr_plot_path = os.path.join(output_dir, "correlations_comparison.png")
            print(f"Saving correlation plot to {corr_plot_path}...")
            plot_vertical_correlations(target_corrs, gen_corrs, corr_plot_path)

    # Compile results
    results = {
        "mean_ln_vs_rmse": float(mean_mismatch)
        if not np.isnan(mean_mismatch)
        else None,
        "std_ln_vs_rmse": float(std_mismatch) if not np.isnan(std_mismatch) else None,
        "n_generated_profiles": len(generated_sequences),
        "n_vs_samples": len(all_vs_gen),
    }

    if gen_corrs is not None:
        results["correlation_errors"] = {}
        for lag in sorted(target_corrs["mean_correlations"].keys()):
            target_mean = target_corrs["mean_correlations"][lag]
            gen_mean = gen_corrs["mean_correlations"][lag]
            results["correlation_errors"][f"lag_{lag}m"] = float(
                abs(target_mean - gen_mean)
            )

    return results


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate depth-dependent statistics for generated profiles"
    )
    parser.add_argument(
        "--generated-profiles",
        type=str,
        required=True,
        help="Path to generated profiles .npy file",
    )
    parser.add_argument(
        "--target-stats",
        type=str,
        required=True,
        help="Path to target depth statistics .pkl file (or .npy for backward compatibility)",
    )
    parser.add_argument(
        "--target-corr",
        type=str,
        default=None,
        help="Path to target correlations .pkl file (or .npy for backward compatibility, optional)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots (default: plots_dir from config)",
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

    args = parser.parse_args()

    cfg = cfg_mod.cfg

    # Determine output directory
    if args.output_dir is None:
        output_dir = cfg.plots_dir
    else:
        output_dir = args.output_dir

    # Load generated profiles
    print(f"Loading generated profiles from {args.generated_profiles}...")
    generated_profiles_array = np.load(args.generated_profiles, allow_pickle=True)
    generated_sequences = []
    for seq in generated_profiles_array:
        if isinstance(seq, np.ndarray):
            generated_sequences.append(seq)
        else:
            generated_sequences.append(np.array(seq))

    print(f"Loaded {len(generated_sequences)} generated profiles")

    # Load target statistics
    print(f"Loading target statistics from {args.target_stats}...")
    target_stats, target_corrs = load_target_statistics(
        args.target_stats, args.target_corr
    )

    if args.target_corr is not None:
        print(f"Loaded target correlations from {args.target_corr}")
    else:
        target_corrs = None

    # Evaluate
    results = evaluate_depth_statistics(
        generated_sequences,
        target_stats,
        target_corrs=target_corrs,
        bin_width=args.bin_width,
        max_depth=args.max_depth,
        correlation_lags=args.lags,
        output_dir=output_dir,
    )

    # Print summary
    print("\n" + "=" * 80)
    print("Evaluation Summary")
    print("=" * 80)
    print(f"Mean ln(Vs) RMSE: {results['mean_ln_vs_rmse']:.6f}")
    print(f"Std ln(Vs) RMSE:  {results['std_ln_vs_rmse']:.6f}")
    if "correlation_errors" in results:
        print("\nCorrelation Errors:")
        for lag_key, error in results["correlation_errors"].items():
            print(f"  {lag_key}: {error:.6f}")

    print(f"\nPlots saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
