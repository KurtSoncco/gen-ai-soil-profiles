"""Compute and visualize depth-dependent Vs statistics (mean and variance).

Used for both real data analysis and model validation.
Adapted to work with [TTS, depth] sequences from FlowMatchingDataset.
"""

from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


def prepend_origin(sequence: np.ndarray) -> np.ndarray:
    """Prepend (0, 0) origin point to a sequence for visualization."""
    origin = np.array([[0.0, 0.0]], dtype=sequence.dtype)
    return np.vstack([origin, sequence])


def reconstruct_vs_profile(
    sequence: np.ndarray,
    apply_expm1: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct full Vs profile from [TTS, depth] sequence.

    Args:
        sequence: (n, 2) array with [TTS, depth] pairs
        apply_expm1: Whether to apply expm1 (if TTS was log1p normalized)

    Returns:
        depths: Depth values (m) - layer boundaries (n+1 points)
        vs_values: Shear wave velocity (m/s) - per layer (n values)
    """
    if len(sequence) < 2:
        if len(sequence) == 1:
            return np.array([0.0, sequence[0, 1]]), np.array([])
        else:
            return np.array([0.0]), np.array([])

    # Prepend origin
    seq_with_origin = prepend_origin(sequence)

    # Extract TTS and depth
    tts = seq_with_origin[:, 0]
    depths = seq_with_origin[:, 1]

    if apply_expm1:
        tts = np.expm1(tts)

    # Calculate layer thicknesses
    thicknesses = np.diff(depths)

    # Calculate TTS differences (incremental TTS for each layer)
    dtts = np.diff(tts)

    # Convert to Vs: Vs = dz / dt
    # Add small epsilon to avoid division by zero
    vs_values = thicknesses / (dtts + 1e-9)

    # Ensure positive velocities (clip negative values)
    vs_values = np.maximum(vs_values, 0.1)

    return depths, vs_values


def bin_by_depth(
    depths: np.ndarray,
    vs_values: np.ndarray,
    bin_edges: np.ndarray,
    apply_log: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bin Vs values by depth and compute statistics.

    Args:
        depths: 1D array of depths (layer center depths, same length as vs_values)
        vs_values: 1D array of Vs values (per layer)
        bin_edges: Depth bin edges (e.g., np.arange(0, 500, 10))
        apply_log: If True, compute ln(Vs) statistics

    Returns:
        bin_centers: Center of each depth bin
        mean_vs: Mean Vs (or ln Vs) per bin
        std_vs: Std of Vs (or ln Vs) per bin
        bin_counts: Number of samples in each bin
    """
    if len(depths) == 0 or len(vs_values) == 0:
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return (
            bin_centers,
            np.full(len(bin_centers), np.nan),
            np.full(len(bin_centers), np.nan),
            np.zeros(len(bin_centers)),
        )

    # Ensure depths and vs_values have the same length
    if len(depths) != len(vs_values):
        min_len = min(len(depths), len(vs_values))
        depths = depths[:min_len]
        vs_values = vs_values[:min_len]

    # depths are already layer center depths (not boundaries)
    layer_depths = depths

    vs_to_use = np.log(vs_values) if apply_log else vs_values

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    means = []
    stds = []
    counts = []

    for i in range(len(bin_edges) - 1):
        mask = (layer_depths >= bin_edges[i]) & (layer_depths < bin_edges[i + 1])
        if mask.sum() > 0:
            means.append(vs_to_use[mask].mean())
            stds.append(vs_to_use[mask].std())
            counts.append(mask.sum())
        else:
            means.append(np.nan)
            stds.append(np.nan)
            counts.append(0)

    return bin_centers, np.array(means), np.array(stds), np.array(counts)


def compute_real_data_statistics(
    dataset,
    bin_width: float = 10.0,
    max_depth: float = 500.0,
) -> Dict:
    """Compute depth-dependent statistics from real dataset.

    Args:
        dataset: FlowMatchingDataset with denormalize_sequence capability
        bin_width: Depth bin width in meters
        max_depth: Maximum depth to consider

    Returns:
        Dictionary with:
        - bin_centers: Depth bin centers
        - mean_ln_vs: Mean ln(Vs) per depth bin
        - std_ln_vs: Std ln(Vs) per depth bin
        - bin_counts: Number of samples per bin
        - all_depths: All depth values (for debugging)
        - all_vs: All Vs values (for debugging)
    """
    bin_edges = np.arange(0, max_depth + bin_width, bin_width)

    all_depths = []
    all_vs = []

    # Extract Vs profiles from dataset
    print("Extracting Vs profiles from dataset...")
    for i in range(len(dataset)):
        if i % 1000 == 0:
            print(f"  Processing sample {i}/{len(dataset)}...")

        try:
            # Get normalized sequence
            seq_normalized = dataset.sequences[i]

            # Denormalize
            if dataset.normalize:
                seq_denorm = dataset.denormalize_sequence(seq_normalized, i)
            else:
                seq_denorm = seq_normalized

            # Reconstruct full profile with Vs
            depths, vs_values = reconstruct_vs_profile(seq_denorm, apply_expm1=False)

            if len(vs_values) > 0:
                # Map Vs values to layer center depths
                layer_depths = (depths[:-1] + depths[1:]) / 2.0
                all_depths.extend(layer_depths)
                all_vs.extend(vs_values)
        except Exception as e:
            if i < 10:  # Only print first few errors
                print(f"  Warning: Could not process sample {i}: {e}")
            continue

    all_depths = np.array(all_depths)
    all_vs = np.array(all_vs)

    print(f"Extracted {len(all_vs)} Vs values from {len(dataset)} profiles")

    # Bin by depth
    bin_centers, mean_ln_vs, std_ln_vs, bin_counts = bin_by_depth(
        all_depths, all_vs, bin_edges, apply_log=True
    )

    return {
        "bin_centers": bin_centers,
        "mean_ln_vs": mean_ln_vs,
        "std_ln_vs": std_ln_vs,
        "bin_counts": bin_counts,
        "all_depths": all_depths,
        "all_vs": all_vs,
        "bin_edges": bin_edges,
    }


def plot_depth_statistics(
    real_stats: Dict,
    generated_stats: Dict,
    output_path: str,
) -> None:
    """Plot real vs generated depth-dependent statistics.

    Plots:
    - μ(ln Vs) vs depth
    - σ(ln Vs) vs depth

    If real_stats and generated_stats are the same (baseline plot),
    only the real data is plotted.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Check if this is a baseline plot (same data)
    is_baseline = real_stats is generated_stats or (
        np.array_equal(
            real_stats.get("mean_ln_vs", []),
            generated_stats.get("mean_ln_vs", []),
            equal_nan=True,
        )
        if "mean_ln_vs" in real_stats and "mean_ln_vs" in generated_stats
        else False
    )

    # Plot 1: Mean ln(Vs) vs depth
    ax = axes[0]
    real_valid = ~np.isnan(real_stats["mean_ln_vs"])
    gen_valid = (
        ~np.isnan(generated_stats["mean_ln_vs"]) if not is_baseline else np.array([])
    )

    if real_valid.any():
        ax.plot(
            real_stats["bin_centers"][real_valid],
            real_stats["mean_ln_vs"][real_valid],
            "o-",
            label="Real" if not is_baseline else "Real Data",
            linewidth=2,
            markersize=8,
            color="blue",
        )
        ax.fill_between(
            real_stats["bin_centers"][real_valid],
            real_stats["mean_ln_vs"][real_valid] - real_stats["std_ln_vs"][real_valid],
            real_stats["mean_ln_vs"][real_valid] + real_stats["std_ln_vs"][real_valid],
            alpha=0.2,
            color="blue",
            label="Real ±1σ" if not is_baseline else None,
        )

    if gen_valid.any() and not is_baseline:
        ax.plot(
            generated_stats["bin_centers"][gen_valid],
            generated_stats["mean_ln_vs"][gen_valid],
            "s--",
            label="Generated",
            linewidth=2,
            markersize=6,
            color="red",
        )

    ax.set_xlabel("Depth (m)", fontsize=12)
    ax.set_ylabel("ln(Vs) [log m/s]", fontsize=12)
    ax.set_title("Mean ln(Vs) vs Depth", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.invert_yaxis()  # Depth increases downward

    # Plot 2: Std ln(Vs) vs depth
    ax = axes[1]
    if real_valid.any():
        ax.plot(
            real_stats["bin_centers"][real_valid],
            real_stats["std_ln_vs"][real_valid],
            "o-",
            label="Real" if not is_baseline else "Real Data",
            linewidth=2,
            markersize=8,
            color="blue",
        )
    if gen_valid.any() and not is_baseline:
        ax.plot(
            generated_stats["bin_centers"][gen_valid],
            generated_stats["std_ln_vs"][gen_valid],
            "s--",
            label="Generated",
            linewidth=2,
            markersize=6,
            color="red",
        )
    ax.set_xlabel("Depth (m)", fontsize=12)
    ax.set_ylabel("σ(ln Vs)", fontsize=12)
    ax.set_title("Uncertainty σ(ln Vs) vs Depth", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.invert_yaxis()  # Depth increases downward

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved depth statistics plot to {output_path}")


# Usage example:
if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add project root to path
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

    try:
        from experiments.flow_matching_simplified.data import (
            FlowMatchingDataLoader,
            FlowMatchingDataset,
        )
        from experiments.flow_matching_simplified import config as cfg_mod
        from experiments.flow_matching_simplified.split_utils import (
            get_train_val_test_indices,
        )
    except ImportError:
        # Fallback: try relative imports when running from the directory
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from data import FlowMatchingDataLoader, FlowMatchingDataset  # type: ignore
        import config as cfg_mod  # type: ignore
        from split_utils import get_train_val_test_indices  # type: ignore

    # Load dataset
    cfg = cfg_mod.cfg
    data_loader = FlowMatchingDataLoader(data_path=Path(cfg.data_path))
    data_loader.load_data()

    assert data_loader.sequences is not None, "Sequences must be loaded"
    n_total = len(data_loader.sequences)

    train_indices, _, _ = get_train_val_test_indices(n_total)
    train_sequences = [data_loader.sequences[i] for i in train_indices]

    train_dataset = FlowMatchingDataset(
        train_sequences,
        max_length=cfg.max_length,
        pad_token=cfg.pad_token,
        normalize=cfg.normalize,
    )

    # Compute real data statistics
    real_stats = compute_real_data_statistics(
        train_dataset, bin_width=10.0, max_depth=500.0
    )

    print("\n=== Real Data Depth Statistics ===")
    print(f"Bin centers: {real_stats['bin_centers'][:5]}...")
    print(f"Mean ln(Vs): {real_stats['mean_ln_vs'][:5]}...")
    print(f"Std ln(Vs):  {real_stats['std_ln_vs'][:5]}...")
