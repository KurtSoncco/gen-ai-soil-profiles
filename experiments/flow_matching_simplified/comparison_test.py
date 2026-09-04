"""Compare real and generated profiles: TTS vs Depth and Vs vs Depth.

This script loads 100 real profiles and 100 generated profiles,
converts TTS/depth to Vs profiles, and creates comparison plots.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

try:
    from experiments.flow_matching_simplified import config as cfg_mod
    from experiments.flow_matching_simplified.data import (
        FlowMatchingDataLoader,
        FlowMatchingDataset,
    )
    from experiments.flow_matching_simplified.split_utils import get_train_val_test_indices
except ImportError:
    import config as cfg_mod  # type: ignore
    from split_utils import get_train_val_test_indices  # type: ignore

    from data import FlowMatchingDataLoader, FlowMatchingDataset  # type: ignore


def prepend_origin(sequence: np.ndarray) -> np.ndarray:
    """Prepend (0, 0) origin point to a sequence for visualization."""
    origin = np.array([[0.0, 0.0]], dtype=sequence.dtype)
    return np.vstack([origin, sequence])


def tts_depth_to_vs(sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert TTS/depth sequence to Vs velocity profile.

    Args:
        sequence: Array of shape (n, 2) with [TTS, depth] pairs

    Returns:
        Tuple of (depths, vs_values) where:
        - depths: depth values for each layer boundary (n+1 points)
        - vs_values: Vs velocity for each layer (n points)
    """
    if len(sequence) < 2:
        # Not enough points
        if len(sequence) == 1:
            return np.array([0.0, sequence[0, 1]]), np.array([sequence[0, 0]])
        else:
            return np.array([0.0]), np.array([])

    # Prepend origin
    seq_with_origin = prepend_origin(sequence)

    # Extract TTS and depth
    tts = seq_with_origin[:, 0]
    depths = seq_with_origin[:, 1]

    # Calculate layer thicknesses
    thicknesses = np.diff(depths)

    # Calculate TTS differences (incremental TTS for each layer)
    dtts = np.diff(tts)

    # Convert to Vs: Vs = dz / dt (NOT 2 * dz / dt)
    # Add small epsilon to avoid division by zero
    vs_values = thicknesses / (dtts + 1e-9)

    # Ensure positive velocities (clip negative values)
    vs_values = np.maximum(vs_values, 0.1)

    return depths, vs_values


def has_valid_vs(sequence: np.ndarray, min_vs: float = 100.0) -> bool:
    """Check if a sequence has all Vs values >= min_vs.

    Args:
        sequence: Array of shape (n, 2) with [TTS, depth] pairs
        min_vs: Minimum Vs value in m/s (default: 100.0)

    Returns:
        True if all Vs values are >= min_vs, False otherwise
    """
    if len(sequence) == 0:
        return False

    _, vs_values = tts_depth_to_vs(sequence)

    if len(vs_values) == 0:
        return False

    return bool(np.all(vs_values >= min_vs))


def plot_tts_vs_depth_comparison(
    real_sequences: list[np.ndarray],
    generated_sequences: list[np.ndarray],
    output_path: str,
    n_profiles: int = 100,
):
    """Plot TTS vs Depth comparison for real and generated profiles.

    Args:
        real_sequences: List of real sequences, each of shape (n, 2) with [TTS, depth]
        generated_sequences: List of generated sequences, same format
        output_path: Path to save the plot
        n_profiles: Number of profiles to plot
    """
    n_profiles = min(n_profiles, len(real_sequences), len(generated_sequences))

    # Get colorblind-friendly colors - use three different colors
    colors = sns.color_palette("colorblind")
    color1 = colors[0]
    color2 = colors[1]
    color3 = colors[2]

    # Sample different profiles for second and third subplots
    np.random.seed(42)  # For reproducibility
    n_gen = len(generated_sequences)
    if n_gen >= 2 * n_profiles:
        # Sample different profiles for second and third subplots
        gen_indices_2 = np.random.choice(n_gen, n_profiles, replace=False)
        remaining_indices = np.setdiff1d(np.arange(n_gen), gen_indices_2)
        gen_indices_3 = np.random.choice(remaining_indices, n_profiles, replace=False)
    else:
        # If not enough profiles, sample with replacement but ensure different sets
        gen_indices_2 = np.random.choice(n_gen, n_profiles, replace=True)
        gen_indices_3 = np.random.choice(n_gen, n_profiles, replace=True)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 8), sharey=True)

    # Collect all TTS and depth values to determine axis limits
    all_tts_values = []
    all_depth_values = []

    # Plot real profiles (first subplot)
    for i in range(n_profiles):
        real_seq = real_sequences[i]
        if len(real_seq) > 0:
            seq_with_origin = prepend_origin(real_seq)
            all_tts_values.extend(seq_with_origin[:, 0])
            all_depth_values.extend(seq_with_origin[:, 1])
            ax1.plot(
                seq_with_origin[:, 0],
                seq_with_origin[:, 1],
                color=color1,
                linestyle="-",
                linewidth=2.0,
                alpha=0.3,
            )

    # Plot generated profiles (second subplot)
    for idx in gen_indices_2:
        gen_seq = generated_sequences[idx]
        if len(gen_seq) > 0:
            seq_with_origin = prepend_origin(gen_seq)
            all_tts_values.extend(seq_with_origin[:, 0])
            all_depth_values.extend(seq_with_origin[:, 1])
            ax2.plot(
                seq_with_origin[:, 0],
                seq_with_origin[:, 1],
                color=color2,
                linestyle="-",
                linewidth=2.0,
                alpha=0.3,
            )

    # Plot generated profiles (third subplot) - different samples
    for idx in gen_indices_3:
        gen_seq = generated_sequences[idx]
        if len(gen_seq) > 0:
            seq_with_origin = prepend_origin(gen_seq)
            all_tts_values.extend(seq_with_origin[:, 0])
            all_depth_values.extend(seq_with_origin[:, 1])
            ax3.plot(
                seq_with_origin[:, 0],
                seq_with_origin[:, 1],
                color=color3,
                linestyle="-",
                linewidth=2.0,
                alpha=0.3,
            )

    # Set consistent x-axis limits starting at 0
    if len(all_tts_values) > 0:
        max_tts = max(all_tts_values)
        x_max = max_tts * 1.05  # Add 5% padding
        ax1.set_xlim(0, x_max)
        ax2.set_xlim(0, x_max)
        ax3.set_xlim(0, x_max)

    # Set consistent y-axis limits starting at 0
    if len(all_depth_values) > 0:
        max_depth = max(all_depth_values)
        y_max = max_depth * 1.05  # Add 5% padding
        ax1.set_ylim(0, y_max)
        ax2.set_ylim(0, y_max)
        ax3.set_ylim(0, y_max)

    ax1.set_xlabel("TTS (s)", fontsize=12)
    ax1.set_ylabel("Depth (m)", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()
    # ax1.legend(fontsize=11)

    ax2.set_xlabel("TTS (s)", fontsize=12)
    ax2.set_ylabel("Depth (m)", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()
    # ax2.legend(fontsize=11)

    ax3.set_xlabel("TTS (s)", fontsize=12)
    ax3.set_ylabel("Depth (m)", fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.invert_yaxis()
    # ax3.legend(fontsize=11)

    # Move all xlabel and ticks to the top
    ax1.xaxis.set_label_position("top")
    ax1.xaxis.tick_top()
    ax2.xaxis.set_label_position("top")
    ax2.xaxis.tick_top()
    ax3.xaxis.set_label_position("top")
    ax3.xaxis.tick_top()
    plt.suptitle("TTS vs Depth Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved TTS vs Depth comparison to {output_path}")


def plot_vs_vs_depth_comparison(
    real_sequences: list[np.ndarray],
    generated_sequences: list[np.ndarray],
    output_path: str,
    n_profiles: int = 100,
):
    """Plot Vs vs Depth comparison for real and generated profiles.

    Args:
        real_sequences: List of real sequences, each of shape (n, 2) with [TTS, depth]
        generated_sequences: List of generated sequences, same format
        output_path: Path to save the plot
        n_profiles: Number of profiles to plot
    """
    n_profiles = min(n_profiles, len(real_sequences), len(generated_sequences))

    # Get colorblind-friendly colors - use three different colors
    colors = sns.color_palette("colorblind")
    color1 = colors[0]
    color2 = colors[1]
    color3 = colors[2]

    # Sample different profiles for second and third subplots
    np.random.seed(42)  # For reproducibility
    n_gen = len(generated_sequences)
    if n_gen >= 2 * n_profiles:
        # Sample different profiles for second and third subplots
        gen_indices_2 = np.random.choice(n_gen, n_profiles, replace=False)
        remaining_indices = np.setdiff1d(np.arange(n_gen), gen_indices_2)
        gen_indices_3 = np.random.choice(remaining_indices, n_profiles, replace=False)
    else:
        # If not enough profiles, sample with replacement but ensure different sets
        gen_indices_2 = np.random.choice(n_gen, n_profiles, replace=True)
        gen_indices_3 = np.random.choice(n_gen, n_profiles, replace=True)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 8), sharey=True)

    # Collect all Vs and depth values to determine axis limits
    all_vs_values = []
    all_depth_values = []

    # Plot real profiles (first subplot)
    for i in range(n_profiles):
        real_seq = real_sequences[i]
        if len(real_seq) > 0:
            depths, vs_values = tts_depth_to_vs(real_seq)
            if len(vs_values) > 0 and len(depths) > 1:
                # Create step plot: Vs is constant within each layer
                # For each layer, Vs[i] applies from depth[i] to depth[i+1]
                step_depths = []
                step_vs = []

                for j in range(len(vs_values)):
                    # Start of layer
                    step_depths.append(depths[j])
                    step_vs.append(vs_values[j])
                    # End of layer
                    step_depths.append(depths[j + 1])
                    step_vs.append(vs_values[j])

                all_vs_values.extend(step_vs)
                all_depth_values.extend(step_depths)
                ax1.plot(
                    step_vs,
                    step_depths,
                    color=color1,
                    linestyle="-",
                    linewidth=2.0,
                    alpha=0.3,
                )

    # Plot generated profiles (second subplot)
    for idx in gen_indices_2:
        gen_seq = generated_sequences[idx]
        if len(gen_seq) > 0:
            depths, vs_values = tts_depth_to_vs(gen_seq)
            if len(vs_values) > 0 and len(depths) > 1:
                # Create step plot: Vs is constant within each layer
                step_depths = []
                step_vs = []

                for j in range(len(vs_values)):
                    # Start of layer
                    step_depths.append(depths[j])
                    step_vs.append(vs_values[j])
                    # End of layer
                    step_depths.append(depths[j + 1])
                    step_vs.append(vs_values[j])

                all_vs_values.extend(step_vs)
                all_depth_values.extend(step_depths)
                ax2.plot(
                    step_vs,
                    step_depths,
                    color=color2,
                    linestyle="-",
                    linewidth=2.0,
                    alpha=0.3,
                )

    # Plot generated profiles (third subplot) - different samples
    for idx in gen_indices_3:
        gen_seq = generated_sequences[idx]
        if len(gen_seq) > 0:
            depths, vs_values = tts_depth_to_vs(gen_seq)
            if len(vs_values) > 0 and len(depths) > 1:
                # Create step plot: Vs is constant within each layer
                step_depths = []
                step_vs = []

                for j in range(len(vs_values)):
                    # Start of layer
                    step_depths.append(depths[j])
                    step_vs.append(vs_values[j])
                    # End of layer
                    step_depths.append(depths[j + 1])
                    step_vs.append(vs_values[j])

                all_vs_values.extend(step_vs)
                all_depth_values.extend(step_depths)
                ax3.plot(
                    step_vs,
                    step_depths,
                    color=color3,
                    linestyle="-",
                    linewidth=2.0,
                    alpha=0.3,
                )

    # Set consistent x-axis limits starting at 0
    if len(all_vs_values) > 0:
        max_vs = max(all_vs_values)
        x_max = max_vs * 1.05  # Add 5% padding
        ax1.set_xlim(0, x_max)
        ax2.set_xlim(0, x_max)
        ax3.set_xlim(0, x_max)

    # Set consistent y-axis limits starting at 0
    if len(all_depth_values) > 0:
        max_depth = max(all_depth_values)
        y_max = max_depth * 1.05  # Add 5% padding
        ax1.set_ylim(0, y_max)
        ax2.set_ylim(0, y_max)
        ax3.set_ylim(0, y_max)

    ax1.set_xlabel("Vs (m/s)", fontsize=12)
    ax1.set_ylabel("Depth (m)", fontsize=12)

    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()
    # ax1.legend(fontsize=11)

    ax2.set_xlabel("Vs (m/s)", fontsize=12)
    ax2.set_ylabel("Depth (m)", fontsize=12)

    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()
    # ax2.legend(fontsize=11)

    ax3.set_xlabel("Vs (m/s)", fontsize=12)
    ax3.set_ylabel("Depth (m)", fontsize=12)

    ax3.grid(True, alpha=0.3)
    ax3.invert_yaxis()
    # ax3.legend(fontsize=11)

    # Move all xlabel and ticks to the top
    ax1.xaxis.set_label_position("top")
    ax1.xaxis.tick_top()
    ax2.xaxis.set_label_position("top")
    ax2.xaxis.tick_top()
    ax3.xaxis.set_label_position("top")
    ax3.xaxis.tick_top()

    plt.suptitle("Vs vs Depth Comparison", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved Vs vs Depth comparison to {output_path}")


def main():
    """Main function to load profiles and create comparison plots."""
    import argparse

    parser = argparse.ArgumentParser(description="Compare real and generated profiles")
    parser.add_argument(
        "--generated-profiles",
        type=str,
        default=None,
        help="Path to generated profiles .npy file (default: auto-detect)",
    )
    parser.add_argument(
        "--n-profiles",
        type=int,
        default=30,
        help="Number of profiles to compare (default: 30)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots (default: plots_dir from config)",
    )

    args = parser.parse_args()

    cfg = cfg_mod.cfg

    # Determine generated profiles path
    if args.generated_profiles is None:
        # Look for the most recent filtered_profiles file
        samples_dir = cfg.samples_dir
        if os.path.exists(samples_dir):
            npy_files = list(Path(samples_dir).glob("filtered_profiles*.npy"))
            if npy_files:
                # Sort by modification time, get most recent
                generated_profiles_path = max(
                    npy_files, key=lambda p: p.stat().st_mtime
                )
                generated_profiles_path = str(generated_profiles_path)
            else:
                raise FileNotFoundError(
                    f"No filtered_profiles*.npy found in {samples_dir}"
                )
        else:
            raise FileNotFoundError(f"Samples directory not found: {samples_dir}")
    else:
        generated_profiles_path = args.generated_profiles

    if not os.path.exists(generated_profiles_path):
        raise FileNotFoundError(
            f"Generated profiles not found: {generated_profiles_path}"
        )

    # Load generated profiles
    print(f"Loading generated profiles from {generated_profiles_path}")
    generated_profiles_array = np.load(generated_profiles_path, allow_pickle=True)
    generated_sequences = []
    for seq in generated_profiles_array:
        if isinstance(seq, np.ndarray):
            generated_sequences.append(seq)
        else:
            generated_sequences.append(np.array(seq))

    print(f"Loaded {len(generated_sequences)} generated profiles")

    # Filter out sequences with Vs < 100 m/s
    print("Filtering sequences with Vs < 100 m/s...")
    generated_sequences_filtered = [
        seq for seq in generated_sequences if has_valid_vs(seq, min_vs=100.0)
    ]
    print(
        f"Filtered {len(generated_sequences)} -> {len(generated_sequences_filtered)} generated profiles"
    )
    generated_sequences = generated_sequences_filtered

    # Load real profiles from test set
    print("Loading real profiles from test set...")
    data_loader = FlowMatchingDataLoader(data_path=Path(cfg.data_path))
    data_loader.load_data()

    assert data_loader.sequences is not None, "Sequences must be loaded"
    n_total = len(data_loader.sequences)

    _, _, test_indices = get_train_val_test_indices(n_total)

    # Get test dataset
    assert data_loader.sequences is not None
    test_sequences = [data_loader.sequences[i] for i in test_indices]
    test_dataset = FlowMatchingDataset(
        test_sequences,
        max_length=cfg.max_length,
        pad_token=cfg.pad_token,
        normalize=cfg.normalize,
    )

    # Get real sequences from test set and denormalize if needed
    real_sequences = []
    for i in range(len(test_dataset)):
        seq_normalized = test_dataset.sequences[i]
        if cfg.normalize and test_dataset.normalize:
            seq_denorm = test_dataset.denormalize_sequence(seq_normalized, i)
            real_sequences.append(seq_denorm)
        else:
            real_sequences.append(seq_normalized)

    print(f"Loaded {len(real_sequences)} real sequences from test set")

    # Filter out sequences with Vs < 100 m/s
    print("Filtering real sequences with Vs < 100 m/s...")
    real_sequences_filtered = [
        seq for seq in real_sequences if has_valid_vs(seq, min_vs=100.0)
    ]
    print(
        f"Filtered {len(real_sequences)} -> {len(real_sequences_filtered)} real sequences"
    )
    real_sequences = real_sequences_filtered

    # Sample n_profiles from each (ensure at least 30)
    n_profiles = max(
        30, min(args.n_profiles, len(real_sequences), len(generated_sequences))
    )
    np.random.seed(24)  # For reproducibility
    real_indices = np.random.choice(len(real_sequences), n_profiles, replace=False)
    gen_indices = np.random.choice(len(generated_sequences), n_profiles, replace=False)

    sampled_real = [real_sequences[i] for i in real_indices]
    sampled_generated = [generated_sequences[i] for i in gen_indices]

    print(f"Sampled {n_profiles} profiles from each set")

    # Determine output directory
    if args.output_dir is None:
        output_dir = cfg.plots_dir
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Create TTS vs Depth plot
    print("\nCreating TTS vs Depth comparison plot...")
    tts_output_path = os.path.join(output_dir, "tts_vs_depth_comparison.png")
    plot_tts_vs_depth_comparison(
        sampled_real, sampled_generated, tts_output_path, n_profiles=n_profiles
    )

    # Create Vs vs Depth plot
    print("Creating Vs vs Depth comparison plot...")
    vs_output_path = os.path.join(output_dir, "vs_vs_depth_comparison.png")
    plot_vs_vs_depth_comparison(
        sampled_real, sampled_generated, vs_output_path, n_profiles=n_profiles
    )

    print(f"\nDone! Plots saved to {output_dir}")
    print(f"  - TTS vs Depth: {tts_output_path}")
    print(f"  - Vs vs Depth: {vs_output_path}")


if __name__ == "__main__":
    main()
