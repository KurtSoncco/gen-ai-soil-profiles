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
import torch

try:
    from experiments.flow_matching_simplified import config as cfg_mod
    from experiments.flow_matching_simplified.data import (
        FlowMatchingDataLoader,
        FlowMatchingDataset,
    )
except ImportError:
    import config as cfg_mod  # type: ignore

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

    # Get colorblind-friendly colors
    colors = sns.color_palette("colorblind")
    real_color = colors[0]  # First color for real
    gen_color = colors[1]  # Second color for generated

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8), sharey=True)

    # Plot real profiles
    for i in range(n_profiles):
        real_seq = real_sequences[i]
        if len(real_seq) > 0:
            seq_with_origin = prepend_origin(real_seq)
            ax1.plot(
                seq_with_origin[:, 0],
                seq_with_origin[:, 1],
                color=real_color,
                linestyle="-",
                linewidth=2.0,
                alpha=0.3,
                label="Real" if i == 0 else "",
            )

    # Plot generated profiles
    for i in range(n_profiles):
        gen_seq = generated_sequences[i]
        if len(gen_seq) > 0:
            seq_with_origin = prepend_origin(gen_seq)
            ax2.plot(
                seq_with_origin[:, 0],
                seq_with_origin[:, 1],
                color=gen_color,
                linestyle="--",
                linewidth=2.0,
                alpha=0.3,
                label="Generated" if i == 0 else "",
            )

    ax1.set_xlabel("TTS (s)", fontsize=12)
    ax1.set_ylabel("Depth (m)", fontsize=12)
    ax1.grid(True, alpha=0.3)
    # ax1.legend(fontsize=11)

    ax2.set_xlabel("TTS (s)", fontsize=12)
    ax2.set_ylabel("Depth (m)", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()
    # ax2.legend(fontsize=11)

    # Move all xlabel and ticks to the top
    ax1.xaxis.set_label_position("top")
    ax1.xaxis.tick_top()
    ax2.xaxis.set_label_position("top")
    ax2.xaxis.tick_top()
    plt.suptitle(
        f"TTS vs Depth Comparison (n={n_profiles})", fontsize=14, fontweight="bold"
    )
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

    # Get colorblind-friendly colors
    colors = sns.color_palette("colorblind")
    real_color = colors[0]  # First color for real
    gen_color = colors[1]  # Second color for generated

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8), sharey=True)

    # Plot real profiles
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

                ax1.plot(
                    step_vs,
                    step_depths,
                    color=real_color,
                    linestyle="-",
                    linewidth=2.0,
                    alpha=0.3,
                    label="Real" if i == 0 else "",
                )

    # Plot generated profiles
    for i in range(n_profiles):
        gen_seq = generated_sequences[i]
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

                ax2.plot(
                    step_vs,
                    step_depths,
                    color=gen_color,
                    linestyle="-",
                    linewidth=2.0,
                    alpha=0.3,
                    label="Generated" if i == 0 else "",
                )

    ax1.set_xlabel("Vs (m/s)", fontsize=12)
    ax1.set_ylabel("Depth (m)", fontsize=12)

    ax1.grid(True, alpha=0.3)
    # ax1.legend(fontsize=11)

    ax2.set_xlabel("Vs (m/s)", fontsize=12)
    ax2.set_ylabel("Depth (m)", fontsize=12)

    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()
    # ax2.legend(fontsize=11)

    # Move all xlabel and ticks to the top
    ax1.xaxis.set_label_position("top")
    ax1.xaxis.tick_top()
    ax2.xaxis.set_label_position("top")
    ax2.xaxis.tick_top()

    plt.suptitle(
        f"Vs vs Depth Comparison (n={n_profiles})", fontsize=14, fontweight="bold"
    )

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
        default=20,
        help="Number of profiles to compare (default: 20)",
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

    # Load real profiles from test set
    print("Loading real profiles from test set...")
    data_loader = FlowMatchingDataLoader(data_path=Path(cfg.data_path))
    data_loader.load_data()

    assert data_loader.sequences is not None, "Sequences must be loaded"
    n_total = len(data_loader.sequences)

    # Create train/val/test splits (same as train.py and comprehensive_eval.py)
    all_indices = torch.randperm(n_total)
    n_train = int(cfg.train_val_test_split[0] * n_total)
    n_val = int(cfg.train_val_test_split[1] * n_total)

    test_indices = all_indices[n_train + n_val :].tolist()

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

    # Sample n_profiles from each
    n_profiles = min(args.n_profiles, len(real_sequences), len(generated_sequences))
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
