"""Compare multiple model outputs: Vs vs Depth visualization.

This script creates Vs vs Depth comparison plots showing multiple models
side-by-side, similar to vs_vs_depth_comparison.png format.
"""

import argparse
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from experiments.flow_matching_simplified import config as cfg_mod
    from experiments.flow_matching_simplified.data import (
        FlowMatchingDataLoader,
        FlowMatchingDataset,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
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

    # Convert to Vs: Vs = dz / dt
    vs_values = thicknesses / (dtts + 1e-9)

    # Ensure positive velocities
    vs_values = np.maximum(vs_values, 0.1)

    return depths, vs_values


def has_valid_vs(sequence: np.ndarray, min_vs: float = 100.0) -> bool:
    """Check if a sequence has all Vs values >= min_vs."""
    if len(sequence) == 0:
        return False

    _, vs_values = tts_depth_to_vs(sequence)

    if len(vs_values) == 0:
        return False

    return bool(np.all(vs_values >= min_vs))


def plot_vs_vs_depth_comparison(
    real_sequences: List[np.ndarray],
    model_sequences_list: List[List[np.ndarray]],
    model_names: List[str],
    output_path: str,
    n_profiles: int = 30,
):
    """Plot Vs vs Depth comparison for real data and multiple models.

    Args:
        real_sequences: List of real sequences
        model_sequences_list: List of lists, each containing sequences for one model
        model_names: List of model names/labels
        output_path: Path to save the plot
        n_profiles: Number of profiles to plot per panel
    """
    n_models = len(model_sequences_list)

    # Create subplots: 1 for real + 1 for each model
    n_cols = n_models + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 8), sharey=True)

    if n_cols == 1:
        axes = [axes]

    # Get colorblind-friendly colors
    colors = sns.color_palette("colorblind", n_colors=n_cols)

    # Collect all Vs and depth values for consistent axis limits
    all_vs_values = []
    all_depth_values = []

    # Sample profiles for reproducibility
    np.random.seed(42)

    # Plot real profiles (first subplot)
    n_real = min(n_profiles, len(real_sequences))
    real_indices = np.random.choice(len(real_sequences), n_real, replace=False)

    ax = axes[0]
    for idx in real_indices:
        real_seq = real_sequences[idx]
        if len(real_seq) > 0:
            depths, vs_values = tts_depth_to_vs(real_seq)
            if len(vs_values) > 0 and len(depths) > 1:
                # Create step plot: Vs is constant within each layer
                step_depths = []
                step_vs = []

                for j in range(len(vs_values)):
                    step_depths.append(depths[j])
                    step_vs.append(vs_values[j])
                    step_depths.append(depths[j + 1])
                    step_vs.append(vs_values[j])

                all_vs_values.extend(step_vs)
                all_depth_values.extend(step_depths)
                ax.plot(
                    step_vs,
                    step_depths,
                    color=colors[0],
                    linestyle="-",
                    linewidth=2.0,
                    alpha=0.3,
                )

    ax.set_xlabel("Vs (m/s)", fontsize=12)
    ax.set_ylabel("Depth (m)", fontsize=12)
    ax.set_title("Real Data", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    # Plot each model
    for model_idx, (model_sequences, model_name, color) in enumerate(
        zip(model_sequences_list, model_names, colors[1:]), start=1
    ):
        ax = axes[model_idx]

        n_model = min(n_profiles, len(model_sequences))
        model_indices = np.random.choice(len(model_sequences), n_model, replace=False)

        for idx in model_indices:
            gen_seq = model_sequences[idx]
            if len(gen_seq) > 0:
                depths, vs_values = tts_depth_to_vs(gen_seq)
                if len(vs_values) > 0 and len(depths) > 1:
                    # Create step plot
                    step_depths = []
                    step_vs = []

                    for j in range(len(vs_values)):
                        step_depths.append(depths[j])
                        step_vs.append(vs_values[j])
                        step_depths.append(depths[j + 1])
                        step_vs.append(vs_values[j])

                    all_vs_values.extend(step_vs)
                    all_depth_values.extend(step_depths)
                    ax.plot(
                        step_vs,
                        step_depths,
                        color=color,
                        linestyle="-",
                        linewidth=2.0,
                        alpha=0.3,
                    )

        ax.set_xlabel("Vs (m/s)", fontsize=12)
        ax.set_ylabel("Depth (m)", fontsize=12)
        ax.set_title(model_name, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()

    # Set consistent axis limits
    if len(all_vs_values) > 0:
        max_vs = max(all_vs_values)
        x_max = max_vs * 1.05
        for ax in axes:
            ax.set_xlim(0, x_max)

    if len(all_depth_values) > 0:
        max_depth = max(all_depth_values)
        y_max = max_depth * 1.05
        for ax in axes:
            ax.set_ylim(0, y_max)

    ax.invert_yaxis()
    plt.suptitle("Vs vs Depth Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved Vs vs Depth comparison to {output_path}")


def load_real_sequences(cfg) -> List[np.ndarray]:
    """Load real sequences from test set."""
    print("Loading real profiles from test set...")
    data_loader = FlowMatchingDataLoader(data_path=Path(cfg.data_path))
    data_loader.load_data()

    assert data_loader.sequences is not None, "Sequences must be loaded"
    n_total = len(data_loader.sequences)

    # Create train/val/test splits
    import torch

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

    # Get real sequences and denormalize if needed
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

    return real_sequences_filtered


def main():
    """Main comparison function."""
    parser = argparse.ArgumentParser(
        description="Compare multiple model outputs: Vs vs Depth visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two models
  python compare_models.py \\
      --model baseline.npy "Baseline Model" \\
      --model improved.npy "Improved Model" \\
      --output comparison.png

  # Compare multiple models
  python compare_models.py \\
      --model model1.npy "Model 1" \\
      --model model2.npy "Model 2" \\
      --model model3.npy "Model 3" \\
      --output comparison.png
        """,
    )
    parser.add_argument(
        "--model",
        nargs=2,
        metavar=("PROFILES", "NAME"),
        action="append",
        required=True,
        help="Generated profiles .npy file and model name (can be used multiple times)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="vs_vs_depth_comparison.png",
        help="Output path for comparison plot (default: vs_vs_depth_comparison.png)",
    )
    parser.add_argument(
        "--n-profiles",
        type=int,
        default=30,
        help="Number of profiles to plot per panel (default: 30)",
    )
    parser.add_argument(
        "--no-real",
        action="store_true",
        help="Don't include real data panel (only show models)",
    )

    args = parser.parse_args()

    cfg = cfg_mod.cfg

    # Load real sequences if requested
    real_sequences = []
    if not args.no_real:
        real_sequences = load_real_sequences(cfg)

    # Process each model
    model_sequences_list = []
    model_names = []

    for profiles_path, model_name in args.model:
        print(f"\nProcessing {model_name} from {profiles_path}...")

        # Load profiles
        generated_profiles_array = np.load(profiles_path, allow_pickle=True)
        generated_sequences = []
        for seq in generated_profiles_array:
            if isinstance(seq, np.ndarray):
                generated_sequences.append(seq)
            else:
                generated_sequences.append(np.array(seq))

        print(f"  Loaded {len(generated_sequences)} profiles")

        # Filter out sequences with Vs < 100 m/s
        print("  Filtering sequences with Vs < 100 m/s...")
        generated_sequences_filtered = [
            seq for seq in generated_sequences if has_valid_vs(seq, min_vs=100.0)
        ]
        print(
            f"  Filtered {len(generated_sequences)} -> {len(generated_sequences_filtered)} profiles"
        )

        model_sequences_list.append(generated_sequences_filtered)
        model_names.append(model_name)

    # Create comparison plot
    print("\nCreating Vs vs Depth comparison plot...")
    plot_vs_vs_depth_comparison(
        real_sequences,
        model_sequences_list,
        model_names,
        args.output,
        n_profiles=args.n_profiles,
    )

    print(f"\nComparison plot saved to: {args.output}")


if __name__ == "__main__":
    main()
