"""Comprehensive evaluation script for Flow Matching Breakpoints.

This script generates:
- Profile panels: 20 real vs predicted TTS-depth curves
- Histograms + KS tests for: Final depth, Total TTS, Breakpoint count
- Monotonicity rate and final-depth physical checks
- Diversity: mean/std pairwise distance on interpolated profiles
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats as scipy_stats
from scipy.interpolate import interp1d

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
    from experiments.flow_matching_simplified.train import sample_sequences
    from experiments.flow_matching_simplified.utils import (
        check_min_dt,
        check_vs_bounds,
        compute_vs_from_sequence,
    )
except ImportError:  # fallback when running as script
    import config as cfg_mod  # type: ignore
    from model import TransformerModel  # type: ignore
    from train import sample_sequences  # type: ignore
    from utils import (  # type: ignore
        check_min_dt,
        check_vs_bounds,
        compute_vs_from_sequence,
    )

    from data import FlowMatchingDataLoader, FlowMatchingDataset  # type: ignore


def convert_to_json_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
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


def prepend_origin(sequence: np.ndarray) -> np.ndarray:
    """Prepend (0, 0) origin point to a sequence for reconstruction/visualization.

    During training, sequences don't include the deterministic (0,0) origin.
    This function adds it back for evaluation and visualization.

    Args:
        sequence: Array of shape (n, 2) with [ts, depth] pairs

    Returns:
        Array of shape (n+1, 2) with (0,0) prepended
    """
    origin = np.array([[0.0, 0.0]], dtype=sequence.dtype)
    return np.vstack([origin, sequence])


def load_checkpoint(
    checkpoint_path: str, device: torch.device, config: cfg_mod.Config
) -> tuple[TransformerModel, int]:
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
    # Checkpoint uses "model" key (from train.py save_checkpoint)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        raise KeyError(
            f"Checkpoint missing model state dict. Keys: {checkpoint.keys()}"
        )
    epoch = checkpoint.get("epoch", 0)
    return model, epoch


def compute_final_depth(sequences: list[np.ndarray]) -> np.ndarray:
    """Compute final depth for each sequence (last depth value)."""
    final_depths = []
    for seq in sequences:
        if len(seq) > 0:
            # Final depth is the last depth value (absolute)
            final_depths.append(seq[-1, 1])
        else:
            final_depths.append(0.0)
    return np.array(final_depths)


def compute_total_tts(sequences: list[np.ndarray]) -> np.ndarray:
    """Compute total TTS for each sequence (sum of all TTS values)."""
    total_tts = []
    for seq in sequences:
        if len(seq) > 0:
            total_tts.append(np.sum(seq[:, 0]))
        else:
            total_tts.append(0.0)
    return np.array(total_tts)


def compute_breakpoint_count(sequences: list[np.ndarray]) -> np.ndarray:
    """Compute number of breakpoints (sequence length) for each sequence."""
    return np.array([len(seq) for seq in sequences])


def check_monotonicity(sequences: list[np.ndarray]) -> dict[str, float]:
    """Check monotonicity of depth values (should be non-decreasing).

    Returns:
        Dictionary with monotonicity rate and other metrics
    """
    monotonic_count = 0
    total_count = 0

    for seq in sequences:
        if len(seq) < 2:
            monotonic_count += 1
            total_count += 1
            continue

        # Check if depth is non-decreasing
        depths = seq[:, 1]
        is_monotonic = np.all(depths[1:] >= depths[:-1])

        if is_monotonic:
            monotonic_count += 1
        total_count += 1

    monotonic_rate = monotonic_count / total_count if total_count > 0 else 0.0

    return {
        "monotonic_rate": float(monotonic_rate),
        "monotonic_count": int(monotonic_count),
        "total_count": int(total_count),
    }


def check_final_depth_physical(sequences: list[np.ndarray]) -> dict[str, float]:
    """Check physical constraints on final depth.

    Physical checks:
    - Final depth should be positive
    - Final depth should be reasonable (e.g., < 1000m for most cases)
    """
    final_depths = compute_final_depth(sequences)

    positive_count = np.sum(final_depths > 0)
    reasonable_count = np.sum((final_depths > 0) & (final_depths < 1000))

    return {
        "positive_rate": float(positive_count / len(final_depths))
        if len(final_depths) > 0
        else 0.0,
        "reasonable_rate": float(reasonable_count / len(final_depths))
        if len(final_depths) > 0
        else 0.0,
        "mean_final_depth": float(np.mean(final_depths[final_depths > 0]))
        if np.any(final_depths > 0)
        else 0.0,
        "max_final_depth": float(np.max(final_depths))
        if len(final_depths) > 0
        else 0.0,
    }


def check_vs_constraints(
    sequences: list[np.ndarray],
    vs_min: float = 100.0,
    vs_max: float = 5000.0,
    min_dt: float = 1e-6,
) -> dict[str, float]:
    """Check Vs velocity constraints.

    Args:
        sequences: List of sequences, each of shape (n, 2) with [TTS, depth]
        vs_min: Minimum allowed Vs value (m/s)
        vs_max: Maximum allowed Vs value (m/s)
        min_dt: Minimum allowed Δt value

    Returns:
        Dictionary with Vs constraint statistics
    """
    vs_bounds_count = 0
    min_dt_count = 0
    total_count = 0
    all_vs_values = []

    for seq in sequences:
        if len(seq) < 2:
            total_count += 1
            vs_bounds_count += 1
            min_dt_count += 1
            continue

        # Check Vs bounds
        if check_vs_bounds(seq, vs_min, vs_max):
            vs_bounds_count += 1

        # Check min_dt
        if check_min_dt(seq, min_dt):
            min_dt_count += 1

        # Collect Vs values for statistics
        vs_values = compute_vs_from_sequence(seq)
        if len(vs_values) > 0:
            all_vs_values.extend(vs_values.tolist())

        total_count += 1

    vs_bounds_rate = vs_bounds_count / total_count if total_count > 0 else 0.0
    min_dt_rate = min_dt_count / total_count if total_count > 0 else 0.0

    result = {
        "vs_bounds_rate": float(vs_bounds_rate),
        "vs_bounds_count": int(vs_bounds_count),
        "min_dt_rate": float(min_dt_rate),
        "min_dt_count": int(min_dt_count),
        "total_count": int(total_count),
    }

    if len(all_vs_values) > 0:
        all_vs_array = np.array(all_vs_values)
        result.update(
            {
                "mean_vs": float(np.mean(all_vs_array)),
                "std_vs": float(np.std(all_vs_array)),
                "min_vs": float(np.min(all_vs_array)),
                "max_vs": float(np.max(all_vs_array)),
                "vs_below_min": int(np.sum(all_vs_array < vs_min)),
                "vs_above_max": int(np.sum(all_vs_array > vs_max)),
            }
        )
    else:
        result.update(
            {
                "mean_vs": 0.0,
                "std_vs": 0.0,
                "min_vs": 0.0,
                "max_vs": 0.0,
                "vs_below_min": 0,
                "vs_above_max": 0,
            }
        )

    return result


def interpolate_profiles(
    sequences: list[np.ndarray], n_points: int = 100
) -> list[np.ndarray]:
    """Interpolate profiles to fixed number of points for distance computation.

    Uses absolute TTS and depth values, interpolating TTS as a function of depth.
    """
    interpolated = []

    for seq in sequences:
        if len(seq) < 2:
            # If too few points, pad with zeros
            interp_seq = np.zeros((n_points, 2))
            interpolated.append(interp_seq)
            continue

        # Prepend (0,0) origin for proper reconstruction
        seq_with_origin = prepend_origin(seq)

        # Use absolute values: seq[:, 0] = TTS, seq[:, 1] = depth
        depths = seq_with_origin[:, 1]
        tts_values = seq_with_origin[:, 0]

        # Create depth range for interpolation (from 0 to max depth)
        if len(depths) > 1 and depths[-1] > depths[0]:
            depth_range = np.linspace(depths[0], depths[-1], n_points)
        else:
            # If depths are not increasing, use original range
            depth_range = np.linspace(np.min(depths), np.max(depths), n_points)

        # Interpolate TTS as a function of depth
        if len(depths) > 1 and np.max(depths) > np.min(depths):
            # Use extrapolation for values outside bounds
            # fill_value can be tuple (left, right) but type checker doesn't recognize it
            f_tts = interp1d(
                depths,
                tts_values,
                kind="linear",
                fill_value=(float(tts_values[0]), float(tts_values[-1])),  # type: ignore
                bounds_error=False,
            )
            interp_tts = f_tts(depth_range)
        else:
            # If all depths are the same, use constant TTS
            interp_tts = np.full(
                n_points, tts_values[0] if len(tts_values) > 0 else 0.0
            )

        # Create interpolated sequence [ts, depth]
        interp_seq = np.column_stack([interp_tts, depth_range])
        interpolated.append(interp_seq)

    return interpolated


def compute_pairwise_distance(sequences: list[np.ndarray]) -> dict[str, float]:
    """Compute pairwise distances between interpolated profiles.

    Returns:
        Dictionary with mean and std of pairwise distances
    """
    if len(sequences) < 2:
        return {"mean_distance": 0.0, "std_distance": 0.0}

    # Interpolate all sequences
    interp_sequences = interpolate_profiles(sequences, n_points=100)

    # Compute pairwise distances
    distances = []
    for i in range(len(interp_sequences)):
        for j in range(i + 1, len(interp_sequences)):
            seq1 = interp_sequences[i]
            seq2 = interp_sequences[j]

            # Euclidean distance on interpolated points
            dist = np.sqrt(np.sum((seq1 - seq2) ** 2))
            distances.append(dist)

    distances = np.array(distances)

    return {
        "mean_distance": float(np.mean(distances)) if len(distances) > 0 else 0.0,
        "std_distance": float(np.std(distances)) if len(distances) > 0 else 0.0,
        "min_distance": float(np.min(distances)) if len(distances) > 0 else 0.0,
        "max_distance": float(np.max(distances)) if len(distances) > 0 else 0.0,
    }


def plot_profile_comparison_side_by_side(
    real_sequences: list[np.ndarray],
    generated_sequences: list[np.ndarray],
    output_path: str,
    n_profiles: int = 15,
):
    """Plot real and generated profiles side by side (1 row, 2 columns).

    Left subplot: n_profiles real profiles
    Right subplot: n_profiles generated profiles
    """
    n_profiles = min(n_profiles, len(real_sequences), len(generated_sequences))

    # Create subplots: 1 row x 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Left subplot: Real profiles
    ax_real = axes[0]
    for i in range(n_profiles):
        real_seq = real_sequences[i]
        if len(real_seq) > 0:
            # Prepend (0,0) origin for visualization
            seq_with_origin = prepend_origin(real_seq)
            ax_real.plot(
                seq_with_origin[:, 0],
                seq_with_origin[:, 1],
                "b-",
                linewidth=1.5,
                alpha=0.7,
                marker="o",
                markersize=3,
            )

    ax_real.set_xlabel("TTS (absolute)", fontsize=12)
    ax_real.set_ylabel("Depth (absolute)", fontsize=12)
    ax_real.set_title(f"Real Profiles (n={n_profiles})", fontsize=14, fontweight="bold")
    ax_real.grid(True, alpha=0.3)
    ax_real.invert_yaxis()

    # Right subplot: Generated profiles
    ax_gen = axes[1]
    for i in range(n_profiles):
        gen_seq = generated_sequences[i]
        if len(gen_seq) > 0:
            # Prepend (0,0) origin for visualization
            seq_with_origin = prepend_origin(gen_seq)
            ax_gen.plot(
                seq_with_origin[:, 0],
                seq_with_origin[:, 1],
                "r-",
                linewidth=1.5,
                alpha=0.7,
                marker="s",
                markersize=3,
            )

    ax_gen.set_xlabel("TTS (absolute)", fontsize=12)
    ax_gen.set_ylabel("Depth (absolute)", fontsize=12)
    ax_gen.set_title(
        f"Generated Profiles (n={n_profiles})", fontsize=14, fontweight="bold"
    )
    ax_gen.grid(True, alpha=0.3)
    ax_gen.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved side-by-side profile comparison to {output_path}")


def plot_profile_panels(
    real_sequences: list[np.ndarray],
    generated_sequences: list[np.ndarray],
    output_path: str,
    n_profiles: int = 20,
):
    """Plot 20 real vs predicted TTS-depth curves."""
    n_profiles = min(n_profiles, len(real_sequences), len(generated_sequences))

    # Create subplots: 4 rows x 5 columns
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.flatten()

    for i in range(n_profiles):
        ax = axes[i]

        # Real profile - use absolute values
        real_seq = real_sequences[i]
        if len(real_seq) > 0:
            # Prepend (0,0) origin for visualization
            real_seq_with_origin = prepend_origin(real_seq)
            ax.plot(
                real_seq_with_origin[:, 0],
                real_seq_with_origin[:, 1],
                "b-",
                linewidth=2,
                label="Real",
                alpha=0.8,
                marker="o",
                markersize=4,
            )

        # Generated profile - use absolute values
        gen_seq = generated_sequences[i]
        if len(gen_seq) > 0:
            # Prepend (0,0) origin for visualization
            gen_seq_with_origin = prepend_origin(gen_seq)
            ax.plot(
                gen_seq_with_origin[:, 0],
                gen_seq_with_origin[:, 1],
                "r--",
                linewidth=2,
                label="Generated",
                alpha=0.8,
                marker="s",
                markersize=4,
            )

        ax.set_xlabel("TTS (absolute)")
        ax.set_ylabel("Depth (absolute)")
        ax.set_title(f"Profile {i + 1}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.invert_yaxis()

    # Hide unused subplots
    for i in range(n_profiles, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved profile panels to {output_path}")


def compute_all_vs_values(sequences: list[np.ndarray]) -> np.ndarray:
    """Compute all Vs values from a list of sequences.

    Args:
        sequences: List of sequences, each of shape (n, 2) with [TTS, depth]

    Returns:
        Array of all Vs values from all sequences
    """
    all_vs = []
    for seq in sequences:
        vs_values = compute_vs_from_sequence(seq)
        if len(vs_values) > 0:
            all_vs.extend(vs_values.tolist())
    return np.array(all_vs)


def plot_histograms_with_ks(
    real_sequences: list[np.ndarray],
    generated_sequences: list[np.ndarray],
    output_path: str,
):
    """Plot histograms with KS tests for Final depth, Total TTS, and Breakpoint count."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Final depth
    real_final_depth = compute_final_depth(real_sequences)
    gen_final_depth = compute_final_depth(generated_sequences)
    ks_final_depth = scipy_stats.ks_2samp(real_final_depth, gen_final_depth)

    axes[0].hist(
        real_final_depth, bins=30, alpha=0.7, label="Real", density=True, color="blue"
    )
    axes[0].hist(
        gen_final_depth,
        bins=30,
        alpha=0.7,
        label="Generated",
        density=True,
        color="red",
    )
    ks_stat_fd = getattr(ks_final_depth, "statistic", 0.0)
    ks_pval_fd = getattr(ks_final_depth, "pvalue", 1.0)
    axes[0].set_title(
        f"Final Depth Distribution\nKS statistic: {ks_stat_fd:.4f}, p-value: {ks_pval_fd:.4f}"
    )
    axes[0].set_xlabel("Final Depth")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Total TTS
    real_total_tts = compute_total_tts(real_sequences)
    gen_total_tts = compute_total_tts(generated_sequences)
    ks_total_tts = scipy_stats.ks_2samp(real_total_tts, gen_total_tts)

    axes[1].hist(
        real_total_tts, bins=30, alpha=0.7, label="Real", density=True, color="blue"
    )
    axes[1].hist(
        gen_total_tts, bins=30, alpha=0.7, label="Generated", density=True, color="red"
    )
    ks_stat_tts = getattr(ks_total_tts, "statistic", 0.0)
    ks_pval_tts = getattr(ks_total_tts, "pvalue", 1.0)
    axes[1].set_title(
        f"Total TTS Distribution\nKS statistic: {ks_stat_tts:.4f}, p-value: {ks_pval_tts:.4f}"
    )
    axes[1].set_xlabel("Total TTS")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. Breakpoint count
    real_breakpoints = compute_breakpoint_count(real_sequences)
    gen_breakpoints = compute_breakpoint_count(generated_sequences)
    ks_breakpoints = scipy_stats.ks_2samp(real_breakpoints, gen_breakpoints)

    axes[2].hist(
        real_breakpoints,
        bins=range(int(np.max(real_breakpoints)) + 2),
        alpha=0.7,
        label="Real",
        density=True,
        color="blue",
    )
    axes[2].hist(
        gen_breakpoints,
        bins=range(int(np.max(gen_breakpoints)) + 2),
        alpha=0.7,
        label="Generated",
        density=True,
        color="red",
    )
    ks_stat_bp = getattr(ks_breakpoints, "statistic", 0.0)
    ks_pval_bp = getattr(ks_breakpoints, "pvalue", 1.0)
    axes[2].set_title(
        f"Breakpoint Count Distribution\nKS statistic: {ks_stat_bp:.4f}, p-value: {ks_pval_bp:.4f}"
    )
    axes[2].set_xlabel("Number of Breakpoints")
    axes[2].set_ylabel("Density")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved histograms with KS tests to {output_path}")

    return {
        "final_depth_ks_statistic": float(ks_stat_fd),
        "final_depth_ks_pvalue": float(ks_pval_fd),
        "total_tts_ks_statistic": float(ks_stat_tts),
        "total_tts_ks_pvalue": float(ks_pval_tts),
        "breakpoints_ks_statistic": float(ks_stat_bp),
        "breakpoints_ks_pvalue": float(ks_pval_bp),
    }


def plot_vs_histograms_with_ks(
    real_sequences: list[np.ndarray],
    generated_sequences: list[np.ndarray],
    output_path: str,
    vs_min: float = 100.0,
    vs_max: float = 5000.0,
):
    """Plot Vs velocity histograms with KS test for real vs generated profiles.

    Args:
        real_sequences: List of real sequences, each of shape (n, 2) with [TTS, depth]
        generated_sequences: List of generated sequences, same format
        output_path: Path to save the plot
        vs_min: Minimum Vs value for histogram range
        vs_max: Maximum Vs value for histogram range
    """
    # Compute all Vs values
    real_vs = compute_all_vs_values(real_sequences)
    gen_vs = compute_all_vs_values(generated_sequences)

    # Filter out invalid values (NaN, inf, or outside reasonable range)
    real_vs = real_vs[np.isfinite(real_vs) & (real_vs > 0) & (real_vs < 1e6)]
    gen_vs = gen_vs[np.isfinite(gen_vs) & (gen_vs > 0) & (gen_vs < 1e6)]

    if len(real_vs) == 0 or len(gen_vs) == 0:
        print("Warning: No valid Vs values found for histogram")
        return {}

    # Perform KS test
    ks_vs = scipy_stats.ks_2samp(real_vs, gen_vs)

    # Create histogram
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Determine bin range
    vs_min_actual = min(np.min(real_vs), np.min(gen_vs))
    vs_max_actual = max(np.max(real_vs), np.max(gen_vs))
    bins = np.linspace(
        max(0, vs_min_actual * 0.9), min(vs_max_actual * 1.1, vs_max), 50
    ).tolist()

    ax.hist(
        real_vs,
        bins=bins,
        alpha=0.7,
        label="Real",
        density=True,
        color="blue",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.hist(
        gen_vs,
        bins=bins,
        alpha=0.7,
        label="Generated",
        density=True,
        color="red",
        edgecolor="black",
        linewidth=0.5,
    )

    ks_stat_vs = getattr(ks_vs, "statistic", 0.0)
    ks_pval_vs = getattr(ks_vs, "pvalue", 1.0)

    ax.set_title(
        f"Vs Velocity Distribution\nKS statistic: {ks_stat_vs:.4f}, p-value: {ks_pval_vs:.4f}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Vs (m/s)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add statistics text
    stats_text = (
        f"Real: mean={np.mean(real_vs):.1f}, std={np.std(real_vs):.1f}\n"
        f"Generated: mean={np.mean(gen_vs):.1f}, std={np.std(gen_vs):.1f}"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved Vs histograms with KS test to {output_path}")

    return {
        "vs_ks_statistic": float(ks_stat_vs),
        "vs_ks_pvalue": float(ks_pval_vs),
        "real_vs_mean": float(np.mean(real_vs)),
        "real_vs_std": float(np.std(real_vs)),
        "real_vs_min": float(np.min(real_vs)),
        "real_vs_max": float(np.max(real_vs)),
        "gen_vs_mean": float(np.mean(gen_vs)),
        "gen_vs_std": float(np.std(gen_vs)),
        "gen_vs_min": float(np.min(gen_vs)),
        "gen_vs_max": float(np.max(gen_vs)),
    }


def main():
    """Main function - uses config file for all settings."""
    cfg = cfg_mod.cfg
    device = torch.device(cfg.device)

    # Use config file settings
    checkpoint_path = os.path.join(cfg.checkpoints_dir, "latest.pt")
    num_samples = cfg.num_eval_samples
    output_dir = cfg.plots_dir

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load model
    print(f"Loading model from {checkpoint_path}")
    model, epoch = load_checkpoint(checkpoint_path, device, cfg)

    # Load data
    print("Loading data...")
    data_loader = FlowMatchingDataLoader(data_path=Path(cfg.data_path))
    data_loader.load_data()

    assert data_loader.sequences is not None, "Sequences must be loaded"
    n_total = len(data_loader.sequences)

    # Create train/val/test splits (same as train.py)
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

    # Create test dataset (with per-profile normalization)
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
            # Denormalize using per-profile statistics
            seq_denorm = test_dataset.denormalize_sequence(seq_normalized, i)
            real_sequences.append(seq_denorm)
        else:
            real_sequences.append(seq_normalized)
    print(f"Loaded {len(real_sequences)} real sequences from test set (denormalized)")

    # Generate samples
    print(f"Generating {num_samples} samples...")
    generated_sequences_array, _ = sample_sequences(
        model,
        num_samples,
        cfg.max_length,
        device,
        train_dataset,
        cfg.ode_steps,
        wandb_run=None,
    )

    # Convert array to list of sequences (sample_sequences returns np.array with dtype=object)
    # Note: sample_sequences already denormalizes the generated sequences
    generated_sequences = []
    for i in range(len(generated_sequences_array)):
        seq = generated_sequences_array[i]
        if isinstance(seq, np.ndarray):
            generated_sequences.append(seq)
        else:
            # Handle case where seq might be a scalar or other type
            generated_sequences.append(np.array(seq))

    print(f"Generated {len(generated_sequences)} sequences (already denormalized)")

    # Set output directory
    os.makedirs(output_dir, exist_ok=True)

    # 1. Side-by-side profile comparison
    print("Creating side-by-side profile comparison...")
    side_by_side_path = os.path.join(
        output_dir, f"profiles_comparison_epoch_{epoch}.png"
    )
    plot_profile_comparison_side_by_side(
        real_sequences, generated_sequences, side_by_side_path, n_profiles=15
    )

    # 2. Profile panels (detailed comparison)
    print("Creating detailed profile panels...")
    profile_panels_path = os.path.join(output_dir, f"profile_panels_epoch_{epoch}.png")
    plot_profile_panels(
        real_sequences, generated_sequences, profile_panels_path, n_profiles=20
    )

    # 3. Histograms with KS tests
    print("Creating histograms with KS tests...")
    histograms_path = os.path.join(output_dir, f"histograms_ks_epoch_{epoch}.png")
    ks_results = plot_histograms_with_ks(
        real_sequences, generated_sequences, histograms_path
    )

    # 3b. Vs velocity histograms with KS test
    print("Creating Vs velocity histograms with KS test...")
    vs_histograms_path = os.path.join(output_dir, f"vs_histograms_ks_epoch_{epoch}.png")
    vs_ks_results = plot_vs_histograms_with_ks(
        real_sequences,
        generated_sequences,
        vs_histograms_path,
        vs_min=cfg.vs_min,
        vs_max=cfg.vs_max,
    )

    # 4. Monotonicity and physical checks
    print("Computing monotonicity and physical checks...")
    real_monotonicity = check_monotonicity(real_sequences)
    gen_monotonicity = check_monotonicity(generated_sequences)
    real_physical = check_final_depth_physical(real_sequences)
    gen_physical = check_final_depth_physical(generated_sequences)

    print(f"Real monotonicity rate: {real_monotonicity['monotonic_rate']:.4f}")
    print(f"Generated monotonicity rate: {gen_monotonicity['monotonic_rate']:.4f}")
    print(f"Real final depth positive rate: {real_physical['positive_rate']:.4f}")
    print(f"Generated final depth positive rate: {gen_physical['positive_rate']:.4f}")

    # 4b. Vs constraints
    print("Computing Vs constraint checks...")
    real_vs_constraints = check_vs_constraints(
        real_sequences, cfg.vs_min, cfg.vs_max, cfg.min_dt
    )
    gen_vs_constraints = check_vs_constraints(
        generated_sequences, cfg.vs_min, cfg.vs_max, cfg.min_dt
    )

    print(f"Real Vs bounds rate: {real_vs_constraints['vs_bounds_rate']:.4f}")
    print(f"Generated Vs bounds rate: {gen_vs_constraints['vs_bounds_rate']:.4f}")
    print(f"Real min_dt rate: {real_vs_constraints['min_dt_rate']:.4f}")
    print(f"Generated min_dt rate: {gen_vs_constraints['min_dt_rate']:.4f}")
    if gen_vs_constraints["total_count"] > 0:
        print(
            f"Generated Vs range: [{gen_vs_constraints['min_vs']:.2f}, {gen_vs_constraints['max_vs']:.2f}] m/s"
        )
        print(
            f"Generated Vs violations: {gen_vs_constraints['vs_below_min']} below min, {gen_vs_constraints['vs_above_max']} above max"
        )

    # 5. Diversity metrics
    print("Computing diversity metrics...")
    real_diversity = compute_pairwise_distance(
        real_sequences[: min(100, len(real_sequences))]
    )
    gen_diversity = compute_pairwise_distance(
        generated_sequences[: min(100, len(generated_sequences))]
    )

    print(
        f"Real mean pairwise distance: {real_diversity['mean_distance']:.4f} ± {real_diversity['std_distance']:.4f}"
    )
    print(
        f"Generated mean pairwise distance: {gen_diversity['mean_distance']:.4f} ± {gen_diversity['std_distance']:.4f}"
    )

    # Save results
    results = {
        "epoch": epoch,
        "ks_tests": ks_results,
        "vs_ks_tests": vs_ks_results,
        "real_monotonicity": real_monotonicity,
        "generated_monotonicity": gen_monotonicity,
        "real_physical_checks": real_physical,
        "generated_physical_checks": gen_physical,
        "real_vs_constraints": real_vs_constraints,
        "generated_vs_constraints": gen_vs_constraints,
        "real_diversity": real_diversity,
        "generated_diversity": gen_diversity,
    }

    results_path = os.path.join(output_dir, f"comprehensive_eval_epoch_{epoch}.json")
    with open(results_path, "w") as f:
        json.dump(convert_to_json_serializable(results), f, indent=2)

    print(f"\nResults saved to {results_path}")
    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
