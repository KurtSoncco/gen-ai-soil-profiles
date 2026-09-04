#!/usr/bin/env python3
"""
Vs Pairs FFM Evaluation Script

This script evaluates a trained flow matching model on vs pairs.
It computes Mean MSE and Std Dev MSE for each dimension and creates
comprehensive distribution comparison plots.
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FuncFormatter

try:
    from . import config as cfg_mod
    from . import models as models_mod
    from . import utils as utils_mod
    from .data import create_dataloader
except ImportError:  # fallback when running as script
    import config as cfg_mod
    import models as models_mod
    import utils as utils_mod

    from data import create_dataloader  # type: ignore


def load_latest_checkpoint(dir_path: str) -> str:
    """Load the latest checkpoint from the directory."""
    dir_path = str(Path(dir_path))
    files = [
        f
        for f in Path(dir_path).iterdir()
        if f.is_file() and f.name.startswith("checkpoint_") and f.name.endswith(".pt")
    ]
    if not files:
        raise FileNotFoundError(f"No checkpoints found in {dir_path}")

    # Sort by step number
    files.sort(
        key=lambda x: (
            int(x.stem.split("_")[-1]) if x.stem != "checkpoint_final" else float("inf")
        )
    )
    return str(files[-1])


def compute_mse_metrics(real_pairs: np.ndarray, generated_pairs: np.ndarray) -> dict:
    """
    Compute Mean MSE and Std Dev MSE for each vs dimension.

    Args:
        real_pairs: Real vs pairs (N, 4)
        generated_pairs: Generated vs pairs (N, 4)

    Returns:
        Dictionary with metrics for each dimension
    """
    feature_names = ["vs30", "vs50", "vs100", "vs200"]
    metrics = {}

    for idx, name in enumerate(feature_names):
        real_vals = real_pairs[:, idx]
        gen_vals = generated_pairs[:, idx]

        # Compute squared errors for each sample
        se = (real_vals - gen_vals) ** 2

        # Mean MSE
        metrics[f"{name}_mse_mean"] = float(np.mean(se))

        # Std Dev MSE
        metrics[f"{name}_mse_std"] = float(np.std(se))

        # Root Mean Squared Error (RMSE)
        metrics[f"{name}_rmse"] = float(np.sqrt(np.mean(se)))

        # Mean Absolute Error
        metrics[f"{name}_mae"] = float(np.mean(np.abs(real_vals - gen_vals)))

        # Correlation
        if len(real_vals) > 1:
            correlation = np.corrcoef(real_vals, gen_vals)[0, 1]
            metrics[f"{name}_correlation"] = float(correlation)
        else:
            metrics[f"{name}_correlation"] = float("nan")

    return metrics


def create_comprehensive_plot(
    real_pairs: np.ndarray,
    generated_pairs: np.ndarray,
    output_dir: str,
    step: int,
    metrics: dict,
    all_metrics: dict = {},
) -> None:
    """
    Create a comprehensive comparison plot with distributions and metrics.

    Args:
        real_pairs: Real vs pairs (N, 4)
        generated_pairs: Generated vs pairs (N, 4)
        output_dir: Output directory for saving plots
        step: Training step
        metrics: Dictionary of computed MSE metrics
        all_metrics: Dictionary of all computed metrics (mean, std, etc.)
    """
    os.makedirs(output_dir, exist_ok=True)

    feature_names = ["vs30", "vs50", "vs100", "vs200"]

    # Create figure with subplots - 4 rows to accommodate MSE plot
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 4, hspace=0.5, wspace=0.35, height_ratios=[1, 1, 1, 1])

    # Plot 1: Distribution comparisons
    for idx, name in enumerate(feature_names):
        ax = fig.add_subplot(gs[0, idx])

        real_vals = real_pairs[:, idx]
        gen_vals = generated_pairs[:, idx]

        # Create bins
        all_vals = np.concatenate([real_vals, gen_vals])
        bins = np.linspace(all_vals.min(), all_vals.max(), 50)

        ax.hist(
            real_vals, bins=bins, alpha=0.5, label="Real", density=True, color="blue"
        )
        ax.hist(
            gen_vals, bins=bins, alpha=0.5, label="Generated", density=True, color="red"
        )

        ax.set_xlabel(f"{name} (m/s)")
        ax.set_ylabel("Density")
        ax.set_title(f"{name} Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add KS statistic
        ks = utils_mod.ks_statistic(real_vals, gen_vals)
        ax.text(
            0.05,
            0.95,
            f"KS={ks:.3f}\nRMSE={metrics.get(f'{name}_rmse', 0):.1f}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    # Plot 2: Scatter plots vs30 vs vs50
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(
        real_pairs[:, 0], real_pairs[:, 1], alpha=0.3, s=20, label="Real", color="blue"
    )
    ax2.scatter(
        generated_pairs[:, 0],
        generated_pairs[:, 1],
        alpha=0.3,
        s=20,
        label="Generated",
        color="red",
    )
    ax2.set_xlabel("vs30")
    ax2.set_ylabel("vs50")
    ax2.set_title("vs30 vs vs50")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Scatter plots vs100 vs vs200
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.scatter(
        real_pairs[:, 2], real_pairs[:, 3], alpha=0.3, s=20, label="Real", color="blue"
    )
    ax3.scatter(
        generated_pairs[:, 2],
        generated_pairs[:, 3],
        alpha=0.3,
        s=20,
        label="Generated",
        color="red",
    )
    ax3.set_xlabel("vs100")
    ax3.set_ylabel("vs200")
    ax3.set_title("vs100 vs vs200")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 3b: Mean comparison bar chart
    if all_metrics:
        ax3b = fig.add_subplot(gs[1, 2])
        mean_real = [all_metrics[f"{name}_mean_real"] for name in feature_names]
        mean_gen = [all_metrics[f"{name}_mean_gen"] for name in feature_names]

        x_pos = np.arange(len(feature_names))
        width = 0.35

        bars1 = ax3b.bar(
            x_pos - width / 2,
            mean_real,
            width,
            label="Real",
            color="blue",
            alpha=0.7,
            edgecolor="navy",
            linewidth=1.5,
        )
        bars2 = ax3b.bar(
            x_pos + width / 2,
            mean_gen,
            width,
            label="Generated",
            color="red",
            alpha=0.7,
            edgecolor="maroon",
            linewidth=1.5,
        )

        ax3b.set_xlabel("Vs Dimension", fontweight="bold")
        ax3b.set_ylabel("Mean Value", fontweight="bold")
        ax3b.set_title("Mean Comparison", fontweight="bold")
        ax3b.set_xticks(x_pos)
        ax3b.set_xticklabels(feature_names, fontweight="bold")
        ax3b.legend()
        ax3b.grid(True, alpha=0.3, axis="y", linestyle="--")
        ax3b.set_ylim(bottom=0)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax3b.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + height * 0.02,
                    f"{height:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    # Plot 3c: Std comparison bar chart
    if all_metrics:
        ax3c = fig.add_subplot(gs[1, 3])
        std_real = [all_metrics[f"{name}_std_real"] for name in feature_names]
        std_gen = [all_metrics[f"{name}_std_gen"] for name in feature_names]

        x_pos = np.arange(len(feature_names))
        width = 0.35

        bars1 = ax3c.bar(
            x_pos - width / 2,
            std_real,
            width,
            label="Real",
            color="green",
            alpha=0.7,
            edgecolor="darkgreen",
            linewidth=1.5,
        )
        bars2 = ax3c.bar(
            x_pos + width / 2,
            std_gen,
            width,
            label="Generated",
            color="orange",
            alpha=0.7,
            edgecolor="darkorange",
            linewidth=1.5,
        )

        ax3c.set_xlabel("Vs Dimension", fontweight="bold")
        ax3c.set_ylabel("Standard Deviation", fontweight="bold")
        ax3c.set_title("Std Dev Comparison", fontweight="bold")
        ax3c.set_xticks(x_pos)
        ax3c.set_xticklabels(feature_names, fontweight="bold")
        ax3c.legend()
        ax3c.grid(True, alpha=0.3, axis="y", linestyle="--")
        ax3c.set_ylim(bottom=0)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax3c.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + height * 0.02,
                    f"{height:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    # Plot 4: MSE comparison bar chart with scientific notation (moved to row 2)
    ax4 = fig.add_subplot(gs[2, :])
    mse_means = [metrics[f"{name}_mse_mean"] for name in feature_names]
    mse_stds = [metrics[f"{name}_mse_std"] for name in feature_names]

    x_pos = np.arange(len(feature_names))
    bars = ax4.bar(
        x_pos,
        mse_means,
        yerr=mse_stds,
        alpha=0.7,
        capsize=5,
        color="skyblue",
        edgecolor="navy",
        linewidth=1.5,
    )

    # Use scientific notation for y-axis

    ax4.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:.2e}"))

    # Add value labels on top of bars
    for i, (bar, mean, std) in enumerate(zip(bars, mse_means, mse_stds)):
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + std + height * 0.05,
            f"{mean:.2e}\n±{std:.2e}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )

    ax4.set_xlabel("Vs Dimension", fontweight="bold")
    ax4.set_ylabel("Mean Squared Error", fontweight="bold")
    ax4.set_title("MSE by Dimension (with Std Dev)", fontweight="bold")
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(feature_names, fontweight="bold")
    ax4.grid(True, alpha=0.3, axis="y", linestyle="--")
    ax4.set_ylim(bottom=0)

    # Plot 5: Q-Q plots for each dimension (moved to row 3)
    for idx, name in enumerate(feature_names):
        ax = fig.add_subplot(gs[3, idx])

        real_vals = real_pairs[:, idx]
        gen_vals = generated_pairs[:, idx]

        # Create Q-Q plot
        real_sorted = np.sort(real_vals)
        gen_sorted = np.sort(gen_vals)

        # Percentile-based matching
        percentiles = np.linspace(0, 100, min(len(real_sorted), len(gen_sorted)))
        real_percentiles = np.percentile(real_vals, percentiles)
        gen_percentiles = np.percentile(gen_vals, percentiles)

        ax.scatter(real_percentiles, gen_percentiles, alpha=0.5, s=20)

        # Add diagonal line for reference
        min_val = min(real_percentiles.min(), gen_percentiles.min())
        max_val = max(real_percentiles.max(), gen_percentiles.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.5)

        ax.set_xlabel(f"Real {name}")
        ax.set_ylabel(f"Generated {name}")
        ax.set_title(f"{name} Q-Q Plot")
        ax.grid(True, alpha=0.3)

    # Add overall title with statistics
    overall_title = f"Vs Pairs Flow Matching Evaluation @ Step {step}\n"
    overall_title += (
        f"MSE: vs30={metrics['vs30_mse_mean']:.0f}±{metrics['vs30_mse_std']:.0f}, "
    )
    overall_title += (
        f"vs50={metrics['vs50_mse_mean']:.0f}±{metrics['vs50_mse_std']:.0f}, "
    )
    overall_title += (
        f"vs100={metrics['vs100_mse_mean']:.0f}±{metrics['vs100_mse_std']:.0f}, "
    )
    overall_title += (
        f"vs200={metrics['vs200_mse_mean']:.0f}±{metrics['vs200_mse_std']:.0f}"
    )

    fig.suptitle(overall_title, fontsize=14, y=0.995)

    # Save plot
    output_path = os.path.join(output_dir, f"evaluation_comprehensive_{step}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved comprehensive evaluation plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Vs Pairs Flow Matching Model"
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default=None, help="Path to checkpoint file"
    )
    parser.add_argument(
        "--num_samples", type=int, default=1000, help="Number of samples to generate"
    )
    parser.add_argument(
        "--ode_steps", type=int, default=None, help="Number of ODE integration steps"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Output directory for results"
    )
    args = parser.parse_args()

    cfg = cfg_mod.cfg
    device = torch.device(cfg.device)

    # Determine checkpoint path
    if args.checkpoint_path is None:
        checkpoint_path = load_latest_checkpoint(cfg.out_dir)
    else:
        checkpoint_path = args.checkpoint_path

    print(f"Loading checkpoint from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    step = checkpoint.get("step", 0)

    # Get dataset info for normalization
    _, max_length, dataset = create_dataloader(
        batch_size=cfg.batch_size, num_workers=0, shuffle=False
    )

    # Create model
    model = models_mod.create_model(cfg.model_type, cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    print(
        f"Loaded {cfg.model_type.upper()} model with {sum(p.numel() for p in model.parameters()):,} parameters"
    )
    print(f"Model trained for {step} steps")

    # Generate samples
    num_samples = args.num_samples
    ode_steps = args.ode_steps or cfg.ode_steps

    print(f"Generating {num_samples} samples with {ode_steps} ODE steps...")

    # Create initial noise
    initial_noise = torch.randn(num_samples, 1, max_length).to(device)

    # Generate samples
    with torch.no_grad():
        samples_normalized = utils_mod.sample_ffm(
            model, initial_noise, ode_steps, device
        )

    # Denormalize samples
    samples = dataset.denormalize_batch(samples_normalized)
    samples_np = samples.squeeze(1).cpu().numpy()

    # Load real data for comparison
    # Sample same number as generated for fair comparison
    real_pairs_all = dataset.raw_pairs
    if len(real_pairs_all) >= num_samples:
        indices = np.random.choice(len(real_pairs_all), size=num_samples, replace=False)
        real_pairs_np = real_pairs_all[indices]
    else:
        real_pairs_np = real_pairs_all
        print(f"Warning: Dataset has only {len(real_pairs_all)} samples, using all")

    # Compute metrics
    print("\nComputing metrics...")
    mse_metrics = compute_mse_metrics(real_pairs_np, samples_np)

    # Print metrics
    print("\n" + "=" * 80)
    print("EVALUATION METRICS")
    print("=" * 80)
    feature_names = ["vs30", "vs50", "vs100", "vs200"]

    for name in feature_names:
        print(f"\n{name}:")
        print(
            f"  MSE:     {mse_metrics[f'{name}_mse_mean']:.2f} ± {mse_metrics[f'{name}_mse_std']:.2f}"
        )
        print(f"  RMSE:    {mse_metrics[f'{name}_rmse']:.2f}")
        print(f"  MAE:     {mse_metrics[f'{name}_mae']:.2f}")
        print(f"  Corr:    {mse_metrics[f'{name}_correlation']:.3f}")

    # Also compute additional statistics
    all_metrics = utils_mod.compute_metrics(real_pairs_np, samples_np)

    print("\n" + "=" * 80)
    print("DISTRIBUTION STATISTICS")
    print("=" * 80)
    for name in feature_names:
        print(f"\n{name}:")
        print(f"  Real Mean:    {all_metrics[f'{name}_mean_real']:.2f}")
        print(f"  Real Std:     {all_metrics[f'{name}_std_real']:.2f}")
        print(f"  Gen Mean:     {all_metrics[f'{name}_mean_gen']:.2f}")
        print(f"  Gen Std:      {all_metrics[f'{name}_std_gen']:.2f}")
        print(f"  KS Statistic: {all_metrics[f'{name}_ks']:.3f}")

    # Create comprehensive plot
    output_dir = args.output_dir or cfg.plots_dir
    print("\nCreating comprehensive evaluation plots...")
    create_comprehensive_plot(
        real_pairs_np, samples_np, output_dir, step, mse_metrics, all_metrics
    )

    # Save metrics to file
    import json

    # Convert all metrics to native Python types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        return obj

    all_metrics_converted = convert_to_native({**mse_metrics, **all_metrics})
    metrics_file = os.path.join(cfg.out_dir, f"evaluation_metrics_step_{step}.json")
    with open(metrics_file, "w") as f:
        json.dump(all_metrics_converted, f, indent=2)
    print(f"✓ Saved evaluation metrics to {metrics_file}")

    print("\n" + "=" * 80)
    print("Evaluation completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
