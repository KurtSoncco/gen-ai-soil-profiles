"""Utility functions for Vs Pairs Flow Matching.

This module provides:
- Sampling functions for generating vs pairs
- Metrics computation
- Plotting and visualization utilities
- Statistical analysis functions
"""

from __future__ import annotations

import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch


def sample_ffm(model, initial_noise, ode_steps, device):
    """
    Sample from trained flow matching model using RK4 (Runge-Kutta 4) ODE integration.

    Args:
        model: Trained flow matching model
        initial_noise: Initial noise tensor (batch_size, 1, 4)
        ode_steps: Number of ODE integration steps
        device: PyTorch device

    Returns:
        Generated samples with same shape as initial_noise
    """
    model.eval()

    # Start with noise at t=0
    u = initial_noise.to(device)
    dt = 1.0 / ode_steps

    for i in range(ode_steps):
        # Current time
        t_val = i * dt
        t = torch.full((u.shape[0], 1), t_val).to(device)

        # Predict vector field at current state
        k1 = model(u, t) * dt

        # Second stage: midpoint
        k2_t = t_val + 0.5 * dt
        k2 = model(u + 0.5 * k1, torch.full((u.shape[0], 1), k2_t).to(device)) * dt

        # Third stage: midpoint
        k3 = model(u + 0.5 * k2, torch.full((u.shape[0], 1), k2_t).to(device)) * dt

        # Fourth stage: endpoint
        k4_t = t_val + dt
        k4 = model(u + k3, torch.full((u.shape[0], 1), k4_t).to(device)) * dt

        # Update u using weighted average of all stages
        u = u + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

    return u


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Kolmogorov-Smirnov statistic between two distributions."""
    if a.size == 0 or b.size == 0:
        return float("nan")
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    ia = ib = 0
    na, nb = len(a_sorted), len(b_sorted)
    d = 0.0
    while ia < na and ib < nb:
        if a_sorted[ia] <= b_sorted[ib]:
            ia += 1
        else:
            ib += 1
        fa = ia / na
        fb = ib / nb
        d = max(d, abs(fa - fb))
    return d


def plot_loss_curves(loss_history: List[float], output_dir: str, step: int) -> None:
    """Plot training loss curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title(f"FFM Training Loss (Step {step})")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)

    output_path = os.path.join(output_dir, f"loss_curve_{step}.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved loss curve to {output_path}")


def plot_pairs_distribution(
    real_pairs: np.ndarray,
    generated_pairs: np.ndarray,
    output_dir: str,
    step: int,
) -> None:
    """Plot distributions of vs pairs (scatter plots and histograms)."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    feature_names = ["vs30", "vs50", "vs100", "vs200"]

    for idx, (ax, name) in enumerate(zip(axes, feature_names)):
        # Extract the column
        real_vals = real_pairs[:, idx]
        gen_vals = generated_pairs[:, idx]

        # Plot histograms
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

        # Compute KS statistic
        ks = ks_statistic(real_vals, gen_vals)
        ax.text(
            0.05,
            0.95,
            f"KS={ks:.3f}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.suptitle(f"Vs Pairs Distribution Comparison @ Step {step}", fontsize=16, y=1.01)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"pairs_distribution_{step}.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved pairs distribution plot to {output_path}")


def plot_pairs_scatter(
    real_pairs: np.ndarray,
    generated_pairs: np.ndarray,
    output_dir: str,
    step: int,
) -> None:
    """Plot scatter plots comparing pairs (e.g., vs30 vs vs50, vs100 vs vs200)."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    feature_names = ["vs30", "vs50", "vs100", "vs200"]

    # Plot vs30 vs vs50
    axes[0].scatter(
        real_pairs[:, 0], real_pairs[:, 1], alpha=0.3, s=20, label="Real", color="blue"
    )
    axes[0].scatter(
        generated_pairs[:, 0],
        generated_pairs[:, 1],
        alpha=0.3,
        s=20,
        label="Generated",
        color="red",
    )
    axes[0].set_xlabel(feature_names[0])
    axes[0].set_ylabel(feature_names[1])
    axes[0].set_title(f"{feature_names[0]} vs {feature_names[1]}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot vs100 vs vs200
    axes[1].scatter(
        real_pairs[:, 2], real_pairs[:, 3], alpha=0.3, s=20, label="Real", color="blue"
    )
    axes[1].scatter(
        generated_pairs[:, 2],
        generated_pairs[:, 3],
        alpha=0.3,
        s=20,
        label="Generated",
        color="red",
    )
    axes[1].set_xlabel(feature_names[2])
    axes[1].set_ylabel(feature_names[3])
    axes[1].set_title(f"{feature_names[2]} vs {feature_names[3]}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"Vs Pairs Scatter Comparison @ Step {step}", fontsize=16)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"pairs_scatter_{step}.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved pairs scatter plot to {output_path}")


def compute_metrics(real_pairs: np.ndarray, generated_pairs: np.ndarray) -> dict:
    """Compute metrics comparing real and generated vs pairs."""
    feature_names = ["vs30", "vs50", "vs100", "vs200"]
    metrics = {}

    for idx, name in enumerate(feature_names):
        real_vals = real_pairs[:, idx]
        gen_vals = generated_pairs[:, idx]

        # KS statistic
        ks = ks_statistic(real_vals, gen_vals)
        metrics[f"{name}_ks"] = ks

        # Mean and std
        metrics[f"{name}_mean_real"] = np.mean(real_vals)
        metrics[f"{name}_mean_gen"] = np.mean(gen_vals)
        metrics[f"{name}_std_real"] = np.std(real_vals)
        metrics[f"{name}_std_gen"] = np.std(gen_vals)

        # Mean absolute error
        metrics[f"{name}_mae"] = np.mean(np.abs(real_vals - gen_vals))

    return metrics


def log_metrics(model, cfg, device, step, output_dir, dataset, wandb=None):
    """Log metrics and create plots during training."""
    # Generate samples for evaluation
    num_eval_samples = min(256, len(dataset))
    z_eval = torch.randn(num_eval_samples, 1, 4).to(device)

    with torch.no_grad():
        model.eval()
        generated_samples = sample_ffm(model, z_eval, cfg.ode_steps, device)
        model.train()

    # Denormalize samples
    generated_samples_denorm = dataset.denormalize_batch(generated_samples)
    generated_samples_np = generated_samples_denorm.squeeze(1).cpu().numpy()

    # Load real data for comparison - randomly sample to match generated samples
    real_pairs_all = dataset.raw_pairs
    indices = np.random.choice(
        len(real_pairs_all), size=num_eval_samples, replace=False
    )
    real_pairs_np = real_pairs_all[indices]

    # Compute metrics
    metrics = compute_metrics(real_pairs_np, generated_samples_np)

    print(f"[metrics] step={step}")
    for name, value in metrics.items():
        print(f"  {name}={value:.4f}")

    # Log to wandb if available
    if wandb is not None:
        try:
            log_dict = {"step": step}
            log_dict.update({f"metrics/{k}": v for k, v in metrics.items()})
            wandb.log(log_dict)
        except Exception as e:
            print(f"[info] wandb logging failed: {e}")

    # Create plots periodically
    if step % cfg.checkpoint_every == 0 or step == cfg.num_steps - 1:
        plot_pairs_distribution(
            real_pairs_np, generated_samples_np, cfg.plots_dir, step
        )
        plot_pairs_scatter(real_pairs_np, generated_samples_np, cfg.plots_dir, step)

    return metrics


if __name__ == "__main__":
    print("Testing utils...")

    # Test sample_ffm with dummy model
    class DummyModel(torch.nn.Module):
        def forward(self, x, t):
            return torch.zeros_like(x)

    model = DummyModel()
    noise = torch.randn(4, 1, 4)
    samples = sample_ffm(model, noise, 10, "cpu")
    print(f"Sample shape: {samples.shape}")

    # Test metrics
    real = np.random.rand(100, 4) * 500 + 200
    gen = np.random.rand(100, 4) * 500 + 200
    metrics = compute_metrics(real, gen)
    print(f"Computed {len(metrics)} metrics")

    print("All utils tests passed!")
