from __future__ import annotations

import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def compute_real_vs30_and_density(parquet_path: str) -> tuple[np.ndarray, float]:
    """Compute real Vs30 distribution and average samples per meter from parquet data."""
    df = pd.read_parquet(parquet_path)
    required = {"velocity_metadata_id", "depth", "vs_value"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Parquet missing columns required for Vs30: {missing}")

    vs30_values = []
    samples_per_meter = []
    for _, g in df.groupby("velocity_metadata_id"):
        g = g.sort_values("depth")
        depths = g["depth"].to_numpy(dtype=float)
        vs = g["vs_value"].to_numpy(dtype=float)
        if depths.size == 0:
            continue
        # ensure starts at 0m if needed
        if depths[0] > 0:
            depths = np.insert(depths, 0, 0.0)
            vs = np.insert(vs, 0, vs[0])
        # limit to top 30 m
        top_mask = depths <= 30.0
        if not np.any(top_mask):
            continue
        d = depths[top_mask]
        v = vs[top_mask]
        if d[-1] < 30.0:
            d = np.append(d, 30.0)
            v = np.append(v, v[-1])
        dz = np.diff(d)
        v_mid = v[1:]  # piecewise-constant per segment
        denom = np.sum(dz / np.maximum(v_mid, 1e-6))
        if denom > 0:
            vs30_values.append(30.0 / denom)
        samples_per_meter.append((np.count_nonzero(top_mask)) / 30.0)

    if not vs30_values:
        return np.array([]), 1.0
    return np.asarray(vs30_values, dtype=float), float(np.mean(samples_per_meter))


def compute_generated_vs30(samples: np.ndarray, samples_per_meter: float) -> np.ndarray:
    """Compute Vs30 for generated samples."""
    # samples: (N, 1, L)
    if samples_per_meter <= 0:
        samples_per_meter = 1.0

    n_take = max(1, int(30.0 * samples_per_meter))
    n_take = min(n_take, samples.shape[-1])
    dz = 1.0 / samples_per_meter

    # Extract velocity values for top 30m
    v = samples[:, 0, :n_take]

    # Ensure positive velocities (Vs should be > 0)
    v = np.maximum(v, 50.0)  # Minimum reasonable Vs value

    # Calculate Vs30 using harmonic mean
    denom = np.sum((dz / v), axis=1)
    vs30_gen = 30.0 / np.maximum(denom, 1e-9)

    # Clamp to reasonable range
    vs30_gen = np.clip(vs30_gen, 50.0, 2000.0)

    return vs30_gen


def compute_vs100(samples: np.ndarray, samples_per_meter: float) -> np.ndarray:
    """Compute Vs100 for generated samples."""
    # Find the layer closest to 100m depth
    depth_per_sample = 1.0 / samples_per_meter
    target_samples = int(100.0 / depth_per_sample)
    target_samples = min(target_samples, samples.shape[-1] - 1)

    return samples[:, 0, target_samples]


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


def compute_vs30(profile: np.ndarray, depth_step: float = 0.5) -> float:
    """
    Compute Vs30 (time-averaged shear-wave velocity in top 30m) for a profile.

    Args:
        profile: Vs values array
        depth_step: Depth step in meters (default 0.5m)

    Returns:
        Vs30 value in m/s
    """
    # Calculate depths
    depths = np.arange(len(profile)) * depth_step

    # Find layers within 30m
    mask = depths < 30.0
    if not np.any(mask):
        return float(np.mean(profile))

    profile_30m = profile[mask]
    depths_30m = depths[mask]

    # Calculate layer thicknesses
    layer_thicknesses = np.diff(np.concatenate([[0], depths_30m]))

    # Calculate travel time for each layer
    travel_times = layer_thicknesses / profile_30m[:-1]  # Exclude last point

    # Vs30 = 30 / total_travel_time
    total_travel_time = np.sum(travel_times)
    if total_travel_time > 0:
        vs30 = 30.0 / total_travel_time
    else:
        vs30 = float(np.mean(profile_30m))

    return vs30


def compute_vs30_distribution(
    profiles: np.ndarray, depth_step: float = 0.5
) -> np.ndarray:
    """Compute Vs30 distribution for a batch of profiles."""
    vs30_values = []
    for i in range(profiles.shape[0]):
        profile = profiles[i, 0, :]  # Remove channel dimension
        vs30 = compute_vs30(profile, depth_step)
        vs30_values.append(vs30)
    return np.array(vs30_values)


def log_vs30_metrics(
    model,
    cfg,
    device,
    step,
    output_dir,
    real_vs30,
    avg_samples_per_meter,
    max_length,
    wandb=None,
):
    """Log Vs30 metrics and create plots during training."""
    if real_vs30.size == 0:
        print("[metrics] Skipping Vs30 metrics: real distribution unavailable.")
        return

    # Generate samples for evaluation
    num_eval_samples = 512
    z_eval = torch.randn(num_eval_samples, 1, max_length).to(device)

    with torch.no_grad():
        model.eval()
        generated_samples = sample_ffm(model, z_eval, cfg.ode_steps, device)
        model.train()

    # Convert to numpy
    generated_samples_np = generated_samples.cpu().numpy()

    # Debug: Print sample statistics
    print(f"[debug] Generated samples shape: {generated_samples_np.shape}")
    print(
        f"[debug] Generated samples range: [{np.min(generated_samples_np):.3f}, {np.max(generated_samples_np):.3f}]"
    )
    print(f"[debug] Generated samples mean: {np.mean(generated_samples_np):.3f}")
    print(f"[debug] Samples per meter: {avg_samples_per_meter}")

    # Check if generated samples are in reasonable range
    sample_range = np.max(generated_samples_np) - np.min(generated_samples_np)
    if sample_range < 10:  # Very small range indicates poor training
        print(
            f"[warning] Generated samples have very small range ({sample_range:.3f}). Model may not be properly trained."
        )

    gen_vs30 = compute_generated_vs30(generated_samples_np, avg_samples_per_meter)

    # Debug: Print Vs30 statistics
    print(
        f"[debug] Generated Vs30 range: [{np.min(gen_vs30):.3f}, {np.max(gen_vs30):.3f}]"
    )
    print(f"[debug] Generated Vs30 mean: {np.mean(gen_vs30):.3f}")
    print(
        f"[debug] Real Vs30 range: [{np.min(real_vs30):.3f}, {np.max(real_vs30):.3f}]"
    )
    print(f"[debug] Real Vs30 mean: {np.mean(real_vs30):.3f}")

    ks = ks_statistic(real_vs30, gen_vs30)
    print(f"[metrics] step={step} KS(real_vs30, gen_vs30)={ks:.4f}")

    # Log to wandb if available
    if wandb is not None:
        try:
            wandb.log(
                {
                    "step": step,
                    "metrics/generated_vs30_mean": np.mean(gen_vs30),
                    "metrics/generated_vs30_std": np.std(gen_vs30),
                    "metrics/generated_vs30_min": np.min(gen_vs30),
                    "metrics/generated_vs30_max": np.max(gen_vs30),
                    "metrics/generated_samples_mean": np.mean(generated_samples_np),
                    "metrics/generated_samples_std": np.std(generated_samples_np),
                    "metrics/generated_samples_min": np.min(generated_samples_np),
                    "metrics/generated_samples_max": np.max(generated_samples_np),
                    "metrics/vs30_ks_statistic": ks,
                    "metrics/sample_range": sample_range,
                }
            )

            # Log histograms to wandb
            wandb.log(
                {
                    "histograms/vs30_distribution": wandb.Histogram(gen_vs30),
                    "histograms/generated_samples": wandb.Histogram(
                        generated_samples_np.flatten()
                    ),
                }
            )
        except Exception as e:
            print(f"[info] wandb logging failed: {e}, continuing without it")
    else:
        print("[info] wandb not available, skipping wandb logging")

    # Only save detailed plots for final checkpoint or every 50 steps
    if step % 50 == 0 or step == cfg.num_steps - 1:
        # save histogram plot
        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(12, 8))

        # Create subplots for better visualization
        plt.subplot(2, 2, 1)
        # Use consistent bin range for both histograms
        all_vs30 = np.concatenate([real_vs30, gen_vs30])
        bin_range = (np.min(all_vs30), np.max(all_vs30))
        bins = np.linspace(bin_range[0], bin_range[1], 50)

        plt.hist(
            real_vs30, bins=bins, alpha=0.6, label="real", density=True, color="blue"
        )
        plt.hist(
            gen_vs30, bins=bins, alpha=0.6, label="generated", density=True, color="red"
        )
        plt.xlabel("Vs30 (m/s)")
        plt.ylabel("Density")
        plt.title(f"Vs30 Distribution @ Step {step}\nKS={ks:.3f}")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add sample distribution plot
        plt.subplot(2, 2, 2)
        plt.hist(generated_samples_np.flatten(), bins=50, alpha=0.7, color="green")
        plt.xlabel("Generated Sample Values")
        plt.ylabel("Frequency")
        plt.title(
            f"Generated Sample Distribution\nRange: [{np.min(generated_samples_np):.3f}, {np.max(generated_samples_np):.3f}]"
        )
        plt.grid(True, alpha=0.3)

        # Add Vs30 comparison
        plt.subplot(2, 2, 3)
        plt.scatter(
            range(len(gen_vs30)),
            gen_vs30,
            alpha=0.6,
            s=10,
            color="red",
            label="Generated",
        )
        plt.scatter(
            range(len(real_vs30[: len(gen_vs30)])),
            real_vs30[: len(gen_vs30)],
            alpha=0.6,
            s=10,
            color="blue",
            label="Real",
        )
        plt.xlabel("Sample Index")
        plt.ylabel("Vs30 (m/s)")
        plt.title("Vs30 Values Comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add statistics text
        plt.subplot(2, 2, 4)
        plt.axis("off")
        stats_text = f"""Statistics Summary (Step {step})

Generated Samples:
  Range: [{np.min(generated_samples_np):.3f}, {np.max(generated_samples_np):.3f}]
  Mean: {np.mean(generated_samples_np):.3f}
  Std: {np.std(generated_samples_np):.3f}

Generated Vs30:
  Range: [{np.min(gen_vs30):.1f}, {np.max(gen_vs30):.1f}]
  Mean: {np.mean(gen_vs30):.1f}

Real Vs30:
  Range: [{np.min(real_vs30):.1f}, {np.max(real_vs30):.1f}]
  Mean: {np.mean(real_vs30):.1f}

KS Statistic: {ks:.4f}
"""

        plt.text(
            0.1,
            0.9,
            stats_text,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
        )

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"vs30_hist_{step}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print(f"[metrics] Vs30 histogram saved: vs30_hist_{step}.png")


def plot_loss_curves(loss_history: List[float], output_dir: str, step: int) -> None:
    """Plot training loss curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title(f"FFM Training Loss (Step {step})")
    plt.xlabel("Training Step")
    plt.ylabel("MSE Loss")
    plt.grid(True, alpha=0.3)
    plt.yscale("log")

    output_path = os.path.join(output_dir, f"loss_curve_step_{step}.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved loss curve to: {output_path}")


def plot_profile_comparison(
    real_profiles: np.ndarray,
    generated_profiles: np.ndarray,
    output_dir: str,
    step: int,
    max_profiles: int = 8,
) -> None:
    """Plot comparison between real and generated profiles."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot individual profiles
    ax1 = axes[0, 0]
    depth_axis = np.arange(real_profiles.shape[-1]) * 0.5  # 0.5m depth step

    for i in range(min(max_profiles, real_profiles.shape[0])):
        ax1.plot(
            real_profiles[i, 0, :], depth_axis, alpha=0.7, color="blue", linewidth=1
        )
    ax1.set_title("Real Profiles")
    ax1.set_xlabel("Vs (m/s)")
    ax1.set_ylabel("Depth (m)")
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    for i in range(min(max_profiles, generated_profiles.shape[0])):
        ax2.plot(
            generated_profiles[i, 0, :], depth_axis, alpha=0.7, color="red", linewidth=1
        )
    ax2.set_title("Generated Profiles")
    ax2.set_xlabel("Vs (m/s)")
    ax2.set_ylabel("Depth (m)")
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3)

    # Plot Vs30 distributions
    ax3 = axes[1, 0]
    real_vs30 = compute_vs30_distribution(real_profiles)
    gen_vs30 = compute_vs30_distribution(generated_profiles)

    ax3.hist(real_vs30, bins=30, alpha=0.7, label="Real", color="blue", density=True)
    ax3.hist(gen_vs30, bins=30, alpha=0.7, label="Generated", color="red", density=True)
    ax3.set_title("Vs30 Distribution")
    ax3.set_xlabel("Vs30 (m/s)")
    ax3.set_ylabel("Density")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot statistics comparison
    ax4 = axes[1, 1]
    stats_data = {
        "Mean": [real_profiles.mean(), generated_profiles.mean()],
        "Std": [real_profiles.std(), generated_profiles.std()],
        "Min": [real_profiles.min(), generated_profiles.min()],
        "Max": [real_profiles.max(), generated_profiles.max()],
    }

    x = np.arange(len(stats_data))
    width = 0.35

    ax4.bar(
        x - width / 2,
        [stats_data[k][0] for k in stats_data.keys()],
        width,
        label="Real",
        alpha=0.7,
        color="blue",
    )
    ax4.bar(
        x + width / 2,
        [stats_data[k][1] for k in stats_data.keys()],
        width,
        label="Generated",
        alpha=0.7,
        color="red",
    )

    ax4.set_title("Profile Statistics")
    ax4.set_xlabel("Statistic")
    ax4.set_ylabel("Value")
    ax4.set_xticks(x)
    ax4.set_xticklabels(stats_data.keys())
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"profile_comparison_step_{step}.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved profile comparison to: {output_path}")


def plot_generation_trajectory(
    trajectory: np.ndarray, output_dir: str, step: int
) -> None:
    """Plot the generation trajectory showing evolution from noise to final profile."""
    plt.figure(figsize=(12, 8))

    # trajectory shape: (Steps, Batch, 1, Points)
    # Plot the evolution of the first sample
    sample_trajectory = trajectory[:, 0, 0, :]

    plt.imshow(
        sample_trajectory,
        aspect="auto",
        origin="lower",
        extent=(0, sample_trajectory.shape[1], 0, sample_trajectory.shape[0]),
        cmap="viridis",
    )
    plt.title(f"Generation Trajectory (Step {step})")
    plt.xlabel("Profile Points")
    plt.ylabel("ODE Integration Step")
    plt.colorbar(label="Normalized Vs")

    output_path = os.path.join(output_dir, f"trajectory_step_{step}.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved trajectory plot to: {output_path}")


def sample_ffm(model, initial_noise, ode_steps, device):
    """Helper function to sample from FFM model."""
    model.eval()

    # Start with noise
    u = initial_noise.to(device)
    dt = 1.0 / ode_steps

    for i in range(ode_steps):
        # Current time
        t_val = i * dt
        t = torch.full((u.shape[0], 1), t_val).to(device)

        # Predict vector field
        v_pred = model(u, t)

        # Euler step: u_{t+dt} = u_t + v_pred * dt
        u = u + v_pred * dt

    return u


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")

    # Create mock data
    real_profiles = np.random.randn(10, 1, 128) * 100 + 300
    generated_profiles = np.random.randn(10, 1, 128) * 80 + 320

    # Test Vs30 computation
    vs30 = compute_vs30(real_profiles[0, 0, :])
    print(f"Sample Vs30: {vs30:.1f} m/s")

    # Test distribution computation
    vs30_dist = compute_vs30_distribution(real_profiles)
    print(f"Vs30 distribution: mean={vs30_dist.mean():.1f}, std={vs30_dist.std():.1f}")

    # Test KS statistic
    ks = ks_statistic(real_profiles.flatten(), generated_profiles.flatten())
    print(f"KS statistic: {ks:.3f}")

    print("Utility functions test completed!")
