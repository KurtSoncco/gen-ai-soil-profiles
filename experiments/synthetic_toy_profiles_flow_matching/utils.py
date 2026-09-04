"""Utility functions for Synthetic Toy Profiles Flow Matching.

This module provides:
- Sampling functions for generating profiles
- Metrics computation
- Plotting and visualization utilities
"""

from __future__ import annotations

import os
from typing import List

import matplotlib

# Set non-interactive backend before importing pyplot (for headless environments/WSL)
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


def sample_ffm(model, initial_noise, ode_steps, device, ode_solver: str = "rk4"):
    """
    Sample from trained flow matching model using ODE integration.
    Works with breakpoints (3 values) instead of full profiles.

    Args:
        model: Trained flow matching model
        initial_noise: Initial noise tensor (batch_size, 3) for breakpoints
        ode_steps: Number of ODE integration steps
        device: PyTorch device
        ode_solver: ODE solver method - "euler" or "rk4" (Runge-Kutta 4)

    Returns:
        Generated breakpoints with same shape as initial_noise (batch_size, 3)
    """
    model.eval()

    # Start with noise at t=0 - shape should be (batch_size, 3)
    u = initial_noise.to(device)  # (batch_size, 3)

    dt = 1.0 / ode_steps

    if ode_solver == "euler":
        # Euler method: simple first-order integration
        for i in range(ode_steps):
            t_val = i * dt
            t = torch.full((u.shape[0], 1), t_val).to(device)
            v = model(u, t)  # Predict vector field (batch_size, 3)
            u = u + v * dt  # Update state

    elif ode_solver == "rk4":
        # RK4 (Runge-Kutta 4th order): more accurate 4th-order integration
        for i in range(ode_steps):
            # Current time
            t_val = i * dt
            t = torch.full((u.shape[0], 1), t_val).to(device)

            # Predict vector field at current state
            k1 = model(u, t) * dt  # (batch_size, 3)

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
    else:
        raise ValueError(f"Unknown ODE solver: {ode_solver}. Must be 'euler' or 'rk4'")

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
    plt.title(f"Synthetic Profiles FFM Training Loss (Step {step})")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)

    output_path = os.path.join(output_dir, f"loss_curve_{step}.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved loss curve to {output_path}")


def plot_profile_comparison(
    real_profiles: np.ndarray,
    generated_profiles: np.ndarray,
    output_dir: str,
    step: int,
    max_profiles: int = 16,
) -> None:
    """Plot side-by-side comparison of real and generated profiles."""
    os.makedirs(output_dir, exist_ok=True)

    n_profiles = min(max_profiles, len(real_profiles), len(generated_profiles))
    n_cols = 4
    n_rows = (n_profiles + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for i in range(n_profiles):
        ax = axes[i]
        real_profile = real_profiles[i, 0, :]
        gen_profile = generated_profiles[i, 0, :]

        # Create depth array starting at 0
        depth = np.arange(len(real_profile))

        # Plot travel time vs depth (x: travel time, y: depth)
        ax.plot(real_profile, depth, label="Real", alpha=0.7, linewidth=1.5)
        ax.plot(gen_profile, depth, label="Generated", alpha=0.7, linewidth=1.5)
        ax.set_xlabel("Travel Time")
        ax.set_ylabel("Depth")
        ax.set_title(f"Profile {i + 1}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Invert y-axis so depth increases downward (starting at 0 at top)
        ax.invert_yaxis()
        ax.set_ylim(len(real_profile) - 1, 0)  # Ensure depth starts at 0 at top

        # Move xlabel and ticks to top
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()

    # Hide unused subplots
    for i in range(n_profiles, len(axes)):
        axes[i].axis("off")

    plt.suptitle(f"Profile Comparison @ Step {step}", fontsize=16, y=0.998)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"profile_comparison_{step}.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved profile comparison to {output_path}")


def compute_layer_statistics(
    profiles: np.ndarray, jump_threshold: float | None = None
) -> dict:
    """
    Compute statistics for the 2-layer synthetic profiles.

    For each profile, detect layer 1 (v1) and layer 2 (v2) values.

    Args:
        profiles: Array of shape (N, 1, L) where N is number of profiles, L is length
        jump_threshold: Threshold for detecting significant jumps. If None, auto-detect
            based on data range (10% of range for denormalized data, 0.1 for normalized data)

    Returns:
        Dictionary with layer statistics
    """
    layer1_values = []
    layer2_values = []
    boundary_depths = []
    jump_sizes = []  # Track the magnitude of jumps at boundaries

    # Auto-detect jump threshold if not provided
    if jump_threshold is None:
        # Estimate data range from first profile
        sample_profile = profiles[0, 0, :]
        data_range = np.max(sample_profile) - np.min(sample_profile)
        if data_range < 10:
            # Likely normalized data (0-1 range)
            jump_threshold = 0.1
        else:
            # Likely denormalized data (travel_time values)
            jump_threshold = max(
                50.0, data_range * 0.1
            )  # 10% of range or 50, whichever is larger

    for i in range(profiles.shape[0]):
        profile = profiles[i, 0, :]

        # Find the boundary - look for where the value changes significantly
        # Use a better method: find all significant jumps, then use the largest one
        diff = np.abs(np.diff(profile))
        significant_jumps = diff > jump_threshold

        if np.any(significant_jumps):
            # Find the largest significant jump
            jump_indices = np.where(significant_jumps)[0]
            jump_sizes_all = diff[jump_indices]
            largest_jump_idx = jump_indices[np.argmax(jump_sizes_all)]
            boundary_idx = largest_jump_idx + 1
            largest_jump_size = jump_sizes_all[np.argmax(jump_sizes_all)]

            layer1_values.append(float(np.mean(profile[:boundary_idx])))
            layer2_values.append(float(np.mean(profile[boundary_idx:])))
            boundary_depths.append(boundary_idx)
            jump_sizes.append(float(largest_jump_size))
        else:
            # Fallback: assume middle split
            mid = len(profile) // 2
            layer1_values.append(float(np.mean(profile[:mid])))
            layer2_values.append(float(np.mean(profile[mid:])))
            boundary_depths.append(mid)
            jump_sizes.append(0.0)  # No significant jump found

    return {
        "layer1_mean": np.mean(layer1_values),
        "layer1_std": np.std(layer1_values),
        "layer2_mean": np.mean(layer2_values),
        "layer2_std": np.std(layer2_values),
        "boundary_mean": np.mean(boundary_depths),
        "boundary_std": np.std(boundary_depths),
        "jump_size_mean": np.mean(jump_sizes),
        "jump_size_std": np.std(jump_sizes),
        "layer1_values": np.array(layer1_values),
        "layer2_values": np.array(layer2_values),
        "boundary_depths": np.array(boundary_depths),
        "jump_sizes": np.array(jump_sizes),
    }


def compute_breakpoint_statistics(breakpoints: np.ndarray) -> dict:
    """
    Compute statistics for breakpoints.

    Args:
        breakpoints: Array of shape (N, 3) with [depth1, tts1, tts_end] for each profile

    Returns:
        Dictionary with breakpoint statistics
    """
    if breakpoints.shape[1] != 3:
        raise ValueError(f"Expected breakpoints shape (N, 3), got {breakpoints.shape}")

    # Point 1: origin (0, 0) - implicit, no distribution
    # Point 2: breakpoint (depth1, tts1)
    # Point 3: end point (500, tts_end)

    depth1_values = breakpoints[:, 0]
    tts1_values = breakpoints[:, 1]
    tts_end_values = breakpoints[:, 2]

    return {
        "depth1_mean": float(np.mean(depth1_values)),
        "depth1_std": float(np.std(depth1_values)),
        "depth1_min": float(np.min(depth1_values)),
        "depth1_max": float(np.max(depth1_values)),
        "tts1_mean": float(np.mean(tts1_values)),
        "tts1_std": float(np.std(tts1_values)),
        "tts1_min": float(np.min(tts1_values)),
        "tts1_max": float(np.max(tts1_values)),
        "tts_end_mean": float(np.mean(tts_end_values)),
        "tts_end_std": float(np.std(tts_end_values)),
        "tts_end_min": float(np.min(tts_end_values)),
        "tts_end_max": float(np.max(tts_end_values)),
        "depth1_values": depth1_values,
        "tts1_values": tts1_values,
        "tts_end_values": tts_end_values,
    }


def log_profiles_metrics(model, cfg, device, step, output_dir, dataset, wandb=None):
    """Log metrics and create plots during training for breakpoints."""
    # Import reconstruct_profile from data module
    try:
        from .data import reconstruct_profile
    except ImportError:
        from data import reconstruct_profile

    # Generate samples for evaluation (breakpoints)
    num_eval_samples = min(256, len(dataset))
    z_eval = torch.randn(num_eval_samples, 3).to(device)

    with torch.no_grad():
        model.eval()
        generated_breakpoints = sample_ffm(
            model, z_eval, cfg.ode_steps, device, cfg.ode_solver
        )
        model.train()

    # Denormalize samples (breakpoints)
    generated_breakpoints_denorm = dataset.denormalize_batch(generated_breakpoints)
    generated_breakpoints_np = generated_breakpoints_denorm.cpu().numpy()

    # Load real data for comparison (breakpoints)
    real_breakpoints_np = []
    for i in range(num_eval_samples):
        real_breakpoints_np.append(dataset[i].cpu().numpy())
    real_breakpoints_np = np.stack(real_breakpoints_np, axis=0)

    # Denormalize real data
    real_breakpoints_np_normalized = torch.from_numpy(real_breakpoints_np)
    real_breakpoints_np = (
        dataset.denormalize_batch(real_breakpoints_np_normalized).cpu().numpy()
    )

    print(f"[metrics] step={step}")

    # Compute breakpoint statistics
    real_stats = compute_breakpoint_statistics(real_breakpoints_np)
    gen_stats = compute_breakpoint_statistics(generated_breakpoints_np)

    print("  Point 1 (origin): (0, 0) - implicit")
    print("  Point 2 (breakpoint):")
    print(
        f"    depth1: Real={real_stats['depth1_mean']:.2f}±{real_stats['depth1_std']:.2f}, "
        f"Gen={gen_stats['depth1_mean']:.2f}±{gen_stats['depth1_std']:.2f}"
    )
    print(
        f"    tts1: Real={real_stats['tts1_mean']:.4f}±{real_stats['tts1_std']:.4f}, "
        f"Gen={gen_stats['tts1_mean']:.4f}±{gen_stats['tts1_std']:.4f}"
    )
    print("  Point 3 (end):")
    print(
        f"    tts_end: Real={real_stats['tts_end_mean']:.4f}±{real_stats['tts_end_std']:.4f}, "
        f"Gen={gen_stats['tts_end_mean']:.4f}±{gen_stats['tts_end_std']:.4f}"
    )

    # Compute KS statistics for distributions
    ks_depth1 = ks_statistic(real_stats["depth1_values"], gen_stats["depth1_values"])
    ks_tts1 = ks_statistic(real_stats["tts1_values"], gen_stats["tts1_values"])
    ks_tts_end = ks_statistic(real_stats["tts_end_values"], gen_stats["tts_end_values"])

    print("  KS Statistics:")
    print(f"    depth1: {ks_depth1:.4f}")
    print(f"    tts1: {ks_tts1:.4f}")
    print(f"    tts_end: {ks_tts_end:.4f}")

    # Log to wandb if available
    if wandb is not None:
        try:
            wandb.log(
                {
                    "step": step,
                    "metrics/point2_depth1_mean": float(gen_stats["depth1_mean"]),
                    "metrics/point2_depth1_std": float(gen_stats["depth1_std"]),
                    "metrics/point2_tts1_mean": float(gen_stats["tts1_mean"]),
                    "metrics/point2_tts1_std": float(gen_stats["tts1_std"]),
                    "metrics/point3_tts_end_mean": float(gen_stats["tts_end_mean"]),
                    "metrics/point3_tts_end_std": float(gen_stats["tts_end_std"]),
                    "metrics/ks_depth1": float(ks_depth1),
                    "metrics/ks_tts1": float(ks_tts1),
                    "metrics/ks_tts_end": float(ks_tts_end),
                }
            )
        except Exception as e:
            print(f"[info] wandb logging failed: {e}")

    # Reconstruct profiles for visualization
    if step % cfg.checkpoint_every == 0 or step == cfg.num_steps - 1:
        # Reconstruct profiles from breakpoints (already denormalized, so apply_expm1=False)
        real_profiles = np.array(
            [
                reconstruct_profile(bp, dataset.max_depth, apply_expm1=False)
                for bp in real_breakpoints_np
            ]
        )
        gen_profiles = np.array(
            [
                reconstruct_profile(bp, dataset.max_depth, apply_expm1=False)
                for bp in generated_breakpoints_np
            ]
        )

        # Reshape for plotting (add channel dimension)
        real_profiles = real_profiles[:, np.newaxis, :]  # (N, 1, 500)
        gen_profiles = gen_profiles[:, np.newaxis, :]  # (N, 1, 500)

        plot_profile_comparison(real_profiles, gen_profiles, cfg.plots_dir, step)


def tts_to_vs(tts_profile: np.ndarray, dz: float = 1.0) -> np.ndarray:
    """
    Convert travel time profile to Vs (shear wave velocity) profile.

    For a piecewise linear travel time profile:
    - Travel time increment per depth step: d(tts) = slope * dz
    - Vs = dz / d(tts) = dz / (slope * dz) = 1 / slope

    For piecewise linear: vs = depth_range / tts_range

    Args:
        tts_profile: Array of travel time values (length,)
        dz: Depth increment in meters (default 1.0)

    Returns:
        Array of Vs values (length,)
    """
    vs_profile = np.zeros_like(tts_profile)

    # Compute finite differences (slope of travel time)
    # d_tts[i] = tts[i] - tts[i-1] for i > 0, and d_tts[0] = tts[0] - 0
    d_tts = np.diff(tts_profile, prepend=0.0)

    # Avoid division by zero - use a small epsilon
    d_tts = np.where(np.abs(d_tts) < 1e-9, 1e-9, d_tts)

    # Vs = dz / d(tts) where d(tts) is the travel time increment per depth step
    # For piecewise linear: d(tts) = slope * dz, so vs = dz / (slope * dz) = 1 / slope
    # But we have d_tts = tts[i] - tts[i-1], so:
    # vs = dz / d_tts
    vs_profile = dz / d_tts

    # Clip to reasonable Vs range (e.g., 100-5000 m/s)
    vs_profile = np.clip(vs_profile, 100.0, 5000.0)

    return vs_profile.astype(np.float32)


def plot_tts_vs_samples(
    breakpoints: np.ndarray,
    output_path: str,
    max_depth: int = 500,
    dz: float = 1.0,
    num_samples: int = 100,
    title: str = "Generated Samples",
) -> None:
    """
    Plot travel time and corresponding Vs profiles for generated samples.

    Args:
        breakpoints: Array of shape (N, 3) with breakpoints [depth1, tts1, tts_end]
        output_path: Path to save the plot
        max_depth: Maximum depth (default 500)
        dz: Depth increment in meters (default 1.0)
        num_samples: Number of samples to plot (default 100)
        title: Plot title
    """
    import matplotlib.pyplot as plt

    try:
        from .data import reconstruct_profile
    except ImportError:
        from data import reconstruct_profile

    # Limit number of samples
    num_samples = min(num_samples, len(breakpoints))
    breakpoints = breakpoints[:num_samples]

    # Create figure with 1 row, 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    # Depth array (multiply index by dz)
    depths = np.arange(max_depth) * dz

    # Plot all samples
    for i, bp in enumerate(breakpoints):
        # Reconstruct travel time profile
        # Breakpoints are expected to be in original space (after expm1), so use apply_expm1=False
        tts_profile = reconstruct_profile(bp, max_depth, apply_expm1=False)

        # Convert to Vs profile
        vs_profile = tts_to_vs(tts_profile, dz)

        # Plot travel time (left)
        axes[0].plot(tts_profile, depths, alpha=0.3, linewidth=0.5, color="blue")

        # Plot Vs (right)
        axes[1].plot(vs_profile, depths, alpha=0.3, linewidth=0.5, color="red")

    # Add median and geometric mean profiles for reference
    if len(breakpoints) > 1:
        # Reconstruct all profiles
        all_tts = np.array(
            [
                reconstruct_profile(bp, max_depth, apply_expm1=False)
                for bp in breakpoints
            ]
        )
        all_vs = np.array([tts_to_vs(tts, dz) for tts in all_tts])

        # Compute median
        median_tts = np.median(all_tts, axis=0)
        median_vs = np.median(all_vs, axis=0)

        # Compute geometric mean (exp of mean of log)
        # Use log to avoid issues with zeros/negative values
        log_tts = np.log(np.maximum(all_tts, 1e-9))  # Avoid log(0)
        log_vs = np.log(np.maximum(all_vs, 1e-9))  # Avoid log(0)
        geomean_tts = np.exp(np.mean(log_tts, axis=0))
        geomean_vs = np.exp(np.mean(log_vs, axis=0))

        # Plot median and geometric mean with thicker lines
        axes[0].plot(
            median_tts,
            depths,
            alpha=0.9,
            linewidth=2,
            color="darkblue",
            label="Median",
            linestyle="-",
        )
        axes[0].plot(
            geomean_tts,
            depths,
            alpha=0.9,
            linewidth=2,
            color="navy",
            label="GeoMean",
            linestyle="--",
        )
        axes[1].plot(
            median_vs,
            depths,
            alpha=0.9,
            linewidth=2,
            color="darkred",
            label="Median",
            linestyle="-",
        )
        axes[1].plot(
            geomean_vs,
            depths,
            alpha=0.9,
            linewidth=2,
            color="darkorange",
            label="GeoMean",
            linestyle="--",
        )

        axes[0].legend()
        axes[1].legend()

    # Configure travel time plot (left)
    axes[0].set_xlabel("Travel Time (s)", fontsize=12)
    axes[0].set_ylabel("Depth (m)", fontsize=12)
    axes[0].set_title("Travel Time Profiles", fontsize=14, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].invert_yaxis()  # Depth increases downward
    axes[0].set_ylim(depths[-1], depths[0])

    # Configure Vs plot (right)
    axes[1].set_xlabel("Vs (m/s)", fontsize=12)
    axes[1].set_ylabel("Depth (m)", fontsize=12)
    axes[1].set_title("Vs Profiles", fontsize=14, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].invert_yaxis()  # Depth increases downward
    axes[1].set_ylim(depths[-1], depths[0])

    # Overall title
    fig.suptitle(
        f"{title} ({num_samples} samples, dz={dz}m)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved TTS/Vs comparison plot to {output_path}")


if __name__ == "__main__":
    print("Testing utils...")

    # Test sample_ffm with dummy model
    class DummyModel(torch.nn.Module):
        def forward(self, x, t):
            return torch.zeros_like(x)

    model = DummyModel()
    noise = torch.randn(4, 1, 500)
    samples = sample_ffm(model, noise, 10, "cpu")
    print(f"Sample shape: {samples.shape}")

    # Test layer statistics
    profiles = np.random.rand(10, 1, 500) * 1000 + 500
    stats = compute_layer_statistics(profiles)
    print(f"Computed stats: {len(stats)} metrics")

    # Test tts_to_vs
    tts_test = np.linspace(0, 1, 500)
    vs_test = tts_to_vs(tts_test, dz=1.0)
    print(
        f"TTS to Vs conversion test: tts shape={tts_test.shape}, vs shape={vs_test.shape}"
    )

    print("All utils tests passed!")
