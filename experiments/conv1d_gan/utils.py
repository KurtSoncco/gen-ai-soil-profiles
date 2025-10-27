from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def compute_real_vs30_and_density(parquet_path: str) -> tuple[np.ndarray, float]:
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


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
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


def log_vs30_metrics(
    G: torch.nn.Module,
    cfg,
    device: torch.device,
    step: int,
    out_dir: str,
    real_vs30: np.ndarray,
    avg_samples_per_meter: float,
    wandb_instance=None,
) -> None:
    if real_vs30.size == 0:
        print("[metrics] Skipping Vs30 metrics: real distribution unavailable.")
        return
    with torch.no_grad():
        z_eval = torch.randn(512, cfg.latent_dim, device=device)
        fake = G(z_eval).cpu().numpy()

    # Debug: Print sample statistics
    print(f"[debug] Generated samples shape: {fake.shape}")
    print(f"[debug] Generated samples range: [{np.min(fake):.3f}, {np.max(fake):.3f}]")
    print(f"[debug] Generated samples mean: {np.mean(fake):.3f}")
    print(f"[debug] Samples per meter: {avg_samples_per_meter}")

    # Check if generated samples are in reasonable range
    sample_range = np.max(fake) - np.min(fake)
    if sample_range < 10:  # Very small range indicates poor training
        print(
            f"[warning] Generated samples have very small range ({sample_range:.3f}). Generator may not be properly trained."
        )

    gen_vs30 = compute_generated_vs30(fake, avg_samples_per_meter)

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
    if wandb_instance is not None:
        try:
            wandb_instance.log(
                {
                    "step": step,
                    "metrics/generated_vs30_mean": np.mean(gen_vs30),
                    "metrics/generated_vs30_std": np.std(gen_vs30),
                    "metrics/generated_vs30_min": np.min(gen_vs30),
                    "metrics/generated_vs30_max": np.max(gen_vs30),
                    "metrics/generated_samples_mean": np.mean(fake),
                    "metrics/generated_samples_std": np.std(fake),
                    "metrics/generated_samples_min": np.min(fake),
                    "metrics/generated_samples_max": np.max(fake),
                    "metrics/vs30_ks_statistic": ks,
                    "metrics/sample_range": sample_range,
                }
            )

            # Log histograms to wandb
            wandb_instance.log(
                {
                    "histograms/vs30_distribution": wandb_instance.Histogram(gen_vs30),
                    "histograms/generated_samples": wandb_instance.Histogram(
                        fake.flatten()
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
        os.makedirs(out_dir, exist_ok=True)
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
        plt.hist(fake.flatten(), bins=50, alpha=0.7, color="green")
        plt.xlabel("Generated Sample Values")
        plt.ylabel("Frequency")
        plt.title(
            f"Generated Sample Distribution\nRange: [{np.min(fake):.3f}, {np.max(fake):.3f}]"
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
  Range: [{np.min(fake):.3f}, {np.max(fake):.3f}]
  Mean: {np.mean(fake):.3f}
  Std: {np.std(fake):.3f}

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
            os.path.join(out_dir, f"vs30_hist_{step}.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()

        print(f"[metrics] Vs30 histogram saved: vs30_hist_{step}.png")


def plot_loss_curves(loss_history: dict, out_dir: str, step: int) -> None:
    """
    Plot and save loss curves.

    Args:
        loss_history: Dictionary containing loss history
        out_dir: Output directory
        step: Current training step
    """
    if not loss_history["steps"]:
        return

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(
        loss_history["steps"],
        loss_history["loss_D"],
        "b-",
        label="Discriminator Loss",
        linewidth=1,
    )
    plt.plot(
        loss_history["steps"],
        loss_history["loss_G"],
        "r-",
        label="Generator Loss",
        linewidth=1,
    )
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    # Plot recent losses (last 50 steps) for better visibility
    recent_steps = loss_history["steps"][-50:]
    recent_loss_D = loss_history["loss_D"][-50:]
    recent_loss_G = loss_history["loss_G"][-50:]

    plt.plot(recent_steps, recent_loss_D, "b-", label="Discriminator Loss", linewidth=2)
    plt.plot(recent_steps, recent_loss_G, "r-", label="Generator Loss", linewidth=2)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Recent Losses (Last 50 Steps)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f"training_losses_step_{step}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


__all__ = [
    "compute_real_vs30_and_density",
    "compute_generated_vs30",
    "ks_statistic",
    "log_vs30_metrics",
    "plot_loss_curves",
]
