import numpy as np
from scipy import stats
from typing import List, Dict
import matplotlib.pyplot as plt
from .utils import calculate_vs30


def compute_weighted_metrics(
    real_profiles: np.ndarray,
    generated_profiles: np.ndarray,
    depths: np.ndarray,
    weights: np.ndarray,
) -> Dict[str, float]:
    """Compute weighted metrics for profile comparison."""

    # Ensure both arrays have the same number of samples
    min_samples = min(real_profiles.shape[0], generated_profiles.shape[0])
    real_profiles = real_profiles[:min_samples]
    generated_profiles = generated_profiles[:min_samples]

    # Layer-wise weighted MSE
    layer_errors = np.mean((real_profiles - generated_profiles) ** 2, axis=0)
    weighted_mse = np.sum(weights * layer_errors)

    # Layer-wise weighted MAE
    layer_mae = np.mean(np.abs(real_profiles - generated_profiles), axis=0)
    weighted_mae = np.sum(weights * layer_mae)

    # Total variation (smoothness) comparison
    real_tv = np.mean(np.sum(np.abs(np.diff(real_profiles, axis=1)), axis=1))
    gen_tv = np.mean(np.sum(np.abs(np.diff(generated_profiles, axis=1)), axis=1))

    return {
        "weighted_mse": weighted_mse,
        "weighted_mae": weighted_mae,
        "real_tv": real_tv,
        "gen_tv": gen_tv,
        "tv_ratio": gen_tv / real_tv if real_tv > 0 else float("inf"),
    }


def compute_vs30_metrics(
    real_vs30: List[float], generated_vs30: List[float]
) -> Dict[str, float]:
    """Compute Vs30 distribution comparison metrics."""

    # Remove NaN values
    real_vs30_clean = [v for v in real_vs30 if not np.isnan(v) and v > 0]
    gen_vs30_clean = [v for v in generated_vs30 if not np.isnan(v) and v > 0]

    if len(real_vs30_clean) == 0 or len(gen_vs30_clean) == 0:
        return {
            "ks_statistic": 1.0,
            "ks_pvalue": 0.0,
            "mean_ratio": 1.0,
            "std_ratio": 1.0,
        }

    # Kolmogorov-Smirnov test
    ks_stat, ks_pvalue = stats.ks_2samp(real_vs30_clean, gen_vs30_clean)

    # Distribution statistics
    real_mean, real_std = np.mean(real_vs30_clean), np.std(real_vs30_clean)
    gen_mean, gen_std = np.mean(gen_vs30_clean), np.std(gen_vs30_clean)

    return {
        "ks_statistic": ks_stat,
        "ks_pvalue": ks_pvalue,
        "mean_ratio": gen_mean / real_mean if real_mean > 0 else 1.0,
        "std_ratio": gen_std / real_std if real_std > 0 else 1.0,
        "real_mean": real_mean,
        "real_std": real_std,
        "gen_mean": gen_mean,
        "gen_std": gen_std,
    }


def compute_depth_wise_metrics(
    real_profiles: np.ndarray, generated_profiles: np.ndarray, depths: np.ndarray
) -> Dict[str, np.ndarray]:
    """Compute metrics at different depth intervals."""

    # Calculate layer mid-depths
    layer_mid_depths = (depths[:-1] + depths[1:]) / 2

    # Define depth intervals
    depth_intervals = [(0, 10), (10, 30), (30, 100), (100, 200), (200, 500)]

    interval_metrics = {}

    for start_depth, end_depth in depth_intervals:
        # Find layers in this depth interval
        mask = (layer_mid_depths >= start_depth) & (layer_mid_depths < end_depth)

        if not np.any(mask):
            continue

        real_interval = real_profiles[:, mask]
        gen_interval = generated_profiles[:, mask]

        # Compute statistics for this interval
        real_mean = np.mean(real_interval, axis=0)
        gen_mean = np.mean(gen_interval, axis=0)

        real_std = np.std(real_interval, axis=0)
        gen_std = np.std(gen_interval, axis=0)

        interval_metrics[f"{start_depth}_{end_depth}"] = {
            "real_mean": real_mean,
            "gen_mean": gen_mean,
            "real_std": real_std,
            "gen_std": gen_std,
            "mse": np.mean((real_interval - gen_interval) ** 2, axis=0),
            "mae": np.mean(np.abs(real_interval - gen_interval), axis=0),
        }

    return interval_metrics


def plot_comprehensive_evaluation(
    real_profiles: np.ndarray,
    generated_profiles: np.ndarray,
    depths: np.ndarray,
    vs30_metrics: Dict[str, float],
    save_path: str = None,
):
    """Create comprehensive evaluation plots."""

    # Ensure both arrays have the same number of samples
    min_samples = min(real_profiles.shape[0], generated_profiles.shape[0])
    real_profiles = real_profiles[:min_samples]
    generated_profiles = generated_profiles[:min_samples]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Vs30 distribution comparison
    real_vs30 = [calculate_vs30(p, depths) for p in real_profiles]
    gen_vs30 = [calculate_vs30(p, depths) for p in generated_profiles]

    real_vs30_clean = [v for v in real_vs30 if not np.isnan(v) and v > 0]
    gen_vs30_clean = [v for v in gen_vs30 if not np.isnan(v) and v > 0]

    axes[0, 0].hist(real_vs30_clean, bins=30, alpha=0.7, label="Real", density=True)
    axes[0, 0].hist(gen_vs30_clean, bins=30, alpha=0.7, label="Generated", density=True)
    axes[0, 0].set_xlabel("Vs30 (m/s)")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].set_title(f"Vs30 Distribution\nKS={vs30_metrics['ks_statistic']:.3f}")
    axes[0, 0].legend()

    # 2. Profile examples
    for i in range(min(5, real_profiles.shape[0])):
        axes[0, 1].plot(real_profiles[i], depths[:-1], "b-", alpha=0.7, linewidth=1)
        axes[0, 1].plot(
            generated_profiles[i], depths[:-1], "r--", alpha=0.7, linewidth=1
        )
    axes[0, 1].set_xlabel("Vs (m/s)")
    axes[0, 1].set_ylabel("Depth (m)")
    axes[0, 1].set_title("Profile Examples")
    axes[0, 1].invert_yaxis()
    axes[0, 1].grid(True)

    # 3. Mean and std comparison by depth
    layer_mid_depths = (depths[:-1] + depths[1:]) / 2
    real_mean = np.mean(real_profiles, axis=0)
    gen_mean = np.mean(generated_profiles, axis=0)
    real_std = np.std(real_profiles, axis=0)
    gen_std = np.std(generated_profiles, axis=0)

    axes[0, 2].plot(real_mean, layer_mid_depths, "b-", label="Real Mean")
    axes[0, 2].plot(gen_mean, layer_mid_depths, "r--", label="Generated Mean")
    axes[0, 2].fill_betweenx(
        layer_mid_depths,
        real_mean - real_std,
        real_mean + real_std,
        alpha=0.3,
        color="blue",
        label="Real ±1σ",
    )
    axes[0, 2].fill_betweenx(
        layer_mid_depths,
        gen_mean - gen_std,
        gen_mean + gen_std,
        alpha=0.3,
        color="red",
        label="Generated ±1σ",
    )
    axes[0, 2].set_xlabel("Vs (m/s)")
    axes[0, 2].set_ylabel("Depth (m)")
    axes[0, 2].set_title("Mean ± Std by Depth")
    axes[0, 2].invert_yaxis()
    axes[0, 2].legend()
    axes[0, 2].grid(True)

    # 4. Error by depth
    mse_by_depth = np.mean((real_profiles - generated_profiles) ** 2, axis=0)
    mae_by_depth = np.mean(np.abs(real_profiles - generated_profiles), axis=0)

    axes[1, 0].plot(mse_by_depth, layer_mid_depths, "g-", label="MSE")
    axes[1, 0].set_xlabel("MSE")
    axes[1, 0].set_ylabel("Depth (m)")
    axes[1, 0].set_title("Reconstruction Error by Depth")
    axes[1, 0].invert_yaxis()
    axes[1, 0].grid(True)

    # 5. Error distribution
    all_errors = (real_profiles - generated_profiles).flatten()
    axes[1, 1].hist(all_errors, bins=50, alpha=0.7)
    axes[1, 1].set_xlabel("Reconstruction Error (m/s)")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Error Distribution")
    axes[1, 1].grid(True)

    # 6. Summary statistics
    axes[1, 2].axis("off")
    summary_text = f"""
    Summary Statistics:
    
    Vs30 Metrics:
    KS Statistic: {vs30_metrics["ks_statistic"]:.3f}
    Mean Ratio: {vs30_metrics["mean_ratio"]:.3f}
    Std Ratio: {vs30_metrics["std_ratio"]:.3f}
    
    Overall Metrics:
    Mean MSE: {np.mean(mse_by_depth):.2f}
    Mean MAE: {np.mean(mae_by_depth):.2f}
    Max Error: {np.max(np.abs(all_errors)):.2f}
    
    Profile Count:
    Real: {len(real_vs30_clean)}
    Generated: {len(gen_vs30_clean)}
    """
    axes[1, 2].text(
        0.1,
        0.9,
        summary_text,
        transform=axes[1, 2].transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Comprehensive evaluation plot saved to: {save_path}")

    plt.close()  # Close instead of showing
