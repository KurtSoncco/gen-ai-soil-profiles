import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import wandb


def plot_stair_profiles(profiles, depths, title, max_depth, vs_max):
    """Plot multiple Vs profiles as stair plots."""
    plt.figure(figsize=(10, 8))
    for profile in profiles:
        # Create stair plot points
        y_stair = np.repeat(depths, 2)[1:-1]
        x_stair = np.repeat(profile, 2)
        plt.plot(x_stair, y_stair)

    plt.title(title)
    plt.xlabel("Shear Wave Velocity (Vs) [m/s]")
    plt.ylabel("Depth [m]")
    plt.ylim(max_depth, 0)
    plt.xlim(0, vs_max)
    plt.grid(True)
    wandb.log({title: wandb.Image(plt)})
    plt.savefig(
        Path(__file__).parent / "generated_vs_profiles.png", bbox_inches="tight"
    )
    plt.close()


def tts_to_Vs(d_tts, dz):
    """
    Convert a profile of travel time differences (d_tts) to a layered Vs profile.
    """
    # Add a small epsilon to avoid division by zero
    Vs = dz / (d_tts + 1e-9)
    return Vs


def Vs30_calc(depth, vs):
    """
    Calculate Vs30 from a single velocity profile with non-uniform layers.
    `depth` represents the top of each layer.
    Formula is 30 / sum(di/vsi) for layers until the depth sums up to 30.
    """
    # Convert to numpy arrays to handle pandas Series indexing and ensure correct sorting
    depth = np.array(depth)
    vs = np.array(vs)
    sorted_indices = np.argsort(depth)
    depth = depth[sorted_indices]
    vs = vs[sorted_indices]

    if not len(depth) or not len(vs) or depth[0] > 30:
        return np.nan

    # If the profile is deeper than 30m, we might not need all layers
    if depth[-1] < 30:
        # If the last layer starts below 30m but the profile extends to 30m,
        # we need to handle it. This case is complex with non-uniform layers.
        # For simplicity, we return NaN if the last measured layer top is less than 30.
        # A more sophisticated approach would be to extrapolate, but that can be unreliable.
        return np.nan

    time_sum = 0
    for i in range(len(vs)):
        top_depth = depth[i]
        # The bottom of the last layer is effectively infinite, but we only care up to 30m.
        bottom_depth = depth[i + 1] if i + 1 < len(depth) else 30.0

        if top_depth >= 30:
            break

        layer_thickness = min(bottom_depth, 30.0) - top_depth
        if vs[i] > 0:
            time_sum += layer_thickness / vs[i]
        else:  # If vs is zero or negative, we can't calculate a meaningful Vs30
            return np.nan

    return 30.0 / time_sum if time_sum > 0 else np.nan


def calculate_vs30(vs_profile, depths):
    """
    Calculate Vs30 from a standardized Vs profile with uniform layers.
    `depths` are the layer boundaries (N+1 points for N layers).
    `vs_profile` has N points, one for each layer.
    """
    if depths[-1] < 30:
        return np.nan  # Not enough depth

    total_time = 0
    # `depths` has N+1 points for N layers. `vs_profile` has N points.
    for i in range(len(vs_profile)):
        depth_start = depths[i]
        depth_end = depths[i + 1]

        if depth_start >= 30:
            break

        # Layer thickness is the portion of this layer that is above 30m
        layer_thickness = min(depth_end, 30.0) - depth_start
        if vs_profile[i] > 0:
            total_time += layer_thickness / vs_profile[i]

    if total_time <= 0:
        return np.nan

    return 30.0 / total_time


def evaluate_generation(real_vs, generated_vs, standard_depths):
    """
    Evaluates the quality of generated Vs profiles against real profiles.
    - Compares Vs30 distributions.
    - Compares Vs distributions at different depth intervals.
    """
    # 1. Compare Vs30
    real_vs30 = [calculate_vs30(p, standard_depths) for p in real_vs]
    real_vs30 = [v for v in real_vs30 if not np.isnan(v) and v > 0]

    generated_vs30 = [calculate_vs30(p, standard_depths) for p in generated_vs]
    generated_vs30 = [v for v in generated_vs30 if not np.isnan(v) and v > 0]

    plt.figure(figsize=(10, 5))
    sns.histplot(real_vs30, color="blue", label="Real Vs30", kde=True, stat="density")
    sns.histplot(
        generated_vs30, color="red", label="Generated Vs30", kde=True, stat="density"
    )
    plt.title("Comparison of Vs30 Distributions")
    plt.xlabel("Vs30 [m/s]")
    plt.legend()
    wandb.log({"Comparison of Vs30 Distributions": wandb.Image(plt)})
    plt.savefig(Path(__file__).parent / "vs30_comparison.png", bbox_inches="tight")
    plt.close()

    logging.info(
        f"Real Vs30 stats: mean={np.mean(real_vs30):.2f}, std={np.std(real_vs30):.2f}"
    )
    logging.info(
        f"Generated Vs30 stats: mean={np.mean(generated_vs30):.2f}, std={np.std(generated_vs30):.2f}"
    )

    wandb.log(
        {
            "eval_real_vs30_mean": np.mean(real_vs30),
            "eval_real_vs30_std": np.std(real_vs30),
            "eval_generated_vs30_mean": np.mean(generated_vs30),
            "eval_generated_vs30_std": np.std(generated_vs30),
        }
    )

    # 2. Compare Vs distributions at different depths
    depth_intervals = [(0, 10), (10, 30), (30, 100)]
    max_vs = np.percentile(real_vs, 99)

    # Calculate layer mid-depths for correct masking
    layer_mid_depths = (standard_depths[:-1] + standard_depths[1:]) / 2

    # Create a long-form dataframe for easier plotting with seaborn
    real_vs_long = pd.DataFrame(real_vs).melt(var_name="layer_index", value_name="vs")
    real_vs_long["depth"] = real_vs_long["layer_index"].apply(
        lambda i: layer_mid_depths[i]
    )
    real_vs_long["type"] = "Real"

    gen_vs_long = pd.DataFrame(generated_vs).melt(
        var_name="layer_index", value_name="vs"
    )
    gen_vs_long["depth"] = gen_vs_long["layer_index"].apply(
        lambda i: layer_mid_depths[i]
    )
    gen_vs_long["type"] = "Generated"

    combined_df = pd.concat([real_vs_long, gen_vs_long], ignore_index=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for i, (start_depth, end_depth) in enumerate(depth_intervals):
        # Filter data for the current depth interval
        data_in_interval = combined_df[
            (combined_df["depth"] >= start_depth) & (combined_df["depth"] < end_depth)
        ]

        if data_in_interval.empty:
            continue

        sns.histplot(
            data=data_in_interval,
            x="vs",
            hue="type",
            ax=axes[i],
            kde=True,
            stat="density",
            common_norm=False,
        )
        axes[i].set_title(f"Vs dist. at {start_depth}-{end_depth} m")
        axes[i].set_xlabel("Vs [m/s]")
        axes[i].set_xlim(0, max_vs)
        axes[i].legend()

    plt.tight_layout()
    wandb.log({"Vs Distribution at Depths": wandb.Image(fig)})
    plt.savefig(Path(__file__).parent / "vs_dist_comparison.png", bbox_inches="tight")
    plt.close()
