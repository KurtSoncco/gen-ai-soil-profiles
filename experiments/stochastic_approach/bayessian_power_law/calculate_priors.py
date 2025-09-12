# calculate_priors.py
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from scipy.stats import lognorm

from soilgen_ai.logging_config import setup_logging

sns.set_palette("colorblind")
logger = setup_logging()

# --- Configuration ---
DATA_PATH = Path(__file__).cwd() / "data" / "vspdb_tts_profiles.parquet"
OUTPUT_PATH = Path(__file__).resolve().parent / "priors.json"
logger.info(f"Data path set to: {DATA_PATH}")
logger.info(f"Output path set to: {OUTPUT_PATH}")


# --- Core Functions ---
def power_law_model(depth: NDArray, a: float, b: float, c: float) -> NDArray:
    """The power-law function we want to fit: tts = a * depth^b + c."""
    return a * (depth**b) + c


def load_all_data(file_path: Path) -> Dict[str, pd.DataFrame]:
    """Loads and provides a basic filter on the profile data."""
    try:
        import pyarrow.parquet as pq

        if not file_path.exists():
            raise FileNotFoundError("Data file not found.")
        df = pq.read_table(file_path).to_pandas()
        grouped = df.groupby("velocity_metadata_id")
        return {
            name: group[["depth", "tts"]].reset_index(drop=True)
            for name, group in grouped
            if len(group) > 20  # Basic filter for sufficient data points
        }
    except (ImportError, FileNotFoundError) as e:
        logger.warning(f"Error loading data: {e}")
        return {}


def plot_priors_and_data(
    profiles: Dict[str, pd.DataFrame],
    prior_distributions: Dict[str, Dict[str, float]],
    num_samples: int = 200,
):
    """
    Plots TTS vs. depth, including real data, and samples from the prior distribution.

    Args:
        profiles: Dictionary of real data profiles.
        prior_distributions: Calculated prior distribution parameters for 'a', 'b', 'c'.
        num_samples: Number of profiles to sample from the prior distribution.
    """
    all_depths = np.linspace(0, 2000, 200)
    sampled_tts_curves = []

    # Sample from the prior distributions
    for _ in range(num_samples):
        a = lognorm.rvs(
            s=prior_distributions["a"]["s"], scale=prior_distributions["a"]["scale"]
        )
        b = lognorm.rvs(
            s=prior_distributions["b"]["s"], scale=prior_distributions["b"]["scale"]
        )
        c = lognorm.rvs(
            s=prior_distributions["c"]["s"], scale=prior_distributions["c"]["scale"]
        )
        sampled_tts_curves.append(power_law_model(all_depths, a, b, c))

    sampled_tts_curves = np.array(sampled_tts_curves)
    median_tts = np.median(sampled_tts_curves, axis=0)
    percentile_2_5 = np.percentile(sampled_tts_curves, 2.5, axis=0)
    percentile_97_5 = np.percentile(sampled_tts_curves, 97.5, axis=0)

    plt.figure(figsize=(12, 8))

    # Plot the median and 95% CI of the sampled priors
    plt.plot(
        median_tts, all_depths, color="blue", label="Median of Priors", linewidth=2
    )
    plt.fill_betweenx(
        all_depths,
        percentile_2_5,
        percentile_97_5,
        color="lightblue",
        alpha=0.5,
        label="95% Confidence Interval",
    )

    # Overlay the real data
    for name, profile_df in profiles.items():
        plt.scatter(
            profile_df["tts"],
            profile_df["depth"],
            alpha=0.1,
            color="red",
            s=10,
            label="Real Data" if name == list(profiles.keys())[0] else "",
        )

    plt.ylabel("Depth (m)")
    plt.xlabel("Two-Way Travel Time (s)")
    plt.title("Prior Distribution Samples vs. Real Data")
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    # Xlabel and ticks to the top
    plt.gca().xaxis.set_label_position("top")
    plt.gca().xaxis.tick_top()
    plt.gca().invert_yaxis()  # Depth increases downwards
    plt.savefig(
        Path(__file__).cwd()
        / "outputs"
        / "figures/bayesian_approach"
        / "priors_vs_real_data.png",
        dpi=300,
    )
    plt.show()


# -- Main Execution ---
def main():
    """
    Fits a power-law model to all profiles, calculates the distribution of the
    fitted parameters, and saves them to a JSON file.
    """
    logger.info(f"Loading data from {DATA_PATH}...")
    profiles = load_all_data(DATA_PATH)
    if not profiles:
        logger.warning("No suitable profiles found to analyze.")
        return

    fit_params = {"a": [], "b": [], "c": []}
    logger.info(f"Analyzing {len(profiles)} profiles to determine priors...")

    for name, profile_df in profiles.items():
        depth_data = profile_df["depth"].to_numpy()
        tts_data = profile_df["tts"].to_numpy()

        try:
            # Use curve_fit to find the best a, b, c for this profile
            # Bounds are crucial to guide the optimizer to physical solutions
            popt, _ = curve_fit(
                power_law_model,
                depth_data,
                tts_data,
                p0=[0.01, 1.1, 0.05],  # Initial guess
                bounds=([0, 0.1, 0], [1.0, 3.0, 1.0]),  # (a, b, c) bounds
                maxfev=5000,
            )
            fit_params["a"].append(popt[0])
            fit_params["b"].append(popt[1])
            fit_params["c"].append(popt[2])
        except RuntimeError:
            # This can happen if a profile is too noisy for the optimizer
            print(f"  - Could not fit profile {name}, skipping.")
            continue

    if not fit_params["a"]:
        logger.warning("Could not successfully fit any profiles.")
        return

    logger.info(f"\nSuccessfully fit {len(fit_params['a'])} profiles.")
    logger.info("Calculating log-normal distribution parameters for priors...")

    prior_distributions = {}
    for param_name, values in fit_params.items():
        # Use lognorm.fit to find the shape(s), loc, and scale parameters
        # that best describe the distribution of the fitted values.
        s, loc, scale = lognorm.fit(values, floc=0)  # floc=0 for standard lognorm
        prior_distributions[param_name] = {"s": s, "scale": scale}
        logger.info(f"  - Parameter '{param_name}': s={s:.4f}, scale={scale:.4f}")

    # Plotting the distributions for visual verification
    ## Plot of histograms and fitted PDFs
    logger.info("\nPlotting fitted prior distributions...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x_vals = np.linspace(0.001, 3, 1000)

    for ax, (param_name, values) in zip(axes, fit_params.items()):
        sns.histplot(values, bins=30, kde=False, stat="density", ax=ax)
        s = prior_distributions[param_name]["s"]
        scale = prior_distributions[param_name]["scale"]
        pdf_vals = lognorm.pdf(x_vals, s, scale=scale)
        ax.plot(x_vals, pdf_vals, "r-", lw=2)
        ax.set_title(f"Fitted '{param_name}' Distribution")
        ax.set_xlabel(param_name)
        ax.set_ylabel("Density")

    plt.tight_layout()
    plt.savefig(
        Path(__file__).cwd()
        / "outputs"
        / "figures/bayesian_approach"
        / "fitted_priors.png",
        dpi=300,
    )
    plt.show()

    # Plot the prior distribution against the real data
    logger.info("\nPlotting prior distribution against real data...")
    plot_priors_and_data(profiles, prior_distributions)

    logger.info(f"\nSaving calculated priors to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(prior_distributions, f, indent=4)

    logger.info("Done.")


if __name__ == "__main__":
    logger.info("Starting prior calculation...")
    main()
