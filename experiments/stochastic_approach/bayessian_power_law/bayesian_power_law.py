import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import arviz as az
import jax
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
import seaborn as sns
from numpy.random import Generator, default_rng
from numpy.typing import NDArray
from numpyro.infer import MCMC, NUTS
from scipy.optimize import curve_fit
from scipy.stats import lognorm, rv_continuous

from soilgen_ai.logging_config import setup_logging

# --- Configuration & Setup ---
logger = setup_logging()
sns.set_palette("colorblind")


# --- MODIFIED: Load data-driven priors from the JSON file ---
def load_priors_from_json(path: Path) -> Dict[str, rv_continuous | Any]:
    """Loads prior distribution parameters from a JSON file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Prior file not found at {path}. Please run calculate_priors.py first."
        )
    with open(path, "r") as f:
        params = json.load(f)

    return {name: lognorm(s=p["s"], scale=p["scale"]) for name, p in params.items()}


CONFIG = {
    "rng_seed": 42,
    "num_realizations": 2000,
    "max_depth": 2000.0,
    "num_points": 201,
    "data_path": Path(__file__).cwd() / "data" / "vspdb_tts_profiles.parquet",
    "priors_path": Path(__file__).resolve().parent / "priors.json",
}
# Initialize priors dynamically
try:
    CONFIG["priors"] = load_priors_from_json(CONFIG["priors_path"])
    logger.info("Successfully loaded data-driven priors from priors.json.")
except FileNotFoundError as e:
    logger.error(e)
    exit()

# --- Utility Functions ---


def power_law_model(depth: NDArray, a: float, b: float, c: float) -> NDArray:
    """The power-law function: tts = a * depth^b + c."""
    return a * (depth**b) + c


def load_data(file_path: Path, rng: Generator) -> Dict[str, pd.DataFrame]:
    """Loads real data, or generates synthetic data if not found."""
    try:
        import pyarrow.parquet as pq

        if not file_path.exists():
            raise FileNotFoundError("Data file not found.")
        logger.info(f"Loading real data from '{file_path}'...")
        df = pq.read_table(file_path).to_pandas()
        grouped = df.groupby("velocity_metadata_id")
        # REMOVED: The rigid _is_profile_suitable check is no longer needed.
        return {
            name: group[["depth", "tts"]].reset_index(drop=True)
            for name, group in grouped
            if len(group) > 20
        }
    except (ImportError, FileNotFoundError):
        logger.warning(
            f"Could not load data from '{file_path}'. Generating synthetic data."
        )
        data_dict = {}
        for i in range(50):
            num_points = rng.integers(20, 100)
            max_depth = rng.uniform(1000, 7000)
            depth = np.sort(rng.uniform(0, max_depth, num_points))
            tts = np.cumsum(rng.uniform(0.1, 0.5, num_points))
            data_dict[f"synthetic_{i}"] = pd.DataFrame({"depth": depth, "tts": tts})
        return data_dict


# --- Core Modeling Class ---
class StochasticTTSModel:
    """
    A class to manage fitting and generation of TTS profiles with NumPyro.
    The class supports both prior-based and Bayesian posterior-based generation.
    """

    def __init__(
        self, max_depth: float, num_points: int, num_realizations: int, seed: int
    ):
        self.max_depth = max_depth
        self.num_points = num_points
        self.num_realizations = num_realizations
        self.rng = default_rng(seed)
        self.depths = np.linspace(0, self.max_depth, self.num_points)
        self.tts_profiles: Optional[NDArray] = None
        self.posterior_samples: Optional[Dict[str, NDArray]] = None
        self.mcmc_run: Optional[MCMC] = None

    def _numpyro_model(self, depths: NDArray, tts: Optional[NDArray] = None):
        """Defines the Bayesian power-law model in NumPyro."""
        param_a = numpyro.sample("a", dist.HalfNormal(0.1))
        param_b = numpyro.sample("b", dist.Normal(1.0, 0.5))
        param_c = numpyro.sample("c", dist.HalfNormal(0.1))
        sigma = numpyro.sample("sigma", dist.HalfNormal(0.05))

        mu = param_a * (depths + 1e-6) ** param_b + param_c
        numpyro.sample("tts_likelihood", dist.Normal(mu, sigma), obs=tts)

    def fit_numpyro_posterior(self, profile_df: pd.DataFrame) -> None:
        """Fits the Bayesian model to a profile using NumPyro's MCMC."""
        logger.info("\n--- Fitting Bayesian Model with NumPyro ---")
        depths_data = profile_df["depth"].to_numpy()
        tts_data = profile_df["tts"].to_numpy()

        rng_key = jax.random.PRNGKey(self.rng.integers(2**32 - 1))
        kernel = NUTS(self._numpyro_model)
        self.mcmc_run = MCMC(
            kernel,
            num_warmup=1000,
            num_samples=2000,
            num_chains=1,
            progress_bar=True,
        )

        try:
            logger.info(
                f"Running MCMC sampler on profile with {len(depths_data)} points..."
            )
            self.mcmc_run.run(rng_key, depths=depths_data, tts=tts_data)
            self.posterior_samples = self.mcmc_run.get_samples()
            logger.info("MCMC sampling complete.")
        except Exception as e:
            logger.error(f"Error during MCMC sampling: {e}")
            self.mcmc_run = None

    def _generate_from_params(
        self, a_params: NDArray, b_params: NDArray, c_params: NDArray
    ) -> None:
        """Vectorized core logic to generate profiles from parameter arrays."""
        tts_profiles = (
            a_params[:, np.newaxis] * (self.depths + 1e-6) ** b_params[:, np.newaxis]
            + c_params[:, np.newaxis]
        )
        self.tts_profiles = np.maximum.accumulate(tts_profiles, axis=1)

    def generate_profiles(
        self,
        source: str = "prior",
        priors: Optional[Dict[str, rv_continuous]] = None,
    ) -> None:
        """Generates profiles from either predefined priors or a fitted posterior."""
        logger.info(f"\n--- Generating Profiles from '{source.title()}' Source ---")
        if source == "prior":
            if not priors:
                raise ValueError("Priors must be provided for 'prior' source.")
            a = np.atleast_1d(
                priors["a"].rvs(size=self.num_realizations, random_state=self.rng)
            )
            b = np.atleast_1d(
                priors["b"].rvs(size=self.num_realizations, random_state=self.rng)
            )
            c = np.atleast_1d(
                priors["c"].rvs(size=self.num_realizations, random_state=self.rng)
            )
            self._generate_from_params(a, b, c)

        elif source == "bayesian":
            if self.posterior_samples is None:
                raise RuntimeError("Bayesian model must be fit before generating.")
            num_posterior_samples = self.posterior_samples["a"].shape[0]
            indices = self.rng.choice(
                num_posterior_samples, size=self.num_realizations, replace=True
            )
            a = self.posterior_samples["a"][indices]
            b = self.posterior_samples["b"][indices]
            c = self.posterior_samples["c"][indices]
            self._generate_from_params(a, b, c)
        else:
            raise ValueError(f"Unknown source: '{source}'.")
        logger.info("Generation complete!")

    def plot_results(self) -> None:
        """Creates and displays plots for the generated profiles."""
        if self.tts_profiles is None:
            return
        logger.info("Plotting results...")
        fig, ax = plt.subplots(figsize=(10, 8))
        num_to_plot = min(self.num_realizations, 50)
        indices = self.rng.choice(self.num_realizations, num_to_plot, replace=False)
        ax.plot(
            self.tts_profiles[indices].T,
            self.depths,
            alpha=0.2,
            color="gray",
            zorder=1,
        )
        tts_mean = self.tts_profiles.mean(axis=0)
        tts_std = self.tts_profiles.std(axis=0)
        ax.plot(
            tts_mean,
            self.depths,
            color="red",
            lw=3,
            label="Mean $tts$ Profile",
            zorder=2,
        )
        ax.fill_betweenx(
            self.depths,
            tts_mean - tts_std,
            tts_mean + tts_std,
            color="red",
            alpha=0.2,
            label=r"Mean $\pm$ 1 Std Dev",
            zorder=0,
        )
        ax.set_xlabel("Two-Way Travel Time, $tts$ (s)")
        ax.set_ylabel("Depth (m)")
        ax.set_title(f"Stochastic TTS Profiles ({self.num_realizations} Realizations)")
        ax.invert_yaxis()
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.show()

        if self.mcmc_run:
            logger.info("Displaying posterior parameter distributions...")
            inference_data = az.from_numpyro(self.mcmc_run)
            az.plot_posterior(inference_data, var_names=["a", "b", "c", "sigma"])
            plt.tight_layout()
            plt.show()


# --- Main Execution ---


def main():
    """Main execution function with command-line argument parsing."""
    parser = argparse.ArgumentParser(description="Generate synthetic TTS profiles.")
    parser.add_argument("mode", choices=["prior", "bayesian"], help="Generation mode.")
    args = parser.parse_args()

    rng = default_rng(CONFIG["rng_seed"])
    model = StochasticTTSModel(
        max_depth=CONFIG["max_depth"],
        num_points=CONFIG["num_points"],
        num_realizations=CONFIG["num_realizations"],
        seed=CONFIG["rng_seed"],
    )

    if args.mode == "prior":
        model.generate_profiles(source="prior", priors=CONFIG["priors"])
        model.plot_results()
    elif args.mode == "bayesian":
        profiles = load_data(CONFIG["data_path"], rng)
        if not profiles:
            logger.warning("No suitable data profiles found. Exiting.")
            return

        # --- NEW: Profile Extension Logic ---
        profile_to_fit = next(iter(profiles.values()))
        logger.info(
            f"Selected profile with {len(profile_to_fit)} points and max depth of {profile_to_fit['depth'].max():.1f}m."
        )

        # 1. Fit the selected profile to get its specific power-law curve for extrapolation
        try:
            popt, _ = curve_fit(
                power_law_model, profile_to_fit["depth"], profile_to_fit["tts"]
            )
        except RuntimeError:
            logger.warning(
                "Could not fit the selected profile for extrapolation. Exiting."
            )
            return

        # 2. Extend the profile to the fixed max depth
        max_real_depth = profile_to_fit["depth"].max()
        if max_real_depth < CONFIG["max_depth"]:
            logger.info(
                f"Extending profile from {max_real_depth:.1f}m to {CONFIG['max_depth']}m..."
            )
            # Generate new depths from the end of the real data to the target depth
            num_ext_points = int(
                CONFIG["num_points"] / 4
            )  # Add a reasonable number of new points
            extended_depths = np.linspace(
                max_real_depth, CONFIG["max_depth"], num_ext_points
            )

            # Calculate tts for these new depths using the fitted curve
            extended_tts = power_law_model(extended_depths, *popt)

            # Combine original and extended data
            full_depth = np.concatenate([profile_to_fit["depth"], extended_depths])
            full_tts = np.concatenate([profile_to_fit["tts"], extended_tts])

            # Create a new DataFrame for fitting
            extended_profile = pd.DataFrame({"depth": full_depth, "tts": full_tts})
        else:
            extended_profile = profile_to_fit

        # 3. Fit the Bayesian model on the (potentially extended) profile
        model.fit_numpyro_posterior(extended_profile)

        if model.mcmc_run:
            model.generate_profiles(source="bayesian")
            model.plot_results()
        else:
            logger.warning("Model fitting failed. Cannot proceed with generation.")


if __name__ == "__main__":
    main()
