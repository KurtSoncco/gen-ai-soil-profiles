# -*- coding: utf-8 -*-
"""
Generates synthetic Two-Way Travel Time (tts) profiles using a stochastic
power-law model with NumPyro as the Bayesian backend.
"""

import warnings
from pathlib import Path
from typing import Dict, Optional

import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
from numpy.random import Generator, default_rng
from numpy.typing import NDArray
from numpyro.infer import MCMC, NUTS
from scipy.stats import lognorm, rv_continuous

from soilgen_ai.logging_config import setup_logging

# --- Configuration & Setup ---

# IMPROVEMENT: Set numpyro to use multiple chains by default for better diagnostics
numpyro.set_host_device_count(4)

logger = setup_logging()

sns.set_palette("colorblind")

CONFIG = {
    "rng_seed": 42,
    "num_realizations": 2000,
    "max_depth": 2000.0,
    "num_points": 101,
    "data_path": Path(__file__).cwd() / "data" / "vspdb_tts_profiles.parquet",
    "priors": {
        "a": lognorm(s=0.8, scale=0.01),
        "b": lognorm(s=0.2, scale=1.1),
        "c": lognorm(s=0.5, scale=0.05),
    },
}


# --- Utility Functions ---
def _is_profile_suitable(df: pd.DataFrame, r2_threshold: float = 0.9) -> bool:
    """Checks if a profile is suitable for fitting using a quick linear check."""
    # This function is well-implemented and remains unchanged.
    try:
        from sklearn.linear_model import LinearRegression
    except ImportError:
        warnings.warn("scikit-learn not installed. Skipping profile suitability check.")
        return True

    depth = df["depth"].to_numpy().reshape(-1, 1)
    tts = df["tts"].to_numpy()

    # Use log-log space to approximate the power law with a linear model
    log_depth = np.log(depth + 1e-6)
    model = LinearRegression()
    try:
        model.fit(log_depth, tts)
        return bool(model.score(log_depth, tts) > r2_threshold)
    except ValueError:
        return False


def load_data(file_path: Path, rng: Generator) -> Dict[str, pd.DataFrame]:
    """Tries to load real data; otherwise, generates synthetic data."""
    # This function is also well-implemented and remains unchanged.
    try:
        if not file_path.exists():
            raise FileNotFoundError
        logger.info(f"Loading real data from '{file_path}'...")
        df = pq.read_table(file_path).to_pandas()
        logger.info("Filtering profiles for suitability...")
        grouped = df.groupby("velocity_metadata_id")
        suitable_groups = grouped.filter(
            lambda g: len(g) > 20 and _is_profile_suitable(g)
        )
        return {
            name: group[["depth", "tts"]].reset_index(drop=True)
            for name, group in suitable_groups.groupby("velocity_metadata_id")
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
    """Manages fitting and generation of TTS profiles with NumPyro."""

    def __init__(
        self, max_depth: float, num_points: int, num_realizations: int, seed: int
    ):
        self.max_depth = max_depth
        self.num_points = num_points
        self.num_realizations = num_realizations
        self.rng = default_rng(seed)
        self.jax_rng_key = jax.random.PRNGKey(seed)
        # IMPROVEMENT: Use JAX arrays where computations will happen on device (GPU/TPU)
        self.depths = jnp.linspace(0, self.max_depth, self.num_points)
        self.tts_profiles: Optional[NDArray] = None
        self.posterior_samples: Optional[Dict[str, NDArray]] = None
        self.mcmc_run: Optional[MCMC] = None

    @staticmethod
    def _numpyro_model(depths: jnp.ndarray, tts: Optional[jnp.ndarray] = None) -> None:
        """Defines the Bayesian power-law model in NumPyro."""
        param_a = numpyro.sample("a", dist.HalfNormal(0.1))
        param_b = numpyro.sample("b", dist.Normal(1.0, 0.5))
        param_c = numpyro.sample("c", dist.HalfNormal(0.1))
        sigma = numpyro.sample("sigma", dist.HalfNormal(0.05))

        mu = param_a * (depths + 1e-6) ** param_b + param_c
        numpyro.sample("tts_likelihood", dist.Normal(mu, sigma), obs=tts)

    def fit_numpyro_posterior(self, profile_df: pd.DataFrame) -> None:
        """Fits the Bayesian model using NumPyro's NUTS sampler."""
        logger.info("\n--- Fitting Bayesian Model with NumPyro ---")
        # IMPROVEMENT: Convert data to JAX arrays for GPU/TPU compatibility
        depths_data = jnp.array(profile_df["depth"].values)
        tts_data = jnp.array(profile_df["tts"].values)

        self.jax_rng_key, fit_key = jax.random.split(self.jax_rng_key)

        # IMPROVEMENT: Let NumPyro run chains in parallel by default.
        kernel = NUTS(self._numpyro_model)
        self.mcmc_run = MCMC(
            kernel, num_warmup=1000, num_samples=2000, progress_bar=True
        )

        try:
            logger.info("Running MCMC sampler...")
            self.mcmc_run.run(fit_key, depths=depths_data, tts=tts_data)
            # IMPROVEMENT: Device array to host array conversion for broader compatibility.
            self.posterior_samples = {
                k: np.asarray(v) for k, v in self.mcmc_run.get_samples().items()
            }
            logger.info("MCMC sampling complete.")
            self.mcmc_run.print_summary()
        except Exception as e:
            logger.error(f"Error during MCMC sampling: {e}")
            self.mcmc_run = None

    def _generate_from_params(
        self, a_params: NDArray, b_params: NDArray, c_params: NDArray
    ) -> None:
        """Vectorized core logic to generate profiles from parameter arrays."""
        # Use NumPy for generation as it's a host-side operation
        depths_np = np.asarray(self.depths)
        tts_profiles = (
            a_params[:, np.newaxis] * (depths_np + 1e-6) ** b_params[:, np.newaxis]
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
            a = np.asarray(
                priors["a"].rvs(size=self.num_realizations, random_state=self.rng)
            )
            b = np.asarray(
                priors["b"].rvs(size=self.num_realizations, random_state=self.rng)
            )
            c = np.asarray(
                priors["c"].rvs(size=self.num_realizations, random_state=self.rng)
            )
            self._generate_from_params(a, b, c)

        elif source == "bayesian":
            if self.posterior_samples is None:
                raise RuntimeError("Bayesian model must be fit before generating.")

            # The posterior samples are already flattened across chains
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
            logger.warning("No profiles generated to plot.")
            return

        logger.info("Plotting results...")
        fig, ax = plt.subplots(figsize=(10, 8))
        num_to_plot = min(self.num_realizations, 50)
        indices = self.rng.choice(self.num_realizations, num_to_plot, replace=False)
        ax.plot(
            self.tts_profiles[indices].T,
            np.asarray(self.depths),
            alpha=0.2,
            color="gray",
            zorder=1,
        )

        tts_mean = self.tts_profiles.mean(axis=0)
        tts_std = self.tts_profiles.std(axis=0)
        ax.plot(
            tts_mean,
            np.asarray(self.depths),
            color="red",
            lw=3,
            label="Mean $tts$ Profile",
            zorder=2,
        )
        ax.fill_betweenx(
            np.asarray(self.depths),
            tts_mean - tts_std,
            tts_mean + tts_std,
            color="red",
            alpha=0.2,
            label=r"Mean $\pm$ 1 Std Dev",
            zorder=0,
        )
        ax.set(
            xlabel="Two-Way Travel Time, $tts$ (s)",
            ylabel="Depth (m)",
            title=f"Stochastic TTS Profiles ({self.num_realizations} Realizations)",
        )
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
    """Main execution function to demonstrate both prior and bayesian modes."""
    # The NumPy Generator is still useful for host-side randomness
    rng = default_rng(CONFIG["rng_seed"])
    model = StochasticTTSModel(
        max_depth=CONFIG["max_depth"],
        num_points=CONFIG["num_points"],
        num_realizations=CONFIG["num_realizations"],
        seed=CONFIG["rng_seed"],
    )

    # --- Run Prior Mode ---
    logger.info("\n\n--- Running in Prior Mode ---")
    model.generate_profiles(source="prior", priors=CONFIG["priors"])
    model.plot_results()

    # --- Run Bayesian Mode ---
    logger.info("\n\n--- Running in Bayesian Mode ---")
    profiles = load_data(CONFIG["data_path"], rng)
    if not profiles:
        logger.warning("No suitable data profiles found for Bayesian mode. Exiting.")
        return

    # Fit on the first suitable profile
    profile_to_fit = next(iter(profiles.values()))
    model.fit_numpyro_posterior(profile_to_fit)

    if model.mcmc_run:
        model.generate_profiles(source="bayesian")
        model.plot_results()
    else:
        logger.warning("Model fitting failed. Cannot proceed with Bayesian generation.")


if __name__ == "__main__":
    main()
