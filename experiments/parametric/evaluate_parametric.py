#!/usr/bin/env python3
"""
Parametric Profile Evaluation and Visualization

This script provides detailed evaluation and visualization of the parametric
profile modeling results, including comparison with real data and statistical analysis.
"""

import logging
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# These imports must come after sys.path.append
from experiments.parametric.src.generative_models import ParameterGenerator  # noqa: E402
from experiments.parametric.src.parametric_models import ParametricProfileFitter  # noqa: E402
from experiments.VAE.src.utils import calculate_vs30  # noqa: E402
from soilgen_ai.logging_config import setup_logging  # noqa: E402


class ParametricProfileEvaluator:
    """
    Comprehensive evaluator for parametric profile models.
    """

    def __init__(self, standard_depths):
        """
        Initialize the evaluator.

        Args:
            standard_depths: Array of depth values for the profiles
        """
        self.standard_depths = standard_depths
        self.layer_centers = (standard_depths[:-1] + standard_depths[1:]) / 2

    def calculate_vs30(self, vs_profile):
        """Calculate Vs30 for a profile."""
        return calculate_vs30(vs_profile, self.standard_depths)

    def calculate_vs100(self, vs_profile):
        """Calculate Vs100 for a profile."""
        # Find the layer closest to 100m depth
        target_depth = 100.0
        layer_idx = np.argmin(np.abs(self.layer_centers - target_depth))
        return vs_profile[layer_idx]

    def calculate_vs500(self, vs_profile):
        """Calculate Vs500 for a profile."""
        # Find the layer closest to 500m depth
        target_depth = 500.0
        layer_idx = np.argmin(np.abs(self.layer_centers - target_depth))
        return vs_profile[layer_idx]

    def calculate_profile_statistics(self, vs_profiles):
        """
        Calculate comprehensive statistics for a set of profiles.

        Args:
            vs_profiles: Array of shape (n_profiles, n_layers)

        Returns:
            Dictionary containing various statistics
        """
        stats_dict = {}

        # Basic statistics
        stats_dict["mean_vs"] = np.mean(vs_profiles, axis=0)
        stats_dict["std_vs"] = np.std(vs_profiles, axis=0)
        stats_dict["min_vs"] = np.min(vs_profiles, axis=0)
        stats_dict["max_vs"] = np.max(vs_profiles, axis=0)

        # Vs30, Vs100, Vs500 statistics
        vs30_values = [self.calculate_vs30(p) for p in vs_profiles]
        vs100_values = [self.calculate_vs100(p) for p in vs_profiles]
        vs500_values = [self.calculate_vs500(p) for p in vs_profiles]

        stats_dict["vs30_mean"] = np.mean(vs30_values)
        stats_dict["vs30_std"] = np.std(vs30_values)
        stats_dict["vs30_min"] = np.min(vs30_values)
        stats_dict["vs30_max"] = np.max(vs30_values)

        stats_dict["vs100_mean"] = np.mean(vs100_values)
        stats_dict["vs100_std"] = np.std(vs100_values)
        stats_dict["vs100_min"] = np.min(vs100_values)
        stats_dict["vs100_max"] = np.max(vs100_values)

        stats_dict["vs500_mean"] = np.mean(vs500_values)
        stats_dict["vs500_std"] = np.std(vs500_values)
        stats_dict["vs500_min"] = np.min(vs500_values)
        stats_dict["vs500_max"] = np.max(vs500_values)

        # Profile shape statistics
        stats_dict["gradient_mean"] = np.mean(np.gradient(vs_profiles, axis=1), axis=0)
        stats_dict["gradient_std"] = np.std(np.gradient(vs_profiles, axis=1), axis=0)

        return stats_dict

    def compare_profiles(self, real_profiles, generated_profiles, model_name):
        """
        Compare real and generated profiles.

        Args:
            real_profiles: Array of real profiles
            generated_profiles: Array of generated profiles
            model_name: Name of the model for logging

        Returns:
            Dictionary containing comparison metrics
        """
        logging.info(f"Comparing profiles for {model_name}...")

        # Calculate statistics for both sets
        real_stats = self.calculate_profile_statistics(real_profiles)
        gen_stats = self.calculate_profile_statistics(generated_profiles)

        # Calculate comparison metrics
        comparison = {}

        # Profile-level metrics
        comparison["profile_mse"] = mean_squared_error(
            real_stats["mean_vs"], gen_stats["mean_vs"]
        )
        comparison["profile_mae"] = mean_absolute_error(
            real_stats["mean_vs"], gen_stats["mean_vs"]
        )

        # Vs30 comparison
        real_vs30 = [self.calculate_vs30(p) for p in real_profiles]
        gen_vs30 = [self.calculate_vs30(p) for p in generated_profiles]

        comparison["vs30_mse"] = mean_squared_error(real_vs30, gen_vs30)
        comparison["vs30_mae"] = mean_absolute_error(real_vs30, gen_vs30)
        comparison["vs30_ks_statistic"] = stats.ks_2samp(real_vs30, gen_vs30).statistic
        comparison["vs30_ks_pvalue"] = stats.ks_2samp(real_vs30, gen_vs30).pvalue

        # Vs100 comparison
        real_vs100 = [self.calculate_vs100(p) for p in real_profiles]
        gen_vs100 = [self.calculate_vs100(p) for p in generated_profiles]

        comparison["vs100_mse"] = mean_squared_error(real_vs100, gen_vs100)
        comparison["vs100_mae"] = mean_absolute_error(real_vs100, gen_vs100)
        comparison["vs100_ks_statistic"] = stats.ks_2samp(
            real_vs100, gen_vs100
        ).statistic

        # Vs500 comparison
        real_vs500 = [self.calculate_vs500(p) for p in real_profiles]
        gen_vs500 = [self.calculate_vs500(p) for p in generated_profiles]

        comparison["vs500_mse"] = mean_squared_error(real_vs500, gen_vs500)
        comparison["vs500_mae"] = mean_absolute_error(real_vs500, gen_vs500)
        comparison["vs500_ks_statistic"] = stats.ks_2samp(
            real_vs500, gen_vs500
        ).statistic

        # Store statistics
        comparison["real_stats"] = real_stats
        comparison["gen_stats"] = gen_stats

        # Log results
        logging.info(f"  Profile MSE: {comparison['profile_mse']:.4f}")
        logging.info(f"  Profile MAE: {comparison['profile_mae']:.4f}")
        logging.info(f"  Vs30 MSE: {comparison['vs30_mse']:.4f}")
        logging.info(f"  Vs30 KS statistic: {comparison['vs30_ks_statistic']:.4f}")
        logging.info(f"  Vs100 MSE: {comparison['vs100_mse']:.4f}")
        logging.info(f"  Vs500 MSE: {comparison['vs500_mse']:.4f}")

        return comparison

    def plot_comprehensive_comparison(
        self, real_profiles, generated_profiles_dict, model_names, save_path
    ):
        """
        Create comprehensive comparison plots.

        Args:
            real_profiles: Array of real profiles
            generated_profiles_dict: Dictionary of generated profiles by model
            model_names: List of model names
            save_path: Path to save the plot
        """
        logging.info("Creating comprehensive comparison plots...")

        # Set up the plot
        plt.figure(figsize=(20, 16))

        # 1. Profile comparison (top row)
        for i, model_name in enumerate(model_names):
            ax = plt.subplot(4, len(model_names), i + 1)

            generated_profiles = generated_profiles_dict[model_name]

            # Plot real profiles
            for j in range(min(20, real_profiles.shape[0])):
                ax.plot(
                    real_profiles[j], self.layer_centers, "b-", alpha=0.3, linewidth=0.5
                )

            # Plot generated profiles
            for j in range(min(20, generated_profiles.shape[0])):
                ax.plot(
                    generated_profiles[j],
                    self.layer_centers,
                    "r-",
                    alpha=0.3,
                    linewidth=0.5,
                )

            ax.set_title(f"{model_name.title()} Profiles")
            ax.set_xlabel("Vs (m/s)")
            ax.set_ylabel("Depth (m)")
            ax.set_ylim(max(self.standard_depths), 0)
            ax.grid(True, alpha=0.3)

        # 2. Vs30 comparison (second row)
        for i, model_name in enumerate(model_names):
            ax = plt.subplot(4, len(model_names), len(model_names) + i + 1)

            generated_profiles = generated_profiles_dict[model_name]

            real_vs30 = [self.calculate_vs30(p) for p in real_profiles]
            gen_vs30 = [self.calculate_vs30(p) for p in generated_profiles]

            ax.hist(real_vs30, bins=30, alpha=0.7, label="Real", density=True)
            ax.hist(gen_vs30, bins=30, alpha=0.7, label="Generated", density=True)

            ax.set_title(f"{model_name.title()} Vs30 Distribution")
            ax.set_xlabel("Vs30 (m/s)")
            ax.set_ylabel("Density")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 3. Mean profile comparison (third row)
        for i, model_name in enumerate(model_names):
            ax = plt.subplot(4, len(model_names), 2 * len(model_names) + i + 1)

            generated_profiles = generated_profiles_dict[model_name]

            real_mean = np.mean(real_profiles, axis=0)
            real_std = np.std(real_profiles, axis=0)
            gen_mean = np.mean(generated_profiles, axis=0)
            gen_std = np.std(generated_profiles, axis=0)

            ax.plot(real_mean, self.layer_centers, "b-", label="Real Mean", linewidth=2)
            ax.fill_betweenx(
                self.layer_centers,
                real_mean - real_std,
                real_mean + real_std,
                alpha=0.3,
                color="blue",
            )

            ax.plot(
                gen_mean, self.layer_centers, "r-", label="Generated Mean", linewidth=2
            )
            ax.fill_betweenx(
                self.layer_centers,
                gen_mean - gen_std,
                gen_mean + gen_std,
                alpha=0.3,
                color="red",
            )

            ax.set_title(f"{model_name.title()} Mean Profiles")
            ax.set_xlabel("Vs (m/s)")
            ax.set_ylabel("Depth (m)")
            ax.set_ylim(max(self.standard_depths), 0)
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 4. Gradient comparison (fourth row)
        for i, model_name in enumerate(model_names):
            ax = plt.subplot(4, len(model_names), 3 * len(model_names) + i + 1)

            generated_profiles = generated_profiles_dict[model_name]

            real_gradients = np.gradient(real_profiles, axis=1)
            gen_gradients = np.gradient(generated_profiles, axis=1)

            real_grad_mean = np.mean(real_gradients, axis=0)
            gen_grad_mean = np.mean(gen_gradients, axis=0)

            ax.plot(
                real_grad_mean, self.layer_centers[:-1], "b-", label="Real", linewidth=2
            )
            ax.plot(
                gen_grad_mean,
                self.layer_centers[:-1],
                "r-",
                label="Generated",
                linewidth=2,
            )

            ax.set_title(f"{model_name.title()} Velocity Gradients")
            ax.set_xlabel("dV/dz (m/s/m)")
            ax.set_ylabel("Depth (m)")
            ax.set_ylim(max(self.standard_depths), 0)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logging.info(f"Comprehensive comparison plot saved to {save_path}")

    def plot_parameter_analysis(
        self, fitted_parameters_dict, generated_parameters_dict, fitters, save_path
    ):
        """
        Plot parameter analysis and comparison.

        Args:
            fitted_parameters_dict: Dictionary of fitted parameters by model
            generated_parameters_dict: Dictionary of generated parameters by model
            fitters: Dictionary of fitters by model
            save_path: Path to save the plot
        """
        logging.info("Creating parameter analysis plots...")

        n_models = len(fitted_parameters_dict)
        fig, axes = plt.subplots(2, n_models, figsize=(5 * n_models, 10))

        if n_models == 1:
            axes = axes.reshape(2, 1)

        for i, (model_name, fitted_params) in enumerate(fitted_parameters_dict.items()):
            generated_params = generated_parameters_dict[model_name]
            param_names = fitters[model_name].get_parameter_names()

            # Plot fitted parameters
            ax_fitted = axes[0, i]
            for j, name in enumerate(param_names):
                ax_fitted.hist(fitted_params[:, j], bins=30, alpha=0.7, label=name)
            ax_fitted.set_title(f"{model_name.title()} - Fitted Parameters")
            ax_fitted.set_xlabel("Parameter Value")
            ax_fitted.set_ylabel("Frequency")
            ax_fitted.legend()
            ax_fitted.grid(True, alpha=0.3)

            # Plot generated parameters
            ax_generated = axes[1, i]
            for j, name in enumerate(param_names):
                ax_generated.hist(
                    generated_params[:, j], bins=30, alpha=0.7, label=name
                )
            ax_generated.set_title(f"{model_name.title()} - Generated Parameters")
            ax_generated.set_xlabel("Parameter Value")
            ax_generated.set_ylabel("Frequency")
            ax_generated.legend()
            ax_generated.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logging.info(f"Parameter analysis plot saved to {save_path}")


def load_results(results_dir):
    """Load results from the parametric experiment."""
    logging.info("Loading parametric experiment results...")

    results_dir = Path(results_dir)

    # Load generated profiles
    generated_parameters = {}

    for model_file in results_dir.glob("*.pkl"):
        if "generator" in model_file.name:
            model_name = model_file.name.split("_")[0]
            gen_type = model_file.name.split("_")[1]

            # Load generator
            generator = ParameterGenerator(
                model_type="gmm" if gen_type == "gmm" else "mlp"
            )
            generator.load(str(model_file))

            # Generate samples for evaluation
            n_samples = 1000
            generated_params = generator.generate(n_samples)

            if model_name not in generated_parameters:
                generated_parameters[model_name] = {}
            generated_parameters[model_name][gen_type] = generated_params

    return generated_parameters


def main():
    """Main evaluation function."""
    # Setup logging
    setup_logging()
    logging.info("Starting Parametric Profile Evaluation")

    # Configuration
    results_dir = Path(__file__).parent / "results"
    output_dir = Path(__file__).parent / "evaluation"
    output_dir.mkdir(exist_ok=True)

    # Load the original data for comparison
    logging.info("Loading original data for comparison...")

    # This would normally load the same data as the training script
    # For now, we'll create a placeholder
    # In practice, you would load the same profiles_dict and convert to vs_profiles

    # Load results
    generated_parameters = load_results(results_dir)

    # Create evaluator
    standard_depths = np.linspace(0, 2000, 101)  # Same as training
    evaluator = ParametricProfileEvaluator(standard_depths)

    # For demonstration, create some sample data
    # In practice, you would load the real data here
    logging.info(
        "Note: This is a demonstration script. In practice, load real data here."
    )

    # Create sample real profiles (replace with actual data loading)
    n_real_profiles = 500
    n_layers = 100
    real_profiles = np.random.uniform(200, 2000, (n_real_profiles, n_layers))

    # Generate profiles from parameters (this would use the actual fitters)
    logging.info("Generating profiles from parameters for evaluation...")

    # This is a simplified version - in practice you would load the actual fitters
    generated_profiles_dict = {}
    for model_name, model_params in generated_parameters.items():
        # Use the first generator type for simplicity
        gen_type = list(model_params.keys())[0]
        params = model_params[gen_type]

        # Create a simple fitter for demonstration
        fitter = ParametricProfileFitter(standard_depths, model_name)
        generated_profiles = fitter.generate_profiles(params)
        generated_profiles_dict[model_name] = generated_profiles

    # Perform comprehensive evaluation
    logging.info("Performing comprehensive evaluation...")

    comparisons = {}
    for model_name, generated_profiles in generated_profiles_dict.items():
        comparison = evaluator.compare_profiles(
            real_profiles, generated_profiles, model_name
        )
        comparisons[model_name] = comparison

    # Create comprehensive plots
    model_names = list(generated_profiles_dict.keys())
    evaluator.plot_comprehensive_comparison(
        real_profiles,
        generated_profiles_dict,
        model_names,
        output_dir / "comprehensive_comparison.png",
    )

    # Create parameter analysis plots
    evaluator.plot_parameter_analysis(
        {},
        generated_parameters,
        {},  # Empty dicts for demonstration
        output_dir / "parameter_analysis.png",
    )

    # Save evaluation results
    results_file = output_dir / "evaluation_results.pkl"
    with open(results_file, "wb") as f:
        pickle.dump(comparisons, f)

    logging.info(f"Evaluation results saved to {results_file}")
    logging.info("Parametric Profile Evaluation completed successfully!")


if __name__ == "__main__":
    main()
