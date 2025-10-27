#!/usr/bin/env python3
"""
FFM Evaluation and Visualization

This script provides comprehensive evaluation and visualization of the FFM
results, including comparison with real data, statistical analysis, and plots.
"""

import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from . import config as cfg_mod
    from . import models as models_mod
    from . import utils as utils_mod
    from .data import create_dataloader, VsProfilesDataset
except Exception:  # fallback when running as script
    import config as cfg_mod
    import models as models_mod
    import utils as utils_mod

    from data import create_dataloader, VsProfilesDataset  # type: ignore


class FFMEvaluator:
    """
    Comprehensive evaluator for FFM models.
    """

    def __init__(self, config, device: torch.device):
        """
        Initialize the evaluator.

        Args:
            config: Configuration object
            device: PyTorch device
        """
        self.config = config
        self.device = device
        self.results_dir = Path(config.out_dir)
        self.plots_dir = self.results_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)

        # Load real data for comparison
        self.real_data, self.real_vs30, self.avg_samples_per_meter, self.dataset = (
            self._load_real_data()
        )

    def _load_real_data(self) -> Tuple[np.ndarray, np.ndarray, float, VsProfilesDataset]:
        """Load real data for comparison."""
        logging.info("Loading real data for comparison...")

        # Load real profiles
        loader, max_length, dataset = create_dataloader(
            self.config.batch_size, self.config.num_workers, shuffle=False
        )
        real_profiles = []

        for real in loader:
            # Denormalize real data for proper comparison
            real_denorm = dataset.denormalize_batch(real)
            real_profiles.append(real_denorm.numpy())
            if (
                len(real_profiles) * self.config.batch_size >= 1000
            ):  # Limit for evaluation
                break

        real_profiles = np.concatenate(real_profiles, axis=0)

        # Compute real Vs30 distribution
        real_vs30, avg_samples_per_meter = utils_mod.compute_real_vs30_and_density(
            self.config.parquet_path
        )

        logging.info(f"Loaded {len(real_profiles)} real profiles")
        logging.info(
            f"Real Vs30: mean={np.mean(real_vs30):.2f}, std={np.std(real_vs30):.2f}"
        )

        return real_profiles, real_vs30, avg_samples_per_meter, dataset

    def load_trained_model(self, checkpoint_path: str) -> Tuple[torch.nn.Module, int]:
        """
        Load a trained model from checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file

        Returns:
            Tuple of (model, step)
        """
        logging.info(f"Loading model from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Create model
        model = models_mod.create_model(self.config.model_type, self.config).to(
            self.device
        )

        model.load_state_dict(checkpoint["model"])
        model.eval()

        step = checkpoint.get("step", 0)

        logging.info(f"Loaded {self.config.model_type.upper()} model from step {step}")
        return model, step

    def generate_samples(
        self, model: torch.nn.Module, n_samples: int = 1000
    ) -> np.ndarray:
        """
        Generate samples from the trained FFM model.

        Args:
            model: Trained FFM model
            n_samples: Number of samples to generate

        Returns:
            Generated samples array (denormalized)
        """
        model.eval()
        generated_samples = []

        with torch.no_grad():
            for _ in range(0, n_samples, self.config.batch_size):
                batch_size = min(
                    self.config.batch_size, n_samples - len(generated_samples)
                )
                # Create initial noise
                initial_noise = torch.randn(batch_size, 1, self.real_data.shape[-1]).to(
                    self.device
                )

                # Generate samples using ODE solver
                if self.config.use_pcfm:
                    samples_normalized = utils_mod.sample_ffm_pcfm(
                        model, 
                        initial_noise, 
                        self.config.ode_steps, 
                        self.device,
                        self.dataset,
                        guidance_strength=self.config.pcfm_guidance_strength,
                        monotonic_weight=self.config.pcfm_monotonic_weight,
                        positivity_weight=self.config.pcfm_positivity_weight
                    )
                else:
                    samples_normalized = utils_mod.sample_ffm(
                        model, initial_noise, self.config.ode_steps, self.device
                    )
                
                # Denormalize samples before returning
                samples_denorm = self.dataset.denormalize_batch(samples_normalized)
                generated_samples.append(samples_denorm.cpu().numpy())

        return np.concatenate(generated_samples, axis=0)

    def calculate_vs30(self, profiles: np.ndarray) -> np.ndarray:
        """Calculate Vs30 for profiles."""
        return utils_mod.compute_generated_vs30(profiles, self.avg_samples_per_meter)

    def calculate_vs100(self, profiles: np.ndarray) -> np.ndarray:
        """Calculate Vs100 for profiles."""
        return utils_mod.compute_vs100(profiles, self.avg_samples_per_meter)

    def calculate_profile_statistics(self, profiles: np.ndarray) -> Dict:
        """
        Calculate comprehensive statistics for a set of profiles.

        Args:
            profiles: Array of shape (n_profiles, 1, n_layers)

        Returns:
            Dictionary containing various statistics
        """
        stats_dict = {}

        # Remove channel dimension for easier computation
        profiles_flat = profiles[:, 0, :]  # (n_profiles, n_layers)

        # Basic statistics
        stats_dict["mean_vs"] = np.mean(profiles_flat, axis=0)
        stats_dict["std_vs"] = np.std(profiles_flat, axis=0)
        stats_dict["min_vs"] = np.min(profiles_flat, axis=0)
        stats_dict["max_vs"] = np.max(profiles_flat, axis=0)

        # Vs30, Vs100 statistics
        vs30_values = self.calculate_vs30(profiles)
        vs100_values = self.calculate_vs100(profiles)

        stats_dict["vs30_mean"] = np.mean(vs30_values)
        stats_dict["vs30_std"] = np.std(vs30_values)
        stats_dict["vs30_min"] = np.min(vs30_values)
        stats_dict["vs30_max"] = np.max(vs30_values)

        stats_dict["vs100_mean"] = np.mean(vs100_values)
        stats_dict["vs100_std"] = np.std(vs100_values)
        stats_dict["vs100_min"] = np.min(vs100_values)
        stats_dict["vs100_max"] = np.max(vs100_values)

        # Profile shape statistics
        gradients = np.gradient(profiles_flat, axis=1)
        stats_dict["gradient_mean"] = np.mean(gradients, axis=0)
        stats_dict["gradient_std"] = np.std(gradients, axis=0)

        return stats_dict

    def compare_profiles(
        self, real_profiles: np.ndarray, generated_profiles: np.ndarray, model_name: str
    ) -> Dict:
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
        real_vs30 = self.calculate_vs30(real_profiles)
        gen_vs30 = self.calculate_vs30(generated_profiles)

        comparison["vs30_mse"] = mean_squared_error(real_vs30, gen_vs30)
        comparison["vs30_mae"] = mean_absolute_error(real_vs30, gen_vs30)
        comparison["vs30_ks_statistic"] = stats.ks_2samp(real_vs30, gen_vs30).statistic  # type: ignore
        comparison["vs30_ks_pvalue"] = stats.ks_2samp(real_vs30, gen_vs30).pvalue  # type: ignore

        # Vs100 comparison
        real_vs100 = self.calculate_vs100(real_profiles)
        gen_vs100 = self.calculate_vs100(generated_profiles)

        comparison["vs100_mse"] = mean_squared_error(real_vs100, gen_vs100)
        comparison["vs100_mae"] = mean_absolute_error(real_vs100, gen_vs100)
        comparison["vs100_ks_statistic"] = stats.ks_2samp(
            real_vs100, gen_vs100
        ).statistic  # type: ignore
        comparison["vs100_ks_pvalue"] = stats.ks_2samp(real_vs100, gen_vs100).pvalue  # type: ignore

        # Store statistics
        comparison["real_stats"] = real_stats
        comparison["gen_stats"] = gen_stats

        # Log results
        logging.info(f"  Profile MSE: {comparison['profile_mse']:.4f}")
        logging.info(f"  Profile MAE: {comparison['profile_mae']:.4f}")
        logging.info(f"  Vs30 MSE: {comparison['vs30_mse']:.4f}")
        logging.info(f"  Vs30 KS statistic: {comparison['vs30_ks_statistic']:.4f}")
        logging.info(f"  Vs100 MSE: {comparison['vs100_mse']:.4f}")
        logging.info(f"  Vs100 KS statistic: {comparison['vs100_ks_statistic']:.4f}")

        return comparison

    def plot_generated_profiles(
        self, generated_profiles: np.ndarray, step: int, save_path: Optional[str] = None
    ) -> None:
        """
        Plot generated profiles.

        Args:
            generated_profiles: Generated profiles array
            step: Training step
            save_path: Path to save the plot
        """
        if save_path is None:
            save_path = str(self.plots_dir / f"generated_profiles_step_{step}.png")

        plt.figure(figsize=(12, 8))

        # Plot first 20 profiles
        n_profiles = min(20, generated_profiles.shape[0])
        depths = np.arange(generated_profiles.shape[-1]) / self.avg_samples_per_meter

        for i in range(n_profiles):
            plt.plot(
                generated_profiles[i, 0, :], depths, "b-", alpha=0.6, linewidth=0.8
            )

        plt.xlabel("Vs (m/s)")
        plt.ylabel("Depth (m)")
        plt.title(f"Generated Profiles - Step {step}")
        plt.gca().invert_yaxis()  # Invert y-axis so depth increases downward
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logging.info(f"Generated profiles plot saved to {save_path}")

    def plot_vs30_comparison(
        self,
        real_vs30: np.ndarray,
        generated_vs30: np.ndarray,
        step: int,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot Vs30 distribution comparison.

        Args:
            real_vs30: Real Vs30 values
            generated_vs30: Generated Vs30 values
            step: Training step
            save_path: Path to save the plot
        """
        if save_path is None:
            save_path = str(self.plots_dir / f"vs30_comparison_step_{step}.png")

        plt.figure(figsize=(10, 6))

        plt.hist(
            real_vs30, bins=50, alpha=0.7, label="Real", density=True, color="blue"
        )
        plt.hist(
            generated_vs30,
            bins=50,
            alpha=0.7,
            label="Generated",
            density=True,
            color="red",
        )

        # Calculate KS statistic
        ks_stat = stats.ks_2samp(real_vs30, generated_vs30).statistic  # type: ignore

        plt.xlabel("Vs30 (m/s)")
        plt.ylabel("Density")
        plt.title(
            f"Vs30 Distribution Comparison - Step {step}\nKS Statistic: {ks_stat:.4f}"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logging.info(f"Vs30 comparison plot saved to {save_path}")

    def plot_vs100_comparison(
        self,
        real_vs100: np.ndarray,
        generated_vs100: np.ndarray,
        step: int,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot Vs100 distribution comparison.

        Args:
            real_vs100: Real Vs100 values
            generated_vs100: Generated Vs100 values
            step: Training step
            save_path: Path to save the plot
        """
        if save_path is None:
            save_path = str(self.plots_dir / f"vs100_comparison_step_{step}.png")

        plt.figure(figsize=(10, 6))

        plt.hist(
            real_vs100, bins=50, alpha=0.7, label="Real", density=True, color="blue"
        )
        plt.hist(
            generated_vs100,
            bins=50,
            alpha=0.7,
            label="Generated",
            density=True,
            color="red",
        )

        # Calculate KS statistic
        ks_stat = stats.ks_2samp(real_vs100, generated_vs100).statistic  # type: ignore

        plt.xlabel("Vs100 (m/s)")
        plt.ylabel("Density")
        plt.title(
            f"Vs100 Distribution Comparison - Step {step}\nKS Statistic: {ks_stat:.4f}"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logging.info(f"Vs100 comparison plot saved to {save_path}")

    def plot_comprehensive_comparison(
        self,
        real_profiles: np.ndarray,
        generated_profiles: np.ndarray,
        step: int,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Create comprehensive comparison plots.

        Args:
            real_profiles: Real profiles array
            generated_profiles: Generated profiles array
            step: Training step
            save_path: Path to save the plot
        """
        if save_path is None:
            save_path = str(
                self.plots_dir / f"comprehensive_comparison_step_{step}.png"
            )

        logging.info("Creating comprehensive comparison plots...")

        # Set up the plot
        plt.figure(figsize=(20, 12))

        # Calculate metrics
        real_vs30 = self.calculate_vs30(real_profiles)
        gen_vs30 = self.calculate_vs30(generated_profiles)
        real_vs100 = self.calculate_vs100(real_profiles)
        gen_vs100 = self.calculate_vs100(generated_profiles)

        # 1. Profile comparison (top row)
        ax1 = plt.subplot(2, 3, 1)
        depths = np.arange(real_profiles.shape[-1]) / self.avg_samples_per_meter

        # Plot real profiles
        for i in range(min(20, real_profiles.shape[0])):
            ax1.plot(real_profiles[i, 0, :], depths, "b-", alpha=0.3, linewidth=0.5)

        # Plot generated profiles
        for i in range(min(20, generated_profiles.shape[0])):
            ax1.plot(
                generated_profiles[i, 0, :], depths, "r-", alpha=0.3, linewidth=0.5
            )

        ax1.set_title("Profile Comparison")
        ax1.set_xlabel("Vs (m/s)")
        ax1.set_ylabel("Depth (m)")
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3)

        # 2. Mean profile comparison
        ax2 = plt.subplot(2, 3, 2)

        real_mean = np.mean(real_profiles[:, 0, :], axis=0)
        real_std = np.std(real_profiles[:, 0, :], axis=0)
        gen_mean = np.mean(generated_profiles[:, 0, :], axis=0)
        gen_std = np.std(generated_profiles[:, 0, :], axis=0)

        ax2.plot(real_mean, depths, "b-", label="Real Mean", linewidth=2)
        ax2.fill_betweenx(
            depths, real_mean - real_std, real_mean + real_std, alpha=0.3, color="blue"
        )

        ax2.plot(gen_mean, depths, "r-", label="Generated Mean", linewidth=2)
        ax2.fill_betweenx(
            depths, gen_mean - gen_std, gen_mean + gen_std, alpha=0.3, color="red"
        )

        ax2.set_title("Mean Profile Comparison")
        ax2.set_xlabel("Vs (m/s)")
        ax2.set_ylabel("Depth (m)")
        ax2.invert_yaxis()
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Vs30 comparison
        ax3 = plt.subplot(2, 3, 3)
        ks_vs30 = stats.ks_2samp(real_vs30, gen_vs30).statistic  # type: ignore

        ax3.hist(
            real_vs30, bins=30, alpha=0.7, label="Real", density=True, color="blue"
        )
        ax3.hist(
            gen_vs30, bins=30, alpha=0.7, label="Generated", density=True, color="red"
        )

        ax3.set_title(f"Vs30 Distribution\nKS: {ks_vs30:.4f}")
        ax3.set_xlabel("Vs30 (m/s)")
        ax3.set_ylabel("Density")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Vs100 comparison
        ax4 = plt.subplot(2, 3, 4)
        ks_vs100 = stats.ks_2samp(real_vs100, gen_vs100).statistic  # type: ignore

        ax4.hist(
            real_vs100, bins=30, alpha=0.7, label="Real", density=True, color="blue"
        )
        ax4.hist(
            gen_vs100, bins=30, alpha=0.7, label="Generated", density=True, color="red"
        )

        ax4.set_title(f"Vs100 Distribution\nKS: {ks_vs100:.4f}")
        ax4.set_xlabel("Vs100 (m/s)")
        ax4.set_ylabel("Density")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Gradient comparison
        ax5 = plt.subplot(2, 3, 5)

        real_gradients = np.gradient(real_profiles[:, 0, :], axis=1)
        gen_gradients = np.gradient(generated_profiles[:, 0, :], axis=1)

        real_grad_mean = np.mean(real_gradients, axis=0)
        gen_grad_mean = np.mean(gen_gradients, axis=0)

        # Ensure dimensions match for plotting
        min_len = min(len(real_grad_mean), len(gen_grad_mean), len(depths))
        ax5.plot(
            real_grad_mean[:min_len], depths[:min_len], "b-", label="Real", linewidth=2
        )
        ax5.plot(
            gen_grad_mean[:min_len],
            depths[:min_len],
            "r-",
            label="Generated",
            linewidth=2,
        )

        ax5.set_title("Velocity Gradients")
        ax5.set_xlabel("dV/dz (m/s/m)")
        ax5.set_ylabel("Depth (m)")
        ax5.invert_yaxis()
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Statistics summary
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis("off")

        # Calculate summary statistics
        profile_mse = mean_squared_error(real_mean, gen_mean)
        vs30_mse = mean_squared_error(real_vs30, gen_vs30)
        vs100_mse = mean_squared_error(real_vs100, gen_vs100)

        stats_text = f"""Statistics Summary (Step {step})
        
Profile MSE: {profile_mse:.4f}
Vs30 MSE: {vs30_mse:.4f}
Vs100 MSE: {vs100_mse:.4f}

Vs30 KS: {ks_vs30:.4f}
Vs100 KS: {ks_vs100:.4f}

Real Vs30: {np.mean(real_vs30):.1f} ± {np.std(real_vs30):.1f}
Gen Vs30: {np.mean(gen_vs30):.1f} ± {np.std(gen_vs30):.1f}

Real Vs100: {np.mean(real_vs100):.1f} ± {np.std(real_vs100):.1f}
Gen Vs100: {np.mean(gen_vs100):.1f} ± {np.std(gen_vs100):.1f}
"""

        ax6.text(
            0.1,
            0.9,
            stats_text,
            transform=ax6.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logging.info(f"Comprehensive comparison plot saved to {save_path}")

    def evaluate_checkpoint(self, checkpoint_path: str) -> Dict:
        """
        Evaluate a specific checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file

        Returns:
            Dictionary containing evaluation results
        """
        logging.info(f"Evaluating checkpoint: {checkpoint_path}")

        # Load model
        model, step = self.load_trained_model(checkpoint_path)

        # Generate samples
        generated_profiles = self.generate_samples(model, n_samples=1000)

        # Compare with real data
        comparison = self.compare_profiles(
            self.real_data,
            generated_profiles,
            f"ffm_{self.config.model_type}_step_{step}",
        )

        # Create plots
        self.plot_generated_profiles(generated_profiles, step)

        real_vs30 = self.calculate_vs30(self.real_data)
        gen_vs30 = self.calculate_vs30(generated_profiles)
        self.plot_vs30_comparison(real_vs30, gen_vs30, step)

        real_vs100 = self.calculate_vs100(self.real_data)
        gen_vs100 = self.calculate_vs100(generated_profiles)
        self.plot_vs100_comparison(real_vs100, gen_vs100, step)

        self.plot_comprehensive_comparison(self.real_data, generated_profiles, step)

        # Add step information
        comparison["step"] = step
        comparison["checkpoint_path"] = checkpoint_path

        return comparison

    def evaluate_all_checkpoints(self) -> Dict:
        """
        Evaluate all available checkpoints.

        Returns:
            Dictionary containing evaluation results for all checkpoints
        """
        logging.info("Evaluating all checkpoints...")

        checkpoint_files = list(self.results_dir.glob("checkpoint_*.pt"))
        checkpoint_files.sort(
            key=lambda x: int(x.stem.split("_")[1])
            if x.stem != "checkpoint_final"
            else float("inf")
        )

        if not checkpoint_files:
            logging.warning("No checkpoint files found!")
            return {}

        all_results = {}

        for checkpoint_file in checkpoint_files:
            try:
                results = self.evaluate_checkpoint(str(checkpoint_file))
                all_results[checkpoint_file.stem] = results
            except Exception as e:
                logging.error(f"Error evaluating {checkpoint_file}: {e}")
                continue

        # Save all results
        results_file = self.results_dir / "evaluation_results.pkl"
        with open(results_file, "wb") as f:
            pickle.dump(all_results, f)

        logging.info(f"All evaluation results saved to {results_file}")

        return all_results


def main():
    """Main evaluation function."""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting FFM Evaluation")

    # Configuration
    config = cfg_mod.cfg
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create evaluator
    evaluator = FFMEvaluator(config, device)

    # Find the latest checkpoint
    checkpoint_files = list(evaluator.results_dir.glob("checkpoint_*.pt"))
    if not checkpoint_files:
        logging.warning("No checkpoint files found!")
        return

    # Sort by step number and get the latest
    checkpoint_files.sort(
        key=lambda x: int(x.stem.split("_")[1])
        if x.stem != "checkpoint_final"
        else float("inf")
    )
    latest_checkpoint = checkpoint_files[-1]

    logging.info(f"Evaluating latest checkpoint: {latest_checkpoint}")

    # Evaluate the latest checkpoint
    result = evaluator.evaluate_checkpoint(str(latest_checkpoint))

    if result:
        logging.info("Evaluation completed successfully!")

        # Print summary
        step = result.get("step", "unknown")
        vs30_ks = result.get("vs30_ks_statistic", "N/A")
        vs100_ks = result.get("vs100_ks_statistic", "N/A")
        logging.info(
            f"Final checkpoint (step {step}): Vs30 KS={vs30_ks:.4f}, Vs100 KS={vs100_ks:.4f}"
        )

        # Save final results
        results_file = evaluator.results_dir / "final_evaluation_results.pkl"
        with open(results_file, "wb") as f:
            pickle.dump(result, f)
        logging.info(f"Final evaluation results saved to {results_file}")
    else:
        logging.warning("Evaluation failed")


if __name__ == "__main__":
    main()
