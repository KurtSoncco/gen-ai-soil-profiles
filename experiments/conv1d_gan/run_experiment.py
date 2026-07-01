#!/usr/bin/env python3
"""
Conv1D GAN Experiment Runner

This script runs the Conv1D GAN experiment and evaluation, creating a complete
analysis similar to the parametric and VAE experiments.
"""

import logging
import subprocess
import sys
from pathlib import Path


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("conv1d_gan_experiment.log"),
        ],
    )


def main():
    """Run the Conv1D GAN experiment and evaluation."""
    setup_logging()
    logging.info("🚀 Starting Conv1D GAN experiment...")

    # Get the directory of this script
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent  # Go up to project root

    print("🚀 Starting Conv1D GAN experiment...")
    print(f"📁 Working directory: {project_root}")
    print("=" * 50)

    try:
        # Step 1: Run training
        logging.info("Step 1: Training Conv1D GAN...")
        print("Step 1: Training Conv1D GAN...")

        result = subprocess.run(
            [sys.executable, "-m", "experiments.conv1d_gan.train"],
            cwd=project_root,
            check=True,
        )

        print("✅ Training completed successfully!")
        logging.info("Training completed successfully!")

        # Step 2: Run evaluation
        logging.info("Step 2: Running evaluation...")
        print("Step 2: Running evaluation...")

        result = subprocess.run(
            [sys.executable, "-m", "experiments.conv1d_gan.evaluate_conv1d_gan"],
            cwd=project_root,
            check=True,
        )

        assert result is not None

        print("✅ Evaluation completed successfully!")
        logging.info("Evaluation completed successfully!")

        # Step 3: Create summary
        logging.info("Step 3: Creating experiment summary...")
        print("Step 3: Creating experiment summary...")

        create_experiment_summary(script_dir)

        print("✅ Conv1D GAN experiment completed successfully!")
        logging.info("Conv1D GAN experiment completed successfully!")

        return 0

    except subprocess.CalledProcessError as e:
        error_msg = f"Error running experiment: {e}"
        print(f"❌ {error_msg}")
        logging.error(error_msg)
        return e.returncode
    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        print(f"❌ {error_msg}")
        logging.error(error_msg)
        return 1


def create_experiment_summary(script_dir: Path):
    """Create a summary of the experiment results."""
    import glob
    import os

    results_dir = script_dir / ".." / ".." / "outputs" / "conv1d_gan"
    plots_dir = results_dir / "plots"

    # Create summary file
    summary_file = results_dir / "experiment_summary.md"

    with open(summary_file, "w") as f:
        f.write("# Conv1D GAN Experiment Summary\n\n")
        f.write("## Overview\n")
        f.write(
            "This experiment trains a Conv1D GAN to generate Vs profiles and evaluates its performance.\n\n"
        )

        f.write("## Files Generated\n\n")

        # List checkpoint files
        checkpoint_files = glob.glob(str(results_dir / "checkpoint_*.pt"))
        if checkpoint_files:
            f.write("### Checkpoints\n")
            for checkpoint in sorted(checkpoint_files):
                f.write(f"- {os.path.basename(checkpoint)}\n")
            f.write("\n")

        # List sample files
        sample_files = glob.glob(str(results_dir / "samples_*.npy"))
        if sample_files:
            f.write("### Generated Samples\n")
            for sample in sorted(sample_files):
                f.write(f"- {os.path.basename(sample)}\n")
            f.write("\n")

        # List plot files
        if plots_dir.exists():
            plot_files = glob.glob(str(plots_dir / "*.png"))
            if plot_files:
                f.write("### Plots\n")
                for plot in sorted(plot_files):
                    f.write(f"- {os.path.basename(plot)}\n")
                f.write("\n")

        f.write("## Evaluation Results\n")
        f.write(
            "The evaluation script generates comprehensive comparison plots including:\n"
        )
        f.write("- Generated vs Real profile comparisons\n")
        f.write("- Vs30 and Vs100 distribution comparisons\n")
        f.write("- Training loss curves\n")
        f.write("- Statistical metrics (MSE, MAE, KS statistics)\n\n")

        f.write("## Usage\n")
        f.write("To run individual components:\n")
        f.write("```bash\n")
        f.write("# Training only\n")
        f.write("python -m experiments.conv1d_gan.train\n\n")
        f.write("# Evaluation only\n")
        f.write("python -m experiments.conv1d_gan.evaluate_conv1d_gan\n")
        f.write("```\n")

    print(f"📄 Experiment summary created: {summary_file}")
    logging.info(f"Experiment summary created: {summary_file}")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
