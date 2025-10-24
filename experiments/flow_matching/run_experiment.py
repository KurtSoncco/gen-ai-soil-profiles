#!/usr/bin/env python3
"""
FFM Experiment Runner

This script runs the FFM experiment and evaluation, creating a complete
analysis similar to the conv1d_gan experiment.
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
            logging.FileHandler("ffm_experiment.log"),
        ],
    )


def main():
    """Run the FFM experiment and evaluation."""
    setup_logging()
    logging.info("🚀 Starting FFM experiment...")

    # Get the directory of this script
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent  # Go up to project root

    print("🚀 Starting FFM experiment...")
    print(f"📁 Working directory: {project_root}")
    print("=" * 50)

    try:
        # Step 1: Run training
        logging.info("Step 1: Training FFM model...")
        print("Step 1: Training FFM model...")

        result = subprocess.run(
            [sys.executable, "-m", "experiments.flow_matching.train"],
            cwd=project_root,
            check=True,
        )

        print("✅ Training completed successfully!")
        logging.info("Training completed successfully!")

        # Step 2: Run evaluation
        logging.info("Step 2: Running evaluation...")
        print("Step 2: Running evaluation...")

        result = subprocess.run(
            [sys.executable, "-m", "experiments.flow_matching.evaluate_ffm"],
            cwd=project_root,
            check=True,
        )

        assert result is not None

        print("✅ Evaluation completed successfully!")
        logging.info("Evaluation completed successfully!")

        # Step 3: Run sampling
        logging.info("Step 3: Generating samples...")
        print("Step 3: Generating samples...")

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.flow_matching.sample",
                "--num_samples",
                "100",
            ],
            cwd=project_root,
            check=True,
        )

        assert result is not None

        print("✅ Sampling completed successfully!")
        logging.info("Sampling completed successfully!")

        # Step 4: Create summary
        logging.info("Step 4: Creating experiment summary...")
        print("Step 4: Creating experiment summary...")

        create_experiment_summary(script_dir)

        print("✅ FFM experiment completed successfully!")
        logging.info("FFM experiment completed successfully!")

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

    results_dir = script_dir / ".." / ".." / "outputs" / "flow_matching"
    plots_dir = results_dir / "plots"

    # Create summary file
    summary_file = results_dir / "experiment_summary.md"

    with open(summary_file, "w") as f:
        f.write("# FFM Experiment Summary\n\n")
        f.write("## Overview\n")
        f.write(
            "This experiment trains a Functional Flow Matching (FFM) model to generate Vs profiles "
            "using either UNet or FNO neural field architectures.\n\n"
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

        f.write("## FFM Method\n")
        f.write(
            "Functional Flow Matching learns a vector field v_θ(u, t) that transforms noise "
        )
        f.write("into realistic profiles through ODE integration:\n\n")
        f.write("```\n")
        f.write("du/dt = v_θ(u, t)\n")
        f.write("u(0) = noise, u(1) = realistic profile\n")
        f.write("```\n\n")

        f.write("## Architecture Options\n")
        f.write("- **UNet**: 1D U-Net with skip connections and time conditioning\n")
        f.write("- **FNO**: Fourier Neural Operator with spectral convolutions\n\n")

        f.write("## Evaluation Results\n")
        f.write("The experiment generates comprehensive analysis including:\n")
        f.write("- Generated vs Real profile comparisons\n")
        f.write("- Vs30 and Vs100 distribution comparisons\n")
        f.write("- Training loss curves\n")
        f.write("- ODE integration trajectory visualization\n")
        f.write("- Statistical metrics (MSE, MAE, KS statistics)\n")
        f.write("- Comprehensive evaluation plots and metrics\n\n")

        f.write("## Usage\n")
        f.write("To run individual components:\n")
        f.write("```bash\n")
        f.write("# Training only\n")
        f.write("python -m experiments.flow_matching.train\n\n")
        f.write("# Evaluation only\n")
        f.write("python -m experiments.flow_matching.evaluate_ffm\n\n")
        f.write("# Sampling only\n")
        f.write("python -m experiments.flow_matching.sample --num_samples 100\n\n")
        f.write("# Full experiment\n")
        f.write("python -m experiments.flow_matching.run_experiment\n")
        f.write("```\n\n")

        f.write("## Configuration\n")
        f.write("Edit `experiments/flow_matching/config.py` to modify:\n")
        f.write("- Model architecture (unet/fno)\n")
        f.write("- Training parameters (learning rate, steps, batch size)\n")
        f.write("- ODE integration steps for sampling\n")
        f.write("- Data paths and normalization\n")

    print(f"📄 Experiment summary created: {summary_file}")
    logging.info(f"Experiment summary created: {summary_file}")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
