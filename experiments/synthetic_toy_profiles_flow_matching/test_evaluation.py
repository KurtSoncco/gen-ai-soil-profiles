#!/usr/bin/env python3
"""
Synthetic Toy Profiles FFM Evaluation Script

This script evaluates a trained flow matching model on synthetic toy profiles.
It computes layer statistics, creates profile comparison plots, and analyzes
the learned 2-layer structure.
"""

import argparse
import os
from pathlib import Path

import matplotlib

# Set non-interactive backend before importing pyplot (for headless environments/WSL)
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from . import config as cfg_mod
    from . import models as models_mod
    from . import utils as utils_mod
    from .data import create_dataloader
except ImportError:  # fallback when running as script
    import config as cfg_mod
    import models as models_mod
    import utils as utils_mod

    from data import create_dataloader  # type: ignore


def load_latest_checkpoint(dir_path: str) -> str:
    """Load the latest checkpoint from the directory."""
    dir_path = str(Path(dir_path))
    files = [
        f
        for f in Path(dir_path).iterdir()
        if f.is_file() and f.name.startswith("checkpoint_") and f.name.endswith(".pt")
    ]
    if not files:
        raise FileNotFoundError(f"No checkpoints found in {dir_path}")

    files.sort(
        key=lambda x: (
            int(x.stem.split("_")[-1]) if x.stem != "checkpoint_final" else float("inf")
        )
    )
    return str(files[-1])


def create_comprehensive_plot(
    real_profiles: np.ndarray,
    generated_profiles: np.ndarray,
    output_dir: str,
    step: int,
    real_breakpoints: np.ndarray | None = None,
    gen_breakpoints: np.ndarray | None = None,
) -> None:
    """Create comprehensive evaluation visualization using breakpoint statistics."""
    os.makedirs(output_dir, exist_ok=True)

    # Use breakpoint statistics if provided, otherwise fall back to layer statistics
    if real_breakpoints is not None and gen_breakpoints is not None:
        real_stats = utils_mod.compute_breakpoint_statistics(real_breakpoints)
        gen_stats = utils_mod.compute_breakpoint_statistics(gen_breakpoints)
        use_breakpoints = True
    else:
        # Fallback to layer statistics for backward compatibility
        real_stats = utils_mod.compute_layer_statistics(real_profiles)
        gen_stats = utils_mod.compute_layer_statistics(generated_profiles)
        use_breakpoints = False

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    if use_breakpoints:
        # Plot 1: Point 2 - tts1 distribution (breakpoint travel time)
        ax = axes[0, 0]
        ax.hist(
            real_stats["tts1_values"],
            bins=50,
            alpha=0.5,
            label="Real",
            density=True,
            color="blue",
        )
        ax.hist(
            gen_stats["tts1_values"],
            bins=50,
            alpha=0.5,
            label="Generated",
            density=True,
            color="red",
        )
        ax.set_xlabel("Point 2: tts1 (Travel Time)")
        ax.set_ylabel("Density")
        ax.set_title("Point 2: tts1 Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ks1 = utils_mod.ks_statistic(
            real_stats["tts1_values"], gen_stats["tts1_values"]
        )
        ax.text(
            0.05,
            0.95,
            f"KS={ks1:.3f}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # Plot 2: Point 2 - depth1 distribution (breakpoint depth)
        ax = axes[0, 1]
        ax.hist(
            real_stats["depth1_values"],
            bins=50,
            alpha=0.5,
            label="Real",
            density=True,
            color="blue",
        )
        ax.hist(
            gen_stats["depth1_values"],
            bins=50,
            alpha=0.5,
            label="Generated",
            density=True,
            color="red",
        )
        ax.set_xlabel("Point 2: depth1 (Depth)")
        ax.set_ylabel("Density")
        ax.set_title("Point 2: depth1 Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ks2 = utils_mod.ks_statistic(
            real_stats["depth1_values"], gen_stats["depth1_values"]
        )
        ax.text(
            0.05,
            0.95,
            f"KS={ks2:.3f}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # Plot 3: Point 3 - tts_end distribution
        ax = axes[0, 2]
        ax.hist(
            real_stats["tts_end_values"],
            bins=50,
            alpha=0.5,
            label="Real",
            density=True,
            color="blue",
        )
        ax.hist(
            gen_stats["tts_end_values"],
            bins=50,
            alpha=0.5,
            label="Generated",
            density=True,
            color="red",
        )
        ax.set_xlabel("Point 3: tts_end (Travel Time)")
        ax.set_ylabel("Density")
        ax.set_title("Point 3: tts_end Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ks_b = utils_mod.ks_statistic(
            real_stats["tts_end_values"], gen_stats["tts_end_values"]
        )
        ax.text(
            0.05,
            0.95,
            f"KS={ks_b:.3f}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
    else:
        # Fallback to old layer statistics
        # Plot 1: Layer 1 distribution
        ax = axes[0, 0]
        ax.hist(
            real_stats["layer1_values"],
            bins=50,
            alpha=0.5,
            label="Real",
            density=True,
            color="blue",
        )
        ax.hist(
            gen_stats["layer1_values"],
            bins=50,
            alpha=0.5,
            label="Generated",
            density=True,
            color="red",
        )
        ax.set_xlabel("Layer 1 Value (m/s)")
        ax.set_ylabel("Density")
        ax.set_title("Layer 1 Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ks1 = utils_mod.ks_statistic(
            real_stats["layer1_values"], gen_stats["layer1_values"]
        )
        ax.text(
            0.05,
            0.95,
            f"KS={ks1:.3f}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # Plot 2: Layer 2 distribution
        ax = axes[0, 1]
        ax.hist(
            real_stats["layer2_values"],
            bins=50,
            alpha=0.5,
            label="Real",
            density=True,
            color="blue",
        )
        ax.hist(
            gen_stats["layer2_values"],
            bins=50,
            alpha=0.5,
            label="Generated",
            density=True,
            color="red",
        )
        ax.set_xlabel("Layer 2 Value (m/s)")
        ax.set_ylabel("Density")
        ax.set_title("Layer 2 Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ks2 = utils_mod.ks_statistic(
            real_stats["layer2_values"], gen_stats["layer2_values"]
        )
        ax.text(
            0.05,
            0.95,
            f"KS={ks2:.3f}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # Plot 3: Boundary depth distribution
        ax = axes[0, 2]
        ax.hist(
            real_stats["boundary_depths"],
            bins=50,
            alpha=0.5,
            label="Real",
            density=True,
            color="blue",
        )
        ax.hist(
            gen_stats["boundary_depths"],
            bins=50,
            alpha=0.5,
            label="Generated",
            density=True,
            color="red",
        )
        ax.set_xlabel("Boundary Depth (index)")
        ax.set_ylabel("Density")
        ax.set_title("Boundary Depth Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ks_b = utils_mod.ks_statistic(
            real_stats["boundary_depths"], gen_stats["boundary_depths"]
        )
        ax.text(
            0.05,
            0.95,
            f"KS={ks_b:.3f}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    # Plot 4-6: Sample profile comparisons
    n_show = min(3, len(real_profiles), len(generated_profiles))
    for i in range(n_show):
        ax = axes[1, i]

        # Create depth array starting at 0
        profile_len = real_profiles.shape[-1]
        depth = np.arange(profile_len)

        # Plot Vs vs depth (x: Vs, y: depth)
        ax.plot(
            real_profiles[i, 0, :],
            depth,
            label="Real",
            alpha=0.7,
            linewidth=2,
            color="blue",
        )
        ax.plot(
            generated_profiles[i, 0, :],
            depth,
            label="Generated",
            alpha=0.7,
            linewidth=2,
            color="red",
        )
        ax.set_xlabel("Travel Time")
        ax.set_ylabel("Depth")
        ax.set_title(f"Sample Profile {i + 1}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Invert y-axis so depth increases downward (starting at 0 at top)
        ax.invert_yaxis()
        ax.set_ylim(profile_len - 1, 0)  # Ensure depth starts at 0 at top

        # Move xlabel and ticks to top
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()

    # Add overall title
    if use_breakpoints:
        title = f"Synthetic Toy Profiles Evaluation @ Step {step}\n"
        title += f"tts1 KS={ks1:.3f}, depth1 KS={ks2:.3f}, tts_end KS={ks_b:.3f}"
    else:
        title = f"Synthetic Toy Profiles Evaluation @ Step {step}\n"
        title += f"Layer1 KS={ks1:.3f}, Layer2 KS={ks2:.3f}, Boundary KS={ks_b:.3f}"
    fig.suptitle(title, fontsize=14, y=0.995)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"evaluation_comprehensive_{step}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved comprehensive evaluation to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Synthetic Toy Profiles Flow Matching Model"
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default=None, help="Path to checkpoint file"
    )
    parser.add_argument(
        "--num_samples", type=int, default=1000, help="Number of samples to generate"
    )
    parser.add_argument(
        "--ode_steps", type=int, default=None, help="Number of ODE integration steps"
    )
    args = parser.parse_args()

    cfg = cfg_mod.cfg
    device = torch.device(cfg.device)

    # Determine checkpoint path
    if args.checkpoint_path is None:
        checkpoint_path = load_latest_checkpoint(cfg.out_dir)
    else:
        checkpoint_path = args.checkpoint_path

    print(f"Loading checkpoint from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    step = checkpoint.get("step", 0)

    # Auto-detect model type
    checkpoint_config = checkpoint.get("config", {})
    model_type = checkpoint_config.get("model_type", cfg.model_type)

    # Get dataset info
    _, max_length, dataset = create_dataloader(
        batch_size=cfg.batch_size, num_workers=0, shuffle=False
    )

    # Create model
    model = models_mod.create_model(model_type, cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    print(
        f"Loaded {model_type.upper()} model with {sum(p.numel() for p in model.parameters()):,} parameters"
    )
    print(f"Model trained for {step} steps")

    # Generate samples (breakpoints)
    num_samples = args.num_samples
    ode_steps = args.ode_steps or cfg.ode_steps

    print(f"Generating {num_samples} breakpoint samples with {ode_steps} ODE steps...")

    initial_noise = torch.randn(num_samples, 3).to(device)

    with torch.no_grad():
        breakpoints_normalized = utils_mod.sample_ffm(
            model, initial_noise, ode_steps, device, cfg.ode_solver
        )

    breakpoints = dataset.denormalize_batch(breakpoints_normalized)
    breakpoints_np = breakpoints.cpu().numpy()

    # Load real data for comparison (breakpoints)
    real_breakpoints_np = []
    for i in range(min(num_samples, len(dataset))):
        real_breakpoints_np.append(dataset[i].cpu().numpy())
    real_breakpoints_np = np.stack(real_breakpoints_np, axis=0)

    # Denormalize real data
    real_breakpoints_np_normalized = torch.from_numpy(real_breakpoints_np)
    real_breakpoints_np = (
        dataset.denormalize_batch(real_breakpoints_np_normalized).cpu().numpy()
    )

    # Import reconstruct_profile
    try:
        from .data import reconstruct_profile
    except ImportError:
        from data import reconstruct_profile  # type: ignore

    # Reconstruct profiles from breakpoints for visualization (already denormalized, so apply_expm1=False)
    real_profiles_np = np.array(
        [
            reconstruct_profile(bp, dataset.max_depth, apply_expm1=False)
            for bp in real_breakpoints_np
        ]
    )
    samples_np = np.array(
        [
            reconstruct_profile(bp, dataset.max_depth, apply_expm1=False)
            for bp in breakpoints_np
        ]
    )

    # Reshape for compatibility with existing plotting code
    real_profiles_np = real_profiles_np[:, np.newaxis, :]  # (N, 1, 500)
    samples_np = samples_np[:, np.newaxis, :]  # (N, 1, 500)

    # Compute breakpoint statistics
    print("\n" + "=" * 80)
    print("EVALUATION METRICS - BREAKPOINTS")
    print("=" * 80)

    real_stats = utils_mod.compute_breakpoint_statistics(real_breakpoints_np)
    gen_stats = utils_mod.compute_breakpoint_statistics(breakpoints_np)

    print("\nPoint 1 (origin): (0, 0) - implicit")
    print("\nPoint 2 (breakpoint):")
    print(
        f"  depth1: Real={real_stats['depth1_mean']:.2f}±{real_stats['depth1_std']:.2f}, "
        f"Gen={gen_stats['depth1_mean']:.2f}±{gen_stats['depth1_std']:.2f}"
    )
    print(
        f"  tts1: Real={real_stats['tts1_mean']:.4f}±{real_stats['tts1_std']:.4f}, "
        f"Gen={gen_stats['tts1_mean']:.4f}±{gen_stats['tts1_std']:.4f}"
    )

    print("\nPoint 3 (end):")
    print(
        f"  tts_end: Real={real_stats['tts_end_mean']:.4f}±{real_stats['tts_end_std']:.4f}, "
        f"Gen={gen_stats['tts_end_mean']:.4f}±{gen_stats['tts_end_std']:.4f}"
    )

    # KS statistics for breakpoint distributions
    ks_depth1 = utils_mod.ks_statistic(
        real_stats["depth1_values"], gen_stats["depth1_values"]
    )
    ks_tts1 = utils_mod.ks_statistic(
        real_stats["tts1_values"], gen_stats["tts1_values"]
    )
    ks_tts_end = utils_mod.ks_statistic(
        real_stats["tts_end_values"], gen_stats["tts_end_values"]
    )

    print("\nKS Statistics (for breakpoint distributions):")
    print(f"  depth1:   {ks_depth1:.4f}")
    print(f"  tts1:     {ks_tts1:.4f}")
    print(f"  tts_end:  {ks_tts_end:.4f}")

    # Create plot
    print("\nCreating comprehensive evaluation plots...")
    create_comprehensive_plot(
        real_profiles_np,
        samples_np,
        cfg.plots_dir,
        step,
        real_breakpoints=real_breakpoints_np,
        gen_breakpoints=breakpoints_np,
    )

    print("\n" + "=" * 80)
    print("Evaluation completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
