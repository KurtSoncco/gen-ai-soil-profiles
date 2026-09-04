#!/usr/bin/env python3
"""Flow Matching Sampling Script for Synthetic Toy Profiles.

Generate synthetic profiles using a trained flow matching model.
Uses RK4 ODE integration to sample from the learned vector field.
"""

import argparse
import os

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
    files = [
        f
        for f in os.listdir(dir_path)
        if f.startswith("checkpoint_") and f.endswith(".pt")
    ]
    if not files:
        raise FileNotFoundError(f"No checkpoints found in {dir_path}")

    files.sort(
        key=lambda x: (
            int(x.split("_")[-1].split(".")[0])
            if x != "checkpoint_final.pt"
            else float("inf")
        )
    )
    return os.path.join(dir_path, files[-1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Synthetic Profiles using trained FFM model"
    )
    parser.add_argument(
        "--num_samples", type=int, default=16, help="Number of samples to generate"
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default=None, help="Path to checkpoint file"
    )
    parser.add_argument(
        "--ode_steps", type=int, default=None, help="Number of ODE integration steps"
    )
    parser.add_argument(
        "--output_path", type=str, default=None, help="Output path for samples"
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

    # Auto-detect model type from checkpoint
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
    print(f"Model trained for {checkpoint['step']} steps")

    # Generate samples (breakpoints)
    num_samples = args.num_samples
    ode_steps = args.ode_steps or cfg.ode_steps

    print(f"Generating {num_samples} breakpoint samples with {ode_steps} ODE steps...")

    # Create initial noise for breakpoints (3 values)
    initial_noise = torch.randn(num_samples, 3).to(device)

    # Generate breakpoints
    breakpoints_normalized = utils_mod.sample_ffm(
        model, initial_noise, ode_steps, device, cfg.ode_solver
    )

    # Denormalize breakpoints
    breakpoints = dataset.denormalize_batch(breakpoints_normalized)

    # Import reconstruct_profile
    try:
        from .data import reconstruct_profile  # type: ignore
    except ImportError:
        from data import reconstruct_profile  # type: ignore

    # Reconstruct profiles from breakpoints (already denormalized, so apply_expm1=False)
    breakpoints_np = breakpoints.cpu().numpy()
    profiles = np.array(
        [
            reconstruct_profile(bp, dataset.max_depth, apply_expm1=False)
            for bp in breakpoints_np
        ]
    )

    # Save breakpoints
    if args.output_path is None:
        output_path = os.path.join(cfg.out_dir, "samples_latest.npy")
    else:
        output_path = args.output_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, breakpoints_np)

    # Also save reconstructed profiles
    profiles_path = output_path.replace(".npy", "_profiles.npy")
    np.save(profiles_path, profiles)

    print(f"Saved {num_samples} breakpoint samples to: {output_path}")
    print(f"Saved {num_samples} reconstructed profiles to: {profiles_path}")
    print("Breakpoint statistics:")
    print(
        f"  depth1: Mean={breakpoints_np[:, 0].mean():.2f}, Std={breakpoints_np[:, 0].std():.2f}"
    )
    print(
        f"  tts1: Mean={breakpoints_np[:, 1].mean():.4f}, Std={breakpoints_np[:, 1].std():.4f}"
    )
    print(
        f"  tts_end: Mean={breakpoints_np[:, 2].mean():.4f}, Std={breakpoints_np[:, 2].std():.4f}"
    )

    # Create TTS/Vs comparison plot
    plot_path = os.path.join(os.path.dirname(output_path), "tts_vs_samples.png")
    utils_mod.plot_tts_vs_samples(
        breakpoints_np,
        plot_path,
        max_depth=dataset.max_depth,
        dz=cfg.depth_increment,
        num_samples=100,
        title="Generated Travel Time and Vs Profiles",
    )
    print(f"Saved TTS/Vs comparison plot to: {plot_path}")


if __name__ == "__main__":
    main()
