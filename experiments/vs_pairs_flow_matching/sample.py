#!/usr/bin/env python3
"""Flow Matching Sampling Script for Vs Pairs.

Generate vs pairs using a trained flow matching model.
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

    # Sort by step number
    files.sort(
        key=lambda x: int(x.split("_")[-1].split(".")[0])
        if x != "checkpoint_final.pt"
        else float("inf")
    )
    return os.path.join(dir_path, files[-1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Vs pairs using trained FFM model"
    )
    parser.add_argument(
        "--num_samples", type=int, default=64, help="Number of samples to generate"
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

    # Auto-detect model type from checkpoint config
    checkpoint_config = checkpoint.get("config", {})
    model_type = checkpoint_config.get("model_type", cfg.model_type)

    # Get dataset info for normalization
    _, max_length, dataset = create_dataloader(
        batch_size=cfg.batch_size, num_workers=0, shuffle=False
    )

    # Create model with detected type
    model = models_mod.create_model(model_type, cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    print(
        f"Loaded {model_type.upper()} model with {sum(p.numel() for p in model.parameters()):,} parameters"
    )
    print(f"Model trained for {checkpoint['step']} steps")

    # Generate samples
    num_samples = args.num_samples
    ode_steps = args.ode_steps or cfg.ode_steps

    print(f"Generating {num_samples} samples with {ode_steps} ODE steps...")

    # Create initial noise
    initial_noise = torch.randn(num_samples, 1, max_length).to(device)

    # Generate samples
    samples_normalized = utils_mod.sample_ffm(model, initial_noise, ode_steps, device)

    # Denormalize samples
    samples = dataset.denormalize_batch(samples_normalized)

    # Save samples
    if args.output_path is None:
        output_path = os.path.join(cfg.out_dir, "samples_latest.npy")
    else:
        output_path = args.output_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, samples.cpu().numpy())

    print(f"Saved {num_samples} samples to: {output_path}")
    print("Sample statistics:")
    samples_np = samples.squeeze(1).cpu().numpy()
    print(f"  Mean: {samples_np.mean(axis=0)}")
    print(f"  Std:  {samples_np.std(axis=0)}")
    print(f"  Min:  {samples_np.min(axis=0)}")
    print(f"  Max:  {samples_np.max(axis=0)}")


if __name__ == "__main__":
    main()
