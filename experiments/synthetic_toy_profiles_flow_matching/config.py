"""Configuration for Synthetic Toy Profiles Flow Matching training and evaluation.

This module defines all hyperparameters and paths for training flow matching models
on synthetic travel time profiles using breakpoints (origin, breakpoint, end point).
Profiles are represented as 3 breakpoints: [depth1, tts1, tts_end] instead of full profiles.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import torch


@dataclass
class Config:
    """Configuration class for Synthetic Toy Profiles Flow Matching experiments.

    This configuration supports MLP-based flow matching models that generate
    breakpoints [depth1, tts1, tts_end] for piecewise linear travel time profiles.
    Profiles are reconstructed from breakpoints as piecewise linear functions.
    """

    # Data
    parquet_path: str = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "data",
            "synthetic_toy_profiles.parquet",
        )
    )
    feature_column: str = "travel_time"
    group_column: str | None = "velocity_metadata_id"
    max_length: int | None = None  # if None, inferred from dataset (should be 500)
    pad_value: float = 0.0  # With z-score normalization, padding uses 0
    depth_increment: float = (
        1.0  # Depth increment in meters (dz) for converting depth index to actual depth
    )
    batch_size: int = 128
    num_workers: int = 4

    # Model Architecture
    model_type: str = "fno"  # "unet", "fno", or "mlp" (all are MLPs for 3 breakpoints)
    unet_dim: int = 64  # base dimension for UNet
    fno_modes: int = 16  # number of Fourier modes for FNO
    fno_width: int = 64  # width of FNO layers
    time_emb_dim: int = 128  # dimension of time embedding
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # FFM Training
    learning_rate: float = 5e-4
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-5
    loss_type: Literal["l1", "l2"] = (
        "l2"  # Reconstruction loss type: "l1" (MAE) or "l2" (MSE) - use L2 for breakpoints
    )
    reconstruction_loss_weight: float = (
        1.0  # Weight factor for reconstruction loss (L1/L2)
    )
    num_steps: int = 5000  # number of training steps
    log_every: int = 100  # log every N steps
    checkpoint_every: int = 250  # save checkpoint every N steps

    # GAN Discriminator
    use_gan_loss: bool = True  # Enable GAN adversarial loss
    gan_weight: float = 0.1  # Weight for GAN loss relative to reconstruction loss
    discriminator_lr: float = 1e-4  # Learning rate for discriminator
    discriminator_dim: int = 64  # Base dimension for discriminator
    d_steps_per_g: int = 1  # Number of discriminator steps per generator step

    # LR Scheduler
    use_scheduler: bool = True
    scheduler_patience: int = 500
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6
    scheduler_mode: Literal["min", "max"] = "min"

    # Regularization (disabled for unconstrained experiment)
    tvd_weight: float = 0.0
    kinetic_energy_weight: float = 0.0
    monotonicity_weight: float = 1.0  # Weight for monotonicity penalty (enforces positive slopes for travel_time profiles)

    # FFM Sampling
    ode_steps: int = 100
    ode_solver: Literal["euler", "rk4"] = (
        "rk4"  # ODE solver: "euler" or "rk4" (Runge-Kutta 4)
    )
    num_samples: int = 64  # number of samples to generate during evaluation

    # IO
    out_dir: str = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "outputs",
            "synthetic_toy_profiles_flow_matching",
        )
    )
    plots_dir: str = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "outputs",
            "synthetic_toy_profiles_flow_matching",
            "plots",
        )
    )
    results_dir: str = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "outputs",
            "synthetic_toy_profiles_flow_matching",
            "results",
        )
    )

    # Wandb
    wandb_project: str = "ffm-synthetic-toy-profiles-travel-time"
    wandb_name: str | None = None  # if None, auto-generated

    # Misc
    seed: int = 42


cfg = Config()
