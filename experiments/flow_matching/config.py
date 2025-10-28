"""Configuration for Flow Matching training and evaluation.

This module defines all hyperparameters and paths for training flow matching models
on Vs (shear wave velocity) profiles.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch


@dataclass
class Config:
    """Configuration class for Flow Matching experiments.

    This configuration supports both UNet and FNO architectures for unguided
    flow matching without physics constraints.
    """

    # Data
    parquet_path: str = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "vspdb_vs_profiles.parquet"
        )
    )
    feature_column: str = "vs_value"  # column name containing the profile values
    group_column: str | None = (
        "velocity_metadata_id"  # optional grouping id per profile; None assumes single-column array-like
    )
    max_length: int | None = None  # if None, inferred from dataset
    pad_value: float = (
        0.0  # Note: With z-score normalization, padding uses 0 (mean-centered)
    )
    batch_size: int = 32
    num_workers: int = 4

    # Model Architecture
    model_type: str = "fno"  # "unet" or "fno"
    unet_dim: int = 64  # base dimension for UNet (reduced from 64)
    fno_modes: int = 16  # number of Fourier modes for FNO (reduced from 16)
    fno_width: int = 64  # width of FNO layers (reduced from 64)
    time_emb_dim: int = 128  # dimension of time embedding (reduced from 128)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # FFM Training
    learning_rate: float = 1e-4
    betas: tuple[float, float] = (0.6, 0.8)
    weight_decay: float = 1e-4
    num_steps: int = 5000  # number of training steps
    log_every: int = 250  # log every N steps
    checkpoint_every: int = 500  # save checkpoint every N steps

    # LR Scheduler
    use_scheduler: bool = True  # whether to use LR scheduler
    scheduler_patience: int = (
        1000  # ReduceLROnPlateau patience (recommended: 1000-1500 for 15k steps)
    )
    scheduler_factor: float = 0.5  # ReduceLROnPlateau factor
    scheduler_min_lr: float = 1e-6  # minimum learning rate
    scheduler_mode: str = "min"  # "min" or "max"

    # Regularization
    tvd_weight: float = 0.00  # Total Variation Diminishing regularization weight
    kinetic_energy_weight: float = (
        0.1  # Kinetic energy regularization weight (penalizes ||v||²)
    )

    # FFM Sampling
    ode_steps: int = (
        250  # number of ODE integration steps for sampling (increased for RK45)
    )
    num_samples: int = 16  # number of samples to generate during evaluation

    # IO
    out_dir: str = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "flow_matching")
    )
    plots_dir: str = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "outputs", "flow_matching", "plots"
        )
    )
    results_dir: str = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "outputs", "flow_matching", "results"
        )
    )

    # Wandb
    wandb_project: str = "ffm-soil-profiles"
    wandb_name: str | None = None  # if None, auto-generated

    # Misc
    seed: int = 42


cfg = Config()
