"""Configuration for Flow Matching training and evaluation on Vs pairs.

This module defines all hyperparameters and paths for training flow matching models
on 4-dimensional Vs pairs [vs30, vs50, vs100, vs200].
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

import torch


@dataclass
class Config:
    """Configuration class for Vs Pairs Flow Matching experiments.

    This configuration supports both UNet and FNO architectures for unguided
    flow matching without physics constraints on 4D vs pairs.
    """

    # Data
    parquet_path: str = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "vs_pairs.parquet")
    )
    feature_columns: list[str] = field(
        default_factory=lambda: ["vs30", "vs50", "vs100", "vs200"]
    )  # columns containing the vs pairs
    max_length: int = 4  # Fixed: we have exactly 4 features
    pad_value: float = 0.0
    batch_size: int = 256  # Can use larger batches with simpler data
    num_workers: int = 4

    # Model Architecture
    model_type: str = "mlp"  # "mlp" or "fno" for 4D vs pairs
    mlp_dim: int = 128  # hidden dimension for MLP
    mlp_layers: int = 4  # number of layers for MLP
    unet_dim: int = 64  # base dimension for UNet
    fno_modes: int = 3  # number of Fourier modes for FNO (max 3 for 4D vectors)
    fno_width: int = 64  # width of FNO layers
    time_emb_dim: int = 64  # dimension of time embedding
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # FFM Training
    learning_rate: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-5
    num_steps: int = 1000  # number of training steps
    log_every: int = 100  # log every N steps
    checkpoint_every: int = 100  # save checkpoint every N steps

    # LR Scheduler
    use_scheduler: bool = True
    scheduler_patience: int = 200
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6
    scheduler_mode: Literal["min", "max"] = "min"

    # Regularization (disabled for unconstrained experiment)
    tvd_weight: float = 0.0
    kinetic_energy_weight: float = 0.0

    # FFM Sampling
    ode_steps: int = 50
    num_samples: int = 64  # generate more samples for visualization

    # IO
    out_dir: str = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "outputs", "vs_pairs_flow_matching"
        )
    )
    plots_dir: str = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "outputs",
            "vs_pairs_flow_matching",
            "plots",
        )
    )
    results_dir: str = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "outputs",
            "vs_pairs_flow_matching",
            "results",
        )
    )

    # Wandb
    wandb_project: str = "ffm-vs-pairs"
    wandb_name: str | None = None  # if None, auto-generated

    # Misc
    seed: int = 42


cfg = Config()
