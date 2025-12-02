"""Configuration for Flow Matching training and evaluation.

This module defines all hyperparameters and paths for training flow matching models
on variable-length paired token breakpoints [ts, depth].
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Config:
    """Configuration class for Flow Matching experiments with variable-length breakpoints."""

    # Data
    data_path: str = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "data", "breakpoints.parquet")
    )
    batch_size: int = 32
    num_workers: int = 4
    max_length: int = 20
    train_val_test_split: tuple[float, float, float] = (0.8, 0.1, 0.1)
    pad_token: float = 0.0
    normalize: bool = True

    # Model Architecture
    input_dim: int = 2  # [ts, depth]
    output_dim: int = 2  # predicted vector field
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    time_emb_dim: int = 128
    use_sequence_stats: bool = True  # Use per-sequence statistics conditioning
    stats_dim: int = 4  # ts_mean, ts_std, depth_mean, depth_std
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Training
    learning_rate: float = 1e-4
    betas: tuple[float, float] = (0.6, 0.999)
    weight_decay: float = 1e-4
    num_epochs: int = 1000
    gradient_clip: float = 1.0

    # LR Scheduler
    use_scheduler: bool = True
    scheduler_patience: int = 10  # epochs
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6
    scheduler_mode: str = "min"

    # Evaluation
    eval_every: int = 50  # epochs
    num_eval_samples: int = 200
    ode_steps: int = 50  # number of ODE integration steps for sampling

    # Vs constraints
    vs_min: float = 100.0  # Minimum allowed Vs value (m/s)
    vs_max: float = 5000.0  # Maximum allowed Vs value (m/s)
    min_dt: float = 1e-6  # Minimum Δt to prevent infinite/huge Vs
    vs_penalty_weight: float = (
        1e-2  # Weight for Vs regularization in training (start small)
    )
    use_vs_regularization: bool = True  # Toggle for training penalty

    # Logging
    log_every: int = 10  # batches
    checkpoint_every: int = 10  # epochs

    # Wandb
    wandb_project: str = "flow-matching-breakpoints"
    wandb_name: str | None = None  # if None, auto-generated

    # Output directories (all under outputs/flow_matching_simplified/)
    output_dir: str = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "outputs", "flow_matching_simplified"
        )
    )

    @property
    def checkpoints_dir(self) -> str:
        """Directory for model checkpoints."""
        return os.path.join(self.output_dir, "checkpoints")

    @property
    def plots_dir(self) -> str:
        """Directory for evaluation plots."""
        return os.path.join(self.output_dir, "plots")

    @property
    def results_dir(self) -> str:
        """Directory for JSON results and metrics."""
        return os.path.join(self.output_dir, "results")

    @property
    def samples_dir(self) -> str:
        """Directory for generated sample sequences."""
        return os.path.join(self.output_dir, "samples")

    # Misc
    seed: int = np.random.randint(0, 1000000)


cfg = Config()
