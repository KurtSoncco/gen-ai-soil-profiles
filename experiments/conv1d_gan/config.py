from __future__ import annotations

import os
from dataclasses import dataclass
import torch


@dataclass
class Config:
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
    pad_value: float = 0.0
    batch_size: int = 64
    num_workers: int = 2

    # Model
    latent_dim: int = 128
    base_channels: int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Training
    lr: float = 2e-4
    betas: tuple[float, float] = (0.5, 0.999)
    num_steps: int = 500
    d_steps_per_g: int = 1
    log_every: int = 50
    checkpoint_every: int = 50

    # IO
    out_dir: str = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "conv1d_gan")
    )
    plots_dir: str = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "outputs", "conv1d_gan", "plots"
        )
    )
    results_dir: str = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "outputs", "conv1d_gan", "results"
        )
    )
    seed: int = 42


cfg = Config()
