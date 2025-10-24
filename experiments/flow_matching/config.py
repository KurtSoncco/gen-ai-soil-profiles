from __future__ import annotations

import os
from dataclasses import dataclass
import torch


@dataclass
class Config:
    # Data
    parquet_path: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "vspdb_vs_profiles.parquet"))
    feature_column: str = "vs_value"  # column name containing the profile values
    group_column: str | None = "velocity_metadata_id"  # optional grouping id per profile; None assumes single-column array-like
    max_length: int | None = None  # if None, inferred from dataset
    pad_value: float = 0.0
    batch_size: int = 16  # Reduced from 32
    num_workers: int = 2

    # Model Architecture
    model_type: str = "fno"  # "unet" or "fno"
    unet_dim: int = 32  # base dimension for UNet (reduced from 64)
    fno_modes: int = 8  # number of Fourier modes for FNO (reduced from 16)
    fno_width: int = 32  # width of FNO layers (reduced from 64)
    time_emb_dim: int = 64  # dimension of time embedding (reduced from 128)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # FFM Training
    learning_rate: float = 1e-4
    num_steps: int = 100  # number of training steps
    log_every: int = 10  # log every N steps
    checkpoint_every: int = 50  # save checkpoint every N steps
    
    # Regularization
    tvd_weight: float = 0.01  # Total Variation Diminishing regularization weight
    
    # FFM Sampling
    ode_steps: int = 100  # number of ODE integration steps for sampling
    num_samples: int = 16  # number of samples to generate during evaluation

    # IO
    out_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "flow_matching"))
    plots_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "flow_matching", "plots"))
    results_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "flow_matching", "results"))
    
    # Wandb
    wandb_project: str = "ffm-soil-profiles"
    wandb_name: str | None = None  # if None, auto-generated
    
    # Misc
    seed: int = 42


cfg = Config()
