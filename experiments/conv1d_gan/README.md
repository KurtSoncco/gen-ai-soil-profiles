Conv1D GAN for Soil Vs Profiles

Overview

This experiment trains a 1D Convolutional GAN to synthesize soil shear-wave velocity (Vs) profiles from borehole data. It handles variable profile depths via pad-and-mask to a fixed maximum length, while enabling variable-length generation at sampling time.

Structure

- `config.py`: Hyperparameters and paths
- `data.py`: Parquet loader, padding/masking, dataset and dataloader builders
- `models.py`: Conv1D Generator and Discriminator
- `train.py`: Training loop with logging and checkpointing
- `sample.py`: Utilities to sample synthetic profiles with optional variable-length truncation

Setup

1) Ensure dependencies are installed (project root uses `pyproject.toml`).
2) Place your dataset at `data/vspdb_vs_profiles.parquet` (default) or update `config.py`.

Run Training

```bash
source .venv/bin/activate
uv run python experiments/conv1d_gan/train.py
```

Sampling

```bash
uv run python experiments/conv1d_gan/sample.py --num 16 --truncate heuristic
```

Variable-Length Generation

- Train GAN to output fixed-length profiles (max depth). At sampling time:
  - Truncate by sampling a target depth from the empirical depth distribution; or
  - Use an end-of-profile token (optional; not implemented by default).

Notes

- This experiment uses Conv1D layers to learn local transitions and sharp boundaries between layers.
- Logging integrates with `wandb` if `WANDB_PROJECT` is set; otherwise logs locally.


