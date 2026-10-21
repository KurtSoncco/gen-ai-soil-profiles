"""Train the flagship on one-way breakpoints without editing the trainer.

Sets seed 42, the one-way parquet, and a separate output directory, then
calls ``train_with_depth_losses.main``. Checkpoints stay out of the
original ``checkpoints/`` directory.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

os.environ["WANDB_MODE"] = "disabled"

from experiments.flow_matching_simplified import config as cfg_mod
from experiments.flow_matching_simplified.depth_dependent_stats.train_with_depth_losses import (
    main,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "flow_matching_simplified_onewaytime"


def launch() -> None:
    cfg = cfg_mod.cfg
    cfg.seed = 42
    cfg.data_path = str(
        ROOT
        / "experiments"
        / "flow_matching_simplified"
        / "data"
        / "breakpoints_onewaytime.parquet"
    )
    cfg.output_dir = str(OUT)
    sys.argv = [
        "train_onewaytime",
        "--target-stats",
        str(
            ROOT
            / "experiments"
            / "flow_matching_simplified"
            / "depth_dependent_stats"
            / "real_depth_stats_onewaytime.pkl"
        ),
        "--target-corr",
        str(
            ROOT
            / "experiments"
            / "flow_matching_simplified"
            / "depth_dependent_stats"
            / "real_correlations_onewaytime.pkl"
        ),
    ]
    main()
    ckpt = OUT / "checkpoints"
    named = ROOT / "outputs" / "flow_matching_simplified" / "checkpoints"
    for src, dst in (
        ("latest.pt", "latest_onewaytime.pt"),
        ("best.pt", "best_onewaytime.pt"),
    ):
        shutil.copy2(ckpt / src, named / dst)
        print(f"copied {src} -> {named / dst}", flush=True)


if __name__ == "__main__":
    launch()
