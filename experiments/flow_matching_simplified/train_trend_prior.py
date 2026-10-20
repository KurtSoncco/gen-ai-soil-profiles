"""Train the flagship on the same-profile projection prior.

The seed-42 split, per-profile normalization, and optimizer hyperparameters
stay fixed. The run seed changes initialization and the Eq. 5 noise.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

os.environ["WANDB_MODE"] = "disabled"

from experiments.flow_matching_simplified import config as cfg_mod
from experiments.flow_matching_simplified.depth_dependent_stats.train_with_depth_losses import (
    main,
)

ROOT = Path(__file__).resolve().parents[2]
SPLIT = ROOT / "outputs" / "flow_matching_simplified" / "results" / "split_indices_seed42.json"
DATA = ROOT / "experiments" / "flow_matching_simplified" / "data" / "breakpoints_onewaytime.parquet"
STATS = (
    ROOT
    / "experiments"
    / "flow_matching_simplified"
    / "depth_dependent_stats"
    / "real_depth_stats_onewaytime.pkl"
)
CORR = (
    ROOT
    / "experiments"
    / "flow_matching_simplified"
    / "depth_dependent_stats"
    / "real_correlations_onewaytime.pkl"
)
NAMED = ROOT / "outputs" / "flow_matching_simplified" / "checkpoints"


def launch(seed: int, epochs: int = 1000, output_dir: Path | None = None) -> None:
    cfg = cfg_mod.cfg
    cfg.seed = seed
    cfg.num_epochs = epochs
    cfg.prior_mode = "projection"
    cfg.fixed_split_path = str(SPLIT)
    cfg.dataloader_seed = 42
    cfg.num_workers = 0
    cfg.data_path = str(DATA)
    out = output_dir or (ROOT / "outputs" / f"flow_matching_simplified_trend_s{seed}")
    cfg.output_dir = str(out)
    sys.argv = ["train_trend_prior", "--target-stats", str(STATS), "--target-corr", str(CORR)]
    t0 = time.perf_counter()
    main()
    elapsed = time.perf_counter() - t0
    print(f"seed {seed} wall_seconds {elapsed:.1f}", flush=True)
    if epochs == 1000:
        ckpt = out / "checkpoints"
        NAMED.mkdir(parents=True, exist_ok=True)
        for src, dst in (
            ("latest.pt", f"latest_trend_s{seed}.pt"),
            ("best.pt", f"best_trend_s{seed}.pt"),
        ):
            shutil.copy2(ckpt / src, NAMED / dst)
            print(f"copied {src} -> {NAMED / dst}", flush=True)


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    launch(seed, epochs)
