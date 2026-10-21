"""Time one short fixed-grid flow. Does not train the flagship.

The state is ln(Vs) on the 20 knots at 10 m through 200 m. The prior is the
same-profile projection: average groups of 4 valid knots, linearly upsample,
then x0 = (1 - 0.5) T(x1) + 0.5 * noise. This is a wall-clock check, not a result.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.flow_matching_simplified.data import FlowMatchingDataLoader
from experiments.flow_matching_simplified.eval_corrected import (
    TREND_DEPTHS_M,
    ln_vs_matrix,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = (
    ROOT
    / "experiments"
    / "flow_matching_simplified"
    / "data"
    / "breakpoints_onewaytime.parquet"
)
SPLIT = (
    ROOT
    / "outputs"
    / "flow_matching_simplified"
    / "results"
    / "split_indices_seed42.json"
)
OUT = (
    ROOT / "outputs" / "flow_matching_simplified" / "results" / "fixed_grid_timing.json"
)

FACTOR = 4
ALPHA = 0.5
BATCH = 32
WARMUP_EPOCHS = 1
TIMED_EPOCHS = 10
HIDDEN = 256
N_LAYERS = 4
LR = 1e-4


def project_rows(values: np.ndarray, valid: np.ndarray, factor: int) -> np.ndarray:
    """Average contiguous groups of ``factor`` valid knots and interpolate back."""
    out = np.zeros_like(values)
    for i in range(len(values)):
        idx = np.flatnonzero(valid[i])
        if len(idx) == 0:
            continue
        vals = values[i, idx]
        centers = []
        coarse = []
        n = len(vals)
        for start in range(0, n, factor):
            sl = np.arange(start, min(start + factor, n))
            centers.append(float(sl.mean()))
            coarse.append(float(vals[sl].mean()))
        if len(coarse) == 1:
            up = np.full(n, coarse[0])
        else:
            up = np.interp(np.arange(n), np.asarray(centers), np.asarray(coarse))
        out[i, idx] = up
    return out


class GridFlow(nn.Module):
    def __init__(self, n_knots: int) -> None:
        super().__init__()
        self.time = nn.Sequential(
            nn.Linear(1, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, HIDDEN)
        )
        blocks: list[nn.Module] = []
        width = n_knots * 2 + HIDDEN
        for _ in range(N_LAYERS):
            blocks.append(nn.Linear(width, HIDDEN))
            blocks.append(nn.SiLU())
            width = HIDDEN
        blocks.append(nn.Linear(HIDDEN, n_knots))
        self.net = nn.Sequential(*blocks)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        h = torch.cat([x, mask, self.time(t)], dim=-1)
        return self.net(h) * mask


def run_epoch(
    model: GridFlow,
    opt: torch.optim.Optimizer,
    x1: torch.Tensor,
    proj: torch.Tensor,
    mask: torch.Tensor,
    order: torch.Tensor,
) -> float:
    model.train()
    total = 0.0
    seen = 0
    for start in range(0, len(order), BATCH):
        rows = order[start : start + BATCH]
        target = x1[rows]
        m = mask[rows]
        eps = torch.randn_like(target)
        x0 = ((1.0 - ALPHA) * proj[rows] + ALPHA * eps) * m
        t = torch.rand(len(rows), 1, device=target.device)
        xt = ((1.0 - t) * x0 + t * target) * m
        pred = model(xt, m, t)
        err = (pred - (target - x0)) ** 2
        loss = (err * m).sum() / m.sum().clamp(min=1.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        total += float(loss) * len(rows)
        seen += len(rows)
    return total / seen


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = FlowMatchingDataLoader(DATA)
    loader.load_data()
    assert loader.sequences is not None
    split = json.loads(SPLIT.read_text())
    train = [loader.sequences[i] for i in split["train"]]
    raw = ln_vs_matrix(train, TREND_DEPTHS_M)
    valid = np.isfinite(raw)
    mu = np.zeros(raw.shape[1])
    sig = np.ones(raw.shape[1])
    for j in range(raw.shape[1]):
        col = raw[valid[:, j], j]
        if len(col):
            mu[j] = float(col.mean())
            sig[j] = float(col.std() + 1e-3)
    norm = np.where(valid, (raw - mu) / sig, 0.0)
    physical = np.where(valid, raw, 0.0)
    physical_proj = project_rows(physical, valid, FACTOR)
    proj = np.where(valid, (physical_proj - mu) / sig, 0.0)

    x1 = torch.tensor(norm, dtype=torch.float32, device=device)
    projected = torch.tensor(proj, dtype=torch.float32, device=device)
    m = torch.tensor(valid, dtype=torch.float32, device=device)
    model = GridFlow(len(TREND_DEPTHS_M)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n_params = sum(p.numel() for p in model.parameters())
    rng = np.random.default_rng(42)

    def epoch(seed_epoch: int) -> tuple[float, float]:
        order = torch.tensor(rng.permutation(len(train)), device=device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = run_epoch(model, opt, x1, projected, m, order)
        if device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - t0, loss

    for i in range(WARMUP_EPOCHS):
        epoch(i)
    times = []
    losses = []
    for i in range(TIMED_EPOCHS):
        dt, loss = epoch(WARMUP_EPOCHS + i)
        times.append(dt)
        losses.append(loss)
    median = float(np.median(times))
    record = {
        "device": str(device),
        "torch": torch.__version__,
        "n_train": len(train),
        "n_knots": int(len(TREND_DEPTHS_M)),
        "factor": FACTOR,
        "alpha": ALPHA,
        "batch": BATCH,
        "hidden": HIDDEN,
        "n_layers": N_LAYERS,
        "n_params": n_params,
        "warmup_epochs": WARMUP_EPOCHS,
        "timed_epochs": TIMED_EPOCHS,
        "epoch_seconds": times,
        "epoch_loss": losses,
        "median_epoch_seconds": median,
        "extrapolated_1000_epoch_hours": median * 1000 / 3600,
        "extrapolated_3_seed_hours": median * 1000 * 3 / 3600,
        "note": "Short timing run. Not a Stage 6 result.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
