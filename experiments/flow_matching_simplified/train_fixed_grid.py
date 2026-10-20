"""Fixed-grid MLP with the same-profile projection prior.

The state is standardized ln(Vs) on the 20 knots. The seed-42 train split
and the knot standardization are shared. The run seed changes initialization
and the Eq. 5 noise. Batch order is seeded with 42 for every run.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.flow_matching_simplified.data import FlowMatchingDataLoader
from experiments.flow_matching_simplified.eval_corrected import TREND_DEPTHS_M, ln_vs_matrix
from experiments.flow_matching_simplified.projection import (
    ALPHA,
    FACTOR,
    project_vector,
)
from experiments.flow_matching_simplified.train import set_seed

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "experiments" / "flow_matching_simplified" / "data" / "breakpoints_onewaytime.parquet"
SPLIT = ROOT / "outputs" / "flow_matching_simplified" / "results" / "split_indices_seed42.json"
OUT = ROOT / "outputs" / "flow_matching_simplified_fixed_grid"
HIDDEN = 256
N_LAYERS = 4
BATCH = 32
LR = 1e-4
EPOCHS = 1000


class GridFlow(nn.Module):
    def __init__(self, n_knots: int) -> None:
        super().__init__()
        self.time = nn.Sequential(nn.Linear(1, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, HIDDEN))
        blocks: list[nn.Module] = []
        width = n_knots * 2 + HIDDEN
        for _ in range(N_LAYERS):
            blocks.append(nn.Linear(width, HIDDEN))
            blocks.append(nn.SiLU())
            width = HIDDEN
        blocks.append(nn.Linear(HIDDEN, n_knots))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = torch.cat([x, mask, self.time(t)], dim=-1)
        return self.net(h) * mask


def _matrices(sequences: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = ln_vs_matrix(sequences, TREND_DEPTHS_M)
    valid = np.isfinite(raw)
    mu = np.zeros(raw.shape[1])
    sig = np.ones(raw.shape[1])
    for j in range(raw.shape[1]):
        col = raw[valid[:, j], j]
        if len(col):
            mu[j] = float(col.mean())
            sig[j] = float(col.std() + 1e-3)
    physical = np.where(valid, raw, 0.0)
    projected = np.zeros_like(physical)
    for i in range(len(physical)):
        projected[i] = project_vector(physical[i], valid[i], FACTOR)
    x1 = np.where(valid, (physical - mu) / sig, 0.0)
    proj = np.where(valid, (projected - mu) / sig, 0.0)
    return x1.astype(np.float32), proj.astype(np.float32), valid, np.column_stack([mu, sig])


def train_one(seed: int, x1, proj, valid, scale, device) -> dict:
    set_seed(seed)
    model = GridFlow(x1.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    xt = torch.tensor(x1, device=device)
    pt = torch.tensor(proj, device=device)
    mask = torch.tensor(valid, dtype=torch.float32, device=device)
    order_rng = np.random.default_rng(42)
    t0 = time.perf_counter()
    last_loss = None
    for epoch in range(EPOCHS):
        model.train()
        order = torch.tensor(order_rng.permutation(len(x1)), device=device)
        total = 0.0
        seen = 0
        for start in range(0, len(order), BATCH):
            rows = order[start : start + BATCH]
            target = xt[rows]
            m = mask[rows]
            eps = torch.randn_like(target)
            x0 = ((1.0 - ALPHA) * pt[rows] + ALPHA * eps) * m
            t = torch.rand(len(rows), 1, device=device)
            state = ((1.0 - t) * x0 + t * target) * m
            pred = model(state, m, t)
            err = (pred - (target - x0)) ** 2
            loss = (err * m).sum() / m.sum().clamp(min=1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(rows)
            seen += len(rows)
        last_loss = total / seen
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"fixed-grid seed {seed} epoch {epoch + 1} loss {last_loss:.4f}", flush=True)
    elapsed = time.perf_counter() - t0
    path = OUT / f"grid_s{seed}.pt"
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "seed": seed,
            "scale": scale,
            "epoch": EPOCHS - 1,
            "n_params": sum(p.numel() for p in model.parameters()),
        },
        path,
    )
    return {"seed": seed, "final_loss": last_loss, "seconds": elapsed, "checkpoint": str(path)}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = FlowMatchingDataLoader(DATA)
    loader.load_data()
    assert loader.sequences is not None
    split = json.loads(SPLIT.read_text())
    train = [loader.sequences[i] for i in split["train"]]
    x1, proj, valid, scale = _matrices(train)
    records = []
    for seed in (42, 43, 44):
        records.append(train_one(seed, x1, proj, valid, scale, device))
    summary = {
        "split": str(SPLIT),
        "train_index_sum": int(sum(split["train"])),
        "n_train": len(train),
        "epochs": EPOCHS,
        "seeds": records,
    }
    (OUT / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
