"""Stage 8 diagnostics for the one-way flagship. CPU only, no training.

Reproduces the numbers in ``docs/stage8_generation_critique.md``:

- how small layer increments are in per-profile normalized token space,
- how often the flagship emits invalid geometry (first token at or above the
  surface, non-increasing channels, non-positive or non-finite Vs),
- where the bound violations sit,
- whether drawing length and sequence statistics from one profile changes that,
- the Vs penalty's gradient with respect to the network.

Run: ``python -m experiments.flow_matching_simplified.stage8_diagnostics``
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from experiments.flow_matching_simplified import config as cfg_mod
from experiments.flow_matching_simplified.baseline_metrics import euler_integrate
from experiments.flow_matching_simplified.comprehensive_eval import load_checkpoint
from experiments.flow_matching_simplified.data import (
    FlowMatchingDataLoader,
    FlowMatchingDataset,
)
from experiments.flow_matching_simplified.eval_corrected import (
    VS_MAX,
    VS_MIN,
    bound_rate,
    vs_from_travel_time,
)
from experiments.flow_matching_simplified.utils import compute_vs_penalty

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "experiments/flow_matching_simplified/data/breakpoints_onewaytime.parquet"
SPLIT = ROOT / "outputs/flow_matching_simplified/results/split_indices_seed42.json"
CKPT = ROOT / "outputs/flow_matching_simplified/checkpoints/latest_onewaytime.pt"
RESULT = ROOT / "outputs/flow_matching_simplified/results/stage8_diagnostics.json"
N_SAMPLES = 200
STEPS = 50


def increment_scale(train: list[np.ndarray]) -> dict[str, float]:
    """Layer increments in units of each profile's own channel standard deviation."""
    first, smallest = [], []
    for seq in train:
        if len(seq) < 2:
            continue
        z = np.r_[0.0, seq[:, 1]]
        t = np.r_[0.0, seq[:, 0]]
        dz, dt = np.diff(z), np.diff(t)
        z_sd, t_sd = seq[:, 1].std(), seq[:, 0].std()
        first.append(dz[0] / z_sd)
        smallest.append(min(dz.min() / z_sd, dt.min() / t_sd))
    return {
        "first_layer_median": float(np.median(first)),
        "first_layer_p10": float(np.percentile(first, 10)),
        "smallest_increment_median": float(np.median(smallest)),
        "smallest_increment_p10": float(np.percentile(smallest, 10)),
    }


def geometry_report(sequences: list[np.ndarray]) -> dict[str, float]:
    first_nonpos = 0
    strictly_increasing = 0
    invalid_vs = 0
    viol_first = 0
    viol_other = 0
    thick = []
    for seq in sequences:
        if seq[0, 0] <= 0 or seq[0, 1] <= 0:
            first_nonpos += 1
        z = np.r_[0.0, seq[:, 1]]
        t = np.r_[0.0, seq[:, 0]]
        if np.all(np.diff(z) > 0) and np.all(np.diff(t) > 0):
            strictly_increasing += 1
        _, _, vs = vs_from_travel_time(seq)
        if np.any(~np.isfinite(vs) | (vs <= 0)):
            invalid_vs += 1
        bad = ~(np.isfinite(vs) & (vs >= VS_MIN) & (vs <= VS_MAX))
        viol_first += int(bad[0])
        viol_other += int(bad[1:].sum())
        dz = np.diff(z)
        thick.extend(dz[dz > 0].tolist())
    n = len(sequences)
    thick_a = np.asarray(thick)
    return {
        "n": n,
        "bound_rate": bound_rate(sequences),
        "first_token_at_or_above_surface": first_nonpos / n,
        "strictly_increasing_rate": strictly_increasing / n,
        "profiles_with_nonpositive_or_nonfinite_vs": invalid_vs,
        "violating_first_layers": viol_first,
        "violating_other_layers": viol_other,
        "fraction_positive_layers_under_1m": float(np.mean(thick_a < 1.0)),
    }


def _draw(model, train, stats, matched: bool, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    lens = np.array([len(s) for s in train])
    li = rng.integers(0, len(train), N_SAMPLES)
    si = li if matched else rng.integers(0, len(train), N_SAMPLES)
    lengths = np.minimum(lens[li], 20)
    mask = torch.zeros(N_SAMPLES, 20, dtype=torch.bool)
    for i, n in enumerate(lengths):
        mask[i, :n] = True
    cond = torch.tensor(stats[si], dtype=torch.float32)
    u0 = torch.randn(N_SAMPLES, 20, 2) * mask.unsqueeze(-1)
    u = euler_integrate(model, u0, mask, cond, STEPS).numpy()
    out = []
    for i, n in enumerate(lengths):
        x = u[i, :n]
        s = stats[si[i]]
        out.append(np.column_stack([x[:, 0] * s[1] + s[0], x[:, 1] * s[3] + s[2]]))
    return out


def penalty_gradient(model, ds: FlowMatchingDataset, cfg) -> dict[str, float]:
    """Gradient of the Vs penalty, as the trainer calls it, with respect to the weights."""
    batch = [ds[i] for i in range(32)]
    u1 = torch.stack([b["tokens"] for b in batch])
    mask = torch.stack([b["attention_mask"] for b in batch])
    stats = torch.stack([b["sequence_stats"] for b in batch])
    u0 = torch.randn_like(u1) * mask.unsqueeze(-1).float()
    t = torch.rand(32, 1, 1)
    ut = (1 - t) * u0 + t * u1
    model.zero_grad()
    pen = compute_vs_penalty(
        ut, mask, stats, cfg.vs_min, cfg.vs_max, cfg.min_dt, normalize=True
    )
    grads_exist = pen.requires_grad
    if grads_exist:
        pen.backward()
    total = sum(
        float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None
    )
    return {
        "penalty_value": float(pen.detach()),
        "penalty_requires_grad": bool(grads_exist),
        "sum_abs_grad_over_weights": total,
    }


def main() -> None:
    cfg = cfg_mod.cfg
    device = torch.device("cpu")
    loader = FlowMatchingDataLoader(DATA)
    loader.load_data()
    assert loader.sequences is not None
    split = json.loads(SPLIT.read_text())
    train = [loader.sequences[i] for i in split["train"]]
    test = [loader.sequences[i] for i in split["test"]]
    ds = FlowMatchingDataset(train, max_length=20, normalize=True)
    stats = np.array(
        [
            [s["ts_mean"], s["ts_std"], s["depth_mean"], s["depth_std"]]
            for s in ds.sequence_stats
        ]
    )
    model, epoch = load_checkpoint(str(CKPT), device, cfg)
    model.eval()

    lens = np.array([len(s) for s in train])
    rank = lambda a: np.argsort(np.argsort(a))  # noqa: E731
    results = {
        "checkpoint": str(CKPT.relative_to(ROOT)),
        "epoch_index": int(epoch),
        "increment_scale": increment_scale(train),
        "length_vs_stats_spearman": {
            "depth_mean": float(np.corrcoef(rank(lens), rank(stats[:, 2]))[0, 1]),
            "depth_std": float(np.corrcoef(rank(lens), rank(stats[:, 3]))[0, 1]),
        },
        "real_test": geometry_report(test),
        "draws": {},
        "penalty_gradient": penalty_gradient(model, ds, cfg),
    }
    for seed in (42, 43):
        for matched in (False, True):
            key = f"seed{seed}_{'matched' if matched else 'independent'}"
            gen = _draw(model, train, stats, matched, seed)
            results["draws"][key] = geometry_report(gen)
            print(key, json.dumps(results["draws"][key]), flush=True)
    print("penalty", results["penalty_gradient"], flush=True)
    RESULT.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {RESULT}")


if __name__ == "__main__":
    main()
