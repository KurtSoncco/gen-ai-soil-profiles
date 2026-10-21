"""Score trend-prior checkpoints under trend and oracle starts.

The trend model is fit on the seed-42 train split. Oracle starts use each
held-out test profile's own projection and are not the decision-rule rows.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.flow_matching_simplified import config as cfg_mod
from experiments.flow_matching_simplified.baseline_metrics import (
    _denorm_generated,
    euler_integrate,
)
from experiments.flow_matching_simplified.comprehensive_eval import load_checkpoint
from experiments.flow_matching_simplified.data import (
    FlowMatchingDataLoader,
    FlowMatchingDataset,
)
from experiments.flow_matching_simplified.eval_corrected import (
    bound_rate,
    bootstrap_vertical,
)
from experiments.flow_matching_simplified.projection import (
    ALPHA,
    fit_trend,
    draw_trend_knots,
    knot_outside_fraction,
    projection_tokens,
    tokens_from_knots,
)
from experiments.flow_matching_simplified.run_corrected_eval import (
    CURVATURE_STEPS,
    N_BOOT,
    N_CURVATURE,
    N_SAMPLES,
    STEP_COUNTS,
    _pairwise,
    _score_set,
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
NAMED = ROOT / "outputs" / "flow_matching_simplified" / "checkpoints"
OUT = (
    ROOT / "outputs" / "flow_matching_simplified" / "results" / "stage6_trend_eval.json"
)


def _boundaries(sequence: np.ndarray) -> np.ndarray:
    depths = np.asarray(sequence[:, 1], dtype=np.float64)
    if len(depths) == 0 or depths[0] != 0.0:
        depths = np.concatenate([[0.0], depths])
    return depths


def _pack(tokens, stats_list, max_length, device, noise_seed):
    n = len(tokens)
    clean = torch.zeros(n, max_length, 2, device=device)
    mask = torch.zeros(n, max_length, dtype=torch.bool, device=device)
    stats = torch.zeros(n, 4, device=device)
    lengths = []
    for i, (tok, st) in enumerate(zip(tokens, stats_list)):
        ts = (tok[:, 0] - st["ts_mean"]) / st["ts_std"]
        depth = (tok[:, 1] - st["depth_mean"]) / st["depth_std"]
        norm = np.column_stack([ts, depth]).astype(np.float32)
        length = min(len(norm), max_length)
        lengths.append(length)
        clean[i, :length] = torch.as_tensor(norm[:length], device=device)
        mask[i, :length] = True
        stats[i] = torch.tensor(
            [st["ts_mean"], st["ts_std"], st["depth_mean"], st["depth_std"]],
            dtype=torch.float32,
            device=device,
        )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(noise_seed))
    eps = torch.randn(clean.shape, generator=generator).to(device)
    x0 = ((1.0 - ALPHA) * clean + ALPHA * eps) * mask.unsqueeze(-1).float()
    return x0, mask, stats, stats_list, np.asarray(lengths)


def _trend_batch(train_raw, train_stats, fit, n, seed, max_length, device):
    rng = np.random.default_rng(seed)
    tokens = []
    stats = []
    outside = 0
    total = 0
    for _ in range(n):
        index = int(rng.integers(0, len(train_raw)))
        bounds = _boundaries(train_raw[index])
        ln, valid, n_out, n_kn = draw_trend_knots(
            fit, float(bounds[-1]), rng, clip=True
        )
        outside += n_out
        total += n_kn
        tokens.append(tokens_from_knots(bounds, ln, valid))
        stats.append(train_stats[index])
    packed = _pack(tokens, stats, max_length, device, noise_seed=seed + 17)
    report = {
        "n_knots": total,
        "n_outside": outside,
        "fraction_outside": float(outside / total) if total else float("nan"),
        "clipped": True,
    }
    return packed, report


def _oracle_batch(test_raw, test_stats, max_length, device, seed):
    tokens = [projection_tokens(seq) for seq in test_raw]
    return _pack(tokens, test_stats, max_length, device, noise_seed=seed + 17000)


def _curvature(model, make_batch, device) -> dict:
    t0 = time.perf_counter()
    remaining = N_CURVATURE
    acc = 0.0
    count = 0.0
    cursor = 0
    while remaining > 0:
        n = min(64, remaining)
        u0, masks, stats, _, _ = make_batch(n, cursor)
        cursor += 1
        u = u0.clone()
        velocities = []
        dt = 1.0 / CURVATURE_STEPS
        with torch.no_grad():
            for step in range(CURVATURE_STEPS):
                t = torch.full((u0.shape[0], 1), step * dt, device=device)
                v = model(u, masks, t, stats)
                velocities.append(v)
                u = (u + v * dt) * masks.unsqueeze(-1).float()
        target = u - u0
        pred = torch.stack(velocities, dim=0)
        mask = masks.unsqueeze(-1).float().expand_as(target).unsqueeze(0)
        err = (pred - target.unsqueeze(0)) ** 2
        acc += float((err * mask).sum())
        count += float(mask.sum())
        remaining -= n
    return {
        "protocol": "10000 generated trajectories, 128 Euler steps, Euler state",
        "curvature": acc / count if count else float("nan"),
        "n_samples": N_CURVATURE,
        "seconds": time.perf_counter() - t0,
    }


def _dataset(raw):
    ds = FlowMatchingDataset(raw, max_length=cfg_mod.cfg.max_length, normalize=True)
    return ds.sequence_stats


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = json.loads(SPLIT.read_text())
    loader = FlowMatchingDataLoader(DATA)
    loader.load_data()
    assert loader.sequences is not None
    train_raw = [loader.sequences[i] for i in split["train"]]
    test_raw = [loader.sequences[i] for i in split["test"]]
    train_stats = _dataset(train_raw)
    test_stats = _dataset(test_raw)
    fit = fit_trend(train_raw)
    train_bounds = knot_outside_fraction(train_raw)
    real = []
    test_ds = FlowMatchingDataset(
        test_raw, max_length=cfg_mod.cfg.max_length, normalize=True
    )
    for i in range(len(test_raw)):
        real.append(test_ds.denormalize_sequence(test_ds.sequences[i], i))
    real_boot = bootstrap_vertical(real, n_boot=N_BOOT, seed=42)
    max_length = cfg_mod.cfg.max_length
    results = {
        "split": str(SPLIT),
        "train_index_sum": int(sum(split["train"])),
        "n_train": len(train_raw),
        "n_test": len(test_raw),
        "trend_fit_length_m": fit["length_m"],
        "training_projection_knots_outside": train_bounds,
        "real_bound_rate": bound_rate(real),
        "real_pairwise": _pairwise(real),
        "seeds": {},
    }
    for seed in (42, 43, 44):
        ckpt = NAMED / f"latest_trend_s{seed}.pt"
        print(f"scoring {ckpt.name}", flush=True)
        model, epoch = load_checkpoint(str(ckpt), device, cfg_mod.cfg)
        model.eval()
        block = {"epoch_index": int(epoch), "conditions": {}}
        trend_pack, trend_report = _trend_batch(
            train_raw, train_stats, fit, N_SAMPLES, seed, max_length, device
        )
        oracle_pack = _oracle_batch(test_raw, test_stats, max_length, device, seed)
        block["trend_start_knots_outside_before_clip"] = trend_report
        for name, pack in (("trend", trend_pack), ("oracle", oracle_pack)):
            u0, masks, stats, stats_list, lengths = pack
            cond = {"n": int(len(lengths)), "sweeps": {}}
            for steps in STEP_COUNTS:
                t0 = time.perf_counter()
                gen = _denorm_generated(
                    euler_integrate(model, u0, masks, stats, steps), lengths, stats_list
                )
                cond["sweeps"][str(steps)] = _score_set(real, gen, real_boot, lengths)
                cond["sweeps"][str(steps)]["seconds"] = time.perf_counter() - t0
                row = cond["sweeps"][str(steps)]
                print(
                    f"  seed {seed} {name} euler {steps} "
                    f"trend {row['depth_trend']['weighted_mean_ln_rmse']:.4f} "
                    f"bounds {row['bound_rate']:.3f}",
                    flush=True,
                )
            if name == "trend":
                cond["curvature"] = _curvature(
                    model,
                    lambda n, cursor: _trend_batch(
                        train_raw,
                        train_stats,
                        fit,
                        n,
                        seed + 1000 + cursor,
                        max_length,
                        device,
                    )[0],
                    device,
                )
            else:
                cond["curvature"] = _curvature(
                    model,
                    lambda n, cursor, raw=test_raw, st=test_stats: _oracle_batch(
                        [raw[(cursor * 64 + i) % len(raw)] for i in range(n)],
                        [st[(cursor * 64 + i) % len(st)] for i in range(n)],
                        max_length,
                        device,
                        seed + 5000 + cursor,
                    ),
                    device,
                )
            print(
                f"  curvature {name} {cond['curvature']['curvature']:.4f}", flush=True
            )
            block["conditions"][name] = cond
        results["seeds"][str(seed)] = block
        OUT.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
