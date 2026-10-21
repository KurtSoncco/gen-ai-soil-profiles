"""Stage 7 checks. No training.

1. Catalog the unclipped training-projection knots outside [100, 5000] m/s
   and compare them with Euler-50 trend-start bound violations from the
   existing checkpoints.
2. Show whether the vertical-correlation loss can send a gradient into
   the velocity field, and what correlation it targets.
"""

from __future__ import annotations

import json
import pickle
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
from experiments.flow_matching_simplified.depth_dependent_stats.losses import (
    compute_vertical_correlation_loss,
)
from experiments.flow_matching_simplified.eval_corrected import (
    TREND_DEPTHS_M,
    bound_rate,
    profile_in_bounds,
    vs_from_travel_time,
)
from experiments.flow_matching_simplified.eval_trend_prior import _trend_batch
from experiments.flow_matching_simplified.projection import (
    KNOTS,
    LN_MAX,
    LN_MIN,
    fit_trend,
    knot_ln,
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
CORR = (
    ROOT
    / "experiments"
    / "flow_matching_simplified"
    / "depth_dependent_stats"
    / "real_correlations_onewaytime.pkl"
)
NAMED = ROOT / "outputs" / "flow_matching_simplified" / "checkpoints"
OUT = ROOT / "outputs" / "flow_matching_simplified" / "results" / "stage7_checks.json"
VS_MIN = 100.0
VS_MAX = 5000.0


def _raw_at_knot(sequence: np.ndarray, depth: float) -> float:
    z0, z1, vs = vs_from_travel_time(sequence)
    if len(vs) == 0:
        return float("nan")
    for a, b, v in zip(z0, z1, vs):
        if a < depth <= b:
            return float(v)
    return float("nan")


def _layer_rows(sequence: np.ndarray) -> list[dict]:
    z0, z1, vs = vs_from_travel_time(sequence)
    rows = []
    for a, b, v in zip(z0, z1, vs):
        if not np.isfinite(v) or v < VS_MIN or v > VS_MAX:
            rows.append(
                {
                    "z0": float(a),
                    "z1": float(b),
                    "vs": float(v) if np.isfinite(v) else None,
                    "side": "low"
                    if np.isfinite(v) and v < VS_MIN
                    else "high"
                    if np.isfinite(v)
                    else "nonfinite",
                }
            )
    return rows


def catalog_training(train_raw: list[np.ndarray]) -> dict:
    knots = []
    raw_layers = []
    profiles_raw = 0
    profiles_proj = 0
    n_raw_layers = 0
    for index, seq in enumerate(train_raw):
        ln, valid = knot_ln(seq)
        outside = valid & ((ln < LN_MIN) | (ln > LN_MAX))
        if outside.any():
            profiles_proj += 1
        for j in np.flatnonzero(outside):
            raw_vs = _raw_at_knot(seq, float(KNOTS[j]))
            knots.append(
                {
                    "train_row": index,
                    "depth_m": float(KNOTS[j]),
                    "projected_vs": float(np.exp(ln[j])),
                    "raw_layer_vs": raw_vs,
                    "side": "low" if ln[j] < LN_MIN else "high",
                }
            )
        layers = _layer_rows(seq)
        n_raw_layers += len(vs_from_travel_time(seq)[2])
        if layers:
            profiles_raw += 1
            for layer in layers:
                raw_layers.append({"train_row": index, **layer})
    shallow = sum(1 for row in raw_layers if row["z1"] <= 10.0)
    on_knot = sum(
        1 for row in raw_layers if any(row["z0"] < z <= row["z1"] for z in KNOTS)
    )
    depth_counts: dict[str, int] = {}
    for knot in knots:
        key = str(int(knot["depth_m"]))
        depth_counts[key] = depth_counts.get(key, 0) + 1
    return {
        "n_profiles": len(train_raw),
        "n_projected_knots_outside": len(knots),
        "n_profiles_with_projected_knot_outside": profiles_proj,
        "projected_outside_by_depth_m": depth_counts,
        "projected_knots": knots,
        "n_raw_layers": n_raw_layers,
        "n_raw_layers_outside": len(raw_layers),
        "n_profiles_with_raw_layer_outside": profiles_raw,
        "n_raw_outside_ending_by_10m": shallow,
        "n_raw_outside_covering_a_knot": on_knot,
        "raw_layers_outside": raw_layers,
    }


def correlation_check(train_raw: list[np.ndarray]) -> dict:
    with CORR.open("rb") as handle:
        stored = pickle.load(handle)
    targets = {str(k): float(v) for k, v in stored["mean_correlations"].items()}

    seq = np.asarray(train_raw[0], dtype=np.float64)
    stats = {
        "ts_mean": float(seq[:, 0].mean()),
        "ts_std": float(seq[:, 0].std() + 1e-6),
        "depth_mean": float(seq[:, 1].mean()),
        "depth_std": float(seq[:, 1].std() + 1e-6),
    }
    norm = np.column_stack(
        [
            (seq[:, 0] - stats["ts_mean"]) / stats["ts_std"],
            (seq[:, 1] - stats["depth_mean"]) / stats["depth_std"],
        ]
    ).astype(np.float32)
    tokens = torch.zeros(1, cfg_mod.cfg.max_length, 2)
    mask = torch.zeros(1, cfg_mod.cfg.max_length, dtype=torch.bool)
    length = min(len(norm), cfg_mod.cfg.max_length)
    tokens[0, :length] = torch.as_tensor(norm[:length])
    mask[0, :length] = True
    sequence_stats = torch.tensor(
        [[stats["ts_mean"], stats["ts_std"], stats["depth_mean"], stats["depth_std"]]],
        dtype=torch.float32,
    )
    target = {float(k): float(v) for k, v in stored["mean_correlations"].items()}
    plain = compute_vertical_correlation_loss(
        tokens,
        mask,
        sequence_stats,
        target,
        lags=[1.0, 2.0, 5.0, 10.0],
        weight=0.025,
    )
    paired = None
    for seq in train_raw:
        if len(seq) < 6:
            continue
        stats = {
            "ts_mean": float(seq[:, 0].mean()),
            "ts_std": float(seq[:, 0].std() + 1e-6),
            "depth_mean": float(seq[:, 1].mean()),
            "depth_std": float(seq[:, 1].std() + 1e-6),
        }
        norm = np.column_stack(
            [
                (seq[:, 0] - stats["ts_mean"]) / stats["ts_std"],
                (seq[:, 1] - stats["depth_mean"]) / stats["depth_std"],
            ]
        ).astype(np.float32)
        tokens_i = torch.zeros(1, cfg_mod.cfg.max_length, 2)
        mask_i = torch.zeros(1, cfg_mod.cfg.max_length, dtype=torch.bool)
        length_i = min(len(norm), cfg_mod.cfg.max_length)
        tokens_i[0, :length_i] = torch.as_tensor(norm[:length_i])
        mask_i[0, :length_i] = True
        stats_i = torch.tensor(
            [
                [
                    stats["ts_mean"],
                    stats["ts_std"],
                    stats["depth_mean"],
                    stats["depth_std"],
                ]
            ],
            dtype=torch.float32,
        )
        value = compute_vertical_correlation_loss(
            tokens_i,
            mask_i,
            stats_i,
            target,
            lags=[1.0, 2.0, 5.0, 10.0],
            weight=0.025,
        )
        if float(value.detach()) > 0.0:
            paired = {
                "n_tokens": int(length_i),
                "requires_grad": bool(value.requires_grad),
                "grad_fn": value.grad_fn is not None,
                "value": float(value.detach()),
            }
            break
    connected = tokens.detach().clone().requires_grad_(True)
    numpy_error = None
    try:
        compute_vertical_correlation_loss(
            connected,
            mask,
            sequence_stats,
            target,
            lags=[1.0, 2.0, 5.0, 10.0],
            weight=0.025,
        )
    except RuntimeError as exc:
        numpy_error = str(exc)

    leaf = torch.tensor(0.42, dtype=torch.float32)
    return {
        "stored_mean_correlations": targets,
        "weight": 0.025,
        "short_lag_weight": 2.0,
        "lags": [1.0, 2.0, 5.0, 10.0],
        "loss_on_data_requires_grad": bool(plain.requires_grad),
        "loss_on_data_grad_fn": plain.grad_fn is not None,
        "loss_on_data_value": float(plain.detach()),
        "loss_on_profile_with_pairs": paired,
        "requires_grad_input_error": numpy_error,
        "fresh_tensor_requires_grad": bool(leaf.requires_grad),
        "fresh_tensor_grad_fn": leaf.grad_fn is not None,
    }


def sample_violations(train_raw, train_stats, fit) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_length = cfg_mod.cfg.max_length
    seeds = {}
    for seed in (42, 43, 44):
        model, epoch = load_checkpoint(
            str(NAMED / f"latest_trend_s{seed}.pt"), device, cfg_mod.cfg
        )
        model.eval()
        pack, report = _trend_batch(
            train_raw, train_stats, fit, 200, seed, max_length, device
        )
        u0, masks, stats, stats_list, lengths = pack
        gen = _denorm_generated(
            euler_integrate(model, u0, masks, stats, 50), lengths, stats_list
        )
        violations = []
        n_layers = 0
        for profile_i, seq in enumerate(gen):
            _, _, vs = vs_from_travel_time(seq)
            n_layers += len(vs)
            for layer in _layer_rows(seq):
                violations.append({"profile": profile_i, **layer})
        by_depth = {str(int(z)): 0 for z in TREND_DEPTHS_M}
        n_cover = 0
        for row in violations:
            covered = False
            for z in TREND_DEPTHS_M:
                if row["z0"] < z <= row["z1"]:
                    by_depth[str(int(z))] += 1
                    covered = True
            n_cover += int(covered)
        seeds[str(seed)] = {
            "epoch_index": int(epoch),
            "device": str(device),
            "bound_rate": bound_rate(gen),
            "n_profiles": len(gen),
            "n_profiles_outside": int(sum(not profile_in_bounds(seq) for seq in gen)),
            "n_layers": n_layers,
            "n_layers_outside": len(violations),
            "n_low": int(sum(row["side"] == "low" for row in violations)),
            "n_high": int(sum(row["side"] == "high" for row in violations)),
            "n_nonfinite": int(sum(row["side"] == "nonfinite" for row in violations)),
            "n_outside_ending_by_10m": int(
                sum(row["z1"] <= 10.0 for row in violations)
            ),
            "n_outside_covering_any_knot": n_cover,
            "violations_covering_knot": by_depth,
            "clip_report": report,
            "median_outside_vs": float(
                np.median([row["vs"] for row in violations if row["vs"] is not None])
            )
            if any(row["vs"] is not None for row in violations)
            else None,
        }
        print(
            f"seed {seed} bound_rate {seeds[str(seed)]['bound_rate']:.3f} "
            f"layers_outside {len(violations)}",
            flush=True,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return seeds


def main() -> None:
    split = json.loads(SPLIT.read_text())
    loader = FlowMatchingDataLoader(DATA)
    loader.load_data()
    assert loader.sequences is not None
    train_raw = [loader.sequences[i] for i in split["train"]]
    train_stats = FlowMatchingDataset(
        train_raw, max_length=cfg_mod.cfg.max_length, normalize=True
    ).sequence_stats
    catalog = catalog_training(train_raw)
    print(
        f"projected outside {catalog['n_projected_knots_outside']} "
        f"raw layers outside {catalog['n_raw_layers_outside']}",
        flush=True,
    )
    corr = correlation_check(train_raw)
    print(f"corr loss requires_grad {corr['loss_on_data_requires_grad']}", flush=True)
    fit = fit_trend(train_raw)
    sampled = sample_violations(train_raw, train_stats, fit)
    payload = {
        "split": str(SPLIT),
        "train_index_sum": int(sum(split["train"])),
        "training": {
            k: v
            for k, v in catalog.items()
            if k not in ("projected_knots", "raw_layers_outside")
        },
        "projected_knots": catalog["projected_knots"],
        "raw_layer_summary": {
            "n_low": int(
                sum(row["side"] == "low" for row in catalog["raw_layers_outside"])
            ),
            "n_high": int(
                sum(row["side"] == "high" for row in catalog["raw_layers_outside"])
            ),
            "depths_m": sorted(
                {round(row["z0"], 1) for row in catalog["raw_layers_outside"]}
            ),
        },
        "correlation": corr,
        "sampler": sampled,
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
