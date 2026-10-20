"""Score best.pt and latest.pt with the corrected evaluator.

Does not train and does not modify the sampling or training modules.
"""

from __future__ import annotations

import json
import platform
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats as scipy_stats

from experiments.flow_matching_simplified import config as cfg_mod
from experiments.flow_matching_simplified.baseline_metrics import (
    _denorm_generated,
    _initial_state,
    euler_integrate,
    sample_fixed_grid_gaussian,
)
from experiments.flow_matching_simplified.comprehensive_eval import (
    check_monotonicity,
    compute_breakpoint_count,
    compute_final_depth,
    compute_pairwise_distance,
    load_checkpoint,
)
from experiments.flow_matching_simplified.data import FlowMatchingDataLoader, FlowMatchingDataset
from experiments.flow_matching_simplified.eval_corrected import (
    bound_rate,
    bootstrap_vertical,
    catalog_convention_report,
    depth_trend_error,
)
from experiments.flow_matching_simplified.utils import sample_sequences_ode

SEED = 42
N_SAMPLES = 200
N_BOOT = 400
STEP_COUNTS = (5, 10, 20, 50)
N_CURVATURE = 10_000
CURVATURE_STEPS = 128


def _denorm_split(loader, indices) -> list[np.ndarray]:
    ds = FlowMatchingDataset(
        [loader.sequences[i] for i in indices],
        max_length=cfg_mod.cfg.max_length,
        normalize=True,
    )
    out = []
    for i, seq in enumerate(ds.sequences):
        out.append(ds.denormalize_sequence(seq, i))
    return out, ds


def _pairwise(sequences: list[np.ndarray]) -> dict[str, float]:
    """Mean pairwise distance. Non-monotonic profiles are excluded and counted.

    ``interp1d`` requires increasing depth. Dropping those profiles keeps the
    distance defined. The count is reported so the exclusion is visible.
    """
    kept = []
    dropped = 0
    for seq in sequences:
        if len(seq) < 2 or np.all(np.diff(seq[:, 1]) >= -1e-8):
            kept.append(seq)
        else:
            dropped += 1
    dist = compute_pairwise_distance(kept)
    return {
        "mean_distance": dist["mean_distance"],
        "n_profiles": len(kept),
        "n_dropped_nonmonotonic": dropped,
    }


def _ks(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return {"statistic": float("nan"), "pvalue": float("nan")}
    res = scipy_stats.ks_2samp(a, b)
    return {"statistic": float(res.statistic), "pvalue": float(res.pvalue)}


def _corr_gap(real_boot, generated) -> dict:
    gen_boot = bootstrap_vertical(generated, n_boot=N_BOOT, seed=SEED)
    out = {}
    for lag, gest in gen_boot.items():
        rest = real_boot[lag]
        point = abs(rest["ln_correlation"] - gest["ln_correlation"])
        out[lag] = {
            "real_correlation": rest["ln_correlation"],
            "real_lo": rest["ln_correlation_lo"],
            "real_hi": rest["ln_correlation_hi"],
            "real_same_layer_fraction": rest["same_layer_fraction"],
            "real_n_pairs": rest["n_pairs"],
            "gen_correlation": gest["ln_correlation"],
            "gen_lo": gest["ln_correlation_lo"],
            "gen_hi": gest["ln_correlation_hi"],
            "gen_same_layer_fraction": gest["same_layer_fraction"],
            "gen_n_pairs": gest["n_pairs"],
            "abs_correlation_gap": point,
        }
    return out


def _score_set(real, generated, real_boot, histogram_lengths) -> dict:
    counts = compute_breakpoint_count(generated)
    if histogram_lengths is None:
        histogram_match = None
    else:
        histogram_match = bool(
            np.array_equal(counts.astype(int), np.asarray(histogram_lengths).astype(int))
        )
    return {
        "depth_monotonicity": check_monotonicity(generated)["monotonic_rate"],
        "bound_rate": bound_rate(generated),
        "final_depth_mean": float(np.mean(compute_final_depth(generated))),
        "final_depth_ks": _ks(compute_final_depth(real), compute_final_depth(generated)),
        "vs_ks": _vs_ks(real, generated),
        "breakpoint_depth_ks": _depth_ks(real, generated),
        "breakpoint_count_ks": _ks(
            compute_breakpoint_count(real), counts
        ),
        "breakpoint_count_equals_histogram_draw": histogram_match,
        "pairwise_distance_generated": _pairwise(generated),
        "depth_trend": depth_trend_error(real, generated, n_boot=N_BOOT, seed=SEED),
        "vertical": _corr_gap(real_boot, generated),
    }


def _vs_values(sequences):
    from experiments.flow_matching_simplified.eval_corrected import vs_from_travel_time

    chunks = []
    for seq in sequences:
        _, _, vs = vs_from_travel_time(seq)
        vs = vs[np.isfinite(vs)]
        if len(vs):
            chunks.append(vs)
    return np.concatenate(chunks) if chunks else np.zeros(0)


def _vs_ks(real, generated):
    return _ks(_vs_values(real), _vs_values(generated))


def _depth_ks(real, generated):
    rd = np.concatenate([seq[:, 1] for seq in real if len(seq)])
    gd = np.concatenate([seq[:, 1] for seq in generated if len(seq)])
    return _ks(rd, gd)


def paper_curvature_clean(model, dataset, device) -> dict[str, float]:
    """App. A.3, verified from full text: 10,000 generated trajectories, 128 Euler steps.

    Residual is ``v_theta(x_t, t) - (x_1 - x_0)``, with ``x_1`` the Euler endpoint
    and padding tokens excluded. Both token features are in the mean.
    """
    t0 = time.perf_counter()
    remaining = N_CURVATURE
    batch = 64
    acc = 0.0
    count = 0.0
    done = 0
    cursor = 0
    while remaining > 0:
        n = min(batch, remaining)
        u0, masks, stats, _, _ = _initial_state(
            dataset, n, dataset.max_length, device, seed=SEED + 1000 + cursor
        )
        cursor += 1
        u = u0.clone()
        velocities = []
        dt = 1.0 / CURVATURE_STEPS
        with torch.no_grad():
            for step in range(CURVATURE_STEPS):
                t = torch.full((n, 1), step * dt, device=device)
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
        done += n
        if done % 2048 == 0 or remaining == 0:
            print(f"  curvature {done}/{N_CURVATURE}", flush=True)
    return {
        "protocol": "10000 generated trajectories, 128 Euler steps, chord residual",
        "source": "arXiv:2606.04092 App. A.3, verified from full text",
        "curvature": acc / count if count else float("nan"),
        "n_samples": N_CURVATURE,
        "euler_steps": CURVATURE_STEPS,
        "seconds": time.perf_counter() - t0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--breakpoints", default=None)
    parser.add_argument("--tts", default="data/vspdb_tts_profiles.parquet")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--split", default=None)
    args = parser.parse_args()
    cfg = cfg_mod.cfg
    bp_path = Path(args.breakpoints) if args.breakpoints else Path(cfg.data_path)
    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else Path(cfg.checkpoints_dir)
    split_path = (
        Path(args.split)
        if args.split
        else Path(cfg.results_dir, "split_indices_seed42.json")
    )
    out = Path(args.output) if args.output else Path(cfg.results_dir) / "corrected_eval.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tts = pd.read_parquet(Path(args.tts))
    bp = pd.read_parquet(bp_path)
    convention = catalog_convention_report(tts, bp)

    split = json.loads(split_path.read_text())
    loader = FlowMatchingDataLoader(bp_path)
    loader.load_data()
    real, _ = _denorm_split(loader, split["test"])
    train_raw, train_ds = _denorm_split(loader, split["train"])
    real_bounds = bound_rate(real)
    real_vert = bootstrap_vertical(real, n_boot=N_BOOT, seed=SEED)
    real_distance = _pairwise(real)

    u0, masks, stats, stats_list, lengths = _initial_state(
        train_ds, N_SAMPLES, cfg.max_length, device, SEED
    )
    results = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "pyproject_pin": "torch==2.9.1",
        },
        "breakpoints": str(bp_path),
        "tts": str(args.tts),
        "split": str(split_path),
        "convention": convention,
        "seed": SEED,
        "n_test": len(real),
        "n_samples": N_SAMPLES,
        "real": {
            "bound_rate": real_bounds,
            "pairwise_distance": real_distance,
            "final_depth_mean": float(np.mean(compute_final_depth(real))),
            "vertical": real_vert,
            "depth_trend_n_real": depth_trend_error(real, real, n_boot=20, seed=SEED)["n_real"],
        },
        "checkpoints": {},
    }
    print(
        f"real bounds passeri={real_bounds:.4f}",
        flush=True,
    )

    for name in ("latest.pt", "best.pt"):
        ckpt = ckpt_dir / name
        model, epoch = load_checkpoint(str(ckpt), device, cfg)
        model.eval()
        block = {"epoch_index": int(epoch), "sweeps": {}}
        for kind in ("euler", "rk4"):
            for steps in STEP_COUNTS:
                t0 = time.perf_counter()
                if kind == "euler":
                    u = euler_integrate(model, u0, masks, stats, steps)
                else:
                    u = sample_sequences_ode(model, u0.clone(), masks, stats, steps, device)
                gen = _denorm_generated(u, lengths, stats_list)
                scored = _score_set(real, gen, real_vert, lengths)
                scored["seconds"] = time.perf_counter() - t0
                block["sweeps"][f"{kind}_{steps}"] = scored
                trend = scored["depth_trend"]["weighted_mean_ln_rmse"]
                print(
                    f"{name} {kind} {steps} trend={trend:.4f} "
                    f"bounds={scored['bound_rate']:.3f} "
                    f"mono={scored['depth_monotonicity']:.3f} sec={scored['seconds']:.1f}",
                    flush=True,
                )
        print(f"{name} curvature", flush=True)
        block["curvature"] = paper_curvature_clean(model, train_ds, device)
        results["checkpoints"][name] = block
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    t0 = time.perf_counter()
    gauss = sample_fixed_grid_gaussian(train_raw, N_SAMPLES, SEED)
    gscore = _score_set(real, gauss, real_vert, None)
    gscore["seconds"] = time.perf_counter() - t0
    gscore["note"] = (
        "Existing fixed-grid Gaussian from baseline_metrics.py. "
        "It draws depth knots from the training profiles. "
        "Scored with Vs = Δz / Δt. "
        "No correlation length is reported."
    )
    results["gaussian"] = gscore
    print(
        f"gaussian trend={gscore['depth_trend']['weighted_mean_ln_rmse']:.4f} "
        f"bounds={gscore['bound_rate']:.3f}",
        flush=True,
    )

    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
