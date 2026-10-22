"""Seed-locked metrics for the flagship baseline.

Does not change training, the loss, or the samplers. Euler integration matches
``train.sample_sequences``. RK4 calls ``utils.sample_sequences_ode``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats as scipy_stats

from experiments.flow_matching_simplified import config as cfg_mod
from experiments.flow_matching_simplified.comprehensive_eval import (
    check_final_depth_physical,
    check_monotonicity,
    check_vs_constraints,
    compute_breakpoint_count,
    compute_final_depth,
    compute_pairwise_distance,
    compute_total_tts,
    load_checkpoint,
)
from experiments.flow_matching_simplified.data import (
    FlowMatchingDataLoader,
    FlowMatchingDataset,
)
from experiments.flow_matching_simplified.depth_dependent_stats.depth_statistics import (
    bin_by_depth,
)
from experiments.flow_matching_simplified.depth_dependent_stats.vertical_correlation import (
    compute_vertical_autocorrelation,
)
from experiments.flow_matching_simplified.train import set_seed
from experiments.flow_matching_simplified.utils import sample_sequences_ode

SEED = 42
N_SAMPLES = 200
STEP_COUNTS = (5, 10, 20, 50)
LAGS = (1.0, 2.0)


def _denorm_list(dataset: FlowMatchingDataset) -> list[np.ndarray]:
    out = []
    for i, seq in enumerate(dataset.sequences):
        if dataset.normalize:
            out.append(dataset.denormalize_sequence(seq, i))
        else:
            out.append(np.asarray(seq, dtype=np.float64))
    return out


def seed42_split(n_total: int) -> dict[str, list[int]]:
    """Match ``train_with_depth_losses.main`` after ``set_seed(42)``."""
    set_seed(SEED)
    all_indices = torch.randperm(n_total)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    return {
        "train": all_indices[:n_train].tolist(),
        "val": all_indices[n_train : n_train + n_val].tolist(),
        "test": all_indices[n_train + n_val :].tolist(),
    }


def load_recorded_split(path: Path, n_total: int) -> dict[str, list[int]]:
    if path.is_file():
        obj = json.loads(path.read_text())
        if all(k in obj for k in ("train", "val", "test")):
            return {"train": obj["train"], "val": obj["val"], "test": obj["test"]}
    return seed42_split(n_total)


def _layer_vs(seq: np.ndarray, passeri: bool) -> tuple[np.ndarray, np.ndarray]:
    """Layer-center depths and Vs. ``passeri=False`` is ``dz/dt`` on the stored series."""
    if len(seq) == 0:
        return np.array([]), np.array([])
    if seq[0, 0] != 0.0 or seq[0, 1] != 0.0:
        seq = np.vstack([np.zeros((1, 2)), seq])
    tts = seq[:, 0].astype(np.float64)
    if passeri:
        # Stored TTS is 2 * h / Vs. Passeri TTS is half of that.
        tts = tts / 2.0
    dz = np.diff(seq[:, 1].astype(np.float64))
    dt = np.diff(tts)
    vs = dz / (dt + 1e-9)
    centers = 0.5 * (seq[:-1, 1] + seq[1:, 1])
    ok = np.isfinite(vs) & (vs > 0) & np.isfinite(centers)
    return centers[ok], vs[ok]


def _bin_stats(
    sequences: list[np.ndarray], passeri: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depths: list[float] = []
    vs: list[float] = []
    for seq in sequences:
        c, v = _layer_vs(seq, passeri=passeri)
        depths.extend(c.tolist())
        vs.extend(v.tolist())
    edges = np.arange(0.0, 500.0 + 10.0, 10.0)
    if not vs:
        nan = np.full(len(edges) - 1, np.nan)
        return nan, nan, np.zeros(len(edges) - 1, dtype=np.int64)
    _, mean_ln, std_ln, counts = bin_by_depth(
        np.asarray(depths), np.asarray(vs), edges, apply_log=True
    )
    return mean_ln, std_ln, counts


def _rmse(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)))


def _mean_corr(
    sequences: list[np.ndarray], passeri: bool, lags: tuple[float, ...]
) -> dict[str, float]:
    acc = {lag: [] for lag in lags}
    for seq in sequences:
        c, v = _layer_vs(seq, passeri=passeri)
        if len(v) < 2:
            continue
        corrs = compute_vertical_autocorrelation(c, v, lags=list(lags))
        for lag in lags:
            val = corrs.get(lag, np.nan)
            if val is not None and np.isfinite(val):
                acc[lag].append(float(val))
    return {
        str(lag): float(np.mean(acc[lag])) if acc[lag] else float("nan") for lag in lags
    }


def _grid_corr(sequences: list[np.ndarray], passeri: bool, lag_m: float) -> float:
    """Population correlation of piecewise-constant ln(Vs) at a fixed depth lag."""
    xs: list[float] = []
    ys: list[float] = []
    for seq in sequences:
        c, v = _layer_vs(seq, passeri=passeri)
        if len(v) < 1:
            continue
        order = np.argsort(c)
        c, v = c[order], v[order]
        z0 = 0.0
        z1 = float(np.max(c) + 1.0)
        if z1 - lag_m <= z0:
            continue
        grid = np.arange(z0, z1 - lag_m + 1e-9, 1.0)
        if len(grid) < 2:
            continue
        # Step profile: Vs of the layer whose center is nearest, restricted to brackets.
        ln = np.interp(grid, c, np.log(v), left=np.log(v[0]), right=np.log(v[-1]))
        ln_lag = np.interp(
            grid + lag_m, c, np.log(v), left=np.log(v[0]), right=np.log(v[-1])
        )
        xs.extend(ln.tolist())
        ys.extend(ln_lag.tolist())
    if len(xs) < 3:
        return float("nan")
    x = np.asarray(xs)
    y = np.asarray(ys)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom < 1e-12:
        return float("nan")
    return float(np.sum(x * y) / denom)


def _ks(real: np.ndarray, gen: np.ndarray) -> dict[str, float]:
    real = real[np.isfinite(real)]
    gen = gen[np.isfinite(gen)]
    if len(real) == 0 or len(gen) == 0:
        return {"statistic": float("nan"), "pvalue": float("nan")}
    res = scipy_stats.ks_2samp(real, gen)
    return {"statistic": float(res.statistic), "pvalue": float(res.pvalue)}


def metric_bundle(
    real: list[np.ndarray], generated: list[np.ndarray]
) -> dict[str, float | dict]:
    out: dict[str, float | dict] = {}
    out["monotonic_depth_rate"] = check_monotonicity(generated)["monotonic_rate"]
    out["final_depth"] = check_final_depth_physical(generated)
    out["pairwise_distance"] = compute_pairwise_distance(generated)
    out["breakpoint_count_ks"] = _ks(
        compute_breakpoint_count(real), compute_breakpoint_count(generated)
    )
    out["final_depth_ks"] = _ks(
        compute_final_depth(real), compute_final_depth(generated)
    )
    out["total_tts_ks"] = _ks(compute_total_tts(real), compute_total_tts(generated))
    real_depths = np.concatenate([seq[:, 1] for seq in real if len(seq)])
    gen_depths = np.concatenate([seq[:, 1] for seq in generated if len(seq)])
    out["breakpoint_depth_ks"] = _ks(real_depths, gen_depths)

    for name, passeri in (("code_inversion", False), ("passeri", True)):
        r_mean, r_std, r_count = _bin_stats(real, passeri=passeri)
        g_mean, g_std, _g_count = _bin_stats(generated, passeri=passeri)
        mask = (
            np.isfinite(r_mean)
            & np.isfinite(g_mean)
            & np.isfinite(r_std)
            & np.isfinite(g_std)
        )
        mask_train = mask & (r_count >= 50)
        bounds = check_vs_constraints(
            [
                (np.column_stack([seq[:, 0] / 2.0, seq[:, 1]]) if passeri else seq)
                for seq in generated
            ]
        )
        real_vs = np.concatenate([_layer_vs(seq, passeri=passeri)[1] for seq in real])
        gen_vs = np.concatenate(
            [_layer_vs(seq, passeri=passeri)[1] for seq in generated]
        )
        r_corr = _mean_corr(real, passeri, LAGS)
        g_corr = _mean_corr(generated, passeri, LAGS)
        block = {
            "depth_mean_ln_rmse_bins_ge_50": _rmse(r_mean, g_mean, mask_train),
            "depth_std_ln_rmse_bins_ge_50": _rmse(r_std, g_std, mask_train),
            "n_bins_ge_50": int(mask_train.sum()),
            "vs_bounds_rate": bounds["vs_bounds_rate"],
            "vs_ks": _ks(real_vs, gen_vs),
            "layer_center_corr_real": r_corr,
            "layer_center_corr_gen": g_corr,
            "layer_center_corr_abs_err_1m": abs(r_corr["1.0"] - g_corr["1.0"])
            if np.isfinite(r_corr["1.0"]) and np.isfinite(g_corr["1.0"])
            else float("nan"),
            "layer_center_corr_abs_err_2m": abs(r_corr["2.0"] - g_corr["2.0"])
            if np.isfinite(r_corr["2.0"]) and np.isfinite(g_corr["2.0"])
            else float("nan"),
            "grid_corr_real_1m": _grid_corr(real, passeri, 1.0),
            "grid_corr_gen_1m": _grid_corr(generated, passeri, 1.0),
            "grid_corr_real_2m": _grid_corr(real, passeri, 2.0),
            "grid_corr_gen_2m": _grid_corr(generated, passeri, 2.0),
        }
        block["grid_corr_abs_err_1m"] = abs(
            block["grid_corr_real_1m"] - block["grid_corr_gen_1m"]
        )
        block["grid_corr_abs_err_2m"] = abs(
            block["grid_corr_real_2m"] - block["grid_corr_gen_2m"]
        )
        out[name] = block
    return out


def _initial_state(
    dataset: FlowMatchingDataset,
    num_samples: int,
    max_length: int,
    device: torch.device,
    seed: int,
):
    """Same draws as ``train.sample_sequences`` lines 309–352."""
    set_seed(seed)
    train_lengths = [len(seq) for seq in dataset.sequences]
    sampled_lengths = np.random.choice(train_lengths, size=num_samples, replace=True)
    u = torch.randn(num_samples, max_length, 2, device=device)
    attention_masks = torch.zeros(
        num_samples, max_length, dtype=torch.bool, device=device
    )
    sequence_stats_list = []
    for i, seq_len in enumerate(sampled_lengths):
        attention_masks[i, : int(seq_len)] = True
        stats_idx = np.random.randint(0, len(dataset.sequence_stats))
        sequence_stats_list.append(dataset.sequence_stats[stats_idx])
    stats_array = np.array(
        [
            [s["ts_mean"], s["ts_std"], s["depth_mean"], s["depth_std"]]
            for s in sequence_stats_list
        ],
        dtype=np.float32,
    )
    sequence_stats = torch.as_tensor(stats_array, device=device)
    u = u * attention_masks.unsqueeze(-1).float()
    return u, attention_masks, sequence_stats, sequence_stats_list, sampled_lengths


def _denorm_generated(
    u: torch.Tensor,
    lengths: np.ndarray,
    stats_list: list[dict[str, float]],
) -> list[np.ndarray]:
    out = []
    for i, seq_len in enumerate(lengths):
        seq = u[i, : int(seq_len), :].detach().cpu().numpy()
        stats = stats_list[i]
        ts = seq[:, 0] * stats["ts_std"] + stats["ts_mean"]
        depth = seq[:, 1] * stats["depth_std"] + stats["depth_mean"]
        out.append(np.column_stack([ts, depth]))
    return out


def euler_integrate(
    model,
    u0: torch.Tensor,
    attention_mask: torch.Tensor,
    sequence_stats: torch.Tensor,
    ode_steps: int,
) -> torch.Tensor:
    """Euler loop from ``train.sample_sequences`` (lines 354–369)."""
    model.eval()
    u = u0.clone()
    dt = 1.0 / ode_steps
    with torch.no_grad():
        for step in range(ode_steps):
            t_val = step * dt
            t = torch.full((u.shape[0], 1), t_val, device=u.device)
            v = model(u, attention_mask, t, sequence_stats)
            u = u + v * dt
            u = u * attention_mask.unsqueeze(-1).float()
    return u


def curvature(
    model,
    u0: torch.Tensor,
    u1: torch.Tensor,
    attention_mask: torch.Tensor,
    sequence_stats: torch.Tensor,
    n_times: int = 16,
) -> float:
    """App. A.3 curvature on the straight chord from this run's noise to its sample."""
    target = u1 - u0
    acc = []
    with torch.no_grad():
        for k in range(n_times):
            t_val = k / n_times
            t = torch.full((u0.shape[0], 1), t_val, device=u0.device)
            xt = (1.0 - t_val) * u0 + t_val * u1
            xt = xt * attention_mask.unsqueeze(-1).float()
            v = model(xt, attention_mask, t, sequence_stats)
            err = (target - v) ** 2
            mask = attention_mask.unsqueeze(-1).float()
            acc.append(float((err * mask).sum() / mask.sum().clamp(min=1.0)))
    return float(np.mean(acc))


def _fit_bins(
    train_seqs: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    centers_all = []
    ln_all = []
    for seq in train_seqs:
        c, v = _layer_vs(seq, passeri=False)
        if len(v):
            centers_all.append(c)
            ln_all.append(np.log(v))
    centers = np.concatenate(centers_all)
    ln = np.concatenate(ln_all)
    edges = np.arange(0.0, 510.0, 10.0)
    means = np.full(len(edges) - 1, np.nan)
    stds = np.full(len(edges) - 1, np.nan)
    for i in range(len(edges) - 1):
        m = (centers >= edges[i]) & (centers < edges[i + 1])
        if m.sum() >= 5:
            means[i] = float(np.mean(ln[m]))
            stds[i] = float(np.std(ln[m]) + 1e-3)
    global_mean = float(np.mean(ln))
    global_std = float(np.std(ln) + 1e-3)
    means = np.where(np.isfinite(means), means, global_mean)
    stds = np.where(np.isfinite(stds), stds, global_std)
    rho = _grid_corr(train_seqs, passeri=False, lag_m=1.0)
    if not np.isfinite(rho) or rho <= 0.0 or rho >= 1.0:
        length = 5.0
    else:
        length = float(-1.0 / np.log(rho))
    return means, stds, edges, length


def sample_fixed_grid_gaussian(
    train_seqs: list[np.ndarray], num_samples: int, seed: int
) -> list[np.ndarray]:
    """Bootstrap train depth knots. Draw ln(Vs) from 10 m bins with exp correlation.

    Layer boundaries are copied from a random training profile. This baseline
    generates velocities, not layering, so it is not like-for-like with a model
    that generates its own boundaries on thickness-dependent metrics.
    """
    means, stds, edges, length = _fit_bins(train_seqs)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(num_samples):
        template = train_seqs[int(rng.integers(0, len(train_seqs)))]
        depths = np.asarray(template[:, 1], dtype=np.float64)
        if depths[0] != 0.0:
            depths = np.concatenate([[0.0], depths])
        dz = np.diff(depths)
        if len(dz) == 0:
            continue
        centers = depths[:-1] + 0.5 * dz
        idx = np.clip(np.digitize(centers, edges) - 1, 0, len(means) - 1)
        mu = means[idx]
        sig = stds[idx]
        dist = np.abs(centers[:, None] - centers[None, :])
        cov = np.exp(-dist / length)
        cov.flat[:: cov.shape[0] + 1] += 1e-6
        z = rng.multivariate_normal(np.zeros(len(centers)), cov)
        ln_vs = mu + sig * z
        vs = np.exp(ln_vs)
        tts = np.concatenate([[0.0], np.cumsum(dz / vs)])
        seq = np.column_stack([tts[1:], depths[1:]])
        out.append(seq.astype(np.float64))
    return out


def main() -> None:
    cfg = cfg_mod.cfg
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_path = Path(cfg.results_dir) / "split_indices_seed42.json"
    ckpt = Path(cfg.checkpoints_dir) / "latest.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)

    loader = FlowMatchingDataLoader(Path(cfg.data_path))
    loader.load_data()
    assert loader.sequences is not None
    split = load_recorded_split(split_path, len(loader.sequences))
    train_ds = FlowMatchingDataset(
        [loader.sequences[i] for i in split["train"]],
        max_length=cfg.max_length,
        pad_token=cfg.pad_token,
        normalize=True,
    )
    test_ds = FlowMatchingDataset(
        [loader.sequences[i] for i in split["test"]],
        max_length=cfg.max_length,
        pad_token=cfg.pad_token,
        normalize=True,
    )
    real = _denorm_list(test_ds)
    train_raw = _denorm_list(train_ds)

    model, epoch = load_checkpoint(str(ckpt), device, cfg)
    model.eval()

    u0, masks, stats, stats_list, lengths = _initial_state(
        train_ds, N_SAMPLES, cfg.max_length, device, SEED
    )
    results: dict = {
        "seed": SEED,
        "n_samples": N_SAMPLES,
        "checkpoint": str(ckpt),
        "epoch": int(epoch),
        "device": str(device),
        "torch": torch.__version__,
        "split_source": str(
            split_path if split_path.is_file() else "recomputed set_seed(42)"
        ),
        "n_train": len(split["train"]),
        "n_test": len(split["test"]),
        "sweeps": {},
    }

    t_curve = time.perf_counter()
    u_ref = euler_integrate(model, u0, masks, stats, 50)
    results["curvature_euler_chord_50"] = curvature(model, u0, u_ref, masks, stats)
    results["curvature_seconds"] = time.perf_counter() - t_curve

    for kind in ("euler", "rk4"):
        for steps in STEP_COUNTS:
            t0 = time.perf_counter()
            if kind == "euler":
                u = euler_integrate(model, u0, masks, stats, steps)
            else:
                u = sample_sequences_ode(model, u0.clone(), masks, stats, steps, device)
            gen = _denorm_generated(u, lengths, stats_list)
            bundle = metric_bundle(real, gen)
            bundle["seconds"] = time.perf_counter() - t0
            results["sweeps"][f"{kind}_{steps}"] = bundle
            code = bundle["code_inversion"]
            print(
                f"{kind} steps={steps} "
                f"mean_rmse={code['depth_mean_ln_rmse_bins_ge_50']:.4f} "
                f"std_rmse={code['depth_std_ln_rmse_bins_ge_50']:.4f} "
                f"grid_corr_err_1m={code['grid_corr_abs_err_1m']:.4f} "
                f"vs_bounds={code['vs_bounds_rate']:.3f} "
                f"depth_ks={bundle['breakpoint_depth_ks']['statistic']:.3f} "
                f"sec={bundle['seconds']:.1f}"
            )

    t0 = time.perf_counter()
    gauss = sample_fixed_grid_gaussian(train_raw, N_SAMPLES, SEED)
    gbundle = metric_bundle(real, gauss)
    gbundle["seconds"] = time.perf_counter() - t0
    gbundle["correlation_length_m"] = _fit_bins(train_raw)[3]
    results["fixed_grid_gaussian"] = gbundle
    print(
        "gaussian",
        gbundle["code_inversion"]["depth_mean_ln_rmse_bins_ge_50"],
        "sec",
        gbundle["seconds"],
    )

    out_path = Path(cfg.results_dir) / "baseline_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
