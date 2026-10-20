"""Corrected evaluation for travel-time profiles.

The training path is unchanged. This module only reads checkpoints and data.

Passeri one-way travel time is ``tts = Σ h / Vs``, so the inversion is
``Vs = Δz / Δt``. A factor of 2 is not applied to ``Δz``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

VS_MIN = 100.0
VS_MAX = 5000.0
# Pre-specified. Matches the training bin width. Not chosen on the test split.
TREND_DEPTHS_M = np.arange(10.0, 210.0, 10.0)
LAGS_M = (1.0, 2.0)
PAIR_STEP_M = 0.5


def travel_time_from_vs(depths: np.ndarray, vs: np.ndarray) -> np.ndarray:
    """Build breakpoint tokens with Passeri one-way time ``Δt = h / Vs``.

    ``depths`` has one more entry than ``vs`` (layer boundaries, surface first).
    Returns ``[tts, depth]`` at the interior and final boundaries, origin omitted.
    """
    depths = np.asarray(depths, dtype=np.float64)
    vs = np.asarray(vs, dtype=np.float64)
    if len(depths) != len(vs) + 1:
        raise ValueError("depths must be layer boundaries, one longer than vs")
    if np.any(vs <= 0):
        raise ValueError("Vs must be positive")
    thickness = np.diff(depths)
    if np.any(thickness <= 0):
        raise ValueError("depths must be strictly increasing")
    tts = np.cumsum(thickness / vs)
    return np.column_stack([tts, depths[1:]])


def vs_from_travel_time(sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Invert a ``[tts, depth]`` sequence with ``Vs = Δz / Δt``.

    Returns layer-start depths, layer-end depths, and Vs.
    """
    sequence = np.asarray(sequence, dtype=np.float64)
    if len(sequence) == 0:
        empty = np.zeros(0)
        return empty, empty, empty
    if sequence[0, 0] != 0.0 or sequence[0, 1] != 0.0:
        sequence = np.vstack([np.zeros((1, 2)), sequence])
    thickness = np.diff(sequence[:, 1])
    increment = np.diff(sequence[:, 0])
    vs = thickness / (increment + 1e-15)
    return sequence[:-1, 1], sequence[1:, 1], vs


def profile_in_bounds(sequence: np.ndarray) -> bool:
    _, _, vs = vs_from_travel_time(sequence)
    if len(vs) == 0:
        return False
    return bool(np.all(np.isfinite(vs) & (vs >= VS_MIN) & (vs <= VS_MAX)))


def bound_rate(sequences: list[np.ndarray]) -> float:
    if not sequences:
        return float("nan")
    return float(np.mean([profile_in_bounds(seq) for seq in sequences]))


def _layer_index_at(boundaries: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Layer index for depths in ``(boundaries[i], boundaries[i+1]]``."""
    # searchsorted side='left' on interior boundaries gives the layer whose end is the next boundary
    idx = np.searchsorted(boundaries, z, side="left") - 1
    return idx


def lag_pairs(sequence: np.ndarray, lag_m: float, step_m: float = PAIR_STEP_M) -> dict[str, np.ndarray]:
    """Pairs ``(z, z+lag)`` on a fixed depth grid inside one piecewise-constant profile.

    Vs is the Passeri inversion ``Δz / Δt``. Pairs in a profile whose depth is not
    strictly increasing are dropped.
    """
    z0, z1, vs = vs_from_travel_time(sequence)
    empty = {
        "ln0": np.zeros(0),
        "ln1": np.zeros(0),
        "same": np.zeros(0, dtype=bool),
    }
    if len(vs) == 0 or np.any(np.diff(np.concatenate([[z0[0]], z1])) <= 0):
        return empty
    if np.any(~np.isfinite(vs) | (vs <= 0)):
        return empty
    z_end = float(z1[-1])
    if z_end <= lag_m:
        return empty
    grid = np.arange(0.0, z_end - lag_m + 1e-9, step_m)
    if len(grid) == 0:
        return empty
    boundaries = np.concatenate([z0[:1], z1])
    i0 = _layer_index_at(boundaries, grid)
    i1 = _layer_index_at(boundaries, grid + lag_m)
    valid = (i0 >= 0) & (i1 >= 0) & (i0 < len(vs)) & (i1 < len(vs))
    if not np.any(valid):
        return empty
    ln = np.log(vs)
    return {
        "ln0": ln[i0[valid]],
        "ln1": ln[i1[valid]],
        "same": i0[valid] == i1[valid],
    }


@dataclass
class LagEstimate:
    lag_m: float
    n_pairs: int
    same_layer_fraction: float
    ln_correlation: float


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom < 1e-15:
        return float("nan")
    return float(np.sum(x * y) / denom)


def vertical_structure(
    sequences: list[np.ndarray], lags: tuple[float, ...] = LAGS_M
) -> list[LagEstimate]:
    """Same-layer fraction and correlation of ``ln(Vs)`` at lag ``h``.

    Pairs are pooled across profiles. Same-layer fraction is the fraction of
    depth pairs at lag ``h`` that sit in one piecewise-constant layer. The
    correlation is the Pearson correlation of the paired ``ln(Vs)`` values,
    including pairs that fall in different layers.
    """
    cached = {lag: [lag_pairs(seq, lag) for seq in sequences] for lag in lags}
    out = []
    for lag in lags:
        parts = cached[lag]
        if not any(len(p["ln0"]) for p in parts):
            out.append(LagEstimate(lag, 0, float("nan"), float("nan")))
            continue
        ln0 = np.concatenate([p["ln0"] for p in parts if len(p["ln0"])])
        ln1 = np.concatenate([p["ln1"] for p in parts if len(p["ln1"])])
        same = np.concatenate([p["same"] for p in parts if len(p["same"])])
        out.append(
            LagEstimate(
                lag_m=lag,
                n_pairs=int(len(ln0)),
                same_layer_fraction=float(np.mean(same)),
                ln_correlation=_corr(ln0, ln1),
            )
        )
    return out


def bootstrap_vertical(
    sequences: list[np.ndarray],
    lags: tuple[float, ...] = LAGS_M,
    n_boot: int = 400,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Profile-resample intervals for same-layer fraction and ``ln(Vs)`` correlation."""
    rng = np.random.default_rng(seed)
    n = len(sequences)
    point = vertical_structure(sequences, lags)
    bags = {
        lag: {"same": [], "corr": []} for lag in lags
    }
    pair_cache = {lag: [lag_pairs(seq, lag) for seq in sequences] for lag in lags}
    for _ in range(n_boot):
        draw = rng.integers(0, n, size=n)
        for lag in lags:
            chosen = [pair_cache[lag][i] for i in draw if len(pair_cache[lag][i]["ln0"])]
            if not chosen:
                bags[lag]["same"].append(np.nan)
                bags[lag]["corr"].append(np.nan)
                continue
            ln0 = np.concatenate([p["ln0"] for p in chosen])
            ln1 = np.concatenate([p["ln1"] for p in chosen])
            same = np.concatenate([p["same"] for p in chosen])
            bags[lag]["same"].append(float(np.mean(same)))
            bags[lag]["corr"].append(_corr(ln0, ln1))
    result = {}
    for est in point:
        same = np.asarray(bags[est.lag_m]["same"], dtype=float)
        corr = np.asarray(bags[est.lag_m]["corr"], dtype=float)
        result[str(est.lag_m)] = {
            "n_pairs": est.n_pairs,
            "same_layer_fraction": est.same_layer_fraction,
            "same_layer_lo": float(np.nanpercentile(same, 2.5)),
            "same_layer_hi": float(np.nanpercentile(same, 97.5)),
            "ln_correlation": est.ln_correlation,
            "ln_correlation_lo": float(np.nanpercentile(corr, 2.5)),
            "ln_correlation_hi": float(np.nanpercentile(corr, 97.5)),
        }
    return result


def ln_vs_matrix(sequences: list[np.ndarray], depths: np.ndarray = TREND_DEPTHS_M) -> np.ndarray:
    """``ln(Vs)`` from ``Vs = Δz / Δt`` at fixed depths. NaN if the profile stops short."""
    mat = np.full((len(sequences), len(depths)), np.nan)
    for i, seq in enumerate(sequences):
        z0, z1, vs = vs_from_travel_time(seq)
        if len(vs) == 0 or np.any(~np.isfinite(vs) | (vs <= 0)):
            continue
        if np.any(np.diff(np.concatenate([[z0[0]], z1])) <= 0):
            continue
        boundaries = np.concatenate([z0[:1], z1])
        ln = np.log(vs)
        for j, z in enumerate(depths):
            if z <= boundaries[0] or z > boundaries[-1]:
                continue
            idx = int(_layer_index_at(boundaries, np.array([z]))[0])
            if 0 <= idx < len(ln):
                mat[i, j] = ln[idx]
    return mat


def depth_trend_error(
    real: list[np.ndarray],
    generated: list[np.ndarray],
    depths: np.ndarray = TREND_DEPTHS_M,
    n_boot: int = 400,
    seed: int = 42,
) -> dict[str, float | list]:
    """Weighted RMSE of the mean ``ln(Vs)`` curve, using every depth that both sets reach.

    Weight at each depth is the number of real profiles that reach it. There is
    no minimum count. Bootstrap resamples profiles.
    """
    real_m = ln_vs_matrix(real, depths)
    gen_m = ln_vs_matrix(generated, depths)
    point, counts = _weighted_rmse(real_m, gen_m)
    rng = np.random.default_rng(seed)
    boots = []
    n_r, n_g = len(real), len(generated)
    for _ in range(n_boot):
        ri = rng.integers(0, n_r, size=n_r)
        gi = rng.integers(0, n_g, size=n_g)
        val, _ = _weighted_rmse(real_m[ri], gen_m[gi])
        boots.append(val)
    boots_a = np.asarray(boots, dtype=float)
    return {
        "weighted_mean_ln_rmse": point,
        "lo": float(np.nanpercentile(boots_a, 2.5)),
        "hi": float(np.nanpercentile(boots_a, 97.5)),
        "depths_m": depths.tolist(),
        "n_real": counts["n_real"],
        "n_gen": counts["n_gen"],
        "n_depths_used": counts["n_depths_used"],
    }


def _weighted_rmse(real_m: np.ndarray, gen_m: np.ndarray) -> tuple[float, dict]:
    n_real = np.sum(np.isfinite(real_m), axis=0).astype(float)
    n_gen = np.sum(np.isfinite(gen_m), axis=0).astype(float)
    with np.errstate(invalid="ignore"):
        mu_r = np.nanmean(real_m, axis=0)
        mu_g = np.nanmean(gen_m, axis=0)
    use = (n_real >= 1) & (n_gen >= 1) & np.isfinite(mu_r) & np.isfinite(mu_g)
    if not np.any(use):
        return float("nan"), {
            "n_real": n_real.tolist(),
            "n_gen": n_gen.tolist(),
            "n_depths_used": 0,
        }
    w = n_real[use]
    err = (mu_r[use] - mu_g[use]) ** 2
    rmse = float(np.sqrt(np.sum(w * err) / np.sum(w)))
    return rmse, {
        "n_real": n_real.tolist(),
        "n_gen": n_gen.tolist(),
        "n_depths_used": int(np.sum(use)),
    }


def catalog_convention_report(tts_frame, breakpoint_frame) -> dict[str, float]:
    """Compare stored increments with ``h/Vs`` and ``2h/Vs`` on the fine grid."""
    ratios = []
    catalog_in = 0
    catalog_n = 0
    profiles_ok = 0
    profiles_n = 0
    for _, group in tts_frame.groupby("velocity_metadata_id", sort=False):
        group = group.sort_values("depth")
        z = group["depth"].to_numpy(float)
        t = group["tts"].to_numpy(float)
        v = group["vs_value"].to_numpy(float)
        h = np.diff(z)
        dt = np.diff(t)
        vv = v[:-1]
        ok = (h > 0) & (vv > 0) & np.isfinite(dt)
        if not np.any(ok):
            continue
        ratios.append(dt[ok] / (h[ok] / vv[ok]))
        catalog_n += int(ok.sum())
        catalog_in += int(np.sum((vv[ok] >= VS_MIN) & (vv[ok] <= VS_MAX)))
        profiles_n += 1
        if np.all((vv[ok] >= VS_MIN) & (vv[ok] <= VS_MAX)):
            profiles_ok += 1
    ratio = np.concatenate(ratios)
    return {
        "fine_layers": int(catalog_n),
        "increment_over_h_over_vs_median": float(np.median(ratio)),
        "fraction_matching_h_over_vs": float(np.mean(np.abs(ratio - 1.0) <= 1e-3)),
        "fraction_matching_2h_over_vs": float(np.mean(np.abs(ratio - 2.0) <= 1e-3)),
        "catalog_layer_bound_rate": float(catalog_in / catalog_n),
        "catalog_profile_bound_rate": float(profiles_ok / profiles_n),
        "breakpoint_rows": int(len(breakpoint_frame)),
    }
