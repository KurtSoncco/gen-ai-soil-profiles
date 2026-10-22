"""Engineering checks on layered Vs profiles, beyond the 0.5 m-grid correlation.

- Adjacent-layer correlation of the trend-removed ln(Vs) residual, the
  quantity Toro (1995) models with ρ(z, t).
- σ_lnVs against depth.
- Layer thickness distribution by depth band (Toro's λ(z) in empirical form).
- Vs30.
- Fundamental frequency f0 from the 1D linear SH transfer function.

By default only valid profiles are scored, and ``validity_rate`` carries the
invalid ones as failures. ``include_invalid`` scores invalid profiles through
``eval_corrected.layer_model`` with clipped Vs. That is a stress check, not a
headline: clipped extremes inflate variance, so pooled means and correlations
can move toward the real values instead of away from them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as scipy_stats
from scipy.spatial.distance import cdist

from experiments.flow_matching_simplified.comprehensive_eval import interpolate_profiles
from experiments.flow_matching_simplified.eval_corrected import (
    TREND_DEPTHS_M,
    layer_model,
    ln_vs_matrix,
)

RESIDUAL_EDGES_M = np.array([0, 5, 10, 20, 30, 50, 75, 100, 150, 200, np.inf])
THICKNESS_BANDS_M = ((0.0, 10.0), (10.0, 30.0), (30.0, 100.0), (100.0, np.inf))
F0_FREQS_HZ = np.logspace(np.log10(0.1), np.log10(25.0), 400)
DAMPING = 0.02


def _layers(
    sequences: list[np.ndarray], include_invalid: bool = False
) -> list[tuple[np.ndarray, np.ndarray]]:
    out = []
    for seq in sequences:
        lm = layer_model(seq, include_invalid=include_invalid)
        if lm is not None:
            out.append(lm)
    return out


@dataclass
class ResidualReference:
    """Median and s.d. of ln(Vs) by depth bin, fit on training layer midpoints."""

    edges: np.ndarray
    median: np.ndarray
    sigma: np.ndarray

    def standardize(self, mid: np.ndarray, ln_vs: np.ndarray) -> np.ndarray:
        j = np.clip(np.digitize(mid, self.edges) - 1, 0, len(self.median) - 1)
        return (ln_vs - self.median[j]) / self.sigma[j]


def fit_residual_reference(
    train: list[np.ndarray], edges: np.ndarray = RESIDUAL_EDGES_M
) -> ResidualReference:
    mids, lns = [], []
    for seq in train:
        lm = layer_model(seq, include_invalid=False)
        if lm is None:
            continue
        b, vs = lm
        mids.append(0.5 * (b[:-1] + b[1:]))
        lns.append(np.log(vs))
    mid = np.concatenate(mids)
    ln = np.concatenate(lns)
    j = np.digitize(mid, edges) - 1
    n_bins = len(edges) - 1
    filled = [k for k in range(n_bins) if np.sum(j == k) >= 2]
    if not filled:
        raise ValueError("no depth bin has two or more training layers")
    median = np.empty(n_bins)
    sigma = np.empty(n_bins)
    for k in range(n_bins):
        src = min(filled, key=lambda f: (abs(f - k), -f))
        median[k] = np.median(ln[j == src])
        sigma[k] = np.std(ln[j == src])
    return ResidualReference(edges=edges, median=median, sigma=sigma)


def residual_correlation(
    sequences: list[np.ndarray],
    ref: ResidualReference,
    include_invalid: bool = False,
) -> dict[str, float]:
    """Correlation of standardized ln(Vs) residuals between adjacent layers.

    Zero-thickness layers (backward depth steps in invalid profiles) are skipped,
    so the pair is the next layers that occupy depth.
    """
    a, b, z_all = [], [], []
    for bounds, vs in _layers(sequences, include_invalid):
        h = np.diff(bounds)
        keep = h > 0
        if not np.any(keep):
            continue
        mid = (0.5 * (bounds[:-1] + bounds[1:]))[keep]
        z = ref.standardize(mid, np.log(vs[keep]))
        z_all.extend(z.tolist())
        a.extend(z[:-1].tolist())
        b.extend(z[1:].tolist())
    if len(a) < 2:
        return {"rho": float("nan"), "residual_sd": float("nan"), "n_pairs": len(a)}
    return {
        "rho": float(np.corrcoef(a, b)[0, 1]),
        "residual_sd": float(np.std(z_all)),
        "n_pairs": len(a),
    }


def sigma_ln_vs_by_depth(
    real: list[np.ndarray],
    generated: list[np.ndarray],
    depths: np.ndarray = TREND_DEPTHS_M,
    min_count: int = 5,
    include_invalid: bool = False,
) -> dict[str, list | float]:
    """Between-profile s.d. of ln(Vs) at fixed depths, and an n-weighted RMSE.

    Depths where either set has fewer than ``min_count`` profiles are left out
    of the RMSE.
    """
    rm = ln_vs_matrix(real, depths, include_invalid=include_invalid)
    gm = ln_vs_matrix(generated, depths, include_invalid=include_invalid)
    n_r = np.sum(np.isfinite(rm), axis=0)
    n_g = np.sum(np.isfinite(gm), axis=0)
    with np.errstate(invalid="ignore"):
        s_r = np.nanstd(rm, axis=0)
        s_g = np.nanstd(gm, axis=0)
    use = (n_r >= min_count) & (n_g >= min_count)
    rmse = (
        float(np.sqrt(np.sum(n_r[use] * (s_r[use] - s_g[use]) ** 2) / np.sum(n_r[use])))
        if np.any(use)
        else float("nan")
    )
    return {
        "depths_m": depths.tolist(),
        "real_sigma": s_r.tolist(),
        "gen_sigma": s_g.tolist(),
        "n_real": n_r.tolist(),
        "n_gen": n_g.tolist(),
        "n_depths_used": int(np.sum(use)),
        "weighted_rmse": rmse,
    }


def _thickness_by_band(
    sequences: list[np.ndarray], include_invalid: bool = False
) -> dict[str, np.ndarray]:
    out = {f"{lo:g}-{hi:g}": [] for lo, hi in THICKNESS_BANDS_M}
    for bounds, _ in _layers(sequences, include_invalid):
        h = np.diff(bounds)
        top = bounds[:-1]
        for lo, hi in THICKNESS_BANDS_M:
            sel = (top >= lo) & (top < hi) & (h > 0)
            out[f"{lo:g}-{hi:g}"].extend(h[sel].tolist())
    return {k: np.asarray(v) for k, v in out.items()}


def thickness_ks(
    real: list[np.ndarray], generated: list[np.ndarray], include_invalid: bool = False
) -> dict:
    """KS distance of positive layer thickness, by depth band of the layer top."""
    r = _thickness_by_band(real, include_invalid)
    g = _thickness_by_band(generated, include_invalid)
    bands = {}
    weighted, weight = 0.0, 0
    for key in r:
        if len(r[key]) and len(g[key]):
            ks = float(scipy_stats.ks_2samp(r[key], g[key]).statistic)
            weighted += ks * len(r[key])
            weight += len(r[key])
        else:
            ks = float("nan")
        bands[key] = {
            "ks": ks,
            "n_real": int(len(r[key])),
            "n_gen": int(len(g[key])),
            "real_median": float(np.median(r[key])) if len(r[key]) else float("nan"),
            "gen_median": float(np.median(g[key])) if len(g[key]) else float("nan"),
        }
    return {
        "bands": bands,
        "weighted_ks": weighted / weight if weight else float("nan"),
    }


def vs_z_or_h(bounds: np.ndarray, vs: np.ndarray, z: float = 30.0) -> float:
    """Time-averaged Vs to ``z``, or to ``H`` when the profile is shallower."""
    return vs_z(bounds, vs, min(float(bounds[-1]), z))


def vs_z(bounds: np.ndarray, vs: np.ndarray, z: float = 30.0) -> float:
    """Time-averaged Vs to depth ``z``. NaN if the profile stops short of ``z``."""
    if bounds[-1] < z:
        return float("nan")
    top = np.minimum(bounds[:-1], z)
    bot = np.minimum(bounds[1:], z)
    travel = np.sum((bot - top) / vs)
    return float(z / travel) if travel > 0 else float("nan")


def transfer_function(
    bounds: np.ndarray,
    vs: np.ndarray,
    freqs: np.ndarray = F0_FREQS_HZ,
    damping: float = DAMPING,
) -> np.ndarray:
    """Surface over outcrop amplitude for vertical SH waves, constant density.

    The last layer is the elastic half-space. Uses the Thomson–Haskell
    up/down-going amplitude recursion with complex modulus ``Vs(1 + iξ)``.
    A zero-thickness layer is transparent. Needs at least two layers.
    """
    h = np.diff(bounds)[:-1]
    vs_c = vs.astype(np.complex128) * (1.0 + 1j * damping)
    omega = 2.0 * np.pi * freqs
    a = np.ones_like(omega, dtype=np.complex128)
    b = np.ones_like(omega, dtype=np.complex128)
    for m in range(len(h)):
        alpha = vs_c[m] / vs_c[m + 1]
        phase = np.exp(1j * omega / vs_c[m] * h[m])
        a, b = (
            0.5 * a * (1 + alpha) * phase + 0.5 * b * (1 - alpha) / phase,
            0.5 * a * (1 - alpha) * phase + 0.5 * b * (1 + alpha) / phase,
        )
    return np.abs(2.0 / (2.0 * a))


def f0_and_peak(bounds: np.ndarray, vs: np.ndarray) -> tuple[float, float]:
    if len(vs) < 2:
        return float("nan"), float("nan")
    tf = transfer_function(bounds, vs)
    i = int(np.argmax(tf))
    return float(F0_FREQS_HZ[i]), float(tf[i])


def _ks(a, b) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return float(scipy_stats.ks_2samp(a, b).statistic)


def site_signatures(
    sequences: list[np.ndarray], include_invalid: bool = False
) -> dict[str, np.ndarray]:
    vs30, f0, peak = [], [], []
    for bounds, vs in _layers(sequences, include_invalid):
        vs30.append(vs_z(bounds, vs, 30.0))
        f, p = f0_and_peak(bounds, vs)
        f0.append(f)
        peak.append(p)
    return {"vs30": np.array(vs30), "f0": np.array(f0), "peak": np.array(peak)}


def _summary(x: np.ndarray) -> dict[str, float]:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0, "median": float("nan"), "p10": float("nan"), "p90": float("nan")}
    return {
        "n": int(len(x)),
        "median": float(np.median(x)),
        "p10": float(np.percentile(x, 10)),
        "p90": float(np.percentile(x, 90)),
    }


def site_report(
    real: list[np.ndarray],
    generated: list[np.ndarray],
    ref: ResidualReference,
    include_invalid: bool = False,
) -> dict:
    """All Stage 8 engineering checks for one generated set against the real test set."""
    rr = residual_correlation(real, ref, include_invalid)
    gr = residual_correlation(generated, ref, include_invalid)
    rs = site_signatures(real, include_invalid)
    gs = site_signatures(generated, include_invalid)
    return {
        "include_invalid": include_invalid,
        "gen_profiles_scored": len(_layers(generated, include_invalid)),
        "residual_correlation": {
            "real": rr,
            "gen": gr,
            "abs_rho_gap": abs(rr["rho"] - gr["rho"]),
        },
        "sigma_ln_vs": sigma_ln_vs_by_depth(
            real, generated, include_invalid=include_invalid
        ),
        "thickness": thickness_ks(real, generated, include_invalid),
        "vs30": {
            "real": _summary(rs["vs30"]),
            "gen": _summary(gs["vs30"]),
            "ks": _ks(rs["vs30"], gs["vs30"]),
        },
        "f0": {
            "real": _summary(rs["f0"]),
            "gen": _summary(gs["f0"]),
            "ks_ln": _ks(np.log(rs["f0"]), np.log(gs["f0"])),
            "peak_real": _summary(rs["peak"]),
            "peak_gen": _summary(gs["peak"]),
        },
    }


VS_Z_DEPTHS_M = (5.0, 10.0, 30.0)
LAYER_VS_BANDS_M = THICKNESS_BANDS_M


def vs_depth_breakdown(
    sequences: list[np.ndarray], include_invalid: bool = False
) -> dict:
    """Time-averaged Vs at 5, 10, and 30 m, plus median layer Vs by depth band."""
    layers = _layers(sequences, include_invalid)
    by_z = {f"vs{int(z)}": [] for z in VS_Z_DEPTHS_M}
    by_band = {f"{a:.0f}-{b if np.isfinite(b) else 'inf'}": [] for a, b in LAYER_VS_BANDS_M}
    for bounds, vs in layers:
        for z in VS_Z_DEPTHS_M:
            by_z[f"vs{int(z)}"].append(vs_z(bounds, vs, z))
        mid = 0.5 * (bounds[:-1] + bounds[1:])
        for (a, b), key in zip(LAYER_VS_BANDS_M, by_band):
            sel = (mid >= a) & (mid < b)
            if np.any(sel):
                by_band[key].extend(vs[sel].tolist())
    return {
        "time_averaged": {k: _summary(np.asarray(v, float)) for k, v in by_z.items()},
        "layer_vs_by_band": {k: _summary(np.asarray(v, float)) for k, v in by_band.items()},
    }


def _interp_vectors(sequences: list[np.ndarray], n_points: int = 100) -> np.ndarray:
    """Flattened interpolated [tts, depth] rows; drops non-increasing depth."""
    kept = []
    for seq in sequences:
        seq = np.asarray(seq, float)
        if len(seq) < 2:
            continue
        z = np.concatenate([[0.0], seq[:, 1]])
        if np.any(np.diff(z) < -1e-8) or not np.all(np.isfinite(seq)):
            continue
        kept.append(seq)
    if not kept:
        return np.zeros((0, n_points * 2))
    return np.stack([row.ravel() for row in interpolate_profiles(kept, n_points)])


def nearest_train_distance(
    generated: list[np.ndarray],
    train: list[np.ndarray],
    test: list[np.ndarray] | None = None,
    n_points: int = 100,
) -> dict:
    """Min interpolated-profile distance from each generated sample to the train set.

    The interpolation is the same as ``compute_pairwise_distance``. ``test`` is
    the held-out baseline: the fraction of generated samples closer to train
    than the median test-to-train distance.
    """
    g = _interp_vectors(generated, n_points)
    t = _interp_vectors(train, n_points)
    empty = {
        "n_generated": int(len(g)),
        "n_train": int(len(t)),
        "median": float("nan"),
        "mean": float("nan"),
        "p10": float("nan"),
        "p90": float("nan"),
        "test_median": float("nan"),
        "frac_closer_than_test_median": float("nan"),
    }
    if len(g) == 0 or len(t) == 0:
        return empty
    mins = cdist(g, t).min(axis=1)
    out = {
        "n_generated": int(len(mins)),
        "n_train": int(len(t)),
        "median": float(np.median(mins)),
        "mean": float(np.mean(mins)),
        "p10": float(np.percentile(mins, 10)),
        "p90": float(np.percentile(mins, 90)),
        "test_median": float("nan"),
        "frac_closer_than_test_median": float("nan"),
    }
    if test is not None:
        te = _interp_vectors(test, n_points)
        if len(te):
            test_mins = cdist(te, t).min(axis=1)
            thresh = float(np.median(test_mins))
            out["test_median"] = thresh
            out["frac_closer_than_test_median"] = float(np.mean(mins < thresh))
    return out
