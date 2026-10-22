"""Toro (1995) Vs profile randomization, fitted on the training split.

Three parts, as summarized in Toro (2022, J. Seismology, doi:10.1007/s10950-022-10084-x):

- Layering: interfaces form a non-homogeneous Poisson process with rate
  ``λ(z) = c3 (z + c1)^(-c2)``. Fitted by maximum likelihood on the interior
  breakpoints of the training profiles, each observed on ``(0, H]``.
- Median and σ of ln(Vs) against depth, from training layer midpoints in
  depth bins, interpolated linearly between bin centres.
- Inter-layer correlation: ``Z_i = ρ Z_{i-1} + sqrt(1 - ρ²) ε`` with
  ``ρ = (1 - ρ_d(d)) ρ_t(t) + ρ_d(d)``, ``ρ_t = ρ0 exp(-t / Δ)``,
  ``ρ_d = ρ200 ((d + d0) / (200 + d0))^b`` for ``d ≤ 200`` m and ``ρ200`` below.
  ``d`` is the mean of two adjacent midpoints and ``t`` their separation.
  Fitted by maximum likelihood on adjacent training layers.

Profile depth ``H`` is bootstrapped from the training profiles; Toro gives no
generic depth model. Vs is not truncated, so the bound rate is reported as is.
Unlike the fixed-grid Gaussian, this baseline generates its own layering.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import minimize

from experiments.flow_matching_simplified.eval_corrected import (
    layer_model,
    travel_time_from_vs,
)

MEDIAN_EDGES_M = np.array(
    [0, 2, 4, 6, 8, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 125, 150, 200, 300, 500, np.inf]
)


@dataclass
class ToroFit:
    c1: float
    c2: float
    c3: float
    rho0: float
    delta: float
    rho200: float
    d0: float
    b: float
    centres: np.ndarray
    median: np.ndarray
    sigma: np.ndarray
    depths: np.ndarray

    def summary(self) -> dict:
        out = {k: v for k, v in asdict(self).items() if not isinstance(v, np.ndarray)}
        out["centres_m"] = self.centres.tolist()
        out["median_ln_vs"] = self.median.tolist()
        out["sigma_ln_vs"] = self.sigma.tolist()
        return out


def _cumulative_rate(z, c1, c2, c3):
    z = np.asarray(z, dtype=np.float64)
    if abs(1.0 - c2) < 1e-6:
        return c3 * np.log((z + c1) / c1)
    return c3 / (1.0 - c2) * ((z + c1) ** (1.0 - c2) - c1 ** (1.0 - c2))


def _inverse_cumulative_rate(s, c1, c2, c3):
    s = np.asarray(s, dtype=np.float64)
    if abs(1.0 - c2) < 1e-6:
        return c1 * np.exp(s / c3) - c1
    base = s * (1.0 - c2) / c3 + c1 ** (1.0 - c2)
    with np.errstate(invalid="ignore"):
        z = np.where(base > 0, base ** (1.0 / (1.0 - c2)) - c1, np.inf)
    return z


def _rho(d, t, rho0, delta, rho200, d0, b):
    rho_t = rho0 * np.exp(-t / delta)
    rho_d = np.where(
        d <= 200.0, rho200 * ((d + d0) / (200.0 + d0)) ** b, rho200
    )
    return (1.0 - rho_d) * rho_t + rho_d


def _valid_layers(train: list[np.ndarray]):
    for seq in train:
        lm = layer_model(seq, include_invalid=False)
        if lm is not None:
            yield lm


def fit_layering(train: list[np.ndarray]) -> tuple[float, float, float]:
    interfaces, depths = [], []
    for bounds, _ in _valid_layers(train):
        interfaces.extend(bounds[1:-1].tolist())
        depths.append(bounds[-1])
    zi = np.asarray(interfaces)
    hp = np.asarray(depths)

    def nll(p):
        c1, c2, c3 = np.exp(p[0]), p[1], np.exp(p[2])
        lam = c3 * (zi + c1) ** (-c2)
        return -(np.sum(np.log(lam)) - np.sum(_cumulative_rate(hp, c1, c2, c3)))

    x0 = np.array([np.log(10.86), 0.89, np.log(1.98)])
    res = minimize(
        nll,
        x0,
        method="L-BFGS-B",
        bounds=[(np.log(0.01), np.log(1000.0)), (0.0, 3.0), (np.log(1e-4), np.log(100.0))],
    )
    return float(np.exp(res.x[0])), float(res.x[1]), float(np.exp(res.x[2]))


def fit_median_sigma(train: list[np.ndarray]):
    mids, lns = [], []
    for bounds, vs in _valid_layers(train):
        mids.append(0.5 * (bounds[:-1] + bounds[1:]))
        lns.append(np.log(vs))
    mid = np.concatenate(mids)
    ln = np.concatenate(lns)
    j = np.digitize(mid, MEDIAN_EDGES_M) - 1
    centres, med, sig = [], [], []
    for k in range(len(MEDIAN_EDGES_M) - 1):
        sel = j == k
        if np.sum(sel) < 5:
            continue
        centres.append(float(np.median(mid[sel])))
        med.append(float(np.median(ln[sel])))
        sig.append(float(np.std(ln[sel])))
    return np.array(centres), np.array(med), np.array(sig)


def _standardize(mid, ln, centres, med, sig):
    return (ln - np.interp(mid, centres, med)) / np.interp(mid, centres, sig)


def fit_correlation(train, centres, med, sig) -> tuple[float, ...]:
    zp, zc, dd, tt = [], [], [], []
    for bounds, vs in _valid_layers(train):
        if len(vs) < 2:
            continue
        mid = 0.5 * (bounds[:-1] + bounds[1:])
        z = _standardize(mid, np.log(vs), centres, med, sig)
        zp.extend(z[:-1])
        zc.extend(z[1:])
        dd.extend(0.5 * (mid[:-1] + mid[1:]))
        tt.extend(np.diff(mid))
    zp, zc, dd, tt = map(np.asarray, (zp, zc, dd, tt))

    def nll(p):
        rho = np.clip(_rho(dd, tt, *p), -0.999, 0.999)
        var = 1.0 - rho**2
        return np.sum(0.5 * np.log(var) + (zc - rho * zp) ** 2 / (2.0 * var))

    x0 = np.array([0.95, 3.4, 0.5, 0.0 + 1.0, 0.3])
    res = minimize(
        nll,
        x0,
        method="L-BFGS-B",
        bounds=[(0.0, 0.99), (0.5, 500.0), (0.0, 0.99), (0.1, 500.0), (0.01, 5.0)],
    )
    return tuple(float(v) for v in res.x)


def fit_toro(train: list[np.ndarray]) -> ToroFit:
    c1, c2, c3 = fit_layering(train)
    centres, med, sig = fit_median_sigma(train)
    rho0, delta, rho200, d0, b = fit_correlation(train, centres, med, sig)
    depths = np.array([bounds[-1] for bounds, _ in _valid_layers(train)])
    return ToroFit(c1, c2, c3, rho0, delta, rho200, d0, b, centres, med, sig, depths)


def sample_toro_bounds(fit: ToroFit, h_total: float, rng: np.random.Generator) -> np.ndarray:
    """Interfaces of a non-homogeneous Poisson process on ``(0, h_total]``."""
    interfaces = []
    s = 0.0
    while True:
        s += rng.exponential(1.0)
        z = float(_inverse_cumulative_rate(s, fit.c1, fit.c2, fit.c3))
        if not np.isfinite(z) or z >= h_total:
            break
        interfaces.append(z)
    return np.array([0.0, *interfaces, h_total], dtype=np.float64)


def match_layer_count(bounds: np.ndarray, n: int) -> np.ndarray:
    """Merge the thinnest or split the thickest layers until there are ``n``."""
    bounds = np.asarray(bounds, dtype=np.float64)
    n = max(int(n), 1)
    while len(bounds) - 1 > n:
        h = np.diff(bounds)
        i = int(np.argmin(h))
        if i == 0:
            bounds = np.concatenate([bounds[:1], bounds[2:]])
        elif i == len(h) - 1:
            bounds = np.concatenate([bounds[:-2], bounds[-1:]])
        elif h[i - 1] <= h[i + 1]:
            bounds = np.delete(bounds, i)
        else:
            bounds = np.delete(bounds, i + 1)
    while len(bounds) - 1 < n:
        h = np.diff(bounds)
        i = int(np.argmax(h))
        mid = 0.5 * (bounds[i] + bounds[i + 1])
        bounds = np.insert(bounds, i + 1, mid)
    return bounds


def fill_toro_vs(fit: ToroFit, bounds: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Toro median plus AR(1) residual at the midpoints of ``bounds``."""
    mid = 0.5 * (bounds[:-1] + bounds[1:])
    z = np.empty(len(mid))
    z[0] = rng.normal()
    for i in range(1, len(mid)):
        rho = float(
            _rho(
                0.5 * (mid[i - 1] + mid[i]),
                mid[i] - mid[i - 1],
                fit.rho0,
                fit.delta,
                fit.rho200,
                fit.d0,
                fit.b,
            )
        )
        z[i] = rho * z[i - 1] + np.sqrt(1.0 - rho**2) * rng.normal()
    ln_vs = np.interp(mid, fit.centres, fit.median) + np.interp(
        mid, fit.centres, fit.sigma
    ) * z
    return np.exp(ln_vs)


def sample_toro_matched(
    fit: ToroFit, h_total: float, n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Toro layering on ``(0, h_total]``, remapped to ``n`` layers, then Toro Vs."""
    bounds = match_layer_count(sample_toro_bounds(fit, h_total, rng), n)
    return bounds, fill_toro_vs(fit, bounds, rng)


def sample_toro(fit: ToroFit, num_samples: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(num_samples):
        h_total = float(fit.depths[rng.integers(0, len(fit.depths))])
        bounds = sample_toro_bounds(fit, h_total, rng)
        vs = fill_toro_vs(fit, bounds, rng)
        out.append(travel_time_from_vs(bounds, vs))
    return out
