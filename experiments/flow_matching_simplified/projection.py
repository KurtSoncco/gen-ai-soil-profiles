"""Same-profile low-frequency projection for the Stage 6 prior.

T is built in ln(Vs) on knots at 10 m through 200 m. Groups of four valid
knots are averaged and linearly interpolated back. Travel time at the
profile's own breakpoints uses ``Δt = Δz / Vs``. Layers outside
[100, 5000] m/s are kept.
"""

from __future__ import annotations

import numpy as np
import torch

from experiments.flow_matching_simplified.eval_corrected import (
    TREND_DEPTHS_M,
    _layer_index_at,
    travel_time_from_vs,
    vs_from_travel_time,
)

FACTOR = 4
ALPHA = 0.5
LN_MIN = float(np.log(100.0))
LN_MAX = float(np.log(5000.0))
KNOTS = np.asarray(TREND_DEPTHS_M, dtype=np.float64)


def project_vector(values: np.ndarray, valid: np.ndarray, factor: int = FACTOR) -> np.ndarray:
    """Average contiguous groups of ``factor`` valid entries and interpolate back."""
    out = np.zeros(len(values), dtype=np.float64)
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        return out
    vals = np.asarray(values, dtype=np.float64)[idx]
    n = len(vals)
    centers = []
    coarse = []
    for start in range(0, n, factor):
        sl = np.arange(start, min(start + factor, n))
        centers.append(float(sl.mean()))
        coarse.append(float(vals[sl].mean()))
    if len(coarse) == 1:
        up = np.full(n, coarse[0])
    else:
        up = np.interp(np.arange(n), np.asarray(centers), np.asarray(coarse))
    out[idx] = up
    return out


def knot_ln(sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Projected ln(Vs) on ``KNOTS``, and the valid-knot mask.

    Returns zeros and an empty mask when the profile cannot be inverted.
    """
    projected = np.zeros(len(KNOTS), dtype=np.float64)
    valid = np.zeros(len(KNOTS), dtype=bool)
    z0, z1, vs = vs_from_travel_time(np.asarray(sequence, dtype=np.float64))
    if len(vs) == 0 or np.any(~np.isfinite(vs) | (vs <= 0)):
        return projected, valid
    boundaries = np.concatenate([z0[:1], z1])
    if np.any(np.diff(boundaries) <= 0):
        return projected, valid
    ln = np.log(vs)
    raw = np.full(len(KNOTS), np.nan)
    for j, z in enumerate(KNOTS):
        if z <= boundaries[0] or z > boundaries[-1]:
            continue
        layer = int(_layer_index_at(boundaries, np.array([z]))[0])
        if 0 <= layer < len(ln):
            raw[j] = ln[layer]
    valid = np.isfinite(raw)
    if not valid.any():
        return projected, valid
    projected = project_vector(np.where(valid, raw, 0.0), valid, FACTOR)
    return projected, valid


def projection_tokens(sequence: np.ndarray) -> np.ndarray:
    """Physical ``[tts, depth]`` of T(x1) at this profile's breakpoint depths."""
    sequence = np.asarray(sequence, dtype=np.float64)
    z0, z1, vs = vs_from_travel_time(sequence)
    fallback = np.array(sequence, dtype=np.float64, copy=True)
    if len(vs) == 0 or np.any(~np.isfinite(vs) | (vs <= 0)):
        return fallback
    boundaries = np.concatenate([z0[:1], z1])
    if np.any(np.diff(boundaries) <= 0):
        return fallback
    projected, valid = knot_ln(sequence)
    if not valid.any():
        return fallback
    known_z = KNOTS[valid]
    known_ln = projected[valid]
    centers = 0.5 * (z0 + z1)
    if len(known_z) == 1:
        layer_ln = np.full(len(centers), known_ln[0])
    else:
        layer_ln = np.interp(centers, known_z, known_ln)
    tokens = travel_time_from_vs(boundaries, np.exp(layer_ln))
    if len(tokens) != len(sequence):
        return fallback
    return tokens


def knot_outside_fraction(sequences: list[np.ndarray]) -> dict[str, float]:
    """Share of projected knots outside [ln 100, ln 5000]. Unclipped."""
    outside = 0
    total = 0
    for seq in sequences:
        ln, valid = knot_ln(seq)
        total += int(valid.sum())
        outside += int(np.sum(valid & ((ln < LN_MIN) | (ln > LN_MAX))))
    return {
        "n_knots": total,
        "n_outside": outside,
        "fraction_outside": float(outside / total) if total else float("nan"),
    }


def fit_trend(sequences: list[np.ndarray]) -> dict[str, list[float] | float]:
    """Mean, std, and exponential length of projected ln(Vs), train split only."""
    cols = [[] for _ in range(len(KNOTS))]
    pair_a = []
    pair_b = []
    for seq in sequences:
        ln, valid = knot_ln(seq)
        for j in range(len(KNOTS)):
            if valid[j]:
                cols[j].append(ln[j])
        idx = np.flatnonzero(valid)
        for a, b in zip(idx[:-1], idx[1:]):
            if KNOTS[b] - KNOTS[a] == 10.0:
                pair_a.append(ln[a])
                pair_b.append(ln[b])
    present = [np.asarray(c, dtype=np.float64) for c in cols if len(c)]
    global_mean = float(np.mean(np.concatenate(present))) if present else 0.0
    global_std = float(np.std(np.concatenate(present)) + 1e-3) if present else 1.0
    mu = []
    sig = []
    for c in cols:
        if len(c) >= 5:
            arr = np.asarray(c, dtype=np.float64)
            mu.append(float(arr.mean()))
            sig.append(float(arr.std() + 1e-3))
        else:
            mu.append(global_mean)
            sig.append(global_std)
    if len(pair_a) >= 5:
        rho = float(np.corrcoef(pair_a, pair_b)[0, 1])
    else:
        rho = 0.5
    if not np.isfinite(rho) or rho <= 0.0 or rho >= 1.0:
        length = 40.0
    else:
        length = float(-10.0 / np.log(rho))
    return {"mu": mu, "sig": sig, "length_m": length, "n_profiles": len(sequences)}


def draw_trend_knots(
    fit: dict,
    final_depth: float,
    rng: np.random.Generator,
    clip: bool,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """One draw of projected ln(Vs). Returns values, mask, n_outside, n_knots."""
    mu = np.asarray(fit["mu"], dtype=np.float64)
    sig = np.asarray(fit["sig"], dtype=np.float64)
    length = float(fit["length_m"])
    valid = KNOTS <= final_depth
    idx = np.flatnonzero(valid)
    ln = np.zeros(len(KNOTS), dtype=np.float64)
    if len(idx) == 0:
        return ln, valid, 0, 0
    dist = np.abs(KNOTS[idx][:, None] - KNOTS[idx][None, :])
    cov = np.exp(-dist / max(length, 1e-6))
    cov.flat[:: cov.shape[0] + 1] += 1e-6
    z = rng.multivariate_normal(np.zeros(len(idx)), cov)
    drawn = mu[idx] + sig[idx] * z
    n_outside = int(np.sum((drawn < LN_MIN) | (drawn > LN_MAX)))
    if clip:
        drawn = np.clip(drawn, LN_MIN, LN_MAX)
    ln[idx] = drawn
    return ln, valid, n_outside, int(len(idx))


def tokens_from_knots(depths: np.ndarray, knot_values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Passeri tokens at ``depths`` (surface included) from a knot field."""
    depths = np.asarray(depths, dtype=np.float64)
    known_z = KNOTS[valid]
    known_ln = np.asarray(knot_values, dtype=np.float64)[valid]
    centers = 0.5 * (depths[:-1] + depths[1:])
    if len(known_z) == 0:
        layer_ln = np.full(len(centers), float(np.mean(knot_values)))
    elif len(known_z) == 1:
        layer_ln = np.full(len(centers), known_ln[0])
    else:
        layer_ln = np.interp(centers, known_z, known_ln)
    return travel_time_from_vs(depths, np.exp(layer_ln))


def normalize_tokens(tokens: np.ndarray, stats: dict[str, float]) -> np.ndarray:
    ts = (tokens[:, 0] - stats["ts_mean"]) / stats["ts_std"]
    depth = (tokens[:, 1] - stats["depth_mean"]) / stats["depth_std"]
    return np.column_stack([ts, depth]).astype(np.float32)


def attach_projections(dataset, raw_sequences: list[np.ndarray]) -> None:
    """Store padded normalized T(x1) on a dataset, aligned with its items."""
    padded = []
    for raw, stats in zip(raw_sequences, dataset.sequence_stats):
        physical = projection_tokens(np.asarray(raw, dtype=np.float64))
        norm = normalize_tokens(physical, stats)
        if len(norm) < dataset.max_length:
            pad = np.zeros((dataset.max_length - len(norm), 2), dtype=np.float32)
            norm = np.vstack([norm, pad])
        else:
            norm = norm[: dataset.max_length]
        padded.append(norm)
    dataset.projected = padded


def mix_projection(
    projection: torch.Tensor,
    attention_mask: torch.Tensor,
    alpha: float = ALPHA,
) -> torch.Tensor:
    """x0 = (1 - α) T + α ε in the model's normalized token space."""
    eps = torch.randn_like(projection)
    mask = attention_mask.unsqueeze(-1).float()
    return ((1.0 - alpha) * projection + alpha * eps) * mask
