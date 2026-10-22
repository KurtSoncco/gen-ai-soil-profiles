"""README figures for the constrained-decoder headline draw.

``s42_e500/latest.pt``, Euler 50, 200 profiles, seed 42, matched
``(n, H, Vs_z)``. Layers are drawn with ``Axes.stairs`` (horizontal).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from scipy import stats as scipy_stats

from experiments.flow_matching_simplified.eval_corrected import (
    TREND_DEPTHS_M,
    layer_model,
    ln_vs_matrix,
    vs_from_travel_time,
)
from experiments.flow_matching_simplified.site_metrics import site_signatures
from experiments.flow_matching_simplified.toro_baseline import fit_toro
from experiments.flow_matching_simplified.train_constrained import (
    OUT,
    ConstrainedTokens,
    build_model,
    load_split,
    sample,
)
from soilgen_ai.figures.style import STYLE

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "docs" / "figures"
CACHE = ROOT / "outputs" / "flow_matching_simplified" / "samples"
TAG = "s42_e500"
N_SAMPLES = 200
SEED = 42
N_SHOW = 10
Z_MAX = 100.0
VS_MIN = 100.0
VS_MAX = 5000.0


def _percentiles(sequences: list[np.ndarray]):
    mat = ln_vs_matrix(sequences, TREND_DEPTHS_M)
    with np.errstate(invalid="ignore"):
        p10 = np.nanpercentile(mat, 10, axis=0)
        p50 = np.nanpercentile(mat, 50, axis=0)
        p90 = np.nanpercentile(mat, 90, axis=0)
        n = np.sum(np.isfinite(mat), axis=0)
    return p10, p50, p90, n


def pick_profiles(
    sequences: list[np.ndarray], n: int = N_SHOW, z_max: float = Z_MAX
) -> list[np.ndarray]:
    """Spread eight profiles by depth so the panel is not only 30 m holes."""
    usable = [seq for seq in sequences if 8.0 <= float(seq[-1, 1]) <= z_max]
    usable.sort(key=lambda seq: float(seq[-1, 1]))
    if len(usable) <= n:
        return usable
    idx = np.linspace(0, len(usable) - 1, n).astype(int)
    return [usable[i] for i in idx]


def layer_stairs(seq: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Vs values and depth edges for ``Axes.stairs(..., orientation='horizontal')``."""
    z0, z1, vs = vs_from_travel_time(seq)
    ok = np.isfinite(vs) & (vs > 0) & (z1 > z0)
    if not np.any(ok):
        return None
    bounds = np.concatenate([z0[ok][:1], z1[ok]])
    return vs[ok], bounds


def draw_stairs(ax: Axes, sequences: list[np.ndarray], color: str) -> None:
    for seq in pick_profiles(sequences):
        layer = layer_stairs(seq)
        if layer is None:
            continue
        vs, bounds = layer
        ax.stairs(
            vs,
            bounds,
            orientation="horizontal",
            baseline=None,
            fill=False,
            color=color,
            lw=0.9,
            alpha=0.88,
        )
    ax.set_xscale("log")
    ax.set_xlim(80, 3500)
    ax.set_ylim(Z_MAX, 0)
    ax.set_xlabel("Vs (m/s)")
    ax.axvline(VS_MIN, color=STYLE.muted, lw=0.4, ls="--")
    ax.axvline(VS_MAX, color=STYLE.muted, lw=0.4, ls="--")


def _finite(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return a[np.isfinite(a)]


def overlay_hist(ax: Axes, real: np.ndarray, generated: np.ndarray, bins: int) -> float:
    real = _finite(real)
    generated = _finite(generated)
    lo = min(float(real.min()), float(generated.min()))
    hi = max(float(real.max()), float(generated.max()))
    edges = np.linspace(lo, hi, bins + 1)
    ax.hist(
        real,
        bins=edges,
        density=True,
        histtype="step",
        color=STYLE.real,
        lw=1.15,
        label="Real test",
    )
    ax.hist(
        generated,
        bins=edges,
        density=True,
        histtype="step",
        color=STYLE.flow,
        lw=1.15,
        label="Generated",
    )
    return float(scipy_stats.ks_2samp(real, generated).statistic)


def thicknesses(sequences: list[np.ndarray]) -> np.ndarray:
    out = []
    for seq in sequences:
        lm = layer_model(seq, include_invalid=False)
        if lm is None:
            continue
        out.extend(np.diff(lm[0]).tolist())
    return np.asarray(out, dtype=float)


def load_or_sample() -> tuple[list[np.ndarray], list[np.ndarray]]:
    cache = CACHE / f"constrained_{TAG}_n{N_SAMPLES}.npy"
    train, _, test = load_split()
    if cache.exists():
        generated = [
            np.asarray(seq, dtype=np.float64)
            for seq in np.load(cache, allow_pickle=True)
        ]
        print(f"loaded {len(generated)} cached samples from {cache}")
        return test, generated

    device = torch.device("cpu")
    ckpt = torch.load(OUT / TAG / "latest.pt", map_location=device, weights_only=False)
    data = ConstrainedTokens(train)
    assert np.allclose(ckpt["scale"], data.scale)
    model = build_model()
    model.load_state_dict(ckpt["model"])
    model.to(device)
    frac = float(
        ckpt.get("toro_source_frac", 1.0 if ckpt.get("use_toro_source", True) else 0.0)
    )
    toro_fit = fit_toro(train) if frac > 0.0 else None
    generated = sample(
        model,
        data,
        N_SAMPLES,
        SEED,
        device,
        toro_fit=toro_fit,
        vs_z_mode="matched",
        toro_source_frac=frac,
    )
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(cache, np.asarray(generated, dtype=object), allow_pickle=True)
    print(
        f"sampled {len(generated)} profiles, epoch {int(ckpt['epoch'])}, wrote {cache}"
    )
    return test, generated


def plot_profiles(real: list[np.ndarray], generated: list[np.ndarray]) -> None:
    STYLE.apply()
    fig, axes = plt.subplots(1, 2, figsize=STYLE.size("double", 0.42), sharey=True)
    draw_stairs(axes[0], real, STYLE.real)
    draw_stairs(axes[1], generated, STYLE.flow)
    axes[0].set_ylabel("Depth (m)")
    for ax, letter in zip(axes, "ab"):
        STYLE.panel(ax, letter)
    fig.tight_layout()
    STYLE.save(fig, FIG / "constrained_vs_profiles")


def plot_ln_vs(real: list[np.ndarray], generated: list[np.ndarray]) -> None:
    STYLE.apply()
    real_lo, real_med, real_hi, _ = _percentiles(real)
    lo, med, hi, n = _percentiles(generated)
    fig, ax = plt.subplots(figsize=STYLE.size("onehalf", 0.72))
    ax.fill_betweenx(
        TREND_DEPTHS_M,
        real_lo,
        real_hi,
        color=STYLE.real,
        alpha=0.18,
        linewidth=0,
        label="Real p10–p90",
    )
    ax.plot(real_med, TREND_DEPTHS_M, color=STYLE.real, lw=1.2, label="Real median")
    ax.plot(med, TREND_DEPTHS_M, color=STYLE.flow, lw=1.2, label="Generated median")
    ax.plot(
        lo, TREND_DEPTHS_M, color=STYLE.flow, lw=0.7, ls="--", label="Generated p10–p90"
    )
    ax.plot(hi, TREND_DEPTHS_M, color=STYLE.flow, lw=0.7, ls="--")
    enough = n >= 5
    if np.any(~enough):
        z_cut = float(TREND_DEPTHS_M[~enough][0])
        ax.annotate(
            "n<5",
            (med[np.where(~enough)[0][0]], z_cut),
            fontsize=6,
            color=STYLE.flow,
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
        )
    ax.set_ylim(200, 0)
    ax.set_xlabel("ln(Vs)")
    ax.set_ylabel("Depth (m)")
    ax.legend(loc="upper right")
    STYLE.panel(ax, "a")
    fig.tight_layout()
    STYLE.save(fig, FIG / "constrained_ln_vs_depth")


def plot_vs30_thickness(
    real: list[np.ndarray], generated: list[np.ndarray]
) -> dict[str, float]:
    STYLE.apply()
    real_site = site_signatures(real)
    gen_site = site_signatures(generated)
    real_h = thicknesses(real)
    gen_h = thicknesses(generated)
    real_h = real_h[real_h <= 40.0]
    gen_h = gen_h[gen_h <= 40.0]
    fig, axes = plt.subplots(1, 2, figsize=STYLE.size("double", 0.42))
    ks_vs30 = overlay_hist(axes[0], real_site["vs30"], gen_site["vs30"], 22)
    axes[0].set_xlim(80, 800)
    axes[0].set_xlabel("Vs30 (m/s)")
    axes[0].set_ylabel("Density")
    ks_h = overlay_hist(axes[1], real_h, gen_h, 22)
    axes[1].set_xlim(0, 40)
    axes[1].set_xlabel("Layer thickness (m)")
    for ax, letter in zip(axes, "ab"):
        STYLE.panel(ax, letter)
    handles = [
        plt.Line2D([0], [0], color=STYLE.real, lw=1.3, label="Real test"),
        plt.Line2D([0], [0], color=STYLE.flow, lw=1.3, label="Generated"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.08),
        frameon=False,
    )
    fig.tight_layout()
    STYLE.save(fig, FIG / "constrained_vs30_thickness")
    return {"vs30": ks_vs30, "thickness": ks_h}


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    real, generated = load_or_sample()
    plot_profiles(real, generated)
    plot_ln_vs(real, generated)
    ks = plot_vs30_thickness(real, generated)
    real_vs30 = _finite(site_signatures(real)["vs30"])
    gen_vs30 = _finite(site_signatures(generated)["vs30"])
    print(
        "vs30 median",
        float(np.median(real_vs30)),
        float(np.median(gen_vs30)),
        "ks",
        ks,
    )


if __name__ == "__main__":
    main()
