"""README figures for the one-way Stage 3 scoring draw.

``latest.pt`` is the checkpoint that is scored. Sampling is Euler, 50 steps,
200 profiles, seed 42. Colors and sizes come from ``soilgen_ai.figures.style``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.flow_matching_simplified import config as cfg_mod
from experiments.flow_matching_simplified.baseline_metrics import (
    _denorm_generated,
    _initial_state,
    euler_integrate,
    sample_fixed_grid_gaussian,
)
from experiments.flow_matching_simplified.comprehensive_eval import load_checkpoint
from experiments.flow_matching_simplified.data import FlowMatchingDataLoader
from experiments.flow_matching_simplified.eval_corrected import (
    TREND_DEPTHS_M,
    ln_vs_matrix,
    vs_from_travel_time,
)
from experiments.flow_matching_simplified.run_corrected_eval import _denorm_split
from soilgen_ai.figures.style import STYLE

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "outputs" / "flow_matching_simplified_onewaytime" / "checkpoints" / "latest.pt"
DATA = ROOT / "experiments" / "flow_matching_simplified" / "data" / "breakpoints_onewaytime.parquet"
SPLIT = ROOT / "outputs" / "flow_matching_simplified" / "results" / "split_indices_seed42.json"
OUT = ROOT / "docs" / "figures"
VS_MIN = 100.0
VS_MAX = 5000.0


def _layers(sequence: np.ndarray):
    return vs_from_travel_time(sequence)


def _steps(z0, z1, vs):
    x, y = [], []
    for a, b, v in zip(z0, z1, vs):
        if not np.isfinite(v) or v <= 0 or b <= a:
            continue
        x.extend([v, v])
        y.extend([a, b])
    return np.asarray(x), np.asarray(y)


def _draw_layers(ax, sequences, color, n=8):
    shown = 0
    for seq in sequences:
        if shown >= n:
            break
        z0, z1, vs = _layers(seq)
        if len(vs) == 0:
            continue
        x, y = _steps(z0, z1, vs)
        if len(x) == 0:
            continue
        ax.plot(x, y, color=color, lw=0.9, solid_capstyle="butt")
        bad = np.isfinite(vs) & (vs > 0) & ((vs < VS_MIN) | (vs > VS_MAX))
        if np.any(bad):
            ax.scatter(
                vs[bad],
                0.5 * (z0[bad] + z1[bad]),
                s=10,
                facecolors="none",
                edgecolors=STYLE.ink,
                linewidths=0.6,
                zorder=3,
            )
        shown += 1
    ax.set_xscale("log")
    ax.set_xlim(20, 8000)
    ax.axvline(VS_MIN, color=STYLE.muted, lw=0.4, ls="--")
    ax.axvline(VS_MAX, color=STYLE.muted, lw=0.4, ls="--")
    ax.set_xlabel("Vs (m/s)")
    ax.set_ylim(200, 0)


def _percentiles(sequences):
    mat = ln_vs_matrix(sequences, TREND_DEPTHS_M)
    with np.errstate(invalid="ignore"):
        p10 = np.nanpercentile(mat, 10, axis=0)
        p50 = np.nanpercentile(mat, 50, axis=0)
        p90 = np.nanpercentile(mat, 90, axis=0)
        n = np.sum(np.isfinite(mat), axis=0)
    return p10, p50, p90, n


def _sample():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = json.loads(SPLIT.read_text())
    loader = FlowMatchingDataLoader(DATA)
    loader.load_data()
    real, _ = _denorm_split(loader, split["test"])
    train_raw, train_ds = _denorm_split(loader, split["train"])
    u0, masks, stats, stats_list, lengths = _initial_state(
        train_ds, 200, cfg_mod.cfg.max_length, device, 42
    )
    model, epoch = load_checkpoint(str(CKPT), device, cfg_mod.cfg)
    model.eval()
    generated = _denorm_generated(euler_integrate(model, u0, masks, stats, 50), lengths, stats_list)
    gauss = sample_fixed_grid_gaussian(train_raw, 200, 42)
    print(f"latest epoch index {int(epoch)} device {device}", flush=True)
    return real, generated, gauss


def plot_layers(real, generated, gauss) -> None:
    STYLE.apply()
    fig, axes = plt.subplots(1, 3, figsize=STYLE.size("double", 0.42), sharey=True)
    _draw_layers(axes[0], real, STYLE.real)
    _draw_layers(axes[1], generated, STYLE.flow)
    _draw_layers(axes[2], gauss, STYLE.baseline)
    axes[0].set_ylabel("Depth (m)")
    for ax, letter in zip(axes, "abc"):
        STYLE.panel(ax, letter)
    fig.tight_layout()
    STYLE.save(fig, OUT / "layered_vs")


def plot_band(real, generated, gauss) -> None:
    STYLE.apply()
    real_lo, real_med, real_hi, n_real = _percentiles(real)
    series = (
        ("latest.pt", generated, STYLE.flow),
        ("Gaussian", gauss, STYLE.baseline),
    )
    fig, axes = plt.subplots(1, 2, figsize=STYLE.size("double", 0.48), sharey=True)
    for ax, (label, seqs, color), letter in zip(axes, series, "ab"):
        lo, med, hi, n = _percentiles(seqs)
        ax.fill_betweenx(
            TREND_DEPTHS_M,
            real_lo,
            real_hi,
            color=STYLE.real,
            alpha=0.18,
            linewidth=0,
        )
        ax.plot(real_med, TREND_DEPTHS_M, color=STYLE.real, lw=1.2, label="Real median")
        ax.plot(med, TREND_DEPTHS_M, color=color, lw=1.2, label=f"{label} median")
        ax.plot(lo, TREND_DEPTHS_M, color=color, lw=0.7, ls="--", label=f"{label} p10–p90")
        ax.plot(hi, TREND_DEPTHS_M, color=color, lw=0.7, ls="--")
        labelled = False
        for z, a, b, m, nr, ng in zip(TREND_DEPTHS_M, real_lo, real_hi, med, n_real, n):
            if nr >= 5 and ng >= 5 and np.isfinite(m) and not (a <= m <= b):
                ax.plot(m, z, "o", ms=3.0, color=color)
            if label == "latest.pt" and ng < 5 and not labelled:
                ax.annotate(
                    "n<5",
                    (m, z),
                    fontsize=6,
                    color=color,
                    xytext=(6, 0),
                    textcoords="offset points",
                    va="center",
                )
                labelled = True
        ax.set_ylim(200, 0)
        ax.set_xlabel("ln(Vs)")
        ax.legend(loc="upper right")
        STYLE.panel(ax, letter)
    axes[0].set_ylabel("Depth (m)")
    fig.tight_layout()
    STYLE.save(fig, OUT / "ln_vs_percentiles")


def main() -> None:
    real, generated, gauss = _sample()
    plot_layers(real, generated, gauss)
    plot_band(real, generated, gauss)


if __name__ == "__main__":
    main()
