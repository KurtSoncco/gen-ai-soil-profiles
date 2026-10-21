"""Redraw the README one-way figures.

Uses the seed-42 test breakpoints and the 200 sequences saved at the end of
one-way training. Style follows a Nature-style two-series contrast: navy for
real profiles, vermillion for generated profiles, thin spines, and no grid.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from scipy import stats as scipy_stats

from experiments.flow_matching_simplified.comprehensive_eval import (
    compute_all_vs_values,
    compute_breakpoint_count,
    compute_final_depth,
    prepend_origin,
)
from experiments.flow_matching_simplified.data import FlowMatchingDataLoader
from experiments.flow_matching_simplified.depth_dependent_stats import (
    bin_by_depth,
    compute_profile_statistics,
    load_target_statistics,
    reconstruct_vs_profile,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "figures"
REAL = "#3C5488"
GENERATED = "#E64B35"
BAND = "#4DBBD5"
INK = "#222222"
MUTED = "#666666"

# Nature Publishing Group palette, high-contrast pair plus the cyan band.
# Real is navy, generated is vermillion. Both stay distinct in grayscale
# because one series is solid and the other is dashed.


def use_nature_style() -> None:
    sns.set_theme(
        context="paper",
        style="ticks",
        font="DejaVu Sans",
        rc={
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.linewidth": 0.6,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        },
    )


def _panel(ax: Axes, letter: str) -> None:
    ax.text(
        -0.14,
        1.06,
        letter,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="left",
        color=INK,
    )


def _save(fig: plt.Figure, name: str) -> None:
    path = OUT / name
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def _load() -> tuple[list[np.ndarray], list[np.ndarray]]:
    loader = FlowMatchingDataLoader(
        ROOT
        / "experiments"
        / "flow_matching_simplified"
        / "data"
        / "breakpoints_onewaytime.parquet"
    )
    loader.load_data()
    assert loader.sequences is not None
    split = json.loads(
        (
            ROOT
            / "outputs"
            / "flow_matching_simplified"
            / "results"
            / "split_indices_seed42.json"
        ).read_text()
    )
    real = [loader.sequences[i] for i in split["test"]]
    generated = [
        np.asarray(seq, dtype=np.float32)
        for seq in np.load(
            ROOT
            / "outputs"
            / "flow_matching_simplified_onewaytime"
            / "samples"
            / "generated_samples.npy",
            allow_pickle=True,
        )
    ]
    return real, generated


def _with_origin(seq: np.ndarray) -> np.ndarray:
    if len(seq) == 0:
        return seq
    return prepend_origin(seq)


def plot_profiles(real: list[np.ndarray], generated: list[np.ndarray]) -> None:
    n = 15
    shown = [_with_origin(seq) for seq in real[:n] + generated[:n]]
    t_max = max(float(np.nanmax(seq[:, 0])) for seq in shown)
    t_min = min(float(np.nanmin(seq[:, 0])) for seq in shown)
    z_max = max(float(np.nanmax(seq[:, 1])) for seq in shown)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharex=True, sharey=True)
    pairs = (
        (axes[0], real[:n], REAL, "Real test profiles"),
        (axes[1], generated[:n], GENERATED, "Generated"),
    )
    for ax, seqs, color, title in pairs:
        for seq in seqs:
            drawn = _with_origin(seq)
            ax.plot(
                drawn[:, 0],
                drawn[:, 1],
                color=color,
                lw=0.9,
                alpha=0.8,
                marker="o",
                ms=2.2,
                mew=0,
            )
        ax.set_title(title, loc="left", pad=6)
        ax.set_xlabel("One-way travel time (s)")
        ax.invert_yaxis()
    axes[0].set_ylabel("Depth (m)")
    pad_t = 0.04 * (t_max - t_min)
    axes[0].set_xlim(t_min - pad_t, t_max + pad_t)
    axes[0].set_ylim(z_max * 1.04, 0)
    sns.despine(fig)
    fig.tight_layout()
    _save(fig, "profiles_oneway.png")


def _step_hist(
    ax: Axes, real: np.ndarray, generated: np.ndarray, xlabel: str, bins: int
) -> float:
    frame = pd.DataFrame(
        {
            "value": np.concatenate([real, generated]),
            "group": ["Real"] * len(real) + ["Generated"] * len(generated),
        }
    )
    sns.histplot(
        data=frame,
        x="value",
        hue="group",
        hue_order=["Real", "Generated"],
        palette={"Real": REAL, "Generated": GENERATED},
        element="step",
        fill=True,
        stat="density",
        common_bins=True,
        common_norm=False,
        bins=bins,
        alpha=0.35,
        linewidth=1.05,
        ax=ax,
        legend=False,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ks = float(scipy_stats.ks_2samp(real, generated).statistic)
    ax.text(
        0.97,
        0.95,
        f"KS {ks:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color=MUTED,
    )
    return ks


def plot_histograms(
    real: list[np.ndarray], generated: list[np.ndarray]
) -> dict[str, float]:
    real_z = compute_final_depth(real)
    gen_z = compute_final_depth(generated)
    real_t = np.array([float(seq[-1, 0]) if len(seq) else 0.0 for seq in real])
    gen_t = np.array([float(seq[-1, 0]) if len(seq) else 0.0 for seq in generated])
    real_n = compute_breakpoint_count(real).astype(float)
    gen_n = compute_breakpoint_count(generated).astype(float)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    ks_z = _step_hist(axes[0], real_z, gen_z, "Final depth (m)", 28)
    ks_t = _step_hist(axes[1], real_t, gen_t, "Total one-way time (s)", 28)
    ks_n = _step_hist(axes[2], real_n, gen_n, "Breakpoints", 18)
    axes[0].set_ylabel("Density")
    axes[0].set_xlim(0, 500)
    titles = ("Final depth", "Total travel time", "Breakpoint count")
    letters = ("a", "b", "c")
    for ax, title, letter in zip(axes, titles, letters, strict=True):
        ax.set_title(title, loc="left", pad=8)
        _panel(ax, letter)
    handles = [
        plt.Line2D([0], [0], color=REAL, lw=1.4, label="Real"),
        plt.Line2D([0], [0], color=GENERATED, lw=1.4, label="Generated"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.55, 1.08),
        frameon=False,
    )
    sns.despine(fig)
    fig.tight_layout()
    _save(fig, "histograms_oneway.png")
    return {"depth": ks_z, "tts": ks_t, "breakpoints": ks_n}


def plot_vs(real: list[np.ndarray], generated: list[np.ndarray]) -> dict[str, float]:
    real_vs = compute_all_vs_values(real)
    gen_vs = compute_all_vs_values(generated)
    real_vs = real_vs[np.isfinite(real_vs) & (real_vs > 0) & (real_vs < 1e6)]
    gen_vs = gen_vs[np.isfinite(gen_vs) & (gen_vs > 0) & (gen_vs < 1e6)]
    ks = float(scipy_stats.ks_2samp(real_vs, gen_vs).statistic)
    hi = 1500.0
    real_show = real_vs[real_vs <= hi]
    gen_show = gen_vs[gen_vs <= hi]
    fig, ax = plt.subplots(figsize=(5.0, 2.7))
    _step_hist(ax, real_show, gen_show, "Vs (m/s)", 32)
    ax.texts[-1].set_visible(False)
    ax.set_ylabel("Density")
    ax.set_title("Layer velocity", loc="left", pad=8)
    _panel(ax, "a")
    ax.axvline(float(np.mean(real_vs)), color=REAL, lw=0.8, ls="--")
    ax.axvline(float(np.mean(gen_vs)), color=GENERATED, lw=0.8, ls="--")
    handles = [
        plt.Line2D([0], [0], color=REAL, lw=1.4, label="Real"),
        plt.Line2D([0], [0], color=GENERATED, lw=1.4, label="Generated"),
        plt.Line2D([0], [0], color=REAL, lw=0.8, ls="--", label="Real mean"),
        plt.Line2D([0], [0], color=GENERATED, lw=0.8, ls="--", label="Generated mean"),
    ]
    ax.legend(handles=handles, loc="upper right", ncol=2, frameon=False)
    n_real_out = int(np.sum(real_vs > hi))
    n_gen_out = int(np.sum(gen_vs > hi))
    sns.despine(fig)
    fig.tight_layout()
    _save(fig, "vs_histograms_oneway.png")
    return {
        "ks": ks,
        "real_mean": float(np.mean(real_vs)),
        "real_std": float(np.std(real_vs)),
        "gen_mean": float(np.mean(gen_vs)),
        "gen_std": float(np.std(gen_vs)),
        "n_real_above": n_real_out,
        "n_gen_above": n_gen_out,
        "hi": hi,
    }


def _layer_cloud(sequences: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    depths: list[float] = []
    vs: list[float] = []
    for seq in sequences:
        bounds, values = reconstruct_vs_profile(seq, apply_expm1=False)
        if len(values) == 0:
            continue
        centers = (bounds[:-1] + bounds[1:]) / 2.0
        depths.extend(centers.tolist())
        vs.extend(values.tolist())
    return np.asarray(depths), np.asarray(vs)


def plot_depth_stats(generated: list[np.ndarray]) -> dict[str, float]:
    stats, _ = load_target_statistics(
        str(
            ROOT
            / "experiments"
            / "flow_matching_simplified"
            / "depth_dependent_stats"
            / "real_depth_stats_onewaytime.pkl"
        ),
        str(
            ROOT
            / "experiments"
            / "flow_matching_simplified"
            / "depth_dependent_stats"
            / "real_correlations_onewaytime.pkl"
        ),
    )
    edges = np.arange(0, 510, 10)
    depths, vs = _layer_cloud(generated)
    centers, mean, std, counts = bin_by_depth(depths, vs, edges, apply_log=True)
    real_mean = np.asarray(stats["mean_ln_vs"], dtype=float)
    real_std = np.asarray(stats["std_ln_vs"], dtype=float)
    real_z = np.asarray(stats["bin_centers"], dtype=float)
    valid = (
        np.isfinite(real_mean)
        & np.isfinite(mean)
        & np.isfinite(real_std)
        & np.isfinite(std)
    )
    rmse_mean = float(np.sqrt(np.mean((real_mean[valid] - mean[valid]) ** 2)))
    rmse_std = float(np.sqrt(np.mean((real_std[valid] - std[valid]) ** 2)))

    show = real_z <= 200
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharex=True)
    axes[0].fill_between(
        real_z[show],
        (real_mean - real_std)[show],
        (real_mean + real_std)[show],
        color=BAND,
        alpha=0.35,
        lw=0,
        label="Real ±1 s.d.",
    )
    axes[0].plot(
        real_z[show],
        real_mean[show],
        color=REAL,
        lw=1.3,
        marker="o",
        ms=3,
        label="Real",
    )
    enough = show & (counts >= 5) & np.isfinite(mean)
    sparse = show & (counts > 0) & (counts < 5) & np.isfinite(mean)
    axes[0].plot(
        centers[enough],
        mean[enough],
        color=GENERATED,
        lw=1.2,
        ls="--",
        marker="s",
        ms=3,
        label="Generated",
    )
    axes[0].scatter(
        centers[sparse],
        mean[sparse],
        s=14,
        facecolors="none",
        edgecolors=GENERATED,
        linewidths=0.7,
        zorder=3,
        label="Generated, n < 5",
    )
    axes[1].plot(
        real_z[show], real_std[show], color=REAL, lw=1.3, marker="o", ms=3, label="Real"
    )
    axes[1].plot(
        centers[enough],
        std[enough],
        color=GENERATED,
        lw=1.2,
        ls="--",
        marker="s",
        ms=3,
        label="Generated",
    )
    axes[1].scatter(
        centers[sparse],
        std[sparse],
        s=14,
        facecolors="none",
        edgecolors=GENERATED,
        linewidths=0.7,
        zorder=3,
    )
    axes[0].set_ylabel("ln(Vs)")
    axes[1].set_ylabel("s.d. of ln(Vs)")
    for ax, title, letter in zip(
        axes,
        ("Mean", "Spread"),
        ("a", "b"),
        strict=True,
    ):
        ax.set_xlabel("Depth (m)")
        ax.set_title(title, loc="left", pad=8)
        _panel(ax, letter)
        ax.set_xlim(0, 200)
    axes[0].legend(loc="lower right", borderaxespad=0.3)
    sns.despine(fig)
    fig.tight_layout()
    _save(fig, "depth_statistics_comparison.png")
    return {"rmse_mean": rmse_mean, "rmse_std": rmse_std}


def plot_correlations(generated: list[np.ndarray]) -> None:
    _, real = load_target_statistics(
        str(
            ROOT
            / "experiments"
            / "flow_matching_simplified"
            / "depth_dependent_stats"
            / "real_depth_stats_onewaytime.pkl"
        ),
        str(
            ROOT
            / "experiments"
            / "flow_matching_simplified"
            / "depth_dependent_stats"
            / "real_correlations_onewaytime.pkl"
        ),
    )

    class _Plain:
        def __init__(self, sequences: list[np.ndarray]) -> None:
            self.sequences = sequences
            self.normalize = False

        def __len__(self) -> int:
            return len(self.sequences)

        def denormalize_sequence(self, seq: np.ndarray, idx: int) -> np.ndarray:
            return seq

    gen = compute_profile_statistics(_Plain(generated), lags=[1.0, 2.0, 5.0, 10.0])
    lags = np.array([1.0, 2.0, 5.0, 10.0])
    real_mean = np.array([real["mean_correlations"][lag] for lag in lags])
    gen_mean = np.array([gen["mean_correlations"][lag] for lag in lags])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for mean, color, label, marker in (
        (real_mean, REAL, "Real", "o"),
        (gen_mean, GENERATED, "Generated", "s"),
    ):
        axes[0].plot(
            lags, mean, color=color, lw=1.3, marker=marker, ms=4.5, label=label
        )
    axes[0].axhline(0, color=MUTED, lw=0.5)
    axes[0].set_xscale("log")
    axes[0].set_xticks(lags)
    axes[0].set_xticklabels(["1", "2", "5", "10"])
    axes[0].set_xlabel("Lag (m)")
    axes[0].set_ylabel("Correlation of ln(Vs)")
    axes[0].set_title("Mean across profiles", loc="left", pad=8)
    axes[0].legend(loc="upper left")
    _panel(axes[0], "a")

    lag = 2.0
    real_vals = np.asarray(real["lag_correlations"][lag], dtype=float)
    gen_vals = np.asarray(gen["lag_correlations"][lag], dtype=float)
    real_vals = real_vals[np.isfinite(real_vals)]
    gen_vals = gen_vals[np.isfinite(gen_vals)]
    _step_hist(axes[1], real_vals, gen_vals, "Correlation at 2 m", 30)
    axes[1].texts[-1].set_visible(False)
    axes[1].set_ylabel("Density")
    axes[1].set_title("Profile-wise, 2 m lag", loc="left", pad=8)
    _panel(axes[1], "b")
    handles = [
        plt.Line2D([0], [0], color=REAL, lw=1.4, label="Real"),
        plt.Line2D([0], [0], color=GENERATED, lw=1.4, label="Generated"),
    ]
    axes[1].legend(handles=handles, loc="upper left")
    sns.despine(fig)
    fig.tight_layout()
    _save(fig, "correlations_comparison.png")
    print("corr real", real_mean)
    print("corr gen", gen_mean)


def main() -> None:
    use_nature_style()
    OUT.mkdir(parents=True, exist_ok=True)
    real, generated = _load()
    plot_profiles(real, generated)
    print("hist", plot_histograms(real, generated))
    print("vs", plot_vs(real, generated))
    print("depth", plot_depth_stats(generated))
    plot_correlations(generated)


if __name__ == "__main__":
    main()
