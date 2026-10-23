"""Held-out Ricker velocity for the trained shear-wave operator."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.latent_path_operator.data import sample_sh_ricker_batch
from experiments.latent_path_operator.model import LatentPathOperator
from experiments.latent_path_operator.ricker import (
    peak_lag_ms,
    peak_ratio,
    ricker,
    time_axis,
    velocity_traces,
)
from experiments.latent_path_operator.spec import SH_SPEC
from experiments.latent_path_operator.train import predict
from soilgen_ai.figures.style import STYLE


def surface_traces(model: LatentPathOperator, batch) -> tuple[np.ndarray, np.ndarray]:
    """Predicted and exact velocity, both (B, 3, T). Exact is ``batch.target``."""
    predicted = predict(model, batch).detach().cpu().numpy()
    wavelet = batch.wavelet.detach().cpu().numpy()
    traces = np.stack(
        [
            velocity_traces(
                predicted[i, ..., 0] + 1j * predicted[i, ..., 1], wavelet[i]
            )
            for i in range(predicted.shape[0])
        ]
    )
    return traces, batch.target.detach().cpu().numpy()


def summarize(predicted: np.ndarray, exact: np.ndarray) -> dict[str, float]:
    surface_p = predicted[:, 0, :]
    surface_e = exact[:, 0, :]
    rel = [
        np.linalg.norm(p - e) / np.linalg.norm(e) for p, e in zip(surface_p, surface_e)
    ]
    ratios = [peak_ratio(p, e) for p, e in zip(surface_p, surface_e)]
    lags = [peak_lag_ms(p, e) for p, e in zip(surface_p, surface_e)]
    return {
        "relative_l2": float(np.mean(rel)),
        "peak_ratio": float(np.mean(ratios)),
        "peak_lag_ms": float(np.mean(lags)),
    }


def plot_ricker(
    predicted: np.ndarray,
    exact: np.ndarray,
    path: Path,
    f_peak: float = 5.0,
) -> None:
    STYLE.apply()
    time = time_axis()
    fig, axes = plt.subplots(2, 2, figsize=STYLE.size("double", 0.85), sharex=True)
    outcrop = ricker(time, f_peak)
    panels = (
        (axes[0, 0], "a", outcrop, None, "Outcrop velocity"),
        (axes[0, 1], "b", exact[0, 0], predicted[0, 0], "Surface, column 1"),
        (axes[1, 0], "c", exact[1, 0], predicted[1, 0], "Surface, column 2"),
        (axes[1, 1], "d", None, None, "Column 1 at depth"),
    )
    for ax, letter, truth, guess, label in panels[:3]:
        ax.plot(
            time,
            truth,
            color=STYLE.real,
            label="Exact" if guess is not None else "Ricker",
        )
        if guess is not None:
            ax.plot(time, guess, color=STYLE.flow, label="Operator")
            ax.legend()
        ax.set_ylabel("Velocity")
        ax.text(
            0.0,
            1.02,
            label,
            transform=ax.transAxes,
            fontsize=STYLE.label_pt,
            va="bottom",
        )
        STYLE.panel(ax, letter)
    ax = axes[1, 1]
    names = ("Surface", "Mid-depth", "Bedrock interface")
    colors = (STYLE.real, STYLE.baseline, STYLE.ink)
    for depth, name, color in zip(exact[0], names, colors):
        ax.plot(time, depth, color=color, label=name)
    for depth, color in zip(predicted[0], colors):
        ax.plot(time, depth, color=color, ls="--")
    ax.plot([], [], color=STYLE.ink, label="Exact")
    ax.plot([], [], color=STYLE.ink, ls="--", label="Operator")
    ax.legend(ncol=1)
    STYLE.panel(ax, "d")
    ax.text(
        0.0,
        1.02,
        "Column 1 at depth",
        transform=ax.transAxes,
        fontsize=STYLE.label_pt,
        va="bottom",
    )
    for ax in axes.ravel():
        ax.set_xlim(0.0, 2.5)
    for ax in axes[1]:
        ax.set_xlabel("Time (s)")
    fig.tight_layout()
    STYLE.save(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score Ricker velocity on held-out columns"
    )
    parser.add_argument("--checkpoint", default="outputs/latent_path_operator/sh.pt")
    parser.add_argument(
        "--out", default="outputs/latent_path_operator/ricker_velocity.png"
    )
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--f-peak", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = LatentPathOperator(SH_SPEC).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    batch = sample_sh_ricker_batch(
        args.n, rng=np.random.default_rng(args.seed), f_peak=args.f_peak
    ).to(device)
    with torch.no_grad():
        predicted, exact = surface_traces(model, batch)
    stats = summarize(predicted, exact)
    print(
        f"surface relative L2 {stats['relative_l2']:.4f}  "
        f"peak ratio {stats['peak_ratio']:.3f}  "
        f"peak lag {stats['peak_lag_ms']:.2f} ms"
    )
    plot_ricker(predicted, exact, Path(args.out), f_peak=args.f_peak)


if __name__ == "__main__":
    main()
