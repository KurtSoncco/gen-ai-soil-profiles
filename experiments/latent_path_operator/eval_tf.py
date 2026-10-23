"""Held-out outcrop transfer function for one transfer architecture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.latent_path_operator.data import TF_FREQS, sample_tf_batch
from experiments.latent_path_operator.operators import build_tf
from experiments.latent_path_operator.train import predict, relative_l1, relative_l2
from soilgen_ai.figures.style import STYLE


def _complex(real_imag: np.ndarray) -> np.ndarray:
    return real_imag[..., 0] + 1j * real_imag[..., 1]


def summarize(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    surface = predicted[:, :, :1]
    truth = target[:, :, :1]
    return {
        "surface_l1": float(relative_l1(surface, truth)),
        "all_depths_l1": float(relative_l1(predicted, target)),
        "surface_l2": float(relative_l2(surface, truth)),
        "all_depths_l2": float(relative_l2(predicted, target)),
    }


def _per_column(predicted: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    """Relative L1, Pearson on concatenated real and imag, and complex misfit."""
    pred = predicted.reshape(len(predicted), -1)
    truth = target.reshape(len(target), -1)
    rel = np.abs(pred - truth).sum(axis=-1) / np.clip(
        np.abs(truth).sum(axis=-1), 1e-8, None
    )
    centered_p = pred - pred.mean(axis=-1, keepdims=True)
    centered_t = truth - truth.mean(axis=-1, keepdims=True)
    denom = np.linalg.norm(centered_p, axis=-1) * np.linalg.norm(centered_t, axis=-1)
    corr = (centered_p * centered_t).sum(axis=-1) / np.clip(denom, 1e-8, None)
    misfit = np.linalg.norm(pred - truth, axis=-1) / np.clip(
        np.linalg.norm(truth, axis=-1), 1e-8, None
    )
    return {"relative_l1": rel, "pearson": corr, "misfit": misfit}


def score_draw(
    predicted: np.ndarray, target: np.ndarray, n_layers: np.ndarray
) -> dict[str, float]:
    """Mean scores for the surface, every depth, the band above 3 Hz, and layer count."""
    high = TF_FREQS >= 3.0
    slices = {
        "surface": (predicted[:, :, :1], target[:, :, :1]),
        "all_depths": (predicted, target),
        "surface_above_3hz": (predicted[:, high, :1], target[:, high, :1]),
        "all_depths_above_3hz": (predicted[:, high], target[:, high]),
    }
    report: dict[str, float] = {}
    for name, (pred, truth) in slices.items():
        columns = _per_column(pred, truth)
        for metric, values in columns.items():
            report[f"{name}_{metric}"] = float(np.mean(values))
    for count in sorted(set(int(n) for n in n_layers)):
        pick = n_layers == count
        if not np.any(pick):
            continue
        columns = _per_column(predicted[pick][:, high], target[pick][:, high])
        for metric, values in columns.items():
            report[f"layers_{count}_above_3hz_{metric}"] = float(np.mean(values))
    deep = n_layers >= 6
    if np.any(deep):
        columns = _per_column(predicted[deep][:, high], target[deep][:, high])
        for metric, values in columns.items():
            report[f"layers_6plus_above_3hz_{metric}"] = float(np.mean(values))
    return report


def peak_metrics(predicted: np.ndarray, exact: np.ndarray) -> dict[str, float]:
    """Error of the largest surface peak on the shared frequency grid."""
    pred_amp = np.abs(predicted[:, :, 0])
    exact_amp = np.abs(exact[:, :, 0])
    pred_at = np.argmax(pred_amp, axis=1)
    exact_at = np.argmax(exact_amp, axis=1)
    rows = np.arange(predicted.shape[0])
    freq_err = np.abs(TF_FREQS[pred_at] - TF_FREQS[exact_at])
    height = pred_amp[rows, pred_at] / np.clip(exact_amp[rows, exact_at], 1e-6, None)
    return {
        "peak_freq_error_hz": float(np.mean(freq_err)),
        "peak_height_ratio": float(np.mean(height)),
    }


def per_column_slices(
    predicted: np.ndarray, target: np.ndarray
) -> dict[str, dict[str, np.ndarray]]:
    """Per-column scores for the surface and for every depth above 3 Hz."""
    high = TF_FREQS >= 3.0
    slices = {
        "surface": (predicted[:, :, :1], target[:, :, :1]),
        "all_depths_above_3hz": (predicted[:, high], target[:, high]),
    }
    return {name: _per_column(pred, truth) for name, (pred, truth) in slices.items()}


def plot_metric_boxes(
    groups: dict[str, dict[str, dict[str, np.ndarray]]],
    path: Path,
) -> None:
    """Two rows (surface, all depths above 3 Hz) and three metrics."""
    STYLE.apply()
    metrics = (
        ("relative_l1", "Relative L1"),
        ("pearson", "Pearson correlation"),
        ("misfit", "Misfit"),
    )
    rows = (
        ("surface", "Surface"),
        ("all_depths_above_3hz", "All depths, above 3 Hz"),
    )
    colors = (STYLE.real, STYLE.flow, STYLE.baseline, STYLE.ink)
    names = list(groups)
    fig, axes = plt.subplots(2, 3, figsize=STYLE.size("double", 1.05))
    letters = "abcdef"
    for row, (slice_key, slice_label) in enumerate(rows):
        for col, (metric, metric_label) in enumerate(metrics):
            ax = axes[row, col]
            data = [groups[name][slice_key][metric] for name in names]
            drawn = ax.boxplot(
                data,
                patch_artist=True,
                widths=0.62,
                medianprops={"color": STYLE.ink, "linewidth": 1.0},
                whiskerprops={"color": STYLE.ink, "linewidth": 0.6},
                capprops={"color": STYLE.ink, "linewidth": 0.6},
                flierprops={
                    "marker": "o",
                    "markersize": 2.5,
                    "markerfacecolor": STYLE.muted,
                    "markeredgecolor": STYLE.muted,
                },
            )
            for patch, color in zip(drawn["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_edgecolor(STYLE.ink)
                patch.set_linewidth(0.6)
            ax.set_xticks(range(1, len(names) + 1))
            ax.set_xticklabels(names)
            ax.set_ylabel(f"{slice_label}\n{metric_label}")
            STYLE.panel(ax, letters[row * 3 + col])
    fig.tight_layout()
    STYLE.save(fig, path)


def collect_operator_columns(
    n: int = 32, seed: int = 11
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Per-column scores for each saved transfer checkpoint on one draw."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = sample_tf_batch(n, rng=np.random.default_rng(seed)).to(device)
    truth = batch.target.detach().cpu().numpy()
    groups = {}
    for arch in ("modal", "laplace", "wavelet", "chain"):
        model = build_tf(arch).to(device)
        ckpt = torch.load(
            f"outputs/latent_path_operator/tf_{arch}.pt",
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(ckpt["model"])
        model.eval()
        with torch.no_grad():
            predicted = predict(model, batch).detach().cpu().numpy()
        groups[arch] = per_column_slices(predicted, truth)
        del model
    return groups


def plot_profile_spectra(
    thickness: list[np.ndarray],
    velocity: list[np.ndarray],
    exact: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    path: Path,
) -> None:
    """Vs profiles and the surface spectra of the solver and both chains."""
    STYLE.apply()
    fig, axes = plt.subplots(2, 3, figsize=STYLE.size("double", 1.05))
    letters = "abcdef"
    for index in range(3):
        ax = axes[0, index]
        edges = np.concatenate([[0.0], np.cumsum(thickness[index])])
        speed = np.concatenate([velocity[index], velocity[index][-1:]])
        ax.step(speed, edges, where="post", color=STYLE.real)
        ax.invert_yaxis()
        ax.set_xlabel("Vs (m/s)")
        ax.set_ylabel("Depth (m)")
        STYLE.panel(ax, letters[index])
        ax = axes[1, index]
        ax.plot(TF_FREQS, exact[index], color=STYLE.real, label="Exact")
        ax.plot(TF_FREQS, before[index], color=STYLE.flow, ls="--", label="Chain")
        ax.plot(
            TF_FREQS, after[index], color=STYLE.baseline, ls="--", label="Fine-tuned"
        )
        ax.set_xscale("log")
        ax.set_xlim(0.01, 20)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Surface |TF|")
        if index == 0:
            ax.legend()
        STYLE.panel(ax, letters[3 + index])
    fig.tight_layout()
    STYLE.save(fig, path)


def plot_tf(predicted: np.ndarray, exact: np.ndarray, path: Path) -> None:
    STYLE.apply()
    fig, axes = plt.subplots(1, 2, figsize=STYLE.size("double", 0.62))
    freq = TF_FREQS
    ax = axes[0]
    for index, label in enumerate(("Column 1", "Column 2", "Column 3")):
        color = (STYLE.real, STYLE.flow, STYLE.baseline)[index]
        ax.plot(
            freq,
            np.abs(exact[index, :, 0]),
            color=color,
            label=f"Exact {label.lower()}",
        )
        ax.plot(freq, np.abs(predicted[index, :, 0]), color=color, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Surface |TF|")
    ax.legend()
    STYLE.panel(ax, "a")

    ax = axes[1]
    ax.plot(freq, np.abs(exact[0, :, 0]), color=STYLE.real, label="Exact surface")
    ax.plot(
        freq, np.abs(predicted[0, :, 0]), color=STYLE.flow, ls="--", label="Operator"
    )
    ax.plot(freq, np.abs(exact[0, :, 1]), color=STYLE.ink, label="Exact mid-depth")
    ax.plot(
        freq,
        np.abs(predicted[0, :, 1]),
        color=STYLE.ink,
        ls="--",
        label="Operator mid-depth",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("|TF|")
    ax.legend()
    STYLE.panel(ax, "b")
    fig.tight_layout()
    STYLE.save(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score one transfer-function operator")
    parser.add_argument(
        "--arch",
        choices=["modal", "laplace", "wavelet", "chain"],
        default="modal",
    )
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--boxes", action="store_true")
    args = parser.parse_args()
    if args.boxes:
        groups = collect_operator_columns(n=args.n, seed=args.seed)
        labels = {
            "modal": "Modal",
            "laplace": "Laplace",
            "wavelet": "Wavelet",
            "chain": "Chain",
        }
        plot_metric_boxes(
            {labels[name]: groups[name] for name in labels},
            Path("outputs/latent_path_operator/tf_metric_boxes.png"),
        )
        return
    checkpoint = args.checkpoint or f"outputs/latent_path_operator/tf_{args.arch}.pt"
    figure = args.out or f"outputs/latent_path_operator/tf_{args.arch}.png"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_tf(args.arch).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    batch = sample_tf_batch(args.n, rng=np.random.default_rng(args.seed)).to(device)
    with torch.no_grad():
        predicted = predict(model, batch)
        stats = summarize(predicted, batch.target)
    curves = predicted.detach().cpu().numpy()
    truth = batch.target.detach().cpu().numpy()
    n_layers = batch.layer_mask.detach().cpu().numpy().sum(axis=-1)
    scores = score_draw(curves, truth, n_layers)
    peaks = peak_metrics(_complex(curves), _complex(truth))
    scores.update(stats)
    scores.update(peaks)
    print(
        f"{args.arch}  surface L1 {scores['surface_relative_l1']:.4f}  "
        f"Pearson {scores['surface_pearson']:.4f}  "
        f"misfit {scores['surface_misfit']:.4f}  "
        f"above 3 Hz L1 {scores['surface_above_3hz_relative_l1']:.4f}  "
        f"Pearson {scores['surface_above_3hz_pearson']:.4f}  "
        f"misfit {scores['surface_above_3hz_misfit']:.4f}"
    )
    if "layers_6plus_above_3hz_relative_l1" in scores:
        print(
            f"{args.arch}  6+ layers above 3 Hz  "
            f"L1 {scores['layers_6plus_above_3hz_relative_l1']:.4f}  "
            f"Pearson {scores['layers_6plus_above_3hz_pearson']:.4f}  "
            f"misfit {scores['layers_6plus_above_3hz_misfit']:.4f}"
        )
    report = Path(checkpoint).with_suffix(".json")
    report.write_text(json.dumps(scores, indent=2, sort_keys=True))
    plot_tf(_complex(curves), _complex(truth), Path(figure))


if __name__ == "__main__":
    main()
