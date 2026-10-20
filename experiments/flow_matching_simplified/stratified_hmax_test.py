"""Generate 3- and 4-layer profiles with bedrock depth H in [15, 100] m.

Pins the attention-mask length and draws sequence_stats from real profiles
already in that (n_layers, H) cell. Reports physics rates, quarter-wavelength
f0 = 1 / (4 * TTS_to_bedrock), and generated-vs-real statistics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats as scipy_stats
from scipy.stats import wasserstein_distance

from experiments.flow_matching_simplified import config as cfg_mod
from experiments.flow_matching_simplified.bias_analysis import (
    _to_json,
    load_model,
    sample_conditional,
    sequence_stats_from_raw,
)
from experiments.flow_matching_simplified.data import FlowMatchingDataLoader
from experiments.flow_matching_simplified.utils import compute_vs_from_sequence

EXPORT = Path("/home/kurt-/fm-vspdb-5k-export")
OUT_DIR = EXPORT / "stratified_hmax_test"
H_MIN = 15.0
H_MAX = 100.0
LAYERS = (3, 4)
TARGET = 1000
SEED = 42
VS_MIN = 100.0
VS_MAX = 5000.0
MIN_THICK = 0.5  # matches real breakpoint floor


def with_origin(seq: np.ndarray) -> np.ndarray:
    seq = np.asarray(seq, dtype=np.float64)
    if len(seq) == 0:
        return seq
    if seq[0, 0] != 0.0 or seq[0, 1] != 0.0:
        return np.vstack([[0.0, 0.0], seq])
    return seq


def profile_physics(seq: np.ndarray) -> dict:
    s = with_origin(seq)
    out = {
        "n_layers": 0,
        "H": np.nan,
        "tts": np.nan,
        "f0": np.nan,
        "vs_eq": np.nan,
        "vs30": np.nan,
        "tts_mono": False,
        "depth_mono": False,
        "nonneg": False,
        "vs_ok": False,
        "h_ok": False,
        "thick_ok": False,
        "valid": False,
        "thickness": [],
        "vs_layers": [],
        "contrast": np.nan,
        "vs_last": np.nan,
        "vs_first": np.nan,
    }
    if len(s) < 2:
        return out
    tts, z = s[:, 0], s[:, 1]
    dz = np.diff(z)
    dt = np.diff(tts)
    vs = compute_vs_from_sequence(seq)
    h = float(z[-1])
    t_end = float(tts[-1])
    n_layers = int(len(dz))
    tts_mono = bool(np.all(dt > 1e-9))
    depth_mono = bool(np.all(dz > 1e-9))
    nonneg = bool(np.all(tts >= -1e-12) and np.all(z >= -1e-12))
    vs_ok = bool(len(vs) and np.all(np.isfinite(vs)) and np.all((vs >= VS_MIN) & (vs <= VS_MAX)))
    thick_ok = bool(np.all(dz >= MIN_THICK - 1e-9))
    h_ok = bool(H_MIN - 1e-9 <= h <= H_MAX + 1e-9)
    vs_eq = h / t_end if t_end > 1e-9 and h > 0 else np.nan
    f0 = 1.0 / (4.0 * t_end) if t_end > 1e-9 else np.nan
    if h >= 30.0 and t_end > 1e-9:
        t30 = float(np.interp(30.0, z, tts))
        vs30 = 30.0 / t30 if t30 > 1e-9 else np.nan
    else:
        vs30 = vs_eq
    contrast = float(np.max(vs) / np.min(vs)) if len(vs) and np.min(vs) > 0 else np.nan
    out.update(
        {
            "n_layers": n_layers,
            "H": h,
            "tts": t_end,
            "f0": float(f0) if np.isfinite(f0) else np.nan,
            "vs_eq": float(vs_eq) if np.isfinite(vs_eq) else np.nan,
            "vs30": float(vs30) if np.isfinite(vs30) else np.nan,
            "tts_mono": tts_mono,
            "depth_mono": depth_mono,
            "nonneg": nonneg,
            "vs_ok": vs_ok,
            "h_ok": h_ok,
            "thick_ok": thick_ok,
            "valid": bool(tts_mono and depth_mono and nonneg and vs_ok and thick_ok and h_ok),
            "thickness": [float(x) for x in dz],
            "vs_layers": [float(x) for x in vs],
            "contrast": contrast,
            "vs_last": float(vs[-1]) if len(vs) else np.nan,
            "vs_first": float(vs[0]) if len(vs) else np.nan,
        }
    )
    return out


def summarize(xs: np.ndarray) -> dict:
    xs = np.asarray(xs, dtype=float)
    xs = xs[np.isfinite(xs)]
    if len(xs) == 0:
        return {}
    qs = [0, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    return {
        "n": int(len(xs)),
        "mean": float(np.mean(xs)),
        "std": float(np.std(xs)),
        "percentiles": {str(q): float(np.percentile(xs, q)) for q in qs},
    }


def two_sample(real: np.ndarray, gen: np.ndarray) -> dict:
    real = np.asarray(real, dtype=float)
    gen = np.asarray(gen, dtype=float)
    real = real[np.isfinite(real)]
    gen = gen[np.isfinite(gen)]
    if len(real) < 5 or len(gen) < 5:
        return {}
    ks = scipy_stats.ks_2samp(real, gen)
    return {
        "ks": float(ks.statistic),
        "ks_p": float(ks.pvalue),
        "w1": float(wasserstein_distance(real, gen)),
        "mean_real": float(np.mean(real)),
        "mean_gen": float(np.mean(gen)),
        "median_real": float(np.median(real)),
        "median_gen": float(np.median(gen)),
        "std_real": float(np.std(real)),
        "std_gen": float(np.std(gen)),
    }


def hist_counts(xs: np.ndarray, edges: np.ndarray) -> list[float]:
    xs = np.asarray(xs, dtype=float)
    xs = xs[np.isfinite(xs)]
    counts, _ = np.histogram(xs, bins=edges, density=True)
    return [float(c) for c in counts]


def layer_means(recs: list[dict], n_layers: int) -> dict:
    th = [[] for _ in range(n_layers)]
    vs = [[] for _ in range(n_layers)]
    for r in recs:
        if len(r["thickness"]) != n_layers:
            continue
        for i in range(n_layers):
            th[i].append(r["thickness"][i])
            vs[i].append(r["vs_layers"][i])
    return {
        "thickness_mean": [float(np.mean(x)) if x else np.nan for x in th],
        "thickness_median": [float(np.median(x)) if x else np.nan for x in th],
        "vs_mean": [float(np.mean(x)) if x else np.nan for x in vs],
        "vs_median": [float(np.median(x)) if x else np.nan for x in vs],
    }


def generate_until(
    model,
    donors: list[dict],
    n_layers: int,
    max_length: int,
    device,
    target: int,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], list[dict], dict]:
    kept_seqs: list[np.ndarray] = []
    kept_rec: list[dict] = []
    rates = {
        "raw": 0,
        "tts_mono": 0,
        "depth_mono": 0,
        "nonneg": 0,
        "vs_ok": 0,
        "thick_ok": 0,
        "h_ok": 0,
        "valid": 0,
    }
    batch = 64
    while len(kept_seqs) < target:
        idx = rng.integers(0, len(donors), size=batch)
        stats = [donors[i]["stats"] for i in idx]
        lengths = [n_layers] * batch
        seqs = sample_conditional(
            model, stats, lengths, max_length, device, ode_steps=50, batch_size=batch
        )
        for seq in seqs:
            rec = profile_physics(seq)
            rec["n_layers_requested"] = n_layers
            rates["raw"] += 1
            rates["tts_mono"] += int(rec["tts_mono"])
            rates["depth_mono"] += int(rec["depth_mono"])
            rates["nonneg"] += int(rec["nonneg"])
            rates["vs_ok"] += int(rec["vs_ok"])
            rates["thick_ok"] += int(rec["thick_ok"])
            rates["h_ok"] += int(rec["h_ok"])
            rates["valid"] += int(rec["valid"] and rec["n_layers"] == n_layers)
            if rec["valid"] and rec["n_layers"] == n_layers:
                kept_seqs.append(seq)
                kept_rec.append(rec)
                if len(kept_seqs) >= target:
                    break
        print(
            f"  L={n_layers} kept {len(kept_seqs)}/{target}  "
            f"accept={rates['valid'] / rates['raw']:.1%}  raw={rates['raw']}",
            flush=True,
        )
    for k, v in list(rates.items()):
        if k != "raw":
            rates[f"{k}_rate"] = v / rates["raw"] if rates["raw"] else 0.0
    return kept_seqs[:target], kept_rec[:target], rates


def main() -> None:
    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = cfg_mod.cfg
    loader = FlowMatchingDataLoader(data_path=Path(cfg.data_path))
    loader.load_data()
    assert loader.sequences is not None

    real_all = []
    for seq in loader.sequences:
        rec = profile_physics(seq)
        rec["stats"] = sequence_stats_from_raw(seq)
        rec["seq"] = seq
        real_all.append(rec)

    device = torch.device("cpu")
    model = load_model(EXPORT / "checkpoints" / "latest.pt", device)

    f0_edges = np.linspace(0.2, 6.0, 25)
    h_edges = np.linspace(H_MIN, H_MAX, 18)
    vs_edges = np.linspace(50, 800, 21)

    results: dict = {
        "spec": {
            "h_min": H_MIN,
            "h_max": H_MAX,
            "layers": list(LAYERS),
            "target": TARGET,
            "f0": "1 / (4 * TTS_to_bedrock), quarter-wavelength, one-way TTS",
            "checkpoint": "latest.pt",
            "seed": SEED,
            "min_thickness_m": MIN_THICK,
            "vs_bounds": [VS_MIN, VS_MAX],
        },
        "by_layers": {},
    }

    for L in LAYERS:
        donors = [
            r
            for r in real_all
            if r["n_layers"] == L and r["h_ok"] and r["tts_mono"] and r["depth_mono"]
        ]
        print(f"Real donors L={L}, H in [{H_MIN},{H_MAX}]: {len(donors)}")
        seqs, recs, rates = generate_until(
            model, donors, L, cfg.max_length, device, TARGET, rng
        )
        np.save(OUT_DIR / f"generated_L{L}.npy", np.array(seqs, dtype=object), allow_pickle=True)

        real_f0 = np.array([r["f0"] for r in donors])
        gen_f0 = np.array([r["f0"] for r in recs])
        real_h = np.array([r["H"] for r in donors])
        gen_h = np.array([r["H"] for r in recs])
        real_tts = np.array([r["tts"] for r in donors])
        gen_tts = np.array([r["tts"] for r in recs])
        real_vseq = np.array([r["vs_eq"] for r in donors])
        gen_vseq = np.array([r["vs_eq"] for r in recs])
        real_vs30 = np.array([r["vs30"] for r in donors])
        gen_vs30 = np.array([r["vs30"] for r in recs])
        real_th = np.concatenate([np.array(r["thickness"]) for r in donors])
        gen_th = np.concatenate([np.array(r["thickness"]) for r in recs])
        real_contr = np.array([r["contrast"] for r in donors])
        gen_contr = np.array([r["contrast"] for r in recs])
        real_vlast = np.array([r["vs_last"] for r in donors])
        gen_vlast = np.array([r["vs_last"] for r in recs])

        results["by_layers"][str(L)] = {
            "n_real_donors": len(donors),
            "n_generated_kept": len(recs),
            "acceptance": rates,
            "real": {
                "f0": summarize(real_f0),
                "H": summarize(real_h),
                "tts": summarize(real_tts),
                "vs_eq": summarize(real_vseq),
                "vs30": summarize(real_vs30),
                "thickness": summarize(real_th),
                "contrast": summarize(real_contr),
                "vs_last": summarize(real_vlast),
                "per_layer": layer_means(donors, L),
            },
            "generated": {
                "f0": summarize(gen_f0),
                "H": summarize(gen_h),
                "tts": summarize(gen_tts),
                "vs_eq": summarize(gen_vseq),
                "vs30": summarize(gen_vs30),
                "thickness": summarize(gen_th),
                "contrast": summarize(gen_contr),
                "vs_last": summarize(gen_vlast),
                "per_layer": layer_means(recs, L),
            },
            "compare": {
                "f0": two_sample(real_f0, gen_f0),
                "H": two_sample(real_h, gen_h),
                "tts": two_sample(real_tts, gen_tts),
                "vs_eq": two_sample(real_vseq, gen_vseq),
                "vs30": two_sample(real_vs30, gen_vs30),
                "thickness": two_sample(real_th, gen_th),
                "contrast": two_sample(real_contr, gen_contr),
                "vs_last": two_sample(real_vlast, gen_vlast),
            },
            "hist": {
                "f0_edges": [float(x) for x in f0_edges],
                "f0_real": hist_counts(real_f0, f0_edges),
                "f0_gen": hist_counts(gen_f0, f0_edges),
                "h_edges": [float(x) for x in h_edges],
                "h_real": hist_counts(real_h, h_edges),
                "h_gen": hist_counts(gen_h, h_edges),
                "vs_eq_edges": [float(x) for x in vs_edges],
                "vs_eq_real": hist_counts(real_vseq, vs_edges),
                "vs_eq_gen": hist_counts(gen_vseq, vs_edges),
            },
        }

    out = OUT_DIR / "results.json"
    out.write_text(json.dumps(_to_json(results), indent=2))
    print(f"Wrote {out}")
    plot_histogram_figure()


def _collect_stratum(n_layers: int) -> tuple[list[dict], list[dict]]:
    cfg = cfg_mod.cfg
    loader = FlowMatchingDataLoader(data_path=Path(cfg.data_path))
    loader.load_data()
    assert loader.sequences is not None
    real = []
    for seq in loader.sequences:
        rec = profile_physics(seq)
        if rec["n_layers"] == n_layers and rec["h_ok"] and rec["tts_mono"] and rec["depth_mono"]:
            real.append(rec)
    gen_path = OUT_DIR / f"generated_L{n_layers}.npy"
    gen_seqs = np.load(gen_path, allow_pickle=True)
    gen = [profile_physics(np.asarray(s, dtype=np.float64)) for s in gen_seqs]
    return real, gen


def _hist(ax, real, gen, bins, xlabel, title, xlim=None):
    real = np.asarray(real, dtype=float)
    gen = np.asarray(gen, dtype=float)
    real = real[np.isfinite(real)]
    gen = gen[np.isfinite(gen)]
    ax.hist(real, bins=bins, density=True, alpha=0.55, color="#4C78A8", label="Real", edgecolor="none")
    ax.hist(gen, bins=bins, density=True, alpha=0.45, color="#E45756", label="Generated", edgecolor="none")
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    if xlim is not None:
        ax.set_xlim(*xlim)


def plot_histogram_figure() -> Path:
    """2×7 histograms: H, f0, TTS, Vs_i for 3-layer (row 1) and 4-layer (row 2)."""
    real3, gen3 = _collect_stratum(3)
    real4, gen4 = _collect_stratum(4)

    h_bins = np.linspace(H_MIN, H_MAX, 22)
    f0_bins = np.linspace(0.2, 8.0, 22)
    tts_bins = np.linspace(0.0, 0.65, 22)
    vs_bins = np.linspace(50, 1000, 22)

    fig, axes = plt.subplots(2, 7, figsize=(22, 7.2), sharey=False)
    fig.suptitle(
        "3- and 4-layer generation vs real donors  ·  H ∈ [15, 100] m  ·  latest.pt",
        fontsize=13,
        y=1.02,
    )

    def row(axes_row, real, gen, n_layers, ylabel):
        _hist(
            axes_row[0],
            [r["H"] for r in real],
            [r["H"] for r in gen],
            h_bins,
            "H (m)",
            f"{n_layers}-layer  total height H",
            xlim=(H_MIN, H_MAX),
        )
        _hist(
            axes_row[1],
            [r["f0"] for r in real],
            [r["f0"] for r in gen],
            f0_bins,
            r"$f_0$ (Hz)",
            f"{n_layers}-layer  $f_0 = 1/(4\\,\\mathrm{{TTS}}_H)$",
            xlim=(0.2, 8.0),
        )
        _hist(
            axes_row[2],
            [r["tts"] for r in real],
            [r["tts"] for r in gen],
            tts_bins,
            "TTS to bedrock (s)",
            f"{n_layers}-layer  total TTS",
            xlim=(0.0, 0.65),
        )
        axes_row[0].set_ylabel("Density")
        for i in range(4):
            ax = axes_row[3 + i]
            if i >= n_layers:
                ax.axis("off")
                continue
            _hist(
                ax,
                [r["vs_layers"][i] for r in real if len(r["vs_layers"]) == n_layers],
                [r["vs_layers"][i] for r in gen if len(r["vs_layers"]) == n_layers],
                vs_bins,
                rf"$V_{{s,{i+1}}}$ (m/s)",
                f"{n_layers}-layer  " + rf"$V_{{s,{i+1}}}$",
                xlim=(50, 1000),
            )
        axes_row[0].legend(loc="upper right", fontsize=8)

    row(axes[0], real3, gen3, 3, "3-layer")
    row(axes[1], real4, gen4, 4, "4-layer")

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "histograms_H_f0_tts_Vs.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    clone_fig_dir = Path(
        "experiments/flow_matching_simplified/outputs/figures"
    )
    clone_fig_dir.mkdir(parents=True, exist_ok=True)
    clone_path = clone_fig_dir / "stratified_hmax_histograms.png"
    fig.savefig(clone_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")
    print(f"Wrote {clone_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Rebuild the histogram figure from saved samples (no generation).",
    )
    args = parser.parse_args()
    if args.plot_only:
        plot_histogram_figure()
    else:
        main()
