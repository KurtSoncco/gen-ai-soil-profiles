"""Bias and conditional-generation analysis for the flagship flow-matching model.

Evaluation protocol follows metrics used in recent flow-matching papers:
- Precision / Recall / Density / Coverage (Naeem et al. 2020; used in Weighted CFM)
- 1-NNA, Coverage (COV), Minimum Matching Distance (TreeFlow, Yang et al.)
- Per-stratum Wasserstein-1 and KS (scientific / conditional FM)
- Conditional fidelity + mean-reversion (dominant-mode bias; SubFlow 2026)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy import stats as scipy_stats
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance

from experiments.flow_matching_simplified import config as cfg_mod
from experiments.flow_matching_simplified.data import FlowMatchingDataLoader
from experiments.flow_matching_simplified.model import TransformerModel
from experiments.flow_matching_simplified.utils import compute_vs_from_sequence

EXPORT = Path("/home/kurt-/fm-vspdb-5k-export")
OUT_DIR = EXPORT / "bias_analysis"
NEHRP_EDGES = [0.0, 180.0, 360.0, 760.0, 1500.0, 1e9]
NEHRP_LABELS = ["E (<180)", "D (180-360)", "C (360-760)", "B (760-1500)", "A (>1500)"]
FEATURE_NAMES = ["log_vs30", "log_H", "n_layers", "log_vs_mean", "log_contrast"]
K_PRDC = 5
SEED = 42


def _to_json(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json(v) for v in obj]
    return obj


def nehrp_class(vs30: float) -> str:
    if not np.isfinite(vs30) or vs30 <= 0:
        return "invalid"
    if vs30 < 180:
        return "E (<180)"
    if vs30 < 360:
        return "D (180-360)"
    if vs30 < 760:
        return "C (360-760)"
    if vs30 < 1500:
        return "B (760-1500)"
    return "A (>1500)"


def vs30_from_sequence(seq: np.ndarray) -> float:
    """Travel-time equivalent Vs30. TTS here is one-way (Vs = dz/dt)."""
    if seq is None or len(seq) == 0:
        return np.nan
    seq = np.asarray(seq, dtype=np.float64)
    if seq[0, 0] != 0.0 or seq[0, 1] != 0.0:
        seq = np.vstack([[0.0, 0.0], seq])
    tts = seq[:, 0]
    depth = seq[:, 1]
    order = np.argsort(depth)
    depth, tts = depth[order], tts[order]
    # drop non-increasing depth
    keep = np.concatenate([[True], np.diff(depth) > 1e-9])
    depth, tts = depth[keep], tts[keep]
    if len(depth) < 2:
        return np.nan
    h = float(depth[-1])
    t_end = float(tts[-1])
    if h <= 0 or t_end <= 1e-9:
        return np.nan
    if h >= 30.0:
        t30 = float(np.interp(30.0, depth, tts))
        if t30 <= 1e-9:
            return np.nan
        return 30.0 / t30
    return h / t_end


def profile_features(seq: np.ndarray) -> dict[str, float]:
    seq = np.asarray(seq, dtype=np.float64)
    vs = compute_vs_from_sequence(seq)
    vs = vs[np.isfinite(vs) & (vs > 0)]
    vs30 = vs30_from_sequence(seq)
    if len(seq) == 0:
        h = np.nan
    else:
        h = float(np.max(seq[:, 1]))
    n_layers = int(len(vs)) if len(vs) else max(len(seq) - 1, 0)
    vs_mean = float(np.mean(vs)) if len(vs) else np.nan
    contrast = float(np.max(vs) / np.clip(np.min(vs), 1e-6, None)) if len(vs) else np.nan
    cls = nehrp_class(vs30)
    return {
        "vs30": float(vs30) if np.isfinite(vs30) else np.nan,
        "H": h,
        "n_layers": float(n_layers),
        "vs_mean": vs_mean,
        "contrast": contrast,
        "class": cls,
        "valid": bool(
            np.isfinite(vs30)
            and np.isfinite(h)
            and h > 0
            and n_layers >= 1
            and np.isfinite(vs_mean)
            and vs_mean > 0
        ),
    }


def feature_matrix(records: list[dict]) -> np.ndarray:
    rows = []
    for r in records:
        if not r["valid"]:
            continue
        contrast = r["contrast"] if np.isfinite(r["contrast"]) and r["contrast"] > 0 else 1.0
        rows.append(
            [
                np.log(max(r["vs30"], 1.0)),
                np.log(max(r["H"], 0.5)),
                r["n_layers"],
                np.log(max(r["vs_mean"], 1.0)),
                np.log(max(contrast, 1.0)),
            ]
        )
    return np.asarray(rows, dtype=np.float64)


def standardize(real: np.ndarray, fake: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = real.mean(axis=0)
    sd = real.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (real - mu) / sd, (fake - mu) / sd


def knn_radii(x: np.ndarray, k: int) -> np.ndarray:
    d = cdist(x, x, metric="euclidean")
    np.fill_diagonal(d, np.inf)
    k = min(k, x.shape[0] - 1)
    return np.sort(d, axis=1)[:, k - 1]


def prdc(real: np.ndarray, fake: np.ndarray, k: int = K_PRDC) -> dict[str, float]:
    """Naeem et al. 2020 density/coverage + Kynkäänniemi precision/recall."""
    if len(real) < k + 1 or len(fake) < k + 1:
        return {"precision": np.nan, "recall": np.nan, "density": np.nan, "coverage": np.nan}
    real_r = knn_radii(real, k)
    fake_r = knn_radii(fake, k)
    d_fr = cdist(fake, real, metric="euclidean")
    precision = float(np.mean(np.any(d_fr < real_r[None, :], axis=1)))
    recall = float(np.mean(np.any(cdist(real, fake) < fake_r[None, :], axis=1)))
    density = float((1.0 / k) * np.mean(np.sum(d_fr < real_r[None, :], axis=1)))
    coverage = float(np.mean(np.any(d_fr < real_r[None, :], axis=0)))
    return {
        "precision": precision,
        "recall": recall,
        "density": density,
        "coverage": coverage,
    }


def nna_cov_mmd(real: np.ndarray, fake: np.ndarray) -> dict[str, float]:
    """1-NNA, COV, MMD in feature space (Yang / TreeFlow protocol)."""
    n_r, n_f = len(real), len(fake)
    if n_r < 2 or n_f < 2:
        return {"one_nna": np.nan, "cov": np.nan, "mmd": np.nan}
    d_rf = cdist(real, fake)
    d_rr = cdist(real, real)
    d_ff = cdist(fake, fake)
    np.fill_diagonal(d_rr, np.inf)
    np.fill_diagonal(d_ff, np.inf)

    real_to_fake = d_rf.min(axis=1)
    fake_to_real = d_rf.min(axis=0)
    real_to_real = d_rr.min(axis=1)
    fake_to_fake = d_ff.min(axis=1)

    real_nn_is_fake = real_to_fake < real_to_real
    fake_nn_is_real = fake_to_real < fake_to_fake
    one_nna = float((real_nn_is_fake.sum() + fake_nn_is_real.sum()) / (n_r + n_f))

    nearest_real_of_fake = d_rf.argmin(axis=0)
    cov = float(len(np.unique(nearest_real_of_fake)) / n_r)
    mmd = float(np.mean(real_to_fake))
    return {"one_nna": one_nna, "cov": cov, "mmd": mmd}


def scalar_tests(real: np.ndarray, fake: np.ndarray) -> dict[str, float]:
    real = real[np.isfinite(real)]
    fake = fake[np.isfinite(fake)]
    if len(real) < 5 or len(fake) < 5:
        return {"ks": np.nan, "ks_p": np.nan, "w1": np.nan, "mean_real": np.nan, "mean_gen": np.nan}
    ks = scipy_stats.ks_2samp(real, fake)
    return {
        "ks": float(ks.statistic),
        "ks_p": float(ks.pvalue),
        "w1": float(wasserstein_distance(real, fake)),
        "mean_real": float(np.mean(real)),
        "mean_gen": float(np.mean(fake)),
        "median_real": float(np.median(real)),
        "median_gen": float(np.median(fake)),
        "std_real": float(np.std(real)),
        "std_gen": float(np.std(fake)),
    }


def class_hist(records: list[dict]) -> dict[str, float]:
    labels = [r["class"] for r in records if r["valid"]]
    n = len(labels)
    out = {}
    for lab in NEHRP_LABELS:
        c = labels.count(lab)
        out[lab] = {"count": c, "frac": (c / n) if n else 0.0}
    out["invalid"] = {"count": sum(1 for r in records if not r["valid"]), "frac": 0.0}
    return out


def subsample(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(x) <= n:
        return x
    idx = rng.choice(len(x), size=n, replace=False)
    return x[idx]


def load_sequences_from_npy(path: Path) -> list[np.ndarray]:
    arr = np.load(path, allow_pickle=True)
    return [np.asarray(s, dtype=np.float64) for s in arr]


def load_model(ckpt: Path, device: torch.device) -> TransformerModel:
    cfg = cfg_mod.cfg
    model = TransformerModel(
        input_dim=cfg.input_dim,
        output_dim=cfg.output_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
        max_length=cfg.max_length,
        time_emb_dim=cfg.time_emb_dim,
        use_sequence_stats=cfg.use_sequence_stats,
        stats_dim=cfg.stats_dim,
        use_depth_conv=getattr(cfg, "use_depth_conv", True),
        depth_conv_kernel_size=getattr(cfg, "depth_conv_kernel_size", 3),
    ).to(device)
    checkpoint = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


@torch.no_grad()
def sample_conditional(
    model: TransformerModel,
    stats: list[dict[str, float]],
    lengths: list[int],
    max_length: int,
    device: torch.device,
    ode_steps: int = 50,
    batch_size: int = 32,
) -> list[np.ndarray]:
    """Generate with pinned (sequence_stats, length) — the model's actual condition."""
    n = len(stats)
    out: list[np.ndarray] = []
    for start in range(0, n, batch_size):
        sl = slice(start, min(start + batch_size, n))
        b_stats = stats[sl]
        b_len = lengths[sl]
        bsz = len(b_stats)
        u = torch.randn(bsz, max_length, 2, device=device)
        mask = torch.zeros(bsz, max_length, dtype=torch.bool, device=device)
        for i, L in enumerate(b_len):
            L = int(max(1, min(L, max_length)))
            mask[i, :L] = True
        stats_t = torch.tensor(
            [
                [s["ts_mean"], s["ts_std"], s["depth_mean"], s["depth_std"]]
                for s in b_stats
            ],
            dtype=torch.float32,
            device=device,
        )
        u = u * mask.unsqueeze(-1).float()
        dt = 1.0 / ode_steps
        for step in range(ode_steps):
            t = torch.full((bsz, 1), step * dt, device=device)
            v = model(u, mask, t, stats_t)
            u = (u + v * dt) * mask.unsqueeze(-1).float()
        u_np = u.cpu().numpy()
        for i, L in enumerate(b_len):
            L = int(max(1, min(L, max_length)))
            seq_n = u_np[i, :L]
            s = b_stats[i]
            seq = np.column_stack(
                [
                    seq_n[:, 0] * s["ts_std"] + s["ts_mean"],
                    seq_n[:, 1] * s["depth_std"] + s["depth_mean"],
                ]
            )
            out.append(seq)
        print(f"  generated {len(out)}/{n}", flush=True)
    return out


def sequence_stats_from_raw(seq: np.ndarray) -> dict[str, float]:
    seq = np.asarray(seq, dtype=np.float64)
    return {
        "ts_mean": float(np.mean(seq[:, 0])),
        "ts_std": float(np.std(seq[:, 0]) + 1e-10),
        "depth_mean": float(np.mean(seq[:, 1])),
        "depth_std": float(np.std(seq[:, 1]) + 1e-10),
    }


def linreg(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 5:
        return {"slope": np.nan, "intercept": np.nan, "r": np.nan, "n": 0}
    slope, intercept, r, p, se = scipy_stats.linregress(x, y)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r": float(r),
        "p": float(p),
        "se": float(se),
        "n": int(len(x)),
        "mean_x": float(np.mean(x)),
        "mean_y": float(np.mean(y)),
    }


def compare_sets(real_rec: list[dict], gen_rec: list[dict], rng: np.random.Generator) -> dict:
    real_f = feature_matrix(real_rec)
    gen_f = feature_matrix(gen_rec)
    n_match = min(len(real_f), len(gen_f))
    real_s, gen_s = standardize(real_f, gen_f)
    real_m = subsample(real_s, n_match, rng)
    gen_m = subsample(gen_s, n_match, rng)
    manifold = prdc(real_m, gen_m)
    manifold.update(nna_cov_mmd(real_m, gen_m))
    real_vs = np.array([r["vs30"] for r in real_rec if r["valid"]])
    gen_vs = np.array([r["vs30"] for r in gen_rec if r["valid"]])
    real_h = np.array([r["H"] for r in real_rec if r["valid"]])
    gen_h = np.array([r["H"] for r in gen_rec if r["valid"]])
    real_n = np.array([r["n_layers"] for r in real_rec if r["valid"]])
    gen_n = np.array([r["n_layers"] for r in gen_rec if r["valid"]])
    return {
        "n_real_valid": int(sum(r["valid"] for r in real_rec)),
        "n_gen_valid": int(sum(r["valid"] for r in gen_rec)),
        "n_gen_total": len(gen_rec),
        "valid_rate": float(np.mean([r["valid"] for r in gen_rec])) if gen_rec else 0.0,
        "manifold": manifold,
        "vs30": scalar_tests(real_vs, gen_vs),
        "H": scalar_tests(real_h, gen_h),
        "n_layers": scalar_tests(real_n, gen_n),
        "class_real": class_hist(real_rec),
        "class_gen": class_hist(gen_rec),
    }


def confusion(real_cls: list[str], gen_cls: list[str]) -> dict:
    labels = NEHRP_LABELS
    mat = {a: {b: 0 for b in labels + ["invalid"]} for a in labels}
    for a, b in zip(real_cls, gen_cls):
        if a not in mat:
            continue
        bb = b if b in mat[a] else "invalid"
        mat[a][bb] += 1
    rates = {}
    for a in labels:
        tot = sum(mat[a].values())
        rates[a] = {
            "n": tot,
            "self": (mat[a][a] / tot) if tot else 0.0,
            "row": {b: (mat[a][b] / tot) if tot else 0.0 for b in labels},
        }
    return {"counts": mat, "self_rate": rates}


def main() -> None:
    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = cfg_mod.cfg
    data_path = Path(cfg.data_path)
    print(f"Loading breakpoints from {data_path}")
    loader = FlowMatchingDataLoader(data_path=data_path)
    loader.load_data()
    assert loader.sequences is not None
    real_seqs = loader.sequences
    real_rec = [profile_features(s) for s in real_seqs]
    for i, r in enumerate(real_rec):
        r["idx"] = i
        r["stats"] = sequence_stats_from_raw(real_seqs[i])
        r["length"] = int(len(real_seqs[i]))

    split = json.loads((EXPORT / "results" / "split_indices.json").read_text())
    test_idx = split["test"]
    train_idx = split["train"]
    test_rec = [real_rec[i] for i in test_idx]
    train_rec = [real_rec[i] for i in train_idx]

    results: dict = {
        "protocol": {
            "papers": [
                "Naeem et al. 2020 — density & coverage (PRDC)",
                "Kynkäänniemi et al. 2019 — improved precision/recall",
                "Tong et al. TMLR 2024 — OT-CFM; MMD on scientific tasks",
                "Stoica et al. ICCV 2025 Contrastive FM — FID/IS/Prec/Rec",
                "TreeFlow 2026 — per-stratum 1-NNA / COV / MMD for conditional FM",
                "SubFlow 2026 — dominant-mode bias of the FM regression target",
            ],
            "condition": "[ts_mean, ts_std, depth_mean, depth_std] + sampled length",
            "site_classes": "NEHRP from travel-time Vs30",
            "seed": SEED,
        },
        "real": {
            "n_total": len(real_rec),
            "n_valid": int(sum(r["valid"] for r in real_rec)),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "class_all": class_hist(real_rec),
            "class_test": class_hist(test_rec),
            "vs30": {
                "mean": float(np.nanmean([r["vs30"] for r in real_rec])),
                "median": float(np.nanmedian([r["vs30"] for r in real_rec])),
                "p10": float(np.nanpercentile([r["vs30"] for r in real_rec], 10)),
                "p90": float(np.nanpercentile([r["vs30"] for r in real_rec], 90)),
            },
        },
        "population": {},
    }

    for tag, npy in [
        ("best", EXPORT / "samples" / "generated_samples_best.npy"),
        ("latest", EXPORT / "samples" / "generated_samples_latest.npy"),
    ]:
        print(f"Population metrics: {tag}")
        gen_seqs = load_sequences_from_npy(npy)
        gen_rec = [profile_features(s) for s in gen_seqs]
        pop = compare_sets(test_rec, gen_rec, rng)
        pop["vs_all_test"] = compare_sets(real_rec, gen_rec, rng)["vs30"]
        results["population"][tag] = pop

    # Conditional generation with latest.pt (better depth-stat RMSE)
    device = torch.device("cpu")
    ckpt = EXPORT / "checkpoints" / "latest.pt"
    print(f"Loading {ckpt}")
    model = load_model(ckpt, device)

    # Paired: one sample per held-out profile, pinned to that profile's stats+length
    print("Conditional paired generation on test set")
    test_stats = [real_rec[i]["stats"] for i in test_idx]
    test_lengths = [real_rec[i]["length"] for i in test_idx]
    paired_seqs = sample_conditional(
        model, test_stats, test_lengths, cfg.max_length, device, ode_steps=50, batch_size=32
    )
    paired_rec = [profile_features(s) for s in paired_seqs]
    np.save(OUT_DIR / "paired_test_latest.npy", np.array(paired_seqs, dtype=object), allow_pickle=True)

    real_vs = np.array([real_rec[i]["vs30"] for i in test_idx])
    gen_vs = np.array([r["vs30"] for r in paired_rec])
    real_h = np.array([real_rec[i]["H"] for i in test_idx])
    gen_h = np.array([r["H"] for r in paired_rec])
    real_cls = [real_rec[i]["class"] for i in test_idx]
    gen_cls = [r["class"] for r in paired_rec]

    results["conditional_paired"] = {
        "checkpoint": "latest.pt",
        "n": len(test_idx),
        "vs30_tracking": linreg(real_vs, gen_vs),
        "H_tracking": linreg(real_h, gen_h),
        "confusion": confusion(real_cls, gen_cls),
        "vs30_abs_err": {
            "mae": float(np.nanmean(np.abs(gen_vs - real_vs))),
            "medae": float(np.nanmedian(np.abs(gen_vs - real_vs))),
            "mape": float(np.nanmean(np.abs(gen_vs - real_vs) / np.clip(real_vs, 1.0, None))),
        },
        "bias_vs30": float(np.nanmean(gen_vs - real_vs)),
        "valid_rate": float(np.mean([r["valid"] for r in paired_rec])),
    }

    # Per NEHRP stratum: subsample train profiles as conditions, generate, compare to real class
    per_class = {}
    n_per = 48
    n_proto = 32
    for lab in NEHRP_LABELS:
        members = [r for r in train_rec if r["valid"] and r["class"] == lab]
        if len(members) < 8:
            print(f"Skip {lab}: only {len(members)} train profiles")
            continue
        pick = rng.choice(len(members), size=min(n_per, len(members)), replace=False)
        cond = [members[i] for i in pick]
        print(f"Class {lab}: generating {len(cond)} paired + {n_proto} prototype")
        seqs = sample_conditional(
            model,
            [c["stats"] for c in cond],
            [c["length"] for c in cond],
            cfg.max_length,
            device,
            ode_steps=50,
            batch_size=32,
        )
        gen_c = [profile_features(s) for s in seqs]
        real_c = [r for r in real_rec if r["valid"] and r["class"] == lab]
        cls_metrics = compare_sets(real_c, gen_c, rng)
        stay = np.mean([g["class"] == lab for g in gen_c])
        vs_track = linreg(
            np.array([c["vs30"] for c in cond]),
            np.array([g["vs30"] for g in gen_c]),
        )

        # Prototype: median stats, median length, many noises
        med_stats = {
            k: float(np.median([c["stats"][k] for c in members]))
            for k in ["ts_mean", "ts_std", "depth_mean", "depth_std"]
        }
        med_len = int(np.median([c["length"] for c in members]))
        proto_seqs = sample_conditional(
            model,
            [med_stats] * n_proto,
            [med_len] * n_proto,
            cfg.max_length,
            device,
            ode_steps=50,
            batch_size=32,
        )
        proto_rec = [profile_features(s) for s in proto_seqs]
        proto_vs = np.array([p["vs30"] for p in proto_rec if p["valid"]])
        real_vs_c = np.array([r["vs30"] for r in real_c])
        per_class[lab] = {
            "n_real": len(real_c),
            "n_cond": len(cond),
            "stay_rate": float(stay),
            "vs30_tracking": vs_track,
            "metrics": cls_metrics,
            "prototype": {
                "median_stats": med_stats,
                "median_length": med_len,
                "n": n_proto,
                "valid_rate": float(np.mean([p["valid"] for p in proto_rec])),
                "stay_rate": float(np.mean([p["class"] == lab for p in proto_rec])),
                "vs30_mean": float(np.nanmean(proto_vs)) if len(proto_vs) else np.nan,
                "vs30_std": float(np.nanstd(proto_vs)) if len(proto_vs) else np.nan,
                "vs30_cv": float(np.nanstd(proto_vs) / np.nanmean(proto_vs))
                if len(proto_vs) and np.nanmean(proto_vs) > 0
                else np.nan,
                "real_vs30_mean": float(np.mean(real_vs_c)),
                "real_vs30_std": float(np.std(real_vs_c)),
                "real_vs30_cv": float(np.std(real_vs_c) / np.mean(real_vs_c)),
            },
        }

    results["per_class"] = per_class

    # Quantile-conditioned generation: soft / typical / stiff from vs30 quantiles
    valid_train = [r for r in train_rec if r["valid"]]
    vs_sorted = sorted(valid_train, key=lambda r: r["vs30"])
    bands = {
        "soft_p10": vs_sorted[: max(1, len(vs_sorted) // 10)],
        "typical_p40_60": vs_sorted[int(0.4 * len(vs_sorted)) : int(0.6 * len(vs_sorted))],
        "stiff_p90": vs_sorted[int(0.9 * len(vs_sorted)) :],
    }
    quantile_out = {}
    for name, band in bands.items():
        n = min(40, len(band))
        pick = [band[i] for i in rng.choice(len(band), size=n, replace=False)]
        print(f"Quantile {name}: {n} samples")
        seqs = sample_conditional(
            model,
            [c["stats"] for c in pick],
            [c["length"] for c in pick],
            cfg.max_length,
            device,
            ode_steps=50,
            batch_size=32,
        )
        gen_b = [profile_features(s) for s in seqs]
        quantile_out[name] = {
            "n": n,
            "real_vs30_mean": float(np.mean([c["vs30"] for c in pick])),
            "gen_vs30_mean": float(np.nanmean([g["vs30"] for g in gen_b])),
            "real_H_mean": float(np.mean([c["H"] for c in pick])),
            "gen_H_mean": float(np.nanmean([g["H"] for g in gen_b])),
            "tracking": linreg(
                np.array([c["vs30"] for c in pick]),
                np.array([g["vs30"] for g in gen_b]),
            ),
            "class_gen": class_hist(gen_b),
        }
    results["quantile_bands"] = quantile_out

    out_path = OUT_DIR / "results.json"
    out_path.write_text(json.dumps(_to_json(results), indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parents[2])
    main()
