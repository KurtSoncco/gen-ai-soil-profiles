from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


def compute_real_vs30_and_density(parquet_path: str) -> tuple[np.ndarray, float]:
    df = pd.read_parquet(parquet_path)
    required = {"velocity_metadata_id", "depth", "vs_value"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Parquet missing columns required for Vs30: {missing}")

    vs30_values = []
    samples_per_meter = []
    for _, g in df.groupby("velocity_metadata_id"):
        g = g.sort_values("depth")
        depths = g["depth"].to_numpy(dtype=float)
        vs = g["vs_value"].to_numpy(dtype=float)
        if depths.size == 0:
            continue
        # ensure starts at 0m if needed
        if depths[0] > 0:
            depths = np.insert(depths, 0, 0.0)
            vs = np.insert(vs, 0, vs[0])
        # limit to top 30 m
        top_mask = depths <= 30.0
        if not np.any(top_mask):
            continue
        d = depths[top_mask]
        v = vs[top_mask]
        if d[-1] < 30.0:
            d = np.append(d, 30.0)
            v = np.append(v, v[-1])
        dz = np.diff(d)
        v_mid = v[1:]  # piecewise-constant per segment
        denom = np.sum(dz / np.maximum(v_mid, 1e-6))
        if denom > 0:
            vs30_values.append(30.0 / denom)
        samples_per_meter.append((np.count_nonzero(top_mask)) / 30.0)

    if not vs30_values:
        return np.array([]), 1.0
    return np.asarray(vs30_values, dtype=float), float(np.mean(samples_per_meter))


def compute_generated_vs30(samples: np.ndarray, samples_per_meter: float) -> np.ndarray:
    # samples: (N, 1, L)
    if samples_per_meter <= 0:
        samples_per_meter = 1.0
    n_take = max(1, int(30.0 * samples_per_meter))
    n_take = min(n_take, samples.shape[-1])
    dz = 1.0 / samples_per_meter
    v = samples[:, 0, :n_take]
    v = np.maximum(v, 1e-6)
    denom = np.sum((dz / v), axis=1)
    vs30_gen = 30.0 / np.maximum(denom, 1e-9)
    return vs30_gen


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    ia = ib = 0
    na, nb = len(a_sorted), len(b_sorted)
    d = 0.0
    while ia < na and ib < nb:
        if a_sorted[ia] <= b_sorted[ib]:
            ia += 1
        else:
            ib += 1
        fa = ia / na
        fb = ib / nb
        d = max(d, abs(fa - fb))
    return d


def log_vs30_metrics(G: torch.nn.Module, cfg, device: torch.device, step: int, out_dir: str, real_vs30: np.ndarray, avg_samples_per_meter: float) -> None:
    if real_vs30.size == 0:
        print("[metrics] Skipping Vs30 metrics: real distribution unavailable.")
        return
    with torch.no_grad():
        z_eval = torch.randn(512, cfg.latent_dim, device=device)
        fake = G(z_eval).cpu().numpy()
    gen_vs30 = compute_generated_vs30(fake, avg_samples_per_meter)
    ks = ks_statistic(real_vs30, gen_vs30)
    print(f"[metrics] step={step} KS(real_vs30, gen_vs30)={ks:.4f}")
    # save histogram plot
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(6,4))
    plt.hist(real_vs30, bins=50, alpha=0.6, label="real", density=True)
    plt.hist(gen_vs30, bins=50, alpha=0.6, label="generated", density=True)
    plt.xlabel("Vs30 (m/s)")
    plt.ylabel("Density")
    plt.title(f"Vs30 distribution @ step {step}\nKS={ks:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"vs30_hist_{step}.png"))
    plt.close()


__all__ = [
    "compute_real_vs30_and_density",
    "compute_generated_vs30",
    "ks_statistic",
    "log_vs30_metrics",
]


