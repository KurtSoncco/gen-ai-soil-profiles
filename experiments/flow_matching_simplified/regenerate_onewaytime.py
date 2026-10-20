"""Rebuild one-way travel-time files. Does not overwrite the old parquets.

Source is ``data/vspdb_vs_profiles.parquet``. Travel time uses
``compute_tts`` (``Δt = h / Vs``). Breakpoints use
``extract_breakpoints``. Depth-stat targets use the recorded seed-42
train indices, not a fresh permutation.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.flow_matching_simplified.breakpoint_analysis import extract_breakpoints
from experiments.flow_matching_simplified.data import FlowMatchingDataLoader, FlowMatchingDataset
from experiments.flow_matching_simplified.depth_dependent_stats.depth_statistics import (
    compute_real_data_statistics,
)
from experiments.flow_matching_simplified.depth_dependent_stats.vertical_correlation import (
    compute_profile_statistics,
)
from soilgen_ai.vs_profiles.vs_calculation import compute_tts

ROOT = Path(__file__).resolve().parents[2]
VS_PATH = ROOT / "data" / "vspdb_vs_profiles.parquet"
OLD_TTS = ROOT / "data" / "vspdb_tts_profiles.parquet"
OLD_BP = ROOT / "experiments" / "flow_matching_simplified" / "data" / "breakpoints.parquet"
NEW_TTS = ROOT / "data" / "vspdb_tts_profiles_onewaytime.parquet"
NEW_BP = ROOT / "experiments" / "flow_matching_simplified" / "data" / "breakpoints_onewaytime.parquet"
SPLIT_PATH = ROOT / "outputs" / "flow_matching_simplified" / "results" / "split_indices_seed42.json"
STATS_PATH = (
    ROOT
    / "experiments"
    / "flow_matching_simplified"
    / "depth_dependent_stats"
    / "real_depth_stats_onewaytime.pkl"
)
CORR_PATH = (
    ROOT
    / "experiments"
    / "flow_matching_simplified"
    / "depth_dependent_stats"
    / "real_correlations_onewaytime.pkl"
)
VS_MIN = 100.0
VS_MAX = 5000.0
RTOL = 1e-5
ATOL = 1e-8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _layer_check(frame: pd.DataFrame, id_col: str) -> dict[str, float]:
    mismatches = 0
    layers = 0
    in_bounds = 0
    profiles = 0
    profiles_ok = 0
    for _, group in frame.groupby(id_col, sort=False):
        group = group.sort_values("depth")
        z = group["depth"].to_numpy(dtype=np.float64)
        t = group["tts"].to_numpy(dtype=np.float64)
        v = group["vs_value"].to_numpy(dtype=np.float64)
        dz = np.diff(z)
        dt = np.diff(t)
        vv = v[:-1]
        ok = (dz > 0) & (vv > 0) & np.isfinite(dt)
        if not np.any(ok):
            continue
        recovered = dz[ok] / dt[ok]
        bad = np.abs(recovered - vv[ok]) > (ATOL + RTOL * np.abs(vv[ok]))
        mismatches += int(bad.sum())
        layers += int(ok.sum())
        inside = (vv[ok] >= VS_MIN) & (vv[ok] <= VS_MAX)
        in_bounds += int(inside.sum())
        profiles += 1
        if bool(np.all(inside)):
            profiles_ok += 1
    return {
        "layers": layers,
        "mismatches": mismatches,
        "layer_bound_rate": in_bounds / layers,
        "profiles": profiles,
        "profile_bound_rate": profiles_ok / profiles,
    }


def _token_bounds(frame: pd.DataFrame) -> dict[str, float]:
    layers = 0
    in_bounds = 0
    profiles = 0
    profiles_ok = 0
    for _, group in frame.groupby("profile_id", sort=False):
        group = group.sort_values("depth")
        z = group["depth"].to_numpy(dtype=np.float64)
        t = group["tts"].to_numpy(dtype=np.float64)
        if z[0] != 0.0 or t[0] != 0.0:
            z = np.concatenate([[0.0], z])
            t = np.concatenate([[0.0], t])
        dz = np.diff(z)
        dt = np.diff(t)
        ok = (dz > 0) & (dt > 0) & np.isfinite(dt)
        if not np.any(ok):
            continue
        vs = dz[ok] / dt[ok]
        inside = (vs >= VS_MIN) & (vs <= VS_MAX) & np.isfinite(vs)
        layers += int(ok.sum())
        in_bounds += int(inside.sum())
        profiles += 1
        if bool(np.all(inside)):
            profiles_ok += 1
    return {
        "layers": layers,
        "layer_bound_rate": in_bounds / layers,
        "profiles": profiles,
        "profile_bound_rate": profiles_ok / profiles,
    }


def main() -> None:
    hashes_before = {str(p): _sha256(p) for p in (VS_PATH, OLD_TTS, OLD_BP)}
    vs = pd.read_parquet(VS_PATH)
    fine_parts = []
    bp_rows = []
    n_ids = vs["velocity_metadata_id"].nunique()
    for i, (pid, group) in enumerate(vs.groupby("velocity_metadata_id", sort=True)):
        built = compute_tts(group[["depth", "vs_value"]].copy())
        if built is None:
            raise RuntimeError(f"compute_tts returned None for {pid}")
        built.insert(0, "velocity_metadata_id", int(pid))
        fine_parts.append(built)
        breakpoints = extract_breakpoints(built)
        for row in breakpoints:
            bp_rows.append(
                {"profile_id": int(pid), "depth": float(row[0]), "tts": float(row[1])}
            )
        if (i + 1) % 400 == 0 or i + 1 == n_ids:
            print(f"profiles {i + 1}/{n_ids}", flush=True)

    fine = pd.concat(fine_parts, ignore_index=True)
    fine.to_parquet(NEW_TTS, index=False)
    bp = pd.DataFrame(bp_rows)
    bp.to_parquet(NEW_BP, index=False)

    fine_report = _layer_check(fine, "velocity_metadata_id")
    raw = vs.copy()
    raw_bounds_layers = int(((raw["vs_value"] >= VS_MIN) & (raw["vs_value"] <= VS_MAX)).sum())
    token_report = _token_bounds(bp)

    old_bp = pd.read_parquet(OLD_BP)
    depth_mismatch_profiles = 0
    tts_ratio_max_abs_err = 0.0
    compared = 0
    for pid, new_g in bp.groupby("profile_id", sort=True):
        old_g = old_bp[old_bp["profile_id"] == pid].sort_values("depth")
        new_g = new_g.sort_values("depth")
        if not np.array_equal(old_g["depth"].to_numpy(), new_g["depth"].to_numpy()):
            depth_mismatch_profiles += 1
            continue
        ratio = new_g["tts"].to_numpy() / np.maximum(old_g["tts"].to_numpy(), 1e-15)
        # Origin rows are 0/0. Compare positive travel times only.
        mask = old_g["tts"].to_numpy() > 0
        if np.any(mask):
            tts_ratio_max_abs_err = max(
                tts_ratio_max_abs_err, float(np.max(np.abs(ratio[mask] - 0.5)))
            )
        compared += 1

    old_loader = FlowMatchingDataLoader(OLD_BP)
    old_loader.load_data()
    new_loader = FlowMatchingDataLoader(NEW_BP)
    new_loader.load_data()
    split = json.loads(SPLIT_PATH.read_text())
    ids_equal = old_loader.profile_ids == new_loader.profile_ids
    split_ids = {
        name: [old_loader.profile_ids[i] for i in split[name]] for name in ("train", "val", "test")
    }
    split_ids_new = {
        name: [new_loader.profile_ids[i] for i in split[name]] for name in ("train", "val", "test")
    }
    split_unchanged = split_ids == split_ids_new

    train_sequences = [new_loader.sequences[i] for i in split["train"]]
    train_ds = FlowMatchingDataset(
        train_sequences, max_length=20, normalize=True
    )
    stats = compute_real_data_statistics(train_ds, bin_width=10.0, max_depth=500.0)
    corrs = compute_profile_statistics(train_ds, lags=[1.0, 2.0, 5.0, 10.0])
    with STATS_PATH.open("wb") as handle:
        pickle.dump(stats, handle)
    with CORR_PATH.open("wb") as handle:
        pickle.dump(corrs, handle)

    hashes_after = {str(p): _sha256(p) for p in (VS_PATH, OLD_TTS, OLD_BP)}
    summary = {
        "fine": fine_report,
        "raw_vs_rows": int(len(vs)),
        "raw_vs_layer_bound_count": raw_bounds_layers,
        "raw_vs_layer_bound_rate": raw_bounds_layers / len(vs),
        "oneway_rows": int(len(fine)),
        "breakpoint_rows": int(len(bp)),
        "breakpoint_profiles": int(bp["profile_id"].nunique()),
        "token_bounds": token_report,
        "breakpoint_depth_mismatch_profiles": depth_mismatch_profiles,
        "breakpoint_profiles_compared": compared,
        "breakpoint_tts_over_old_max_abs_err_from_one_half": tts_ratio_max_abs_err,
        "profile_id_order_equal": ids_equal,
        "n_profiles_loaded": len(new_loader.profile_ids),
        "split_profile_ids_unchanged": split_unchanged,
        "split_counts": {k: len(split[k]) for k in ("train", "val", "test")},
        "train_vs_samples": int(len(stats["all_vs"])),
        "mean_ln_vs_min": float(np.nanmin(stats["mean_ln_vs"])),
        "mean_ln_vs_max": float(np.nanmax(stats["mean_ln_vs"])),
        "corr_1m": float(corrs["mean_correlations"][1.0]),
        "corr_2m": float(corrs["mean_correlations"][2.0]),
        "old_hashes_unchanged": hashes_before == hashes_after,
        "stats_path": str(STATS_PATH),
        "corr_path": str(CORR_PATH),
    }
    out = ROOT / "outputs" / "flow_matching_simplified" / "results" / "onewaytime_regen.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
