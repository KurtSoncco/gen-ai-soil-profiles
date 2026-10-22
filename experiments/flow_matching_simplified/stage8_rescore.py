"""Re-score existing generators under the Stage 8 metrics. CPU, no training.

Sets: the Stage 3 flagship (``latest_onewaytime.pt``, Euler 50, the Stage 3
seed-42 draw), the fixed-grid Gaussian, the three fixed-grid MLP seeds (trend
start, Euler 50), and the Toro (1995) baseline fitted on the train split.
Every set is 200 profiles against the 238 seed-42 test profiles.

Run: ``python -m experiments.flow_matching_simplified.stage8_rescore``
"""

from __future__ import annotations

import json
import time
from pathlib import Path

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
    bootstrap_vertical,
    validity_rate,
)
from experiments.flow_matching_simplified.run_corrected_eval import (
    N_BOOT,
    N_SAMPLES,
    SEED,
    _denorm_split,
    _score_set,
)
from experiments.flow_matching_simplified.site_metrics import (
    ResidualReference,
    fit_residual_reference,
    nearest_train_distance,
    site_report,
)
from experiments.flow_matching_simplified.toro_baseline import fit_toro, sample_toro

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "experiments/flow_matching_simplified/data/breakpoints_onewaytime.parquet"
SPLIT = ROOT / "outputs/flow_matching_simplified/results/split_indices_seed42.json"
FLAGSHIP = ROOT / "outputs/flow_matching_simplified/checkpoints/latest_onewaytime.pt"
GRID_DIR = ROOT / "outputs/flow_matching_simplified_fixed_grid"
RESULT = ROOT / "outputs/flow_matching_simplified/results/stage8_rescore.json"

LAYER_SOURCE = {
    "flagship": "generated",
    "gaussian": "copied from a training profile",
    "fixed_grid_mlp": "copied from a training profile",
    "toro": "generated",
}


def load_reference():
    loader = FlowMatchingDataLoader(DATA)
    loader.load_data()
    split = json.loads(SPLIT.read_text())
    real, _ = _denorm_split(loader, split["test"])
    train_raw, train_ds = _denorm_split(loader, split["train"])
    return real, train_raw, train_ds


def score(
    name: str,
    generated: list[np.ndarray],
    real: list[np.ndarray],
    real_boot: dict,
    ref: ResidualReference,
    layer_source: str,
    lengths=None,
    train: list[np.ndarray] | None = None,
) -> dict:
    t0 = time.perf_counter()
    out = _score_set(real, generated, real_boot, lengths)
    out["site"] = site_report(real, generated, ref)
    out["site_all_profiles"] = site_report(real, generated, ref, include_invalid=True)
    if train is not None:
        out["novelty"] = nearest_train_distance(generated, train, test=real)
    out["layer_source"] = layer_source
    out["seconds"] = time.perf_counter() - t0
    return out


def headline(row: dict) -> dict[str, float]:
    """Validity counts invalid profiles as failures; the other columns use valid profiles."""
    v = row["vertical"]["1.0"]
    site = row["site"]
    return {
        "validity": row["validity_rate"],
        "bounds": row["bound_rate"],
        "n_scored": site["gen_profiles_scored"],
        "trend_rmse": row["depth_trend"]["weighted_mean_ln_rmse"],
        "corr1m_gap": v["abs_correlation_gap"],
        "resid_rho": site["residual_correlation"]["gen"]["rho"],
        "resid_rho_gap": site["residual_correlation"]["abs_rho_gap"],
        "resid_sd": site["residual_correlation"]["gen"]["residual_sd"],
        "sigma_rmse": site["sigma_ln_vs"]["weighted_rmse"],
        "thick_ks": site["thickness"]["weighted_ks"],
        "vs30_ks": site["vs30"]["ks"],
        "f0_ks": site["f0"]["ks_ln"],
        "resid_rho_clipped": row["site_all_profiles"]["residual_correlation"]["gen"][
            "rho"
        ],
        "novelty_median": row.get("novelty", {}).get("median", float("nan")),
        "novelty_frac_closer": row.get("novelty", {}).get(
            "frac_closer_than_test_median", float("nan")
        ),
    }


def print_table(rows: dict[str, dict]) -> None:
    keys = list(next(iter(rows.values())).keys())
    print("set".ljust(20) + "".join(k[:12].rjust(13) for k in keys))
    for name, h in rows.items():
        print(name.ljust(20) + "".join(f"{h[k]:13.3f}" for k in keys))


def main() -> None:
    device = torch.device("cpu")
    cfg = cfg_mod.cfg
    real, train_raw, train_ds = load_reference()
    real_boot = bootstrap_vertical(real, n_boot=N_BOOT, seed=SEED)
    ref = fit_residual_reference(train_raw)
    results: dict = {
        "n_test": len(real),
        "n_samples": N_SAMPLES,
        "real": {
            "validity_rate": validity_rate(real),
            "site_self": site_report(real, real, ref)["residual_correlation"]["real"],
        },
        "sets": {},
    }

    model, epoch = load_checkpoint(str(FLAGSHIP), device, cfg)
    model.eval()
    u0, masks, stats, stats_list, lengths = _initial_state(
        train_ds, N_SAMPLES, cfg.max_length, device, SEED
    )
    flow = _denorm_generated(
        euler_integrate(model, u0, masks, stats, 50), lengths, stats_list
    )
    results["sets"]["flagship"] = score(
        "flagship",
        flow,
        real,
        real_boot,
        ref,
        LAYER_SOURCE["flagship"],
        lengths,
        train=train_raw,
    )
    results["sets"]["flagship"]["epoch_index"] = int(epoch)
    print("flagship scored", flush=True)

    gauss = sample_fixed_grid_gaussian(train_raw, N_SAMPLES, SEED)
    results["sets"]["gaussian"] = score(
        "gaussian", gauss, real, real_boot, ref, LAYER_SOURCE["gaussian"], train=train_raw
    )
    print("gaussian scored", flush=True)

    from experiments.flow_matching_simplified.eval_fixed_grid import _sample
    from experiments.flow_matching_simplified.projection import fit_trend
    from experiments.flow_matching_simplified.train_fixed_grid import GridFlow

    loader = FlowMatchingDataLoader(DATA)
    loader.load_data()
    split = json.loads(SPLIT.read_text())
    train_seq = [loader.sequences[i] for i in split["train"]]
    trend_fit = fit_trend(train_seq)
    for seed in (42, 43, 44):
        ckpt = torch.load(
            GRID_DIR / f"grid_s{seed}.pt", map_location=device, weights_only=False
        )
        grid = GridFlow(20).to(device)
        grid.load_state_dict(ckpt["model"])
        grid.eval()
        seqs, _ = _sample(
            grid, ckpt["scale"], train_seq, trend_fit, N_SAMPLES, seed, device
        )
        name = f"fixed_grid_mlp_s{seed}"
        results["sets"][name] = score(
            name,
            seqs["50"],
            real,
            real_boot,
            ref,
            LAYER_SOURCE["fixed_grid_mlp"],
            train=train_raw,
        )
        print(f"{name} scored", flush=True)

    toro_fit = fit_toro(train_seq)
    results["toro_fit"] = toro_fit.summary()
    toro = sample_toro(toro_fit, N_SAMPLES, SEED)
    results["sets"]["toro"] = score(
        "toro", toro, real, real_boot, ref, LAYER_SOURCE["toro"], train=train_raw
    )
    print("toro scored", flush=True)

    results["headline"] = {k: headline(v) for k, v in results["sets"].items()}
    RESULT.write_text(json.dumps(results, indent=2, default=float) + "\n")
    print_table(results["headline"])
    print(f"real residual rho {results['real']['site_self']}")
    print(f"wrote {RESULT}")


if __name__ == "__main__":
    main()
