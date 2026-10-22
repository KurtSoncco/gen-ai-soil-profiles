"""CPU diagnostics for Stage 9: where Vs30 bias lives, H-renormalize, novelty.

Resamples the three layered seeds with the Stage 8 scorer seed, plus the
flagship, Gaussian, and Toro draws. No training.

Run: ``python -m experiments.flow_matching_simplified.stage9_diagnose``
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from experiments.flow_matching_simplified.baseline_metrics import (
    _denorm_generated,
    _initial_state,
    euler_integrate,
    sample_fixed_grid_gaussian,
)
from experiments.flow_matching_simplified.comprehensive_eval import load_checkpoint
from experiments.flow_matching_simplified.eval_corrected import (
    bootstrap_vertical,
    layer_model,
    travel_time_from_vs,
)
from experiments.flow_matching_simplified.run_corrected_eval import (
    N_BOOT,
    N_SAMPLES,
    SEED,
)
from experiments.flow_matching_simplified.site_metrics import (
    fit_residual_reference,
    nearest_train_distance,
    vs_depth_breakdown,
)
from experiments.flow_matching_simplified.stage8_rescore import (
    FLAGSHIP,
    RESULT as RESCORE_RESULT,
    headline,
    load_reference,
    print_table,
    score,
)
from experiments.flow_matching_simplified.toro_baseline import fit_toro, sample_toro
from experiments.flow_matching_simplified.train_layered import (
    OUT as LAYERED_OUT,
    LayerTokens,
    build_model,
    load_split,
    sample,
)
from experiments.flow_matching_simplified import config as cfg_mod

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "outputs/flow_matching_simplified/results/stage9_diagnose.json"
TAGS = ("s42_e500", "s43_e500", "s44_e500")


def intended_depths(data: LayerTokens, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(data.lengths), n)
    return data.depths[idx]


def renormalize_to_h(seq: np.ndarray, h_total: float) -> np.ndarray:
    lm = layer_model(seq, include_invalid=False)
    if lm is None:
        return seq
    bounds, vs = lm
    h = np.diff(bounds)
    h = h * (h_total / h.sum())
    return travel_time_from_vs(np.concatenate([[0.0], np.cumsum(h)]), vs)


def _sample_layered(tag: str, data: LayerTokens, device: torch.device):
    ckpt = torch.load(
        LAYERED_OUT / tag / "latest.pt", map_location=device, weights_only=False
    )
    model = build_model()
    model.load_state_dict(ckpt["model"])
    return sample(model, data, N_SAMPLES, SEED, device)


def main() -> None:
    device = torch.device("cpu")
    real, train_raw, train_ds = load_reference()
    real_boot = bootstrap_vertical(real, n_boot=N_BOOT, seed=SEED)
    ref = fit_residual_reference(train_raw)
    train_seq, _, _ = load_split()
    data = LayerTokens(train_seq)
    results: dict = {"sets": {}, "vs_breakdown": {}, "renormalized": {}}

    for tag in TAGS:
        gen = _sample_layered(tag, data, device)
        name = f"layered_{tag}"
        results["sets"][name] = score(
            name, gen, real, real_boot, ref, "generated", train=train_raw
        )
        results["vs_breakdown"][name] = {
            "real": vs_depth_breakdown(real),
            "gen": vs_depth_breakdown(gen),
        }
        h_cond = intended_depths(data, N_SAMPLES, SEED)
        renorm = [renormalize_to_h(seq, float(h)) for seq, h in zip(gen, h_cond)]
        results["renormalized"][name] = score(
            name + "_renorm", renorm, real, real_boot, ref, "generated", train=train_raw
        )
        print(f"{name} scored", flush=True)

    cfg = cfg_mod.cfg
    model, _ = load_checkpoint(str(FLAGSHIP), device, cfg)
    model.eval()
    u0, masks, stats, stats_list, lengths = _initial_state(
        train_ds, N_SAMPLES, cfg.max_length, device, SEED
    )
    flow = _denorm_generated(
        euler_integrate(model, u0, masks, stats, 50), lengths, stats_list
    )
    results["sets"]["flagship"] = score(
        "flagship", flow, real, real_boot, ref, "generated", lengths, train=train_raw
    )
    print("flagship scored", flush=True)

    gauss = sample_fixed_grid_gaussian(train_raw, N_SAMPLES, SEED)
    results["sets"]["gaussian"] = score(
        "gaussian", gauss, real, real_boot, ref, "copied", train=train_raw
    )
    print("gaussian scored", flush=True)

    toro = sample_toro(fit_toro(train_seq), N_SAMPLES, SEED)
    results["sets"]["toro"] = score(
        "toro", toro, real, real_boot, ref, "generated", train=train_raw
    )
    print("toro scored", flush=True)

    test_novelty = nearest_train_distance(real, train_raw, test=real)
    results["test_novelty"] = test_novelty
    results["headline"] = {k: headline(v) for k, v in results["sets"].items()}
    results["headline_renorm"] = {
        k: headline(v) for k, v in results["renormalized"].items()
    }
    RESULT.write_text(json.dumps(results, indent=2, default=float) + "\n")
    print("raw")
    print_table(results["headline"])
    print("H-renormalized layered")
    print_table(results["headline_renorm"])
    print(f"test-to-train novelty median {test_novelty['test_median']}")
    print(f"wrote {RESULT}")
    # Keep a pointer so later scoring can merge if needed.
    if RESCORE_RESULT.exists():
        print(f"stage8 rescore still at {RESCORE_RESULT}")


if __name__ == "__main__":
    main()
