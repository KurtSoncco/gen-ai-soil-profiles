"""Score the fixed-grid MLP under trend and oracle starts."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.flow_matching_simplified.data import FlowMatchingDataLoader, FlowMatchingDataset
from experiments.flow_matching_simplified.eval_corrected import (
    TREND_DEPTHS_M,
    bound_rate,
    bootstrap_vertical,
)
from experiments.flow_matching_simplified.projection import (
    ALPHA,
    draw_trend_knots,
    fit_trend,
    knot_ln,
    tokens_from_knots,
)
from experiments.flow_matching_simplified.run_corrected_eval import (
    N_BOOT,
    N_SAMPLES,
    _pairwise,
    _score_set,
)
from experiments.flow_matching_simplified.train_fixed_grid import OUT, GridFlow

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "experiments" / "flow_matching_simplified" / "data" / "breakpoints_onewaytime.parquet"
SPLIT = ROOT / "outputs" / "flow_matching_simplified" / "results" / "split_indices_seed42.json"
RESULT = ROOT / "outputs" / "flow_matching_simplified" / "results" / "stage6_fixed_grid_eval.json"


def _boundaries(sequence: np.ndarray) -> np.ndarray:
    depths = np.asarray(sequence[:, 1], dtype=np.float64)
    if len(depths) == 0 or depths[0] != 0.0:
        depths = np.concatenate([[0.0], depths])
    return depths


def _field_to_tokens(bounds, physical, valid):
    return tokens_from_knots(bounds, physical, valid)


def _integrate(model, x0, mask, steps, device):
    u = x0.clone()
    dt = 1.0 / steps
    with torch.no_grad():
        for step in range(steps):
            t = torch.full((x0.shape[0], 1), step * dt, device=device)
            u = (u + model(u, mask, t) * dt) * mask
    return u


def _sample(model, scale, train_raw, fit, n, seed, device, oracle_raw=None):
    mu = torch.tensor(scale[:, 0], dtype=torch.float32, device=device)
    sig = torch.tensor(scale[:, 1], dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed + 17))
    fields = []
    masks = []
    bounds = []
    outside = 0
    total = 0
    source = oracle_raw if oracle_raw is not None else None
    for i in range(n):
        if source is None:
            index = int(rng.integers(0, len(train_raw)))
            template = train_raw[index]
            edge = _boundaries(template)
            ln, valid, n_out, n_kn = draw_trend_knots(fit, float(edge[-1]), rng, clip=True)
            outside += n_out
            total += n_kn
        else:
            template = source[i % len(source)]
            edge = _boundaries(template)
            ln, valid = knot_ln(template)
        fields.append(ln)
        masks.append(valid)
        bounds.append(edge)
    physical = torch.tensor(np.stack(fields), dtype=torch.float32, device=device)
    mask = torch.tensor(np.stack(masks), dtype=torch.float32, device=device)
    standard = torch.where(mask.bool(), (physical - mu) / sig, torch.zeros_like(physical))
    eps = torch.randn(standard.shape, generator=generator).to(device)
    x0 = ((1.0 - ALPHA) * standard + ALPHA * eps) * mask
    sequences = {}
    report = {
        "n_knots": total,
        "n_outside": outside,
        "fraction_outside": float(outside / total) if total else float("nan"),
    }
    for steps in (5, 10, 20, 50):
        end = _integrate(model, x0, mask, steps, device)
        recovered = (end * sig + mu).detach().cpu().numpy()
        seqs = []
        for i in range(n):
            valid = masks[i]
            seqs.append(_field_to_tokens(bounds[i], recovered[i], valid))
        sequences[str(steps)] = seqs
    return sequences, report


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = json.loads(SPLIT.read_text())
    loader = FlowMatchingDataLoader(DATA)
    loader.load_data()
    assert loader.sequences is not None
    train_raw = [loader.sequences[i] for i in split["train"]]
    test_raw = [loader.sequences[i] for i in split["test"]]
    fit = fit_trend(train_raw)
    test_ds = FlowMatchingDataset(test_raw, max_length=20, normalize=True)
    real = [test_ds.denormalize_sequence(test_ds.sequences[i], i) for i in range(len(test_raw))]
    real_boot = bootstrap_vertical(real, n_boot=N_BOOT, seed=42)
    n_knots = len(TREND_DEPTHS_M)
    results = {
        "split": str(SPLIT),
        "train_index_sum": int(sum(split["train"])),
        "real_bound_rate": bound_rate(real),
        "real_pairwise": _pairwise(real),
        "seeds": {},
    }
    for seed in (42, 43, 44):
        ckpt = torch.load(OUT / f"grid_s{seed}.pt", map_location=device, weights_only=False)
        model = GridFlow(n_knots).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        print(f"scoring fixed grid seed {seed}", flush=True)
        block = {"conditions": {}}
        trend_seqs, trend_report = _sample(
            model, ckpt["scale"], train_raw, fit, N_SAMPLES, seed, device
        )
        oracle_seqs, _ = _sample(
            model,
            ckpt["scale"],
            train_raw,
            fit,
            len(test_raw),
            seed + 9000,
            device,
            oracle_raw=test_raw,
        )
        block["trend_start_knots_outside_before_clip"] = trend_report
        for name, groups in (("trend", trend_seqs), ("oracle", oracle_seqs)):
            cond = {"sweeps": {}}
            for steps, seqs in groups.items():
                t0 = time.perf_counter()
                cond["sweeps"][steps] = _score_set(real, seqs, real_boot, None)
                cond["sweeps"][steps]["seconds"] = time.perf_counter() - t0
                row = cond["sweeps"][steps]
                print(
                    f"  grid {seed} {name} {steps} "
                    f"trend {row['depth_trend']['weighted_mean_ln_rmse']:.4f} bounds {row['bound_rate']:.3f}",
                    flush=True,
                )
            block["conditions"][name] = cond
        results["seeds"][str(seed)] = block
        RESULT.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {RESULT}", flush=True)


if __name__ == "__main__":
    main()
