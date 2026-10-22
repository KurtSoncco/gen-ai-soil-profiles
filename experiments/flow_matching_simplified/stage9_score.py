"""Score the constrained-decoder flow under the Stage 8 metrics. CPU only.

Run: ``python -m experiments.flow_matching_simplified.stage9_score --tag s42_e500``
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from experiments.flow_matching_simplified.eval_corrected import bootstrap_vertical
from experiments.flow_matching_simplified.run_corrected_eval import (
    N_BOOT,
    N_SAMPLES,
    SEED,
)
from experiments.flow_matching_simplified.site_metrics import (
    fit_residual_reference,
    vs_depth_breakdown,
)
from experiments.flow_matching_simplified.stage8_rescore import (
    RESULT as RESCORE_RESULT,
    headline,
    load_reference,
    print_table,
    score,
)
from experiments.flow_matching_simplified.toro_baseline import fit_toro
from experiments.flow_matching_simplified.train_constrained import (
    OUT,
    ConstrainedTokens,
    build_model,
    load_split,
    sample,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="s42_e500")
    args = parser.parse_args()
    device = torch.device("cpu")
    real, train_raw, _ = load_reference()
    real_boot = bootstrap_vertical(real, n_boot=N_BOOT, seed=SEED)
    ref = fit_residual_reference(train_raw)
    train_seq, _, _ = load_split()
    data = ConstrainedTokens(train_seq)
    toro_fit = None
    results = {"tag": args.tag, "sets": {}}
    for name in ("latest", "best"):
        ckpt = torch.load(
            OUT / args.tag / f"{name}.pt", map_location=device, weights_only=False
        )
        assert np.allclose(ckpt["scale"], data.scale)
        model = build_model()
        model.load_state_dict(ckpt["model"])
        frac = float(
            ckpt.get(
                "toro_source_frac", 1.0 if ckpt.get("use_toro_source", True) else 0.0
            )
        )
        if frac > 0.0:
            if toro_fit is None:
                toro_fit = fit_toro(train_seq)
            source = toro_fit
        else:
            source = None
        for mode in ("matched", "independent"):
            gen = sample(
                model,
                data,
                N_SAMPLES,
                SEED,
                device,
                toro_fit=source,
                vs_z_mode=mode,
                toro_source_frac=frac,
            )
            key = f"constrained_{name}_{mode}"
            results["sets"][key] = score(
                key, gen, real, real_boot, ref, "generated", train=train_raw
            )
            results["sets"][key]["epoch_index"] = int(ckpt["epoch"])
            results["sets"][key]["vs_z_mode"] = mode
            results["sets"][key]["use_toro_source"] = bool(
                ckpt.get("use_toro_source", True)
            )
            results["sets"][key]["vs_breakdown"] = vs_depth_breakdown(gen)
            results["sets"][key]["final_depth_mean"] = float(
                np.mean([s[-1, 1] for s in gen])
            )
            print(f"{key} (epoch {ckpt['epoch']}) scored", flush=True)
    rows = {k: headline(v) for k, v in results["sets"].items()}
    previous = json.loads(RESCORE_RESULT.read_text())["headline"]
    # Older stage8 JSON may lack novelty columns; fill so the table still prints.
    for row in previous.values():
        row.setdefault("novelty_median", float("nan"))
        row.setdefault("novelty_frac_closer", float("nan"))
    rows = {**{k: previous[k] for k in ("flagship", "toro", "gaussian")}, **rows}
    results["headline"] = rows
    # Layered latest medians, if present, for the write-up.
    layered_path = RESCORE_RESULT.with_name("stage8_layered_s42_e500.json")
    if layered_path.exists():
        layered = json.loads(layered_path.read_text())["headline"]
        for k, v in layered.items():
            if k.startswith("layered_"):
                v.setdefault("novelty_median", float("nan"))
                v.setdefault("novelty_frac_closer", float("nan"))
                rows[k + "_s42"] = v
    out = RESCORE_RESULT.with_name(f"stage9_constrained_{args.tag}.json")
    out.write_text(json.dumps(results, indent=2, default=float) + "\n")
    print_table(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
