"""Score the per-layer-token flow under the Stage 8 metrics. CPU only.

Run: ``python -m experiments.flow_matching_simplified.stage8_score_layered --tag s42_e500``
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
from experiments.flow_matching_simplified.site_metrics import fit_residual_reference
from experiments.flow_matching_simplified.stage8_rescore import (
    RESULT as RESCORE_RESULT,
    headline,
    load_reference,
    print_table,
    score,
)
from experiments.flow_matching_simplified.train_layered import (
    OUT,
    LayerTokens,
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
    data = LayerTokens(train_seq)
    results = {"tag": args.tag, "sets": {}}
    for name in ("latest", "best"):
        ckpt = torch.load(
            OUT / args.tag / f"{name}.pt", map_location=device, weights_only=False
        )
        assert np.allclose(ckpt["scale"], data.scale)
        model = build_model()
        model.load_state_dict(ckpt["model"])
        gen = sample(model, data, N_SAMPLES, SEED, device)
        key = f"layered_{name}"
        results["sets"][key] = score(
            key, gen, real, real_boot, ref, "generated", train=train_raw
        )
        results["sets"][key]["epoch_index"] = int(ckpt["epoch"])
        results["sets"][key]["final_depth_mean"] = float(
            np.mean([s[-1, 1] for s in gen])
        )
        print(f"{key} (epoch {ckpt['epoch']}) scored", flush=True)
    rows = {k: headline(v) for k, v in results["sets"].items()}
    previous = json.loads(RESCORE_RESULT.read_text())["headline"]
    rows = {**{k: previous[k] for k in ("flagship", "toro", "gaussian")}, **rows}
    results["headline"] = rows
    out = RESCORE_RESULT.with_name(f"stage8_layered_{args.tag}.json")
    out.write_text(json.dumps(results, indent=2, default=float) + "\n")
    print_table(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
