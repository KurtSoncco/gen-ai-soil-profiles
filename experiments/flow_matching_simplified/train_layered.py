"""Transformer flow matching on per-layer tokens.

Each token is one layer, ``[ln h, s]``, where ``h`` is the thickness and
``s = logit((ln Vs − ln 100) / (ln 5000 − ln 100))``. Thickness is positive and
Vs lies in [100, 5000] m/s for every sample, by construction. Catalog layers
outside the bounds are clipped to just inside them, and the clipped fraction
is recorded.

Both channels are standardized with one mean and s.d. over all training
layers. The condition is ``[ln H, n / 20]``, the total depth and layer count of
one training profile, drawn together at sampling time. The optional thin-layer
penalty is applied to the predicted endpoint ``x̂1 = ut + (1 − t)·v_θ``, so it
has a gradient with respect to the network.

Run: ``python -m experiments.flow_matching_simplified.train_layered --seed 42``
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from experiments.flow_matching_simplified.data import FlowMatchingDataLoader
from experiments.flow_matching_simplified.eval_corrected import (
    VS_MAX,
    VS_MIN,
    layer_model,
    travel_time_from_vs,
)
from experiments.flow_matching_simplified.model import TransformerModel
from experiments.flow_matching_simplified.train import set_seed

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "experiments/flow_matching_simplified/data/breakpoints_onewaytime.parquet"
SPLIT = ROOT / "outputs/flow_matching_simplified/results/split_indices_seed42.json"
OUT = ROOT / "outputs/flow_matching_simplified_layered"
MAX_LAYERS = 20
LN_LO = float(np.log(VS_MIN))
LN_HI = float(np.log(VS_MAX))
SQUASH_EPS = 1e-3
BATCH = 32
LR = 1e-4
THIN_PENALTY_WEIGHT = 1e-2


def squash(ln_vs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Logit of the position of ln Vs inside the bounds. Returns values and a clipped flag."""
    u = (ln_vs - LN_LO) / (LN_HI - LN_LO)
    clipped = (u < SQUASH_EPS) | (u > 1.0 - SQUASH_EPS)
    u = np.clip(u, SQUASH_EPS, 1.0 - SQUASH_EPS)
    return np.log(u / (1.0 - u)), clipped


def unsquash(s: np.ndarray) -> np.ndarray:
    return LN_LO + (LN_HI - LN_LO) / (1.0 + np.exp(-s))


class LayerTokens:
    """Per-layer tokens for a list of breakpoint sequences, with shared scaling."""

    def __init__(self, sequences: list[np.ndarray], scale: np.ndarray | None = None):
        tokens, depths, n_clip, n_layers = [], [], 0, 0
        for seq in sequences:
            lm = layer_model(seq, include_invalid=False)
            if lm is None:
                continue
            bounds, vs = lm
            bounds, vs = bounds[: MAX_LAYERS + 1], vs[:MAX_LAYERS]
            s, clipped = squash(np.log(vs))
            n_clip += int(clipped.sum())
            n_layers += len(vs)
            tokens.append(np.column_stack([np.log(np.diff(bounds)), s]))
            depths.append(float(bounds[-1]))
        self.raw = tokens
        self.depths = np.asarray(depths)
        self.lengths = np.array([len(t) for t in tokens])
        self.clipped_fraction = n_clip / max(n_layers, 1)
        if scale is None:
            flat = np.concatenate(tokens)
            scale = np.stack([flat.mean(axis=0), flat.std(axis=0)])
        self.scale = scale
        n = len(tokens)
        self.x = np.zeros((n, MAX_LAYERS, 2), dtype=np.float32)
        self.mask = np.zeros((n, MAX_LAYERS), dtype=bool)
        for i, t in enumerate(tokens):
            self.x[i, : len(t)] = (t - scale[0]) / scale[1]
            self.mask[i, : len(t)] = True
        self.cond = condition(self.depths, self.lengths)

    def ln_h_floor(self, quantile: float = 1.0) -> float:
        return float(
            np.percentile(np.concatenate([t[:, 0] for t in self.raw]), quantile)
        )


def condition(depths: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    return np.column_stack([np.log(depths), lengths / MAX_LAYERS]).astype(np.float32)


def build_model() -> TransformerModel:
    return TransformerModel(
        input_dim=2,
        output_dim=2,
        hidden_dim=256,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        max_length=MAX_LAYERS,
        time_emb_dim=128,
        use_sequence_stats=True,
        stats_dim=2,
        use_depth_conv=True,
    )


def flow_loss(
    model: TransformerModel,
    x1: torch.Tensor,
    mask: torch.Tensor,
    cond: torch.Tensor,
    scale: torch.Tensor,
    ln_h_floor: float,
    penalty_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Conditional flow-matching loss plus the thin-layer penalty on ``x̂1``."""
    m = mask.unsqueeze(-1).float()
    x0 = torch.randn_like(x1) * m
    t = torch.rand(x1.shape[0], 1, device=x1.device)
    tb = t.unsqueeze(-1)
    xt = ((1.0 - tb) * x0 + tb * x1) * m
    v = model(xt, mask, t, cond)
    cfm = (((v - (x1 - x0)) ** 2) * m).sum() / (m.sum() * 2.0)
    x1_hat = xt + (1.0 - tb) * v
    ln_h = x1_hat[..., 0] * scale[1, 0] + scale[0, 0]
    short = torch.relu(ln_h_floor - ln_h) ** 2
    penalty = (short * mask.float()).sum() / mask.float().sum()
    return cfm + penalty_weight * penalty, penalty


@torch.no_grad()
def sample(
    model: TransformerModel,
    data: LayerTokens,
    n: int,
    seed: int,
    device: torch.device,
    steps: int = 50,
) -> list[np.ndarray]:
    """Euler integration. Length and total depth come from one training profile."""
    model.eval()
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    idx = rng.integers(0, len(data.lengths), n)
    lengths = data.lengths[idx]
    cond = torch.tensor(condition(data.depths[idx], lengths), device=device)
    mask = torch.zeros(n, MAX_LAYERS, dtype=torch.bool)
    for i, k in enumerate(lengths):
        mask[i, :k] = True
    mask = mask.to(device)
    m = mask.unsqueeze(-1).float()
    x = torch.randn(n, MAX_LAYERS, 2, generator=gen).to(device) * m
    dt = 1.0 / steps
    for k in range(steps):
        t = torch.full((n, 1), k * dt, device=device)
        x = (x + model(x, mask, t, cond) * dt) * m
    x = x.cpu().numpy()
    out = []
    for i, k in enumerate(lengths):
        tok = x[i, :k] * data.scale[1] + data.scale[0]
        bounds = np.concatenate([[0.0], np.cumsum(np.exp(tok[:, 0]))])
        vs = np.exp(unsquash(tok[:, 1]))
        out.append(travel_time_from_vs(bounds, vs))
    return out


def load_split():
    loader = FlowMatchingDataLoader(DATA)
    loader.load_data()
    split = json.loads(SPLIT.read_text())
    pick = lambda key: [loader.sequences[i] for i in split[key]]  # noqa: E731
    return pick("train"), pick("val"), pick("test")


def train(seed: int, epochs: int, device: torch.device, tag: str) -> dict:
    set_seed(seed)
    train_seq, val_seq, _ = load_split()
    tr = LayerTokens(train_seq)
    va = LayerTokens(val_seq, scale=tr.scale)
    floor = tr.ln_h_floor()
    scale_t = torch.tensor(tr.scale, dtype=torch.float32, device=device)
    model = build_model().to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=(0.6, 0.999), weight_decay=1e-4
    )
    sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10, min_lr=1e-6)
    tensors = {
        name: tuple(torch.tensor(a, device=device) for a in (d.x, d.mask, d.cond))
        for name, d in (("train", tr), ("val", va))
    }
    order_rng = np.random.default_rng(42)
    out_dir = OUT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    history = []
    t0 = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        x1, mk, cd = tensors["train"]
        order = torch.tensor(order_rng.permutation(len(x1)), device=device)
        total, pen_total, seen = 0.0, 0.0, 0
        for start in range(0, len(order), BATCH):
            rows = order[start : start + BATCH]
            loss, pen = flow_loss(
                model, x1[rows], mk[rows], cd[rows], scale_t, floor, THIN_PENALTY_WEIGHT
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach()) * len(rows)
            pen_total += float(pen.detach()) * len(rows)
            seen += len(rows)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                torch.manual_seed(1234)
                vx, vm, vc = tensors["val"]
                val, _ = flow_loss(model, vx, vm, vc, scale_t, floor, 0.0)
            val = float(val)
            sched.step(val)
            history.append(
                {
                    "epoch": epoch,
                    "train": total / seen,
                    "penalty": pen_total / seen,
                    "val": val,
                }
            )
            state = {"model": model.state_dict(), "epoch": epoch, "scale": tr.scale}
            if val < best:
                best = val
                torch.save(state, out_dir / "best.pt")
            torch.save(state, out_dir / "latest.pt")
            if (epoch + 1) % 50 == 0 or epoch == 0:
                print(
                    f"[{tag}] epoch {epoch + 1} train {total / seen:.4f} "
                    f"penalty {pen_total / seen:.4f} val {val:.4f} "
                    f"lr {opt.param_groups[0]['lr']:.1e} {time.perf_counter() - t0:.0f}s",
                    flush=True,
                )
    summary = {
        "seed": seed,
        "epochs": epochs,
        "seconds": time.perf_counter() - t0,
        "best_val": best,
        "n_params": sum(p.numel() for p in model.parameters()),
        "train_clipped_layer_fraction": tr.clipped_fraction,
        "ln_h_floor": floor,
        "history": history,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = args.tag or f"s{args.seed}"
    summary = train(args.seed, args.epochs, device, tag)
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
