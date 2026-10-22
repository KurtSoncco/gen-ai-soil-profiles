"""Transformer flow matching with a constrained layer decoder.

Tokens are ``[log p, s]`` where ``p = h / H`` and ``s`` is the logit-squash of
ln Vs used by ``train_layered``. Decode with ``h = softmax(u) * H``, so every
sample has positive thicknesses that sum to the conditioned depth. The
condition is ``[ln H, n / 20, ln Vs_z]``, with Vs_z the time-averaged Vs to
30 m or to H when the profile is shallower.

The interpolant starts from a Toro (1995) profile with the same ``n`` and
``H``, remapped by merging the thinnest or splitting the thickest layers.
Pass ``--no-toro-source`` to fall back to Gaussian noise.

Run: ``python -m experiments.flow_matching_simplified.train_constrained --seed 42``
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
    VS_MIN,
    layer_model,
    travel_time_from_vs,
)
from experiments.flow_matching_simplified.model import TransformerModel
from experiments.flow_matching_simplified.site_metrics import vs_z_or_h
from experiments.flow_matching_simplified.toro_baseline import (
    ToroFit,
    fit_toro,
    sample_toro_matched,
)
from experiments.flow_matching_simplified.train import set_seed
from experiments.flow_matching_simplified.train_layered import squash, unsquash

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "experiments/flow_matching_simplified/data/breakpoints_onewaytime.parquet"
SPLIT = ROOT / "outputs/flow_matching_simplified/results/split_indices_seed42.json"
OUT = ROOT / "outputs/flow_matching_simplified_constrained"
MAX_LAYERS = 20
BATCH = 32
LR = 1e-4
THIN_PENALTY_WEIGHT = 1e-2


def softmax_np(u: np.ndarray) -> np.ndarray:
    e = np.exp(u - np.max(u))
    return e / e.sum()


def decode_tokens(tok: np.ndarray, h_total: float) -> np.ndarray:
    """Map unconstrained ``[u, s]`` to a ``[tts, depth]`` sequence with sum h = H."""
    h = softmax_np(tok[:, 0]) * h_total
    bounds = np.concatenate([[0.0], np.cumsum(h)])
    vs = np.exp(unsquash(tok[:, 1]))
    return travel_time_from_vs(bounds, vs)


def encode_layers(bounds: np.ndarray, vs: np.ndarray) -> np.ndarray:
    h = np.diff(bounds)
    u = np.log(h / h.sum())
    s, _ = squash(np.log(vs))
    return np.column_stack([u, s])


class ConstrainedTokens:
    """Per-layer simplex tokens with a three-dimensional site condition."""

    def __init__(self, sequences: list[np.ndarray], scale: np.ndarray | None = None):
        tokens, depths, vs_z_vals, n_clip, n_layers = [], [], [], 0, 0
        for seq in sequences:
            lm = layer_model(seq, include_invalid=False)
            if lm is None:
                continue
            bounds, vs = lm
            bounds, vs = bounds[: MAX_LAYERS + 1], vs[:MAX_LAYERS]
            raw = encode_layers(bounds, vs)
            _, clipped = squash(np.log(vs))
            n_clip += int(clipped.sum())
            n_layers += len(vs)
            tokens.append(raw)
            depths.append(float(bounds[-1]))
            vs_z_vals.append(max(vs_z_or_h(bounds, vs), VS_MIN))
        self.raw = tokens
        self.depths = np.asarray(depths)
        self.vs_z = np.asarray(vs_z_vals)
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
        self.cond = condition(self.depths, self.lengths, self.vs_z)

    def h_floor(self, quantile: float = 1.0) -> float:
        return float(np.percentile(np.concatenate([np.exp(t[:, 0]) * d for t, d in zip(self.raw, self.depths)]), quantile))


def condition(depths: np.ndarray, lengths: np.ndarray, vs_z: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [np.log(depths), np.asarray(lengths) / MAX_LAYERS, np.log(vs_z)]
    ).astype(np.float32)


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
        stats_dim=3,
        use_depth_conv=True,
    )


def masked_softmax(u: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    filled = u.masked_fill(~mask, torch.finfo(u.dtype).min)
    return torch.softmax(filled, dim=-1)


def encode_toro_batch(
    fit: ToroFit,
    depths: np.ndarray,
    lengths: np.ndarray,
    scale: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Toro tokens with the same ``n`` and ``H`` as each row, already scaled."""
    out = np.zeros((len(depths), MAX_LAYERS, 2), dtype=np.float32)
    for i, (h_total, n) in enumerate(zip(depths, lengths)):
        bounds, vs = sample_toro_matched(fit, float(h_total), int(n), rng)
        tok = (encode_layers(bounds, vs) - scale[0]) / scale[1]
        out[i, : int(n)] = tok
    return out


def flow_loss(
    model: TransformerModel,
    x1: torch.Tensor,
    mask: torch.Tensor,
    cond: torch.Tensor,
    scale: torch.Tensor,
    h_floor: float,
    penalty_weight: float,
    x0: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CFM plus a thin-layer penalty on the predicted clean sample."""
    m = mask.unsqueeze(-1).float()
    if x0 is None:
        x0 = torch.randn_like(x1) * m
    else:
        x0 = x0 * m
    t = torch.rand(x1.shape[0], 1, device=x1.device)
    tb = t.unsqueeze(-1)
    xt = ((1.0 - tb) * x0 + tb * x1) * m
    v = model(xt, mask, t, cond)
    cfm = (((v - (x1 - x0)) ** 2) * m).sum() / (m.sum() * 2.0)
    x1_hat = xt + (1.0 - tb) * v
    u = x1_hat[..., 0] * scale[1, 0] + scale[0, 0]
    h = masked_softmax(u, mask) * torch.exp(cond[:, 0]).unsqueeze(-1)
    short = torch.relu(h_floor - h) ** 2
    penalty = (short * mask.float()).sum() / mask.float().sum()
    return cfm + penalty_weight * penalty, penalty


@torch.no_grad()
def sample(
    model: TransformerModel,
    data: ConstrainedTokens,
    n: int,
    seed: int,
    device: torch.device,
    steps: int = 50,
    toro_fit: ToroFit | None = None,
    vs_z_mode: str = "matched",
    toro_source_frac: float = 1.0,
) -> list[np.ndarray]:
    """Euler integration. ``(n, H, Vs_z)`` come from one training profile."""
    model.eval()
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    idx = rng.integers(0, len(data.lengths), n)
    lengths = data.lengths[idx]
    depths = data.depths[idx]
    if vs_z_mode == "independent":
        vs_z = data.vs_z[rng.integers(0, len(data.vs_z), n)]
    elif vs_z_mode == "matched":
        vs_z = data.vs_z[idx]
    else:
        raise ValueError(f"unknown vs_z_mode {vs_z_mode}")
    cond = torch.tensor(condition(depths, lengths, vs_z), device=device)
    mask = torch.zeros(n, MAX_LAYERS, dtype=torch.bool)
    for i, k in enumerate(lengths):
        mask[i, :k] = True
    mask = mask.to(device)
    m = mask.unsqueeze(-1).float()
    x = torch.randn(n, MAX_LAYERS, 2, generator=gen).to(device) * m
    if toro_fit is not None:
        toro_x = torch.tensor(
            encode_toro_batch(toro_fit, depths, lengths, data.scale, rng),
            device=device,
        ) * m
        if toro_source_frac >= 1.0:
            x = toro_x
        elif toro_source_frac > 0.0:
            pick = torch.tensor(rng.random(n) < toro_source_frac, device=device)
            x = torch.where(pick.view(n, 1, 1), toro_x, x)
    dt = 1.0 / steps
    for k in range(steps):
        t = torch.full((n, 1), k * dt, device=device)
        x = (x + model(x, mask, t, cond) * dt) * m
    x = x.cpu().numpy()
    out = []
    for i, k in enumerate(lengths):
        tok = x[i, :k] * data.scale[1] + data.scale[0]
        out.append(decode_tokens(tok, float(depths[i])))
    return out


def load_split():
    loader = FlowMatchingDataLoader(DATA)
    loader.load_data()
    split = json.loads(SPLIT.read_text())
    pick = lambda key: [loader.sequences[i] for i in split[key]]  # noqa: E731
    return pick("train"), pick("val"), pick("test")


def train(
    seed: int,
    epochs: int,
    device: torch.device,
    tag: str,
    use_toro_source: bool = True,
    toro_source_frac: float = 1.0,
) -> dict:
    set_seed(seed)
    train_seq, val_seq, _ = load_split()
    tr = ConstrainedTokens(train_seq)
    va = ConstrainedTokens(val_seq, scale=tr.scale)
    floor = tr.h_floor()
    scale_t = torch.tensor(tr.scale, dtype=torch.float32, device=device)
    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.6, 0.999), weight_decay=1e-4)
    sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10, min_lr=1e-6)
    tensors = {
        name: tuple(torch.tensor(a, device=device) for a in (d.x, d.mask, d.cond))
        for name, d in (("train", tr), ("val", va))
    }
    if not use_toro_source:
        toro_source_frac = 0.0
    toro_fit = fit_toro(train_seq) if toro_source_frac > 0.0 else None
    order_rng = np.random.default_rng(42)
    source_rng = np.random.default_rng(seed + 17)
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
            rows_cpu = rows.detach().cpu().numpy()
            if toro_fit is None:
                x0 = None
            else:
                x0 = torch.tensor(
                    encode_toro_batch(
                        toro_fit,
                        tr.depths[rows_cpu],
                        tr.lengths[rows_cpu],
                        tr.scale,
                        source_rng,
                    ),
                    device=device,
                )
                if toro_source_frac < 1.0:
                    noise = torch.randn_like(x0) * mk[rows].unsqueeze(-1).float()
                    pick = torch.tensor(
                        source_rng.random(len(rows_cpu)) < toro_source_frac,
                        device=device,
                    )
                    x0 = torch.where(pick.view(len(rows_cpu), 1, 1), x0, noise)
            loss, pen = flow_loss(
                model, x1[rows], mk[rows], cd[rows], scale_t, floor, THIN_PENALTY_WEIGHT, x0
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
                if toro_fit is None:
                    x0_val = None
                else:
                    val_rng = np.random.default_rng(1234)
                    x0_val = torch.tensor(
                        encode_toro_batch(toro_fit, va.depths, va.lengths, tr.scale, val_rng),
                        device=device,
                    )
                val, _ = flow_loss(model, vx, vm, vc, scale_t, floor, 0.0, x0_val)
            val = float(val)
            sched.step(val)
            history.append(
                {"epoch": epoch, "train": total / seen, "penalty": pen_total / seen, "val": val}
            )
            state = {
                "model": model.state_dict(),
                "epoch": epoch,
                "scale": tr.scale,
                "use_toro_source": use_toro_source,
                "toro_source_frac": toro_source_frac,
            }
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
        "h_floor": floor,
        "use_toro_source": use_toro_source,
        "toro_source_frac": toro_source_frac,
        "history": history,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--no-toro-source", action="store_true")
    parser.add_argument(
        "--toro-source-frac",
        type=float,
        default=1.0,
        help="Fraction of the batch that starts from a Toro profile (rest is Gaussian).",
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = args.tag or f"s{args.seed}"
    summary = train(
        args.seed,
        args.epochs,
        device,
        tag,
        use_toro_source=not args.no_toro_source,
        toro_source_frac=0.0 if args.no_toro_source else args.toro_source_frac,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
