"""Train the latent path operator. No physics residual.

Shear and static models use relative L2. The transfer function uses a
relative L1 averaged over frequency, so a short secondary peak counts as
much as an error on the fundamental.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from experiments.latent_path_operator.data import (
    ColumnBatch,
    sample_sh_ricker_batch,
    sample_static_batch,
    sample_tf_batch,
)
from experiments.latent_path_operator.model import LatentPathOperator
from experiments.latent_path_operator.operators import build_tf
from experiments.latent_path_operator.spec import spec_by_name


def relative_l2(pred: Tensor, target: Tensor) -> Tensor:
    """Mean over the batch of ||pred - target|| / ||target||."""
    err = (pred - target).reshape(pred.shape[0], -1)
    ref = target.reshape(target.shape[0], -1)
    return (err.norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-8)).mean()


def relative_l1(pred: Tensor, target: Tensor) -> Tensor:
    """Mean over the batch of sum(|pred - target|) / sum(|target|)."""
    err = (pred - target).abs().reshape(pred.shape[0], -1).sum(dim=-1)
    ref = target.abs().reshape(target.shape[0], -1).sum(dim=-1)
    return (err / ref.clamp_min(1e-8)).mean()


def relative_l1_by_frequency(pred: Tensor, target: Tensor) -> Tensor:
    """Relative L1 averaged over frequencies, so each peak counts the same.

    The denominator is at least 5% of that column's mean absolute response,
    which keeps a near-null frequency from dominating the mean.
    """
    err = (pred - target).abs().flatten(start_dim=2).sum(dim=-1)
    ref = target.abs().flatten(start_dim=2).sum(dim=-1)
    floor = (0.05 * ref.mean(dim=-1, keepdim=True)).clamp_min(1e-4)
    return (err / torch.maximum(ref, floor)).mean()


def ricker_velocity(disp: Tensor, wavelet: Tensor, n_time: int) -> Tensor:
    """Time-domain velocity at each query depth. ``disp`` is (B, F, M, 2)."""
    displacement = torch.complex(disp[..., 0], disp[..., 1])
    spectrum = displacement.new_zeros(
        displacement.shape[0], displacement.shape[1] + 1, displacement.shape[2]
    )
    spectrum[:, 1:, :] = displacement * wavelet[:, :, None].to(displacement.dtype)
    traces = torch.fft.irfft(spectrum, n=n_time, dim=1)
    return traces.transpose(1, 2)


def batch_loss(
    model: torch.nn.Module,
    batch: ColumnBatch,
    loss=relative_l2,
) -> Tensor:
    predicted = predict(model, batch)
    if batch.wavelet is None:
        return loss(predicted, batch.target)
    velocity = ricker_velocity(predicted, batch.wavelet, batch.target.shape[-1])
    return loss(velocity, batch.target)


def predict(model: torch.nn.Module, batch: ColumnBatch) -> Tensor:
    return model(
        thickness=batch.thickness,
        sensor_z=batch.sensor_z,
        sensor_props=batch.sensor_props,
        sensor_mask=batch.sensor_mask,
        layer_mask=batch.layer_mask,
        query_z=batch.query_z,
        bc=batch.bc,
        freq=batch.freq,
    )


def fit_batch(
    model: torch.nn.Module,
    batch: ColumnBatch,
    steps: int,
    lr: float = 1e-3,
    loss=relative_l2,
) -> list[float]:
    """Gradient steps on one frozen batch. Returns the loss after each step."""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        value = batch_loss(model, batch, loss)
        value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(value.detach()))
    return losses


def _batch(
    model_name: str,
    batch_size: int,
    rng: np.random.Generator,
    f_peak: float | None = None,
    f_range: tuple[float, float] = (2.0, 8.0),
) -> ColumnBatch:
    if model_name == "sh":
        return sample_sh_ricker_batch(
            batch_size, rng=rng, f_peak=f_peak, f_range=f_range
        )
    if model_name == "tf":
        return sample_tf_batch(batch_size, rng=rng)
    return sample_static_batch(batch_size, rng=rng)


def train(
    model_name: str = "sh",
    steps: int = 200,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 0,
    out_dir: str | Path = "outputs/latent_path_operator",
    use_wandb: bool = False,
    resume: bool = False,
    arch: str = "modal",
) -> list[float]:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name == "tf":
        model = build_tf(arch).to(device)
    else:
        model = LatentPathOperator(spec_by_name(model_name)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    out = Path(out_dir)
    stem = f"tf_{arch}" if model_name == "tf" else model_name
    ckpt_path = out / f"{stem}.pt"
    start = 0
    losses: list[float] = []
    if resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        losses = list(ckpt.get("train_loss", []))
        start = int(ckpt.get("steps", len(losses)))
        if ckpt.get("opt") is not None:
            opt.load_state_dict(ckpt["opt"])
        print(f"resumed {model_name} at step {start}", flush=True)
    if use_wandb:
        import wandb

        wandb.init(
            project="latent-path-operator",
            config={"model": model_name, "steps": steps, "device": str(device)},
        )
    model.train()
    stop = start + steps
    score = relative_l1_by_frequency if model_name == "tf" else relative_l2
    label = "L1" if model_name == "tf" else "L2"
    print(f"{model_name} on {device}", flush=True)
    for step in range(start, stop):
        # Low peak frequencies first, so a travel-time error is a fraction of
        # the wavelet instead of a full cycle.
        if model_name == "sh" and start == 0 and step < steps // 2:
            f_range = (2.0, 4.0)
        else:
            f_range = (2.0, 8.0)
        batch = _batch(
            model_name,
            batch_size,
            np.random.default_rng(seed + step),
            f_range=f_range,
        ).to(device)
        opt.zero_grad(set_to_none=True)
        loss = batch_loss(model, batch, score)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
        if (step - start) % 50 == 0 or step + 1 == stop:
            print(
                f"step {step + 1}/{stop}  train relative {label} {losses[-1]:.4f}",
                flush=True,
            )
        if use_wandb:
            import wandb

            wandb.log({"loss": losses[-1], "step": step})
    model.eval()
    held_losses = []
    with torch.no_grad():
        for extra in range(8):
            held_batch = _batch(
                model_name, batch_size, np.random.default_rng(seed + 100_000 + extra)
            ).to(device)
            held_losses.append(float(batch_loss(model, held_batch, score)))
    held_out = float(np.mean(held_losses))
    tail = float(np.mean(losses[-50:]))
    print(
        f"train relative {label} {losses[0]:.4f} -> {tail:.4f} over the last 50 steps",
        flush=True,
    )
    print(f"held-out relative {label} {held_out:.4f} on 8 batches", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "spec": model_name,
            "arch": arch if model_name == "tf" else "",
            "steps": stop,
            "train_loss": losses,
            "held_out_relative_l2": held_out,
            "loss": label,
        },
        ckpt_path,
    )
    return losses


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the linear latent path operator"
    )
    parser.add_argument("--model", choices=["sh", "static", "tf"], default="sh")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs/latent_path_operator")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--arch",
        choices=["modal", "laplace", "wavelet", "chain"],
        default="modal",
    )
    args = parser.parse_args()
    losses = train(
        model_name=args.model,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        out_dir=args.out,
        use_wandb=args.wandb,
        resume=args.resume,
        arch=args.arch,
    )
    label = "L1" if args.model == "tf" else "L2"
    print(f"final train relative {label} {losses[-1]:.4f}")


if __name__ == "__main__":
    main()
