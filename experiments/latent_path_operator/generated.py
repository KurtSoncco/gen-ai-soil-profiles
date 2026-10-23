"""Profiles from the constrained flow matcher, labeled by the Rayleigh solver.

The deepest layer is the half-space. The layers above it are the soil the
chain reads. A one-layer profile has no soil and is dropped.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.flow_matching_simplified.eval_corrected import layer_model
from experiments.flow_matching_simplified.toro_baseline import fit_toro
from experiments.flow_matching_simplified.train_constrained import (
    ConstrainedTokens,
    build_model,
    load_split,
    sample,
)
from experiments.latent_path_operator.data import TF_FREQS, _pad
from experiments.latent_path_operator.eval_tf import (
    per_column_slices,
    plot_metric_boxes,
    plot_profile_spectra,
)
from experiments.latent_path_operator.operators import build_tf
from experiments.latent_path_operator.solvers import rayleigh_field
from experiments.latent_path_operator.train import (
    batch_loss,
    predict,
    relative_l1_by_frequency,
)

# The default tag ``s42`` was not the run that was saved. This is the seed-42
# constrained generator.
GENERATOR_CKPT = Path("outputs/flow_matching_simplified_constrained/s42_e500/best.pt")
POOL_PATH = Path("outputs/latent_path_operator/generated_pool.pt")
HOLDOUT_PATH = Path("outputs/latent_path_operator/generated_holdout.pt")
CHAIN_OUT = Path("outputs/latent_path_operator/tf_chain_generated.pt")


def soil_layers(
    sequence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Soil thickness, soil Vs, and half-space Vs. ``None`` when unusable."""
    modeled = layer_model(sequence, include_invalid=False)
    if modeled is None:
        return None
    bounds, velocity = modeled
    if len(velocity) < 2:
        return None
    thickness = np.diff(bounds[:-1])
    if np.any(thickness <= 0) or np.any(velocity[:-1] <= 0):
        return None
    return thickness, velocity[:-1].copy(), float(velocity[-1])


def batch_from_columns(
    columns: list[tuple[np.ndarray, np.ndarray, float]],
    n_query: int = 8,
    rng: np.random.Generator | None = None,
    dtype: torch.dtype = torch.float32,
):
    """Outcrop labels on the fixed frequency grid. Query 0 is the surface."""
    from experiments.latent_path_operator.data import ColumnBatch

    rng = np.random.default_rng() if rng is None else rng
    thicknesses, velocities, queries, targets, beds = [], [], [], [], []
    for thickness, velocity, bedrock in columns:
        depth = float(thickness.sum())
        query = np.empty(n_query, dtype=np.float64)
        query[0] = 0.0
        query[1] = 0.5 * depth
        if n_query > 2:
            query[2:] = rng.uniform(0.0, depth, size=n_query - 2)
        field, _ = rayleigh_field(thickness, velocity, bedrock, TF_FREQS, query)
        thicknesses.append(thickness)
        velocities.append(velocity[:, None])
        queries.append(query)
        targets.append(np.stack([field.real, field.imag], axis=-1))
        beds.append(bedrock)
    width = max(len(row) for row in thicknesses)
    thickness_a, mask = _pad(thicknesses, width)
    props, _ = _pad(velocities, width)
    bounds = np.cumsum(
        np.concatenate([np.zeros((len(columns), 1)), thickness_a], axis=1), axis=1
    )
    sensor_z = bounds[:, :-1] + 0.5 * thickness_a
    freq = np.broadcast_to(TF_FREQS, (len(columns), len(TF_FREQS))).copy()
    return ColumnBatch(
        thickness=torch.tensor(thickness_a, dtype=dtype),
        props=torch.tensor(props, dtype=dtype),
        layer_mask=torch.tensor(mask),
        sensor_z=torch.tensor(sensor_z, dtype=dtype),
        sensor_props=torch.tensor(props, dtype=dtype),
        sensor_mask=torch.tensor(mask),
        query_z=torch.tensor(np.stack(queries), dtype=dtype),
        bc=torch.tensor(beds, dtype=dtype)[:, None],
        freq=torch.tensor(freq, dtype=dtype),
        target=torch.tensor(np.stack(targets), dtype=dtype),
    )


def _subset(batch, index: np.ndarray):
    from experiments.latent_path_operator.data import ColumnBatch

    choice = torch.as_tensor(index, dtype=torch.long)

    def take(value):
        return None if value is None else value.index_select(0, choice)

    return ColumnBatch(
        thickness=take(batch.thickness),
        props=take(batch.props),
        layer_mask=take(batch.layer_mask),
        sensor_z=take(batch.sensor_z),
        sensor_props=take(batch.sensor_props),
        sensor_mask=take(batch.sensor_mask),
        query_z=take(batch.query_z),
        bc=take(batch.bc),
        freq=take(batch.freq),
        target=take(batch.target),
        wavelet=take(batch.wavelet),
    )


def _load_generator(device: torch.device):
    if not GENERATOR_CKPT.exists():
        raise SystemExit(
            f"missing generator checkpoint {GENERATOR_CKPT}; "
            "the generator is not trained from here"
        )
    ckpt = torch.load(GENERATOR_CKPT, map_location=device, weights_only=False)
    model = build_model().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def draw_columns(
    n: int,
    seed: int,
    device: torch.device | None = None,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Valid soil columns conditioned on the test split."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = _load_generator(device)
    train_seq, _, test_seq = load_split()
    sites = ConstrainedTokens(test_seq, scale=np.asarray(ckpt["scale"]))
    fraction = float(ckpt.get("toro_source_frac", 1.0))
    if not ckpt.get("use_toro_source", True):
        fraction = 0.0
    toro = fit_toro(train_seq) if fraction > 0.0 else None
    columns: list[tuple[np.ndarray, np.ndarray, float]] = []
    attempt = seed
    while len(columns) < n:
        sequences = sample(
            model,
            sites,
            min(64, n),
            attempt,
            device,
            toro_fit=toro,
            toro_source_frac=fraction,
        )
        for sequence in sequences:
            column = soil_layers(sequence)
            if column is None:
                continue
            columns.append(column)
            if len(columns) == n:
                break
        attempt += 1
        if attempt > seed + 30:
            raise RuntimeError(f"only {len(columns)} valid profiles after 30 draws")
    return columns


def cache_splits(pool_n: int = 512, holdout_n: int = 64) -> None:
    """Label a training pool and a holdout that the fine-tune never sees."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    POOL_PATH.parent.mkdir(parents=True, exist_ok=True)
    pool = batch_from_columns(
        draw_columns(pool_n, seed=0, device=device),
        rng=np.random.default_rng(0),
    )
    holdout = batch_from_columns(
        draw_columns(holdout_n, seed=10_000, device=device),
        rng=np.random.default_rng(1),
    )
    torch.save(pool, POOL_PATH)
    torch.save(holdout, HOLDOUT_PATH)
    print(f"wrote {POOL_PATH} ({pool_n}) and {HOLDOUT_PATH} ({holdout_n})", flush=True)


def _load_chain(path: Path, device: torch.device):
    model = build_tf("chain").to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def score_holdout(checkpoint: Path, holdout_path: Path = HOLDOUT_PATH):
    """Per-column slices and surface magnitudes for one chain checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    holdout = torch.load(holdout_path, map_location="cpu", weights_only=False)
    model = _load_chain(checkpoint, device)
    predicted = []
    truth = []
    with torch.no_grad():
        for start in range(0, holdout.thickness.shape[0], 16):
            index = np.arange(start, min(start + 16, holdout.thickness.shape[0]))
            batch = _subset(holdout, index).to(device)
            predicted.append(predict(model, batch).detach().cpu().numpy())
            truth.append(batch.target.detach().cpu().numpy())
    curves = np.concatenate(predicted, axis=0)
    exact = np.concatenate(truth, axis=0)
    return per_column_slices(curves, exact), curves, exact, holdout


def finetune_chain(
    steps: int = 1000,
    batch_size: int = 32,
    lr: float = 3e-4,
    seed: int = 0,
) -> None:
    """Continue the Rayleigh chain on the generated pool. A fresh optimizer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pool = torch.load(POOL_PATH, map_location="cpu", weights_only=False)
    model = _load_chain(Path("outputs/latent_path_operator/tf_chain.pt"), device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    n = pool.thickness.shape[0]
    for step in range(steps):
        index = rng.integers(0, n, size=batch_size)
        batch = _subset(pool, index).to(device)
        opt.zero_grad(set_to_none=True)
        value = batch_loss(model, batch, relative_l1_by_frequency)
        value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step + 1 == steps:
            print(
                f"step {step + 1}/{steps}  train relative L1 {float(value.detach()):.4f}",
                flush=True,
            )
    CHAIN_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "spec": "tf",
            "arch": "chain",
            "steps": steps,
            "loss": "L1",
        },
        CHAIN_OUT,
    )
    print(f"wrote {CHAIN_OUT}", flush=True)


def _surface_abs(real_imag: np.ndarray) -> np.ndarray:
    return np.abs(real_imag[:, :, 0, 0] + 1j * real_imag[:, :, 0, 1])


def write_generated_figures(before_curves: np.ndarray, before_groups, holdout) -> None:
    after_groups, after_curves, exact, _ = score_holdout(CHAIN_OUT)
    plot_metric_boxes(
        {"Chain": before_groups, "Fine-tuned": after_groups},
        Path("outputs/latent_path_operator/tf_generated_boxes.png"),
    )
    thickness = holdout.thickness.numpy()
    mask = holdout.layer_mask.numpy()
    velocity = holdout.props.numpy()[..., 0]
    profiles_h, profiles_v = [], []
    for index in range(3):
        live = mask[index]
        profiles_h.append(thickness[index, live])
        profiles_v.append(velocity[index, live])
    plot_profile_spectra(
        profiles_h,
        profiles_v,
        _surface_abs(exact),
        _surface_abs(before_curves),
        _surface_abs(after_curves),
        Path("outputs/latent_path_operator/tf_generated_profiles.png"),
    )
    for label, groups in (("chain", before_groups), ("fine-tuned", after_groups)):
        surface = groups["surface"]["relative_l1"].mean()
        high = groups["all_depths_above_3hz"]["relative_l1"].mean()
        print(
            f"{label}  holdout surface L1 {surface:.4f}  all-depth above 3 Hz L1 {high:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Couple the generator and the chain")
    parser.add_argument(
        "--stage",
        choices=["cache", "score", "finetune", "figures", "all"],
        default="all",
    )
    args = parser.parse_args()
    if args.stage in ("cache", "all"):
        cache_splits()
    if args.stage in ("score", "all"):
        groups, curves, _, holdout = score_holdout(
            Path("outputs/latent_path_operator/tf_chain.pt")
        )
        torch.save(
            {"groups": groups, "curves": curves, "holdout": holdout},
            Path("outputs/latent_path_operator/generated_before.pt"),
        )
        surface = groups["surface"]["relative_l1"].mean()
        print(f"pretrained chain holdout surface L1 {surface:.4f}", flush=True)
    if args.stage in ("finetune", "all"):
        finetune_chain()
    if args.stage in ("figures", "all"):
        saved = torch.load(
            Path("outputs/latent_path_operator/generated_before.pt"),
            map_location="cpu",
            weights_only=False,
        )
        write_generated_figures(saved["curves"], saved["groups"], saved["holdout"])


if __name__ == "__main__":
    main()
