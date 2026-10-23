"""On-the-fly collocation batches for the linear column operators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from experiments.latent_path_operator.ricker import (
    DT,
    N_TIME,
    positive_frequencies,
    rfft_coefficients,
    velocity_traces,
)
from experiments.latent_path_operator.solvers import rayleigh_field, static_field

MAX_LAYERS = 8
# Half-space under every transfer-function column. The soil profile changes;
# this bedrock velocity does not.
BEDROCK_VS = 1500.0
# 200 log points cover the tail and the narrow fundamental. 800 log points
# from 1 to 20 Hz are where the higher peaks sit.
_TF_LOW = np.logspace(np.log10(0.01), np.log10(1.0), 200, endpoint=False)
_TF_HIGH = np.logspace(np.log10(1.0), np.log10(20.0), 800)
TF_FREQS = np.concatenate([_TF_LOW, _TF_HIGH])


@dataclass
class ColumnBatch:
    thickness: Tensor
    props: Tensor
    layer_mask: Tensor
    sensor_z: Tensor
    sensor_props: Tensor
    sensor_mask: Tensor
    query_z: Tensor
    bc: Tensor
    freq: Tensor | None
    target: Tensor
    wavelet: Tensor | None = None

    def to(self, device: torch.device) -> "ColumnBatch":
        def move(value):
            return None if value is None else value.to(device)

        return ColumnBatch(
            thickness=move(self.thickness),
            props=move(self.props),
            layer_mask=move(self.layer_mask),
            sensor_z=move(self.sensor_z),
            sensor_props=move(self.sensor_props),
            sensor_mask=move(self.sensor_mask),
            query_z=move(self.query_z),
            bc=move(self.bc),
            freq=move(self.freq),
            target=move(self.target),
            wavelet=move(self.wavelet),
        )


def _pad(rows: list[np.ndarray], width: int) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros((len(rows), width) + rows[0].shape[1:], dtype=np.float64)
    mask = np.zeros((len(rows), width), dtype=bool)
    for i, row in enumerate(rows):
        n = len(row)
        out[i, :n] = row
        mask[i, :n] = True
    return out, mask


def sample_sh_batch(
    batch_size: int,
    n_freq: int = 3,
    n_query: int = 6,
    rng: np.random.Generator | None = None,
    dtype: torch.dtype = torch.float32,
) -> ColumnBatch:
    """Soil layers plus a half-space. N in the Haskell column is 2 to 8."""
    rng = np.random.default_rng() if rng is None else rng
    thicknesses, velocities, beds, freqs, queries, targets = [], [], [], [], [], []
    for _ in range(batch_size):
        n_soil = int(rng.integers(1, MAX_LAYERS))
        thickness = rng.uniform(2.0, 25.0, size=n_soil)
        vs = np.exp(rng.uniform(np.log(150.0), np.log(800.0), size=n_soil))
        vs_bed = float(np.exp(rng.uniform(np.log(800.0), np.log(2000.0))))
        freq = np.exp(rng.uniform(np.log(0.5), np.log(20.0), size=n_freq))
        depth = float(thickness.sum())
        query = np.empty(n_query, dtype=np.float64)
        query[0] = 0.0
        query[1] = depth
        if n_query > 2:
            query[2:] = rng.uniform(0.0, depth, size=n_query - 2)
        field, _ = rayleigh_field(thickness, vs, vs_bed, freq, query)
        thicknesses.append(thickness)
        velocities.append(vs[:, None])
        beds.append(vs_bed)
        freqs.append(freq)
        queries.append(query)
        targets.append(np.stack([field.real, field.imag], axis=-1))
    thickness_a, mask = _pad(thicknesses, MAX_LAYERS)
    props, _ = _pad(velocities, MAX_LAYERS)
    # One midpoint sensor per layer slot. Padded slots are masked.
    bounds = np.cumsum(
        np.concatenate([np.zeros((batch_size, 1)), thickness_a], axis=1), axis=1
    )
    sensor_z = bounds[:, :-1] + 0.5 * thickness_a
    return ColumnBatch(
        thickness=torch.tensor(thickness_a, dtype=dtype),
        props=torch.tensor(props, dtype=dtype),
        layer_mask=torch.tensor(mask),
        sensor_z=torch.tensor(sensor_z, dtype=dtype),
        sensor_props=torch.tensor(props, dtype=dtype),
        sensor_mask=torch.tensor(mask),
        query_z=torch.tensor(np.stack(queries), dtype=dtype),
        bc=torch.tensor(beds, dtype=dtype)[:, None],
        freq=torch.tensor(np.stack(freqs), dtype=dtype),
        target=torch.tensor(np.stack(targets), dtype=dtype),
    )


def sample_static_batch(
    batch_size: int,
    n_query: int = 6,
    rng: np.random.Generator | None = None,
    dtype: torch.dtype = torch.float32,
) -> ColumnBatch:
    rng = np.random.default_rng() if rng is None else rng
    thicknesses, props_l, loads, queries, targets = [], [], [], [], []
    for _ in range(batch_size):
        n_layers = int(rng.integers(1, MAX_LAYERS + 1))
        thickness = rng.uniform(1.0, 10.0, size=n_layers)
        gamma = rng.uniform(15.0e3, 22.0e3, size=n_layers)
        modulus = np.exp(rng.uniform(np.log(10.0e6), np.log(200.0e6), size=n_layers))
        load = float(rng.uniform(10.0e3, 200.0e3))
        depth = float(thickness.sum())
        query = np.empty(n_query, dtype=np.float64)
        query[0] = 0.0
        query[1] = depth
        if n_query > 2:
            query[2:] = rng.uniform(0.0, depth, size=n_query - 2)
        field = static_field(thickness, gamma, modulus, load, query)
        thicknesses.append(thickness)
        props_l.append(np.stack([gamma, modulus], axis=-1))
        loads.append(load)
        queries.append(query)
        targets.append(field)
    thickness_a, mask = _pad(thicknesses, MAX_LAYERS)
    props, _ = _pad(props_l, MAX_LAYERS)
    bounds = np.cumsum(
        np.concatenate([np.zeros((batch_size, 1)), thickness_a], axis=1), axis=1
    )
    sensor_z = bounds[:, :-1] + 0.5 * thickness_a
    return ColumnBatch(
        thickness=torch.tensor(thickness_a, dtype=dtype),
        props=torch.tensor(props, dtype=dtype),
        layer_mask=torch.tensor(mask),
        sensor_z=torch.tensor(sensor_z, dtype=dtype),
        sensor_props=torch.tensor(props, dtype=dtype),
        sensor_mask=torch.tensor(mask),
        query_z=torch.tensor(np.stack(queries), dtype=dtype),
        bc=torch.tensor(loads, dtype=dtype)[:, None],
        freq=None,
        target=torch.tensor(np.stack(targets), dtype=dtype),
    )


def sample_sh_ricker_batch(
    batch_size: int,
    rng: np.random.Generator | None = None,
    dtype: torch.dtype = torch.float32,
    f_peak: float | None = None,
    f_range: tuple[float, float] = (2.0, 8.0),
) -> ColumnBatch:
    """Rayleigh displacement at the Ricker bins, and the velocity traces as targets."""
    rng = np.random.default_rng() if rng is None else rng
    freqs = positive_frequencies()
    thicknesses, velocities, beds, queries, targets, waves = [], [], [], [], [], []
    for _ in range(batch_size):
        n_soil = int(rng.integers(1, MAX_LAYERS))
        thickness = rng.uniform(2.0, 25.0, size=n_soil)
        vs = np.exp(rng.uniform(np.log(150.0), np.log(800.0), size=n_soil))
        vs_bed = float(np.exp(rng.uniform(np.log(800.0), np.log(2000.0))))
        peak = float(f_peak) if f_peak is not None else float(rng.uniform(*f_range))
        depth = float(thickness.sum())
        query = np.array([0.0, 0.5 * depth, depth])
        field, _ = rayleigh_field(thickness, vs, vs_bed, freqs, query)
        coeff = rfft_coefficients(freqs, peak, DT)
        traces = velocity_traces(field, coeff, N_TIME)
        thicknesses.append(thickness)
        velocities.append(vs[:, None])
        beds.append(vs_bed)
        queries.append(query)
        targets.append(traces)
        waves.append(coeff)
    thickness_a, mask = _pad(thicknesses, MAX_LAYERS)
    props, _ = _pad(velocities, MAX_LAYERS)
    bounds = np.cumsum(
        np.concatenate([np.zeros((batch_size, 1)), thickness_a], axis=1), axis=1
    )
    sensor_z = bounds[:, :-1] + 0.5 * thickness_a
    freq = np.broadcast_to(freqs, (batch_size, len(freqs))).copy()
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
        wavelet=torch.tensor(np.stack(waves), dtype=torch.complex64),
    )


def sample_tf_batch(
    batch_size: int,
    n_query: int = 8,
    rng: np.random.Generator | None = None,
    dtype: torch.dtype = torch.float32,
) -> ColumnBatch:
    """Outcrop transfer function ``u(z, f) / u_outcrop`` on a fixed half-space.

    Query 0 is the surface. The unit outcrop is the fixed bedrock reference,
    so a profile only has to move the surface and the interior depths.
    """
    rng = np.random.default_rng() if rng is None else rng
    thicknesses, velocities, queries, targets = [], [], [], []
    for _ in range(batch_size):
        n_soil = int(rng.integers(1, MAX_LAYERS))
        thickness = rng.uniform(2.0, 25.0, size=n_soil)
        vs = np.exp(rng.uniform(np.log(150.0), np.log(800.0), size=n_soil))
        depth = float(thickness.sum())
        query = np.empty(n_query, dtype=np.float64)
        query[0] = 0.0
        query[1] = 0.5 * depth
        if n_query > 2:
            query[2:] = rng.uniform(0.0, depth, size=n_query - 2)
        field, _ = rayleigh_field(thickness, vs, BEDROCK_VS, TF_FREQS, query)
        thicknesses.append(thickness)
        velocities.append(vs[:, None])
        queries.append(query)
        targets.append(np.stack([field.real, field.imag], axis=-1))
    thickness_a, mask = _pad(thicknesses, MAX_LAYERS)
    props, _ = _pad(velocities, MAX_LAYERS)
    bounds = np.cumsum(
        np.concatenate([np.zeros((batch_size, 1)), thickness_a], axis=1), axis=1
    )
    sensor_z = bounds[:, :-1] + 0.5 * thickness_a
    freq = np.broadcast_to(TF_FREQS, (batch_size, len(TF_FREQS))).copy()
    return ColumnBatch(
        thickness=torch.tensor(thickness_a, dtype=dtype),
        props=torch.tensor(props, dtype=dtype),
        layer_mask=torch.tensor(mask),
        sensor_z=torch.tensor(sensor_z, dtype=dtype),
        sensor_props=torch.tensor(props, dtype=dtype),
        sensor_mask=torch.tensor(mask),
        query_z=torch.tensor(np.stack(queries), dtype=dtype),
        bc=torch.full((batch_size, 1), BEDROCK_VS, dtype=dtype),
        freq=torch.tensor(freq, dtype=dtype),
        target=torch.tensor(np.stack(targets), dtype=dtype),
    )
