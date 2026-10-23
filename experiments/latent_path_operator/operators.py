"""Three transfer-function operators, each with one encoder.

Laplace and wavelet are unstacked DeepONets: one branch emits one coefficient
vector, one trunk evaluates ``(f, z/H)``, and the unit outcrop sits outside
the product. The Rayleigh chain is one encoder over ordered layers, then the
existing layer matrix.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from experiments.latent_path_operator.propagator import (
    RAYLEIGH_A0,
    RAYLEIGH_A1,
    column_bounds,
    propagate_and_query,
    rayleigh_boundary,
    rayleigh_generator,
)

N_POLES = 12
N_WAVE_CENTERS = 16
N_WAVE_DEPTH = 3
COMPLIANCE_SCALE = 0.05


class SequenceBranch(nn.Module):
    """Ordered layers to one coefficient vector. A zero readout stays zero."""

    def __init__(self, out_dim: int, hidden: int = 128):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        block = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=4,
            dim_feedforward=hidden * 2,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=2)
        self.read = nn.Linear(hidden, out_dim)
        nn.init.zeros_(self.read.weight)
        nn.init.zeros_(self.read.bias)

    def forward(self, thickness: Tensor, vs: Tensor, mask: Tensor) -> Tensor:
        token = self._tokens(thickness, vs, mask)
        pooled = (token * mask.unsqueeze(-1)).sum(dim=1)
        pooled = pooled / mask.sum(dim=1).clamp_min(1).unsqueeze(-1).to(pooled.dtype)
        return self.read(pooled)

    def tokens(self, thickness: Tensor, vs: Tensor, mask: Tensor) -> Tensor:
        return self._tokens(thickness, vs, mask)

    def _tokens(self, thickness: Tensor, vs: Tensor, mask: Tensor) -> Tensor:
        live = mask.to(thickness.dtype)
        height = (thickness * live).sum(dim=-1).clamp_min(1e-6)
        bounds = column_bounds(thickness, mask)
        mid = (bounds[:, :-1] + 0.5 * thickness) / height[:, None]
        fraction = (thickness * live) / height[:, None]
        features = torch.stack(
            [torch.log(vs.clamp_min(1.0)), fraction, mid],
            dim=-1,
        )
        token = self.embed(features)
        return self.encoder(token, src_key_padding_mask=~mask)


def _depth_basis(zeta: Tensor) -> Tensor:
    """``(B, M, 3)``. The first column is 1, so a zero shape still passes the residue."""
    return torch.stack(
        [
            torch.ones_like(zeta),
            torch.cos(math.pi * zeta) - 1.0,
            torch.cos(2.0 * math.pi * zeta) - 1.0,
        ],
        dim=-1,
    )


def _zeta(thickness: Tensor, mask: Tensor, query_z: Tensor) -> Tensor:
    height = (thickness * mask.to(thickness.dtype)).sum(dim=-1).clamp_min(1e-6)
    return (query_z / height[:, None]).clamp(0.0, 1.0)


def _vs(sensor_props: Tensor) -> Tensor:
    return sensor_props[..., 0]


class LaplaceOperator(nn.Module):
    """One branch, one pole-residue trunk. ``1`` at initialization."""

    def __init__(self, n_poles: int = N_POLES, hidden: int = 128):
        super().__init__()
        self.n_poles = n_poles
        self.branch = SequenceBranch(8 * n_poles, hidden)
        centers = torch.logspace(math.log10(0.3), math.log10(16.0), n_poles)
        self.register_buffer("f_base", centers)

    def forward(
        self,
        thickness: Tensor,
        sensor_z: Tensor,
        sensor_props: Tensor,
        sensor_mask: Tensor,
        layer_mask: Tensor,
        query_z: Tensor,
        bc: Tensor,
        freq: Tensor | None = None,
    ) -> Tensor:
        del sensor_z, sensor_mask, bc
        if freq is None:
            raise ValueError("transfer queries need a frequency")
        coeff = self.branch(thickness, _vs(sensor_props), layer_mask)
        zeta = _zeta(thickness, layer_mask, query_z)
        transfer = 1.0 + self._kernel(freq, zeta, coeff)
        return torch.stack((transfer.real, transfer.imag), dim=-1)

    def _kernel(self, freq: Tensor, zeta: Tensor, coeff: Tensor) -> Tensor:
        n = self.n_poles
        shift, damp, residue, shape = coeff.split([n, n, 2 * n, 4 * n], dim=-1)
        omega_p = (2.0 * math.pi) * self.f_base * torch.exp(0.6 * torch.tanh(shift))
        sigma = omega_p * (0.08 * torch.exp(0.6 * torch.tanh(damp)))
        beta = torch.tanh(residue.view(-1, n, 2))
        beta_c = torch.complex(beta[..., 0], beta[..., 1])
        omega = (2.0 * math.pi) * freq
        denom = sigma[:, None, :] + 1j * (omega[:, :, None] - omega_p[:, None, :])
        kernel = beta_c[:, None, :] / denom
        phi = self._shape(zeta, shape)
        return (kernel[:, :, None, :] * phi[:, None, :, :]).sum(dim=-1)

    def _shape(self, zeta: Tensor, raw: Tensor) -> Tensor:
        packed = torch.tanh(raw.view(-1, self.n_poles, 2, 2))
        basis = _depth_basis(zeta)
        first = torch.complex(packed[..., 0, 0], packed[..., 0, 1])
        second = torch.complex(packed[..., 1, 0], packed[..., 1, 1])
        return (
            basis[:, :, None, 0].to(first.dtype)
            + first[:, None, :] * basis[:, :, None, 1].to(first.dtype)
            + second[:, None, :] * basis[:, :, None, 2].to(second.dtype)
        )


class WaveletOperator(nn.Module):
    """One branch dotted with a fixed Morlet bank. ``1`` at initialization."""

    def __init__(self, hidden: int = 128):
        super().__init__()
        n_basis = N_WAVE_CENTERS * N_WAVE_DEPTH * 2
        self.n_basis = n_basis
        self.branch = SequenceBranch(2 * n_basis, hidden)
        centers = torch.logspace(math.log10(0.3), math.log10(18.0), N_WAVE_CENTERS)
        self.register_buffer("centers", centers)

    def forward(
        self,
        thickness: Tensor,
        sensor_z: Tensor,
        sensor_props: Tensor,
        sensor_mask: Tensor,
        layer_mask: Tensor,
        query_z: Tensor,
        bc: Tensor,
        freq: Tensor | None = None,
    ) -> Tensor:
        del sensor_z, sensor_mask, bc
        if freq is None:
            raise ValueError("transfer queries need a frequency")
        coeff = self.branch(thickness, _vs(sensor_props), layer_mask)
        zeta = _zeta(thickness, layer_mask, query_z)
        bank = self._bank(freq, zeta)
        weight = coeff.view(-1, 2, self.n_basis)
        real = (bank * weight[:, None, None, 0, :]).sum(dim=-1)
        imag = (bank * weight[:, None, None, 1, :]).sum(dim=-1)
        correction = torch.complex(4.0 * torch.tanh(real), 4.0 * torch.tanh(imag))
        transfer = 1.0 + correction
        return torch.stack((transfer.real, transfer.imag), dim=-1)

    def _bank(self, freq: Tensor, zeta: Tensor) -> Tensor:
        log_f = freq.clamp_min(1e-4).log()
        delta = log_f[:, :, None] - self.centers.log()
        envelope = torch.exp(-0.5 * (delta / 0.35) ** 2)
        oscillate = 5.0 * delta
        wave = torch.stack(
            [envelope * torch.cos(oscillate), envelope * torch.sin(oscillate)],
            dim=-1,
        )
        depth = _depth_basis(zeta)
        bank = wave[:, :, None, :, None, :] * depth[:, None, :, None, :, None]
        return bank.reshape(freq.shape[0], freq.shape[1], zeta.shape[1], -1)


class RayleighChain(nn.Module):
    """One encoder, one compliance per layer, then the Rayleigh matrix product."""

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        block = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=4,
            dim_feedforward=hidden * 2,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=2)
        self.read = nn.Linear(hidden, 1)
        nn.init.zeros_(self.read.weight)
        nn.init.constant_(self.read.bias, -5.3)

    def forward(
        self,
        thickness: Tensor,
        sensor_z: Tensor,
        sensor_props: Tensor,
        sensor_mask: Tensor,
        layer_mask: Tensor,
        query_z: Tensor,
        bc: Tensor,
        freq: Tensor | None = None,
    ) -> Tensor:
        del sensor_z, sensor_mask
        if freq is None:
            raise ValueError("transfer queries need a frequency")
        compliance = self._compliance(thickness, _vs(sensor_props), layer_mask)
        system, force = rayleigh_generator(
            compliance, freq, a0=RAYLEIGH_A0, a1=RAYLEIGH_A1
        )
        batch, n_freq = freq.shape
        n_query = query_z.shape[1]
        flat = batch * n_freq
        state_dim = 4
        system = system.reshape(flat, -1, state_dim, state_dim)
        force = force.reshape(flat, -1, state_dim)
        thick = thickness[:, None, :].expand(batch, n_freq, -1).reshape(flat, -1)
        live = layer_mask[:, None, :].expand(batch, n_freq, -1).reshape(flat, -1)
        query = query_z[:, None, :].expand(batch, n_freq, n_query).reshape(flat, -1)
        vs_bed = bc[:, :1].expand_as(freq).reshape(flat)
        b_top, c_top, b_base, c_base = rayleigh_boundary(
            vs_bed, freq.reshape(flat), a0=RAYLEIGH_A0, a1=RAYLEIGH_A1
        )
        state = propagate_and_query(
            system, force, thick, live, query, b_top, c_top, b_base, c_base, ridge=1e-6
        )
        state = state.reshape(batch, n_freq, n_query, state_dim)
        return state[..., :2]

    def _compliance(self, thickness: Tensor, vs: Tensor, mask: Tensor) -> Tensor:
        live = mask.to(thickness.dtype)
        height = (thickness * live).sum(dim=-1).clamp_min(1e-6)
        bounds = column_bounds(thickness, mask)
        mid = (bounds[:, :-1] + 0.5 * thickness) / height[:, None]
        fraction = (thickness * live) / height[:, None]
        features = torch.stack(
            [torch.log(vs.clamp_min(1.0)), fraction, mid],
            dim=-1,
        )
        token = self.encoder(self.embed(features), src_key_padding_mask=~mask)
        raw = self.read(token).squeeze(-1)
        return torch.exp(raw).clamp(max=COMPLIANCE_SCALE) * live


def build_tf(arch: str) -> nn.Module:
    """Transfer-function architecture. ``modal`` is the existing DeepONet."""
    from experiments.latent_path_operator.deeponet import TransferDeepONet

    if arch == "modal":
        return TransferDeepONet()
    if arch == "laplace":
        return LaplaceOperator()
    if arch == "wavelet":
        return WaveletOperator()
    if arch == "chain":
        return RayleighChain()
    raise ValueError(f"unknown transfer architecture {arch!r}")
