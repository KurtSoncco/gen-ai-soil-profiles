"""Latent path operator: path graph, latent slices, and a layer matrix exponential."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from experiments.latent_path_operator.propagator import (
    RAYLEIGH_A0,
    RAYLEIGH_A1,
    SIGMA_REF,
    assemble_generators,
    column_bounds,
    physical_generators,
    propagate_and_query,
    rayleigh_boundary,
    rayleigh_generator,
    static_boundary,
)
from experiments.latent_path_operator.spec import ColumnSpec
from experiments.flow_matching_simplified.site_metrics import DAMPING


# Caps ê above Vs_ref / Vs_min² for the sampled shear velocities, so one Adam
# step cannot leave the range where the layer matrix exponential stays finite.
COMPLIANCE_SCALE = 0.05


def pool_sensors(
    sensor_z: Tensor,
    sensor_props: Tensor,
    sensor_mask: Tensor,
    thickness: Tensor,
    layer_mask: Tensor,
) -> Tensor:
    """Mean of sensor properties in each layer. Extra sensors in a layer do not add channels."""
    bounds = column_bounds(thickness, layer_mask)
    z = sensor_z[..., None]
    top = bounds[:, None, :-1]
    bot = bounds[:, None, 1:]
    n_real = layer_mask.to(torch.long).sum(dim=-1)
    layer_ids = torch.arange(thickness.shape[-1], device=thickness.device)
    is_last = layer_ids[None, :] == (n_real[:, None] - 1)
    inside = (
        sensor_mask[:, :, None]
        & layer_mask[:, None, :]
        & (z >= top)
        & ((z < bot) | (is_last[:, None, :] & (z <= bot + 1e-8)))
    )
    weight = inside.to(sensor_props.dtype)
    summed = torch.einsum("bsn,bsc->bnc", weight, sensor_props)
    count = weight.sum(dim=1)
    return torch.where(
        count[..., None] > 0,
        summed / count.clamp_min(1)[..., None],
        torch.zeros_like(summed),
    )


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, out_dim),
    )


class SpectralMix(nn.Module):
    """Fourier mix on the fixed latent grid. Lowest modes only."""

    def __init__(self, hidden: int, n_modes: int):
        super().__init__()
        self.n_modes = n_modes
        self.weight = nn.Parameter(torch.randn(hidden, n_modes, 2) / math.sqrt(hidden))
        self.skip = nn.Linear(hidden, hidden)

    def forward(self, latent: Tensor) -> Tensor:
        n_tokens = latent.shape[1]
        spectrum = torch.fft.rfft(latent, dim=1)
        weight = torch.view_as_complex(self.weight.contiguous())
        n_keep = min(self.n_modes, spectrum.shape[1])
        mixed = torch.zeros_like(spectrum)
        mixed[:, :n_keep, :] = spectrum[:, :n_keep, :] * weight[:, :n_keep].permute(
            1, 0
        ).unsqueeze(0)
        return torch.fft.irfft(mixed, n=n_tokens, dim=1) + self.skip(latent)


class PathRound(nn.Module):
    """One round of messages to the layers above and below."""

    def __init__(self, hidden: int):
        super().__init__()
        self.down = _mlp(3 * hidden, hidden, hidden)
        self.up = _mlp(3 * hidden, hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, token: Tensor, mask: Tensor) -> Tensor:
        left = torch.zeros_like(token)
        left[:, 1:] = token[:, :-1]
        left_ok = torch.zeros_like(mask)
        left_ok[:, 1:] = mask[:, :-1]
        right = torch.zeros_like(token)
        right[:, :-1] = token[:, 1:]
        right_ok = torch.zeros_like(mask)
        right_ok[:, :-1] = mask[:, 1:]
        down = self.down(torch.cat([token, left, token - left], dim=-1))
        up = self.up(torch.cat([token, right, token - right], dim=-1))
        updated = (
            token
            + down * left_ok.unsqueeze(-1).to(token.dtype)
            + up * right_ok.unsqueeze(-1).to(token.dtype)
        )
        return self.norm(updated) * mask.unsqueeze(-1).to(token.dtype)


class LatentPathOperator(nn.Module):
    """Shared trunk for any linear column spec. Heads and boundary data differ."""

    def __init__(self, spec: ColumnSpec, damping: float = DAMPING):
        super().__init__()
        self.spec = spec
        self.damping = damping
        hidden = spec.hidden
        self.encoder = _mlp(spec.n_props + 2 + spec.n_bc, hidden, hidden)
        self.path = nn.ModuleList([PathRound(hidden) for _ in range(2)])
        self.score = nn.Linear(hidden, spec.n_latent)
        self.mix = SpectralMix(hidden, spec.n_modes)
        self.latent_norm = nn.LayerNorm(hidden)
        if spec.name == "sh":
            self.compliance = nn.Linear(hidden, 1)
            nn.init.zeros_(self.compliance.weight)
            # log of a mid-range compliance. exp keeps ê positive with a slope
            # that does not vanish between the stiff and soft soils.
            nn.init.constant_(self.compliance.bias, -5.3)
            self.head = None
            self.trunk = None
        else:
            self.compliance = None
            self.head = nn.Linear(
                hidden, spec.n_generators * (spec.state_dim**2 + spec.state_dim)
            )
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
            self.trunk = None

    def branch_tokens(
        self, props: Tensor, thickness: Tensor, bc: Tensor, mask: Tensor
    ) -> Tensor:
        bounds = column_bounds(thickness, mask)
        depth = (bounds[:, -1] - bounds[:, 0]).clamp_min(1e-6)
        mid = bounds[:, :-1] + 0.5 * thickness
        features = torch.cat(
            [
                torch.log(props.clamp_min(1e-8)),
                torch.log(thickness.clamp_min(1e-6)).unsqueeze(-1),
                (mid / depth.unsqueeze(-1)).unsqueeze(-1),
                torch.log(bc.clamp_min(1e-8))[:, None, :].expand(
                    -1, thickness.shape[1], -1
                ),
            ],
            dim=-1,
        )
        token = self.encoder(features) * mask.unsqueeze(-1).to(features.dtype)
        for block in self.path:
            token = block(token, mask)
        scores = self.score(token).masked_fill(~mask.unsqueeze(-1), -1e9)
        weights = torch.softmax(scores, dim=1)
        latent = torch.einsum("bnk,bnh->bkh", weights, token)
        delta = torch.einsum("bnk,bkh->bnh", weights, self.mix(latent))
        return self.latent_norm(token + delta) * mask.unsqueeze(-1).to(token.dtype)

    def elastic_compliance(self, token: Tensor, mask: Tensor) -> Tensor:
        """Positive ê(Vs), capped so the layer exponential stays finite."""
        raw = self.compliance(token).squeeze(-1)
        compliance = torch.exp(raw).clamp(max=COMPLIANCE_SCALE)
        return compliance * mask.to(compliance.dtype)

    def generators(self, token: Tensor) -> tuple[Tensor, Tensor]:
        batch, n_layers, _ = token.shape
        state_dim = self.spec.state_dim
        n_gen = self.spec.n_generators
        raw = self.head(token).view(
            batch, n_layers, n_gen, state_dim * state_dim + state_dim
        )
        system = raw[..., : state_dim * state_dim].reshape(
            batch, n_layers, n_gen, state_dim, state_dim
        )
        force = raw[..., state_dim * state_dim :]
        return system, force

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
        pooled = pool_sensors(
            sensor_z, sensor_props, sensor_mask, thickness, layer_mask
        )
        token = self.branch_tokens(pooled, thickness, bc, layer_mask)
        batch, _, n_query = query_z.shape[0], thickness.shape[1], query_z.shape[1]
        if self.spec.name == "sh":
            if freq is None:
                raise ValueError("SH columns need frequency collocation points")
            n_freq = freq.shape[1]
            compliance = self.elastic_compliance(token, layer_mask)
            system, force = rayleigh_generator(
                compliance, freq, a0=RAYLEIGH_A0, a1=RAYLEIGH_A1
            )
        else:
            coeff_a, coeff_g = self.generators(token)
            trunk = torch.ones(
                batch,
                1,
                self.spec.n_generators,
                device=thickness.device,
                dtype=thickness.dtype,
            )
            n_freq = 1
            system, force = physical_generators(
                *assemble_generators(coeff_a, coeff_g, trunk), self.spec.name
            )
        flat = batch * n_freq
        state_dim = self.spec.state_dim
        system = system.reshape(flat, -1, state_dim, state_dim)
        force = force.reshape(flat, -1, state_dim)
        thick = thickness[:, None, :].expand(batch, n_freq, -1).reshape(flat, -1)
        live = layer_mask[:, None, :].expand(batch, n_freq, -1).reshape(flat, -1)
        query = (
            query_z[:, None, :].expand(batch, n_freq, n_query).reshape(flat, n_query)
        )
        b_top, c_top, b_base, c_base = self._boundary(bc, freq, flat)
        state = propagate_and_query(
            system, force, thick, live, query, b_top, c_top, b_base, c_base, ridge=1e-6
        )
        state = state.reshape(batch, n_freq, n_query, state_dim)
        response = self._response(state)
        if not self.spec.use_frequency:
            return response[:, 0]
        return response

    def _boundary(self, bc: Tensor, freq: Tensor | None, flat: int):
        if self.spec.name == "sh":
            vs_bed = bc[:, 0][:, None].expand_as(freq).reshape(flat)
            freq_flat = freq.reshape(flat)
            return rayleigh_boundary(vs_bed, freq_flat, a0=RAYLEIGH_A0, a1=RAYLEIGH_A1)
        return static_boundary(bc[:, 0], SIGMA_REF)

    def _response(self, state: Tensor) -> Tensor:
        if self.spec.name == "static":
            return torch.cat([state[..., :1], state[..., 1:2] * SIGMA_REF], dim=-1)
        index = list(self.spec.response_index)
        return state[..., index]
