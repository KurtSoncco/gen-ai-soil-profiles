"""Modal DeepONet for the outcrop transfer function.

The branch reads the soil column and emits a few resonances plus a smooth
correction. Lower resonances are centered on the odd quarter-wave harmonics,
where a layered column actually peaks, and a bounded Fourier term fits the
broadband shape those poles cannot. The leading term is the unit bedrock
outcrop. A finite modal sum returns to that leading 1 at high frequency,
while the outcrop transfer function does not, so one extra depth-shaped
gain carries the high-frequency floor. The half-space velocity is not an input.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from experiments.latent_path_operator.propagator import column_bounds

N_RESONANCES = 8
N_SMOOTH = 32
XI_REF = 0.06


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, out_dim),
    )


class TransferDeepONet(nn.Module):
    """Column to ``u(z, f) / u_outcrop`` as resonances plus a smooth correction.

    The lower modes sit on the odd quarter-wave harmonics. Each has a damping
    ratio, a complex amplitude, and a depth shape that is 1 at the surface.
    A zero amplitude and a zero smooth head leave every depth at ``1 + 0j``.
    """

    def __init__(self, hidden: int = 128, n_modes: int = N_RESONANCES):
        super().__init__()
        self.n_modes = n_modes
        self.layer_mlp = _mlp(3, hidden, hidden)
        width = hidden + 1
        self.freq_head = nn.Linear(width, n_modes)
        self.damp_head = nn.Linear(width, n_modes)
        self.shape_head = nn.Linear(width, 4 * n_modes)
        self.amp_head = nn.Linear(width, 2 * n_modes)
        self.floor_head = nn.Linear(width, 2)
        self.corner_head = nn.Linear(width, 1)
        self.floor_shape = nn.Linear(width, 4)
        self.trunk = _mlp(32, hidden, N_SMOOTH)
        self.smooth_head = nn.Linear(width, 2 * N_SMOOTH)
        nn.init.zeros_(self.freq_head.weight)
        nn.init.zeros_(self.freq_head.bias)
        nn.init.zeros_(self.damp_head.weight)
        nn.init.zeros_(self.damp_head.bias)
        nn.init.zeros_(self.amp_head.weight)
        nn.init.zeros_(self.amp_head.bias)
        nn.init.zeros_(self.floor_head.weight)
        nn.init.zeros_(self.floor_head.bias)
        nn.init.zeros_(self.smooth_head.weight)
        nn.init.zeros_(self.smooth_head.bias)

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
        vs = sensor_props[..., 0]
        summary = self._summary(thickness, vs, layer_mask)
        zeta = self._zeta(thickness, layer_mask, query_z)
        live = layer_mask.to(thickness.dtype)
        travel = ((thickness * live) / vs.clamp_min(1.0)).sum(dim=-1)
        frequencies = self._frequencies(self.freq_head(summary), travel)
        damping = XI_REF * torch.exp(0.7 * torch.tanh(self.damp_head(summary)))
        amplitude = 2.0 * torch.tanh(self.amp_head(summary).view(-1, self.n_modes, 2))
        amp = torch.complex(amplitude[..., 0], amplitude[..., 1])
        modes = self._resonators(freq, frequencies, damping, amp)
        shapes = self._shapes(zeta, self.shape_head(summary)).to(modes.dtype)
        response = (modes[:, :, None, :] * shapes[:, None, :, :]).sum(dim=-1)
        transfer = (
            response
            + 1.0
            + self._floor(freq, zeta, summary)
            + self._smooth(summary, freq, travel, zeta)
        )
        return torch.stack((transfer.real, transfer.imag), dim=-1)

    def _summary(self, thickness: Tensor, vs: Tensor, mask: Tensor) -> Tensor:
        live = mask.to(thickness.dtype)
        height = (thickness * live).sum(dim=-1).clamp_min(1e-6)
        bounds = column_bounds(thickness, mask)
        mid = (bounds[:, :-1] + 0.5 * thickness) / height[:, None]
        fraction = (thickness * live) / height[:, None]
        features = torch.stack(
            [torch.log(vs.clamp_min(1e-8)), fraction, mid],
            dim=-1,
        )
        local = self.layer_mlp(features) * live.unsqueeze(-1)
        pooled = local.sum(dim=1)
        return torch.cat([pooled, torch.log(height).unsqueeze(-1)], dim=-1)

    def _zeta(self, thickness: Tensor, mask: Tensor, query_z: Tensor) -> Tensor:
        height = (thickness * mask.to(thickness.dtype)).sum(dim=-1).clamp_min(1e-6)
        return (query_z / height[:, None]).clamp(0.0, 1.0)

    def _frequencies(self, raw: Tensor, travel: Tensor) -> Tensor:
        """Odd harmonics for the lower modes, then log-spaced up through the band.

        At initialization each mode sits on that center, so its amplitude
        gradient lines up with a real peak. The slide stays inside the gap to
        the neighboring center.
        """
        fundamental = (1.0 / (4.0 * travel.clamp_min(1e-3))).clamp(0.08, 14.0)
        n_low = self.n_modes // 2
        odds = torch.arange(1, 2 * n_low, 2, device=raw.device, dtype=raw.dtype)
        low = fundamental[:, None] * odds
        high_start = low[:, -1] * 1.45
        high_end = torch.maximum(high_start * 1.15, fundamental.new_full((), 16.0))
        high_end = torch.maximum(high_end, high_start * 1.05).clamp(max=22.0)
        grid = torch.linspace(
            0.0, 1.0, self.n_modes - n_low, device=raw.device, dtype=raw.dtype
        )
        log_high = (
            torch.log(high_start)[:, None] * (1.0 - grid)
            + torch.log(high_end)[:, None] * grid
        )
        centers = torch.cat([low, log_high.exp()], dim=-1)
        log_c = torch.log(centers.clamp(0.05, 24.0))
        gap = log_c[:, 1:] - log_c[:, :-1]
        prev_gap = torch.cat([gap[:, :1], gap], dim=1)
        next_gap = torch.cat([gap, gap[:, -1:]], dim=1)
        half = 0.48 * torch.minimum(prev_gap, next_gap)
        return torch.exp(log_c + half * torch.tanh(raw))

    def _smooth(
        self, summary: Tensor, freq: Tensor, travel: Tensor, zeta: Tensor
    ) -> Tensor:
        """Bounded Fourier correction, shape (B, F, M) complex. Zero at initialization."""
        basis = self.trunk(self._query_features(freq, travel, zeta))
        coeff = self.smooth_head(summary).view(-1, 2, N_SMOOTH)
        real = (basis * coeff[:, None, None, 0, :]).sum(dim=-1)
        imag = (basis * coeff[:, None, None, 1, :]).sum(dim=-1)
        return torch.complex(6.0 * torch.tanh(real), 6.0 * torch.tanh(imag))

    def _query_features(self, freq: Tensor, travel: Tensor, zeta: Tensor) -> Tensor:
        harmonics = torch.arange(1, 9, device=freq.device, dtype=freq.dtype)
        cycles = freq * travel[:, None]
        ang_f = (2.0 * math.pi) * cycles.unsqueeze(-1) * harmonics
        ang_z = math.pi * zeta.unsqueeze(-1) * harmonics
        sin_f = ang_f.sin()[:, :, None, :].expand(-1, -1, zeta.shape[1], -1)
        cos_f = ang_f.cos()[:, :, None, :].expand(-1, -1, zeta.shape[1], -1)
        sin_z = ang_z.sin()[:, None, :, :].expand(-1, freq.shape[1], -1, -1)
        cos_z = ang_z.cos()[:, None, :, :].expand(-1, freq.shape[1], -1, -1)
        return torch.cat([sin_f, cos_f, sin_z, cos_z], dim=-1)

    def _resonators(
        self, freq: Tensor, f_mode: Tensor, damping: Tensor, amplitude: Tensor
    ) -> Tensor:
        """Complex modal gain, shape (B, F, K)."""
        ratio = freq.unsqueeze(-1) / f_mode.unsqueeze(1)
        complex_dtype = (
            torch.complex64 if freq.dtype == torch.float32 else torch.complex128
        )
        ratio_c = ratio.to(complex_dtype)
        damp_c = damping.to(complex_dtype)
        denom = 1.0 - ratio_c**2 + (2j * damp_c.unsqueeze(1)) * ratio_c
        return amplitude.unsqueeze(1) / denom

    def _floor(self, freq: Tensor, zeta: Tensor, summary: Tensor) -> Tensor:
        """High-frequency gain, 0 at low frequency, shape (B, F, M) complex."""
        corner = 6.0 * torch.exp(torch.tanh(self.corner_head(summary)))
        gain = freq**2 / (freq**2 + corner**2)
        raw = torch.tanh(self.floor_head(summary))
        amplitude = torch.complex(raw[:, 0], raw[:, 1])
        coeff = torch.tanh(self.floor_shape(summary).view(-1, 1, 2, 2))
        shape = self._depth_gain(zeta, coeff)[:, :, 0]
        return (
            gain[:, :, None].to(amplitude.dtype)
            * amplitude[:, None, None]
            * shape[:, None, :].to(amplitude.dtype)
        )

    def _shapes(self, zeta: Tensor, raw: Tensor) -> Tensor:
        """Depth gain, shape (B, M, K). One at the surface, two complex cosines below."""
        coeff = torch.tanh(raw.view(-1, self.n_modes, 2, 2))
        return self._depth_gain(zeta, coeff)

    def _depth_gain(self, zeta: Tensor, coeff: Tensor) -> Tensor:
        """``1`` at the surface. Complex cosines let the phase change with depth."""
        cos1 = torch.cos(math.pi * zeta) - 1.0
        cos2 = torch.cos(2.0 * math.pi * zeta) - 1.0
        first = torch.complex(coeff[..., 0, 0], coeff[..., 0, 1])
        second = torch.complex(coeff[..., 1, 0], coeff[..., 1, 1])
        return (
            1.0
            + first[:, None, :] * cos1[:, :, None].to(first.dtype)
            + second[:, None, :] * cos2[:, :, None].to(second.dtype)
        )
