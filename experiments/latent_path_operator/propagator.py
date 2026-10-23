"""Affine layer propagator: expm of h * [A, g], then a linear boundary solve.

Stress is scaled so the generator stays O(wavenumber) and ``matrix_exp`` is
stable. Displacement is unscaled. Density cancels for constant-density SH.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

RHO = 2000.0
VS_REF = 500.0
SIGMA_REF = 1.0e5
RAYLEIGH_XI = 0.02
RAYLEIGH_F1 = 1.0
RAYLEIGH_F2 = 10.0


def rayleigh_coefficients(
    xi: float = RAYLEIGH_XI,
    f1: float = RAYLEIGH_F1,
    f2: float = RAYLEIGH_F2,
) -> tuple[float, float]:
    """Mass and stiffness coefficients that match ``xi`` at ``f1`` and ``f2``."""
    omega_1 = 2.0 * math.pi * f1
    omega_2 = 2.0 * math.pi * f2
    a0 = 2.0 * xi * omega_1 * omega_2 / (omega_1 + omega_2)
    a1 = 2.0 * xi / (omega_1 + omega_2)
    return a0, a1


RAYLEIGH_A0, RAYLEIGH_A1 = rayleigh_coefficients()


def column_bounds(thickness: Tensor, mask: Tensor) -> Tensor:
    """Depth of each interface. Padding layers add no depth."""
    h = thickness * mask.to(thickness.dtype)
    zeros = torch.zeros(*h.shape[:-1], 1, device=h.device, dtype=h.dtype)
    return torch.cumsum(torch.cat([zeros, h], dim=-1), dim=-1)


def _softplus_shift(raw: Tensor) -> Tensor:
    """softplus(raw) - softplus(0), so a zero head stays a zero generator."""
    return torch.nn.functional.softplus(raw) - torch.nn.functional.softplus(
        raw.new_zeros(())
    )


def physical_generators(
    system: Tensor, force: Tensor, kind: str
) -> tuple[Tensor, Tensor]:
    """Keep A and g in the family of the linear soil model.

    Shear waves use a complex modulus and a real inertia term. Static compression
    uses a positive compliance and a positive unit weight. Entries outside that
    pattern are dropped so the matrix exponential stays a layer transfer map.
    """
    if kind == "sh":
        # a ~ ω Vs_ref / Vs^2 is a few units; c ~ -ω / Vs_ref is smaller.
        a = _softplus_shift(system[..., 0, 2].clamp(-2.0, 6.0))
        b = system[..., 0, 3].clamp(-2.0, 2.0).tanh()
        c = -_softplus_shift(system[..., 2, 0].clamp(-2.0, 2.0))
        zero = torch.zeros_like(a)
        row0 = torch.stack([zero, zero, a, b], dim=-1)
        row1 = torch.stack([zero, zero, -b, a], dim=-1)
        row2 = torch.stack([c, zero, zero, zero], dim=-1)
        row3 = torch.stack([zero, c, zero, zero], dim=-1)
        realized = torch.stack([row0, row1, row2, row3], dim=-2)
        return realized, torch.zeros_like(force)
    if kind == "static":
        compliance = _softplus_shift(system[..., 0, 1].clamp(-4.0, 4.0))
        weight = _softplus_shift(force[..., 1].clamp(-4.0, 4.0))
        zero = torch.zeros_like(compliance)
        row0 = torch.stack([zero, compliance], dim=-1)
        row1 = torch.stack([zero, zero], dim=-1)
        realized = torch.stack([row0, row1], dim=-2)
        return realized, torch.stack([zero, weight], dim=-1)
    raise ValueError(f"unknown generator family {kind!r}")


def assemble_generators(
    coeff_a: Tensor, coeff_g: Tensor, trunk: Tensor
) -> tuple[Tensor, Tensor]:
    """DeepONet product A(f) = sum_k t_k(f) A_k, and the same for g.

    coeff_a: (B, N, P, d, d), coeff_g: (B, N, P, d), trunk: (B, F, P).
    """
    system = torch.einsum("bfp,bnpij->bfnij", trunk, coeff_a)
    force = torch.einsum("bfp,bnpi->bfni", trunk, coeff_g)
    return system, force


def augmented_expm(system: Tensor, force: Tensor, step: Tensor) -> Tensor:
    """expm(step * [[A, g], [0, 0]]) with shapes (..., d, d), (..., d), (...)."""
    upper = torch.cat([system, force.unsqueeze(-1)], dim=-1)
    lower = system.new_zeros(*system.shape[:-2], 1, system.shape[-1] + 1)
    generator = torch.cat([upper, lower], dim=-2)
    return torch.matrix_exp(generator * step[..., None, None])


def _apply(transform: Tensor, state: Tensor) -> Tensor:
    d = state.shape[-1]
    aug = torch.cat([state, torch.ones_like(state[..., :1])], dim=-1)
    out = torch.matmul(transform, aug.unsqueeze(-1)).squeeze(-1)
    return out[..., :d]


def propagate_and_query(
    system: Tensor,
    force: Tensor,
    thickness: Tensor,
    mask: Tensor,
    query_z: Tensor,
    b_top: Tensor,
    c_top: Tensor,
    b_base: Tensor,
    c_base: Tensor,
    ridge: float = 0.0,
) -> Tensor:
    """State at collocation depths.

    system (B, N, d, d), force (B, N, d), query_z (B, M).
    Returns (B, M, d). Masked layers are identity maps.
    """
    batch, n_layers, state_dim, _ = system.shape
    live = mask.to(system.dtype)
    system = system * live[:, :, None, None]
    force = force * live[:, :, None]
    step = augmented_expm(system, force, thickness * live)

    eye = torch.eye(state_dim + 1, device=system.device, dtype=system.dtype).expand(
        batch, -1, -1
    )
    maps = [eye]
    for i in range(n_layers):
        maps.append(torch.matmul(step[:, i], maps[-1]))
    prefix = torch.stack(maps, dim=1)

    total = prefix[:, -1]
    phi = total[:, :state_dim, :state_dim]
    psi = total[:, :state_dim, state_dim]
    base_map = torch.matmul(b_base, phi)
    stacked = torch.cat([b_top, base_map], dim=1)
    if ridge:
        stacked = stacked + ridge * torch.eye(
            state_dim, device=system.device, dtype=system.dtype
        )
    rhs = torch.cat(
        [c_top, c_base - torch.matmul(b_base, psi.unsqueeze(-1)).squeeze(-1)], dim=-1
    )
    state0 = torch.linalg.solve(stacked, rhs)

    bounds = column_bounds(thickness, mask)
    n_real = mask.to(torch.long).sum(dim=-1)
    # bounds[k] <= z. z = H hits the bedrock interface, which is the base of the last real layer.
    index = (bounds[:, None, :] <= query_z[:, :, None]).sum(dim=-1) - 1
    index = torch.minimum(index, n_real[:, None] - 1).clamp(min=0)

    n_query = query_z.shape[1]
    gather_i = index[:, :, None, None].expand(
        batch, n_query, state_dim + 1, state_dim + 1
    )
    top_map = torch.gather(prefix, 1, gather_i)
    state_top = _apply(top_map, state0[:, None, :].expand(batch, n_query, state_dim))

    a_index = index[:, :, None, None].expand(batch, n_query, state_dim, state_dim)
    g_index = index[:, :, None].expand(batch, n_query, state_dim)
    h_index = index
    system_q = torch.gather(system, 1, a_index)
    force_q = torch.gather(force, 1, g_index)
    h_q = torch.gather(thickness * mask.to(thickness.dtype), 1, h_index)
    z_top = torch.gather(bounds[:, :-1], 1, h_index)
    xi = torch.where(
        h_q > 0, ((query_z - z_top) / h_q).clamp(0, 1), torch.zeros_like(query_z)
    )
    partial = augmented_expm(system_q, force_q, xi * h_q)
    return _apply(partial, state_top)


def sh_generator(
    vs: Tensor, freq: Tensor, damping: float, vs_ref: float = VS_REF, rho: float = RHO
) -> tuple[Tensor, Tensor]:
    """Real 4x4 generator for hysteretic SH. State is [Re u, Im u, Re s, Im s], s = τ / (ρ Vs_ref ω)."""
    omega = freq * (2.0 * math.pi)
    vs_i = damping * vs
    mu_r = rho * (vs**2 - vs_i**2)
    mu_i = rho * (2.0 * vs * vs_i)
    beta = rho * vs_ref * omega
    scale = beta / (mu_r**2 + mu_i**2)
    system = vs.new_zeros(*vs.shape, 4, 4)
    system[..., 0, 2] = scale * mu_r
    system[..., 0, 3] = scale * mu_i
    system[..., 1, 2] = -scale * mu_i
    system[..., 1, 3] = scale * mu_r
    system[..., 2, 0] = -omega / vs_ref
    system[..., 3, 1] = -omega / vs_ref
    force = vs.new_zeros(*vs.shape, 4)
    return system, force


def rayleigh_generator(
    compliance: Tensor,
    freq: Tensor,
    a0: float = RAYLEIGH_A0,
    a1: float = RAYLEIGH_A1,
    vs_ref: float = VS_REF,
) -> tuple[Tensor, Tensor]:
    """Shear generator from elastic compliance ``ê ≈ Vs_ref / Vs²``.

    Compliance is ``(ω ê) / (1 + i ω a1)`` and inertia is ``(-ω + i a0) / Vs_ref``.
    ``compliance`` is (B, N) and ``freq`` is (B, F). Returns A, g with A shaped
    (B, F, N, 4, 4).
    """
    omega = freq * (2.0 * math.pi)
    elastic = compliance[:, None, :]
    angular = omega[:, :, None]
    gamma = (angular * elastic) / (1.0 + (angular * a1) ** 2)
    viscous = angular * a1
    system = compliance.new_zeros(
        freq.shape[0], freq.shape[1], compliance.shape[1], 4, 4
    )
    system[..., 0, 2] = gamma
    system[..., 0, 3] = gamma * viscous
    system[..., 1, 2] = -gamma * viscous
    system[..., 1, 3] = gamma
    system[..., 2, 0] = -angular / vs_ref
    system[..., 2, 1] = -a0 / vs_ref
    system[..., 3, 0] = a0 / vs_ref
    system[..., 3, 1] = -angular / vs_ref
    force = compliance.new_zeros(freq.shape[0], freq.shape[1], compliance.shape[1], 4)
    return system, force


def rayleigh_boundary(
    vs_bedrock: Tensor,
    freq: Tensor,
    a0: float = RAYLEIGH_A0,
    a1: float = RAYLEIGH_A1,
    vs_ref: float = VS_REF,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Free surface and unit outcrop radiation for a Rayleigh half-space."""
    omega = freq * (2.0 * math.pi)
    complex_dtype = torch.complex128 if freq.dtype == torch.float64 else torch.complex64
    angular = omega.to(complex_dtype)
    vs_c = vs_bedrock.to(complex_dtype)
    modulus = 1.0 + 1j * angular * a1
    k2 = (angular**2 - 1j * angular * a0) / (vs_c**2 * modulus)
    k = torch.sqrt(k2)
    k = torch.where(k.real < 0, -k, k)
    gain = 1j * (vs_c**2 / vs_ref) * modulus * k / angular
    gr = gain.real.to(freq.dtype)
    gi = gain.imag.to(freq.dtype)
    batch = freq.shape[0]
    b_top = freq.new_zeros(batch, 2, 4)
    b_top[:, 0, 2] = 1.0
    b_top[:, 1, 3] = 1.0
    c_top = freq.new_zeros(batch, 2)
    b_base = freq.new_zeros(batch, 2, 4)
    b_base[:, 0, 0] = gr
    b_base[:, 0, 1] = -gi
    b_base[:, 0, 2] = 1.0
    b_base[:, 1, 0] = gi
    b_base[:, 1, 1] = gr
    b_base[:, 1, 3] = 1.0
    c_base = torch.stack([gr, gi], dim=-1)
    return b_top, c_top, b_base, c_base


def sh_boundary(
    vs_bedrock: Tensor,
    freq: Tensor,
    damping: float,
    vs_ref: float = VS_REF,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Free surface and unit bedrock-outcrop radiation.

    ``freq`` is part of the call so each collocation frequency has its own
    constraint tensor. After stress is scaled by ω it cancels, so the matrix
    depends on the bedrock velocity and the damping only.
    """
    del freq
    gr = -damping * vs_bedrock / vs_ref
    gi = vs_bedrock / vs_ref
    batch = vs_bedrock.shape[0]
    b_top = vs_bedrock.new_zeros(batch, 2, 4)
    b_top[:, 0, 2] = 1.0
    b_top[:, 1, 3] = 1.0
    c_top = vs_bedrock.new_zeros(batch, 2)
    b_base = vs_bedrock.new_zeros(batch, 2, 4)
    b_base[:, 0, 0] = gr
    b_base[:, 0, 1] = -gi
    b_base[:, 0, 2] = 1.0
    b_base[:, 1, 0] = gi
    b_base[:, 1, 1] = gr
    b_base[:, 1, 3] = 1.0
    c_base = torch.stack([gr, gi], dim=-1)
    return b_top, c_top, b_base, c_base


def static_generator(
    gamma: Tensor, modulus: Tensor, sigma_ref: float = SIGMA_REF
) -> tuple[Tensor, Tensor]:
    """State [u, σ/σ_ref]. du/dz = σ/M, dσ/dz = γ."""
    system = gamma.new_zeros(*gamma.shape, 2, 2)
    system[..., 0, 1] = sigma_ref / modulus
    force = gamma.new_zeros(*gamma.shape, 2)
    force[..., 1] = gamma / sigma_ref
    return system, force


def static_boundary(
    load: Tensor, sigma_ref: float = SIGMA_REF
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """σ(0) = q and u(H) = 0, in the scaled stress coordinate."""
    batch = load.shape[0]
    b_top = load.new_zeros(batch, 1, 2)
    b_top[:, 0, 1] = 1.0
    c_top = (load / sigma_ref).unsqueeze(-1)
    b_base = load.new_zeros(batch, 1, 2)
    b_base[:, 0, 0] = 1.0
    c_base = load.new_zeros(batch, 1)
    return b_top, c_top, b_base, c_base
