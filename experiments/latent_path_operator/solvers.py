"""Analytic linear labels. Not used as a loss term.

Shear-wave labels follow the Thomson–Haskell recursion in
``site_metrics.transfer_function``: constant density, hysteretic damping,
last value is the half-space, surface motion over bedrock-outcrop motion.
"""

from __future__ import annotations

import numpy as np

from experiments.flow_matching_simplified.site_metrics import DAMPING
from experiments.latent_path_operator.propagator import RAYLEIGH_A0, RAYLEIGH_A1, RHO


def haskell_field(
    thickness: np.ndarray,
    vs: np.ndarray,
    vs_bedrock: float,
    freqs: np.ndarray,
    query_z: np.ndarray,
    damping: float = DAMPING,
) -> tuple[np.ndarray, np.ndarray]:
    """Complex displacement on a unit outcrop, and the complex surface transfer.

    Returns field (n_freq, n_query) and transfer (n_freq,). The surface value
    equals the transfer. Modulus of the transfer matches ``transfer_function``.
    """
    thickness = np.asarray(thickness, dtype=np.float64)
    vs = np.asarray(vs, dtype=np.float64)
    freqs = np.atleast_1d(np.asarray(freqs, dtype=np.float64))
    query_z = np.atleast_1d(np.asarray(query_z, dtype=np.float64))
    n_layers = len(thickness)
    bounds = np.concatenate([[0.0], np.cumsum(thickness)])
    depth = float(bounds[-1])
    vs_c = np.concatenate([vs, [vs_bedrock]]) * (1.0 + 1j * damping)
    field = np.zeros((len(freqs), len(query_z)), dtype=np.complex128)
    transfer = np.zeros(len(freqs), dtype=np.complex128)
    for i, freq in enumerate(freqs):
        omega = 2.0 * np.pi * float(freq)
        up = 1.0 + 0.0j
        down = 1.0 + 0.0j
        amplitudes = np.zeros((n_layers, 2), dtype=np.complex128)
        for layer in range(n_layers):
            amplitudes[layer, 0] = up
            amplitudes[layer, 1] = down
            alpha = vs_c[layer] / vs_c[layer + 1]
            phase = np.exp(1j * omega / vs_c[layer] * thickness[layer])
            up, down = (
                0.5 * up * (1.0 + alpha) * phase + 0.5 * down * (1.0 - alpha) / phase,
                0.5 * up * (1.0 - alpha) * phase + 0.5 * down * (1.0 + alpha) / phase,
            )
        scale = 2.0 * up
        transfer[i] = 1.0 / up
        for j, depth_q in enumerate(query_z):
            if depth_q >= depth:
                field[i, j] = (up + down) / scale
                continue
            layer = int(np.searchsorted(bounds, depth_q, side="right") - 1)
            layer = min(max(layer, 0), n_layers - 1)
            zeta = depth_q - bounds[layer]
            k = omega / vs_c[layer]
            amp_up, amp_down = amplitudes[layer]
            field[i, j] = (
                amp_up * np.exp(1j * k * zeta) + amp_down * np.exp(-1j * k * zeta)
            ) / scale
    return field, transfer


def _rayleigh_k(vs: np.ndarray, omega: complex, a0: float, a1: float) -> np.ndarray:
    modulus = 1.0 + 1j * omega * a1
    k2 = (omega**2 - 1j * omega * a0) / (vs**2 * modulus)
    k = np.sqrt(k2)
    return np.where(np.real(k) < 0, -k, k)


def rayleigh_field(
    thickness: np.ndarray,
    vs: np.ndarray,
    vs_bedrock: float,
    freqs: np.ndarray,
    query_z: np.ndarray,
    a0: float = RAYLEIGH_A0,
    a1: float = RAYLEIGH_A1,
) -> tuple[np.ndarray, np.ndarray]:
    """Complex displacement for unit outcrop under Rayleigh damping.

    Same up/down recursion as the hysteretic solver, with wavenumber and
    impedance from ``μ(1 + i ω a1)`` and ``-ρω² + i ω a0 ρ``.
    """
    thickness = np.asarray(thickness, dtype=np.float64)
    vs = np.asarray(vs, dtype=np.float64)
    freqs = np.atleast_1d(np.asarray(freqs, dtype=np.float64))
    query_z = np.atleast_1d(np.asarray(query_z, dtype=np.float64))
    n_layers = len(thickness)
    bounds = np.concatenate([[0.0], np.cumsum(thickness)])
    depth = float(bounds[-1])
    vs_all = np.concatenate([vs, [vs_bedrock]])
    omega = 2.0 * np.pi * freqs
    k = _rayleigh_k(vs_all[None, :], omega[:, None], a0, a1)
    impedance = RHO * vs_all[None, :] ** 2 * (1.0 + 1j * omega[:, None] * a1) * k
    up = np.ones(len(freqs), dtype=np.complex128)
    down = np.ones(len(freqs), dtype=np.complex128)
    amplitudes = np.zeros((len(freqs), n_layers, 2), dtype=np.complex128)
    for layer in range(n_layers):
        amplitudes[:, layer, 0] = up
        amplitudes[:, layer, 1] = down
        alpha = impedance[:, layer] / impedance[:, layer + 1]
        phase = np.exp(1j * k[:, layer] * thickness[layer])
        up, down = (
            0.5 * up * (1.0 + alpha) * phase + 0.5 * down * (1.0 - alpha) / phase,
            0.5 * up * (1.0 - alpha) * phase + 0.5 * down * (1.0 + alpha) / phase,
        )
    scale = 2.0 * up
    transfer = 1.0 / up
    field = np.zeros((len(freqs), len(query_z)), dtype=np.complex128)
    for j, depth_q in enumerate(query_z):
        if depth_q >= depth:
            field[:, j] = (up + down) / scale
            continue
        layer = int(np.searchsorted(bounds, depth_q, side="right") - 1)
        layer = min(max(layer, 0), n_layers - 1)
        zeta = depth_q - bounds[layer]
        amp_up = amplitudes[:, layer, 0]
        amp_down = amplitudes[:, layer, 1]
        field[:, j] = (
            amp_up * np.exp(1j * k[:, layer] * zeta)
            + amp_down * np.exp(-1j * k[:, layer] * zeta)
        ) / scale
    return field, transfer


def metrics_column(
    thickness: np.ndarray, vs: np.ndarray, vs_bedrock: float
) -> tuple[np.ndarray, np.ndarray]:
    """Bounds and Vs laid out the way ``transfer_function`` expects them."""
    edges = np.concatenate([[0.0], np.cumsum(np.asarray(thickness, dtype=np.float64))])
    bounds = np.concatenate([edges, [edges[-1] + 1.0]])
    velocities = np.concatenate([np.asarray(vs, dtype=np.float64), [vs_bedrock]])
    return bounds, velocities


def static_field(
    thickness: np.ndarray,
    gamma: np.ndarray,
    modulus: np.ndarray,
    load: float,
    query_z: np.ndarray,
) -> np.ndarray:
    """Displacement and vertical stress. σ(0) = load, u(H) = 0, dσ/dz = γ, du/dz = σ/M.

    With z downward and this sign, compressive stress makes u increase with depth,
    so the fixed base puts the surface value below zero.
    """
    thickness = np.asarray(thickness, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    modulus = np.asarray(modulus, dtype=np.float64)
    query_z = np.atleast_1d(np.asarray(query_z, dtype=np.float64))
    bounds = np.concatenate([[0.0], np.cumsum(thickness)])
    stress_top = np.zeros(len(thickness), dtype=np.float64)
    stress = float(load)
    for layer, (h, weight) in enumerate(zip(thickness, gamma)):
        stress_top[layer] = stress
        stress = stress + weight * h
    disp_top = np.zeros(len(thickness), dtype=np.float64)
    disp_below = 0.0
    for layer in range(len(thickness) - 1, -1, -1):
        h = thickness[layer]
        integral = (stress_top[layer] * h + 0.5 * gamma[layer] * h**2) / modulus[layer]
        disp_top[layer] = disp_below - integral
        disp_below = disp_top[layer]
    out = np.zeros((len(query_z), 2), dtype=np.float64)
    depth = float(bounds[-1])
    for j, depth_q in enumerate(query_z):
        if depth_q >= depth:
            layer = len(thickness) - 1
            zeta = thickness[layer]
        else:
            layer = int(np.searchsorted(bounds, depth_q, side="right") - 1)
            layer = min(max(layer, 0), len(thickness) - 1)
            zeta = depth_q - bounds[layer]
        stress_q = stress_top[layer] + gamma[layer] * zeta
        out[j, 0] = (
            disp_top[layer]
            + (stress_top[layer] * zeta + 0.5 * gamma[layer] * zeta**2) / modulus[layer]
        )
        out[j, 1] = stress_q
    return out
