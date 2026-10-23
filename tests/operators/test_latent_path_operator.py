"""Exact linear maps, transfer-function labels, and a short SH overfit."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from experiments.flow_matching_simplified.eval_corrected import travel_time_from_vs
from experiments.flow_matching_simplified.site_metrics import DAMPING, transfer_function
from experiments.latent_path_operator.data import (
    BEDROCK_VS,
    TF_FREQS,
    sample_sh_batch,
    sample_static_batch,
    sample_tf_batch,
)
from experiments.latent_path_operator.generated import batch_from_columns, soil_layers
from experiments.latent_path_operator.operators import (
    LaplaceOperator,
    RayleighChain,
    WaveletOperator,
)
from experiments.latent_path_operator.deeponet import TransferDeepONet
from experiments.latent_path_operator.model import LatentPathOperator, pool_sensors
from experiments.latent_path_operator.propagator import (
    RAYLEIGH_A0,
    RAYLEIGH_A1,
    SIGMA_REF,
    VS_REF,
    column_bounds,
    propagate_and_query,
    rayleigh_boundary,
    rayleigh_coefficients,
    rayleigh_generator,
    sh_boundary,
    sh_generator,
    static_boundary,
    static_generator,
)
from experiments.latent_path_operator.ricker import (
    DT,
    N_TIME,
    positive_frequencies,
    rfft_coefficients,
    ricker,
    ricker_spectrum,
    time_axis,
    velocity_traces,
)
from experiments.latent_path_operator.solvers import (
    haskell_field,
    metrics_column,
    rayleigh_field,
    static_field,
)
from experiments.latent_path_operator.spec import SH_SPEC, STATIC_SPEC
from experiments.latent_path_operator.train import (
    fit_batch,
    predict,
    relative_l1,
    relative_l1_by_frequency,
    relative_l2,
)


def _complex(state: torch.Tensor) -> np.ndarray:
    return (state[..., 0] + 1j * state[..., 1]).detach().cpu().numpy()


def test_undamped_layer_and_static_match_closed_forms():
    vs, vs_b, height, freq = 400.0, 1200.0, 20.0, 4.0
    system, force = sh_generator(
        torch.tensor([[vs]], dtype=torch.float64),
        torch.tensor(freq, dtype=torch.float64),
        damping=0.0,
    )
    b_top, c_top, b_base, c_base = sh_boundary(
        torch.tensor([vs_b], dtype=torch.float64),
        torch.tensor([freq], dtype=torch.float64),
        damping=0.0,
    )
    query = torch.tensor([[0.0, 0.5 * height, height]], dtype=torch.float64)
    state = propagate_and_query(
        system,
        force,
        torch.tensor([[height]], dtype=torch.float64),
        torch.ones(1, 1, dtype=torch.bool),
        query,
        b_top,
        c_top,
        b_base,
        c_base,
    )
    k = 2.0 * math.pi * freq / vs
    alpha = vs / vs_b
    denom = math.cos(k * height) + 1j * alpha * math.sin(k * height)
    got = _complex(state)[0]
    for zeta, value in zip((0.0, 0.5 * height, height), got):
        expected = math.cos(k * zeta) / denom
        assert value == pytest.approx(expected)
    # Surface over incident amplitude. Outcrop motion is twice the incident amplitude.
    assert 2.0 * got[0] == pytest.approx(2.0 / denom)

    thickness = np.array([2.0, 4.0])
    gamma = np.array([20.0e3, 16.0e3])
    modulus = np.array([50.0e6, 80.0e6])
    load = 100.0e3
    query_np = np.array([0.0, 1.0, 2.0, 5.0, 6.0])
    expected_static = static_field(thickness, gamma, modulus, load, query_np)
    system_s, force_s = static_generator(
        torch.tensor(gamma, dtype=torch.float64)[None, :],
        torch.tensor(modulus, dtype=torch.float64)[None, :],
    )
    b0, c0, bh, ch = static_boundary(torch.tensor([load], dtype=torch.float64))
    state_s = propagate_and_query(
        system_s,
        force_s,
        torch.tensor(thickness, dtype=torch.float64)[None, :],
        torch.ones(1, 2, dtype=torch.bool),
        torch.tensor(query_np, dtype=torch.float64)[None, :],
        b0,
        c0,
        bh,
        ch,
    )
    got_u = state_s[0, :, 0].detach().cpu().numpy()
    got_s = state_s[0, :, 1].detach().cpu().numpy() * SIGMA_REF
    assert np.allclose(got_u, expected_static[:, 0], rtol=1e-8, atol=1e-10)
    assert np.allclose(got_s, expected_static[:, 1], rtol=1e-8, atol=1e-6)


def test_propagator_matches_damped_haskell_between_interfaces():
    thickness = np.array([5.0, 8.0, 3.0])
    vs = np.array([220.0, 450.0, 700.0])
    vs_b = 1400.0
    freq = 3.5
    query = np.array([0.0, 2.5, 5.0, 9.0, 16.0])
    field, _ = haskell_field(
        thickness, vs, vs_b, np.array([freq]), query, damping=DAMPING
    )
    system, force = sh_generator(
        torch.tensor(vs, dtype=torch.float64)[None, :],
        torch.tensor(freq, dtype=torch.float64),
        damping=DAMPING,
    )
    b_top, c_top, b_base, c_base = sh_boundary(
        torch.tensor([vs_b], dtype=torch.float64),
        torch.tensor([freq], dtype=torch.float64),
        damping=DAMPING,
    )
    state = propagate_and_query(
        system,
        force,
        torch.tensor(thickness, dtype=torch.float64)[None, :],
        torch.ones(1, 3, dtype=torch.bool),
        torch.tensor(query, dtype=torch.float64)[None, :],
        b_top,
        c_top,
        b_base,
        c_base,
    )
    assert np.allclose(_complex(state)[0], field[0], rtol=1e-6, atol=1e-8)


def test_masked_layers_do_not_change_the_field():
    vs, vs_b, height, freq = 400.0, 1200.0, 20.0, 4.0
    system, force = sh_generator(
        torch.tensor([[vs]], dtype=torch.float64),
        torch.tensor(freq, dtype=torch.float64),
        damping=0.0,
    )
    query = torch.tensor([[0.0, 7.0, height]], dtype=torch.float64)
    b_top, c_top, b_base, c_base = sh_boundary(
        torch.tensor([vs_b], dtype=torch.float64),
        torch.tensor([freq], dtype=torch.float64),
        damping=0.0,
    )
    kwargs = dict(query_z=query, b_top=b_top, c_top=c_top, b_base=b_base, c_base=c_base)
    clean = propagate_and_query(
        system,
        force,
        torch.tensor([[height]], dtype=torch.float64),
        torch.ones(1, 1, dtype=torch.bool),
        **kwargs,
    )
    junk = torch.randn(1, 1, 4, 4, dtype=torch.float64)
    padded_system = torch.cat([system, junk], dim=1)
    padded_force = torch.cat([force, torch.randn(1, 1, 4, dtype=torch.float64)], dim=1)
    padded = propagate_and_query(
        padded_system,
        padded_force,
        torch.tensor([[height, 9.0]], dtype=torch.float64),
        torch.tensor([[True, False]]),
        **kwargs,
    )
    assert torch.allclose(clean, padded, atol=1e-8, rtol=1e-8)


def test_transfer_modulus_matches_site_metrics():
    rng = np.random.default_rng(1)
    for _ in range(4):
        n_soil = int(rng.integers(1, 6))
        thickness = rng.uniform(2.0, 15.0, size=n_soil)
        vs = np.exp(rng.uniform(np.log(180.0), np.log(900.0), size=n_soil))
        vs_b = float(np.exp(rng.uniform(np.log(1000.0), np.log(1800.0))))
        freqs = np.exp(rng.uniform(np.log(0.4), np.log(12.0), size=5))
        depth = float(thickness.sum())
        query = np.array([0.0, 0.3 * depth, depth])
        field, transfer = haskell_field(thickness, vs, vs_b, freqs, query)
        bounds, velocities = metrics_column(thickness, vs, vs_b)
        amplitude = transfer_function(bounds, velocities, freqs, DAMPING)
        assert np.allclose(np.abs(transfer), amplitude, rtol=1e-10, atol=1e-10)
        assert np.allclose(field[:, 0], transfer, rtol=1e-10, atol=1e-10)


def test_depth_increases_and_extra_sensors_pool_to_the_same_layers():
    thickness = torch.tensor([[1.0, 2.0, 3.5]])
    bounds = column_bounds(thickness, torch.ones(1, 3, dtype=torch.bool))
    assert torch.all(bounds[:, 1:] > bounds[:, :-1])

    layers = torch.tensor([[4.0, 6.0]])
    mask = torch.tensor([[True, True]])
    coarse_z = torch.tensor([[2.0, 7.0]])
    coarse_p = torch.tensor([[[100.0], [300.0]]])
    fine_z = torch.tensor([[1.0, 3.0, 5.0, 9.0]])
    fine_p = torch.tensor([[[100.0], [100.0], [300.0], [300.0]]])
    coarse = pool_sensors(
        coarse_z, coarse_p, torch.ones(1, 2, dtype=torch.bool), layers, mask
    )
    fine = pool_sensors(
        fine_z, fine_p, torch.ones(1, 4, dtype=torch.bool), layers, mask
    )
    assert torch.allclose(coarse, fine)
    assert torch.allclose(coarse, coarse_p)


def test_both_specs_return_finite_collocation_outputs():
    sh = sample_sh_batch(2, n_freq=3, n_query=5, rng=np.random.default_rng(2))
    sh_out = predict(LatentPathOperator(SH_SPEC), sh)
    assert sh_out.shape == (2, 3, 5, 2)
    assert torch.isfinite(sh_out).all()
    assert torch.isfinite(sh_out[:, :, 0, :]).all()
    assert torch.all(sh.query_z[:, 0] == 0)
    depth = (sh.thickness * sh.layer_mask.to(sh.thickness.dtype)).sum(dim=-1)
    assert torch.allclose(sh.query_z[:, 1], depth)

    static = sample_static_batch(2, n_query=5, rng=np.random.default_rng(3))
    static_out = predict(LatentPathOperator(STATIC_SPEC), static)
    assert static_out.shape == (2, 5, 2)
    assert torch.isfinite(static_out).all()


def test_criterion_is_relative_l2_only():
    target = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    pred = torch.zeros_like(target)
    assert relative_l2(target, target).item() == 0.0
    assert relative_l2(pred, target).item() == pytest.approx(1.0)
    assert relative_l1(target, target).item() == 0.0
    assert relative_l1(pred, target).item() == pytest.approx(1.0)
    import experiments.latent_path_operator.train as train_mod

    assert not hasattr(train_mod, "physics_loss")


def test_sh_overfit_decreases_loss():
    torch.manual_seed(0)
    batch = sample_sh_batch(16, n_freq=2, n_query=4, rng=np.random.default_rng(4))
    losses = fit_batch(LatentPathOperator(SH_SPEC), batch, steps=48, lr=3e-3)
    assert math.isfinite(losses[-1])
    assert losses[-1] < losses[0]


def test_ricker_spectrum_roundtrip():
    f_peak = 5.0
    t0 = 1.0 / f_peak
    time = time_axis()
    wave = ricker(time, f_peak, t0)
    freq = np.fft.rfftfreq(N_TIME, DT)
    recon = np.fft.irfft(ricker_spectrum(freq, f_peak, t0) / DT, n=N_TIME)
    assert np.allclose(recon, wave, atol=5e-4)
    assert int(np.argmax(recon)) == int(np.argmax(wave))


def test_rayleigh_generator_writes_mass_and_stiffness():
    vs = 400.0
    freq = 4.0
    compliance = torch.tensor([[VS_REF / vs**2]], dtype=torch.float64)
    freq_t = torch.tensor([[freq]], dtype=torch.float64)
    omega = 2.0 * math.pi * freq
    bare, _ = rayleigh_generator(compliance, freq_t, a0=0.0, a1=0.0)
    assert bare[0, 0, 0, 2, 0].item() == pytest.approx(-omega / VS_REF)
    assert bare[0, 0, 0, 3, 1].item() == pytest.approx(-omega / VS_REF)
    assert bare[0, 0, 0, 2, 1].item() == pytest.approx(0.0)
    assert bare[0, 0, 0, 0, 2].item() == pytest.approx(omega * VS_REF / vs**2)

    a0, a1 = RAYLEIGH_A0, RAYLEIGH_A1
    damped, force = rayleigh_generator(compliance, freq_t, a0=a0, a1=a1)
    gamma = (omega * VS_REF / vs**2) / (1.0 + (omega * a1) ** 2)
    assert damped[0, 0, 0, 2, 0].item() == pytest.approx(-omega / VS_REF)
    assert damped[0, 0, 0, 2, 1].item() == pytest.approx(-a0 / VS_REF)
    assert damped[0, 0, 0, 3, 0].item() == pytest.approx(a0 / VS_REF)
    assert damped[0, 0, 0, 0, 2].item() == pytest.approx(gamma)
    assert damped[0, 0, 0, 0, 3].item() == pytest.approx(gamma * omega * a1)
    assert damped[0, 0, 0, 1, 2].item() == pytest.approx(-gamma * omega * a1)
    assert torch.count_nonzero(force) == 0


def test_rayleigh_solver_matches_true_compliance():
    thickness = np.array([5.0, 8.0, 3.0])
    vs = np.array([220.0, 450.0, 700.0])
    vs_b = 1400.0
    freqs = np.array([1.5, 5.0, 12.0])
    query = np.array([0.0, 4.0, 8.0, 16.0])
    field, _ = rayleigh_field(thickness, vs, vs_b, freqs, query)
    system, force = rayleigh_generator(
        torch.tensor(VS_REF / vs**2, dtype=torch.float64)[None, :],
        torch.tensor(freqs, dtype=torch.float64)[None, :],
    )
    n_freq = len(freqs)
    b_top, c_top, b_base, c_base = rayleigh_boundary(
        torch.full((n_freq,), vs_b, dtype=torch.float64),
        torch.tensor(freqs, dtype=torch.float64),
    )
    state = propagate_and_query(
        system[0],
        force[0],
        torch.tensor(thickness, dtype=torch.float64)[None, :].expand(n_freq, -1),
        torch.ones(n_freq, len(thickness), dtype=torch.bool),
        torch.tensor(query, dtype=torch.float64)[None, :].expand(n_freq, -1),
        b_top,
        c_top,
        b_base,
        c_base,
    )
    assert np.allclose(_complex(state), field, rtol=1e-5, atol=1e-6)


def test_uniform_layer_ricker_is_delayed_and_damped():
    f_peak = 5.0
    source = ricker(time_axis(), f_peak)

    def surface(vs: float, a0: float, a1: float) -> np.ndarray:
        freqs = positive_frequencies()
        field, _ = rayleigh_field(
            np.array([20.0]),
            np.array([vs]),
            1500.0,
            freqs,
            np.array([0.0]),
            a0,
            a1,
        )
        return velocity_traces(field, rfft_coefficients(freqs, f_peak, DT))[0]

    def lag(trace: np.ndarray) -> int:
        corr = np.correlate(trace, source, mode="full")
        return int(np.argmax(corr) - (len(source) - 1))

    a0, a1 = rayleigh_coefficients()
    assert lag(surface(200.0, a0, a1)) > lag(surface(800.0, a0, a1))
    damped = surface(400.0, a0, a1)
    bare = surface(400.0, 0.0, 0.0)
    assert np.max(np.abs(damped)) < np.max(np.abs(bare))


def test_outcrop_transfer_changes_at_the_surface():
    freqs = np.logspace(np.log10(0.5), np.log10(20.0), 16)
    soft, _ = rayleigh_field(
        np.array([12.0, 18.0]),
        np.array([180.0, 400.0]),
        BEDROCK_VS,
        freqs,
        np.array([0.0, 15.0]),
    )
    stiff, _ = rayleigh_field(
        np.array([12.0, 18.0]),
        np.array([600.0, 400.0]),
        BEDROCK_VS,
        freqs,
        np.array([0.0, 15.0]),
    )
    assert not np.allclose(soft[:, 0], stiff[:, 0])
    assert np.max(np.abs(soft[:, 0])) < 20.0


def test_deeponet_starts_at_the_unit_outcrop():
    torch.manual_seed(0)
    batch = sample_tf_batch(4, n_query=4, rng=np.random.default_rng(5))
    surface = predict(TransferDeepONet(), batch)
    assert surface.shape[:3] == batch.target.shape[:3]
    assert torch.allclose(surface[..., 0], torch.ones_like(surface[..., 0]), atol=1e-5)
    assert torch.allclose(surface[..., 1], torch.zeros_like(surface[..., 1]), atol=1e-5)


def test_tf_overfit_decreases_loss():
    torch.manual_seed(0)
    batch = sample_tf_batch(8, n_query=4, rng=np.random.default_rng(6))
    losses = fit_batch(
        TransferDeepONet(), batch, steps=40, lr=1e-3, loss=relative_l1_by_frequency
    )
    assert math.isfinite(losses[-1])
    assert losses[-1] < losses[0]


def test_uniform_layer_peaks_follow_the_quarter_wavelength():
    vs, height = 400.0, 20.0
    fundamental = vs / (4.0 * height)
    freqs = np.linspace(0.5, 20.0, 600)
    field, _ = rayleigh_field(
        np.array([height]),
        np.array([vs]),
        BEDROCK_VS,
        freqs,
        np.array([0.0]),
    )
    amp = np.abs(field[:, 0])
    peaks = [
        index
        for index in range(1, len(amp) - 1)
        if amp[index] >= amp[index - 1]
        and amp[index] > amp[index + 1]
        and amp[index] > 1.5
    ]
    assert len(peaks) >= 2
    first, second = freqs[peaks[0]], freqs[peaks[1]]
    assert first == pytest.approx(fundamental, rel=0.25)
    assert second > first * 1.5


def test_one_column_modal_fit_beats_the_smooth_trunk():
    torch.manual_seed(0)
    batch = sample_tf_batch(1, n_query=6, rng=np.random.default_rng(8))
    losses = fit_batch(
        TransferDeepONet(), batch, steps=200, lr=1e-3, loss=relative_l1_by_frequency
    )
    assert math.isfinite(losses[-1])
    assert losses[-1] < losses[0]
    assert losses[-1] < 0.45


def test_transfer_grid_puts_most_points_above_1hz():
    assert len(TF_FREQS) == 1000
    assert TF_FREQS[0] == pytest.approx(0.01)
    assert TF_FREQS[-1] == pytest.approx(20.0)
    assert np.sum(TF_FREQS >= 1.0) == 800
    assert np.all(np.diff(TF_FREQS) > 0)


@pytest.mark.parametrize("factory", [LaplaceOperator, WaveletOperator])
def test_unstacked_branch_starts_at_the_unit_outcrop(factory):
    torch.manual_seed(0)
    batch = sample_tf_batch(2, n_query=3, rng=np.random.default_rng(5))
    predicted = predict(factory(), batch)
    assert torch.isfinite(predicted).all()
    assert torch.allclose(
        predicted[..., 0], torch.ones_like(predicted[..., 0]), atol=1e-5
    )
    assert torch.allclose(
        predicted[..., 1], torch.zeros_like(predicted[..., 1]), atol=1e-5
    )


def test_rayleigh_chain_is_finite():
    torch.manual_seed(0)
    batch = sample_tf_batch(2, n_query=3, rng=np.random.default_rng(5))
    predicted = predict(RayleighChain(), batch)
    assert predicted.shape == batch.target.shape
    assert torch.isfinite(predicted).all()


@pytest.mark.parametrize("factory", [LaplaceOperator, WaveletOperator, RayleighChain])
def test_one_column_operator_loss_decreases(factory):
    torch.manual_seed(0)
    batch = sample_tf_batch(1, n_query=4, rng=np.random.default_rng(8))
    losses = fit_batch(
        factory(), batch, steps=12, lr=1e-3, loss=relative_l1_by_frequency
    )
    assert math.isfinite(losses[-1])
    assert losses[-1] < losses[0]


def test_travel_time_profile_is_a_chain_column():
    sequence = travel_time_from_vs(
        np.array([0.0, 12.0, 30.0]), np.array([250.0, 1400.0])
    )
    column = soil_layers(sequence)
    assert column is not None
    thickness, velocity, bedrock = column
    assert thickness == pytest.approx([12.0])
    assert velocity == pytest.approx([250.0])
    assert bedrock == pytest.approx(1400.0)
    one_layer = travel_time_from_vs(np.array([0.0, 20.0]), np.array([400.0]))
    assert soil_layers(one_layer) is None
    batch = batch_from_columns([column], n_query=4, rng=np.random.default_rng(0))
    assert batch.target.shape[1] == len(TF_FREQS)
    assert batch.target.shape[-1] == 2
    torch.manual_seed(0)
    predicted = predict(RayleighChain(), batch)
    assert predicted.shape == batch.target.shape
    assert torch.isfinite(predicted).all()
