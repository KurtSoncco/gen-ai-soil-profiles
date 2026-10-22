"""Checks for the Stage 8 engineering metrics."""

import numpy as np

from experiments.flow_matching_simplified.eval_corrected import travel_time_from_vs
from experiments.flow_matching_simplified.site_metrics import (
    f0_and_peak,
    fit_residual_reference,
    residual_correlation,
    thickness_ks,
    vs_z,
    vs_z_or_h,
)


def test_single_layer_resonates_at_quarter_wavelength():
    bounds = np.array([0.0, 20.0, 60.0])
    vs = np.array([200.0, 1000.0])
    f0, peak = f0_and_peak(bounds, vs)
    assert abs(f0 - 200.0 / (4 * 20.0)) / 2.5 < 0.03
    # Undamped peak is the impedance ratio, 5. Damping 2% lowers it.
    assert 3.5 < peak < 5.0


def test_vs30_is_time_averaged():
    bounds = np.array([0.0, 10.0, 40.0])
    vs = np.array([150.0, 600.0])
    expected = 30.0 / (10.0 / 150.0 + 20.0 / 600.0)
    assert abs(vs_z(bounds, vs, 30.0) - expected) < 1e-9
    assert np.isnan(vs_z(np.array([0.0, 20.0]), np.array([300.0]), 30.0))
    assert abs(vs_z_or_h(np.array([0.0, 20.0]), np.array([300.0]), 30.0) - 300.0) < 1e-9


def test_residual_correlation_recovers_ar1():
    rng = np.random.default_rng(1)
    rho = 0.7
    seqs = []
    for _ in range(400):
        bounds = np.arange(0.0, 41.0, 4.0)
        z = np.empty(len(bounds) - 1)
        z[0] = rng.normal()
        for i in range(1, len(z)):
            z[i] = rho * z[i - 1] + np.sqrt(1 - rho**2) * rng.normal()
        vs = np.exp(5.5 + 0.4 * z)
        seqs.append(travel_time_from_vs(bounds, vs))
    ref = fit_residual_reference(seqs)
    out = residual_correlation(seqs, ref)
    assert abs(out["rho"] - rho) < 0.05
    assert abs(out["residual_sd"] - 1.0) < 0.1


def test_thickness_ks_zero_on_identical_sets():
    seq = travel_time_from_vs(
        np.array([0.0, 2.0, 12.0, 50.0]), np.array([150.0, 300.0, 600.0])
    )
    assert thickness_ks([seq] * 5, [seq] * 5)["weighted_ks"] == 0.0
