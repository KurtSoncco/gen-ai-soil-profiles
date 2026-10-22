"""Toro baseline: fitting recovers a known layering rate, and samples are valid."""

import numpy as np

from experiments.flow_matching_simplified.eval_corrected import validity_rate
from experiments.flow_matching_simplified.toro_baseline import (
    ToroFit,
    _cumulative_rate,
    fit_toro,
    sample_toro,
)


def _truth() -> ToroFit:
    return ToroFit(
        c1=10.86,
        c2=0.89,
        c3=1.98,
        rho0=0.6,
        delta=10.0,
        rho200=0.5,
        d0=1.0,
        b=0.3,
        centres=np.array([1.0, 50.0, 200.0]),
        median=np.array([5.2, 5.8, 6.4]),
        sigma=np.array([0.4, 0.35, 0.3]),
        depths=np.array([30.0, 60.0, 120.0]),
    )


def test_fit_recovers_expected_interface_count():
    truth = _truth()
    profiles = sample_toro(truth, 600, seed=3)
    assert validity_rate(profiles) == 1.0
    fit = fit_toro(profiles)
    for h in (30.0, 60.0, 120.0):
        expected = _cumulative_rate(h, truth.c1, truth.c2, truth.c3)
        fitted = _cumulative_rate(h, fit.c1, fit.c2, fit.c3)
        assert abs(fitted - expected) / expected < 0.1
