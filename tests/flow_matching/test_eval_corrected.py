"""Round-trip and synthetic checks for the corrected evaluator."""

import numpy as np

from experiments.flow_matching_simplified.eval_corrected import (
    _layer_index_at,
    bound_rate,
    lag_pairs,
    travel_time_from_vs,
    vertical_structure,
    vs_from_travel_time,
)


def _layered(depths, vs):
    return travel_time_from_vs(np.asarray(depths, float), np.asarray(vs, float))


def test_round_trip_known_profile():
    depths = np.array([0.0, 5.0, 15.0, 30.0])
    vs = np.array([200.0, 400.0, 800.0])
    tokens = _layered(depths, vs)
    expected_dt = np.diff(depths) / vs
    np.testing.assert_allclose(
        np.diff(np.concatenate([[0.0], tokens[:, 0]])), expected_dt
    )
    _, _, recovered = vs_from_travel_time(tokens)
    np.testing.assert_allclose(recovered, vs)
    doubled = 2.0 * np.diff(depths) / expected_dt
    assert not np.allclose(doubled, vs)
    assert bound_rate([tokens]) == 1.0
    slow = _layered([0.0, 10.0], [150.0])
    _, _, slow_vs = vs_from_travel_time(slow)
    np.testing.assert_allclose(slow_vs, [150.0])
    assert bound_rate([slow]) == 1.0


def test_same_layer_fraction_matches_enumeration():
    # Thickness 4 m, lag 1 m, grid 0.5 m. P(same layer) = (4-1)/4 on a tiled domain.
    thickness = 4.0
    lag = 1.0
    n_layers = 10
    boundaries = np.arange(0.0, thickness * n_layers + 1e-9, thickness)
    vs = np.linspace(200.0, 800.0, n_layers)
    tokens = _layered(boundaries, vs)
    pairs = lag_pairs(tokens, lag, step_m=0.5)
    grid = np.arange(0.0, boundaries[-1] - lag + 1e-9, 0.5)
    i0 = _layer_index_at(boundaries, grid)
    i1 = _layer_index_at(boundaries, grid + lag)
    valid = (i0 >= 0) & (i1 >= 0) & (i0 < n_layers) & (i1 < n_layers)
    expected = i0[valid] == i1[valid]
    np.testing.assert_array_equal(pairs["same"], expected)
    # Continuous renewal formula (T-h)/T. The 0.5 m grid shifts this by less than 0.02.
    assert abs(float(pairs["same"].mean()) - (thickness - lag) / thickness) < 0.02


def test_correlation_separates_independent_layers_from_profile_constant():
    rng = np.random.default_rng(0)
    thickness = 4.0
    n_layers = 8
    boundaries = np.arange(0.0, thickness * n_layers + 1e-9, thickness)
    independent = []
    constant = []
    for _ in range(80):
        vs_i = np.exp(rng.normal(5.7, 0.35, size=n_layers))
        independent.append(_layered(boundaries, vs_i))
        vs_c = np.full(n_layers, float(np.exp(rng.normal(5.7, 0.35))))
        constant.append(_layered(boundaries, vs_c))
    ind = vertical_structure(independent, lags=(1.0,))[0]
    con = vertical_structure(constant, lags=(1.0,))[0]
    # i.i.d. layers => corr equals the same-layer fraction.
    assert abs(ind.ln_correlation - ind.same_layer_fraction) < 0.03
    assert ind.same_layer_fraction == con.same_layer_fraction
    # One Vs per profile => paired values match even across layers.
    assert con.ln_correlation > 0.98
    assert ind.ln_correlation < con.ln_correlation - 0.15
