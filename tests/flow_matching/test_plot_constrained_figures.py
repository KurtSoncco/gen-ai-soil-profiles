"""Stair-edge helper for the constrained README figures."""

import numpy as np

from experiments.flow_matching_simplified.eval_corrected import travel_time_from_vs
from experiments.flow_matching_simplified.plot_constrained_figures import layer_stairs


def test_layer_stairs_matches_layer_bounds():
    bounds = np.array([0.0, 5.0, 15.0, 30.0])
    vs = np.array([200.0, 400.0, 800.0])
    values, edges = layer_stairs(travel_time_from_vs(bounds, vs))
    np.testing.assert_allclose(values, vs)
    np.testing.assert_allclose(edges, bounds)
