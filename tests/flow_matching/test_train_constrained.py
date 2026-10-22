"""Constrained decoder: simplex round trip, exact H, penalty gradient, Toro mask."""

import numpy as np
import torch

from experiments.flow_matching_simplified.eval_corrected import (
    VS_MAX,
    VS_MIN,
    is_valid_profile,
    travel_time_from_vs,
    vs_from_travel_time,
)
from experiments.flow_matching_simplified.site_metrics import nearest_train_distance
from experiments.flow_matching_simplified.toro_baseline import match_layer_count
from experiments.flow_matching_simplified.train_constrained import (
    ConstrainedTokens,
    build_model,
    decode_tokens,
    encode_layers,
    flow_loss,
    sample,
)


def _profiles():
    rng = np.random.default_rng(0)
    out = []
    for _ in range(40):
        n = int(rng.integers(1, 8))
        bounds = np.concatenate([[0.0], np.cumsum(rng.uniform(1.0, 20.0, n))])
        out.append(travel_time_from_vs(bounds, rng.uniform(150.0, 1500.0, n)))
    return out


def test_simplex_round_trip_and_sum_h():
    seqs = _profiles()
    data = ConstrainedTokens(seqs)
    assert data.clipped_fraction == 0.0
    assert data.cond.shape[1] == 3
    for i, seq in enumerate(seqs):
        k = data.lengths[i]
        tok = data.x[i, :k] * data.scale[1] + data.scale[0]
        rebuilt = decode_tokens(tok, float(data.depths[i]))
        np.testing.assert_allclose(rebuilt, seq, rtol=1e-4)
        assert abs(rebuilt[-1, 1] - data.depths[i]) < 1e-6


def test_untrained_samples_sum_to_h_and_stay_in_bounds():
    data = ConstrainedTokens(_profiles())
    model = build_model()
    gen = sample(model, data, 16, seed=0, device=torch.device("cpu"), steps=4)
    for i, seq in enumerate(gen):
        assert is_valid_profile(seq)
        _, _, vs = vs_from_travel_time(seq)
        assert np.all((vs >= VS_MIN * 0.999) & (vs <= VS_MAX * 1.001))
    # The sample draw uses the same seed as intended depths.
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(data.lengths), 16)
    for seq, h in zip(gen, data.depths[idx]):
        assert abs(seq[-1, 1] - h) < 1e-5


def test_endpoint_penalty_has_gradient():
    data = ConstrainedTokens(_profiles())
    model = build_model()
    x1 = torch.tensor(data.x[:8])
    mask = torch.tensor(data.mask[:8])
    cond = torch.tensor(data.cond[:8])
    scale = torch.tensor(data.scale, dtype=torch.float32)
    torch.manual_seed(0)
    _, penalty = flow_loss(model, x1, mask, cond, scale, h_floor=50.0, penalty_weight=1.0)
    penalty.backward()
    grad = sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
    assert penalty.item() > 0
    assert grad > 0


def test_match_layer_count_preserves_h_and_n():
    bounds = np.array([0.0, 2.0, 5.0, 20.0, 80.0])
    for n in (1, 2, 4, 7):
        out = match_layer_count(bounds, n)
        assert len(out) == n + 1
        assert abs(out[0]) < 1e-12
        assert abs(out[-1] - 80.0) < 1e-12
        assert np.all(np.diff(out) > 0)


def test_encode_layers_is_softmax_preimage():
    bounds = np.array([0.0, 3.0, 10.0, 40.0])
    vs = np.array([200.0, 400.0, 800.0])
    tok = encode_layers(bounds, vs)
    rebuilt = decode_tokens(tok, 40.0)
    np.testing.assert_allclose(rebuilt[-1, 1], 40.0)
    np.testing.assert_allclose(np.diff(np.concatenate([[0.0], rebuilt[:, 1]])), np.diff(bounds))


def test_nearest_train_distance_is_zero_for_copies():
    seqs = _profiles()
    rng = np.random.default_rng(1)
    other = []
    for _ in range(8):
        n = int(rng.integers(1, 8))
        bounds = np.concatenate([[0.0], np.cumsum(rng.uniform(30.0, 80.0, n))])
        other.append(travel_time_from_vs(bounds, rng.uniform(150.0, 1500.0, n)))
    report = nearest_train_distance(seqs[:8], seqs, test=other)
    assert report["median"] == 0.0
    assert report["frac_closer_than_test_median"] == 1.0
