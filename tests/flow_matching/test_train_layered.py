"""Per-layer tokens: round trip, hard bounds, and a penalty that reaches the network."""

import numpy as np
import torch

from experiments.flow_matching_simplified.eval_corrected import (
    VS_MAX,
    VS_MIN,
    is_valid_profile,
    travel_time_from_vs,
    vs_from_travel_time,
)
from experiments.flow_matching_simplified.train_layered import (
    LayerTokens,
    build_model,
    flow_loss,
    sample,
    unsquash,
)


def _profiles():
    rng = np.random.default_rng(0)
    out = []
    for _ in range(40):
        n = int(rng.integers(1, 8))
        bounds = np.concatenate([[0.0], np.cumsum(rng.uniform(1.0, 20.0, n))])
        out.append(travel_time_from_vs(bounds, rng.uniform(150.0, 1500.0, n)))
    return out


def test_tokens_round_trip():
    seqs = _profiles()
    data = LayerTokens(seqs)
    assert data.clipped_fraction == 0.0
    for i, seq in enumerate(seqs):
        k = data.lengths[i]
        tok = data.x[i, :k] * data.scale[1] + data.scale[0]
        bounds = np.concatenate([[0.0], np.cumsum(np.exp(tok[:, 0]))])
        rebuilt = travel_time_from_vs(bounds, np.exp(unsquash(tok[:, 1])))
        np.testing.assert_allclose(rebuilt, seq, rtol=1e-4)


def test_untrained_samples_are_valid_and_in_bounds():
    data = LayerTokens(_profiles())
    model = build_model()
    gen = sample(model, data, 16, seed=0, device=torch.device("cpu"), steps=4)
    for seq in gen:
        assert is_valid_profile(seq)
        _, _, vs = vs_from_travel_time(seq)
        assert np.all((vs >= VS_MIN * 0.999) & (vs <= VS_MAX * 1.001))


def test_endpoint_penalty_has_gradient():
    data = LayerTokens(_profiles())
    model = build_model()
    x1 = torch.tensor(data.x[:8])
    mask = torch.tensor(data.mask[:8])
    cond = torch.tensor(data.cond[:8])
    scale = torch.tensor(data.scale, dtype=torch.float32)
    torch.manual_seed(0)
    _, penalty = flow_loss(model, x1, mask, cond, scale, ln_h_floor=10.0, penalty_weight=1.0)
    penalty.backward()
    grad = sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
    assert penalty.item() > 0
    assert grad > 0
