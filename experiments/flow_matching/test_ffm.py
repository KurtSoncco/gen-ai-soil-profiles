#!/usr/bin/env python3
"""
Lightweight FFM smoke tests - just verify imports and basic functionality.
"""

import torch

from experiments.flow_matching import config
from experiments.flow_matching.models import create_model


def test_ffm_implementation():
    """Basic smoke test for FFM implementation."""
    cfg = config.cfg
    device = torch.device(cfg.device)

    # Just test that models can be created
    for arch in ["unet", "fno"]:
        model = create_model(arch, cfg)
        assert model is not None
        num_params = sum(p.numel() for p in model.parameters())
        assert num_params > 0

        # Test model can be moved to device (but don't do heavy operations)
        if torch.cuda.is_available() and device.type == "cuda":
            model = model.to(device)
            # Just verify it's on the right device
            assert next(model.parameters()).is_cuda

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
