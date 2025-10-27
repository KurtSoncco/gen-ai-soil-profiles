#!/usr/bin/env python3
"""
FFM Evaluation Test Script

This script demonstrates the complete evaluation functionality for FFM,
including all the same metrics and plots used in conv1d_gan.
"""

import numpy as np

from experiments.flow_matching.utils import (
    compute_generated_vs30,
    compute_vs100,
    ks_statistic,
)


def test_evaluation_metrics():
    """Lightweight smoke test for evaluation metrics."""
    # Test that utility functions can be imported and basic operations work
    # Simple test of computation functions
    # Shape needs to be (n_samples, n_channels, n_points)
    test_profile = np.array([[[200, 300, 400, 500, 600]]])
    samples_per_meter = 2.0
    
    # Test compute_generated_vs30
    gen_vs30 = compute_generated_vs30(test_profile, samples_per_meter)
    assert isinstance(gen_vs30, np.ndarray)
    assert len(gen_vs30) == test_profile.shape[0]
    
    # Test compute_vs100
    gen_vs100 = compute_vs100(test_profile, samples_per_meter)
    assert isinstance(gen_vs100, np.ndarray)
    
    # Test ks_statistic
    real_data = np.random.randn(100)
    gen_data = np.random.randn(100)
    ks = ks_statistic(real_data, gen_data)
    assert 0 <= ks <= 1


if __name__ == "__main__":
    test_evaluation_metrics()
