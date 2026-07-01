"""Pytest configuration and fixtures for tests."""

import pytest
import torch


@pytest.fixture(autouse=True)
def clear_cuda_cache():
    """Clear CUDA cache before and after each test to prevent memory issues."""
    # Clear cache before test
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    yield

    # Clear cache after test
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
