"""Depth-Dependent Vs Statistics and Vertical Correlation Matching.

This module provides tools for computing and matching depth-dependent statistics
and vertical correlation structure in Vs profiles, addressing Toro-style
depth-dependent trends.
"""

from .depth_statistics import (
    bin_by_depth,
    compute_real_data_statistics,
    plot_depth_statistics,
    reconstruct_vs_profile,
)
from .vertical_correlation import (
    compute_profile_statistics,
    compute_vertical_autocorrelation,
    plot_vertical_correlations,
)
from .losses import (
    compute_depth_statistics_loss,
    compute_vertical_correlation_loss,
)
from .utils import (
    denormalize_batch_sequences,
    load_target_statistics,
    sequence_to_vs_profile,
)

__all__ = [
    # Depth statistics
    "bin_by_depth",
    "compute_real_data_statistics",
    "plot_depth_statistics",
    "reconstruct_vs_profile",
    # Vertical correlation
    "compute_profile_statistics",
    "compute_vertical_autocorrelation",
    "plot_vertical_correlations",
    # Losses
    "compute_depth_statistics_loss",
    "compute_vertical_correlation_loss",
    # Utils
    "denormalize_batch_sequences",
    "load_target_statistics",
    "sequence_to_vs_profile",
]

