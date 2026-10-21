"""Configuration extensions for depth-dependent statistics training.

This module extends the base Config class with hyperparameters for
depth-dependent statistics and vertical correlation matching.
"""

from dataclasses import dataclass, field
from typing import List

try:
    from experiments.flow_matching_simplified import config as cfg_mod
except ImportError:
    import config as cfg_mod  # type: ignore


@dataclass
class DepthStatsConfig:
    """Configuration for depth-dependent statistics training.

    This can be used to extend the base Config or used standalone.
    """

    # Depth-dependent regularization
    depth_stats_weight: float = 0.01  # Loss weight for matching σ(ln Vs) and μ(ln Vs)
    vertical_corr_weight: float = (
        0.025  # Loss weight for matching vertical correlations (increased from 0.01)
    )
    depth_bin_width: float = 10.0  # Bin width for depth statistics (m)
    correlation_lags: List[float] = field(
        default_factory=lambda: [1.0, 2.0, 5.0, 10.0]
    )  # Depth lags for correlation computation (focus on 1-2m where gap is largest)
    max_depth_for_stats: float = 500.0  # Maximum depth to consider for statistics (m)

    # Loss tuning parameters
    min_bin_count: int = (
        50  # Minimum samples per bin to include in loss (avoid noisy targets)
    )
    mean_weight_ratio: float = (
        2.0  # Weight ratio: mean_loss / std_loss (emphasize mean trend correction)
    )
    short_lag_weight: float = (
        2.0  # Extra weight for short lags (1-2m) in correlation loss
    )

    # Paths for pre-computed statistics
    target_stats_path: str = (
        "real_depth_stats.pkl"  # Path to pre-computed depth statistics
    )
    target_corr_path: str = "real_correlations.pkl"  # Path to pre-computed correlations

    # Enable/disable depth-aware losses
    use_depth_stats_loss: bool = True  # Enable depth statistics loss
    use_vertical_corr_loss: bool = True  # Enable vertical correlation loss


def extend_config(base_config: cfg_mod.Config) -> cfg_mod.Config:
    """Extend base config with depth statistics hyperparameters.

    This function adds the depth statistics config attributes to the base config.
    Note: This modifies the config object in place.

    Args:
        base_config: Base Config instance

    Returns:
        Extended config (same object, modified in place)
    """
    depth_config = DepthStatsConfig()

    # Add attributes to base config
    base_config.depth_stats_weight = depth_config.depth_stats_weight
    base_config.vertical_corr_weight = depth_config.vertical_corr_weight
    base_config.depth_bin_width = depth_config.depth_bin_width
    base_config.correlation_lags = depth_config.correlation_lags
    base_config.max_depth_for_stats = depth_config.max_depth_for_stats
    base_config.target_stats_path = depth_config.target_stats_path
    base_config.target_corr_path = depth_config.target_corr_path
    base_config.use_depth_stats_loss = depth_config.use_depth_stats_loss
    base_config.use_vertical_corr_loss = depth_config.use_vertical_corr_loss
    base_config.min_bin_count = depth_config.min_bin_count
    base_config.mean_weight_ratio = depth_config.mean_weight_ratio
    base_config.short_lag_weight = depth_config.short_lag_weight

    return base_config


def get_depth_stats_config(base_config: cfg_mod.Config = None) -> DepthStatsConfig:
    """Get depth statistics config, optionally extending base config.

    Args:
        base_config: Optional base config to extend

    Returns:
        DepthStatsConfig instance
    """
    if base_config is None:
        base_config = cfg_mod.cfg

    # Extend base config if needed
    if not hasattr(base_config, "depth_stats_weight"):
        extend_config(base_config)

    return DepthStatsConfig(
        depth_stats_weight=getattr(base_config, "depth_stats_weight", 0.01),
        vertical_corr_weight=getattr(base_config, "vertical_corr_weight", 0.025),
        depth_bin_width=getattr(base_config, "depth_bin_width", 10.0),
        correlation_lags=getattr(
            base_config, "correlation_lags", [1.0, 2.0, 5.0, 10.0]
        ),
        max_depth_for_stats=getattr(base_config, "max_depth_for_stats", 500.0),
        target_stats_path=getattr(
            base_config, "target_stats_path", "real_depth_stats.pkl"
        ),
        target_corr_path=getattr(
            base_config, "target_corr_path", "real_correlations.pkl"
        ),
        use_depth_stats_loss=getattr(base_config, "use_depth_stats_loss", True),
        use_vertical_corr_loss=getattr(base_config, "use_vertical_corr_loss", True),
        min_bin_count=getattr(base_config, "min_bin_count", 50),
        mean_weight_ratio=getattr(base_config, "mean_weight_ratio", 2.0),
        short_lag_weight=getattr(base_config, "short_lag_weight", 2.0),
    )
