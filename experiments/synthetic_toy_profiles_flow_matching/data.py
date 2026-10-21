from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from . import config
except ImportError:
    import config

__all__ = [
    "TravelTimeProfilesDataset",
    "create_dataloader",
    "extract_breakpoints",
    "reconstruct_profile",
]


def extract_breakpoints(
    profile: np.ndarray, max_depth: int = 500, apply_log1p: bool = True
) -> np.ndarray:
    """
    Extract breakpoints from a travel time profile.

    Returns breakpoints as: [depth1, tts1, tts_end]
    where:
    - depth1: depth of breakpoint (layer1 depth)
    - tts1: travel time at breakpoint (optionally log1p transformed)
    - tts_end: travel time at end (optionally log1p transformed)

    The origin (0, 0) is implicit and not included.

    Args:
        profile: Array of travel time values (length,)
        max_depth: Maximum depth (default 500)
        apply_log1p: If True, apply log1p transformation to tts values to enforce positivity

    Returns:
        Array of shape (3,) with [depth1, tts1, tts_end]
    """
    profile_len = len(profile)
    if profile_len == 0:
        return np.array([max_depth / 2, 0.0, 0.0], dtype=np.float32)

    # Find breakpoint by detecting significant change in slope
    # For 2-layer profiles, look for the largest jump in values
    diff = np.abs(np.diff(profile))

    if len(diff) > 0:
        # Find the largest jump
        jump_idx = np.argmax(diff)
        boundary_idx = jump_idx + 1
    else:
        boundary_idx = profile_len // 2

    # Ensure boundary_idx is valid
    boundary_idx = min(boundary_idx, profile_len - 1)
    boundary_idx = max(boundary_idx, 1)

    # Extract values
    depth1 = float(boundary_idx)
    tts1 = float(profile[boundary_idx])
    tts_end = float(profile[-1])

    # Apply log1p transformation to travel time values to enforce positivity
    if apply_log1p:
        tts1 = np.log1p(tts1)
        tts_end = np.log1p(tts_end)

    return np.array([depth1, tts1, tts_end], dtype=np.float32)


def reconstruct_profile(
    breakpoints: np.ndarray, max_depth: int = 500, apply_expm1: bool = True
) -> np.ndarray:
    """
    Reconstruct a piecewise linear profile from breakpoints.

    Args:
        breakpoints: Array of shape (3,) with [depth1, tts1, tts_end]
        max_depth: Maximum depth (default 500)
        apply_expm1: If True, apply expm1 transformation to tts values (inverse of log1p)

    Returns:
        Array of shape (max_depth,) with reconstructed profile
    """
    depth1, tts1, tts_end = breakpoints[0], breakpoints[1], breakpoints[2]

    # Apply expm1 transformation if breakpoints are in log space
    if apply_expm1:
        tts1 = np.expm1(tts1)
        tts_end = np.expm1(tts_end)

    # Clamp depth1 to valid range
    depth1 = max(1, min(depth1, max_depth - 1))
    depth1_int = int(round(depth1))

    # Origin is (0, 0), breakpoint is (depth1, tts1), end is (max_depth-1, tts_end)
    # Piecewise linear interpolation
    profile = np.zeros(max_depth, dtype=np.float32)

    # First segment: from (0, 0) to (depth1, tts1)
    if depth1_int > 0:
        profile[:depth1_int] = np.linspace(0.0, tts1, depth1_int)

    # Second segment: from (depth1, tts1) to (max_depth-1, tts_end)
    if depth1_int < max_depth:
        remaining_depths = max_depth - depth1_int
        profile[depth1_int:] = np.linspace(tts1, tts_end, remaining_depths)

    return profile


class TravelTimeProfilesDataset(Dataset):
    """Loads travel time profiles and extracts breakpoints instead of full profiles.

    This dataset extracts breakpoints: [depth1, tts1, tts_end] from each profile.
    The origin (0, 0) is implicit.

    Expected parquet formats:
    1) Long format: columns [velocity_metadata_id, depth, travel_time]
       - Set cfg.group_column to the profile id column name, cfg.feature_column to value column
    2) Wide format: one row per profile with list-like/array in cfg.feature_column
    """

    def __init__(
        self,
        parquet_path: str,
        feature_column: str,
        group_column: Optional[str],
        pad_value: float,
        max_length: Optional[int] = None,
    ):
        self.parquet_path = parquet_path
        self.feature_column = feature_column
        self.group_column = group_column
        self.pad_value = pad_value
        self.max_depth = max_length or 500

        df = pd.read_parquet(parquet_path)

        # Check if we need to compute travel_time from vs_value
        if feature_column == "travel_time" and "travel_time" not in df.columns:
            # Compute travel_time from vs_value
            print("Computing travel_time from vs_value...")
            if (
                group_column is not None
                and group_column in df.columns
                and "vs_value" in df.columns
            ):
                # Compute travel_time for each profile
                # Travel time = cumulative sum of (depth_increment / vs_value)
                df_sorted = df.sort_values([group_column, "depth"]).copy()

                # Compute travel_time for each group and assign back
                # Select only vs_value before groupby to avoid the warning
                travel_time_series = df_sorted.groupby(group_column, group_keys=False)[
                    "vs_value"
                ].apply(
                    lambda x: pd.Series(
                        np.cumsum(1.0 / np.maximum(x.values, 1e-6)), index=x.index
                    )
                )
                df_sorted["travel_time"] = travel_time_series
                df = df_sorted
            else:
                raise ValueError(
                    "Cannot compute travel_time: need group_column and vs_value column"
                )

        if group_column is not None and group_column in df.columns:
            sequences = (
                df.sort_values([group_column, "depth"])
                .groupby(group_column)[feature_column]
                .apply(lambda s: np.asarray(list(s), dtype=np.float32))
                .tolist()
            )
        else:
            # Wide format: assume each row's feature_column contains list/array-like
            col = df[feature_column].tolist()
            sequences = []
            for item in col:
                if isinstance(item, (list, tuple, np.ndarray)):
                    sequences.append(np.asarray(item, dtype=np.float32))
                else:
                    # Fallback: treat scalar as single-length sequence
                    sequences.append(np.asarray([float(item)], dtype=np.float32))

        # Extract breakpoints from each profile (with log1p transformation)
        self.breakpoints: List[np.ndarray] = []
        for seq in sequences:
            # Ensure sequence is at least max_depth length or pad/truncate
            if len(seq) < self.max_depth:
                # Pad with last value
                padded = np.pad(seq, (0, self.max_depth - len(seq)), mode="edge")
            else:
                padded = seq[: self.max_depth]

            # Extract breakpoints with log1p transformation for tts values
            breakpoints = extract_breakpoints(padded, self.max_depth, apply_log1p=True)
            self.breakpoints.append(breakpoints)

        # Store statistics for normalization
        all_breakpoints = np.array(self.breakpoints)
        self.min_vals = all_breakpoints.min(axis=0)
        self.max_vals = all_breakpoints.max(axis=0)
        self.mean_vals = all_breakpoints.mean(axis=0)
        self.std_vals = all_breakpoints.std(axis=0)

        # Avoid division by zero
        self.std_vals = np.where(self.std_vals < 1e-8, 1.0, self.std_vals)

        print(f"Extracted {len(self.breakpoints)} breakpoint sets")
        print("Breakpoint ranges (in log1p space, before normalization):")
        print(f"  depth1: [{self.min_vals[0]:.2f}, {self.max_vals[0]:.2f}]")
        print(f"  tts1:   [{self.min_vals[1]:.4f}, {self.max_vals[1]:.4f}] (log1p)")
        print(f"  tts_end: [{self.min_vals[2]:.4f}, {self.max_vals[2]:.4f}] (log1p)")
        print("Breakpoint statistics (for normalization):")
        print(f"  depth1: mean={self.mean_vals[0]:.2f}, std={self.std_vals[0]:.2f}")
        print(f"  tts1:   mean={self.mean_vals[1]:.4f}, std={self.std_vals[1]:.4f}")
        print(f"  tts_end: mean={self.mean_vals[2]:.4f}, std={self.std_vals[2]:.4f}")
        print("Note: Applying z-score normalization to all breakpoint values.")

    def __len__(self) -> int:
        return len(self.breakpoints)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Returns normalized breakpoints as tensor of shape (3,) with [depth1, tts1, tts_end]"""
        breakpoints = self.breakpoints[idx]

        # Apply z-score normalization: (x - mean) / std
        normalized = (breakpoints - self.mean_vals) / self.std_vals

        return torch.from_numpy(normalized).float()

    def denormalize_batch(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        """Denormalize a batch of breakpoints back to original scale.

        First reverses z-score normalization, then applies expm1 transformation
        to tts values (inverse of log1p).
        """
        import torch

        # Convert to numpy
        batch_np = batch_tensor.detach().cpu().numpy()

        # Reverse z-score normalization: x = normalized * std + mean
        batch_np = batch_np * self.std_vals + self.mean_vals

        # Apply expm1 to tts values (columns 1 and 2: tts1 and tts_end)
        # Column 0 is depth1, which doesn't need transformation
        batch_np[:, 1] = np.expm1(batch_np[:, 1])  # tts1
        batch_np[:, 2] = np.expm1(batch_np[:, 2])  # tts_end

        return torch.from_numpy(batch_np).float()


def create_dataloader(
    batch_size: int, num_workers: int, shuffle: bool = True
) -> Tuple[DataLoader, int, TravelTimeProfilesDataset]:
    dataset = TravelTimeProfilesDataset(
        parquet_path=config.cfg.parquet_path,
        feature_column=config.cfg.feature_column,
        group_column=config.cfg.group_column,
        pad_value=config.cfg.pad_value,
        max_length=config.cfg.max_length,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
    )
    # Return 3 as the "max_length" since we now have 3 breakpoint values
    return loader, 3, dataset


if __name__ == "__main__":
    print("Testing TravelTimeProfilesDataset on synthetic toy profiles...")

    loader, max_length, dataset = create_dataloader(
        batch_size=4, num_workers=0, shuffle=False
    )
    print(f"Loaded dataloader with max_length={max_length}")
    for i, x in enumerate(loader):
        print(f"Batch {i}: x.shape={x.shape}")
        print(f"Sample range: [{x.min().item():.3f}, {x.max().item():.3f}]")
        # Check a sample profile
        sample_profile = x[0].squeeze().numpy()
        print(f"Sample profile first 10 values: {sample_profile[:10]}")
        if i >= 1:
            break
    print("Dataset test completed successfully!")
