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

__all__ = ["VsProfilesDataset", "create_dataloader"]


class VsProfilesDataset(Dataset):
    """Loads Vs profiles from a Parquet file and pads to a fixed length.

    This dataset applies a log1p transform followed by z-score normalization
    (mean=0, std=1) for stable FFM training with a Gaussian base distribution,
    and stores parameters to invert back to the original Vs scale using expm1.

    Expected parquet formats:
    1) Long format: columns [velocity_metadata_id, depth, vs_value]
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

        df = pd.read_parquet(parquet_path)

        if group_column is not None and group_column in df.columns:
            sequences = (
                df.sort_values([group_column])
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

        self.raw_sequences: List[np.ndarray] = sequences
        self.max_length = max_length or max(len(x) for x in self.raw_sequences)

        # Compute normalization parameters for FFM training
        self._compute_normalization_params()

    def _compute_normalization_params(self):
        """Compute mean/std in log-space for z-score normalization.

        We normalize log1p(Vs): y = log1p(Vs).
        Store mean/std of y so that normalization is (y - mean) / std.
        Denormalization will be: Vs = expm1(std * x + mean).
        """
        all_values = np.concatenate(self.raw_sequences)
        # Ensure positivity for log1p; Vs should be positive but clip just in case
        all_values = np.clip(all_values, a_min=0.0, a_max=None)
        all_values_log = np.log1p(all_values)

        self.mean_val = float(np.mean(all_values_log))
        self.std_val = float(np.std(all_values_log) + 1e-8)

        # Store original min/max (for informational prints)
        self.min_val = float(np.min(all_values))
        self.max_val = float(np.max(all_values))

        print(
            f"Data normalization (log1p z-score): mean={self.mean_val:.4f}, std={self.std_val:.4f}"
        )
        print(f"Data range (Vs): min={self.min_val:.2f}, max={self.max_val:.2f}")

    def _normalize_sequence(self, seq: np.ndarray) -> np.ndarray:
        """Normalize sequence using log1p + z-score.

        x = Vs -> y = log1p(x) -> z = (y - mean) / std
        """
        seq = np.clip(seq, a_min=0.0, a_max=None)
        y = np.log1p(seq)
        return (y - self.mean_val) / self.std_val

    def _denormalize_sequence(self, seq: np.ndarray) -> np.ndarray:
        """Denormalize sequence back to original Vs scale.

        z -> y = z * std + mean -> x = expm1(y)
        """
        y = seq * self.std_val + self.mean_val
        x = np.expm1(y)
        return x

    def __len__(self) -> int:
        return len(self.raw_sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        seq = self.raw_sequences[idx]
        length = len(seq)
        pad_len = self.max_length - length
        if pad_len < 0:
            seq = seq[: self.max_length]
            length = self.max_length
            pad_len = 0

        # Normalize the sequence
        seq_normalized = self._normalize_sequence(seq)

        # Pad the normalized sequence
        # Note: for z-score normalization, padding should be 0 (mean-centered)
        padded = np.pad(
            seq_normalized,
            (0, pad_len),
            mode="constant",
            constant_values=0.0,  # Use 0 for z-score (mean-centered data)
        )

        # Shape to (C=1, L)
        x = torch.from_numpy(padded).unsqueeze(0)  # (1, L)
        return x

    def denormalize_batch(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        """Denormalize a batch of tensors back to original scale."""
        batch_np = batch_tensor.cpu().numpy()
        denormalized = self._denormalize_sequence(batch_np)
        return torch.from_numpy(denormalized)


def create_dataloader(
    batch_size: int, num_workers: int, shuffle: bool = True
) -> Tuple[DataLoader, int, VsProfilesDataset]:
    dataset = VsProfilesDataset(
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
    return loader, dataset.max_length, dataset


if __name__ == "__main__":
    print("Testing VsProfilesDataset...")

    loader, max_length, dataset = create_dataloader(
        batch_size=4, num_workers=0, shuffle=False
    )
    print(f"Loaded dataloader with max_length={max_length}")
    for i, x in enumerate(loader):
        print(f"Batch {i}: x.shape={x.shape}")
        print(f"Sample range: [{x.min().item():.3f}, {x.max().item():.3f}]")
        if i >= 1:
            break
    print("Dataset test completed successfully!")
