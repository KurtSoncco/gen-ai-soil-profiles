from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from . import config
except ImportError:
    import config

__all__ = ["VsPairsDataset", "create_dataloader"]


class VsPairsDataset(Dataset):
    """Loads Vs pairs from a Parquet file.

    This dataset applies a log1p transform followed by z-score normalization
    (mean=0, std=1) for stable FFM training with a Gaussian base distribution,
    and stores parameters to invert back to the original Vs scale using expm1.
    """

    def __init__(self, parquet_path: str, feature_columns: list[str]):
        self.parquet_path = parquet_path
        self.feature_columns = feature_columns

        df = pd.read_parquet(parquet_path)

        # Extract the vs pairs as a numpy array
        self.raw_pairs = df[feature_columns].values.astype(np.float32)

        # Compute normalization parameters for FFM training
        self._compute_normalization_params()

    def _compute_normalization_params(self):
        """Compute mean/std in log-space for z-score normalization.

        We normalize log1p(Vs): y = log1p(Vs).
        Store mean/std of y so that normalization is (y - mean) / std.
        Denormalization will be: Vs = expm1(std * x + mean).
        """
        all_values = self.raw_pairs.flatten()
        # Ensure positivity for log1p
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

        # Also compute per-column statistics for reference
        print("\nPer-column statistics (original scale):")
        # DataFrame construction with numpy array and column names
        df_stats = pd.DataFrame(self.raw_pairs)  # pyright: ignore[reportGeneralTypeIssues]
        df_stats.columns = self.feature_columns
        col_stats = df_stats.describe()
        print(col_stats)

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
        return len(self.raw_pairs)

    def __getitem__(self, idx: int) -> torch.Tensor:
        pairs = self.raw_pairs[idx]

        # Normalize the pairs
        pairs_normalized = self._normalize_sequence(pairs)

        # Shape to (C=1, L) where L=4
        x = torch.from_numpy(pairs_normalized).unsqueeze(0)  # (1, 4)
        return x

    def denormalize_batch(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        """Denormalize a batch of tensors back to original scale."""
        batch_np = batch_tensor.squeeze(1).detach().cpu().numpy()  # (B, 4)
        denormalized = self._denormalize_sequence(batch_np)
        return torch.from_numpy(denormalized).unsqueeze(1)  # (B, 1, 4)


def create_dataloader(
    batch_size: int, num_workers: int, shuffle: bool = True
) -> Tuple[DataLoader, int, VsPairsDataset]:
    dataset = VsPairsDataset(
        parquet_path=config.cfg.parquet_path,
        feature_columns=config.cfg.feature_columns,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
    )
    return loader, dataset.raw_pairs.shape[1], dataset


if __name__ == "__main__":
    print("Testing VsPairsDataset...")

    loader, max_length, dataset = create_dataloader(
        batch_size=4, num_workers=0, shuffle=False
    )
    print(f"Loaded dataloader with max_length={max_length}")
    for i, x in enumerate(loader):
        print(f"Batch {i}: x.shape={x.shape}")
        print(f"Sample range: [{x.min().item():.3f}, {x.max().item():.3f}]")
        print(f"Sample values: {x[0].squeeze()}")

        # Test denormalization
        denorm = dataset.denormalize_batch(x[:1])
        print(f"Denormalized sample: {denorm[0].squeeze()}")
        if i >= 1:
            break
    print("Dataset test completed successfully!")
