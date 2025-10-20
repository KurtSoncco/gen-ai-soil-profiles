from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from config import cfg

__all__ = ["VsProfilesDataset", "create_dataloader"]


class VsProfilesDataset(Dataset):
    """Loads Vs profiles from a Parquet file and pads to a fixed length.

    Expected parquet formats:
    1) Long format: columns [velocity_metadata_id, depth, vs_value]
       - Set cfg.group_column to the profile id column name, cfg.feature_column to value column
    2) Wide format: one row per profile with list-like/array in cfg.feature_column
    """

    def __init__(self, parquet_path: str, feature_column: str, group_column: Optional[str], pad_value: float, max_length: Optional[int] = None):
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

    def __len__(self) -> int:
        return len(self.raw_sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.raw_sequences[idx]
        length = len(seq)
        pad_len = self.max_length - length
        if pad_len < 0:
            seq = seq[: self.max_length]
            length = self.max_length
            pad_len = 0

        padded = np.pad(seq, (0, pad_len), mode="constant", constant_values=self.pad_value)
        mask = np.zeros(self.max_length, dtype=np.float32)
        mask[:length] = 1.0

        # Shape to (C=1, L)
        x = torch.from_numpy(padded).unsqueeze(0)  # (1, L)
        m = torch.from_numpy(mask).unsqueeze(0)    # (1, L)
        return x, m


def create_dataloader(batch_size: int, num_workers: int, shuffle: bool = True) -> Tuple[DataLoader, int]:
    dataset = VsProfilesDataset(
        parquet_path=cfg.parquet_path,
        feature_column=cfg.feature_column,
        group_column=cfg.group_column,
        pad_value=cfg.pad_value,
        max_length=cfg.max_length,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)
    return loader, dataset.max_length


if __name__ == "__main__":
    print("Testing VsProfilesDataset...")
    
    loader, max_length = create_dataloader(batch_size=4, num_workers=0, shuffle=False)
    print(f"Loaded dataloader with max_length={max_length}")
    for i, (x, m) in enumerate(loader):
        print(f"Batch {i}: x.shape={x.shape}, mask.shape={m.shape}, first seq lengths={[int(mask.sum().item()) for mask in m]}")
        if i >= 1:
            break
    print("Dataset test completed successfully!")
