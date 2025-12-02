"""Data Loader for Flow Matching with Variable-Length Paired Token Breakpoints.

This module loads the data for the flow matching with variable-length paired token breakpoints experiment.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class FlowMatchingDataset(Dataset):
    """PyTorch Dataset for Flow Matching with Variable-Length Paired Token Breakpoints.

    Each sample is a sequence of paired tokens: [ts_value, depth_value] per layer.
    Sequences are padded to max_length with padding tokens and attention masks are provided.
    """

    def __init__(
        self,
        sequences: list[np.ndarray],
        max_length: Optional[int] = None,
        pad_token: float = 0.0,
        normalize: bool = True,
    ):
        """
        Args:
            sequences: List of arrays, each of shape (n_tokens, 2) with [ts, depth] pairs
            max_length: Maximum sequence length. If None, uses the maximum length in sequences
            pad_token: Value to use for padding
            normalize: Whether to normalize the data using z-score normalization
        """
        self.sequences = sequences
        self.pad_token = pad_token
        self.normalize = normalize

        # Determine max_length
        if max_length is None:
            self.max_length = max(len(seq) for seq in sequences) if sequences else 0
        else:
            self.max_length = max_length

        # Normalize sequences if requested (per-profile normalization)
        if normalize:
            self.sequences = self._normalize_sequences(sequences)
            # Store per-sequence statistics for denormalization
            self.sequence_stats: list[dict[str, float]] = []
            for seq in sequences:
                ts_mean = float(np.mean(seq[:, 0]))
                ts_std = float(
                    np.std(seq[:, 0]) + 1e-10
                )  # Add epsilon to avoid division by zero
                depth_mean = float(np.mean(seq[:, 1]))
                depth_std = float(np.std(seq[:, 1]) + 1e-10)
                self.sequence_stats.append(
                    {
                        "ts_mean": ts_mean,
                        "ts_std": ts_std,
                        "depth_mean": depth_mean,
                        "depth_std": depth_std,
                    }
                )
        else:
            self.sequence_stats = []

    def _normalize_sequences(self, sequences: list[np.ndarray]) -> list[np.ndarray]:
        """Normalize sequences per-profile (each sequence normalized independently)."""
        normalized = []
        for seq in sequences:
            # Normalize each sequence using its own mean and std
            ts_mean = np.mean(seq[:, 0])
            ts_std = np.std(seq[:, 0]) + 1e-10  # Add epsilon to avoid division by zero
            ts_norm = (seq[:, 0] - ts_mean) / ts_std

            depth_mean = np.mean(seq[:, 1])
            depth_std = np.std(seq[:, 1]) + 1e-10
            depth_norm = (seq[:, 1] - depth_mean) / depth_std

            normalized.append(np.column_stack([ts_norm, depth_norm]))
        return normalized

    def denormalize_sequence(self, sequence: np.ndarray, seq_idx: int) -> np.ndarray:
        """Denormalize a single sequence using per-profile statistics.

        Args:
            sequence: Normalized sequence to denormalize
            seq_idx: Index of the sequence in the dataset (for retrieving stats)

        Returns:
            Denormalized sequence
        """
        if not self.normalize:
            return sequence

        if seq_idx >= len(self.sequence_stats):
            raise IndexError(
                f"Sequence index {seq_idx} out of range for stored statistics"
            )

        stats = self.sequence_stats[seq_idx]
        ts_denorm = sequence[:, 0] * stats["ts_std"] + stats["ts_mean"]
        depth_denorm = sequence[:, 1] * stats["depth_std"] + stats["depth_mean"]
        return np.column_stack([ts_denorm, depth_denorm])

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single sample with padding and attention mask.

        Returns:
            Dictionary with:
                - 'tokens': Padded sequence tensor of shape (max_length, 2)
                - 'attention_mask': Boolean mask of shape (max_length,)
                - 'length': Original sequence length (before padding)
                - 'sequence_stats': Optional tensor of shape (4,) with [ts_mean, ts_std, depth_mean, depth_std]
        """
        seq = self.sequences[idx]
        seq_length = len(seq)

        # Pad sequence
        if seq_length < self.max_length:
            padding = np.full((self.max_length - seq_length, 2), self.pad_token)
            padded_seq = np.vstack([seq, padding])
        else:
            padded_seq = seq[: self.max_length]
            seq_length = self.max_length

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = np.zeros(self.max_length, dtype=bool)
        attention_mask[:seq_length] = True

        result = {
            "tokens": torch.FloatTensor(padded_seq),
            "attention_mask": torch.BoolTensor(attention_mask),
            "length": torch.tensor(seq_length, dtype=torch.long),
        }

        # Add sequence statistics if available (for preserving absolute scale)
        if self.normalize and idx < len(self.sequence_stats):
            stats = self.sequence_stats[idx]
            result["sequence_stats"] = torch.FloatTensor(
                [
                    stats["ts_mean"],
                    stats["ts_std"],
                    stats["depth_mean"],
                    stats["depth_std"],
                ]
            )
        else:
            result["sequence_stats"] = torch.zeros(4)

        return result


class FlowMatchingDataLoader:
    """Data Loader for Flow Matching with Variable-Length Paired Token Breakpoints.

    This class handles loading breakpoints from parquet files and converting them
    to sequences suitable for transformer training.
    """

    def __init__(self, data_path: Path):
        """
        Args:
            data_path: Path to parquet file containing breakpoints with columns:
                      profile_id, depth, tts (column name in parquet file)
        """
        self.data_path = data_path
        self.df: Optional[pd.DataFrame] = None
        self.sequences: Optional[list[np.ndarray]] = None
        self.profile_ids: Optional[list[str]] = None

    def load_data(self) -> None:
        """Load breakpoints data from parquet file."""
        self.df = pd.read_parquet(self.data_path)

        # Group by profile_id and create sequences
        self.sequences = []
        self.profile_ids = []

        for profile_id, group in self.df.groupby("profile_id"):
            # Sort by depth
            group_sorted = group.sort_values("depth")
            depths = np.array(group_sorted["depth"].values)
            ts_values = np.array(group_sorted["tts"].values)

            # Skip empty sequences
            if len(depths) == 0:
                continue

            # Convert to [ts, depth] pairs
            sequence = np.column_stack([ts_values, depths]).astype(np.float32)
            
            # Remove (0,0) origin point if present - it's deterministic and shouldn't be learned
            # The (0,0) point is a fixed boundary condition, not part of the learned distribution.
            # It will be prepended back during evaluation/visualization for reconstruction.
            # Keep only breakpoints with depth > 0 or TTS > 0
            mask = (sequence[:, 1] > 0) | (sequence[:, 0] > 0)  # depth > 0 OR ts > 0
            sequence = sequence[mask]
            
            # Skip if sequence becomes empty after removing origin
            if len(sequence) == 0:
                continue
            
            self.sequences.append(sequence)
            self.profile_ids.append(str(profile_id))

    def get_dataset(
        self,
        max_length: int = 20,
        pad_token: float = 0.0,
        normalize: bool = True,
        train_indices: Optional[list[int]] = None,
        val_indices: Optional[list[int]] = None,
    ) -> tuple[FlowMatchingDataset, FlowMatchingDataset] | FlowMatchingDataset:
        """Create PyTorch Dataset(s) from loaded sequences.

        Args:
            max_length: Maximum sequence length. Defaults to 20.
            pad_token: Value to use for padding
            normalize: Whether to normalize the data
            train_indices: Optional list of indices for training set
            val_indices: Optional list of indices for validation set

        Returns:
            If train_indices and val_indices are provided, returns (train_dataset, val_dataset)
            Otherwise, returns a single dataset with all sequences
        """
        if self.sequences is None:
            self.load_data()

        assert self.sequences is not None, (
            "Sequences must be loaded before creating dataset"
        )
        assert self.profile_ids is not None, (
            "Profile IDs must be loaded before creating dataset"
        )

        if train_indices is not None and val_indices is not None:
            # Split into train and validation
            train_sequences = [self.sequences[i] for i in train_indices]
            val_sequences = [self.sequences[i] for i in val_indices]

            # Fit normalizer on training data only
            train_dataset = FlowMatchingDataset(
                train_sequences,
                max_length=max_length,
                pad_token=pad_token,
                normalize=normalize,
            )

            # Validation dataset with per-profile normalization
            val_dataset = FlowMatchingDataset(
                val_sequences,
                max_length=train_dataset.max_length,
                pad_token=pad_token,
                normalize=normalize,  # Normalize each profile independently
            )

            return train_dataset, val_dataset
        else:
            # Single dataset with all sequences
            return FlowMatchingDataset(
                self.sequences,
                max_length=max_length,
                pad_token=pad_token,
                normalize=normalize,
            )

    def get_dataloader(
        self,
        dataset: FlowMatchingDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> DataLoader:
        """Create a PyTorch DataLoader from a dataset.

        Args:
            dataset: FlowMatchingDataset instance
            batch_size: Batch size
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes for data loading

        Returns:
            PyTorch DataLoader
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
        )

    @staticmethod
    def _collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Collate function for batching samples.

        Args:
            batch: List of samples from __getitem__

        Returns:
            Batched dictionary with:
                - 'tokens': Tensor of shape (batch_size, max_length, 2)
                - 'attention_mask': Tensor of shape (batch_size, max_length)
                - 'length': Tensor of shape (batch_size,)
                - 'sequence_stats': Tensor of shape (batch_size, 4)
        """
        tokens = torch.stack([item["tokens"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        lengths = torch.stack([item["length"] for item in batch])
        sequence_stats = torch.stack([item["sequence_stats"] for item in batch])

        return {
            "tokens": tokens,
            "attention_mask": attention_mask,
            "length": lengths,
            "sequence_stats": sequence_stats,
        }


if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    data_loader = FlowMatchingDataLoader(
        data_path=Path(__file__).parent / "data" / "breakpoints.parquet"
    )
    data_loader.load_data()

    assert data_loader.sequences is not None, "Sequences must be loaded"
    assert data_loader.profile_ids is not None, "Profile IDs must be loaded"

    idx = np.random.randint(0, len(data_loader.sequences))
    print(data_loader.sequences[idx])
    print(data_loader.profile_ids[idx])

    print(f"Number of sequences: {len(data_loader.sequences)}")

    dataset = data_loader.get_dataset(max_length=20)
    print(dataset[0])

    assert isinstance(dataset, FlowMatchingDataset), (
        "Dataset must be a FlowMatchingDataset"
    )

    dataloader = data_loader.get_dataloader(dataset, batch_size=4, shuffle=True)
    for batch in dataloader:
        print(batch)
        break

    # Separate into training, validation and test sets
    n_total = len(data_loader.sequences)
    all_indices = torch.randperm(n_total)

    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)

    train_indices = all_indices[:n_train].tolist()
    val_indices = all_indices[n_train : n_train + n_val].tolist()
    test_indices = all_indices[n_train + n_val :].tolist()

    datasets = data_loader.get_dataset(
        train_indices=train_indices, val_indices=val_indices
    )
    assert isinstance(datasets, tuple), (
        "Expected tuple when train_indices and val_indices are provided"
    )
    train_dataset, val_dataset = datasets

    # Create test dataset separately (get_dataset doesn't support test_indices)
    assert data_loader.sequences is not None
    test_sequences = [data_loader.sequences[i] for i in test_indices]
    test_dataset = FlowMatchingDataset(
        test_sequences,
        max_length=20,
        pad_token=0.0,
        normalize=True,  # Normalize each profile independently
    )

    print(f"Number of training sequences: {len(train_dataset)}")
    print(f"Number of validation sequences: {len(val_dataset)}")
    print(f"Number of test sequences: {len(test_dataset)}")

    # Denormalize the test dataset
    # Get the normalized sequence (not the padded tensor dict)
    normalized_seq = test_dataset.sequences[0]
    print(f"Normalized sequence:\n{normalized_seq}")
    denormalized_seq = test_dataset.denormalize_sequence(normalized_seq, 0)
    print(f"Denormalized sequence:\n{denormalized_seq}")
