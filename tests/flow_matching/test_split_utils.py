"""Tests for persisted train/val/test split indices."""

from pathlib import Path

from experiments.flow_matching_simplified.split_utils import (
    get_train_val_test_indices,
)


def test_split_is_deterministic_and_persisted(tmp_path: Path) -> None:
    split_path = tmp_path / "split_indices.json"
    train_a, val_a, test_a = get_train_val_test_indices(
        n_total=100,
        split=(0.8, 0.1, 0.1),
        seed=42,
        split_path=split_path,
    )
    assert len(train_a) == 80
    assert len(val_a) == 10
    assert len(test_a) == 10
    assert set(train_a + val_a + test_a) == set(range(100))
    assert split_path.exists()

    # Reload must not reshuffle even if a different seed is requested.
    train_b, val_b, test_b = get_train_val_test_indices(
        n_total=100,
        split=(0.8, 0.1, 0.1),
        seed=999,
        split_path=split_path,
    )
    assert train_b == train_a
    assert val_b == val_a
    assert test_b == test_a


def test_split_recreates_when_n_total_changes(tmp_path: Path) -> None:
    split_path = tmp_path / "split_indices.json"
    get_train_val_test_indices(
        n_total=50, seed=42, split_path=split_path, split=(0.8, 0.1, 0.1)
    )
    train, val, test = get_train_val_test_indices(
        n_total=80, seed=42, split_path=split_path, split=(0.8, 0.1, 0.1)
    )
    assert len(train) + len(val) + len(test) == 80
    assert set(train + val + test) == set(range(80))
