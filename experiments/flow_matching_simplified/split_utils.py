"""Persist and reload train/val/test indices for Flow Matching experiments.

Training, evaluation, and precompute scripts must share the same 80/10/10 split
so test metrics are reported on the held-out set. Indices are written to
``outputs/flow_matching_simplified/results/split_indices.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

try:
    from experiments.flow_matching_simplified import config as cfg_mod
except ImportError:
    import config as cfg_mod  # type: ignore


def default_split_path(cfg: cfg_mod.Config | None = None) -> Path:
    """Return the default on-disk path for saved split indices."""
    cfg = cfg or cfg_mod.cfg
    return Path(cfg.results_dir) / "split_indices.json"


def get_train_val_test_indices(
    n_total: int,
    split: tuple[float, float, float] | None = None,
    seed: int | None = None,
    split_path: Path | str | None = None,
    create: bool = True,
) -> tuple[list[int], list[int], list[int]]:
    """Load a saved split if it matches ``n_total``, else create and persist one.

    The permutation layout matches the original trainers: first ``train_frac``
    of a seeded ``torch.randperm`` is train, next ``val_frac`` is val, remainder
    is test. A dedicated ``torch.Generator`` is used so the split does not
    depend on whatever global RNG state the caller happens to have.

    Args:
        n_total: Number of profiles in the dataset.
        split: ``(train, val, test)`` fractions. Defaults to ``cfg.train_val_test_split``.
        seed: RNG seed. Defaults to ``cfg.seed``.
        split_path: JSON path. Defaults to ``cfg.results_dir / split_indices.json``.
        create: If False, raise when no matching saved split exists.

    Returns:
        ``(train_indices, val_indices, test_indices)`` as lists of int.
    """
    cfg = cfg_mod.cfg
    split = split or cfg.train_val_test_split
    seed = cfg.seed if seed is None else seed
    path = Path(split_path) if split_path is not None else default_split_path(cfg)

    if path.exists():
        with open(path) as handle:
            payload = json.load(handle)
        train_indices = [int(i) for i in payload["train"]]
        val_indices = [int(i) for i in payload["val"]]
        test_indices = [int(i) for i in payload["test"]]
        saved_n = int(
            payload.get(
                "n_total",
                len(train_indices) + len(val_indices) + len(test_indices),
            )
        )
        if saved_n == n_total:
            print(f"[info] Loaded train/val/test split from {path}")
            return train_indices, val_indices, test_indices
        print(f"[warning] Saved split n_total={saved_n} != {n_total}; recreating split")

    if not create:
        raise FileNotFoundError(
            f"Split file not found or mismatched at {path}. "
            "Run training or precompute_statistics.py first to persist indices."
        )

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    all_indices = torch.randperm(n_total, generator=generator)
    n_train = int(split[0] * n_total)
    n_val = int(split[1] * n_total)
    train_indices = all_indices[:n_train].tolist()
    val_indices = all_indices[n_train : n_train + n_val].tolist()
    test_indices = all_indices[n_train + n_val :].tolist()

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_total": n_total,
        "seed": int(seed),
        "split": list(split),
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }
    with open(path, "w") as handle:
        json.dump(payload, handle)
    print(
        f"[info] Saved train/val/test split ({n_train}/{n_val}/{len(test_indices)}) to {path}"
    )
    return train_indices, val_indices, test_indices
