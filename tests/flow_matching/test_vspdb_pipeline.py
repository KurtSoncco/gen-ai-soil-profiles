"""End-to-end VSPDB-schema pipeline: Vs → raw TTS → breakpoints → split → model step."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import torch

from experiments.flow_matching_simplified.breakpoint_analysis import (
    extract_breakpoints,
    load_data,
    plot_breakpoints,
)
from experiments.flow_matching_simplified.config import Config
from experiments.flow_matching_simplified.data import FlowMatchingDataLoader
from experiments.flow_matching_simplified.model import TransformerModel
from experiments.flow_matching_simplified.split_utils import get_train_val_test_indices
from experiments.flow_matching_simplified.train import compute_flow_matching_loss
from scripts.build_vspdb_tts import build_tts_profiles


def _synthetic_vs_profiles(n_profiles: int = 24) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for pid in range(n_profiles):
        n_layers = int(rng.integers(4, 10))
        thickness = rng.uniform(2.0, 12.0, size=n_layers)
        bottoms = np.cumsum(thickness)
        vs = rng.uniform(150.0, 800.0, size=n_layers)
        depth = np.arange(0.0, float(bottoms[-1]), 0.5)
        vs_value = np.empty_like(depth)
        top = 0.0
        for bot, vel in zip(bottoms, vs):
            mask = (depth >= top) & (depth < bot)
            vs_value[mask] = vel
            top = bot
        vs_value[np.isnan(vs_value) | (vs_value == 0)] = vs[-1]
        for z, v in zip(depth, vs_value):
            rows.append(
                {
                    "velocity_metadata_id": pid,
                    "depth": float(z),
                    "vs_value": float(v),
                }
            )
    return pd.DataFrame(rows)


def test_vspdb_schema_pipeline_and_one_train_step(tmp_path: Path) -> None:
    vs_path = tmp_path / "vspdb_vs_profiles.parquet"
    tts_path = tmp_path / "vspdb_tts_profiles.parquet"
    bp_path = tmp_path / "breakpoints.parquet"
    fig_path = tmp_path / "breakpoints.png"
    split_path = tmp_path / "split_indices.json"

    _synthetic_vs_profiles().to_parquet(vs_path, index=False)
    tts = build_tts_profiles(vs_path, tts_path, validate=True)
    assert float(tts["tts"].max()) > 0.01
    assert float(tts["tts"].max()) < 50.0

    profiles = load_data(tts_path)
    rows = []
    for profile_id, profile in profiles.items():
        breakpoints = extract_breakpoints(profile)
        for depth, travel in breakpoints:
            rows.append({"profile_id": profile_id, "depth": depth, "tts": travel})
    pd.DataFrame(rows).to_parquet(bp_path, index=False)
    plot_breakpoints(profiles, num_profiles=4, output_paths=[fig_path], seed=42)
    assert fig_path.exists() and fig_path.stat().st_size > 0

    loader = FlowMatchingDataLoader(data_path=bp_path)
    loader.load_data()
    assert loader.sequences is not None
    n_total = len(loader.sequences)
    assert n_total >= 10

    train_idx, val_idx, test_idx = get_train_val_test_indices(
        n_total, seed=42, split_path=split_path
    )
    train_idx2, val_idx2, test_idx2 = get_train_val_test_indices(
        n_total, seed=999, split_path=split_path
    )
    assert train_idx2 == train_idx
    assert val_idx2 == val_idx
    assert test_idx2 == test_idx
    assert set(train_idx + val_idx + test_idx) == set(range(n_total))

    train_ds, val_ds = loader.get_dataset(
        max_length=20,
        normalize=True,
        train_indices=train_idx,
        val_indices=val_idx,
    )
    assert len(train_ds) == len(train_idx)
    assert len(val_ds) == len(val_idx)

    cfg = Config()
    cfg.device = "cpu"
    cfg.batch_size = 4
    model = TransformerModel(
        input_dim=cfg.input_dim,
        output_dim=cfg.output_dim,
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        time_emb_dim=16,
        use_sequence_stats=True,
        stats_dim=4,
        use_depth_conv=True,
        depth_conv_kernel_size=3,
    )
    batch = [train_ds[i] for i in range(min(4, len(train_ds)))]
    tokens = torch.stack([item["tokens"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    sequence_stats = torch.stack([item["sequence_stats"] for item in batch])
    loss = compute_flow_matching_loss(
        model, tokens, attention_mask, sequence_stats, torch.device("cpu"), config=cfg
    )
    assert torch.isfinite(loss)
    loss.backward()
