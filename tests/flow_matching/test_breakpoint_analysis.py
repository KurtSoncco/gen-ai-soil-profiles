"""Tests for VSPDB breakpoint extraction defaults."""

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.flow_matching_simplified.breakpoint_analysis import (
    DEFAULT_TTS_PATH,
    extract_breakpoints,
    load_data,
)


def test_default_tts_path_is_vspdb() -> None:
    assert DEFAULT_TTS_PATH.name == "vspdb_tts_profiles.parquet"
    assert "japan" not in str(DEFAULT_TTS_PATH)


def test_extract_breakpoints_on_piecewise_linear_tts(tmp_path: Path) -> None:
    # Two linear segments in TTS vs depth: 0-10 m and 10-20 m.
    depths = np.concatenate([np.linspace(0, 10, 11), np.linspace(11, 20, 10)])
    tts = np.concatenate([depths[:11] * 0.002, 0.02 + (depths[11:] - 10) * 0.001])
    parquet_path = tmp_path / "tts.parquet"
    pd.DataFrame(
        {
            "velocity_metadata_id": np.full(len(depths), 7),
            "depth": depths,
            "tts": tts,
        }
    ).to_parquet(parquet_path, index=False)

    profiles = load_data(parquet_path)
    assert len(profiles) == 1
    profile = next(iter(profiles.values()))
    breakpoints = extract_breakpoints(profile)
    assert breakpoints.shape[1] == 2
    assert len(breakpoints) >= 2
    assert breakpoints[0, 0] == 0.0
    np.testing.assert_allclose(breakpoints[-1, 0], 20.0)
