"""Tests for VSPDB Vs → raw two-way TTS conversion."""

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_vspdb_tts import build_tts_profiles


def test_build_tts_writes_raw_seconds_not_log1p(tmp_path: Path) -> None:
    vs_path = tmp_path / "vspdb_vs_profiles.parquet"
    out_path = tmp_path / "vspdb_tts_profiles.parquet"
    vs_df = pd.DataFrame(
        {
            "velocity_metadata_id": [1, 1, 1, 2, 2, 2],
            "depth": [0.0, 10.0, 20.0, 0.0, 5.0, 15.0],
            "vs_value": [200.0, 400.0, 800.0, 100.0, 200.0, 400.0],
        }
    )
    vs_df.to_parquet(vs_path, index=False)

    out = build_tts_profiles(vs_path, out_path, validate=True)
    assert out_path.exists()
    assert list(out.columns) == ["velocity_metadata_id", "depth", "vs_value", "tts"]
    assert out["velocity_metadata_id"].nunique() == 2

    profile_1 = out[out["velocity_metadata_id"] == 1].sort_values("depth")
    # Two-way TTS: 2 * 10/200 + 2 * 10/400 = 0.1 + 0.05 = 0.15 s at 20 m
    np.testing.assert_allclose(profile_1["tts"].iloc[0], 0.0)
    np.testing.assert_allclose(profile_1["tts"].iloc[-1], 0.15, rtol=1e-6)
    # Distinct from log1p(0.15) ≈ 0.1398
    assert float(profile_1["tts"].iloc[-1]) != np.log1p(0.15)
