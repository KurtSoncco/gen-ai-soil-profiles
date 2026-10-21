"""Passeri one-way travel time: Δt = Δz / Vs, inverted by Vs = Δz / Δt."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from soilgen_ai.vs_profiles.vs_calculation import compute_tts

PARQUET = Path(__file__).resolve().parents[2] / "data" / "vspdb_tts_profiles.parquet"
ONEWAY = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "vspdb_tts_profiles_onewaytime.parquet"
)
N_PROFILES = 25


def test_passeri_round_trip():
    depths = np.array([0.0, 5.0, 15.0, 30.0])
    vs = np.array([200.0, 400.0, 800.0])
    dt = np.diff(depths) / vs
    recovered = np.diff(depths) / dt
    np.testing.assert_allclose(recovered, vs)
    assert np.allclose(dt, np.array([5.0 / 200.0, 10.0 / 400.0, 15.0 / 800.0]))


def test_compute_tts_is_one_way():
    profile = pd.DataFrame(
        {"depth": [0.0, 5.0, 15.0, 30.0], "vs_value": [200.0, 400.0, 800.0, 800.0]}
    )
    out = compute_tts(profile)
    assert out is not None
    thickness = np.diff(out["depth"].to_numpy())
    expected = np.concatenate(
        [[0.0], np.cumsum(thickness / out["vs_value"].to_numpy()[:-1])]
    )
    np.testing.assert_allclose(out["tts"].to_numpy(), expected, rtol=1e-12, atol=0.0)


def _max_rel_error(path: Path, n_profiles: int | None) -> tuple[float, int]:
    frame = pd.read_parquet(path)
    ids = sorted(frame["velocity_metadata_id"].unique())
    if n_profiles is not None:
        ids = ids[:n_profiles]
    worst = 0.0
    n_layers = 0
    for pid in ids:
        group = frame[frame["velocity_metadata_id"] == pid].sort_values("depth")
        z = group["depth"].to_numpy(dtype=float)
        t = group["tts"].to_numpy(dtype=float)
        v = group["vs_value"].to_numpy(dtype=float)
        dz = np.diff(z)
        dt = np.diff(t)
        vv = v[:-1]
        ok = (dz > 0) & (dt > 0) & (vv > 0) & np.isfinite(vv)
        if not np.any(ok):
            continue
        recovered = dz[ok] / dt[ok]
        rel = np.abs(recovered - vv[ok]) / vv[ok]
        worst = max(worst, float(np.max(rel)))
        n_layers += int(ok.sum())
    return worst, n_layers


def test_stored_parquet_inverts_to_vs_value():
    """The legacy file still stores 2h/Vs, so this stays red on purpose."""
    if not PARQUET.is_file():
        pytest.skip(f"missing {PARQUET}")
    worst, n_layers = _max_rel_error(PARQUET, N_PROFILES)
    assert n_layers > 0
    assert worst < 1e-4, (
        f"max relative error {worst} on {n_layers} layers from {PARQUET.name}"
    )


def test_oneway_parquet_inverts_to_vs_value():
    if not ONEWAY.is_file():
        pytest.skip(f"missing {ONEWAY}")
    worst, n_layers = _max_rel_error(ONEWAY, None)
    assert n_layers == 352092
    assert worst < 1e-4, (
        f"max relative error {worst} on {n_layers} layers from {ONEWAY.name}"
    )
