import numpy as np
import pandas as pd
import pytest

from soilgen_ai.vs_profiles.vs_calculation import (
    compute_tts,
    compute_vs30,
    compute_vs_at_depth,
    compute_vs_rms,
)


@pytest.fixture
def sample_profile():
    profile = pd.DataFrame(
        {
            "depth": [0, 5, 15, 38, 60, 123, 200],
            "vs_value": [200, 300, 400, 500, 600, 700, 800],
        }
    )
    return profile


def test_compute_vs30(sample_profile):
    vs30 = compute_vs30(sample_profile)
    assert isinstance(vs30, float)
    assert vs30 == pytest.approx(313.04, rel=1e-2)  # Approximate expected value


def test_compute_vs_at_depth(sample_profile):
    vs_100 = compute_vs_at_depth(sample_profile, 100)
    assert isinstance(vs_100, float)
    assert vs_100 == pytest.approx(441.50, rel=1e-2)  # Approximate expected value

    vs_250 = compute_vs_at_depth(sample_profile, 250)
    assert isinstance(vs_250, float)
    assert vs_250 == pytest.approx(533.57, rel=1e-2)  # Approximate expected value


def test_compute_vs_rms(sample_profile):
    vs_rms_100 = compute_vs_rms(sample_profile, z=100)
    assert isinstance(vs_rms_100, float)
    assert vs_rms_100 == pytest.approx(496.79, rel=1e-2)  # Approximate expected value

    vs_rms_30 = compute_vs_rms(sample_profile, z=30)
    assert isinstance(vs_rms_30, float)
    assert vs_rms_30 == pytest.approx(341.56, rel=1e-2)  # Approximate expected value


def test_compute_tts(sample_profile):
    tts_profile = compute_tts(sample_profile)
    assert tts_profile is not None
    assert "tts" in tts_profile.columns
    assert len(tts_profile) == len(sample_profile)
    assert tts_profile["tts"].iloc[0] == 0.0  # First TTS value should be zero

    # Check that TTS values are increasing
    assert all(pd.to_numeric(tts_profile["tts"].diff().iloc[1:]) > 0)

    expected_tts = np.array(
        [0.0, 0.05, 0.11666667, 0.231667, 0.319667, 0.529667, 0.749667]
    )
    np.testing.assert_allclose(tts_profile["tts"], expected_tts, rtol=1e-5)

    # Standarization as well
    standardized_tts_profile = compute_tts(sample_profile, standardization=True)
    assert standardized_tts_profile is not None

    expected_standardized_tts = np.log1p(expected_tts)
    np.testing.assert_allclose(
        standardized_tts_profile["tts"], expected_standardized_tts, rtol=1e-5
    )
