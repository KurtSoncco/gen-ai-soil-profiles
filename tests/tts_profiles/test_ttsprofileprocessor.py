import numpy as np
import pandas as pd
import pytest

# Assuming the class is in this location
from soilgen_ai.tts_profiles.check import TTSProfileProcessor


@pytest.fixture
def valid_profile_df() -> pd.DataFrame:
    """Provides a single, perfectly valid profile DataFrame."""
    return pd.DataFrame(
        {
            "depth": [0.0, 0.5, 1.0, 1.5, 2.0],
            "vs_value": [738, 738, 800, 800, 850],
            "tts": [0.00000, 0.00136, 0.00271, 0.00396, 0.00514],
        }
    )


@pytest.fixture
def invalid_tts_profile_df() -> pd.DataFrame:
    """Provides a profile with a non-monotonic TTS value."""
    return pd.DataFrame(
        {
            "depth": [0.0, 0.5, 1.0, 1.5, 2.0],
            "vs_value": [205, 205, 950, 950, 950],
            "tts": [0.00000, 0.00488, 0.00593, 0.00550, 0.00655],  # Decrease here
        }
    )


@pytest.fixture
def invalid_depth_profile_df() -> pd.DataFrame:
    """Provides a profile with a non-monotonic depth value."""
    return pd.DataFrame(
        {
            "depth": [0.0, 0.5, 1.5, 1.0, 2.0],  # Decrease here
            "vs_value": [300, 300, 400, 400, 400],
            "tts": [0.000, 0.0033, 0.0083, 0.0099, 0.011],
        }
    )


# --- Test Class for TTSProfileProcessor ---
class TestTTSProfileProcessor:
    """Groups all tests related to the TTSProfileProcessor."""

    def test_valid_profile_processing(self, valid_profile_df):
        """Tests the successful processing of a valid profile."""
        processor = TTSProfileProcessor({1: valid_profile_df}).process_profiles()

        # Check that the profile is NOT marked as invalid
        assert 1 not in processor.invalid_profiles
        assert 1 in processor.processed_profiles

        # Check that the new columns were added
        processed_df = processor.processed_profiles[1]
        assert "calculated_vs_mps" in processed_df.columns
        assert "vs_diff_percent" in processed_df.columns

        # Check that the velocity column has the correct number of values (N-1)
        assert (
            processed_df["calculated_vs_mps"].notna().sum() == len(valid_profile_df) - 1
        )

    def test_velocity_calculation_is_correct(self, valid_profile_df):
        """Tests the numerical correctness of the velocity calculation."""
        processor = TTSProfileProcessor({1: valid_profile_df}).process_profiles()
        processed_df = processor.processed_profiles[1]

        # Manually calculate the expected interval velocities: vs = 2 * Δz / Δtts
        expected_velocities = np.array(
            [
                2 * 0.5 / (0.00136 - 0.00000),  # Interval 1
                2 * 0.5 / (0.00271 - 0.00136),  # Interval 2
                2 * 0.5 / (0.00396 - 0.00271),  # Interval 3
                2 * 0.5 / (0.00514 - 0.00396),  # Interval 4
            ]
        )

        calculated_velocities = processed_df["calculated_vs_mps"].dropna().to_numpy()
        assert np.allclose(calculated_velocities, expected_velocities)

    def test_invalid_profile_non_monotonic_tts(self, invalid_tts_profile_df):
        """Tests correct identification of a profile with non-monotonic TTS."""
        processor = TTSProfileProcessor({2: invalid_tts_profile_df}).process_profiles()

        assert 2 in processor.invalid_profiles
        assert processor.invalid_profiles[2] == "Non-monotonic TTS"

        # Check that velocity was NOT calculated for the invalid profile
        processed_df = processor.processed_profiles[2]
        assert "calculated_vs_mps" not in processed_df.columns

    def test_invalid_profile_non_monotonic_depth(self, invalid_depth_profile_df):
        """Tests correct identification of a profile with non-monotonic depth."""
        processor = TTSProfileProcessor(
            {3: invalid_depth_profile_df}
        ).process_profiles()

        assert 3 in processor.invalid_profiles
        assert processor.invalid_profiles[3] == "Non-monotonic depth"

    def test_malformed_profile_is_skipped(self):
        """Tests that malformed data (e.g., missing columns) is skipped gracefully."""
        malformed_profiles = {
            4: pd.DataFrame({"depth": [0, 1], "wrong_col": [0, 1]})  # Missing 'tts'
        }
        processor = TTSProfileProcessor(malformed_profiles).process_profiles()

        # The profile should not be in the results or marked as invalid
        assert 4 not in processor.processed_profiles
        assert 4 not in processor.invalid_profiles
