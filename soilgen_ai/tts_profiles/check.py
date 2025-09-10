import numpy as np
import pandas as pd

from soilgen_ai.logging_config import setup_logging

logger = setup_logging()


class TTSProfileProcessor:
    """
    An enhanced class to validate, process, and analyze TTS profiles.

    This workflow now performs three key actions:
    1.  Validates that both 'depth' and 'tts' columns are monotonic.
    2.  Calculates interval velocities for valid profiles.
    3.  Performs a sanity check by comparing calculated velocity with any
        existing 'vs_value' data.
    """

    def __init__(self, profiles_data: dict):
        if not isinstance(profiles_data, dict):
            raise TypeError("profiles_data must be a dictionary.")

        self.profiles_data = profiles_data
        self.processed_profiles = {}
        # Store invalid profiles with a reason for failure
        self.invalid_profiles = {}
        self.is_processed = False

    @staticmethod
    def _is_monotonic_increasing(series: pd.Series) -> bool:
        """Checks if a pandas Series is monotonically increasing."""
        # Using the built-in pandas method is clean and efficient.
        return series.is_monotonic_increasing

    def process_profiles(self):
        """
        Validates all profiles, calculates velocities, and performs sanity checks.
        Returns the instance to allow for method chaining.
        """
        logger.info("--- Starting Enhanced TTS Profile Processing ---")
        for pid, df in self.profiles_data.items():
            # 1. Basic structure validation
            if not isinstance(df, pd.DataFrame) or not {"depth", "tts"}.issubset(
                df.columns
            ):
                logger.warning(
                    f"Profile {pid} is malformed or missing required columns. Skipping."
                )
                continue

            # 2. Advanced physical validation
            if not self._is_monotonic_increasing(df["depth"]):
                self.invalid_profiles[pid] = "Non-monotonic depth"
                self.processed_profiles[pid] = df.copy()
                continue

            if not self._is_monotonic_increasing(df["tts"]):
                self.invalid_profiles[pid] = "Non-monotonic TTS"
                self.processed_profiles[pid] = df.copy()
                continue

            # 3. Calculation for valid profiles
            df_copy = df.copy()
            velocities = self._calculate_velocity(df_copy["tts"], df_copy["depth"])
            df_copy["calculated_vs_mps"] = pd.Series(velocities, index=df.index[:-1])

            # 4. Sanity Check against original vs_value (if it exists)
            if "vs_value" in df_copy.columns:
                # Compare original velocity to the start of the interval it represents
                original_vs = df_copy["vs_value"].iloc[:-1]
                diff_percent = (
                    100 * (df_copy["calculated_vs_mps"] - original_vs) / original_vs
                )
                df_copy["vs_diff_percent"] = diff_percent

            self.processed_profiles[pid] = df_copy

        self.is_processed = True
        logger.info("Processing complete.")
        return (
            self  # Allow chaining, e.g., processor.process_profiles().generate_report()
        )

    def generate_report(self):
        """Prints a comprehensive summary report."""
        if not self.is_processed:
            logger.error("You must run .process_profiles() before generating a report.")
            return

        total = len(self.profiles_data)
        invalid_count = len(self.invalid_profiles)
        valid_count = total - invalid_count

        logger.info("\n--- TTS Profile Validation Report ---")
        logger.info(f"Total Profiles Processed: {total}")
        logger.info(f"✅ Valid Profiles: {valid_count}")
        logger.warning(f"❌ Invalid Profiles: {invalid_count}")

        if self.invalid_profiles:
            logger.warning("\n--- Diagnostics for Invalid Profiles ---")
            for pid, reason in self.invalid_profiles.items():
                logger.warning(f"  - Profile ID: {pid}, Reason: {reason}")
                if reason == "Non-monotonic TTS":
                    self._diagnose_monotonicity_error(
                        self.processed_profiles[pid], "tts"
                    )

        if valid_count > 0:
            self._generate_valid_profile_summary()

    def _generate_valid_profile_summary(self):
        """Generates the summary for valid profiles."""
        logger.info("\n--- Analysis of Valid Profiles ---")
        valid_pids = [
            pid for pid in self.profiles_data if pid not in self.invalid_profiles
        ]
        example_pid = valid_pids[0]
        example_df = self.processed_profiles[example_pid]

        logger.info(f"Showing example from Profile ID: {example_pid}")
        with pd.option_context("display.max_rows", 6, "display.precision", 2):
            print(example_df)

        # Report on the sanity check if the column was created
        if "vs_diff_percent" in example_df.columns:
            all_diffs = pd.concat(
                [self.processed_profiles[pid]["vs_diff_percent"] for pid in valid_pids]
            ).dropna()

            logger.info(
                "\n--- Sanity Check Summary (Calculated Vs vs. Original Vs) ---"
            )
            if not all_diffs.empty:
                logger.info(f"Mean Difference: {all_diffs.mean():.2f}%")
                logger.info(f"Std Dev of Difference: {all_diffs.std():.2f}%")
            else:
                logger.info("No velocity difference data to report.")

    @staticmethod
    def _calculate_velocity(tts: pd.Series, depths: pd.Series) -> np.ndarray:
        """Calculates interval velocity using arrays of TTS and depth."""
        delta_tts = tts.diff().iloc[1:].to_numpy()
        delta_z = depths.diff().iloc[1:].to_numpy()

        # Initialize velocities array with NaNs
        velocities = np.full_like(delta_tts, np.nan)

        # Calculate velocity only where delta_tts is positive
        positive_mask = delta_tts > 0
        velocities[positive_mask] = (2 * delta_z[positive_mask]) / delta_tts[
            positive_mask
        ]

        return velocities

    @staticmethod
    def _diagnose_monotonicity_error(df: pd.DataFrame, col: str):
        """Prints specific locations where a column is not monotonic."""
        values = df[col].to_numpy()
        diffs = np.diff(values)
        problem_indices = np.where(diffs < 0)[0]

        for idx in problem_indices:
            val_before = values[idx]
            val_after = values[idx + 1]
            logger.warning(
                f"    -> Violation at index {idx + 1}: '{col}' decreases from {val_before:.5f} to {val_after:.5f}"
            )
