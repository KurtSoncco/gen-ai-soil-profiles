# preprocessing.py
import logging

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filters the raw DataFrame based on specified criteria."""
    df = df.dropna(subset=["vs_value", "depth"])
    df = df[df["depth"] < 2000]
    index_vals = df[df["vs_value"] >= 2500]["velocity_metadata_id"].unique()
    df = df[~df["velocity_metadata_id"].isin(index_vals)]
    return df.reset_index(drop=True)


def Vs_to_tts(Vs, dz):
    """
    Convert Vs to cumulative travel time profile.
    """
    d_tts = dz / Vs
    tts = np.cumsum(d_tts)
    return tts


def standardize_and_create_tts_profiles(
    df: pd.DataFrame, num_layers: int, max_depth: int
) -> tuple:
    """Calculates cumulative travel time profiles and standardizes them to a fixed size."""
    profiles_dict = {
        metadata_id: group.sort_values("depth")
        for metadata_id, group in df.groupby("velocity_metadata_id")
    }

    tts_profiles_list = []
    standard_depths = np.linspace(0, max_depth, num_layers)

    for profile_id, profile_data in profiles_dict.items():
        depths = profile_data["depth"].to_numpy()
        vs_values = profile_data["vs_value"].to_numpy()

        if not np.all(np.diff(depths) > 0):
            logging.warning(
                f"Skipping profile {profile_id} due to non-monotonic depths."
            )
            continue

        # Calculate TT from raw data
        raw_tts = Vs_to_tts(vs_values, np.diff(np.insert(depths, 0, 0)))

        # Use linear interpolation to resample the TTS profile to the standard depths
        f_interp = interp1d(
            depths,
            raw_tts,
            kind="linear",
            bounds_error=False,
            fill_value=(raw_tts[0], raw_tts[-1]),  # type:ignore
        )
        resampled_tts = f_interp(standard_depths)

        tts_profiles_list.append(resampled_tts)

    tts_profiles_np = np.array(tts_profiles_list)
    tt_max = np.max(tts_profiles_np) if tts_profiles_np.size > 0 else 1.0
    tts_profiles_normalized = tts_profiles_np / tt_max

    return tts_profiles_normalized, standard_depths, tt_max
