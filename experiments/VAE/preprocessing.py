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


def Vs_to_tts(Vs, h):
    """
    Convert Vs to cumulative travel time profile.
    """
    d_tts = h / Vs
    tts_layers = np.cumsum(d_tts)
    return np.cumsum(tts_layers)


def standardize_and_create_tts_profiles(
    df: pd.DataFrame, num_layers: int, max_depth: int
) -> tuple:
    """Calculates cumulative travel time profiles and standardizes them to a fixed size. As well, it standardizes the depth values, and the tts values are normalized by logarithmic transformation.
    Args:
        df (pd.DataFrame): Input DataFrame containing 'depth' and 'vs_value' columns.
        num_layers (int): Number of layers for the output profiles.
        max_depth (int): Maximum depth for the output profiles.
    Returns:
        tuple: A tuple containing the following elements:
            - np.ndarray: The standardized TTS profiles.
            - np.ndarray: The standard depths.
    """
    profiles_dict = {
        metadata_id: group.sort_values("depth")
        for metadata_id, group in df.groupby("velocity_metadata_id")
    }

    tts_profiles_list = []
    # For N layers, we need N+1 depth boundaries.
    standard_depths = np.linspace(0, max_depth, num_layers + 1)

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
            fill_value="extrapolate",  # type: ignore
        )
        # We want the TTS at the layer boundaries.
        resampled_tts = f_interp(standard_depths)

        # The VAE works on the travel times *within* the standardized layers, not the cumulative time.
        # We take the diff, and since the first value is at depth 0, it's always 0.
        # The output of this will be of size `num_layers`.
        resampled_d_tts = np.diff(resampled_tts)

        # Ensure travel time differences are non-negative
        resampled_d_tts[resampled_d_tts < 0] = 0

        tts_profiles_list.append(resampled_d_tts)

    tts_profiles_np = np.array(tts_profiles_list)

    # Normalize the TTS profiles by logarithmic transformation
    tts_profiles_normalized = np.log1p(tts_profiles_np)

    return tts_profiles_normalized, standard_depths
