# preprocessing.py
import logging

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.interpolate import interp1d


def preprocess_data(profiles: dict) -> tuple[dict, list]:
    """
    Filters profiles based on specified criteria:
    - Removes entries with missing vs_value or depth.
    - Excludes profiles with depths >= 2000m.
    - Excludes profiles with any vs_value >= 2500 m/s.
    Args:
        profiles (dict): Input dictionary of profiles, where keys are profile IDs
                         and values are pandas DataFrames.
    Returns:
        tuple: A tuple containing:
            - dict: The filtered dictionary of profiles.
            - list: A list of profile IDs that were dropped.
    """
    initial_ids = set(profiles.keys())
    filtered_profiles = {}
    for profile_id, df in profiles.items():
        # Drop rows with NaN in 'vs_value' or 'depth'
        df_cleaned = df.dropna(subset=["vs_value", "depth"])

        # Check filtering conditions
        if (
            not df_cleaned.empty
            and df_cleaned["depth"].max() < 2000
            and df_cleaned["vs_value"].max() < 2500
        ):
            filtered_profiles[profile_id] = df_cleaned.reset_index(drop=True)

    final_ids = set(filtered_profiles.keys())
    indexes_dropped = list(initial_ids - final_ids)

    return filtered_profiles, indexes_dropped


def Vs_to_tts(profile):
    """
    Convert a velocity profile to a two-way travel time profile.
    Args:
        profile (pd.DataFrame): DataFrame containing 'depth' and 'vs_value' columns.
    Returns:
        pd.DataFrame: DataFrame with an additional 'tts' column representing two-way travel time.
    """
    profile = profile.sort_values("depth").reset_index(drop=True)
    profile["tts"] = 2 * (
        profile["depth"].diff().fillna(profile["depth"]) / profile["vs_value"]
    )
    profile["tts"] = profile["tts"].cumsum()
    return profile


def standardize_and_create_tts_profiles(
    profiles_dict: dict, num_layers: int, max_depth: int
) -> tuple:
    """
    Calculates cumulative travel time profiles and standardizes them to a fixed size.
    It also standardizes the depth values, and normalizes the TTS values via logarithmic transformation.
    Args:
        profiles_dict (dict): Input dictionary of profiles.
        num_layers (int): Number of layers for the output profiles.
        max_depth (int): Maximum depth for the output profiles.
    Returns:
        tuple: A tuple containing:
            - np.ndarray: The standardized and normalized TTS profiles (num_profiles, num_layers).
            - np.ndarray: The standard depth layer boundaries (num_layers + 1).
    """
    # For N layers, we need N+1 depth boundaries.
    standard_depths = np.linspace(0, max_depth, num_layers + 1)

    # Apply TTS conversion to each profile in the dictionary.
    tts_profiles_dict = {
        pid: Vs_to_tts(profile) for pid, profile in profiles_dict.items()
    }

    # Do some checks
    max_tts = [prof["tts"].max() for prof in tts_profiles_dict.values()]
    plt.figure(figsize=(8, 5))
    sns.histplot(max_tts, bins=30, kde=True)
    plt.title("Distribution of Maximum TTS in Profiles (Before Resampling)")
    plt.xlabel("Maximum TTS")
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()

    resampled_profiles = []
    # Iterate over each profile to perform interpolation
    for profile_id, profile_data in tts_profiles_dict.items():
        depths = profile_data["depth"].to_numpy()
        tts = profile_data["tts"].to_numpy()

        # Interpolation requires monotonically increasing depth values.
        if not np.all(np.diff(depths) >= 0):
            logging.warning(
                f"Skipping profile {profile_id} due to non-monotonic depths."
            )
            continue

        # Create an interpolation function for the current profile.
        # This maps depth to two-way travel time.
        f_interp = interp1d(
            depths,
            tts,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",  # type: ignore
        )

        # Resample the TTS profile to the standard depths to get cumulative travel times.
        resampled_tts = f_interp(standard_depths)

        # Calculate the travel time *within* each standardized layer.
        # The result is an array of size `num_layers`.
        resampled_d_tts = np.diff(resampled_tts)

        # Ensure travel time differences are non-negative.
        resampled_d_tts[resampled_d_tts < 0] = 0

        resampled_profiles.append(resampled_d_tts)

    # Convert the list of profiles to a NumPy array for batch processing.
    tts_profiles_np = np.array(resampled_profiles)

    # Normalize the TTS profiles using a logarithmic transformation.
    tts_profiles_normalized = np.log1p(tts_profiles_np)

    # Final assertions to ensure data integrity.
    assert tts_profiles_normalized.shape[1] == num_layers
    assert tts_profiles_normalized.shape[0] == len(resampled_profiles)

    return tts_profiles_normalized, standard_depths
