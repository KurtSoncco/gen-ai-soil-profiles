from typing import Optional

import numpy as np
import pandas as pd


def compute_vs30(profile: pd.DataFrame) -> Optional[float]:
    """
    Computes the time-averaged shear-wave velocity in the top 30 meters (Vs30).

    This function calculates Vs30 based on the standard formula:
    Vs30 = 30 / Σ (h_i / v_i)
    where h_i is the thickness of layer i and v_i is its shear-wave velocity.

    Args:
        profile: A pandas DataFrame with at least two columns:
                 'depth': Depth to the top of the layer (in meters).
                 'vs_value': Shear-wave velocity of the layer (in m/s).

    Returns:
        The calculated Vs30 value as a float, or None if the input is invalid.
        If the profile is shallower than 30m, it returns the time-averaged
        velocity over the available depth.

    Raises:
        ValueError: If required columns are missing, depths are negative,
                    or velocities are non-positive.
    """
    # --- 1. Input Validation and Preparation ---
    if not all(col in profile.columns for col in ["depth", "vs_value"]):
        raise ValueError("Profile must contain 'depth' and 'vs_value' columns.")

    if profile.empty:
        return None

    # Work on a copy to avoid modifying the original DataFrame
    profile = profile.copy()

    # Coerce columns to numeric, turning errors into NaT (which will be dropped)
    profile["depth"] = pd.to_numeric(profile["depth"], errors="coerce")
    profile["vs_value"] = pd.to_numeric(profile["vs_value"], errors="coerce")
    profile.dropna(subset=["depth", "vs_value"], inplace=True)

    if profile.empty:
        return None

    # Validate data integrity
    if (profile["depth"] < 0).any():
        raise ValueError("Depth values must be non-negative.")
    if (profile["vs_value"] <= 0).any():
        raise ValueError("Shear-wave velocity (vs_value) must be positive.")

    # --- 2. Data Processing ---
    profile.sort_values("depth", inplace=True)
    profile.reset_index(drop=True, inplace=True)

    # If the profile doesn't start at the surface, insert a row for depth 0
    if profile.loc[0, "depth"] != 0:
        # Create a new row with depth 0 and the first velocity
        surface_row = pd.DataFrame(
            [{"depth": 0, "vs_value": profile.loc[0, "vs_value"]}]
        )
        profile = pd.concat([surface_row, profile], ignore_index=True)

    # --- 3. Vectorized Calculation ---
    # Calculate the thickness of each layer
    # The thickness is the difference between the depths of subsequent layers
    thickness = profile["depth"].diff().iloc[1:].to_numpy()
    # Velocity corresponds to the layer starting at the previous depth
    velocities = profile["vs_value"].iloc[:-1].to_numpy()

    # Calculate travel time for each layer (thickness / velocity)
    travel_times = thickness / velocities

    # Cumulative sums to find the total time and depth
    cumulative_depths = np.cumsum(thickness)
    cumulative_times = np.cumsum(travel_times)

    total_depth = cumulative_depths[-1]

    # --- 4. Final Vs30 Calculation ---
    target_depth = 30.0

    # Case 1: The entire profile is shallower than 30 meters
    if total_depth < target_depth:
        # Return the time-averaged velocity for the whole profile
        return total_depth / cumulative_times[-1]

    # Case 2
    else:
        # Find the total travel time to exactly 30m using linear interpolation
        # This is more accurate than stopping at the last layer boundary
        total_time_to_30m = np.interp(target_depth, cumulative_depths, cumulative_times)
        return float(target_depth / total_time_to_30m)


def compute_vs_at_depth(profile: pd.DataFrame, z: float) -> Optional[float]:
    """
    Computes the time-averaged shear-wave velocity (Vs) to a specified depth (z).

    This function calculates VsZ based on the standard formula:
    VsZ = Z / Σ (h_i / v_i)
    where h_i is the thickness of layer i and v_i is its shear-wave velocity.

    Think of it like calculating your average speed on a road trip: it's the
    total distance traveled divided by the total time taken, not the average
    of the speeds you drove at.

    Args:
        profile (pd.DataFrame): A pandas DataFrame with at least two columns:
            'depth': Depth to the top of the layer (in meters).
            'vs_value': Shear-wave velocity of the layer (in m/s).
        z (float): The target depth (in meters) for the calculation.

    Returns:
        Optional[float]: The calculated VsZ value. If the profile is shallower
        than z, it returns the time-averaged velocity over the entire profile's
        depth. Returns None for invalid or empty profiles.

    Raises:
        ValueError: If z is not positive, required columns are missing,
                    depths are negative, or velocities are non-positive.
    """
    # --- 1. Input Validation and Preparation ---
    if z <= 0:
        raise ValueError("Target depth z must be a positive number.")

    if not all(col in profile.columns for col in ["depth", "vs_value"]):
        raise ValueError("Profile must contain 'depth' and 'vs_value' columns.")

    if profile.empty:
        return None

    # Work on a copy to avoid modifying the original DataFrame
    profile = profile.copy()
    profile["depth"] = pd.to_numeric(profile["depth"], errors="coerce")
    profile["vs_value"] = pd.to_numeric(profile["vs_value"], errors="coerce")
    profile.dropna(subset=["depth", "vs_value"], inplace=True)

    if profile.empty:
        return None

    if (profile["depth"] < 0).any():
        raise ValueError("Depth values cannot be negative.")
    if (profile["vs_value"] <= 0).any():
        raise ValueError("Shear-wave velocity (vs_value) must be positive.")

    # --- 2. Data Processing ---
    profile.sort_values("depth", inplace=True)
    profile.reset_index(drop=True, inplace=True)

    # Ensure the profile starts at the surface (depth 0)
    if profile.loc[0, "depth"] != 0:
        surface_row = pd.DataFrame(
            [{"depth": 0, "vs_value": profile.loc[0, "vs_value"]}]
        )
        profile = pd.concat([surface_row, profile], ignore_index=True)

    # --- 3. Vectorized Calculation ---
    if len(profile) < 2:
        # Not enough data to form a layer
        return float(profile["vs_value"].iloc[0]) if not profile.empty else None

    # Calculate the thickness of each layer
    thickness = profile["depth"].diff().iloc[1:].to_numpy()
    # Velocity corresponds to the layer starting at the previous depth
    velocities = profile["vs_value"].iloc[:-1].to_numpy()

    # Calculate travel time for each layer (thickness / velocity)
    travel_times = thickness / velocities

    # Calculate cumulative sums to find total depth and travel time at each interface
    cumulative_depths = np.cumsum(thickness)
    cumulative_times = np.cumsum(travel_times)

    profile_max_depth = cumulative_depths[-1]

    # --- 4. Final VsZ Calculation ---
    # Case 1: Target depth is deeper than the entire profile
    if z >= profile_max_depth:
        # Return the time-averaged velocity for the whole profile
        return float(profile_max_depth / cumulative_times[-1])

    # Case 2: Target depth is within the profile
    else:
        # Use linear interpolation to find the exact travel time to depth z.
        # This is highly accurate and avoids a complex loop.
        time_to_z = np.interp(x=z, xp=cumulative_depths, fp=cumulative_times)
        return float(z / time_to_z)


def compute_vs_rms(profile: pd.DataFrame, z: float) -> Optional[float]:
    """
    Computes the root-mean-square (RMS) shear-wave velocity (Vs rms) to a
    specified depth (z).

    The Vs rms is calculated using the formula:
    Vs_rms = sqrt( [Σ (v_i^2 * h_i)] / [Σ h_i] )
    where v_i is the velocity and h_i is the thickness of layer i.

    Args:
        profile (pd.DataFrame): A DataFrame with 'depth' and 'vs_value' columns.
        z (float): The target depth (in meters) for the calculation.

    Returns:
        Optional[float]: The calculated Vs rms value. If the profile is shallower
        than z, it returns the Vs rms for the entire profile's depth. Returns
        None for invalid or empty profiles.

    Raises:
        ValueError: If z is not positive, required columns are missing,
                    depths are negative, or velocities are non-positive.
    """
    # --- 1. Input Validation and Preparation ---
    if z <= 0:
        raise ValueError("Target depth z must be a positive number.")

    if not all(col in profile.columns for col in ["depth", "vs_value"]):
        raise ValueError("Profile must contain 'depth' and 'vs_value' columns.")

    if profile.empty:
        return None

    profile = profile.copy()
    profile["depth"] = pd.to_numeric(profile["depth"], errors="coerce")
    profile["vs_value"] = pd.to_numeric(profile["vs_value"], errors="coerce")
    profile.dropna(subset=["depth", "vs_value"], inplace=True)

    if profile.empty:
        return None

    if (profile["depth"] < 0).any():
        raise ValueError("Depth values cannot be negative.")
    if (profile["vs_value"] <= 0).any():
        raise ValueError("Shear-wave velocity (vs_value) must be positive.")

    # --- 2. Data Processing ---
    profile.sort_values("depth", inplace=True)
    profile.reset_index(drop=True, inplace=True)

    if profile.loc[0, "depth"] != 0:
        surface_row = pd.DataFrame(
            [{"depth": 0, "vs_value": profile.loc[0, "vs_value"]}]
        )
        profile = pd.concat([surface_row, profile], ignore_index=True)

    if len(profile) < 2:
        return float(profile["vs_value"].iloc[0]) if not profile.empty else None

    # --- 3. Vectorized Calculation ---
    thickness = profile["depth"].diff().iloc[1:].to_numpy()
    velocities = profile["vs_value"].iloc[:-1].to_numpy()

    # Calculate the weighted squared velocity (v_i^2 * h_i) for each layer
    weighted_sq_velocity = (velocities**2) * thickness

    # Calculate cumulative sums
    cumulative_depths = np.cumsum(thickness)
    cumulative_weighted_sq_vel_sum = np.cumsum(weighted_sq_velocity)

    profile_max_depth = cumulative_depths[-1]

    # --- 4. Final Vs rms Calculation ---
    # Case 1: Target depth is deeper than the entire profile
    if z >= profile_max_depth:
        total_sum = cumulative_weighted_sq_vel_sum[-1]
        mean_sq_vel = total_sum / profile_max_depth
        return float(np.sqrt(mean_sq_vel))

    # Case 2: Target depth is within the profile
    else:
        # Interpolate to find the cumulative sum at exactly depth z
        sum_to_z = np.interp(
            x=z, xp=cumulative_depths, fp=cumulative_weighted_sq_vel_sum
        )
        mean_sq_vel = sum_to_z / z
        return float(np.sqrt(mean_sq_vel))
