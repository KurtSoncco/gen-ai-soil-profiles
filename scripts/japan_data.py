"""Process Japan borehole data and create a parquet file with velocity profiles.

This script reads the Japan borehole compiled data and creates a standardized
parquet file with columns: velocity_metadata_id, depth, vs, tts.
"""

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def load_raw_data(file_path: Path) -> pd.DataFrame:
    """Load the raw Japan borehole data from text file.

    Args:
        file_path: Path to the input text file.

    Returns:
        DataFrame with the raw data.
    """
    df = pd.read_csv(
        file_path,
        sep="\t",
        dtype={
            "Format": str,
            "Filename": str,
            "Soil_type": str,
            "Layer_No": str,
            "Thickness_m": str,
            "Depth_m": str,
        },
        na_values=["N/A", "N/a", "n/a", ""],
    )

    # Convert numeric columns, handling mixed types
    numeric_columns = [
        "Measurement_Depth_m",
        "N_Value",
        "Density_g_cm3",
        "Initial_soil_column",
        "Final_soil_column",
        "Vp_m_s",
        "Vs_m_s",
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def create_velocity_metadata_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Create unique velocity_metadata_id for each unique Filename.

    Args:
        df: DataFrame with Filename column.

    Returns:
        DataFrame with added velocity_metadata_id column.
    """
    df = df.copy()
    unique_filenames = df["Filename"].unique()
    filename_to_id = {
        filename: idx + 1 for idx, filename in enumerate(sorted(unique_filenames))
    }
    df["velocity_metadata_id"] = df["Filename"].map(lambda x: filename_to_id.get(x, 0))
    return df


def infer_depth_ranges(df_profile: pd.DataFrame) -> pd.DataFrame:
    """Infer depth ranges for rows where Initial/Final_soil_column are missing.

    Args:
        df_profile: DataFrame for a single profile (Filename).

    Returns:
        DataFrame with inferred depth ranges.
    """
    df_profile = df_profile.copy().sort_values("Measurement_Depth_m")
    df_profile = df_profile.reset_index(drop=True)

    # Convert Initial/Final_soil_column to numeric, handling N/A
    df_profile["Initial_soil_column"] = pd.to_numeric(
        df_profile["Initial_soil_column"], errors="coerce"
    )
    df_profile["Final_soil_column"] = pd.to_numeric(
        df_profile["Final_soil_column"], errors="coerce"
    )

    # Infer missing depth ranges
    current_depth = 0.0
    for idx in range(len(df_profile)):
        initial_val = df_profile.loc[idx, "Initial_soil_column"]
        final_val = df_profile.loc[idx, "Final_soil_column"]

        if pd.notna(initial_val) and pd.notna(final_val):
            # Use provided range
            current_depth = float(final_val)
        else:
            # Infer from previous depth or use a default increment
            if idx == 0:
                # First row without depth info - assume starts at 0
                df_profile.loc[idx, "Initial_soil_column"] = 0.0
                df_profile.loc[idx, "Final_soil_column"] = 1.0  # Default 1m increment
                current_depth = 1.0
            else:
                # Continue from previous depth
                prev_final = df_profile.loc[idx - 1, "Final_soil_column"]
                if pd.notna(prev_final):
                    current_depth = float(prev_final)

                # Use a reasonable increment based on measurement depth spacing
                # or default to 1m if we can't infer
                if idx < len(df_profile) - 1:
                    # Try to infer from next row if it has depth info
                    next_initial = df_profile.loc[idx + 1, "Initial_soil_column"]
                    if pd.notna(next_initial) and float(next_initial) > current_depth:
                        increment = (float(next_initial) - current_depth) / 2
                    else:
                        increment = 1.0
                else:
                    increment = 1.0

                df_profile.loc[idx, "Initial_soil_column"] = current_depth
                df_profile.loc[idx, "Final_soil_column"] = current_depth + increment
                current_depth += increment

    return df_profile


def create_depth_vs_mapping(
    df_profile: pd.DataFrame,
) -> List[Tuple[float, float, float]]:
    """Create a list of (depth_start, depth_end, vs) tuples for a profile.

    Args:
        df_profile: DataFrame for a single profile with depth ranges and Vs values.

    Returns:
        List of tuples (depth_start, depth_end, vs).
    """
    mapping = []

    for _, row in df_profile.iterrows():
        vs_val = row["Vs_m_s"]
        # Check for NaN before conversion
        if pd.isna(vs_val):  # type: ignore
            continue
        try:
            vs = float(vs_val)
            if vs <= 0 or np.isnan(vs):
                continue
        except (ValueError, TypeError):
            continue

        depth_start_val = row["Initial_soil_column"]
        depth_end_val = row["Final_soil_column"]

        # Check for NaN before conversion
        if pd.isna(depth_start_val) or pd.isna(depth_end_val):  # type: ignore
            continue

        try:
            depth_start = float(depth_start_val)
            depth_end = float(depth_end_val)
            if np.isnan(depth_start) or np.isnan(depth_end):
                continue
        except (ValueError, TypeError):
            continue

        if depth_start >= depth_end:
            continue

        mapping.append((depth_start, depth_end, vs))

    return mapping


def interpolate_to_regular_depths(
    depth_vs_mapping: List[Tuple[float, float, float]], depth_interval: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate velocity profile to regular depth intervals.

    For each interval, if multiple Vs values exist, use geometric mean.

    Args:
        depth_vs_mapping: List of (depth_start, depth_end, vs) tuples.
        depth_interval: Target depth interval in meters (default 0.5).

    Returns:
        Tuple of (depths, vs_values) arrays.
    """
    if not depth_vs_mapping:
        return np.array([]), np.array([])

    # Find overall depth range
    max_depth = max(m[1] for m in depth_vs_mapping)

    # Create regular depth grid
    depths = np.arange(0, max_depth + depth_interval, depth_interval)

    # For each depth interval, collect all Vs values that overlap
    vs_values = []
    for depth in depths:
        # Find all layers that contain this depth
        overlapping_vs = []
        for depth_start, depth_end, vs in depth_vs_mapping:
            # Check if depth is within [depth_start, depth_end)
            if depth_start <= depth < depth_end:
                overlapping_vs.append(vs)

        if not overlapping_vs:
            # No data at this depth - use NaN or forward fill
            vs_values.append(np.nan)
        elif len(overlapping_vs) == 1:
            vs_values.append(overlapping_vs[0])
        else:
            # Multiple Vs values - use geometric mean
            vs_values.append(np.exp(np.mean(np.log(overlapping_vs))))

    vs_values = np.array(vs_values)

    # Forward fill NaN values (use previous valid value)
    mask = ~np.isnan(vs_values)
    if np.any(mask):
        vs_series = pd.Series(vs_values)
        vs_series = vs_series.ffill().bfill()
        vs_values = vs_series.values

    # Remove depths beyond max_depth
    valid_mask = depths <= max_depth
    depths = depths[valid_mask]
    vs_values = vs_values[valid_mask]

    return depths, np.asarray(vs_values, dtype=float)


def calculate_tts(depths: np.ndarray, vs_values: np.ndarray) -> np.ndarray:
    """Calculate two-way travel time (TTS) from velocity profile.

    TTS is calculated as cumulative two-way travel time:
    tts[i] = sum(2 * thickness[j] / vs[j]) for j from 0 to i-1

    Args:
        depths: Array of depths in meters.
        vs_values: Array of Vs values in m/s.

    Returns:
        Array of TTS values in seconds.
    """
    if len(depths) < 2:
        return np.array([0.0])

    # Ensure profile starts at depth 0
    if depths[0] != 0:
        depths = np.insert(depths, 0, 0.0)
        vs_values = np.insert(vs_values, 0, vs_values[0])

    # Calculate layer thicknesses
    thicknesses = np.diff(depths)

    # Calculate travel time for each layer
    # Use velocities at the start of each layer
    layer_tts = thicknesses / vs_values[:-1]

    # Cumulative sum (TTS at surface is 0)
    tts = np.concatenate(([0.0], np.cumsum(layer_tts)))

    return tts


def process_profile(
    df_profile: pd.DataFrame, depth_interval: float = 0.5
) -> pd.DataFrame:
    """Process a single profile to create interpolated depth, vs, and tts.

    Args:
        df_profile: DataFrame for a single profile.
        depth_interval: Target depth interval in meters.

    Returns:
        DataFrame with columns: velocity_metadata_id, depth, vs, tts.
    """
    # Infer depth ranges if needed
    df_profile = infer_depth_ranges(df_profile)

    # Create depth-Vs mapping
    depth_vs_mapping = create_depth_vs_mapping(df_profile)

    if not depth_vs_mapping:
        return pd.DataFrame(
            {"velocity_metadata_id": [], "depth": [], "vs": [], "tts": []}
        )

    # Interpolate to regular depths
    depths, vs_values = interpolate_to_regular_depths(depth_vs_mapping, depth_interval)

    if len(depths) == 0:
        return pd.DataFrame(
            {"velocity_metadata_id": [], "depth": [], "vs": [], "tts": []}
        )

    # Calculate TTS
    tts = calculate_tts(depths, vs_values)

    # Get velocity_metadata_id
    velocity_metadata_id = df_profile["velocity_metadata_id"].iloc[0]

    # Create result DataFrame
    result_df = pd.DataFrame(
        {
            "velocity_metadata_id": velocity_metadata_id,
            "depth": depths,
            "vs": vs_values,
            "tts": tts,
        }
    )

    return result_df


def process_all_profiles(df: pd.DataFrame, depth_interval: float = 0.5) -> pd.DataFrame:
    """Process all profiles in the dataset.

    Args:
        df: DataFrame with all raw data.
        depth_interval: Target depth interval in meters.

    Returns:
        Combined DataFrame with all processed profiles.
    """
    all_results = []

    for velocity_metadata_id in df["velocity_metadata_id"].unique():
        df_profile = df[df["velocity_metadata_id"] == velocity_metadata_id].copy()

        # Ensure we have a DataFrame, not a Series
        if isinstance(df_profile, pd.Series):
            df_profile = df_profile.to_frame().T

        try:
            result_df = process_profile(df_profile, depth_interval)
            if not result_df.empty:
                all_results.append(result_df)
        except Exception as e:
            print(f"Warning: Failed to process profile {velocity_metadata_id}: {e}")
            continue

    if not all_results:
        return pd.DataFrame(
            {"velocity_metadata_id": [], "depth": [], "vs": [], "tts": []}
        )

    return pd.concat(all_results, ignore_index=True)


def plot_sample_profiles(result_df: pd.DataFrame, n_samples: int = 10) -> None:
    """Plot a sample of the result_df.

    Plotting Vs and tts profiles, where x label, ticks and labels are on top
    and depth is in the y axis, inverted, starting at 0.

    Args:
        result_df: DataFrame with columns velocity_metadata_id, depth, vs, tts.
        n_samples: Number of profiles to sample and plot.
    """
    # Sample unique profiles
    unique_profiles = result_df["velocity_metadata_id"].unique()
    n_samples = min(n_samples, len(unique_profiles))
    sampled_ids = np.random.choice(
        np.array(unique_profiles), size=n_samples, replace=False
    )
    sample_df = result_df[
        result_df["velocity_metadata_id"].isin(list(sampled_ids))
    ].copy()

    # Sort by velocity_metadata_id and depth for proper plotting
    sample_df = sample_df.sort_values(["velocity_metadata_id", "depth"])  # type: ignore

    fig, ax = plt.subplots(1, 2, figsize=(10, 6))

    # Plot each profile
    for profile_id in sampled_ids:
        profile_data = sample_df[sample_df["velocity_metadata_id"] == profile_id]
        ax[0].step(
            profile_data["vs"],
            profile_data["depth"],
            label=f"Profile {profile_id}",
            linewidth=1.5,
            where="post",
        )
        ax[1].plot(
            profile_data["tts"],
            profile_data["depth"],
            label=f"Profile {profile_id}",
            linewidth=1.5,
        )

    # Configure Vs plot
    ax[0].set_xlabel("Vs (m/s)")
    ax[0].set_ylabel("Depth (m)")
    ax[0].set_xlim(left=0)
    ax[0].invert_yaxis()  # Invert y-axis directly on Axes object
    ax[0].grid(True, alpha=0.3)
    ax[0].legend(loc="upper right", fontsize=8)
    ax[0].xaxis.set_label_position("top")
    ax[0].xaxis.tick_top()

    # Configure TTS plot
    ax[1].set_xlabel("TTS (s)")
    ax[1].set_ylabel("Depth (m)")
    ax[1].set_xlim(left=0)
    ax[1].invert_yaxis()  # Invert y-axis directly on Axes object
    ax[1].grid(True, alpha=0.3)
    ax[1].legend(loc="upper right", fontsize=8)
    ax[1].xaxis.set_label_position("top")
    ax[1].xaxis.tick_top()

    plt.tight_layout()
    plt.show()


def main():
    """Main function to process Japan borehole data."""
    # Set up paths
    project_root = Path(__file__).parent.parent
    input_file = project_root / "data" / "Japan_boreholds_compiled_data.txt"
    output_file = project_root / "data" / "japan_borehole_profiles.parquet"

    # Load raw data
    print(f"Loading data from {input_file}...")
    df = load_raw_data(input_file)
    print(f"Loaded {len(df)} rows from {df['Filename'].nunique()} unique files.")

    # Create velocity metadata IDs
    df = create_velocity_metadata_ids(df)
    print(
        f"Created {df['velocity_metadata_id'].nunique()} unique velocity metadata IDs."
    )

    # Process all profiles
    print("Processing profiles...")
    result_df = process_all_profiles(df, depth_interval=0.5)
    print(f"Processed {result_df['velocity_metadata_id'].nunique()} profiles.")
    print(f"Total rows in output: {len(result_df)}")

    # Plot a random sample of the result_df
    plot_sample_profiles(result_df, n_samples=10)

    # Save to parquet
    print(f"Saving to {output_file}...")
    table = pa.Table.from_pandas(result_df)
    pq.write_table(table, output_file)
    print(f"✅ Data saved successfully to '{output_file}'.")


if __name__ == "__main__":
    main()
