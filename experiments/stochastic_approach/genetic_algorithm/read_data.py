from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def load_or_generate_data(file_path: Path) -> Dict[str, pd.DataFrame]:
    """
    Tries to load real data from a Parquet file. If the file is not found,
    it generates synthetic data with similar characteristics for demonstration.
    Args:
        file_path (Path): Path to the Parquet data file.
    Returns:
        Dict[str, pd.DataFrame]: Dictionary of dataframes grouped by 'velocity_metadata_id'.
    """
    if not file_path.exists():
        print(
            f"Warning: Data file not found at '{file_path}'. Generating synthetic data."
        )
        num_profiles = 50
        data_dict = {}
        for i in range(num_profiles):
            num_points = np.random.randint(20, 100)
            max_depth = np.random.uniform(1000, 7000)
            depth = np.sort(np.random.uniform(0, max_depth, num_points))
            # Generate time that generally increases with depth
            time = np.cumsum(np.random.uniform(0.1, 0.5, num_points))
            # Generate velocity values
            vs_value = 1500 + np.cumsum(np.random.normal(0, 50, num_points))
            profile_df = pd.DataFrame(
                {
                    "depth": depth,
                    "vs_value": vs_value,
                    "tts": time,  # Assuming tts is travel time
                }
            )
            data_dict[f"synthetic_{i}"] = profile_df
        return data_dict

    print(f"Loading real data from '{file_path}'...")
    table = pq.read_table(file_path)
    df = table.to_pandas()
    grouped = df.groupby("velocity_metadata_id")
    data_dict = {
        name: group[["depth", "tts"]].reset_index(drop=True) for name, group in grouped
    }
    return data_dict
