import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from dotenv import load_dotenv


class VSPDBClient:
    """
    A client to interact with the VSPDB API for authentication and data retrieval.
    """

    def __init__(self):
        # Load environment variables from the single .env file
        load_dotenv()
        self.email: Optional[str] = os.getenv("AUTH_VSPDB_API_EMAIL")
        self.password: Optional[str] = os.getenv("AUTH_VSPDB_API_PASSWORD")

        if not self.email or not self.password:
            raise ValueError("VSPDB API credentials not found in the .env file.")

        self.token: Optional[str] = None
        self.base_url: str = "https://www.vspdb.org/vsLayerData"
        self.user_agent: str = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) 134.0.6998.118 Safari/537.36"
        )

    def authenticate(self) -> bool:
        """Authenticates with the API and stores the token."""
        url = "http://www.vspdb.org/users/login"
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/json",
        }
        try:
            assert self.email is not None
            assert self.password is not None
            response = requests.get(
                url, headers=headers, auth=(self.email, self.password)
            )
            response.raise_for_status()
            self.token = response.json().get("token")
            return self.token is not None
        except requests.exceptions.RequestException as e:
            print(f"Error during authentication: {e}")
            return False

    def fetch_all_data(self, limit: int = 100) -> pd.DataFrame:
        """Paginates through the API to retrieve all available data.
        Args:
            limit (int): Number of records to fetch per page.
        Returns:
            pd.DataFrame: A DataFrame containing all fetched data.
        """
        if not self.token:
            print("Authentication token is missing. Please authenticate first.")
            return pd.DataFrame()

        headers = {
            "Accept": "application/json",
            "User-Agent": self.user_agent,
            "Authorization": f"Bearer {self.token}",
        }
        all_data = []
        page = 1

        while True:
            url = f"{self.base_url}?contain=VelocityMetadata&page={page}&limit={limit}"
            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()

                if not data:
                    print("✅ All data has been fetched.")
                    break

                all_data.extend(data)
                print(f"Fetched page {page} with {len(data)} records.")
                page += 1
            except requests.exceptions.RequestException as e:
                print(f"❌ Failed to fetch data for page {page}. Error: {e}")
                break

        return pd.DataFrame(all_data)


def create_vs_profile(df: pd.DataFrame, vs_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create depth and vs_array for a given vs_id.
    Args:
        df (pd.DataFrame): The DataFrame containing velocity profile data.
        vs_id (int): The velocity metadata ID to filter the DataFrame.
    Returns:
        Tuple[np.ndarray, np.ndarray]: The depth and vs_array for the given vs_id.
    """
    filtered_df = df[df["velocity_metadata_id"] == vs_id].sort_values("vs_top_depth")

    if filtered_df.empty:
        raise ValueError(f"No data found for vs_id {vs_id}.")

    min_depth = filtered_df["vs_top_depth"].min()
    max_depth = filtered_df["vs_bottom_depth"].max()
    depth = np.arange(min_depth, max_depth, 0.5)  # Intervals of 0.5 meters
    vs_array = np.full_like(depth, np.nan, dtype=float)

    for _, row in filtered_df.iterrows():
        top_depth, bottom_depth, vs_value = (
            row["vs_top_depth"],
            row["vs_bottom_depth"],
            row["vs_layer_value"],
        )
        # Use a more robust indexing approach
        indices = np.where((depth >= top_depth) & (depth < bottom_depth))[0]
        if indices.size > 0:
            vs_array[indices] = vs_value

    if np.isnan(vs_array).any():
        print(
            f"Warning: NaN values found in vs_array for vs_id {vs_id}. Check for depth gaps."
        )
        # Return an empty tuple or raise an error if an incomplete profile is unacceptable
        return np.array([]), np.array([])

    return depth, vs_array


def process_vs_data(df: pd.DataFrame) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Processes the DataFrame to create vs_profiles for each unique vs_id.
    Args:
        df (pd.DataFrame): The DataFrame containing velocity profile data.
    Returns:
        Dict[int, Tuple[np.ndarray, np.ndarray]]: A dictionary mapping vs_id to (depth, vs_array) tuples.
    """
    vs_profiles = {}
    unique_vs_ids = df["velocity_metadata_id"].unique()

    for vs_id in unique_vs_ids:
        try:
            depth, vs_array = create_vs_profile(df, vs_id)
            if depth.size > 0:
                vs_profiles[vs_id] = (depth, vs_array)
        except ValueError as e:
            print(f"Skipping vs_id {vs_id}: {e}")

    return vs_profiles


def main():
    """Main execution function to run the data fetching and processing."""
    client = VSPDBClient()
    if not client.authenticate():
        return

    df = client.fetch_all_data()

    if not df.empty:
        # Save the DataFrame to a Parquet file
        project_root = Path(__file__).parent.parent
        output_path = project_root / "data" / "vspdb_data.parquet"

        table = pa.Table.from_pandas(df)
        pq.write_table(table, output_path)
        print(f"\n✅ Data saved to '{output_path}' successfully.")

        # Process the data to create vs_profiles
        vs_profiles = process_vs_data(df)
        print(
            f"\n✅ Created profiles for {len(vs_profiles)} out of {df['velocity_metadata_id'].nunique()} unique IDs."
        )

        # Save processed Vs profiles to a Parquet file
        profiles_output_path = project_root / "data" / "vspdb_vs_profiles.parquet"
        profiles_df = pd.DataFrame(
            [
                (vs_id, depth, vs)
                for vs_id, (depths, vs_values) in vs_profiles.items()
                for depth, vs in zip(depths, vs_values)
            ],
            columns=["velocity_metadata_id", "depth", "vs_value"],
        )
        table_profiles = pa.Table.from_pandas(profiles_df)
        pq.write_table(table_profiles, profiles_output_path)
        print(f"✅ Processed profiles saved to '{profiles_output_path}' successfully.")
    else:
        print("No data was retrieved. The DataFrame is empty.")


if __name__ == "__main__":
    main()
