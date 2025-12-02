#!/usr/bin/env python3
"""Extract vs30, vs50, vs100, vs200 pairs from profile data.

This script processes the VSPDB profiles and extracts 4-dimensional vectors
containing [vs30, vs50, vs100, vs200] values for each profile.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))  # noqa: E402

from soilgen_ai.vs_profiles.vs_calculation import (  # noqa: E402
    compute_vs30,
    compute_vs_at_depth,
)


def extract_vs_pairs(parquet_path: str) -> pd.DataFrame:
    """
    Extract vs30, vs50, vs100, vs200 pairs from profile data.

    Args:
        parquet_path: Path to the parquet file containing profiles

    Returns:
        DataFrame with columns: [velocity_metadata_id, vs30, vs50, vs100, vs200]
    """
    print(f"Loading profiles from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    print(f"Found {df['velocity_metadata_id'].nunique()} unique profiles")

    results = []
    failed_count = 0

    for profile_id, group in df.groupby("velocity_metadata_id"):
        try:
            # Calculate vs30, vs50, vs100, vs200
            vs30 = compute_vs30(group)
            vs50 = compute_vs_at_depth(group, 50.0)
            vs100 = compute_vs_at_depth(group, 100.0)
            vs200 = compute_vs_at_depth(group, 200.0)

            # Check if all values are valid
            if all(
                v is not None and np.isfinite(v) for v in [vs30, vs50, vs100, vs200]
            ):
                results.append(
                    {
                        "velocity_metadata_id": profile_id,
                        "vs30": vs30,
                        "vs50": vs50,
                        "vs100": vs100,
                        "vs200": vs200,
                    }
                )
            else:
                failed_count += 1
        except Exception as e:
            failed_count += 1
            if len(results) < 10:  # Only print first few errors
                print(f"Warning: Failed to process profile {profile_id}: {e}")

    print(f"Successfully extracted {len(results)} vs pairs")
    if failed_count > 0:
        print(f"Failed to extract {failed_count} profiles")

    result_df = pd.DataFrame(results)

    # Print statistics
    print("\nStatistics:")
    print(result_df.describe())

    return result_df


def main():
    """Main execution function."""
    # Set up paths
    project_root = Path(__file__).parent.parent
    input_path = project_root / "data" / "vspdb_vs_profiles.parquet"
    output_path = project_root / "data" / "vs_pairs.parquet"

    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}")
        return

    # Extract vs pairs
    vs_pairs_df = extract_vs_pairs(str(input_path))

    # Save to parquet
    output_path.parent.mkdir(exist_ok=True)
    vs_pairs_df.to_parquet(output_path, index=False)
    print(f"\n✅ Saved vs pairs to {output_path}")
    print(f"   Shape: {vs_pairs_df.shape}")
    print(f"   Columns: {vs_pairs_df.columns.tolist()}")


if __name__ == "__main__":
    main()
