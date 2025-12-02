#!/usr/bin/env python3
"""Generate synthetic toy Vs profiles for flow matching experiments.

This script creates simple 2-layer profiles with random velocities and boundary depths.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def generate_synthetic_profile(profile_length: int = 500) -> np.ndarray:
    """
    Generate a single synthetic Vs profile with 2 layers.

    Args:
        profile_length: Number of depth points in the profile

    Returns:
        Array of Vs values (profile_length,)
    """
    # Sample Layer 1 Value (v1)
    v1 = np.random.uniform(300, 800)

    # Sample Layer 2 Value (v2)
    v2 = np.random.uniform(900, 2000)

    # Sample Boundary Depth (d1_index)
    d1_index = np.random.randint(100, 400)

    # Construct profile
    profile = np.zeros(profile_length)
    profile[:d1_index] = v1  # Layer 1
    profile[d1_index:] = v2  # Layer 2

    return profile


def generate_synthetic_dataset(
    n_profiles: int = 10000, profile_length: int = 500
) -> pd.DataFrame:
    """
    Generate a synthetic dataset of 2-layer profiles.

    Args:
        n_profiles: Number of profiles to generate
        profile_length: Number of depth points per profile

    Returns:
        DataFrame with columns: [velocity_metadata_id, depth, vs_value]
    """
    profiles = []

    for profile_id in range(n_profiles):
        profile = generate_synthetic_profile(profile_length)

        # Convert to long format
        for depth_idx, vs_value in enumerate(profile):
            profiles.append(
                {
                    "velocity_metadata_id": profile_id,
                    "depth": float(depth_idx),
                    "vs_value": float(vs_value),
                }
            )

    df = pd.DataFrame(profiles)
    return df


def main():
    """Main execution function."""
    # Set up paths
    project_root = Path(__file__).parent.parent
    output_path = project_root / "data" / "synthetic_toy_profiles.parquet"

    # Generate synthetic profiles
    print("Generating synthetic toy profiles...")
    print("  - 10,000 profiles")
    print("  - 500 points each")
    print("  - 2-layer structure")

    df = generate_synthetic_dataset(n_profiles=10000, profile_length=500)

    # Save to parquet
    output_path.parent.mkdir(exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(f"\n✅ Saved synthetic profiles to {output_path}")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}")
    print(f"   Unique profiles: {df['velocity_metadata_id'].nunique()}")

    # Print statistics
    print("\nStatistics:")
    print(df.describe())


if __name__ == "__main__":
    main()
