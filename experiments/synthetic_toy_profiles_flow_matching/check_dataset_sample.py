#!/usr/bin/env python3
"""Check a sample from the travel time dataset before training.

This script loads the dataset and displays information about a sample profile
to verify the data is loaded correctly.
"""

import sys
from pathlib import Path

import matplotlib

# Set non-interactive backend before importing pyplot (for headless environments/WSL)
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.synthetic_toy_profiles_flow_matching import config, data  # noqa: E402


def check_dataset_sample():
    """Load and inspect a sample from the dataset."""
    print("=" * 80)
    print("Checking Travel Time Dataset Sample")
    print("=" * 80)

    # Create dataloader
    print("\nLoading dataset...")
    print(f"  Parquet path: {config.cfg.parquet_path}")
    print(f"  Feature column: {config.cfg.feature_column}")
    print(f"  Group column: {config.cfg.group_column}")

    try:
        loader, max_length, dataset = data.create_dataloader(
            batch_size=4, num_workers=0, shuffle=False
        )
        print("\n✓ Dataset loaded successfully")
        print(f"  Dataset size: {len(dataset)} profiles")
        print(f"  Max length: {max_length}")
        print(f"  Number of batches: {len(loader)}")
    except Exception as e:
        print(f"\n✗ Error loading dataset: {e}")
        print("\nTroubleshooting:")
        print("  - Check if parquet file exists at the specified path")
        print("  - Verify the parquet file contains a 'travel_time' column")
        print("  - Check if the file structure matches expected format")
        return

    # Get a sample batch
    print("\n" + "-" * 80)
    print("Sample Batch Information")
    print("-" * 80)
    batch = next(iter(loader))
    print(f"  Batch shape: {batch.shape}")
    print(f"  Batch dtype: {batch.dtype}")
    print(
        f"  Batch range (normalized): [{batch.min().item():.3f}, {batch.max().item():.3f}]"
    )

    # Get a single sample profile
    print("\n" + "-" * 80)
    print("Sample Profile (First Profile in Batch)")
    print("-" * 80)

    sample_normalized = batch[0].squeeze().numpy()  # Remove channel dimension
    sample_denormalized = dataset.denormalize_batch(batch[0:1]).squeeze().numpy()

    print(f"  Normalized profile shape: {sample_normalized.shape}")
    print(f"  Denormalized profile shape: {sample_denormalized.shape}")
    print(
        f"  Normalized range: [{sample_normalized.min():.3f}, {sample_normalized.max():.3f}]"
    )
    print(
        f"  Denormalized range: [{sample_denormalized.min():.3f}, {sample_denormalized.max():.3f}]"
    )
    print(f"  Denormalized mean: {sample_denormalized.mean():.3f}")
    print(f"  Denormalized std: {sample_denormalized.std():.3f}")

    # Show first and last 10 values
    print(f"\n  First 10 values (denormalized): {sample_denormalized[:10]}")
    print(f"  Last 10 values (denormalized): {sample_denormalized[-10:]}")

    # Check for non-zero values (actual data vs padding)
    non_zero_mask = (
        sample_denormalized > 1e-6
    )  # Small threshold for numerical precision
    actual_length = non_zero_mask.sum()
    print(
        f"  Actual data points (non-zero): {actual_length} / {len(sample_denormalized)}"
    )

    # Plot the sample profile
    print("\n" + "-" * 80)
    print("Creating visualization...")
    print("-" * 80)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    depth_indices = np.arange(len(sample_denormalized))

    # Plot normalized profile (travel time vs depth)
    axes[0].plot(sample_normalized, depth_indices, linewidth=1.5, alpha=0.7)
    axes[0].set_xlabel("Normalized Travel Time")
    axes[0].set_ylabel("Depth")
    axes[0].set_title("Sample Profile (Normalized)")
    axes[0].grid(True, alpha=0.3)
    axes[0].invert_yaxis()  # Depth increases downward
    axes[0].set_ylim(len(sample_normalized) - 1, 0)  # Ensure depth starts at 0 at top
    axes[0].xaxis.set_label_position("top")
    axes[0].xaxis.tick_top()

    # Plot denormalized profile (travel time vs depth)
    axes[1].plot(
        sample_denormalized, depth_indices, linewidth=1.5, alpha=0.7, color="green"
    )
    axes[1].set_xlabel("Travel Time")
    axes[1].set_ylabel("Depth")
    axes[1].set_title("Sample Profile (Denormalized)")
    axes[1].grid(True, alpha=0.3)
    axes[1].invert_yaxis()  # Depth increases downward
    axes[1].set_ylim(len(sample_denormalized) - 1, 0)  # Ensure depth starts at 0 at top
    axes[1].xaxis.set_label_position("top")
    axes[1].xaxis.tick_top()

    plt.tight_layout()

    # Save plot
    output_path = (
        project_root
        / "outputs"
        / "synthetic_toy_profiles_flow_matching"
        / "plots"
        / "dataset_sample_check.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved plot to {output_path}")

    # Statistics on multiple profiles
    print("\n" + "-" * 80)
    print("Statistics on First 10 Profiles")
    print("-" * 80)

    all_profiles = []
    for i in range(min(10, len(dataset))):
        profile = dataset.denormalize_batch(dataset[i].unsqueeze(0)).squeeze().numpy()
        all_profiles.append(profile)

    all_profiles = np.array(all_profiles)
    print(f"  Profiles analyzed: {len(all_profiles)}")
    print(f"  Global min: {all_profiles.min():.3f}")
    print(f"  Global max: {all_profiles.max():.3f}")
    print(f"  Global mean: {all_profiles.mean():.3f}")
    print(f"  Global std: {all_profiles.std():.3f}")

    # Profile lengths (non-zero regions)
    profile_lengths = []
    for profile in all_profiles:
        non_zero = (profile > 1e-6).sum()
        profile_lengths.append(non_zero)

    print(f"  Average profile length (non-zero): {np.mean(profile_lengths):.1f}")
    print(f"  Min profile length: {np.min(profile_lengths)}")
    print(f"  Max profile length: {np.max(profile_lengths)}")

    print("\n" + "=" * 80)
    print("Dataset check completed successfully!")
    print("=" * 80)
    print("\nYou can now proceed with training the model.")


if __name__ == "__main__":
    check_dataset_sample()
