#!/usr/bin/env python3
"""Test script to verify data.py is working correctly."""

import sys
from pathlib import Path

import matplotlib

# Set non-interactive backend before importing pyplot (for headless environments/WSL)
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.synthetic_toy_profiles_flow_matching import config, data  # noqa: E402


@pytest.fixture(scope="module")
def dataset():
    """Load the synthetic toy dataset, or skip when the gitignored parquet is absent."""
    parquet_path = Path(config.cfg.parquet_path)
    if not parquet_path.exists():
        pytest.skip(
            f"Missing {parquet_path}; run scripts/generate_synthetic_profiles.py"
        )
    _, _, ds = data.create_dataloader(batch_size=4, num_workers=0, shuffle=False)
    return ds


def test_data_loading(dataset):
    """Test basic data loading."""
    print("=" * 80)
    print("Testing Data Loading")
    print("=" * 80)

    # Create dataloader
    loader, max_length, _ = data.create_dataloader(
        batch_size=4, num_workers=0, shuffle=False
    )

    print("\n✓ Dataloader created successfully")
    print(f"  Dataset size: {len(dataset)} profiles")
    print(f"  Max length: {max_length}")
    print(f"  Batches: {len(loader)}")

    # Test batch loading
    batch = next(iter(loader))
    print("\n✓ Batch loaded successfully")
    print(f"  Batch shape: {batch.shape}")
    print(f"  Batch dtype: {batch.dtype}")
    print(f"  Batch range: [{batch.min().item():.3f}, {batch.max().item():.3f}]")


def test_normalization(dataset):
    """Test normalization and denormalization."""
    print("\n" + "=" * 80)
    print("Testing Normalization/Denormalization")
    print("=" * 80)

    # Get a batch of real profiles
    batch = []
    for i in range(4):
        batch.append(dataset[i])
    batch = np.stack(batch)

    print("\n✓ Loaded 4 profiles")
    print(f"  Shape: {batch.shape}")

    # Get raw data for comparison
    df = pd.read_parquet(config.cfg.parquet_path)

    for i in range(4):
        # Get normalized profile from dataset
        profile_norm = dataset[i].numpy()[0, :]  # Remove channel dim

        # Get raw profile
        raw_profile = np.asarray(df[df["velocity_metadata_id"] == i]["vs_value"])

        # Denormalize
        profile_denorm_tensor = dataset.denormalize_batch(dataset[i].unsqueeze(0))
        profile_denorm = profile_denorm_tensor.cpu().numpy()[0, 0, :]

        print(f"\nProfile {i}:")
        print(
            f"  Raw unique values: {len(np.unique(raw_profile))} - {np.unique(raw_profile)}"
        )
        print(
            f"  Normalized range: [{profile_norm.min():.3f}, {profile_norm.max():.3f}]"
        )
        print(
            f"  Denormalized unique values: {len(np.unique(profile_denorm.round(1)))} - {np.unique(profile_denorm)}"
        )

        # Check if denormalization is correct
        if len(np.unique(profile_denorm.round(1))) == 2:
            print("  ✓ Correctly denormalized (2 layers preserved)")
        else:
            print("  ✗ Error in denormalization!")


def test_profile_structure(dataset):
    """Test that profiles maintain 2-layer structure."""
    print("\n" + "=" * 80)
    print("Testing Profile Structure")
    print("=" * 80)

    # Load raw data
    pd.read_parquet(config.cfg.parquet_path)

    layer1_values = []
    layer2_values = []
    boundary_depths = []

    for i in range(min(100, len(dataset))):
        # Denormalize profile
        profile = (
            dataset.denormalize_batch(dataset[i].unsqueeze(0)).cpu().numpy()[0, 0, :]
        )

        # Detect layers (same logic as compute_layer_statistics)
        diff = np.abs(np.diff(profile))
        if np.any(diff > 50):
            boundary_idx = np.argmax(diff) + 1
            layer1 = np.mean(profile[:boundary_idx])
            layer2 = np.mean(profile[boundary_idx:])
            layer1_values.append(layer1)
            layer2_values.append(layer2)
            boundary_depths.append(boundary_idx)
        else:
            # Fallback: assume middle split
            mid = len(profile) // 2
            layer1 = np.mean(profile[:mid])
            layer2 = np.mean(profile[mid:])
            layer1_values.append(layer1)
            layer2_values.append(layer2)
            boundary_depths.append(mid)

    print(f"\n✓ Analyzed {len(layer1_values)} profiles")
    print("\nLayer 1 (should be 300-800):")
    print(f"  Mean: {np.mean(layer1_values):.1f}")
    print(f"  Std: {np.std(layer1_values):.1f}")
    print(f"  Min: {np.min(layer1_values):.1f}")
    print(f"  Max: {np.max(layer1_values):.1f}")

    print("\nLayer 2 (should be 900-2000):")
    print(f"  Mean: {np.mean(layer2_values):.1f}")
    print(f"  Std: {np.std(layer2_values):.1f}")
    print(f"  Min: {np.min(layer2_values):.1f}")
    print(f"  Max: {np.max(layer2_values):.1f}")

    print("\nBoundary depth (should be 100-400):")
    print(f"  Mean: {np.mean(boundary_depths):.1f}")
    print(f"  Std: {np.std(boundary_depths):.1f}")
    print(f"  Min: {np.min(boundary_depths):.0f}")
    print(f"  Max: {np.max(boundary_depths):.0f}")

    # Check if values are in expected ranges
    layer1_in_range = np.all(
        (np.array(layer1_values) >= 300) & (np.array(layer1_values) <= 800)
    )
    layer2_in_range = np.all(
        (np.array(layer2_values) >= 900) & (np.array(layer2_values) <= 2000)
    )
    boundary_in_range = np.all(
        (np.array(boundary_depths) >= 100) & (np.array(boundary_depths) <= 400)
    )

    print("\n" + "=" * 80)
    print("Validation")
    print("=" * 80)
    print(f"Layer 1 in range [300, 800]: {'✓' if layer1_in_range else '✗'}")
    print(f"Layer 2 in range [900, 2000]: {'✓' if layer2_in_range else '✗'}")
    print(f"Boundary in range [100, 400]: {'✓' if boundary_in_range else '✗'}")

    if layer1_in_range and layer2_in_range and boundary_in_range:
        print("\n✓ All tests passed! Data processing is working correctly.")
    else:
        print("\n✗ Some tests failed! Check data generation.")


def test_plotting(dataset):
    """Test plotting with correct depth axis."""
    print("\n" + "=" * 80)
    print("Testing Profile Plotting")
    print("=" * 80)

    # Get a sample profile
    profile = dataset.denormalize_batch(dataset[0].unsqueeze(0)).cpu().numpy()[0, 0, :]
    depth = np.arange(len(profile))

    # Create plot with depth on y-axis (inverted)
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.plot(profile, depth, linewidth=2)
    ax.set_xlabel("Vs (m/s)")
    ax.set_ylabel("Depth Index")
    ax.set_title("Synthetic Toy Profile\n(2-layer structure with sharp boundary)")
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # Invert y-axis so depth increases downward

    # Save
    output_path = (
        Path(__file__).parent.parent.parent
        / "outputs"
        / "synthetic_toy_profiles_flow_matching"
        / "plots"
        / "test_profile_plot.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved test plot to {output_path}")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("SYNTHETIC TOY PROFILES DATA TEST")
    print("=" * 80)

    parquet_path = Path(config.cfg.parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Missing {parquet_path}; run scripts/generate_synthetic_profiles.py"
        )
    _, _, dataset = data.create_dataloader(batch_size=4, num_workers=0, shuffle=False)
    test_data_loading(dataset)
    test_normalization(dataset)
    test_profile_structure(dataset)
    test_plotting(dataset)

    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
