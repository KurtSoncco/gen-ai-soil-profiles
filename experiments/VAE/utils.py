from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def tts_to_Vs(tts, dz):
    """
    Convert a cumulative travel time profile to a layered Vs profile.
    """
    # Insert a column of zeros at the beginning of each profile
    tts_with_zero = np.insert(tts, 0, 0, axis=1)
    # Calculate the difference along the layers (axis=1)
    d_tts = np.diff(tts_with_zero, axis=1)
    # Add a small epsilon to avoid division by zero
    Vs = dz / (d_tts + 1e-9)
    return Vs


def plot_stair_profiles(vs_data, depths, title, max_depth, vs_max, num_to_plot=10):
    """Plots generated Vs profiles as stair plots."""
    plt.figure(figsize=(10, 8))

    # Plot a few generated profiles
    for i in range(num_to_plot):
        vs = vs_data[i]
        plt.step(vs, depths, where="post", color="gray", alpha=0.5)

    # Plot the mean and +/- 1 std dev
    vs_mean = np.mean(vs_data, axis=0)
    vs_std = np.std(vs_data, axis=0)
    plt.plot(vs_mean, depths, color="red", linewidth=3, label="Mean Profile")
    plt.fill_betweenx(
        depths,
        vs_mean - vs_std,
        vs_mean + vs_std,
        color="red",
        alpha=0.2,
        label="Mean +/- 1 Std Dev",
    )

    plt.xlabel("Shear Wave Velocity, $V_s$ (m/s)")
    plt.ylabel("Depth (m)")
    plt.title(title)
    plt.ylim(max_depth, 0)
    plt.xlim(0, vs_max)
    plt.grid(True)
    plt.legend()
    plt.savefig(
        Path(__file__).parent / "generated_vs_profiles.png", bbox_inches="tight"
    )
    plt.close()
