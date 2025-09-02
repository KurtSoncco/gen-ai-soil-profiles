from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import lognorm, rv_continuous, uniform

# Use the seaborn 'colorblind' palette for accessibility
sns.set_palette("colorblind")

# --- Function Definitions ---


def generate_soil_profile(
    depths: np.ndarray,
    param_a: rv_continuous,
    param_b: rv_continuous,
    param_c: rv_continuous,
    noise_strength: float = 0.05,
) -> np.ndarray:
    """
    Generates a single shear wave velocity (Vs) profile.

    Args:
        depths (np.ndarray): An array of depths for the profile.
        param_a (lognorm): Log-normal distribution for the 'a' parameter.
        param_b (lognorm): Log-normal distribution for the 'b' parameter.
        param_c (lognorm): Log-normal distribution for the 'c' parameter.
        noise_strength (float): The strength of the multiplicative noise term.

    Returns:
        np.ndarray: An array of Vs values for the generated profile.
    """
    # Sample from the defined parameter distributions
    a = param_a.rvs()
    b = param_b.rvs()
    c = param_c.rvs()

    # Calculate the Vs profile based on the power-law model
    vs_profile = a * (depths**b) + c

    # Add a more realistic, multiplicative noise term.
    # This simulates small, random variations within the soil layers.
    noise = 1 + np.random.normal(loc=0, scale=noise_strength, size=depths.size)
    vs_profile *= noise

    # Ensure Vs increases monotonically with depth, which is physically realistic.
    # We use cumulative maximum to achieve this.
    return np.maximum.accumulate(vs_profile)


def run_monte_carlo_simulation(
    num_realizations: int, max_depth: int, num_points: int, noise_strength: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs a Monte Carlo simulation to generate multiple soil profiles and GWT depths.

    Args:
        num_realizations (int): Number of profiles to generate.
        max_depth (int): Maximum depth of the profiles.
        num_points (int): Number of discretization points.
        noise_strength (float): Strength of the multiplicative noise.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - all_vs_profiles (np.ndarray): Array of all generated Vs profiles.
            - all_gwt_depths (np.ndarray): Array of all generated GWT depths.
            - depths (np.ndarray): The depth array used for the profiles.
    """
    # Create the depth array once
    depths = np.linspace(0, max_depth, num_points)

    # Define distributions locally or pass as arguments
    param_a_dist = lognorm(s=0.5, scale=50)
    param_b_dist = lognorm(s=0.3, scale=0.5)
    param_c_dist = lognorm(s=0.2, scale=100)
    gwt_dist = uniform(loc=2, scale=13)

    # Use NumPy's vectorization for efficiency.
    # Instead of a loop, generate all random variables at once.
    a_params = param_a_dist.rvs(size=num_realizations)
    b_params = param_b_dist.rvs(size=num_realizations)
    c_params = param_c_dist.rvs(size=num_realizations)
    all_gwt_depths = gwt_dist.rvs(size=num_realizations)

    # Pre-allocate array for all Vs profiles
    all_vs_profiles = np.zeros((num_realizations, num_points))

    # Calculate all profiles in a single vectorized operation
    for i in range(num_realizations):
        vs_profile = a_params[i] * (depths ** b_params[i]) + c_params[i]

        # Apply the more realistic multiplicative noise
        noise = 1 + np.random.normal(0, noise_strength, size=num_points)
        vs_profile *= noise

        # Ensure monotonicity
        all_vs_profiles[i] = np.maximum.accumulate(vs_profile)

    return all_vs_profiles, all_gwt_depths, depths


def plot_results(
    vs_profiles: np.ndarray,
    gwt_depths: np.ndarray,
    depths: np.ndarray,
    num_realizations: int,
) -> None:
    """
    Creates and displays the plots for the simulation results.
    """
    # Plotting all Vs profiles, mean, and standard deviation
    plt.figure(figsize=(10, 8))

    # Plot a few random profiles
    for i in np.random.choice(num_realizations, 10, replace=False):
        plt.plot(vs_profiles[i], depths, alpha=0.5, color="gray", zorder=1)

    # Plot the mean and +/- one standard deviation
    vs_mean = np.mean(vs_profiles, axis=0)
    vs_std = np.std(vs_profiles, axis=0)
    plt.plot(
        vs_mean, depths, color="red", linewidth=3, label="Mean $V_s$ Profile", zorder=2
    )
    plt.fill_betweenx(
        depths,
        vs_mean - vs_std,
        vs_mean + vs_std,
        color="red",
        alpha=0.2,
        label=r"Mean $\pm$ 1 Std Dev",
        zorder=0,
    )
    plt.xlabel("Shear Wave Velocity, $V_s$ (m/s)")
    plt.ylabel("Depth (m)")
    plt.title(
        f"Stochastic Shear Wave Velocity Profiles ({num_realizations} Realizations)"
    )
    plt.ylim(depths[-1], 0)  # Invert y-axis to represent depth
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    # Plotting the GWT distribution
    plt.figure(figsize=(8, 6))
    plt.hist(
        gwt_depths,
        bins=30,
        density=True,
        alpha=0.7,
        color="skyblue",
        label="Generated GWT Depths",
    )
    plt.xlabel("Groundwater Table Depth (m)")
    plt.ylabel("Probability Density")
    plt.title("Distribution of Generated Groundwater Table Depths")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


# --- Main simulation ---
if __name__ == "__main__":
    NUM_REALIZATIONS = 2000  # Increased for better statistics
    MAX_DEPTH = 50
    NUM_POINTS = 101

    print(f"Generating {NUM_REALIZATIONS} soil profiles...")
    vs_profiles, gwt_depths, depths = run_monte_carlo_simulation(
        NUM_REALIZATIONS, MAX_DEPTH, NUM_POINTS
    )
    print("Generation complete!")

    # --- Visualization ---
    plot_results(vs_profiles, gwt_depths, depths, NUM_REALIZATIONS)

    # --- Additional Plotting: Distribution at a Specific Depth ---
    # This new plot helps visualize how uncertainty grows with depth
    target_depth_idx = np.abs(
        depths - 30
    ).argmin()  # Find index of depth closest to 30m
    target_depth = depths[target_depth_idx]

    plt.figure(figsize=(8, 6))
    sns.kdeplot(
        vs_profiles[:, target_depth_idx],
        fill=True,
        color="purple",
        label=f"Vs at {target_depth}m",
    )
    plt.axvline(
        x=float(np.mean(vs_profiles[:, target_depth_idx])),
        color="black",
        linestyle="--",
        label="Mean",
    )
    plt.xlabel("Shear Wave Velocity, $V_s$ (m/s)")
    plt.ylabel("Density")
    plt.title(f"Probability Distribution of $V_s$ at {target_depth}m Depth")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()
