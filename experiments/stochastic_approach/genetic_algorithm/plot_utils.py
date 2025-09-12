import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_palette("colorblind")


def plot_best_profile(best_profile: jnp.ndarray):
    """
    Plots the best velocity-time profile obtained from the genetic algorithm.

    Args:
        best_profile (jnp.ndarray): The best individual's chromosome representing
                                    depth and time pairs.

    Returns:
        None: Displays a plot of the velocity-time profile.
    """
    # 1. Extract depth and time from the chromosome
    depths = best_profile[::2]
    times = best_profile[1::2]

    # 2. Filter out the zero-padded values for a clean plot
    mask = depths > 0
    active_depths = depths[mask]
    active_times = times[mask]

    # 3. Add the origin point (0, 0) for a complete profile
    plot_depths = jnp.insert(active_depths, 0, 0)
    plot_times = jnp.insert(active_times, 0, 0)

    # 4. Compute the Vs values from depths and times
    vs_values = jnp.where(
        plot_times[1:] > 0, (2 * jnp.diff(plot_depths)) / jnp.diff(plot_times), 0
    )
    vs_values = jnp.insert(vs_values, 0, 0)  # Insert a zero for the top layer

    # 4. Create the plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(plot_times, plot_depths, marker="o", linestyle="-")
    ax[0].set_title("Depth vs. Two-Way Travel Time")
    ax[0].set_xlabel("Two-Way Travel Time (s)")
    ax[0].set_ylabel("Depth (m)")

    ax[1].step(vs_values, plot_depths, where="post", linestyle="-", marker="o")
    ax[1].set_title("Depth vs. Interval Velocity (Vs)")
    ax[1].set_xlabel("Interval Velocity (m/s)")
    ax[1].set_ylabel("Depth (m)")

    for a in ax:
        a.grid(True, which="both", linestyle="--", linewidth=0.5)
        a.set_ylim(bottom=0)  # Set the bottom limit to 0
        a.set_xlim(left=0)  # Set the left limit to 0
        # Xlabel and ticks to the top
        a.xaxis.set_label_position("top")
        a.xaxis.tick_top()
        a.invert_yaxis()  # Depth increases downwards

    plt.tight_layout()
    plt.show()
