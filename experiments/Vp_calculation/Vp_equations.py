import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.base import Soil

sns.set_palette("colorblind")


def Nagashima_relations(Vs, depth, GWT_depth=4):
    if GWT_depth == 4:
        if depth <= 4:
            Vp_avg = 1.93 * Vs + 380.7
            Vp_std = -6.65 * 1e-4 * Vs + 1.79
        else:
            Vp_avg = 1.39 * Vs + 1189.4
            Vp_std = -6.63 * 1e-5 * Vs + 1.25

    else:
        if depth <= GWT_depth:
            Vp_avg = 1.06 * Vs + 436.72
            Vp_std = -9.51 * 1e-4 * Vs + 1.66
        else:
            Vp_avg = 1.38 * Vs + 1249.40
            Vp_std = -4.51 * 1e-5 * Vs + 1.19

    return Vp_avg, Vp_std


def main():
    # Create a comparison between the relationships using a Soil object
    soil = Soil(name_definition="Example Soil", properties={"Vs": 200, "depth": 5})

    # Create a plot to compare the relationships
    depths = np.linspace(0, 10, 100)
    Vp_avgs = [Nagashima_relations(soil.properties["Vs"], d)[0] for d in depths]
    Vp_stds = [Nagashima_relations(soil.properties["Vs"], d)[1] for d in depths]

    plt.figure(figsize=(10, 6))
    plt.plot(depths, Vp_avgs, label="Vp_avg", color="blue")
    plt.fill_between(
        depths,
        np.array(Vp_avgs) - np.array(Vp_stds),
        np.array(Vp_avgs) + np.array(Vp_stds),
        color="blue",
        alpha=0.2,
    )
    plt.title("Nagashima Relations")
    plt.xlabel("Depth (m)")
    plt.ylabel("Vp (m/s)")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
