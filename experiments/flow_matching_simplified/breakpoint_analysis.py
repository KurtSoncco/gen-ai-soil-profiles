# In this code, we'll load the original data and extract the breakpoints, using the following strategy:
#  Robust breakpoint detection using iterative linear fitting.


import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_TTS_PATH = PROJECT_ROOT / "data" / "vspdb_tts_profiles.parquet"
DEFAULT_BREAKPOINTS_PATH = Path(__file__).resolve().parent / "data" / "breakpoints.parquet"
README_FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures" / "flow_matching"


def load_data(file_path: Path) -> dict[str, pd.DataFrame]:
    """Load the data"""
    df = pd.read_parquet(file_path)

    # join by velocity_metadata_id in different columns
    data_dict = {}

    for metadata_id, group in df.groupby("velocity_metadata_id"):
        data_dict[metadata_id] = group[["depth", "tts"]].copy().sort_values("depth")  # type: ignore

    return data_dict


def extract_breakpoints(
    profile_df: pd.DataFrame, min_segment_length: int = 2, max_error: float = 5e-4
) -> np.ndarray:
    """Extract the breakpoints from a profile
    using iterative linear fitting.

    Args:
        profile_df columns: depth, tts
    Returns:
        breakpoints: np.ndarray of shape (n_breakpoints, 2)
    """

    # Get the profile data
    profile_subset = profile_df[["depth", "tts"]].copy().reset_index(drop=True)  # type: ignore
    profile = profile_subset.to_numpy()
    depths = profile[:, 0]
    tts = profile[:, 1]

    # Handle edge cases
    if len(profile) == 0:
        return np.array([]).reshape(0, 2)
    if len(profile) == 1:
        return profile

    # Normalize for comparison
    tts_norm = (tts - np.min(tts)) / (
        np.max(tts) - np.min(tts) + 1e-10
    )  # Add small epsilon to avoid division by zero

    # Always start with the initial point (index 0)
    breakpoints_idx = [0]
    current_start = 0

    while current_start < len(tts) - 1:
        # Need at least min_segment_length points to fit a line
        if current_start + min_segment_length >= len(tts):
            # Not enough points left, include the last point and break
            if breakpoints_idx[-1] != len(tts) - 1:
                breakpoints_idx.append(len(tts) - 1)
            break

        # Start by validating the minimum segment length
        best_end = current_start + min_segment_length - 1
        segment_valid = False

        # Try to extend segment as far as possible
        for end in range(current_start + min_segment_length, len(tts)):
            segment_depths = depths[current_start : end + 1]
            segment_tts = tts_norm[current_start : end + 1]

            # Fit line to this segment
            A = np.column_stack([segment_depths, np.ones(len(segment_depths))])
            coeffs, residuals, rank, s = np.linalg.lstsq(A, segment_tts, rcond=None)
            m, b = coeffs

            # Compute error
            predicted = m * segment_depths + b
            error = np.mean(np.abs(segment_tts - predicted))

            if error < max_error:
                # Still linear, extend
                best_end = end
                segment_valid = True
            else:
                # Error exceeded, stop extending
                break

        # If no valid segment found, use minimum segment length
        if not segment_valid:
            best_end = current_start + min_segment_length - 1

        # Found a linear segment
        # Ensure best_end doesn't exceed array bounds
        best_end = min(best_end, len(tts) - 1)

        # Only add if it's different from the last breakpoint
        if best_end != breakpoints_idx[-1]:
            breakpoints_idx.append(best_end)

        current_start = best_end

        # Prevent infinite loop
        if current_start == breakpoints_idx[-1] and current_start < len(tts) - 1:
            current_start += 1

    # Ensure last point is included
    if breakpoints_idx[-1] != len(tts) - 1:
        breakpoints_idx.append(len(tts) - 1)

    # Get unique breakpoints and ensure all indices are valid
    unique_indices = np.unique(breakpoints_idx)
    valid_mask = (unique_indices >= 0) & (unique_indices < len(profile))
    return profile[unique_indices[valid_mask]]


def plot_breakpoints(
    profiles_dict: dict[str, pd.DataFrame],
    num_profiles: int = 10,
    output_paths: list[Path] | None = None,
    seed: int = 42,
) -> None:
    """Plot original vs reconstructed for visual inspection.

    Args:
        profiles_dict: Dictionary mapping profile IDs to DataFrames with 'depth' and 'tts' columns
        num_profiles: Number of random profiles to plot
        output_paths: Figure destinations. Defaults to the experiment figure dir
            and the README path ``outputs/figures/flow_matching/breakpoints.png``.
        seed: RNG seed for profile selection.
    """

    rng = np.random.default_rng(seed)

    # Get all profile IDs
    profile_ids = np.array(list(profiles_dict.keys()))

    # Select random profiles
    if len(profile_ids) < num_profiles:
        num_profiles = len(profile_ids)
    selected_ids = rng.choice(profile_ids, num_profiles, replace=False)

    # Select colors from seabron colorblind palette
    colors = sns.color_palette("colorblind", num_profiles)
    colors = [colors[i] for i in range(num_profiles)]

    plt.figure(figsize=(10, 10))

    # Plot each selected profile
    for i, profile_id in enumerate(selected_ids):
        profile_data = profiles_dict[profile_id]
        breakpoints = extract_breakpoints(profile_data)
        plt.plot(
            breakpoints[:, 1],
            breakpoints[:, 0],
            "x-",
            markersize=10,
            color=colors[i],
            label=f"Breakpoints {profile_id}",
        )
        plt.plot(
            profile_data["tts"],
            profile_data["depth"],
            "o--",
            alpha=0.5,
            markersize=2,
            color=colors[i],
            label=f"Original {profile_id}",
        )

    plt.title(f"Breakpoints for {num_profiles} random profiles")
    plt.xlabel("TTS")
    plt.ylabel("Depth")
    plt.legend(loc="best", fontsize="x-small")
    plt.ylim(bottom=0)
    plt.xlim(left=0)

    # Make xlabel and xticks to the top
    plt.gca().xaxis.set_label_position("top")
    plt.gca().xaxis.tick_top()
    plt.gca().invert_yaxis()

    if output_paths is None:
        output_paths = [
            Path(__file__).parent / "outputs" / "figures" / "breakpoints.png",
            README_FIGURE_DIR / "breakpoints.png",
        ]

    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=300)
        print(f"Saved breakpoint overlay to {path}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract piecewise-linear breakpoints from VSPDB TTS profiles"
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_TTS_PATH,
        help="TTS parquet with columns velocity_metadata_id, depth, tts "
        "(default: data/vspdb_tts_profiles.parquet)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_BREAKPOINTS_PATH,
        help="Destination breakpoint parquet (default: experiments/flow_matching_simplified/data/breakpoints.parquet)",
    )
    parser.add_argument(
        "--num-plot-profiles",
        type=int,
        default=10,
        help="Number of random profiles to overlay in the breakpoint figure",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for overlay profile selection",
    )
    args = parser.parse_args()

    if not args.data_path.exists():
        raise FileNotFoundError(
            f"TTS parquet not found: {args.data_path}. "
            "Run scripts/vspdb_request_data.py then scripts/build_vspdb_tts.py first."
        )

    original_data = load_data(args.data_path)

    print(f"Number of different profiles: {len(original_data)}")

    breakpoints = {
        profile_id: extract_breakpoints(profile)
        for profile_id, profile in original_data.items()
    }

    lengths = [len(breakpoints[profile_id]) for profile_id in breakpoints]
    print(f"Number of profiles with breakpoints: {len(breakpoints)}")
    print(f" Average number of breakpoints per profile: {np.mean(lengths)}")
    print(f" Std number of breakpoints per profile: {np.std(lengths)}")
    print(f" Min number of breakpoints per profile: {np.min(lengths)}")
    print(f" Max number of breakpoints per profile: {np.max(lengths)}")
    print(
        f" Number of profiles with no breakpoints: {sum(n == 0 for n in lengths)}"
    )
    print(
        f" Number of profiles with one breakpoint: {sum(n == 1 for n in lengths)}"
    )
    print(
        f" Number of profiles with two breakpoints: {sum(n == 2 for n in lengths)}"
    )
    print(
        f" Number of profiles with three breakpoints: {sum(n == 3 for n in lengths)}"
    )

    all_tts = np.concatenate(
        [df["tts"].to_numpy() for df in original_data.values() if len(df)]
    )
    print(
        f" TTS range (seconds): [{float(all_tts.min()):.6f}, {float(all_tts.max()):.6f}]"
    )
    if float(all_tts.max()) < 0.5:
        print(
            " WARNING: TTS max < 0.5 s; source parquet may still be log1p rather than raw seconds."
        )

    breakpoints_list = []
    for profile_id, breakpoint_array in breakpoints.items():
        if len(breakpoint_array) > 0:
            for breakpoint in breakpoint_array:
                breakpoints_list.append(
                    {
                        "profile_id": profile_id,
                        "depth": breakpoint[0],
                        "tts": breakpoint[1],
                    }
                )

    breakpoints_df = pd.DataFrame(breakpoints_list)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    breakpoints_df.to_parquet(args.output, index=False)
    print(f"Breakpoints saved to {args.output}")

    plot_breakpoints(
        original_data,
        args.num_plot_profiles,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
