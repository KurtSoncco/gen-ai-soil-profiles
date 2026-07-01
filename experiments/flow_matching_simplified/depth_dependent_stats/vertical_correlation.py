"""Compute vertical correlation of ln(Vs) at different depth lags.

Used to characterize the AR(1) or random-field structure.
Adapted to work with [TTS, depth] sequences from FlowMatchingDataset.
"""

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from .depth_statistics import reconstruct_vs_profile


def compute_vertical_autocorrelation(
    depths: np.ndarray,
    vs_values: np.ndarray,
    lags: List[float] = [1.0, 2.0, 5.0, 10.0],
) -> Dict[float, float]:
    """Compute autocorrelation of ln(Vs) at various depth lags.
    
    Args:
        depths: 1D array of depths (layer center depths)
        vs_values: 1D array of Vs (m/s)
        lags: List of depth lags (m) to compute correlation
    
    Returns:
        Dictionary: {lag: correlation_coefficient}
    """
    if len(vs_values) < 2:
        return {lag: 0.0 for lag in lags}
    
    ln_vs = np.log(vs_values)
    ln_vs_centered = ln_vs - np.mean(ln_vs)
    
    correlations = {}
    
    for lag in lags:
        # Find pairs of samples separated by approximately 'lag' depth
        valid_pairs = []
        
        for i in range(len(depths)):
            # Find nearest sample at depth + lag
            target_depth = depths[i] + lag
            
            # Find closest depth within tolerance
            diffs = np.abs(depths - target_depth)
            if diffs.min() < 0.5:  # Within 0.5 m tolerance
                j = diffs.argmin()
                if j != i:  # Don't pair with itself
                    valid_pairs.append((i, j))
        
        if len(valid_pairs) > 1:
            pairs_array = np.array(valid_pairs)
            vs_lag0 = ln_vs_centered[pairs_array[:, 0]]
            vs_lag1 = ln_vs_centered[pairs_array[:, 1]]
            
            # Compute correlation
            if len(vs_lag0) > 1 and np.std(vs_lag0) > 1e-10 and np.std(vs_lag1) > 1e-10:
                correlation = np.corrcoef(vs_lag0, vs_lag1)[0, 1]
                correlations[lag] = correlation if not np.isnan(correlation) else 0.0
            else:
                correlations[lag] = 0.0
        else:
            correlations[lag] = 0.0
    
    return correlations


def compute_profile_statistics(
    dataset,
    lags: List[float] = [1.0, 2.0, 5.0, 10.0],
) -> Dict:
    """Compute vertical correlation statistics for all profiles in dataset.
    
    Args:
        dataset: FlowMatchingDataset
        lags: List of depth lags to compute correlations
    
    Returns:
        Dictionary with:
        - lag_correlations: {lag: [corr1, corr2, ...]} (per-profile correlations)
        - mean_correlations: {lag: mean_corr}
        - std_correlations: {lag: std_corr}
    """
    lag_correlations = {lag: [] for lag in lags}
    
    print("Computing vertical correlations for all profiles...")
    for i in range(len(dataset)):
        if i % 1000 == 0:
            print(f"  Processing profile {i}/{len(dataset)}...")
        
        try:
            # Get normalized sequence
            seq_normalized = dataset.sequences[i]
            
            # Denormalize
            if dataset.normalize:
                seq_denorm = dataset.denormalize_sequence(seq_normalized, i)
            else:
                seq_denorm = seq_normalized
            
            # Reconstruct Vs profile
            depths_boundaries, vs_values = reconstruct_vs_profile(
                seq_denorm, apply_expm1=False
            )
            
            if len(vs_values) < 2:
                continue
            
            # Convert to layer center depths
            layer_depths = (depths_boundaries[:-1] + depths_boundaries[1:]) / 2.0
            
            # Compute correlations for this profile
            corrs = compute_vertical_autocorrelation(layer_depths, vs_values, lags)
            
            for lag in lags:
                lag_correlations[lag].append(corrs[lag])
        
        except Exception as e:
            if i < 10:  # Only print first few errors
                print(f"  Warning: Could not process profile {i}: {e}")
            continue
    
    # Compute statistics
    mean_correlations = {}
    std_correlations = {}
    
    for lag in lags:
        if lag_correlations[lag]:
            valid_corrs = [c for c in lag_correlations[lag] if not np.isnan(c)]
            if valid_corrs:
                mean_correlations[lag] = np.mean(valid_corrs)
                std_correlations[lag] = np.std(valid_corrs)
            else:
                mean_correlations[lag] = 0.0
                std_correlations[lag] = 0.0
        else:
            mean_correlations[lag] = 0.0
            std_correlations[lag] = 0.0
    
    return {
        "lag_correlations": lag_correlations,
        "mean_correlations": mean_correlations,
        "std_correlations": std_correlations,
    }


def plot_vertical_correlations(
    real_corrs: Dict,
    generated_corrs: Dict,
    output_path: str,
) -> None:
    """Plot vertical correlation structure: real vs generated.
    
    Args:
        real_corrs: Dictionary with mean_correlations and std_correlations
        generated_corrs: Dictionary with mean_correlations and std_correlations
        output_path: Path to save plot
    
    If real_corrs and generated_corrs are the same (baseline plot),
    only the real data is plotted.
    """
    lags = sorted(real_corrs["mean_correlations"].keys())
    
    # Check if this is a baseline plot (same data)
    is_baseline = real_corrs is generated_corrs or (
        real_corrs.get("mean_correlations", {}) == generated_corrs.get("mean_correlations", {})
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Mean correlation vs lag
    ax = axes[0]
    real_means = [real_corrs["mean_correlations"][lag] for lag in lags]
    real_stds = [real_corrs["std_correlations"][lag] for lag in lags]
    gen_means = [generated_corrs["mean_correlations"][lag] for lag in lags] if not is_baseline else []
    gen_stds = [generated_corrs["std_correlations"][lag] for lag in lags] if not is_baseline else []
    
    ax.errorbar(
        lags,
        real_means,
        yerr=real_stds,
        fmt="o-",
        label="Real" if not is_baseline else "Real Data",
        linewidth=2,
        markersize=8,
        color="blue",
        capsize=5,
    )
    if not is_baseline:
        ax.errorbar(
            lags,
            gen_means,
            yerr=gen_stds,
            fmt="s--",
            label="Generated",
            linewidth=2,
            markersize=6,
            color="red",
            capsize=5,
        )
    ax.set_xlabel("Depth Lag (m)", fontsize=12)
    ax.set_ylabel("Correlation of ln(Vs)", fontsize=12)
    ax.set_title("Vertical Autocorrelation of ln(Vs)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_xscale("log")
    
    # Plot 2: Distribution of correlations at lag=2m (or first available lag)
    ax = axes[1]
    if len(lags) > 0:
        # Use first lag that exists, prefer 2.0 if available
        plot_lag = 2.0 if 2.0 in lags else lags[0]
        lag_2m_real = real_corrs["lag_correlations"].get(plot_lag, [])
        lag_2m_gen = generated_corrs["lag_correlations"].get(plot_lag, [])
        
        # Filter out NaN values
        lag_2m_real = [c for c in lag_2m_real if not np.isnan(c)]
        lag_2m_gen = [c for c in lag_2m_gen if not np.isnan(c)]
        
        if len(lag_2m_real) > 0:
            ax.hist(
                lag_2m_real,
                bins=30,
                alpha=0.6,
                label="Real" if not is_baseline else "Real Data",
                color="blue",
                density=True,
            )
        if len(lag_2m_gen) > 0 and not is_baseline:
            ax.hist(
                lag_2m_gen,
                bins=30,
                alpha=0.6,
                label="Generated",
                color="red",
                density=True,
            )
        ax.set_xlabel(f"Correlation at lag={plot_lag}m", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(
            f"Distribution of Vertical Correlations (lag={plot_lag}m)",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved correlation plot to {output_path}")


# Usage example:
if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add project root to path
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

    try:
        from experiments.flow_matching_simplified.data import (
            FlowMatchingDataLoader,
            FlowMatchingDataset,
        )
        from experiments.flow_matching_simplified import config as cfg_mod
    except ImportError:
        # Fallback: try relative imports when running from the directory
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from data import FlowMatchingDataLoader, FlowMatchingDataset  # type: ignore
        import config as cfg_mod  # type: ignore

    # Load dataset
    cfg = cfg_mod.cfg
    data_loader = FlowMatchingDataLoader(data_path=Path(cfg.data_path))
    data_loader.load_data()

    assert data_loader.sequences is not None, "Sequences must be loaded"
    n_total = len(data_loader.sequences)

    # Create train/val/test splits
    all_indices = torch.randperm(n_total)
    n_train = int(cfg.train_val_test_split[0] * n_total)
    n_val = int(cfg.train_val_test_split[1] * n_total)

    train_indices = all_indices[:n_train].tolist()
    train_sequences = [data_loader.sequences[i] for i in train_indices]

    train_dataset = FlowMatchingDataset(
        train_sequences,
        max_length=cfg.max_length,
        pad_token=cfg.pad_token,
        normalize=cfg.normalize,
    )

    # Compute vertical correlations
    real_corrs = compute_profile_statistics(train_dataset, lags=[1.0, 2.0, 5.0, 10.0])

    print("\n=== Real Data Vertical Correlations ===")
    for lag in sorted(real_corrs["mean_correlations"].keys()):
        print(
            f"Lag {lag}m: mean={real_corrs['mean_correlations'][lag]:.4f}, "
            f"std={real_corrs['std_correlations'][lag]:.4f}"
        )

