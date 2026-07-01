"""Loss functions for depth-dependent statistics and vertical correlation matching."""

from typing import Dict, List, Optional

import numpy as np
import torch

from .depth_statistics import bin_by_depth
from .utils import compute_batch_vs_profiles
from .vertical_correlation import compute_vertical_autocorrelation


def compute_depth_statistics_loss(
    generated_sequences: torch.Tensor,
    generated_masks: torch.Tensor,
    sequence_stats: torch.Tensor,
    target_mean_ln_vs: torch.Tensor,  # (n_bins,)
    target_std_ln_vs: torch.Tensor,  # (n_bins,)
    bin_edges: np.ndarray,
    normalize: bool = True,
    weight: float = 0.01,
    min_bin_count: int = 50,
    mean_weight_ratio: float = 2.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Loss that encourages generated profiles to match target depth-dependent statistics.
    
    Args:
        generated_sequences: (Batch, max_length, 2) tensor with [TTS, depth] pairs
        generated_masks: (Batch, max_length) boolean attention mask
        sequence_stats: (Batch, 4) tensor with [ts_mean, ts_std, depth_mean, depth_std]
        target_mean_ln_vs: Expected mean ln(Vs) per depth bin (n_bins,)
        target_std_ln_vs: Expected std ln(Vs) per depth bin (n_bins,)
        bin_edges: Depth bin edges (n_bins+1,)
        normalize: Whether sequences are normalized
        weight: Loss weight
        min_bin_count: Minimum number of samples per bin to include in loss (default: 50)
        mean_weight_ratio: Weight ratio for mean vs std loss (default: 2.0, meaning mean is weighted 2x more)
        device: PyTorch device
    
    Returns:
        Scalar loss value
    """
    if device is None:
        device = generated_sequences.device
    
    batch_size = generated_sequences.shape[0]
    
    # Convert sequences to Vs profiles
    all_depths, all_vs = compute_batch_vs_profiles(
        generated_sequences, generated_masks, sequence_stats, normalize
    )
    
    # Collect all depth-Vs pairs for binning
    all_depths_flat = []
    all_vs_flat = []
    
    for depths, vs_values in zip(all_depths, all_vs):
        if len(vs_values) > 0 and len(depths) > 1:
            # Convert to layer center depths
            layer_depths = (depths[:-1] + depths[1:]) / 2.0
            all_depths_flat.extend(layer_depths)
            all_vs_flat.extend(vs_values)
    
    if len(all_vs_flat) == 0:
        # No valid data, return zero loss
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    all_depths_flat = np.array(all_depths_flat)
    all_vs_flat = np.array(all_vs_flat)
    
    # Bin by depth and compute statistics
    bin_centers, mean_ln_vs_gen, std_ln_vs_gen, bin_counts = bin_by_depth(
        all_depths_flat, all_vs_flat, bin_edges, apply_log=True
    )
    
    # Compare to targets (only for bins with sufficient data)
    # Filter by bin count to avoid chasing noisy targets
    valid_bins = (
        ~np.isnan(mean_ln_vs_gen)
        & ~np.isnan(std_ln_vs_gen)
        & ~np.isnan(target_mean_ln_vs.cpu().numpy())
        & ~np.isnan(target_std_ln_vs.cpu().numpy())
        & (bin_counts >= min_bin_count)  # Only bins with sufficient samples
    )
    
    if valid_bins.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Convert to tensors for gradient computation
    mean_gen_tensor = torch.from_numpy(mean_ln_vs_gen[valid_bins]).float().to(device)
    std_gen_tensor = torch.from_numpy(std_ln_vs_gen[valid_bins]).float().to(device)
    
    target_mean_valid = target_mean_ln_vs[valid_bins]
    target_std_valid = target_std_ln_vs[valid_bins]
    
    # Compute MSE losses with emphasis on mean term
    loss_mean = torch.mean((mean_gen_tensor - target_mean_valid) ** 2)
    loss_std = torch.mean((std_gen_tensor - target_std_valid) ** 2)
    
    # Weight mean more heavily than std to correct mean trend shape
    total_loss = (mean_weight_ratio * loss_mean + loss_std) * weight
    
    return total_loss


def compute_vertical_correlation_loss(
    generated_sequences: torch.Tensor,
    generated_masks: torch.Tensor,
    sequence_stats: torch.Tensor,
    target_correlations: Dict[float, float],  # {lag: expected_correlation}
    lags: List[float] = [1.0, 2.0, 5.0],
    normalize: bool = True,
    weight: float = 0.01,
    short_lag_weight: float = 2.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Loss that encourages generated profiles to match target vertical correlations.
    
    Focuses more on short lags (1-2m) where the gap is typically largest.
    
    Args:
        generated_sequences: (Batch, max_length, 2) tensor with [TTS, depth] pairs
        generated_masks: (Batch, max_length) boolean attention mask
        sequence_stats: (Batch, 4) tensor with normalization stats
        target_correlations: Expected correlations at each lag {lag: expected_corr}
        lags: Depth lags to compute correlations
        normalize: Whether sequences are normalized
        weight: Loss weight
        short_lag_weight: Extra weight multiplier for short lags (1-2m) (default: 2.0)
        device: PyTorch device
    
    Returns:
        Scalar loss value
    """
    if device is None:
        device = generated_sequences.device
    
    batch_size = generated_sequences.shape[0]
    
    # Convert sequences to Vs profiles
    all_depths, all_vs = compute_batch_vs_profiles(
        generated_sequences, generated_masks, sequence_stats, normalize
    )
    
    total_loss = 0.0
    total_weight = 0.0
    
    for depths, vs_values in zip(all_depths, all_vs):
        if len(vs_values) < 3:
            continue
        
        # Convert to layer center depths
        layer_depths = (depths[:-1] + depths[1:]) / 2.0
        
        # Compute correlations for this profile
        corrs = compute_vertical_autocorrelation(layer_depths, vs_values, lags)
        
        for lag in lags:
            target_corr = target_correlations.get(lag, 0.0)
            gen_corr = corrs.get(lag, 0.0)
            
            if not np.isnan(gen_corr):
                # Convert to tensor for gradient computation
                gen_corr_tensor = torch.tensor(gen_corr, device=device, dtype=torch.float32)
                target_corr_tensor = torch.tensor(target_corr, device=device, dtype=torch.float32)
                
                loss_corr = (gen_corr_tensor - target_corr_tensor) ** 2
                
                # Weight short lags (1-2m) more heavily since that's where the gap is largest
                lag_weight = short_lag_weight if lag <= 2.0 else 1.0
                total_loss = total_loss + lag_weight * loss_corr
                total_weight += lag_weight
    
    if total_weight == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    return (total_loss / total_weight) * weight

