"""Utility functions for depth-dependent statistics computation."""

import pickle
from typing import Dict, List, Tuple

import numpy as np
import torch

from .depth_statistics import reconstruct_vs_profile


def prepend_origin(sequence: np.ndarray) -> np.ndarray:
    """Prepend (0, 0) origin point to a sequence for visualization."""
    origin = np.array([[0.0, 0.0]], dtype=sequence.dtype)
    return np.vstack([origin, sequence])


def sequence_to_vs_profile(
    sequence: np.ndarray,
    apply_expm1: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert [TTS, depth] sequence to Vs profile.
    
    Wrapper around reconstruct_vs_profile for consistency.
    
    Args:
        sequence: (n, 2) array with [TTS, depth] pairs
        apply_expm1: Whether to apply expm1 (if TTS was log1p normalized)
    
    Returns:
        depths: Depth values (m) - layer boundaries
        vs_values: Shear wave velocity (m/s) - per layer
    """
    return reconstruct_vs_profile(sequence, apply_expm1=apply_expm1)


def denormalize_batch_sequences(
    sequences: torch.Tensor,
    attention_mask: torch.Tensor,
    sequence_stats: torch.Tensor,
    normalize: bool = True,
) -> List[np.ndarray]:
    """Denormalize batched sequences using sequence statistics.
    
    Args:
        sequences: (batch_size, max_length, 2) tensor with normalized [TTS, depth] pairs
        attention_mask: (batch_size, max_length) boolean mask
        sequence_stats: (batch_size, 4) tensor with [ts_mean, ts_std, depth_mean, depth_std]
        normalize: Whether sequences are normalized
    
    Returns:
        List of denormalized sequences (numpy arrays), one per batch item
    """
    batch_size = sequences.shape[0]
    denormalized_sequences = []
    
    for b in range(batch_size):
        # Extract valid sequence (where attention_mask is True)
        valid_mask_b = attention_mask[b].cpu().numpy()
        if valid_mask_b.sum() == 0:
            denormalized_sequences.append(np.array([]).reshape(0, 2))
            continue
        
        seq_normalized = sequences[b, valid_mask_b, :].cpu().numpy()  # (n_valid, 2)
        stats = sequence_stats[b].cpu().numpy()  # (4,)
        
        if normalize:
            # Denormalize: x_denorm = x_norm * std + mean
            ts_denorm = seq_normalized[:, 0] * stats[1] + stats[0]
            depth_denorm = seq_normalized[:, 1] * stats[3] + stats[2]
            seq_denorm = np.column_stack([ts_denorm, depth_denorm])
        else:
            seq_denorm = seq_normalized
        
        denormalized_sequences.append(seq_denorm)
    
    return denormalized_sequences


def load_target_statistics(
    stats_path: str,
    corr_path: str = None,
) -> Tuple[Dict, Dict]:
    """Load pre-computed target statistics from files.
    
    Args:
        stats_path: Path to real_depth_stats.pkl file (or .npy for backward compatibility)
        corr_path: Optional path to real_correlations.pkl file (or .npy for backward compatibility)
    
    Returns:
        Tuple of (depth_stats_dict, correlations_dict)
    """
    # Try pickle first, fall back to numpy for backward compatibility
    if stats_path.endswith('.pkl'):
        with open(stats_path, "rb") as f:
            depth_stats = pickle.load(f)
    else:
        depth_stats = np.load(stats_path, allow_pickle=True).item()
    
    if corr_path is not None:
        if corr_path.endswith('.pkl'):
            with open(corr_path, "rb") as f:
                correlations = pickle.load(f)
        else:
            correlations = np.load(corr_path, allow_pickle=True).item()
    else:
        # Return empty dict if not provided
        correlations = {
            "mean_correlations": {},
            "std_correlations": {},
            "lag_correlations": {},
        }
    
    return depth_stats, correlations


def compute_batch_vs_profiles(
    sequences: torch.Tensor,
    attention_mask: torch.Tensor,
    sequence_stats: torch.Tensor,
    normalize: bool = True,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Convert batched sequences to Vs profiles.
    
    Args:
        sequences: (batch_size, max_length, 2) tensor with [TTS, depth] pairs
        attention_mask: (batch_size, max_length) boolean mask
        sequence_stats: (batch_size, 4) tensor with normalization stats
        normalize: Whether sequences are normalized
    
    Returns:
        Tuple of (list of depths arrays, list of vs_values arrays)
    """
    denorm_sequences = denormalize_batch_sequences(
        sequences, attention_mask, sequence_stats, normalize
    )
    
    all_depths = []
    all_vs = []
    
    for seq in denorm_sequences:
        if len(seq) == 0:
            all_depths.append(np.array([]))
            all_vs.append(np.array([]))
            continue
        
        depths, vs_values = sequence_to_vs_profile(seq, apply_expm1=False)
        all_depths.append(depths)
        all_vs.append(vs_values)
    
    return all_depths, all_vs

