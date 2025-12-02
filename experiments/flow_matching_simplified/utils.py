"""Utility functions for Flow Matching with Variable-Length Breakpoints.

This module provides:
- Sampling functions for generating sequences
- Metrics computation utilities
- Plotting and visualization utilities
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

try:
    from experiments.flow_matching_simplified.model import TransformerModel
except ImportError:
    from model import TransformerModel  # type: ignore


def sample_sequences_ode(
    model: TransformerModel,
    initial_noise: torch.Tensor,
    attention_mask: torch.Tensor,
    sequence_stats: torch.Tensor,
    ode_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Sample sequences using ODE integration (RK4 method).

    Args:
        model: Trained transformer model
        initial_noise: Initial noise tensor (batch_size, max_length, 2)
        attention_mask: Boolean mask (batch_size, max_length)
        sequence_stats: Sequence statistics (batch_size, 4)
        ode_steps: Number of ODE integration steps
        device: PyTorch device

    Returns:
        Generated sequences with same shape as initial_noise
    """
    model.eval()

    # Start with noise at t=0
    u = initial_noise.to(device)
    dt = 1.0 / ode_steps

    with torch.no_grad():
        for i in range(ode_steps):
            # Current time
            t_val = i * dt
            t = torch.full((u.shape[0], 1), t_val).to(device)

            # Predict vector field at current state
            k1 = model(u, attention_mask, t, sequence_stats) * dt

            # Second stage: midpoint
            k2_t = t_val + 0.5 * dt
            k2 = (
                model(
                    u + 0.5 * k1,
                    attention_mask,
                    torch.full((u.shape[0], 1), k2_t).to(device),
                    sequence_stats,
                )
                * dt
            )

            # Third stage: midpoint
            k3 = (
                model(
                    u + 0.5 * k2,
                    attention_mask,
                    torch.full((u.shape[0], 1), k2_t).to(device),
                    sequence_stats,
                )
                * dt
            )

            # Fourth stage: endpoint
            k4_t = t_val + dt
            k4 = (
                model(
                    u + k3,
                    attention_mask,
                    torch.full((u.shape[0], 1), k4_t).to(device),
                    sequence_stats,
                )
                * dt
            )

            # Combine: RK4 integration
            u = u + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

            # Keep padding masked
            u = u * attention_mask.unsqueeze(-1).float()

    return u


def compute_sequence_statistics(sequences: list[np.ndarray]) -> dict[str, float]:
    """
    Compute statistics for a list of sequences.

    Args:
        sequences: List of sequences, each of shape (n_tokens, 2)

    Returns:
        Dictionary with statistics
    """
    if not sequences:
        return {
            "mean_length": 0.0,
            "std_length": 0.0,
            "mean_ts": 0.0,
            "std_ts": 0.0,
            "mean_depth": 0.0,
            "std_depth": 0.0,
        }

    lengths = [len(seq) for seq in sequences]
    all_ts = np.concatenate([seq[:, 0] for seq in sequences])
    all_depth = np.concatenate([seq[:, 1] for seq in sequences])

    return {
        "mean_length": float(np.mean(lengths)),
        "std_length": float(np.std(lengths)),
        "min_length": int(np.min(lengths)),
        "max_length": int(np.max(lengths)),
        "mean_ts": float(np.mean(all_ts)),
        "std_ts": float(np.std(all_ts)),
        "mean_depth": float(np.mean(all_depth)),
        "std_depth": float(np.std(all_depth)),
    }


def compare_distributions(
    real_sequences: list[np.ndarray], generated_sequences: list[np.ndarray]
) -> dict[str, float]:
    """
    Compare distributions between real and generated sequences.

    Args:
        real_sequences: List of real sequences
        generated_sequences: List of generated sequences

    Returns:
        Dictionary with comparison metrics
    """
    real_stats = compute_sequence_statistics(real_sequences)
    gen_stats = compute_sequence_statistics(generated_sequences)

    return {
        "length_mean_diff": abs(real_stats["mean_length"] - gen_stats["mean_length"]),
        "length_std_diff": abs(real_stats["std_length"] - gen_stats["std_length"]),
        "ts_mean_diff": abs(real_stats["mean_ts"] - gen_stats["mean_ts"]),
        "ts_std_diff": abs(real_stats["std_ts"] - gen_stats["std_ts"]),
        "depth_mean_diff": abs(real_stats["mean_depth"] - gen_stats["mean_depth"]),
        "depth_std_diff": abs(real_stats["std_depth"] - gen_stats["std_depth"]),
        "real_mean_length": real_stats["mean_length"],
        "gen_mean_length": gen_stats["mean_length"],
        "real_mean_ts": real_stats["mean_ts"],
        "gen_mean_ts": gen_stats["mean_ts"],
        "real_mean_depth": real_stats["mean_depth"],
        "gen_mean_depth": gen_stats["mean_depth"],
    }


def compute_vs_from_sequence(sequence: np.ndarray) -> np.ndarray:
    """Compute Vs velocity values from TTS/depth sequence.

    Args:
        sequence: Array of shape (n, 2) with [TTS, depth] pairs

    Returns:
        Array of Vs values for each layer (n-1 values), or empty array if insufficient points
    """
    if len(sequence) < 2:
        return np.array([])

    # Prepend origin (0, 0) if not already present
    if len(sequence) > 0 and (sequence[0, 0] != 0.0 or sequence[0, 1] != 0.0):
        origin = np.array([[0.0, 0.0]], dtype=sequence.dtype)
        seq_with_origin = np.vstack([origin, sequence])
    else:
        seq_with_origin = sequence

    # Extract TTS and depth
    tts = seq_with_origin[:, 0]
    depths = seq_with_origin[:, 1]

    # Calculate layer thicknesses
    thicknesses = np.diff(depths)

    # Calculate TTS differences (incremental TTS for each layer)
    dtts = np.diff(tts)

    # Convert to Vs: Vs = dz / dt (NOT 2 * dz / dt)
    # Add small epsilon to avoid division by zero
    vs_values = thicknesses / (dtts + 1e-9)

    return vs_values


def check_vs_bounds(sequence: np.ndarray, vs_min: float, vs_max: float) -> bool:
    """Check if all Vs values are within specified bounds.

    Args:
        sequence: Array of shape (n, 2) with [TTS, depth] pairs
        vs_min: Minimum allowed Vs value (m/s)
        vs_max: Maximum allowed Vs value (m/s)

    Returns:
        True if all Vs values are within bounds, False otherwise
    """
    if len(sequence) < 2:
        return True

    vs_values = compute_vs_from_sequence(sequence)
    if len(vs_values) == 0:
        return True

    return bool(np.all((vs_values >= vs_min) & (vs_values <= vs_max)))


def check_min_dt(sequence: np.ndarray, min_dt: float) -> bool:
    """Check if all Δt (TTS increments) meet minimum threshold.

    This prevents infinite or huge Vs values.

    Args:
        sequence: Array of shape (n, 2) with [TTS, depth] pairs
        min_dt: Minimum allowed Δt value

    Returns:
        True if all Δt values are >= min_dt, False otherwise
    """
    if len(sequence) < 2:
        return True

    # Prepend origin if needed
    if len(sequence) > 0 and (sequence[0, 0] != 0.0 or sequence[0, 1] != 0.0):
        origin = np.array([[0.0, 0.0]], dtype=sequence.dtype)
        seq_with_origin = np.vstack([origin, sequence])
    else:
        seq_with_origin = sequence

    tts = seq_with_origin[:, 0]
    dtts = np.diff(tts)

    # Check that all Δt values are >= min_dt
    # Only check where there's a non-zero thickness (dz > 0)
    depths = seq_with_origin[:, 1]
    dz = np.diff(depths)
    # Only check dt where dz > 0 (non-zero thickness layers)
    valid_mask = dz > 1e-9
    if np.any(valid_mask):
        return bool(np.all(dtts[valid_mask] >= min_dt))
    return True


def compute_vs_penalty(
    sequences: torch.Tensor,
    attention_mask: torch.Tensor,
    sequence_stats: torch.Tensor,
    vs_min: float,
    vs_max: float,
    min_dt: float,
    normalize: bool = True,
) -> torch.Tensor:
    """Compute Vs penalty for training regularization.

    This function denormalizes sequences before computing Vs, ensuring gradients
    flow through the denormalization back to the normalized model outputs.

    Args:
        sequences: Tensor of shape (batch_size, max_length, 2) with [TTS, depth] pairs (normalized)
        attention_mask: Boolean mask (batch_size, max_length) - True for real tokens
        sequence_stats: Tensor of shape (batch_size, 4) with [ts_mean, ts_std, depth_mean, depth_std]
        vs_min: Minimum allowed Vs value (in physical units)
        vs_max: Maximum allowed Vs value (in physical units)
        min_dt: Minimum allowed Δt value (in physical units)
        normalize: Whether sequences are normalized (default: True)

    Returns:
        Scalar penalty value (higher = more violations)
    """
    batch_size, max_length, _ = sequences.shape
    device = sequences.device
    all_penalties = []

    # Extract stats components
    # sequence_stats shape: (batch_size, 4) = [ts_mean, ts_std, depth_mean, depth_std]
    ts_mean = sequence_stats[:, 0].unsqueeze(-1).unsqueeze(-1)  # (batch_size, 1, 1)
    ts_std = sequence_stats[:, 1].unsqueeze(-1).unsqueeze(-1)  # (batch_size, 1, 1)
    depth_mean = sequence_stats[:, 2].unsqueeze(-1).unsqueeze(-1)  # (batch_size, 1, 1)
    depth_std = sequence_stats[:, 3].unsqueeze(-1).unsqueeze(-1)  # (batch_size, 1, 1)

    # Denormalize sequences: x_denorm = x_norm * std + mean
    # sequences shape: (batch_size, max_length, 2)
    if normalize:
        sequences_denorm = sequences.clone()
        # Broadcast: (batch_size, 1, 1) * (batch_size, max_length, 1) -> (batch_size, max_length, 1)
        sequences_denorm[:, :, 0:1] = sequences[:, :, 0:1] * ts_std + ts_mean  # TTS
        sequences_denorm[:, :, 1:2] = (
            sequences[:, :, 1:2] * depth_std + depth_mean
        )  # depth
    else:
        sequences_denorm = sequences

    for b in range(batch_size):
        # Extract valid sequence (where attention_mask is True)
        valid_mask_b = attention_mask[b]  # (max_length,)
        if valid_mask_b.sum() < 2:
            continue

        # Get valid tokens (already denormalized)
        seq = sequences_denorm[b, valid_mask_b, :]  # (n_valid, 2)

        if seq.shape[0] == 0:
            continue

        # Prepend origin (0, 0) in physical units
        # Check if first token is already origin
        first_token = seq[0]
        # Use small epsilon for comparison to handle floating point
        is_origin = (torch.abs(first_token[0]) < 1e-6) & (
            torch.abs(first_token[1]) < 1e-6
        )

        if not is_origin:
            origin = torch.zeros(1, 2, device=device)
            seq_with_origin = torch.cat([origin, seq], dim=0)
        else:
            seq_with_origin = seq

        if seq_with_origin.shape[0] < 2:
            continue

        # Extract TTS and depth (now in physical units)
        tts = seq_with_origin[:, 0]  # (n+1,)
        depths = seq_with_origin[:, 1]  # (n+1,)

        # Compute thicknesses and TTS differences (all in PyTorch, physical units)
        dz = depths[1:] - depths[:-1]  # (n,)
        dt = tts[1:] - tts[:-1]  # (n,)

        # Only consider layers with non-zero thickness and positive dt
        # Ensure both endpoints are valid (dz > 0 and dt > 0)
        valid_mask = (dz > 1e-9) & (dt > 1e-9)

        if valid_mask.any():
            dz_valid = dz[valid_mask]
            dt_valid = dt[valid_mask]

            # Compute Vs values in physical units: Vs = dz / dt
            vs_values = dz_valid / (dt_valid + 1e-9)

            # Hard-clip Vs to cap gradients and prevent extreme values
            vs_clipped = vs_values.clamp(-2 * vs_max, 2 * vs_max)

            # Penalty for Vs outside bounds (dimensionless, normalized)
            # low: violation below vs_min, normalized by vs_min
            vs_penalty_low = F.relu(vs_min - vs_clipped) / vs_min
            # high: violation above vs_max, normalized by vs_max
            vs_penalty_high = F.relu(vs_clipped - vs_max) / vs_max

            # Penalty for dt too small (normalized by min_dt)
            dt_penalty = F.relu(min_dt - dt_valid) / min_dt

            # Mean penalty per violation (dimensionless)
            batch_penalty = (
                vs_penalty_low.mean()
                + vs_penalty_high.mean()
                + dt_penalty.mean()
            )
            all_penalties.append(batch_penalty)

    if len(all_penalties) > 0:
        return torch.stack(all_penalties).mean()  # Average penalty across batches
    else:
        return torch.zeros(1, device=device, dtype=sequences.dtype).squeeze()
