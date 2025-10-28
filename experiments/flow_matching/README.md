# Flow Matching for Soil Profile Generation

This module implements unguided Flow Matching (FFM) for generating Vs (shear wave velocity) profiles.

## Overview

Flow Matching is a generative modeling technique that learns a neural vector field to transport noise to realistic data. This implementation focuses on unguided flow matching without physics constraints.

**Normalization**: Data is normalized using z-score normalization (mean=0, std=1) which aligns with the Gaussian base distribution used during training and sampling. All outputs and metrics are reported in the original (unnormalized) Vs scale.

## Architecture

Two model architectures are supported:
- **UNet**: 1D U-Net with time conditioning and skip connections
- **FNO**: Fourier Neural Operator with spectral convolutions

## Key Files

- `config.py`: Configuration parameters for training, model, and data
- `train.py`: Training script for the flow matching model
- `sample.py`: Script to generate new profiles from trained model
- `evaluate_ffm.py`: Comprehensive evaluation and metrics
- `models.py`: Model architectures (UNet and FNO)
- `utils.py`: Utility functions for sampling, plotting, and metrics
- `data.py`: Data loading and preprocessing

## Usage

### Training

```bash
python -m experiments.flow_matching.train
```

With custom parameters:
```bash
python -m experiments.flow_matching.train --num_steps 20000 --tvd_weight 0.01
```

### Sampling

```bash
python -m experiments.flow_matching.sample --num_samples 100 --ode_steps 200
```

### Evaluation

```bash
python -m experiments.flow_matching.evaluate_ffm
```

## Configuration

Edit `config.py` to modify:
- Model type (unet/fno)
- Training parameters (learning rate, steps, batch size)
- ODE integration steps
- Data paths

## Implementation Details

### Flow Matching

The model learns a vector field `v_θ(u, t)` that transforms noise into realistic profiles:
- Training: MSE loss between predicted and target vector fields
- Sampling: ODE integration from t=0 to t=1 using Euler method

### Loss Function

```
L = MSE(v_predicted, v_target) + λ_TVD * TVD(v_predicted)
```

Where:
- `v_target = (u1 - ut) / (1 - t)` is the conditional flow matching target
- TVD (Total Variation Diminishing) encourages smoothness

### Metrics

- **Vs30**: Time-averaged shear wave velocity in top 30m
- **Vs100**: Velocity at ~100m depth
- **KS statistic**: Kolmogorov-Smirnov distance between distributions
- **Profile MSE/MAE**: Per-sample reconstruction errors

## Output

Trained models and results are saved to `outputs/flow_matching/`:
- `checkpoint_*.pt`: Model checkpoints
- `samples_*.npy`: Generated profiles
- `plots/*`: Visualizations and metrics

