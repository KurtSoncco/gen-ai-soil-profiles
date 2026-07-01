# Flow Matching for Synthetic Toy Profiles

This module implements unguided Flow Matching (FFM) for generating synthetic 2-layer toy profiles as a minimal experiment to observe flow matching feasibility on simple structured data.

## Overview

Flow Matching is a generative modeling technique that learns a neural vector field to transport noise to realistic data. This implementation focuses on **unguided flow matching** without physics constraints, operating on simple 2-layer synthetic profiles.

**Normalization**: Data is transformed with log1p (y = log1p(Vs)) and then z-score normalized (mean=0, std=1), aligning with the Gaussian base distribution used during training and sampling. All outputs are denormalized back to Vs using expm1.

## Architecture

Two model architectures are supported:
- **UNet**: 1D U-Net with time conditioning and skip connections (2.8M parameters)
- **FNO**: Fourier Neural Operator with spectral convolutions (353K parameters)

## Key Files

- `config.py`: Configuration parameters for training, model, and data
- `train.py`: Training script for the flow matching model
- `sample.py`: Script to generate new profiles from trained model
- `models.py`: Model architectures (UNet and FNO)
- `utils.py`: Utility functions for sampling, plotting, and metrics
- `data.py`: Data loading and preprocessing

## Data Preparation

First, generate synthetic toy profiles:

```bash
python scripts/generate_synthetic_profiles.py
```

This creates `data/synthetic_toy_profiles.parquet` with:
- 10,000 profiles
- Each profile: 500 points
- 2 layers per profile:
  - Layer 1 (v1): uniform(300, 800) m/s
  - Layer 2 (v2): uniform(900, 2000) m/s
  - Boundary depth: random integer between 100-400

## Usage

### Training

```bash
python -m experiments.synthetic_toy_profiles_flow_matching.train
```

With custom parameters:
```bash
# Train with UNet
python -m experiments.synthetic_toy_profiles_flow_matching.train --model_type unet --num_steps 10000

# Train with FNO
python -m experiments.synthetic_toy_profiles_flow_matching.train --model_type fno --num_steps 10000
```

### Sampling

```bash
python -m experiments.synthetic_toy_profiles_flow_matching.sample --num_samples 100 --ode_steps 50
```

## Configuration

Edit `config.py` to modify:
- Model architecture (unet or fno)
- Training parameters (learning rate, steps, batch size)
- ODE integration steps
- Data paths

## Implementation Details

### Flow Matching

The model learns a vector field `v_θ(u, t)` that transforms noise into realistic profiles:
- **Training**: MSE loss between predicted and target vector fields (unconstrained)
- **Sampling**: ODE integration from t=0 to t=1 using RK4 (Runge-Kutta 4)

### Loss Function

For this minimal unconstrained experiment:

```
L = MSE(v_predicted, v_target)
```

Where `v_target = (u1 - ut) / (1 - t)` is the conditional flow matching target.

No additional constraints (TVD, kinetic energy, physics losses) are applied.

### Metrics

- **Layer statistics**: Mean and std for layer 1, layer 2, and boundary depth
- **KS statistics**: Distribution similarity for each layer
- **Profile comparison**: Side-by-side visualization of real vs generated profiles

## Output

Trained models and results are saved to `outputs/synthetic_toy_profiles_flow_matching/`:
- `checkpoint_*.pt`: Model checkpoints
- `samples_*.npy`: Generated profiles
- `plots/*`: Visualizations and metrics

## Experiment Purpose

This is a **minimal feasibility study** to:
1. Observe flow matching behavior on simple 2-layer profiles
2. Test unconstrained flow matching without physics losses
3. Compare UNet vs FNO architectures on toy synthetic data
4. Validate the training and sampling pipeline before moving to real profiles
5. Establish baseline metrics and understanding of flow matching dynamics

