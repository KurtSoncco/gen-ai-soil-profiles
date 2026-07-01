# Flow Matching for Vs Pairs Generation

This module implements unguided Flow Matching (FFM) for generating 4-dimensional Vs pairs [vs30, vs50, vs100, vs200] as a minimal experiment to observe flow matching feasibility without additional constraints.

## Overview

Flow Matching is a generative modeling technique that learns a neural vector field to transport noise to realistic data. This implementation focuses on **unguided flow matching** without physics constraints, operating on simplified 4D vectors extracted from full soil profiles.

**Normalization**: Data is transformed with log1p (y = log1p(Vs)) and then z-score normalized (mean=0, std=1), aligning with the Gaussian base distribution used during training and sampling. All outputs and metrics are denormalized back to Vs using expm1.

## Architecture

Two model architectures are supported:
- **MLP**: Multi-Layer Perceptron with time conditioning for 4D vector generation
- **FNO**: Fourier Neural Operator adapted for 4D vectors using spectral convolutions

## Key Files

- `config.py`: Configuration parameters for training, model, and data
- `train.py`: Training script for the flow matching model
- `sample.py`: Script to generate new vs pairs from trained model
- `models.py`: Model architectures (MLP and FNO) for vector field modeling
- `test_evaluation.py`: Comprehensive evaluation script with metrics and plots
- `utils.py`: Utility functions for sampling, plotting, and metrics
- `data.py`: Data loading and preprocessing

## Data Preparation

First, extract vs pairs from the full profiles:

```bash
python scripts/extract_vs_pairs.py
```

This creates `data/vs_pairs.parquet` with columns: [velocity_metadata_id, vs30, vs50, vs100, vs200]

## Usage

### Training

```bash
python -m experiments.vs_pairs_flow_matching.train
```

With custom parameters:
```bash
python -m experiments.vs_pairs_flow_matching.train --num_steps 10000

# Train with FNO architecture:
python -m experiments.vs_pairs_flow_matching.train --model_type fno --num_steps 10000
```

### Sampling

```bash
python -m experiments.vs_pairs_flow_matching.sample --num_samples 100 --ode_steps 50
```

## Configuration

Edit `config.py` to modify:
- Model architecture (mlp or fno)
- Training parameters (learning rate, steps, batch size)
- ODE integration steps
- Data paths

## Implementation Details

### Flow Matching

The model learns a vector field `v_θ(u, t)` that transforms noise into realistic vs pairs:
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

- **Distribution metrics**: KS statistics for each vs dimension
- **Statistical metrics**: Mean, std, min, max for each dimension
- **Scatter plots**: Pairwise relationships between vs30/vs50 and vs100/vs200
- **Histograms**: Per-dimension distribution comparisons

## Output

Trained models and results are saved to `outputs/vs_pairs_flow_matching/`:
- `checkpoint_*.pt`: Model checkpoints
- `samples_*.npy`: Generated vs pairs
- `plots/*`: Visualizations and metrics

## Experiment Purpose

This is a **minimal feasibility study** to:
1. Observe flow matching behavior on simple 4D vectors
2. Test unconstrained flow matching without physics losses
3. Compare MLP vs FNO architectures on vector data
4. Validate the training and sampling pipeline on simplified data
5. Establish baseline metrics before adding constraints

