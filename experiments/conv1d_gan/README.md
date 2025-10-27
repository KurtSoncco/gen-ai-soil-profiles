# Conv1D GAN Experiment

This experiment implements a Conv1D Generative Adversarial Network (GAN) for generating Vs profiles, providing comprehensive evaluation and monitoring similar to the parametric and VAE experiments.

## Overview

The Conv1D GAN uses 1D convolutional layers to generate Vs profiles from random noise. It includes:
- **Generator**: Creates Vs profiles from latent noise using transposed convolutions
- **Discriminator**: Distinguishes between real and generated profiles
- **Comprehensive evaluation**: Metrics, plots, and statistical analysis
- **Wandb integration**: Real-time monitoring and logging

## Files Structure

```
conv1d_gan/
├── config.py              # Configuration parameters
├── data.py                # Data loading utilities
├── models.py              # Generator and Discriminator models
├── train.py               # Training script with wandb integration
├── evaluate_conv1d_gan.py # Comprehensive evaluation script
├── utils.py               # Utility functions (metrics, plotting)
├── run_experiment.py      # Complete pipeline runner
├── sample.py              # Sampling utilities
└── README.md              # This file
```

## Usage

### Complete Pipeline (Recommended)
```bash
# Activate virtual environment
source .venv/bin/activate

# Run complete experiment (training + evaluation)
python -m experiments.conv1d_gan.run_experiment
```

### Individual Components
```bash
# Training only
python -m experiments.conv1d_gan.train

# Evaluation only (after training)
python -m experiments.conv1d_gan.evaluate_conv1d_gan

# Sampling only
python -m experiments.conv1d_gan.sample --num 16
```

## Output Structure

The experiment creates the following organized output:
```
outputs/conv1d_gan/
├── plots/                          # Generated plots
│   ├── vs30_hist_50.png           # Every 50 steps
│   ├── vs30_hist_100.png
│   ├── training_losses_50.png
│   └── training_losses_final.png  # Final checkpoint
├── results/                        # Evaluation results
│   └── final_evaluation_results.pkl
├── checkpoint_*.pt                 # Model checkpoints
├── samples_*.npy                   # Generated samples
└── experiment_summary.md           # Experiment summary
```

## Evaluation Metrics

### Profile Comparisons
- Generated vs Real profile visualizations
- Mean profile comparisons with standard deviations
- Velocity gradient comparisons

### Distribution Comparisons
- **Vs30**: Average shear wave velocity in top 30m
- **Vs100**: Shear wave velocity at 100m depth
- Kolmogorov-Smirnov (KS) statistics for distribution comparison

### Statistical Metrics
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- KS statistics and p-values

### Training Monitoring
- Generator and Discriminator loss curves
- Recent loss trends (last 50 steps)
- Vs30 distribution tracking during training

## Wandb Integration

### Real-time Monitoring
- **Project**: `conv1d-gan-soil-profiles`
- **Loss curves**: Generator and Discriminator losses tracked in real-time
- **Metrics**: Vs30 statistics, sample ranges, KS statistics
- **Histograms**: Vs30 distributions and generated sample distributions

### Logged Metrics
- `train/loss_D`: Discriminator loss
- `train/loss_G`: Generator loss
- `metrics/generated_vs30_mean/std/min/max`
- `metrics/generated_samples_mean/std/min/max`
- `metrics/vs30_ks_statistic`
- `metrics/sample_range`
- `histograms/vs30_distribution`
- `histograms/generated_samples`

## Configuration

Key parameters in `config.py`:
- `latent_dim`: Generator input dimension (default: 128)
- `base_channels`: Base number of channels (default: 128)
- `num_steps`: Training steps (default: 100)
- `batch_size`: Training batch size (default: 64)
- `lr`: Learning rate (default: 2e-4)
- `checkpoint_every`: Save checkpoint every N steps (default: 10)
- `log_every`: Print logs every N steps (default: 10)

## Model Architecture

### Generator
- **Input**: Random noise (latent_dim)
- **Architecture**: Linear projection → Transposed convolutions → Final convolution
- **Output**: Vs profile (1, length)

### Discriminator
- **Input**: Vs profile (1, length)
- **Architecture**: Convolutional layers → Adaptive pooling
- **Output**: Real/fake probability (1)

## Key Features

### Efficient Resource Management
- **Reduced plotting frequency**: Detailed plots only saved every 50 steps and at final checkpoint
- **Wandb logging**: Real-time metrics without I/O overhead
- **Final evaluation focus**: Only latest checkpoint evaluated for comprehensive results

### Robust Error Handling
- **Graceful wandb fallback**: Continues without wandb if not available
- **Import flexibility**: Works as module or script
- **Comprehensive diagnostics**: Clear error messages and warnings

### Consistent with Other Experiments
- **Same evaluation standards**: Comparable metrics and visualization as parametric/VAE experiments
- **Similar structure**: Organized output and documentation
- **Professional monitoring**: Wandb integration for real-time tracking

## Setup

1. Ensure dependencies are installed (project root uses `pyproject.toml`)
2. Place your dataset at `data/vspdb_vs_profiles.parquet` (default) or update `config.py`
3. Activate virtual environment: `source .venv/bin/activate`

## Notes

- **Conv1D layers**: Learn local transitions and sharp boundaries between layers
- **Wandb integration**: Training progress, loss curves, and metrics logged for real-time monitoring
- **Efficient plotting**: Detailed plots only saved every 50 steps to reduce I/O overhead
- **Final results**: Only the latest checkpoint is evaluated to generate comprehensive results
- **Variable-length generation**: Supports truncation for different profile depths

## Troubleshooting

### Common Issues
1. **Import errors**: Ensure you're running from project root with virtual environment activated
2. **Wandb errors**: Check if wandb is installed and logged in
3. **Memory issues**: Reduce batch_size in config.py
4. **Training convergence**: Increase num_steps or adjust learning rate

### Performance Tips
- Use GPU if available (automatically detected)
- Adjust batch_size based on available memory
- Monitor wandb dashboard for real-time progress
- Check generated sample ranges in logs for training quality

The Conv1D GAN experiment provides the same level of comprehensive evaluation and monitoring as your parametric and VAE experiments, with efficient resource usage and professional-grade tracking.