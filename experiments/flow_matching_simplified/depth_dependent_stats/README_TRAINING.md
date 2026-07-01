# Training with Depth-Dependent Statistics

This directory contains scripts for training the flow matching model with depth-dependent statistics losses.

## Quick Start

### 1. Precompute Statistics (if not done already)

First, compute the target statistics from real data:

```bash
python experiments/flow_matching_simplified/depth_dependent_stats/precompute_statistics.py
```

This generates:
- `real_depth_stats.pkl` - Depth-dependent mean and std of ln(Vs)
- `real_correlations.pkl` - Vertical autocorrelations at different lags

### 2. Train the Model

Use the bash script for easy training:

```bash
# Basic training with default settings
./experiments/flow_matching_simplified/depth_dependent_stats/train_model.sh

# With custom loss weights
./experiments/flow_matching_simplified/depth_dependent_stats/train_model.sh \
    --depth-stats-weight 0.01 \
    --vertical-corr-weight 0.03

# With custom statistics files
./experiments/flow_matching_simplified/depth_dependent_stats/train_model.sh \
    --target-stats /path/to/stats.pkl \
    --target-corr /path/to/corr.pkl
```

### 3. Evaluate the Trained Model

After training, evaluate the model:

```bash
./experiments/flow_matching_simplified/depth_dependent_stats/evaluate_model.sh
```

## Training Script Options

The `train_model.sh` script supports the following options:

- `--target-stats PATH`: Path to depth statistics .pkl file (default: `real_depth_stats.pkl`)
- `--target-corr PATH`: Path to correlations .pkl file (default: `real_correlations.pkl`)
- `--depth-stats-weight FLOAT`: Weight for depth statistics loss (overrides config default)
- `--vertical-corr-weight FLOAT`: Weight for vertical correlation loss (overrides config default)
- `--resume PATH`: Resume training from checkpoint (not yet implemented)
- `--help, -h`: Show help message

## Direct Python Usage

You can also run the training script directly:

```bash
python experiments/flow_matching_simplified/depth_dependent_stats/train_with_depth_losses.py \
    --target-stats real_depth_stats.pkl \
    --target-corr real_correlations.pkl \
    --depth-stats-weight 0.01 \
    --vertical-corr-weight 0.025
```

## Configuration

Training hyperparameters are configured in:
- `experiments/flow_matching_simplified/config.py` - Base configuration
- `experiments/flow_matching_simplified/depth_dependent_stats/config_extensions.py` - Depth-aware extensions

Key parameters:
- `depth_stats_weight`: Default 0.01 (weight for matching μ(ln Vs) and σ(ln Vs))
- `vertical_corr_weight`: Default 0.025 (weight for matching vertical correlations)
- `min_bin_count`: Default 50 (minimum samples per bin for loss computation)
- `mean_weight_ratio`: Default 2.0 (emphasize mean trend correction)
- `short_lag_weight`: Default 2.0 (extra weight for 1-2m lags)

## Outputs

Training generates:
- **Checkpoints**: `outputs/flow_matching_simplified/checkpoints/`
  - `latest.pt` - Latest checkpoint
  - `best.pt` - Best validation loss checkpoint
- **Results**: `outputs/flow_matching_simplified/results/`
  - `final_results.json` - Training metrics
- **Samples**: `outputs/flow_matching_simplified/samples/`
  - `generated_samples.npy` - Generated profiles
- **Wandb logs**: If wandb is configured, metrics are logged to your project

## Monitoring Training

If wandb is configured, you can monitor:
- `train/loss` - Total training loss
- `train/reconstruction_loss` - Flow matching reconstruction loss
- `train/depth_stats_loss` - Depth statistics matching loss
- `train/vertical_corr_loss` - Vertical correlation matching loss
- `train/epoch_loss` - Average loss per epoch
- `val/loss` - Validation loss

## Tips

1. **Start with default weights**: The default weights (0.01, 0.025) are small regularization terms. Increase gradually if needed.

2. **Monitor loss components**: Watch both reconstruction loss and depth-aware losses. If depth losses dominate, reduce their weights.

3. **Check evaluation plots**: After training, run `evaluate_model.sh` to see if generated profiles match real data statistics.

4. **Adjust for your data**: If your data has different characteristics, you may need to tune:
   - `depth_bin_width` (default: 10m)
   - `correlation_lags` (default: [1.0, 2.0, 5.0, 10.0])
   - `min_bin_count` (default: 50)

## Troubleshooting

**Error: Target statistics file not found**
- Run `precompute_statistics.py` first to generate the required files

**Training is slow**
- Reduce `num_workers` in config
- Reduce `batch_size` if running out of memory
- Use GPU if available (set `device: "cuda"` in config)

**Loss is not decreasing**
- Check that target statistics files are valid
- Verify data is loading correctly
- Try reducing loss weights if they're too high
- Check learning rate (default: 1e-4)

