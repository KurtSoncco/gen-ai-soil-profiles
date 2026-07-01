# Depth-Dependent Statistics Training Tweaks

This document describes the improvements made to address depth-dependent statistics matching.

## Implemented Improvements

### 1. Increased Vertical Correlation Weight
- **Default**: `vertical_corr_weight = 0.025` (increased from 0.01)
- **Rationale**: Better match vertical correlation structure, especially at short lags
- **Usage**: Can override with `--vertical-corr-weight` argument

### 2. Short Lag Emphasis
- **Parameter**: `short_lag_weight = 2.0`
- **Effect**: Lags ≤ 2m are weighted 2x more heavily than longer lags
- **Rationale**: The gap is typically largest at 1-2m lags, so focus training there

### 3. Bin Count Filtering
- **Parameter**: `min_bin_count = 50`
- **Effect**: Only bins with ≥50 samples contribute to depth statistics loss
- **Rationale**: Avoid chasing noisy targets from sparse bins

### 4. Mean vs Std Weighting
- **Parameter**: `mean_weight_ratio = 2.0`
- **Effect**: Mean ln(Vs) loss is weighted 2x more than std ln(Vs) loss
- **Rationale**: Emphasize correcting the mean trend shape μ(ln Vs)(z)

## Usage

### Training with Tweaks

```bash
python experiments/flow_matching_simplified/depth_dependent_stats/train_with_depth_losses.py \
    --target-stats experiments/flow_matching_simplified/depth_dependent_stats/real_depth_stats.pkl \
    --target-corr experiments/flow_matching_simplified/depth_dependent_stats/real_correlations.pkl \
    --vertical-corr-weight 0.03  # Optional: further increase if needed
```

### Adjusting Parameters

You can modify these in `config_extensions.py` or override via command line:

```python
# In config_extensions.py DepthStatsConfig:
vertical_corr_weight: float = 0.025  # Increase to 0.03-0.05 if needed
short_lag_weight: float = 2.0        # Increase to 3.0 for stronger short-lag focus
mean_weight_ratio: float = 2.0       # Increase to 3.0 for stronger mean emphasis
min_bin_count: int = 50              # Increase to 100 for stricter filtering
```

## Expected Improvements

After training with these tweaks:

1. **Vertical Correlations**: Generated profiles should show higher correlations at 1-2m lags, closer to real data
2. **σ(ln Vs) Shape**: Generated uncertainty envelope should better match real data, with:
   - Lower σ at shallow depths
   - Higher σ at deeper depths
3. **μ(ln Vs) Trend**: Generated mean trend should better match the depth-dependent shape of real data

## Monitoring

Watch these metrics in wandb:
- `train/depth_stats_loss`: Should decrease over time
- `train/vertical_corr_loss`: Should decrease, especially for short lags
- Compare evaluation plots after training to see improvements

## Next Steps

1. Train with these tweaks for several epochs
2. Regenerate evaluation plots using `evaluate_depth_stats.py`
3. Compare to baseline plots - generated curves should be:
   - Closer to real data curves
   - Inside the real ±1σ envelopes
   - Matching correlation magnitudes at 1-2m lags

