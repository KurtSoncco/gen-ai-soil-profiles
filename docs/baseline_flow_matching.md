# Flagship flow-matching baseline

Measured on this machine. The training code, loss, and samplers were not edited. Seed 42 was assigned on the existing config object before `train_with_depth_losses.main`. The recorded split matches `set_seed(42)` then `torch.randperm` (1,895 / 236 / 238).

## Run

| Item | Value |
|---|---|
| Entry | `experiments/flow_matching_simplified/depth_dependent_stats/train_with_depth_losses.py` |
| Data | `experiments/flow_matching_simplified/data/breakpoints.parquet` |
| Targets | `~/fm-vspdb-5k-export/data/real_depth_stats.pkl`, `real_correlations.pkl` (these match the seed-42 train bins; see the audit) |
| Seed | 42, saved in `outputs/flow_matching_simplified/checkpoints/config.json` |
| Epochs | 1,000 |
| Device | RTX 5080 Laptop, local wheel `torch==2.9.1+cu128` (CUDA 12.8). `pyproject.toml` still pins `torch==2.9.1`. |
| Wall time | 5,749 s (95.8 min) |
| Parameters | 3,424,642 |
| Best val loss | 0.1663 at epoch 450 (`best.pt`) |
| Epoch 1000 | train 0.1859, val 0.1922 (`latest.pt`, stored epoch index 999) |
| Test CFM loss | 0.2248 on the seed-42 test loader |
| Checkpoints | `outputs/flow_matching_simplified/checkpoints/{best,latest,latest_ep1000}.pt` |

`WANDB_MODE=disabled` was set. Weights & Biases still printed that it initialized; no run was used for the numbers below.

The prior Lambda 5,000-epoch numbers in `SUMMARY.md` are a different run. They are not repeated here as this baseline.

## `comprehensive_eval.py` (split not seed-locked)

Command: `python -m experiments.flow_matching_simplified.comprehensive_eval` on `latest.pt`. The script never calls `set_seed` (`comprehensive_eval.py` lines 766–772). Sizes were again 1,895 / 236 / 238, but the profiles are not the seed-42 test set. Output: `outputs/flow_matching_simplified/plots/comprehensive_eval_epoch_999.json`. Runtime about 12 s.

| Metric | Real | Generated (200, Euler 50) |
|---|---|---|
| Depth monotonicity | 1.00 | 0.98 |
| Final depth > 0 | 1.00 | 1.00 |
| Final depth < 1000 m | 0.966 | 0.995 |
| Mean final depth (m) | 97.9 | 53.8 |
| Vs in [100, 5000], code inversion `Δz/Δt` | 0.487 | 0.340 |
| min `Δt` rate | 1.00 | 0.95 |
| Vs KS | — | 0.0559 (p = 0.050) |
| Final-depth KS | — | 0.122 |
| Total-TTS KS | — | 0.087 |
| Breakpoint-count KS | — | 0.048 |
| Mean pairwise distance | 533 | 255 |

Real Vs-bound rate 0.49 is the scale bug: stored TTS is `2 h / Vs`, and this eval inverts with Passeri, so physical velocities near 200 m/s are reported near 100 m/s. Generated Vs under that inversion spanned about −1,840 to 6,930 m/s.

## Seed-42 metrics

Script: `experiments/flow_matching_simplified/baseline_metrics.py`. Same 200 draws of length, 4-vector, and noise for every step count (`seed=42`). Real profiles are the seed-42 test split. JSON: `outputs/flow_matching_simplified/results/baseline_metrics.json`.

Depth-bin RMSE uses 10 m bins and the training rule "at least 50 layer centers". On 238 test profiles only **4 of 50 bins** meet that count, because breakpoints are sparse. The RMSE is not a full-depth curve.

`ln(Vs)` errors match for the code inversion and for Passeri (`tts_stored / 2` before `Δz/Δt`), because both sides shift by `ln 2`. Bound rates do not match. The table's bound column is the code inversion; Passeri bounds are in the last column.

Layer-center correlation is the project function (`compute_vertical_autocorrelation`, pair tolerance 0.5 m). Real test means are −0.008 at 1 m and 0.048 at 2 m. Most breakpoint pairs are not 1–2 m apart, so this error is a small difference of noisy means.

Grid correlation resamples piecewise-constant `ln(Vs)` every 1 m. Real values are 0.9986 at 1 m and 0.9955 at 2 m, because those lags usually fall inside one layer. The error near 0.001 does not measure short-lag soil correlation. The Gaussian fit turned that 1 m correlation into a length of 517 m (`L = −1 / ln ρ`). That length is a layer-thickness artifact.

| Sampler | Steps | Mean `ln` RMSE | Std `ln` RMSE | Depth mono. | Depth KS | Vs KS | Bounds `Δz/Δt` | Bounds Passeri | Seconds |
|---|---|---|---|---|---|---|---|---|---|
| Euler | 5 | 0.122 | 0.121 | 0.990 | 0.084 | 0.065 | 0.340 | 0.725 | 0.18 |
| Euler | 10 | 0.114 | 0.098 | 0.980 | 0.084 | 0.071 | 0.345 | 0.715 | 0.19 |
| Euler | 20 | 0.122 | 0.100 | 0.980 | 0.084 | 0.063 | 0.330 | 0.705 | 0.23 |
| Euler | 50 | 0.124 | 0.100 | 0.965 | 0.084 | 0.064 | 0.310 | 0.685 | 0.36 |
| RK4 | 5 | 0.112 | 0.123 | 0.970 | 0.090 | 0.067 | 0.300 | 0.690 | 0.24 |
| RK4 | 10 | 0.116 | 0.104 | 0.970 | 0.086 | 0.068 | 0.300 | 0.690 | 0.33 |
| RK4 | 20 | 0.119 | 0.105 | 0.970 | 0.085 | 0.068 | 0.300 | 0.680 | 0.49 |
| RK4 | 50 | 0.124 | 0.110 | 0.965 | 0.085 | 0.069 | 0.300 | 0.675 | 1.02 |
| Fixed-grid Gaussian | — | 0.058 | 0.115 | 1.000 | 0.056 | 0.077 | 0.570 | 0.945 | 0.23 |

Euler is the integrator `comprehensive_eval` calls. RK4 is `utils.sample_sequences_ode`, which that script does not call. Breakpoint-count KS is 0.066 for every flow row: length is drawn from the training histogram before integration.

Chord curvature of the Euler field, from the shared noise to the 50-step Euler endpoint, averaged over 16 times and 200 sequences: **0.076**. This is not the paper's protocol (10,000 pairs, 128 Euler steps, training pairs).

## Fixed-grid Gaussian

Fit on the seed-42 **train** split only, in the code-inversion `ln(Vs)`. Each sample reuses one train depth-knot vector and draws layer `ln(Vs)` from the 10 m bin mean and std with exponential correlation at the 1 m grid coefficient above. Travel time is `Δt = Δz / Vs_code`, the same convention as the stored file. It is not a neural net. It beats the transformer on depth-bin mean error, final-depth KS, monotonicity, and bound rate. It does not beat it on Vs KS (0.077 vs about 0.065).

## Open questions

- RMSE over the bins with fewer than 50 test points was not stored. With only four bins, the depth-stat ranking is thin.
- `best.pt` (epoch 450, val 0.166) was not the checkpoint scored here. `latest.pt` was.
- A Passeri-scaled copy of the parquet was not trained.
