# Stage 6 structured prior

The contract below was written before the runs and is unchanged. Results are at the end.

## Decision rule

Unchanged, and judged only on the trend-start condition. The oracle start is a diagnostic and does not enter the rule.

Keep the structured prior only if it improves vertical correlation at 1 m or 2 m, or pairwise distance, over both the Gaussian-prior flagship and the fixed-grid Gaussian, by more than seed-to-seed variation, without losing trend accuracy or the Passeri bound rate.

Operational reading, fixed before the runs:

- The flagship numbers are the Stage 3 `latest` Euler-50 point estimates. There is one flagship seed.
- For each metric, the range is the minimum to the maximum of the three trend-prior flagship seeds. It is not a bootstrap interval and not a standard deviation.
- Beats means the median of those three seeds is better than the comparator's median by more than that range. More than: a difference equal to the range does not count. Better means higher for vertical correlation at 1 m and at 2 m, and higher for pairwise distance.
- The decision-rule comparators are the Stage 3 flagship Euler-50 value and the Stage 3 fixed-grid Gaussian value. Each is one number, so its median is that number. The fixed-grid MLP is scored with the same sentence and is not a substitute comparator for the keep-or-drop decision.
- Trend accuracy is not lost when every seed's trend RMSE is at most the flagship's 0.272. The bound rate is not lost when every seed's profile bound rate is at least the flagship's 0.675.
- If the prior beats the flagship but not the Gaussian, the report is that the Transformer adds no value on this 1D data beyond the trend.
- If it does not beat the flagship on correlation or diversity, the report is that the structured prior does not help.

No λ sweep. `best.pt` is not ranked when its depth support is short of 20 knots.

## Training contract for the three flagship seeds

Seeds 42, 43, and 44. 1,000 epochs. Same losses, architecture, and one-way data as Stage 3. The default `N(0, I)` path stays in the code so Stage 3 can be reproduced.

The split does not change with the seed. All three runs load `outputs/flow_matching_simplified/results/split_indices_seed42.json` (1,895 / 236 / 238). The trend model is fit on that train split only, once, and reused. Normalization is the per-profile z-score of those same sequences. AdamW and ReduceLROnPlateau are built with the Stage 3 hyperparameters on every seed. The realized learning rate is what that schedule produces from that seed's own validation loss. It is not copied from seed 42. Data-loader order uses a generator seeded with 42, shared across runs. The run seed changes model initialization, dropout noise, and the α = 0.5 sampling noise.

Each seed records two bound fractions. The training fraction is the share of projected knots, on the seed-42 train split, whose ln(Vs) lies outside [ln 100, ln 5000]. The training projection is not clipped, so this fraction is a property of the split and should match across seeds. The trend-start fraction is the share of knots in that seed's inference draws that were outside the same interval before clipping. A bound-rate change is read against both numbers.

For each profile, T(x1) is its own projection, including layers outside [100, 5000] m/s.

- Knots are 10 m through 200 m. Knots deeper than the profile are left out.
- Factor k = 4: average contiguous groups of four valid knots, then linear interpolation back. A validation sweep of k is not part of these three seeds.
- Convert that ln(Vs) to travel time at the profile's own breakpoint depths with Δt = Δz / Vs.
- Normalize with the real profile's sequence statistics, the same vector the model is conditioned on.
- x0 = (1 − 0.5) T + 0.5 ε in that normalized token space, with fresh ε ~ N(0, I) each step, padding masked.
- The regression target is x1 − x0.

The Gaussian trend is not a training target. At inference it is a sample from the training distribution of this coarse projection, upsampled and converted the same way, then clipped per knot to [ln 100, ln 5000]. The clipped-knot fraction is reported next to the bound rate. The existing unclipped fixed-grid Gaussian remains the decision-rule baseline. It is not the ODE's initial state.

Oracle evaluation, same three checkpoints: start from T(x_i) of the held-out test profiles plus α = 0.5 noise, with no clip. It uses test-profile information and is not the headline result.

The inference prior is clipped and the training projection is not. A bound-rate change can come from that mismatch. The write-up has to say so.

## Fixed-grid timing

`experiments/flow_matching_simplified/fixed_grid_timing.py`, one seed, 1 warmup epoch and 10 timed epochs, on the RTX 5080. Result file: `outputs/flow_matching_simplified/results/fixed_grid_timing.json`.

| Item | Value |
|---|---|
| Parameters | 344,852 |
| Median epoch | 0.137 s |
| 1,000 epochs, one seed | 0.038 h |
| Three seeds | 0.11 h |

The loss after 10 epochs is about 0.39. That is not a result. The three fixed-grid seeds fit beside the 4.02 h flagship runs, so they are included in the plan. They have not been launched.

## What is not launched

The three flagship seeds and the three fixed-grid seeds. Together about 4.13 h of the 5.64 h remaining.

## Results

Written after the runs. The decision rule above is the one that was applied. Headline rows are Euler 50 on `latest_trend_s{seed}.pt` and `grid_s{seed}.pt`, epoch index 999. Every headline row uses 20 depths. Files: `outputs/flow_matching_simplified/results/stage6_trend_eval.json`, `outputs/flow_matching_simplified/results/stage6_fixed_grid_eval.json`.

All three flagship seeds loaded the seed-42 split (`train_index_sum` 2,242,500; 1,895 / 236 / 238) and the same training projection: 21 of 8,183 knots outside [ln 100, ln 5000], fraction 0.002566. Final validation losses were 0.078, 0.077, and 0.072. Best validation losses were 0.070, 0.064, and 0.072. The scored checkpoints are the epoch-1000 weights.

Trend-start knots outside the bounds before clipping, per seed: 2/904 (0.00221), 1/970 (0.00103), 5/958 (0.00522). The fixed-grid draws used the same templates, so the same three fractions. The training projection is unclipped and the inference prior is clipped. Both fractions are about 0.002 to 0.005, so a bound-rate move of several points is not accounted for by that mismatch.

### Trend start, the decision condition

| Model | Seed | Trend RMSE | Bounds | Corr 1 m | Corr 2 m | Pairwise | Dropped |
|---|---|---|---|---|---|---|---|
| Stage 3 flagship | 42 | 0.272 | 0.675 | 0.964 | 0.940 | 524.4 | 6 |
| Stage 3 Gaussian | — | 0.228 | 0.945 | 0.994 | 0.988 | 544.5 | — |
| Trend prior | 42 | 0.092 | 0.625 | 0.977 | 0.959 | 278.6 | 22 |
| Trend prior | 43 | 0.144 | 0.680 | 0.979 | 0.962 | 432.9 | 27 |
| Trend prior | 44 | 0.130 | 0.595 | 0.962 | 0.931 | 404.8 | 19 |
| Fixed-grid MLP | 42 | 0.049 | 0.990 | 0.997 | 0.993 | 555.6 | 0 |
| Fixed-grid MLP | 43 | 0.120 | 0.990 | 0.998 | 0.995 | 694.9 | 0 |
| Fixed-grid MLP | 44 | 0.089 | 1.000 | 0.997 | 0.993 | 727.6 | 0 |

Seed 44’s 200 m bin contains one generated profile. `n_depths_used` is still 20.

Flagship trend-start medians, ranges, and the beat test (median minus comparator, must exceed the range; higher is better):

| Metric | Median | Range | Margin vs flagship  | Beats flagship | Margin vs Gaussian | Beats Gaussian |
|---|---|---|---|---|---|---|
| Corr 1 m | 0.9774 | 0.0170 | +0.0134 | no | −0.0166 | no |
| Corr 2 m | 0.9585 | 0.0307 | +0.0185 | no | −0.0295 | no |
| Pairwise | 404.8 | 154.3 | −119.6 | no | −139.7 | no |

Trend RMSE is at most 0.272 on every seed (0.092, 0.144, 0.130), so trend accuracy is preserved. Curvature on the trend start is 18.35, 17.86, and 23.52, against 5.66 for the Stage 3 flagship.

The keep rule's bound clause requires every seed's profile bound rate to be at least the Stage 3 flagship's 0.675. Seed 43 is 0.680, above that floor and above the Stage 3 flagship. Seeds 42 and 44 are 0.625 and 0.595. Two of the three seeds lose the Passeri bound rate. The bound rate is not uniformly lost, and the trend-prior seeds are not uniformly worse than 0.675. The cause is not established. The clip fractions of 0.002 to 0.005 are too small to account for it. The remaining candidate is the training target: the unclipped projection carries the same 21 out-of-bound knots on every seed, so the flow is trained on values the inference prior clips and the bound metric rejects. That is a hypothesis, not a finding.

The structured prior, as specified, does not beat the Stage 3 flagship or the Gaussian on the decision metrics. Its trend accuracy is preserved. The keep rule also fails its every-seed bound clause on seeds 42 and 44. The prior is dropped. One prior design was tested.

The fixed-grid MLP is scored with the same sentence and is not a substitute comparator. Its 1 m median is 0.9967 (range 0.0009) and its 2 m median is 0.9934 (range 0.0019). Both beat the single Stage 3 flagship value and the single Gaussian value. Its pairwise median is 694.9 (range 172.0). The margins over 524.4 and 544.5 are 170.5 and 150.4, both inside that range. Every MLP seed keeps trend RMSE at or below 0.272 and a bound rate of 0.990 to 1.000. Monotonicity is 1.0 and no profiles are dropped.

A simple fixed-grid flow outperforms the Transformer on the correlation metrics in this 1D setting. That is a result about model class, not about the prior. The MLP is a different model. The comparison is three MLP seeds against one flagship seed and one Gaussian draw, so the flagship's seed-to-seed spread is unmeasured. Whether the correlation advantage survives more seeds, and whether it holds in 2D or 3D, is untested.

### Oracle start, diagnostic only

Same checkpoints, started from T(x_i) of the 238 test profiles plus α = 0.5 noise, no clip. This uses test-profile information and does not enter the rule.

| Seed | Trend RMSE | Bounds | Corr 1 m | Gap 1 m | Pairwise | Curvature |
|---|---|---|---|---|---|---|
| 42 | 0.072 | 0.697 | 0.992 | 0.001 | 392.7 | 2.01 |
| 43 | 0.122 | 0.710 | 0.977 | 0.016 | 406.1 | 1.72 |
| 44 | 0.095 | 0.752 | 0.974 | 0.019 | 307.0 | 1.88 |

Oracle curvature is about 2 on every seed, against 18–24 for the trend start. With the true projection as the start, the trajectory is nearly straight. The 1 m correlation gap is 0.001 on seed 42 and stays at 0.016 and 0.019 on seeds 43 and 44. Straightness does not give the right vertical correlation. The learned velocity field is not reproducing the correlation structure even when the start is the test profile's own projection. That points at the flow's objective or the geostatistics losses, not at the prior. Pairwise distance stays below the Stage 3 flagship on every oracle seed.

The prior was tested. The flow was not. The flow is the remaining question.

### Open questions

- The bound-rate hypothesis above is unchecked. The training projection's 21 out-of-bound knots can be compared with the sampler's bound violations from the existing logs and the training target. That check needs no new flagship training.
- The oracle correlation gap can be read against the vertical-correlation term in the existing training code and checkpoints. That check needs no new flagship training. In the current trainer the term is computed on the real batch `u1`, then rebuilt with `torch.tensor` from a NumPy correlation (`depth_dependent_stats/losses.py`, `train_with_depth_losses.py`). Whether that call site is why the weight cannot move the sampled correlation is the check, not a new weight sweep.
- The fixed-grid MLP's correlation advantage is over a single flagship comparator and one Gaussian comparator, with three seeds on the MLP. A repeat with more seeds would be needed before the model class is recommended. The 2D/3D question is untested.
- A learned G_φ is a different model. Stage 5 said a negative stand-in does not rule one out. That run is not authorized.

Stage 6 is closed. The structured prior, as specified, is dropped. Budget used is 6.35 h of 7.00 h, with 0.65 h left. The proposed next stage is the two checks above, with no new training.
