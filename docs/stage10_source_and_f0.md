# Stage 10: Toro-source ablation and the f0 gap

Stage 9 left residual ρ 0.018–0.027 high and f0 at 7.7–8.0 Hz against a test median of 9.5 Hz. This stage checks both. No graph VAE.

## 1. f0 is a train/test shift

Time-averaged f0 on the seed-42 split, same transfer-function code as Stage 8:

| Split | All f0 | H < 30 m | H ≥ 30 m |
|---|---|---|---|
| Train | 8.04 | 10.3 | 6.18 |
| Val | 8.26 | 9.23 | 7.10 |
| Test | 9.49 | 12.2 | 6.22 |
| Constrained s42 (Toro source) | 7.98 | 9.62 | 5.69 |

Generated f0 matches the train set, which is what `(n, H, Vs_z)` is drawn from. The test set is shifted high, mostly on profiles shallower than 30 m. Conditioning on f0 to hit 9.5 Hz would chase the test split. Not done.

## 2. Toro source moves residual ρ

Same decoder and Vs30 condition. Latest checkpoint, matched `(n, H, Vs_z)`, 200 samples. Train residual ρ is 0.774, test 0.772.

| Source | Seeds | Residual ρ | Trend RMSE | Vs30 KS | Vs30 median | Thickness KS |
|---|---|---|---|---|---|---|
| Toro, frac 1.0 (Stage 9) | 3 | 0.790–0.799 | 0.076–0.102 | 0.075–0.122 | 303–318 | 0.104–0.124 |
| Gaussian, frac 0.0 | 1 (s42) | 0.749 | 0.149 | 0.100 | 307 | 0.091 |
| Mix, frac 0.5 | 3 | 0.765–0.788 | 0.160–0.182 | 0.065–0.079 | 304–320 | 0.099–0.107 |

All-Toro is 0.018–0.027 high. No Toro is 0.023 low and worse on trend. A 50/50 mix lands on 0.772 at the median (range 0.765–0.788) and has the best Vs30 KS, but trend RMSE roughly doubles (0.17 vs 0.08).

`--toro-source-frac` in [`train_constrained.py`](../experiments/flow_matching_simplified/train_constrained.py) sets the batch fraction that starts from Toro; the rest is Gaussian. Score with `python -m experiments.flow_matching_simplified.stage9_score --tag mix50_s42_e500`.

## 3. Headline

Keep the Stage 9 all-Toro model as the headline: best depth trend, Vs30 already on the real median, ρ only slightly high. Use the mix when residual ρ is the decision metric. Do not drop Toro.

GPU: no-Toro seed 42 (281 s) plus three mix seeds (365 + 366 + 342 s), 0.38 h. Ledger total 7.48 of 10.0 h.
