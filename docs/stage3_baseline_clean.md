# Stage 3: baseline on one-way data

Commit `a8f629e3a14b84b541b45a41f26e1fb4bfb7426d`. Branch `fix/onewaytime-and-structured-prior`. Python 3.11.15, `torch==2.9.1+cu128`, CUDA 12.8, NVIDIA GeForce RTX 5080 Laptop GPU. `pyproject.toml` pins `torch==2.9.1`. GPU time this stage: 1.36 h. Running total: 1.36 / 7.00 h.

The trainer was not edited. `train_onewaytime.py` set seed 42, `breakpoints_onewaytime.parquet`, and the one-way target pickles, and wrote to `outputs/flow_matching_simplified_onewaytime/`. Copies: `outputs/flow_matching_simplified/checkpoints/latest_onewaytime.pt` and `best_onewaytime.pt`. The seed-42 index file was not rewritten. `set_seed(42)` then `torch.randperm` matches it (1,895 / 236 / 238).

JSON: `outputs/flow_matching_simplified/results/corrected_eval_onewaytime.json`. Inversion is `Vs = Δz / Δt`. On the one-way fine grid, every layer matches `h/Vs` (352,092 / 352,092) and none match `2h/Vs`.

## Training

| Item | Value |
|---|---|
| Wall time | 4,806.8 s (80.1 min) |
| Parameters | 3,424,642 |
| Epoch 1 | train 1.538, val 1.073 |
| Best val | 0.1680 at stored epoch 449 (printed Epoch 450), `best.pt` |
| Epoch 1000 | train 0.1866, val 0.1949, `latest.pt` stored epoch 999 |
| Test CFM loss | 0.2274 |

`WANDB_MODE=disabled` was set. Weights & Biases still printed that it initialized.

## Real bound rate

Seed-42 test breakpoints, every layer in [100, 5000]: **0.9496**. The full one-way file is 0.9939 of layers and 0.9266 of profiles. The misses are catalog Vs values outside the bound. This is the rate the corrected tokens produce. Training was started after that was already reported.

## Reference checkpoint: `latest.pt`

200 samples, shared seed-42 noise for Euler and RK4. Intervals are 2.5–97.5 percentiles from 400 resamples of this one generated set. One training seed. All 20 depths (10 m through 200 m) are used. Breakpoint-count KS is 0.066 on every flow row and equals the pre-ODE histogram draw (`train.py` line 311).

| Sampler | Steps | Trend `ln` RMSE | Bounds | Mono. | Final-depth KS | Vs KS | Breakpoint-depth KS | Pairwise distance |
|---|---|---|---|---|---|---|---|---|
| Euler | 5 | 0.307 [0.190, 0.439] | 0.730 | 0.990 | 0.108 | 0.086 | 0.084 | 562.3 |
| Euler | 10 | 0.276 [0.138, 0.426] | 0.715 | 0.985 | 0.108 | 0.092 | 0.084 | 561.8 |
| Euler | 20 | 0.280 [0.151, 0.433] | 0.685 | 0.980 | 0.108 | 0.096 | 0.083 | 562.8 |
| Euler | 50 | 0.272 [0.143, 0.417] | 0.675 | 0.970 | 0.103 | 0.097 | 0.084 | 524.4 |
| RK4 | 5 | 0.283 [0.134, 0.422] | 0.690 | 0.970 | 0.114 | 0.105 | 0.088 | 517.3 |
| RK4 | 10 | 0.274 [0.141, 0.420] | 0.675 | 0.970 | 0.091 | 0.100 | 0.085 | 520.8 |
| RK4 | 20 | 0.272 [0.143, 0.419] | 0.680 | 0.970 | 0.098 | 0.100 | 0.084 | 522.3 |
| RK4 | 50 | 0.271 [0.143, 0.418] | 0.675 | 0.970 | 0.098 | 0.100 | 0.084 | 523.1 |
| Gaussian | — | 0.228 [0.124, 0.363] | 0.945 | 1.000 | 0.043 | 0.077 | 0.056 | 544.5 |

Real test pairwise distance is 708.1. Real mean final depth is 89.6 m. `latest` Euler 50 mean final depth is 95.1 m. The Gaussian is 74.3 m.

Vertical structure. Real test, 42,158 pairs at 1 m and 41,682 at 2 m: same-layer fraction 0.948 and 0.902; `ln(Vs)` correlation 0.993 [0.988, 0.995] and 0.988 [0.980, 0.991].

| Row | Same-layer 1 m | Corr 1 m | Gap 1 m | Same-layer 2 m | Corr 2 m | Gap 2 m |
|---|---|---|---|---|---|---|
| latest Euler 50 | 0.899 | 0.964 [0.953, 0.972] | 0.029 | 0.808 | 0.940 | 0.048 |
| latest RK4 50 | 0.899 | 0.962 [0.950, 0.969] | 0.032 | 0.809 | 0.936 | 0.052 |
| Gaussian | 0.936 | 0.994 [0.992, 0.995] | 0.001 | 0.881 | 0.988 | 0.000 |

Profiles reaching each depth, reference rows:

| Depth (m) | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 | 110 | 120 | 130 | 140 | 150 | 160 | 170 | 180 | 190 | 200 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Real | 225 | 197 | 116 | 80 | 58 | 50 | 39 | 36 | 31 | 29 | 28 | 27 | 25 | 22 | 21 | 19 | 19 | 17 | 16 | 15 |
| latest Euler 50 | 147 | 125 | 79 | 39 | 29 | 24 | 19 | 16 | 12 | 11 | 8 | 6 | 4 | 2 | 2 | 2 | 1 | 1 | 1 | 1 |
| Gaussian | 188 | 157 | 98 | 65 | 49 | 41 | 38 | 30 | 25 | 20 | 18 | 16 | 16 | 16 | 14 | 14 | 13 | 13 | 11 | 11 |

Paper-protocol curvature, 10,000 trajectories, 128 Euler steps, Euler state as `x_t` (App. A.3, verified from the full text): `latest.pt` 5.664.

On the full 20-depth support the Gaussian interval contains a lower trend point estimate, matches the real 1 m and 2 m correlation, and has the higher bound rate. The flow intervals for trend overlap the Gaussian. The flow correlation interval at 1 m sits below the real interval.

## `best.pt`, reduced support

Stored epoch 449. Curvature 6.964. In these 200 samples it has zero profiles at 130–200 m, so the trend uses 12 depths. That number is not ranked against `latest` or the Gaussian.

| Depth (m) | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 | 110 | 120 | 130–200 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| best Euler 50 | 141 | 119 | 68 | 30 | 23 | 18 | 13 | 11 | 7 | 4 | 4 | 3 | 0 |

## Open questions

- The clean-data comparison still has the Gaussian ahead on correlation, bounds, and final-depth KS, with overlapping trend intervals. Stage 4 is the audit write-up from these results. I need a yes before writing it.
- Stage 6, if reached, is about 4.8 GPU-hours and fits in the 5.64 h left only if Stages 4 and 5 stay at zero GPU time.
