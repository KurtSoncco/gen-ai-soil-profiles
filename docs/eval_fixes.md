# Corrected evaluation

Phase 1 only. No retraining. `train.py`, `utils.py`, `comprehensive_eval.py`, `baseline_metrics.py`, the parquet, and `pyproject.toml` were not edited. Code: `experiments/flow_matching_simplified/eval_corrected.py`, `run_corrected_eval.py`, `tests/flow_matching/test_eval_corrected.py` (3 passed). Numbers: `outputs/flow_matching_simplified/results/corrected_eval.json`.

Environment in that file: Python 3.11.15, `torch==2.9.1+cu128`, CUDA 12.8, NVIDIA GeForce RTX 5080 Laptop GPU. `pyproject.toml` still pins `torch==2.9.1`.

## Inversion

Passeri one-way time is `tts = Σ h / Vs`. Every Vs number in this file uses

`Vs = Δz / Δt`

(`eval_corrected.py` line 56). The same formula is in `utils.compute_vs_from_sequence` lines 199–201 and `comparison_test.py` lines 66–68. A synthetic profile with Vs 200, 400, 800, encoded as `Δt = h / Vs`, round-trips. A 150 m/s layer stays inside [100, 5000] (`tests/flow_matching/test_eval_corrected.py`).

The parquet writer stores twice that time: `layer_tts = 2 * thickness / velocities` (`vs_calculation.py` line 349). On `data/vspdb_tts_profiles.parquet`, 352,092 / 352,092 fine-layer increments match `2h/Vs` and 0 match `h/Vs`. Reading those tokens with `Vs = Δz / Δt` returns half the `vs_value` column. The `vs_value` column itself lies in [100, 5000] for 0.9939 of layers and for every layer of 0.9266 of profiles. The breakpoint bound rate below is the Passeri reading of the tokens, which is the series the model sees.

Seed-42 test breakpoints, Passeri, every layer in [100, 5000]: **0.4580** (238 profiles).

## Status

1. **Fixed.** Inversion is `Vs = Δz / Δt`. The earlier table that used `2Δz/Δt` is withdrawn.
2. **Fixed.** Every row uses `outputs/flow_matching_simplified/results/split_indices_seed42.json` (1,895 / 236 / 238). `comprehensive_eval.py` lines 766–772 still draw an unseeded `torch.randperm` and were not used.
3. **Fixed.** Depth trend is the n-weighted RMSE of mean `ln(Vs)` at 10, 20, …, 200 m, every depth both sets reach, no 50-count cutoff. Counts are in the last table. The old 4-bin RMSE is not a ranking.
4. **Fixed.** At lag h, on a 0.5 m grid: fraction of pairs in the same piecewise-constant layer, and Pearson correlation of their `ln(Vs)`, with 400 profile-bootstrap percentiles. Synthetic checks: the fraction matches layer-index enumeration; i.i.d. layers give correlation near that fraction; a profile-constant Vs gives correlation above 0.98 while the fraction stays put.
5. **Fixed.** No correlation length is reported.
6. **Fixed.** Euler (`baseline_metrics.euler_integrate`) and RK4 (`utils.sample_sequences_ode`) share one seed-42 initial state per checkpoint.
7. **Fixed.** For every flow row, generated lengths equal the pre-ODE histogram draw (`train.py` line 311). Breakpoint-count KS is 0.066 and does not change with steps or checkpoint. The Gaussian KS is 0.085.
8. **Fixed.** `latest.pt` stored epoch index 999 (the 1,000th epoch). `best.pt` stored index 449, the print "Epoch 450" (`train_with_depth_losses.py` lines 524 and 551). Val 0.1663 is unchanged.
9. **Fixed as the paper protocol, with one named choice.** App. A.3, verified from the full text: `C = E_t ||(x_1 - x_0) - v_θ(x_t, t)||^2`, 10,000 generated samples, 128 Euler steps. This run uses the Euler state as `x_t` and the Euler endpoint as `x_1`. The baseline 0.076 remains a local chord proxy (200 samples, 16 times).
10. **Fixed.** Environment is in the JSON and above.

## Table

200 samples. Intervals are 2.5–97.5 percentiles from 400 resamples of this one generated set and the test set. One training seed. `ln(Vs)` differences are unchanged by a constant scale on both sides; this run recomputed them under `Vs = Δz / Δt` and they match the previous JSON.

`best.pt` has zero generated profiles reaching 130–200 m, so its trend uses 12 depths. `latest.pt` and the Gaussian use all 20.

| Model | Sampler | Steps | Trend `ln` RMSE | Depths | Bounds `Δz/Δt` | Mono. | Final-depth KS | Vs KS | Breakpoint-depth KS |
|---|---|---|---|---|---|---|---|---|---|
| latest, ep 999 | Euler | 5 | 0.306 [0.187, 0.440] | 20 | 0.340 | 0.990 | 0.103 | 0.090 | 0.084 |
| latest | Euler | 10 | 0.291 [0.173, 0.428] | 20 | 0.345 | 0.980 | 0.108 | 0.093 | 0.084 |
| latest | Euler | 20 | 0.256 [0.158, 0.407] | 20 | 0.330 | 0.980 | 0.103 | 0.093 | 0.084 |
| latest | Euler | 50 | 0.253 [0.141, 0.391] | 20 | 0.310 | 0.965 | 0.108 | 0.099 | 0.084 |
| latest | RK4 | 5 | 0.260 [0.153, 0.397] | 20 | 0.300 | 0.970 | 0.104 | 0.099 | 0.090 |
| latest | RK4 | 10 | 0.267 [0.156, 0.411] | 20 | 0.300 | 0.970 | 0.088 | 0.101 | 0.086 |
| latest | RK4 | 20 | 0.260 [0.158, 0.407] | 20 | 0.300 | 0.970 | 0.093 | 0.099 | 0.085 |
| latest | RK4 | 50 | 0.250 [0.135, 0.378] | 20 | 0.300 | 0.965 | 0.103 | 0.099 | 0.085 |
| best, ep 449 | Euler | 5 | 0.147 [0.069, 0.328] | 12 | 0.320 | 0.975 | 0.099 | 0.099 | 0.080 |
| best | Euler | 10 | 0.156 [0.091, 0.347] | 12 | 0.305 | 0.965 | 0.099 | 0.100 | 0.081 |
| best | Euler | 20 | 0.240 [0.082, 0.391] | 13 | 0.295 | 0.955 | 0.099 | 0.104 | 0.081 |
| best | Euler | 50 | 0.243 [0.086, 0.393] | 13 | 0.265 | 0.950 | 0.104 | 0.106 | 0.079 |
| best | RK4 | 5 | 0.146 [0.094, 0.351] | 12 | 0.255 | 0.940 | 0.124 | 0.111 | 0.084 |
| best | RK4 | 10 | 0.141 [0.096, 0.342] | 12 | 0.255 | 0.940 | 0.114 | 0.110 | 0.080 |
| best | RK4 | 20 | 0.123 [0.088, 0.327] | 12 | 0.250 | 0.940 | 0.109 | 0.109 | 0.079 |
| best | RK4 | 50 | 0.245 [0.092, 0.400] | 13 | 0.255 | 0.940 | 0.109 | 0.109 | 0.078 |
| Gaussian | — | — | 0.228 [0.124, 0.363] | 20 | 0.565 | 1.000 | 0.043 | 0.077 | 0.056 |

Real test mean final depth is 89.6 m. Generated means are about 94 m (`latest`), 88 m (`best`), and 74 m (Gaussian).

Paper-protocol curvature: `latest.pt` 5.533, `best.pt` 6.880.

Vertical structure and diversity. Real test, 42,158 pairs at 1 m and 41,682 at 2 m: same-layer fraction 0.948 [0.925, 0.960] and 0.902 [0.860, 0.926]; `ln(Vs)` correlation 0.993 [0.988, 0.995] and 0.988 [0.980, 0.991]. Real pairwise distance 708.1. Generated distances drop non-monotonic profiles because `interp1d` needs increasing depth.

| Model | Row | Same-layer 1 m | Corr 1 m | Gap 1 m | Same-layer 2 m | Corr 2 m | Gap 2 m | Pairwise distance | Dropped |
|---|---|---|---|---|---|---|---|---|---|
| latest | Euler 50 | 0.901 | 0.970 [0.960, 0.976] | 0.023 | 0.812 | 0.948 [0.930, 0.957] | 0.040 | 483.2 | 7 |
| latest | RK4 50 | 0.902 | 0.966 [0.956, 0.972] | 0.027 | 0.815 | 0.941 [0.922, 0.951] | 0.047 | 482.2 | 7 |
| best | RK4 20 | 0.880 | 0.956 [0.943, 0.965] | 0.037 | 0.771 | 0.923 [0.900, 0.939] | 0.064 | 387.6 | 12 |
| Gaussian | — | 0.936 | 0.994 [0.992, 0.995] | 0.001 | 0.881 | 0.988 [0.985, 0.991] | 0.000 | 544.5 | 0 |

Flow correlation intervals at 1 m sit below the real interval. The Gaussian interval contains the real correlation. Trend intervals for `latest` Euler 50 and the Gaussian overlap.

Profiles reaching each depth:

| Depth (m) | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 | 110 | 120 | 130 | 140 | 150 | 160 | 170 | 180 | 190 | 200 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Real | 225 | 197 | 116 | 80 | 58 | 50 | 39 | 36 | 31 | 29 | 28 | 27 | 25 | 22 | 21 | 19 | 19 | 17 | 16 | 15 |
| latest Euler 50 | 146 | 124 | 79 | 39 | 30 | 24 | 19 | 16 | 12 | 11 | 8 | 6 | 4 | 2 | 2 | 2 | 1 | 1 | 1 | 1 |
| best RK4 20 | 142 | 120 | 69 | 31 | 24 | 18 | 14 | 12 | 8 | 5 | 5 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Gaussian | 188 | 157 | 98 | 65 | 49 | 41 | 38 | 30 | 25 | 20 | 18 | 16 | 16 | 16 | 14 | 14 | 13 | 13 | 11 | 11 |

## Phase 2 is not started

Keep a structured prior only if it improves vertical correlation or diversity over both the Gaussian-prior flow and the fixed-grid Gaussian by more than seed-to-seed variation, without losing depth-trend accuracy or Passeri bound rate. A, B, and three λ values of C at 1,000 epochs is 15 runs and 24.0 GPU-hours.

## Open questions

- Passeri on the stored tokens gives a real bound rate of 0.458, because the writer used `2h/Vs`. The `vs_value` column is inside [100, 5000] at 0.994 of layers. Say if later scores should keep reading the tokens as stored, or if the stored time should be replaced before training. I will not multiply `Δz` by 2.
- `best.pt` never reaches 130 m in these 200 samples. I need a rule if Phase 2 should require the same depth support.
- App. A.3 does not say whether `x_t` is the Euler state or the straight chord. This file uses the Euler state.
- Phase 2 needs a yes or no on the 24 GPU-hour plan before any training.
