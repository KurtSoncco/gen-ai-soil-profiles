# Flow-matching audit

Commit `a8f629e3a14b84b541b45a41f26e1fb4bfb7426d`. Branch `fix/onewaytime-and-structured-prior`. Python 3.11.15, `torch==2.9.1+cu128`, CUDA 12.8, NVIDIA GeForce RTX 5080 Laptop GPU. `pyproject.toml` pins `torch==2.9.1`. This write-up used no GPU time. Running total: 1.36 / 7.00 h.

This replaces the earlier draft of the same filename, which was written before the one-way rebuild. Numbers below are from `docs/stage3_baseline_clean.md` and `outputs/flow_matching_simplified/results/corrected_eval_onewaytime.json` unless a count is stated as computed here from the one-way files and the seed-42 split.

## Pipeline

1. `scripts/vspdb_request_data.py` `create_vs_profile` (lines 94–122) writes depth and `vs_value`. It does not write travel time.
2. `soilgen_ai/vs_profiles/vs_calculation.py` `compute_tts` line 349 now sets `layer_tts = thickness / velocities`. The training entry does not call it. The one-way files were built by `regenerate_onewaytime.py`.
3. `breakpoint_analysis.extract_breakpoints` (line 27) keeps `[depth, tts]`. The flagship reads `breakpoints_onewaytime.parquet` (`config.py` lines 21–23 is still the old path; Stage 3 overrode it at runtime).
4. `data.py` `load_data` (lines 186–211) drops the origin and z-scores travel time and depth per profile. `__getitem__` (lines 144–153) conditions on `[ts_mean, ts_std, depth_mean, depth_std]`. There is no `log1p` on these tokens. `compute_tts` applies `log1p` only if asked (line 356), and the rebuild did not.
5. `TransformerModel` (`model.py` line 116): 4 layers, 8 heads, hidden 256, `SinusoidalTimeEmbedding` (lines 162–167), `DepthConvBlock` kernel 3 (lines 175–178).
6. `compute_flow_matching_loss_with_depth` (lines 112–140) is masked CFM MSE. Euler sampling is `train.sample_sequences` (lines 354–369). RK4 is `utils.sample_sequences_ode` (lines 21–99). Length is drawn from the training histogram before integration (`train.py` line 311).
7. The physics term is `compute_vs_penalty` on the interpolant `ut` (`train_with_depth_losses.py` lines 143–153). Depth-stat and correlation losses are computed on the real batch `u1` through NumPy (`losses.py` line 91) and do not train `v_θ`.

## Convention bug and what it changed

The old writer stored `Δt = 2h / Vs` (`vs_calculation.py` line 349 before the edit). All 352,092 fine-layer increments in `data/vspdb_tts_profiles.parquet` matched that factor. The file is still there and was not overwritten. `tests/vs_profiles/test_oneway_convention.py::test_stored_parquet_inverts_to_vs_value` still fails on it, relative error 0.500.

`Vs = Δz / Δt` on that file returns half the catalog velocity, so bound rates fall and `ln(Vs)` shifts by `ln 2`. A gap in `ln(Vs)` between two sets scored with the same inversion does not move. The new depth-bin means sit `ln 2` above the old pickle (max absolute error 9.6e-7 across 50 bins, both with 10,600 train values).

Scope of published numbers:

- `docs/baseline_flow_matching.md` and `outputs/flow_matching_simplified/results/corrected_eval.json` were scored on the doubled tokens. Their bound columns are not the one-way rates.
- `docs/stage3_baseline_clean.md` is the retrain on one-way tokens. Real test profile bound rate rose from the doubled-token Passeri reading (0.458) to 0.9496. The flow's Euler-50 bound rate stayed near 0.67 (0.675 on the clean run, about 0.68 on the doubled-token Passeri column). The convention fix did not close the flow-versus-Gaussian bound gap.
- Trend RMSE and the 1 m correlation gap are on `ln(Vs)`. They are not an artifact of the factor of 2. On the clean run, `latest` Euler 50 is 0.272 [0.143, 0.417] against the Gaussian 0.228 [0.124, 0.363]. The 1 m correlation gaps are 0.029 and 0.001.

## The 0.9496 bound rate does not clear the gate

Stage 3 said the real-data bound rate must be near 1.0, and to stop if it is not. **0.9496 did not clear that gate.** Training had already been approved after the catalog profile rate 0.9266 was reported. This audit does not treat 0.9496 as a pass.

`bound_rate` (`eval_corrected.py` lines 60–70) is the fraction of **profiles** whose every layer satisfies `100 ≤ Vs ≤ 5000`. On the seed-42 test breakpoints that is 226 / 238 = 0.9496.

The same tokens, counted per layer, are 1,326 / 1,351 = 0.9815 inside the bound (25 layers below 100 m/s, none above, none non-positive). The 0.994 figure is a different set: 0.9939 of fine-grid `vs_value` entries in the whole file, not the test breakpoints. On the test fine grid the layer rate is 0.9972 (42,410 / 42,531) and the profile rate is 0.9412 (224 / 238). One slow catalog layer fails the profile. The 0.9496 number is not comparable to 0.994.

## Why the flow bound rate is 0.675 and the Gaussian is 0.945

The penalty uses the same inversion as the evaluation. `compute_vs_penalty` line 357 sets `Vs = dz / dt`. `eval_corrected.vs_from_travel_time` line 56 sets `Vs = thickness / increment`. There is no factor of 2 in either.

It is a weak constraint on the samples the evaluation actually draws.

- The weight is `vs_penalty_weight = 1e-2` (`config.py` lines 68–70), with the comment "start small". It is added to the CFM loss (`train_with_depth_losses.py` line 153). The logged CFM loss at epoch 1000 was 0.187. A layer at 50 m/s contributes `relu(100 - 50) / 100 = 0.5` before the weight, so 0.005 after it. That scale is the formula, not a penalty value saved from the run. The penalty was folded into `reconstruction_loss` and not logged alone.
- The penalty is applied to `ut`, the straight interpolant between noise and the real batch (`train_with_depth_losses.py` lines 122–123 and 145), not to the Euler or RK4 endpoint.
- Layers with `dz ≤ 1e-9` or `dt ≤ 1e-9` are dropped before the penalty (lines 349–351). Negative increments, which the evaluator counts as out of bounds, never enter it.
- `Vs` is clipped to `±2 * vs_max` before the penalty (line 361), so a very large velocity is penalized like 10,000 m/s.

The Gaussian draws `ln(Vs)` from the training bins and emits `Δt = Δz / Vs` (`baseline_metrics.py` lines 387–390). It stays near the catalog and scores 0.945, next to the real profile rate 0.950. The flow's 0.675 is the sampler leaving that support. Stage 6 keeps this loss unchanged, so a prior that improves correlation should not be credited with fixing bounds unless the bound rate actually moves. A prior that leaves bounds near 0.67 has not failed a convention test. It has the same penalty the flagship has.

## The correlation gap is not the data bug

On the clean tokens, real `ln(Vs)` correlation at 1 m is 0.993 [0.988, 0.995]. `latest` Euler 50 is 0.964 [0.953, 0.972], gap 0.029. The Gaussian is 0.994 [0.992, 0.995], gap 0.001. The same ordering was already present on the doubled tokens. A global rescaling of travel time shifts `ln(Vs)` by a constant on both sides and does not change a Pearson correlation. The convention fix cannot explain this gap. It is a difference in the generated layering.

## Tests

`pytest`, 4.37 s: **32 passed, 2 failed, 3 errors, 0 skipped.**

- Failed on purpose: `test_stored_parquet_inverts_to_vs_value`, the legacy doubled file.
- Passed: `test_oneway_parquet_inverts_to_vs_value` (352,092 layers, `Δz/Δt` matches `vs_value`).
- Pre-existing: `test_data_loading` missing `data/synthetic_toy_profiles.parquet`, and three errors in that file because fixture `dataset` is not registered.

## Gaps and risks still open

- `best.pt` (stored epoch 449, val 0.1680) reaches no generated profile at 130–200 m in the 200-sample draw. Its trend is a 12-depth score. The reference row is `latest.pt`.
- `Config.seed` is random at import (`config.py` line 109). `comprehensive_eval.py` lines 766–772 and `precompute_statistics.py` line 118 permute without `set_seed`. Stage 3 avoided both by setting seed 42 before `main` and by using the recorded split for the targets.
- `baseline_metrics._layer_vs` with `passeri=True` still divides stored time by 2 (lines 85–87). That is wrong on the one-way file. The Stage 3 Gaussian path did not use that branch. It used `sample_fixed_grid_gaussian` and `Vs = Δz / Δt`.
- README lines 45, 48, and 83 still say RK4 and two-way travel time. The TTS checker (`check.py` lines 151–153), VAE `Vs_to_tts` (`preprocessing.py` lines 53–56), and the genetic algorithm (`genetic_algorithm.py` lines 220–223) still double.
- `breakpoint_analysis.py` `__main__` (lines 190–192) reads the Japan parquet, not the VSPDB file.
- One training seed. The bootstrap intervals are resamples of one generated set, not seed-to-seed training variation.
- Breakpoint-count KS 0.066 is the histogram draw, not the flow.

## Open questions

- Stage 5 is the paper-applicability note. It needs a yes. It should not use GPU time.
- Stage 6, three seeds at the measured 80.1 min/run, is 4.02 h. Remaining budget is 5.64 h. Say if that note should be the plan, and whether the bound penalty stays at weight 0.01. Changing the weight would make Stage 6 not comparable to this baseline.
