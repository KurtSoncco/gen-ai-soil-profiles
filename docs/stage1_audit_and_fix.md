# Stage 1: audit and convention fix

Commit `a8f629e3a14b84b541b45a41f26e1fb4bfb7426d`. Branch `fix/onewaytime-and-structured-prior`. Python 3.11.15, `torch==2.9.1+cu128`, CUDA 12.8, NVIDIA GeForce RTX 5080 Laptop GPU. `pyproject.toml` pins `torch==2.9.1`. GPU time this stage: 0 h. Running total: 0.00 / 7.00 h (`docs/gpu_budget.md`).

The old parquet files were not modified. The consumer table is `docs/convention_audit.md`.

## 1a. Layout and flagship path

`experiments/`: `VAE`, `Vp_calculation`, `conv1d_gan`, `flow_matching_simplified`, `parametric`, `stochastic_approach`, `synthetic_toy_profiles_flow_matching`, `vs_pairs_flow_matching`.

`soilgen_ai/`: `base.py`, `apis/` (VSPDB and Box clients), `vs_profiles/vs_calculation.py`, `tts_profiles/check.py`.

`scripts/`: `vspdb_request_data.py`, `japan_data.py`, `generate_synthetic_profiles.py`, `extract_vs_pairs.py`.

`tests/`: `test_base.py`, `test_example.py`, `vs_profiles/`, `tts_profiles/`, `flow_matching/`. More tests live under `experiments/*/test_*.py`.

`notebooks/VSPDB/` has no travel-time hits.

README reproduction (`README.md` lines 88–126): clone, `uv sync --extra dev`, `pytest`, `scripts/vspdb_request_data.py`, then `python -m experiments.flow_matching_simplified.train` or `depth_dependent_stats/train_model.sh`, then `comprehensive_eval`.

Flagship path:

1. Ingestion. `scripts/vspdb_request_data.py` `create_vs_profile` (lines 94–122) writes depth and `vs_value` on a 0.5 m grid. It does not write travel time.
2. Travel time. `soilgen_ai/vs_profiles/vs_calculation.py` `compute_tts` (line 349) now sets `layer_tts = thickness / velocities`. Nothing in the training entry calls this function. The file the model reads is still the old parquet.
3. Breakpoints. `breakpoint_analysis.extract_breakpoints` (lines 27–43) keeps `[depth, tts]` wherever the curve leaves a straight line. Its `__main__` (lines 190–192) reads `data/japan_borehole_profiles.parquet` and writes `experiments/flow_matching_simplified/data/breakpoints.parquet` (lines 245–247). The flagship config points at that parquet (`config.py` lines 21–23).
4. `log1p`. Optional inside `compute_tts` only, default off (line 356). `FlowMatchingDataset` does not apply it (`data.py` lines 70–84). README line 36 says models train on `log1p(Vs)`. That sentence does not describe this flagship.
5. Normalization. `load_data` (lines 178–211) groups by `profile_id`, sorts by depth, drops the origin. `_normalize_sequences` z-scores travel time and depth per profile. `__getitem__` (lines 144–153) returns `[ts_mean, ts_std, depth_mean, depth_std]`. Sequences longer than 20 are truncated (lines 126–131).
6. Model. `TransformerModel` (`model.py` line 116): linear input projection (line 156), `SinusoidalTimeEmbedding` (lines 162–167), `PositionalEncoding` (lines 170–172), `DepthConvBlock` kernel 3 (lines 175–178). Widths are in `config.py` lines 34–42: 4 layers, 8 heads, hidden 256, time embedding 128.
7. Flow loss. `compute_flow_matching_loss_with_depth` (lines 112–140): `u0 ~ N(0, I)`, linear interpolant, masked MSE of `v_θ` against `(u1 - ut) / (1 - t)`.
8. Samplers. Euler is `train.sample_sequences` (lines 354–369). RK4 is `utils.sample_sequences_ode` (lines 21–99). `comprehensive_eval` calls Euler. README line 45 says RK4, 50 steps (`config.ode_steps`, line 62).
9. Physics loss. `compute_vs_penalty` (`utils.py` lines 357–358) uses `Vs = Δz / Δt` on denormalized `ut`, added with weight `vs_penalty_weight` (train file lines 143–153). Correction from Stage 8: `ut` contains no network output, so this term has no gradient with respect to the network weights.
10. Geostatistics losses. `compute_depth_statistics_loss` and `compute_vertical_correlation_loss` are called on the real batch `u1`, not on `predicted_v` (lines 171–206). Both go through NumPy (`losses.py` line 91, `torch.from_numpy`), so they do not train `v_θ`.
11. Entry. `depth_dependent_stats/train_with_depth_losses.py` `main` (line 310), launched by `train_model.sh`.

## 1b. Consumers

See `docs/convention_audit.md`. The flagship inversion was already `Vs = Δz / Δt`. The writer, the VAE converter, the TTS checker, and the genetic algorithm were the sites that doubled.

## 1c. Acceptance tests, run before the edit

`tests/vs_profiles/test_oneway_convention.py`:

- `test_passeri_round_trip` passed before the edit.
- `test_compute_tts_is_one_way` failed before the edit. First step was 0.05 against 0.025 (`5/200`).
- `test_stored_parquet_inverts_to_vs_value` failed before the edit: max relative error 0.5000000000000865 on 7,809 layers from the first 25 profile ids in `data/vspdb_tts_profiles.parquet`.

## 1d. Source edit

`compute_tts` line 349 is `layer_tts = thickness / velocities`. The docstring (lines 285–289) now says one-way. Sorting, the surface row, cumulative sum, and optional `log1p` are unchanged. `test_compute_tts` expected values now follow `h / Vs` (`test_vs_calculation.py` lines 60–62). The parquet files were not rewritten.

After the edit, `test_compute_tts_is_one_way` passes. The parquet test still fails with the same 0.500 relative error. That failure is the Stage 2 gate.

## 1e. Hyperparameters and pytest

From `config.py`, used by `train_with_depth_losses` via `extend_config`:

| Item | Value | Line |
|---|---|---|
| Batch size | 32 | 24 |
| Epochs | 1,000 | 49 |
| Learning rate | 1e-4 | 46 |
| Adam betas | (0.6, 0.999) | 47 |
| Weight decay | 1e-4 | 48 |
| Gradient clip | 1.0 | 50 |
| Scheduler | patience 10, factor 0.5, min 1e-6 | 53–56 |
| Eval / ODE steps | every 50 epochs; `ode_steps` 50 | 60–62 |
| Max breakpoints | 20 | 26 |
| Split | 0.8 / 0.1 / 0.1 | 27 |
| Vs penalty | weight 1e-2, bounds 100–5000, `min_dt` 1e-6, on | 65–71 |
| Seed | `np.random.randint` at import | 109 |

Depth-loss block, `config_extensions.py` lines 24–35: depth-stat weight 0.01, correlation weight 0.025, bin width 10 m, lags 1, 2, 5, 10 m, max depth 500 m, min bin count 50, mean/std weight ratio 2, short-lag weight 2.

`pytest` from the repo root, 3.67 s: **31 passed, 2 failed, 3 errors, 0 skipped.**

- Failed: `test_stored_parquet_inverts_to_vs_value` (old parquet, expected until Stage 2).
- Failed: `experiments/synthetic_toy_profiles_flow_matching/test_data.py::test_data_loading` (`data/synthetic_toy_profiles.parquet` missing). Pre-existing.
- Errors: `test_normalization`, `test_profile_structure`, `test_plotting` in that same file. Fixture `dataset` is not registered. Pre-existing.

## 1f. Risks

- Training still reads the doubled parquet. `compute_tts` is not on the training path, so this edit does not change a checkpoint.
- `baseline_metrics._layer_vs` with `passeri=True` divides stored time by 2 (lines 85–87). On a one-way file that halves Vs again.
- `breakpoint_analysis.py` `__main__` reads the Japan parquet, not the VSPDB file. Stage 2 must not treat that `__main__` as the VSPDB rebuild.
- `Config.seed` is random at import (`config.py` line 109). `comprehensive_eval.py` lines 766–772 permute without `set_seed`. `precompute_statistics.py` line 118 does the same.
- README lines 45, 48, and 83 still say RK4 and two-way travel time.
- TTS checker, its test, VAE `Vs_to_tts`, and the genetic algorithm still double. They were left as they are.
- Depth-stat and correlation losses do not backpropagate into `v_θ`.

## Open questions

- Stage 2 should rebuild `breakpoints_onewaytime.parquet` from `vspdb_tts_profiles` with the fixed `compute_tts`, not from `breakpoint_analysis.py` `__main__`. Confirm that source.
- The parquet test will stay red until that new file exists and the test points at it. Say if it should stay pointed at the old file as a permanent regression check.
- I need a yes before Stage 2. It writes two new parquet files and new statistic pickles. It does not train.
