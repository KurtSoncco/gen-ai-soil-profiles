# Travel-time convention audit

Search covered `*.py`, `*.md`, `*.sh`, and `notebooks/**/*.ipynb`. `notebooks/` has no `tts` / `layer_tts` / `two-way` hits. `outputs/` stores metric JSON from the scripts below; it does not define the conversion.

Passeri one-way time is `Δt = Δz / Vs`, inverted by `Vs = Δz / Δt`. "Doubles" means the code multiplies thickness by 2 or divides time by 2.

| Site | Reads stored `Δt` as | Doubles? | Lines |
|---|---|---|---|
| `soilgen_ai/vs_profiles/vs_calculation.py` `compute_tts` | writes `Δt = h / Vs` after this stage | no, after the edit | 348–349 |
| `compute_vs30`, `compute_vs_at_depth` in the same file | time from `h / Vs`, not the `tts` column | no | 73–74, 170–171 |
| `tests/vs_profiles/test_oneway_convention.py` | Passeri round trip; parquet gate | no | whole file |
| `tests/vs_profiles/test_vs_calculation.py` `test_compute_tts` | expects `compute_tts` | no, expected series is now `h / Vs` | 60–62 |
| `experiments/flow_matching_simplified/utils.py` `compute_vs_from_sequence` | `Vs = Δz / Δt` | no; comment rejects `2 Δz / Δt` | 199–201 |
| `utils.compute_vs_penalty` | same, on denormalized `ut` | no | 357–358 |
| `train.py` `compute_flow_matching_loss` | calls that penalty on `ut` | no | 136–148 |
| `depth_dependent_stats/train_with_depth_losses.py` | same penalty on `ut`; depth and correlation losses see `u1` | no | 143–153, 171–176, 194–198 |
| `depth_statistics.reconstruct_vs_profile` | `Vs = Δz / Δt` | no | 57–59 |
| `depth_dependent_stats/utils.py` `sequence_to_vs_profile` | calls `reconstruct_vs_profile` | no | 147 |
| `depth_dependent_stats/compare_models.py` `tts_depth_to_vs` | `Vs = Δz / Δt` | no | 69–70 |
| `comparison_test.py` `tts_depth_to_vs` | `Vs = Δz / Δt` | no | 66–68 |
| `eval_corrected.vs_from_travel_time` | `Vs = Δz / Δt` | no | 56 |
| `comprehensive_eval.py` | `compute_vs_from_sequence` | no | 246, 524 |
| `bias_analysis.py` | `compute_vs_from_sequence` | no | 94 |
| `stratified_hmax_test.py` | profile physics via the same inversion | no | imports `compute_vs_from_sequence` |
| `baseline_metrics._layer_vs` | `Vs = Δz / Δt`, and if `passeri=True` it first divides stored time by 2 | yes, that flag assumes the file is already `2h/Vs` | 85–90 |
| `data.py` `load_data` | stores the `tts` column as tokens; no inversion | neither | 190–197 |
| `breakpoint_analysis.extract_breakpoints` | keeps the `tts` column; no inversion | neither | 40–43 |
| `scripts/japan_data.py` `calculate_tts` | code is `h / Vs` | no in code; docstring says `2 *` | 244–247 vs 269 |
| `scripts/synthetic_toy` data loader | `cumsum(1 / Vs)` on a depth index, not `Δz / Vs` | no | `synthetic_toy_profiles_flow_matching/data.py` 150–159 |
| `experiments/VAE/src/preprocessing.py` `Vs_to_tts` | writes `2 * (Δz / Vs)` | yes | 46–56 |
| `experiments/VAE/src/utils.py` | `h / Vs` for a time sum | no | 78, 106 |
| `soilgen_ai/tts_profiles/check.py` `_calculate_velocity` | `Vs = 2 Δz / Δt` | yes | 151–153 |
| `tests/tts_profiles/test_ttsprofileprocessor.py` | expects that `×2` inversion | yes | 72–79 |
| `genetic_algorithm.py` `_calculate_velocity_penalty` | names times two-way, then `Δt / 2` | yes | 196, 220–223 |
| `plot_utils.py` | `Vs = 2 Δz / Δt` | yes | 36 |
| README | name "two-way" only | name only | 48, 83 |

`2 *` hits that are not a travel-time factor: RK4 weights in `utils.py` line 94, `vs_pairs_flow_matching/utils.py` line 59, `synthetic_toy_profiles_flow_matching/utils.py` line 76; `vs_clipped.clamp(-2 * vs_max, ...)` in `utils.py` line 361; subplot indexing in `evaluate_parametric.py` line 244.

`compute_tts` is called from tests only. No training script calls it. The on-disk parquet is unchanged.
