# Stage 2: one-way data

Commit `a8f629e3a14b84b541b45a41f26e1fb4bfb7426d`. Branch `fix/onewaytime-and-structured-prior`. Python 3.11.15, `torch==2.9.1+cu128`, CUDA 12.8, NVIDIA GeForce RTX 5080 Laptop GPU. `pyproject.toml` pins `torch==2.9.1`. GPU time this stage: 0 h. Running total: 0.00 / 7.00 h.

Script: `experiments/flow_matching_simplified/regenerate_onewaytime.py`. Numbers: `outputs/flow_matching_simplified/results/onewaytime_regen.json`. Old parquet SHA-256 values were unchanged after the write.

## Files

| File | Role |
|---|---|
| `data/vspdb_vs_profiles.parquet` | Source. 354,220 rows, 2,369 ids. Not modified. |
| `data/vspdb_tts_profiles_onewaytime.parquet` | `compute_tts` (`vs_calculation.py` line 349). 354,461 rows. The extra 241 rows are the depth-0 rows that function inserts. |
| `experiments/flow_matching_simplified/data/breakpoints_onewaytime.parquet` | `extract_breakpoints` (`breakpoint_analysis.py` line 27). 15,703 rows, 2,369 profiles. |
| `depth_dependent_stats/real_depth_stats_onewaytime.pkl` | Train split only, 1,895 profiles, 10,600 layer `Vs` values. |
| `depth_dependent_stats/real_correlations_onewaytime.pkl` | Same train profiles. The estimator is `compute_profile_statistics` (`vertical_correlation.py` line 71), the layer-center function, not the lag-pair estimator from `eval_corrected.py`. |

`data/vspdb_tts_profiles.parquet` and `experiments/flow_matching_simplified/data/breakpoints.parquet` were not overwritten.

## Verification

On the new fine grid, `Δz/Δt` matches `vs_value` on 352,092 / 352,092 layers (0 mismatches). `tests/vs_profiles/test_oneway_convention.py::test_oneway_parquet_inverts_to_vs_value` passes. The legacy-file test still fails, relative error 0.500, because that file was left doubled.

Bound rate, every value in [100, 5000]:

| Set | Layer rate | Profile rate (every layer inside) |
|---|---|---|
| Raw `vs_value` rows | 0.9940 (352,104 / 354,220) | — |
| One-way fine grid, layer `vs_value` | 0.9939 (same 352,092 layers as the old audit) | 0.9266 (2,195 / 2,369) |
| One-way breakpoints, `Vs = Δz / Δt` | 0.9746 (12,995 / 13,334) | 0.9278 |

The misses are catalog velocities below 100 m/s (334 breakpoint layers below, 5 above), not a leftover factor of 2. A checked example is profile 47, Vs 58 m/s from 0 to 1 m, and `Δz/Δt` returns 58.

Breakpoint depths match the old file on all 2,369 profiles. Positive breakpoint travel times are exactly half the old travel times (max absolute error from 1/2 is 0). The curve geometry is unchanged. The time unit is now one-way.

Depth-bin mean `ln(Vs)` on the new train targets sits `ln 2` above the old pickle. Across 50 bins the max absolute gap from that shift is 9.6e-7. Both pickles hold 10,600 values. The new mean `ln(Vs)` runs from 5.466 to 8.101. Training-target correlations at 1 m and 2 m are 0.0135 and 0.0524.

## Split

`outputs/flow_matching_simplified/results/split_indices_seed42.json` was not edited. Loading the new breakpoints with `FlowMatchingDataLoader.load_data` (`data.py` lines 186–211) yields the same profile-id order as the old file. Train / val / test ids at those indices match: 1,895 / 236 / 238.

Targets were fit on `split["train"]` only. `precompute_statistics.py` line 118 still permutes without a seed; this rebuild did not call that line.

## Open questions

- Profile-level bound rate is 0.927, not 0.99, because some catalog layers are outside [100, 5000]. The fine-layer rate is 0.9939. Stage 3 says to stop if the real bound rate is not near 1.0. Say whether 0.9939 on layers is enough to train, or whether those profiles should be dropped first.
- The correlation pickle is still the layer-center target the flagship already trains against. Say if Stage 3 should keep it.
- Stage 3 is one 1,000-epoch run, about 1.6 GPU-hours, plus the Gaussian fit and the corrected eval. I need a yes before starting it.
