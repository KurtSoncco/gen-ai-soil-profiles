# Stage 0 snapshot

No files were discarded and nothing was committed.

## Git

| Item | Value |
|---|---|
| Commit | `a8f629e3a14b84b541b45a41f26e1fb4bfb7426d` |
| Message | Rewrite README around the flow matching flagship model |
| Previous branch | `main`, tracking `origin/main` (`https://github.com/KurtSoncco/gen-ai-soil-profiles.git`) |
| Branch now | `fix/onewaytime-and-structured-prior` |
| Tracked diffs | none |

Untracked files, left in place:

- `docs/audit_flow_matching.md`
- `docs/baseline_flow_matching.md`
- `docs/eval_fixes.md`
- `docs/paper_applicability.md`
- `experiments/flow_matching_simplified/SUMMARY.md`
- `experiments/flow_matching_simplified/baseline_metrics.py`
- `experiments/flow_matching_simplified/bias_analysis.py`
- `experiments/flow_matching_simplified/eval_corrected.py`
- `experiments/flow_matching_simplified/run_corrected_eval.py`
- `experiments/flow_matching_simplified/stratified_hmax_test.py`
- `test_integration.png`
- `tests/flow_matching/test_eval_corrected.py`

Ignored and still on disk: `data/vspdb_tts_profiles.parquet`, `experiments/flow_matching_simplified/data/breakpoints.parquet`, `outputs/flow_matching_simplified/checkpoints/latest.pt`, `outputs/flow_matching_simplified/results/corrected_eval.json`.

## Environment

| Item | Value |
|---|---|
| Python | 3.11.15 |
| `torch.__version__` | `2.9.1+cu128` |
| CUDA | 12.8 |
| GPU | NVIDIA GeForce RTX 5080 Laptop GPU |
| `pyproject.toml` pin | `torch==2.9.1` (extra index `cu126`) |
| Commit | `a8f629e3a14b84b541b45a41f26e1fb4bfb7426d` |

GPU time this stage: 0 h. Ledger: `docs/gpu_budget.md`. Authorized total: 7 h.

## Open questions

- The untracked docs from the previous audit stay on this branch. Later stages rewrite `docs/audit_flow_matching.md` and `docs/paper_applicability.md`. Say if those drafts should be copied aside first.
- Stage 1 is the convention audit and the one-line change in `vs_calculation.py`. I need a yes before starting it.
