# GPU budget

Authorized for this task: **10.0 h**. Stop and ask before exceeding it.

Stage 9 raised the cap from 7.0 h to 10.0 h before the first constrained-decoder train.

| Stage | What ran | Hours | Running total |
|---|---|---|---|
| 0 | Snapshot only. CUDA name query, no training. | 0.00 | 0.00 |
| 1 | Audit and `compute_tts` edit. Pytest 3.67 s, no training. | 0.00 | 0.00 |
| 2 | Rebuild one-way parquet, breakpoints, and train-split targets. CPU only, about 3 s. | 0.00 | 0.00 |
| 3 | One 1,000-epoch train (4,806.8 s) plus corrected eval (86.6 s). | 1.36 | 1.36 |
| 4 | Audit write-up. No training. | 0.00 | 1.36 |
| 5 | Paper-applicability note. No training. | 0.00 | 1.36 |
| 6a | Fixed-grid timing only: 1 warmup + 10 epochs. Wall 9.3 s. | 0.00 | 1.36 |
| 6b | Three trend-prior flagship seeds, 1,000 epochs each. Wall 17,323 s. | 4.81 | 6.17 |
| 6c | Trend and oracle eval, then three fixed-grid seeds and their eval. Wall 663 s. | 0.18 | 6.35 |
| 7 | Bound-knot catalog and a 200-profile resample of the three trend checkpoints. Wall about 16 s. No training. | 0.00 | 6.35 |
| 8 | Layer-token model: timing run (71 s), seed 42 at 500 epochs (about 560 s), seeds 43 and 44 at 500 epochs (796 s for both). Scoring on CPU. | 0.40 | 6.75 |
| 9 | Constrained decoder: 200-epoch smoke (149 s), three 500-epoch seeds (372 s, 365 s, 389 s). Scoring on CPU. | 0.35 | 7.10 |
| 10 | Source ablation: no-Toro seed 42 (281 s), mix-50% three seeds (365 + 366 + 342 s). Scoring on CPU. | 0.38 | 7.48 |

Remaining: 2.52 h.

The three flagship seeds used 4.81 h, more than the 4.02 h estimate from the 80.1 min Stage 3 run. Evaluation and the fixed-grid seeds finished inside the remainder. Stage 9 used 0.35 h of the +3.0 h raised before that train. Stage 10 used 0.38 h.
