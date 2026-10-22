# Stage 9: constrained decoder and Vs30

The Stage 8 layer-token model is valid and matches adjacent-layer residual ρ. The remaining errors are global: generated thicknesses do not sum to the conditioned H, Vs30 is high, and the 0–10 m thickness band is the weakest. This stage puts those constraints in a hand-designed decoder and in the condition. It does not train a graph VAE.

CPU diagnostics: `python -m experiments.flow_matching_simplified.stage9_diagnose` writes `outputs/flow_matching_simplified/results/stage9_diagnose.json`. Training: `python -m experiments.flow_matching_simplified.train_constrained --seed 42 --epochs 500 --tag s42_e500`. Scoring: `python -m experiments.flow_matching_simplified.stage9_score --tag s42_e500`.

## 1. Where the Vs30 bias lives

Three Stage 8 layered seeds, latest checkpoint, 200-sample CPU draws. Time-averaged Vs and median layer Vs:

| Set | Vs5 | Vs10 | Vs30 | Layer Vs 0–10 m | 10–30 m | 30–100 m | 100 m+ |
|---|---|---|---|---|---|---|---|
| Real test | 229 | 271 | 309 | 229 | 347 | 452 | 1138 |
| Layered s42 | 247 | 293 | 365 | 251 | 417 | 614 | 1100 |
| Layered s43 | 278 | 330 | 402 | 282 | 443 | 597 | 1136 |
| Layered s44 | 253 | 304 | 376 | 260 | 438 | 594 | 1061 |

The bias starts in the top 5 m and is largest between 10 m and 100 m. The deepest band is fine. Conditioning on Vs_z (Vs30, or Vs_H when the profile is shallower) is the right target; a surface-only fix would not be enough.

## 2. Inference-time H renormalize

`h ← h · H / sum(h)` on the same three draws, using the H that was fed as the condition. Residual ρ does not move (0.759–0.770). Thickness KS improves a little (0.137→0.124 on s42). Vs30 gets worse (KS 0.168→0.216, 0.281→0.285, 0.181→0.232; medians 365→385, 402→408, 376→387). The simplex map has to be trained. Patching H after the fact trades one global error for another.

## 3. Novelty

`nearest_train_distance` is the minimum interpolated-profile distance to the training set, using the same 100-point `[tts, depth]` interpolation as the pairwise metric. The headline is the median of those minima, and the fraction of samples closer to train than the median test-to-train distance (0.036).

| Set | Novelty median | Fraction closer than test |
|---|---|---|
| Real test (self) | 0.036 | 0.50 |
| Layered s42 / s43 / s44 | 0.78 / 0.85 / 0.72 | 0.01 / 0.01 / 0.00 |
| Flagship | 0.60 | 0.05 |
| Toro | 0.060 | 0.30 |
| Gaussian | 0.049 | 0.40 |

The interpolation includes the depth coordinate, so a mismatch in H inflates the distance. After H-renormalize the layered medians fall to 0.041–0.049, next to the test value, with 0.41–0.44 of samples closer than the test median. Layered profiles are not copies once H is aligned. Toro and the Gaussian already draw H from train, which is why they sit near the test baseline. The metric is still useful as a copy check after the simplex decoder makes H exact.

## 4. Model

Tokens `[log(h/H), s]` with `s` the Stage 8 logit-squash of ln Vs. Decode `h = softmax(u) · H`. Condition `[ln H, n/20, ln Vs_z]`. The interpolant starts from a Toro profile with the same `n` and `H` (merge thinnest / split thickest, then Toro Vs). Same transformer, 500 epochs, three seeds. `--no-toro-source` falls back to Gaussian noise if the smoke run collapses ρ or thickness.

## 5. Retrain

The 200-epoch smoke (`smoke_s42_e200`) kept residual ρ at 0.772 and dropped the Vs30 median from 365 m/s to 321 m/s. Thickness and ρ did not collapse, so the Toro source stayed on for the three 500-epoch seeds.

Headline sampler: latest checkpoint, `(n, H, Vs_z)` drawn jointly from one training profile. Range over seeds 42/43/44.

| Set | Validity | Trend RMSE | Residual ρ | Residual s.d. | Thickness KS | Vs30 KS | Vs30 median | 0–10 m thick KS | ln f0 KS | Final-depth KS | Novelty median |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Real test | 1.000 | — | 0.772 | 1.06 | — | — | 309 | — | — | — | 0.036 |
| Constrained, median of 3 | 1.000 | 0.081 [0.076–0.102] | 0.795 [0.790–0.799] | 0.95 [0.94–0.97] | 0.109 [0.104–0.124] | 0.092 [0.075–0.122] | 303–318 | 0.078 [0.071–0.105] | 0.119 [0.095–0.120] | 0.050 [0.047–0.055] | 0.034 [0.029–0.035] |
| Layered, median of 3 | 1.000 | 0.132 | 0.763 | 1.10 | 0.122 | 0.181 | 365–402 | 0.14–0.17 | 0.096 | 0.09–0.15 | 0.72–0.85 |
| Toro (1995), fitted | 1.000 | 0.261 | 0.774 | 0.98 | 0.160 | 0.139 | — | — | 0.186 | — | 0.060 |
| Flagship, 1 seed | 0.805 | 0.241 | 0.669 | 1.27 | 0.115 | 0.107 | — | — | 0.100 | — | 0.60 |

Independent-`Vs_z` control (same checkpoints, Vs_z shuffled against `(n, H)`): Vs30 KS 0.070–0.079, so the condition is used. Trend RMSE rises to 0.15–0.19 when the triple is inconsistent, which is expected.

What closed:

- **Exact H.** Final-depth KS is 0.05, the train-vs-test H gap, not the 0.09–0.15 mismatch of generated vs conditioned depth.
- **Vs30.** Median 303–318 m/s against a real 309. KS 0.075–0.122, better than Toro (0.139) and the layered model (0.17–0.28). The stiffness from 0–100 m is gone.
- **Shallow thickness.** 0–10 m KS 0.071–0.105 against 0.14–0.17.
- **Residual s.d.** 0.94–0.97, next to Toro's 0.98, down from the layered 1.10.
- **Novelty.** Median min-distance 0.034 against a test-to-train 0.036; about half the samples are closer than the test median, the same as the test set against itself. The layered 0.72–0.85 was almost all H mismatch.

What moved the other way:

- **Residual ρ is 0.018–0.027 high** (0.790–0.799 against 0.772). The Toro source pulls correlation up. Still much closer than the Gaussian (0.914) or the flagship (0.669).
- **f0 median 7.7–8.0 Hz** against a real 9.5 Hz. KS is similar to the layered model.

Caveats: bound rate 1.0 by construction (2.7% of training layers were clipped); `(n, H, Vs_z)` still copied from a training profile; 500 epochs; one split; one flagship seed. The independent-`Vs_z` draw is the check that the model is not only copying the triple.

GPU: smoke 149 s plus three seeds 372 + 365 + 389 s (0.35 h). Ledger total 7.10 of 10.0 h.
