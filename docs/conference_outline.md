# Conference outline

A short-paper outline for an ML-for-science or geotechnical venue, about eight pages. This is not a submission draft. Numbers are the Stage 3–7 measurements. Checkpoints are not in git.

The Transformer is not state of the art on the metrics this project froze. The outline says what is ahead on this benchmark, which claims a paper can make now, and which ablations an SOTA sentence would still require.

## 1. Problem

Sample shear-wave velocity profiles `Vs(z)` whose vertical correlation matches real boreholes. A profile is a variable number of layers. Downstream seismic-hazard work needs velocities inside a physical range and a depth trend that is not an artifact of the travel-time definition.

## 2. Convention and protocol

One-way travel time is `Δt = h / Vs`. Inversion is `Vs = Δz / Δt`. The legacy parquet that stored `2h / Vs` is not the training file.

The split is seed 42: 1,895 train, 236 validation, 238 test. It does not change with the run seed. Headline sampling is Euler, 50 steps, 200 samples. RK4 was swept on the Stage 3 checkpoint and did not change the ordering. Trend error is the n-weighted RMSE of mean `ln(Vs)` at 10 m through 200 m. Vertical correlation is a 0.5 m grid. The bound rate is the fraction of profiles whose every layer lies in [100, 5000] m/s.

The Stage 6 keep rule, written before those runs: keep a structured prior only if the median of three seeds beats both the Stage 3 flagship and the fixed-grid Gaussian on 1 m correlation, 2 m correlation, or pairwise distance, by more than the min-to-max range of those three seeds, and every seed keeps trend RMSE at or below 0.272 and a bound rate at or above 0.675.

## 3. Models

- Transformer conditional flow matching. Four layers, eight heads, hidden size 256, about 3.4 million parameters. Tokens are raw one-way travel time and depth, at most 20 breakpoints. The condition is the per-profile mean and standard deviation of those two channels. The default path is `N(0, I)`.
- Fixed-grid Gaussian. Bootstrap training depths, draw `ln(Vs)` in 10 m bins with exponential correlation, emit `Δt = Δz / Vs`. This is a baseline, not an ODE initial state.
- Fixed-grid MLP flow. The same 20 knots, standardized catalog `ln(Vs)`, a four-layer MLP with 344,852 parameters, and the same Gaussian-trend start at inference.
- Gaussian-trend prior. A stand-in for the coarse generator `G_φ` in "Designing Optimal Transport Flows" (arXiv:2606.04092). Training uses each profile's own factor-4 projection, `α = 0.5`. Inference draws the training distribution of that projection and clips knots to [ln 100, ln 5000]. The oracle start, which uses the test profile's own projection, is diagnostic and does not enter the keep rule.

The Vs penalty, weight `1e-2`, is applied to the interpolated state and can affect the vector field. The depth-statistic and vertical-correlation terms are computed on the real batch and do not.

## 4. Results already measured

Stage 3, one Transformer seed, Euler 50, from [`docs/stage3_baseline_clean.md`](stage3_baseline_clean.md):

| Model | Trend RMSE | Bound rate | Corr 1 m | Gap 1 m | Pairwise |
|---|---|---|---|---|---|
| Real test profiles | — | 0.950 | 0.993 | — | 708 |
| Fixed-grid Gaussian | 0.228 | 0.945 | 0.994 | 0.001 | 545 |
| Transformer `latest` | 0.272 | 0.675 | 0.964 | 0.029 | 524 |

Stage 6, three trend-prior seeds, from [`docs/stage6_structured_prior_stage1.md`](stage6_structured_prior_stage1.md). The structured prior, as specified, does not beat the Stage 3 flagship or the Gaussian on the decision metrics. Trend RMSE is preserved (0.092, 0.144, 0.130). The bound rate is not uniform: 0.625, 0.680, 0.595. The prior is dropped. One prior design was tested.

The fixed-grid MLP medians are about 0.997 and 0.993 at 1 m and 2 m, with bound rates 0.990 to 1.000. Pairwise does not clear the three-seed range. The comparison is three MLP seeds against one flagship seed and one Gaussian draw.

Oracle curvature on the trend-prior checkpoints is about 2, against 18–24 from the Gaussian-trend start. The 1 m gap stays at 0.016 and 0.019 on two of three seeds. Straight paths do not produce the eval correlation.

Stage 7, from [`docs/stage7_bound_and_correlation.md`](stage7_bound_and_correlation.md). The 21 unclipped projection knots do not account for the sampler's bound violations. Those violations are mostly layers that end by 10 m, with median Vs of a few metres per second. The correlation weight does not train `v_θ`. Its stored 1 m target is 0.0135. The eval correlation on real profiles is 0.993.

## 5. What is ahead on this benchmark

On correlation and bounds, the fixed-grid Gaussian is ahead of the Transformer. On correlation, the fixed-grid flow is ahead of both the single flagship seed and that Gaussian, with the seed caveat in the next section. The Transformer's variable-length tokenization is the reason it can emit a varying layer count. That is not, by itself, a claim that it matches the borehole correlation or the Passeri bounds.

A paper can call the Gaussian the strongest simple baseline on this protocol. It cannot call the Transformer state of the art.

## 6. Ablations required before an SOTA sentence

Each row is a claim that the current evidence does not support, and the run that would be required before writing the sentence.

| Claim an SOTA sentence would need | Why the evidence is short | Ablation |
|---|---|---|
| The Transformer beats a Gaussian prior across training noise | The flagship comparator is one seed | Repeat the Stage 3 `N(0, I)` run at seeds 43 and 44, same split, same Euler-50 protocol |
| The fixed-grid flow is the model to use | Three MLP seeds against one flagship seed and one Gaussian; pairwise misses its own range | More fixed-grid seeds, and a second Gaussian draw protocol if the baseline itself has seed noise |
| The Transformer beats the VAE and the GAN | The published VAE table is Vs30. It is a different metric | Rerun both on the one-way breakpoints, or publish both on one shared grid, with the same correlation, bound, trend, and pairwise scores |
| The vertical-correlation loss is why the gap is small, or why it remains | The term does not train `v_θ`, and it targets 0.0135 rather than the eval correlation 0.993 | A loss on the integrated sample, using the 0.5 m-grid estimator, with the weight chosen on validation |
| The bound rate is a solved physics constraint | Weight `1e-2` was not varied. Stage 3 is 0.675. The Gaussian is 0.945 | A validation sweep of the penalty weight, then one test report |
| A structured prior helps, or a learned `G_φ` does | The Gaussian-trend stand-in was dropped. Factor `k = 4` was not swept. The paper's own negative-stand-in warning still applies | `k` on validation, then a learned `G_φ` only if that validation says the stand-in was the limit |
| The result holds for 2D or 3D spatial correlation | Every number here is a 1D borehole | A separate dataset and a separate protocol. Do not import the 1D ranking |

Until those rows are filled, the honest sentence is: on this 1D one-way protocol, a fixed-grid Gaussian matches the measured correlation, and a small fixed-grid flow is competitive with it, while the variable-length Transformer does not lead.

## 7. Claims the paper can make, and claims it cannot

It can say:

- The travel-time convention is one-way, and the legacy doubled file is the wrong training target for `Vs = Δz / Δt`.
- Under the frozen rule, the Gaussian-trend prior does not beat the Stage 3 flagship or the Gaussian. Trend accuracy of that prior is preserved. The bound rate is not uniformly lost, and the 21-knot explanation is not supported.
- The vertical-correlation term, as implemented, does not train the vector field.
- An oracle start makes the trajectory nearly straight and does not, on two of three seeds, remove the 1 m correlation gap.

It cannot say:

- The Transformer is state of the art.
- Geostatistics losses improved vertical correlation.
- The fixed-grid MLP should replace the Transformer. That needs more seeds.
- A learned `G_φ` would fail. Stage 6 tested one stand-in.
- The VAE Vs30 table has been surpassed. The metrics differ.

## 8. Open experiments

Remaining GPU budget on the ledger in [`docs/gpu_budget.md`](gpu_budget.md) is 0.65 h of 7.00 h. A new flagship-sized training run does not fit. The ablations in section 6 that retrain the Transformer are a separate decision, after this budget.

No further training is authorized by this outline.
