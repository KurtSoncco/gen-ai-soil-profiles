# Stage 8: critique of the current generation

No new training for the critique itself. Diagnostics: `python -m experiments.flow_matching_simplified.stage8_diagnostics` (CPU, about 30 s), which writes `outputs/flow_matching_simplified/results/stage8_diagnostics.json`. Checkpoint: `latest_onewaytime.pt`, the Stage 3 flagship, Euler 50, 200 samples per draw.

## Summary

The main problem is the representation. The flagship generates cumulative `[one-way travel time, depth]` coordinates, z-scored with each profile's own statistics, and recovers Vs as the ratio of two differences of generated numbers. Nothing keeps those differences positive or inside [100, 5000] m/s. None of the auxiliary training terms reaches the network, so the flow is trained on the conditional flow-matching loss alone. The headline correlation metric mostly measures layer thickness, drops invalid samples, and ranks the Gaussian above the flow, while a trend-removed measure between adjacent layers puts the Gaussian further from the real data than the flow.

Section 9 retrains the same transformer on per-layer `[ln h, bounded ln Vs]` tokens. Over three seeds, every sample is valid, the adjacent-layer residual ρ is 0.759–0.768 against a real 0.772, and the depth trend error is about half the flagship's. Vs30 is still biased high, and shallow thickness is the weakest band.

## 1. The representation cannot reliably produce thin, positive layers

Tokens are absolute cumulative values, normalized per profile (`data.py` lines 70–84). Vs is `Δz / Δt` between neighbouring tokens, with the surface prepended (`eval_corrected.py` lines 43–58).

The precision this needs, in the normalized space the network works in, measured on the 1,895 training profiles:

| Quantity, in units of that profile's channel s.d. | Median | 10th percentile |
|---|---|---|
| First-layer thickness | 0.260 | 0.045 |
| Smallest increment, depth or time | 0.149 | 0.035 |

Vs is a ratio of two such increments. An endpoint error of a few hundredths of a unit can flip a sign or put Vs at a few metres per second.

What the flagship emits, two noise seeds, with length and sequence statistics drawn independently, as the sampler does, or from the same training profile ("matched"):

| Draw | Bound rate | First token at or above surface | Both channels strictly increasing | Profiles with Vs ≤ 0 or non-finite | Violating first layers | Violating deeper layers | Layers under 1 m |
|---|---|---|---|---|---|---|---|
| Real test (238) | 0.950 | 0.000 | 1.000 | 0 | — | — | 0.019 |
| Seed 42, independent | 0.730 | 0.090 | 0.905 | 14 | 40 | 51 | 0.109 |
| Seed 42, matched | 0.700 | 0.155 | 0.800 | 33 | 41 | 52 | 0.048 |
| Seed 43, independent | 0.675 | 0.105 | 0.865 | 21 | 47 | 65 | 0.121 |
| Seed 43, matched | 0.745 | 0.105 | 0.860 | 21 | 44 | 25 | 0.049 |

About half the violating layers are the first layer, which agrees with Stage 7 (violations end by 10 m, median Vs a few m/s). Pairing length and statistics from one profile does not move the bound rate in a consistent direction. In the training data, length is rank-correlated with depth mean (0.32) and depth s.d. (0.48), so the independent draw is a real inconsistency, but it is not the cause of the bound failures.

The established geostatistical models do not generate cumulative coordinates. Toro (1995) draws layer interfaces from a depth-dependent Poisson process and ln Vs per layer around a median, with AR(1) correlation between adjacent layers. Passeri et al. (2020) also randomizes layer thickness with a Poisson process and per-layer travel time in log space, again with AR(1) correlation. Both produce positive thicknesses and velocities by construction.

## 2. Only the flow-matching loss trains the network

The trainer adds the Vs penalty on `ut = (1 − t)·u0 + t·u1` (`train_with_depth_losses.py` lines 141–153). `ut` is built from noise and the real batch and contains no model output. In the diagnostic, the penalty returned by `compute_vs_penalty` has `requires_grad = False`, and the summed absolute gradient over all weights is 0.

Earlier statements that this term "can affect `v_θ`" (README, Stage 1, Stage 7, conference outline) were wrong and have been corrected. Stage 7 had already shown the depth-statistic and correlation terms are detached. The flagship is therefore an unconstrained conditional flow-matching model.

A penalty that trains the network has to see its output. Physics-Informed Diffusion Models (Bastek, Sun, and Kochmann, ICLR 2025) put the residual on the predicted clean sample. The flow-matching analogue is `x̂1 = ut + (1 − t)·v_θ(ut, t)`. Physics-Constrained Flow Matching (Utkarsh et al., NeurIPS 2025) instead enforces hard constraints at inference, by projecting intermediate states. A representation where the bounds hold by construction makes either unnecessary for the bounds.

## 3. The evaluation misranks the models

**The 0.5 m-grid correlation is mostly a layer-thickness statistic.** It is a Pearson correlation of ln(Vs) pooled across profiles. On the real test set, 94.8% of 1 m pairs and 90.2% of 2 m pairs are in the same layer, where the correlation is 1 by construction, and the pooled values also carry variation between sites. A model gets close to 0.99 by emitting thick layers.

**A trend-removed measure between adjacent layers.** Subtract the training median ln(Vs) in depth bins, divide by the bin s.d., then correlate adjacent layers, as Toro's ρ does. Same 200-sample Stage 3 scoring draw:

| Source | Adjacent-layer residual ρ | s.d. of residual |
|---|---|---|
| Real test | 0.772 | 1.06 |
| Transformer flow | 0.669 | 1.26 |
| Fixed-grid Gaussian | 0.914 | 0.97 |

The flow is 0.10 too low and too dispersed. The Gaussian is 0.14 too high. The headline 1 m gaps were 0.029 and 0.001, the other way round.

**Invalid samples are dropped.** `lag_pairs` and `ln_vs_matrix` return nothing for a profile with any non-positive or non-finite Vs. On the scoring draw, the flow's correlation used 160 of 200 profiles and its trend 149 of 200. The bound rate counts them as failures. The trend and correlation columns score only the survivors.

**Two baselines copy real layer boundaries.** `sample_fixed_grid_gaussian` (`baseline_metrics.py` lines 385–388) and the fixed-grid MLP (`eval_fixed_grid.py` lines 96–97) take their layer depths from a random training profile. They generate velocities, not layering. Any metric that depends on layer thickness, including the 0.5 m correlation, compares them against a model that has to generate its own boundaries.

**Engineering checks are missing.** No Vs30, no site period or f0, no transfer function (Passeri's validation uses dispersion curves, transfer functions, and site period), no σ_lnVs against depth, and no layer thickness against depth (Toro's λ(z)).

## 4. The standard baseline is missing

Toro (1995) randomization is still the default in practice, and Passeri et al. (2020) is the model the travel-time convention comes from. A Toro model fitted on the training split is the comparison a geotechnical reviewer will expect before a claim against an ad hoc Gaussian.

## 5. Lower priority

- One flagship training seed.
- At sampling, the model is conditioned on a real training profile's mean and s.d., which reuses training data and limits diversity.
- `DepthConvBlock` mixes neighbouring breakpoint indices, not neighbouring metres, so the "1–2 m coherence" rationale does not hold.

## 6. Evaluation changes

`eval_corrected.py`:

- `is_valid_profile` and `validity_rate`. A profile is valid when depth and one-way travel time both strictly increase from the surface, which is the same as every layer having positive thickness and a positive, finite Vs.
- `layer_model(include_invalid=True)`, and an `include_invalid` flag on `lag_pairs`, `ln_vs_matrix`, `vertical_structure`, `bootstrap_vertical`, and `depth_trend_error`. The default stays `False`, so Stage 3–7 numbers reproduce. With `True`, an invalid profile is scored with Vs = 1 m/s where Δz ≤ 0, 10⁵ m/s where Δt ≤ 0, and clipped elsewhere. `depth_trend_error` now also reports how many generated profiles it used.

`run_corrected_eval._score_set` adds `validity_rate`, the profile count behind each correlation, and `_all_profiles` versions of the trend and correlation.

Clipping does not work as a penalty. On the flagship it lowers the trend RMSE from 0.241 to 0.157 and the 1 m correlation gap from 0.023 to 0.006, because the clipped extremes inflate the variance and partly cancel in the means. The headline therefore follows the convention used for molecule generators. `validity_rate` counts invalid samples as failures. Distribution metrics are computed on valid samples, with the count shown. The clipped variant is stored as a stress check.

`site_metrics.py` adds:

- the adjacent-layer residual correlation (Toro's ρ), against a training median and σ in depth bins;
- σ_lnVs at the 10 m trend depths, as an n-weighted RMSE against the test set;
- layer-thickness KS by depth band of the layer top (0–10, 10–30, 30–100, and over 100 m);
- Vs30 KS on profiles that reach 30 m;
- f0, the peak of the 1D linear SH transfer function (constant density, 2% damping, last layer as half-space), as KS on ln f0.

The fixed-grid Gaussian and the fixed-grid MLP are labelled in code and results as copying layer boundaries from training profiles.

## 7. Toro baseline

`toro_baseline.py` fits Toro (1995) on the seed-42 train split. Layering is a non-homogeneous Poisson process, `λ(z) = c3 (z + c1)^(−c2)`, fitted by maximum likelihood on interior breakpoints, with each profile observed on (0, H]. The median and σ of ln Vs come from 20 depth bins. The inter-layer ρ(d, t) uses Toro's functional form, also fitted by maximum likelihood. Profile depth H is bootstrapped from the training profiles. Vs is not truncated.

Fitted values: c1 = 20.2 m, c2 = 1.33, c3 = 13.8; ρ0 = 0.63, Δ = 52 m, ρ200 = 0.77, d0 = 0.1 m, b = 0.13. The generic Toro (1995) values are c1 = 10.86, c2 = 0.89, c3 = 1.98. Ours describe interfaces between breakpoints of a profile simplified to at most 20 tokens, not interfaces in raw boreholes, so they are not comparable to the generic values.

## 8. Re-score of existing generators

`python -m experiments.flow_matching_simplified.stage8_rescore` (CPU, about 45 s) writes `outputs/flow_matching_simplified/results/stage8_rescore.json`. Each set is 200 profiles against the 238 seed-42 test profiles.

The flagship row uses the Stage 3 sampler and seed, run on CPU. CPU and CUDA random streams differ, so this is not the Stage 3 GPU draw (bound rate 0.685 here, 0.675 there).

| Set | Layering | Validity | Bounds | Scored | Trend RMSE | 1 m corr gap | Residual ρ | Residual s.d. | σ_lnVs RMSE | Thickness KS | Vs30 KS | ln f0 KS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Real test | — | 1.000 | 0.950 | 238 | — | — | 0.772 | 1.06 | — | — | — | — |
| Flagship, Euler 50 | generated | 0.805 | 0.685 | 161 | 0.241 | 0.023 | 0.669 | 1.27 | 0.087 | 0.115 | 0.107 | 0.100 |
| Fixed-grid Gaussian | copied | 1.000 | 0.945 | 200 | 0.228 | 0.001 | 0.914 | 0.98 | 0.169 | 0.082 | 0.118 | 0.360 |
| Fixed-grid MLP s42 | copied | 1.000 | 0.990 | 200 | 0.049 | 0.004 | 0.916 | 1.09 | 0.069 | 0.049 | 0.209 | 0.384 |
| Fixed-grid MLP s43 | copied | 1.000 | 0.990 | 200 | 0.120 | 0.004 | 0.918 | 1.05 | 0.105 | 0.053 | 0.294 | 0.414 |
| Fixed-grid MLP s44 | copied | 1.000 | 1.000 | 200 | 0.089 | 0.004 | 0.911 | 1.05 | 0.061 | 0.074 | 0.176 | 0.449 |
| Toro (1995), fitted | generated | 1.000 | 0.850 | 200 | 0.261 | 0.002 | 0.774 | 0.98 | 0.083 | 0.160 | 0.139 | 0.186 |

"Scored" is the number of profiles behind the columns to its right.

What changes:

- **Toro is the strongest baseline on correlation.** Its residual ρ is 0.774 against a real 0.772, and its 1 m gap is 0.002. It generates its own layering and is always valid. Its ρ is fitted on the train split and scored on the test split, so this is held-out agreement, not a tautology. Its weak points are thickness below 30 m (KS 0.185 at 0–10 m and 0.213 at 30–100 m) and a bound rate of 0.85, because it does not truncate.
- **The Gaussian and the MLP are over-correlated.** Their residual ρ is 0.91–0.92 on every seed. Their near-perfect 1 m correlation comes from copying real layer boundaries. They also have the worst f0 (KS 0.36–0.45, median f0 3–5 Hz against a real 9.5 Hz).
- **The flagship is under-correlated, too dispersed, and invalid on one profile in five.** On the valid profiles, its σ_lnVs, thickness, Vs30, and f0 scores are competitive. Its f0 KS of 0.100 is the best in the table. Its deepest band is thin-layered (median 23.6 m against a real 70 m, KS 0.45).
- No set leads on every column. On the metrics that do not reward copied layering, which are residual ρ, validity, and f0, the generated-layering models (Toro and the flagship) look better than the fixed-grid Gaussian and MLP.

## 9. Layer-token reparameterization

`train_layered.py` keeps the flagship's transformer and conditional flow matching and changes what the tokens are:

- One token per layer, `[ln h, s]`, where h is the layer thickness and `s = logit((ln Vs − ln 100) / (ln 5000 − ln 100))`. Any real token decodes to a positive thickness and a Vs inside [100, 5000] m/s. Catalog layers outside the bounds (2.7% of training layers) are clipped before encoding.
- Global per-channel normalization, not per-profile.
- Condition `[ln H, n / 20]`, both taken from one training profile at sampling time, so length and depth are consistent.
- A penalty of 0.01 × relu(ln h_floor − ln ĥ)² on the predicted clean sample `x̂1 = xt + (1 − t)·v_θ`, so it has a gradient. h_floor is the training 1st percentile, 0.5 m.
- 3.4 M parameters, 500 epochs (half of the flagship's 1,000), AdamW at 1e-4.

Train with `python -m experiments.flow_matching_simplified.train_layered --seed 42 --epochs 500 --tag s42_e500` (GPU, 6–9 min per seed). Score with `python -m experiments.flow_matching_simplified.stage8_score_layered --tag s42_e500` (CPU, about 30 s), which writes `outputs/flow_matching_simplified/results/stage8_layered_<tag>.json`. Same 200-sample scoring as section 8.

Three seeds, final checkpoint (`latest`). The range is shown in brackets.

| Set | Validity | Bounds | Trend RMSE | 1 m corr gap | Residual ρ | Residual s.d. | σ_lnVs RMSE | Thickness KS | Vs30 KS | ln f0 KS |
|---|---|---|---|---|---|---|---|---|---|---|
| Real test | 1.000 | 0.950 | — | — | 0.772 | 1.06 | — | — | — | — |
| Layered, median of 3 | 1.000 | 1.000 | 0.132 [0.124–0.172] | 0.001 [0.001–0.002] | 0.763 [0.759–0.768] | 1.10 [1.10–1.11] | 0.048 [0.047–0.060] | 0.122 [0.122–0.137] | 0.181 [0.168–0.281] | 0.096 [0.094–0.109] |
| Flagship, 1 seed | 0.805 | 0.685 | 0.241 | 0.023 | 0.669 | 1.27 | 0.087 | 0.115 | 0.107 | 0.100 |
| Toro (1995), fitted | 1.000 | 0.850 | 0.261 | 0.002 | 0.774 | 0.98 | 0.083 | 0.160 | 0.139 | 0.186 |
| Fixed-grid Gaussian | 1.000 | 0.945 | 0.228 | 0.001 | 0.914 | 0.98 | 0.169 | 0.082 | 0.118 | 0.360 |

The lowest-validation checkpoint (`best`, epochs 319–459) gives a similar picture: residual ρ 0.755 [0.746–0.767], trend RMSE 0.138 [0.098–0.195], f0 KS 0.077 [0.069–0.112], Vs30 KS 0.229 [0.167–0.246].

What changes:

- **Every sample is valid,** against 0.805 for the flagship, and the scored metrics now cover all 200 profiles.
- **Correlation matches the data.** Residual ρ is within 0.013 of the real 0.772 on every seed, about as close as Toro, where the flagship was 0.10 low. The 1 m gap is at the Gaussian's level without copying layer boundaries.
- **Depth trend and σ_lnVs are the best of the generated-layering models.** Trend RMSE roughly halves against the flagship and Toro, and σ_lnVs RMSE is about 0.05 against 0.08–0.09.
- **f0 holds,** at KS 0.09–0.11 (median 8.3–9.0 Hz against a real 9.5 Hz). The flagship's deep thin-layering is gone: the over-100 m band has a median of 80–90 m against a real 70 m and the flagship's 23.6 m.

What does not improve:

- **Vs30 is worse than the flagship and Toro.** The median is 365–402 m/s against a real 309 m/s, so the shallow profile is too stiff, and the KS spread across seeds is wide (0.17–0.28).
- **Shallow thickness is the weakest band.** At 0–10 m the KS is 0.14–0.17, with layers slightly thinner than real (median 3.0–3.3 m against 3.5 m).
- **The residual s.d. is 1.10,** against a real 1.06. It is less dispersed than the flagship's 1.27 but more than Toro's 0.98.

Caveats:

- A bound rate of 1.000 holds by construction and is above the real 0.95, because catalog layers outside the bounds were clipped. It is not evidence of better fidelity.
- The layer count and H are drawn from a training profile, like the flagship's statistics. The layer count KS is therefore identical across seeds (0.095).
- 500 epochs against the flagship's 1,000, one data split, and one flagship seed. The layered seeds are compared with a single flagship draw.

The reparameterization fixes the first three problems in this note: validity, the penalty that did not train, and under-correlation. It does not yet beat Toro on Vs30 or thickness, which makes Toro the bar for the next stage.

## References

- Toro, G. R. (1995). Probabilistic models of site velocity profiles for generic and site-specific ground-motion amplification studies. Technical Report 779574, Brookhaven National Laboratory. Summarized with the λ(z) and ρ(z, t) forms in Toro (2022), Uncertainty in shear-wave velocity profiles, *Journal of Seismology*, https://doi.org/10.1007/s10950-022-10084-x
- Passeri, F., Foti, S., and Rodriguez-Marek, A. (2020). A new geostatistical model for shear wave velocity profiles. *Soil Dynamics and Earthquake Engineering*, 136, 106247. https://doi.org/10.1016/j.soildyn.2020.106247
- Bastek, J.-H., Sun, W., and Kochmann, D. M. (2025). Physics-informed diffusion models. *ICLR*. https://arxiv.org/abs/2403.14404
- Utkarsh, Cai, P., Edelman, A., Gómez-Bombarelli, R., and Rackauckas, C. (2025). Physics-constrained flow matching: sampling generative models with hard constraints. *NeurIPS*. https://arxiv.org/abs/2506.04171
