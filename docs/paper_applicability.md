# Paper applicability: Gaussian-trend prior as a stand-in for G_φ

Source: the full text of Malnick, Rusanovsky, Fried, and Avidan, "Designing Optimal Transport Flows" (arXiv:2606.04092). Every **[paper]** claim below was checked against that text. None is abstract-only. Soil-specific design choices are marked **[proposal]**.

This stage used no GPU time. Running total: 1.36 / 7.00 h.

## Scope

The paper's G_φ is a learned generator for the coarse prior. For this stage, the prior is a fixed-grid Gaussian depth-trend sample, fit on the training split. This tests the coupling idea without training a second model. It cannot test the paper's full pipeline, which the authors show is limited by G_φ quality **[paper, Sec. 6: replacing G_φ with true low-frequency projections of training images improves CIFAR-10 FID at 1 NFE from 90.16 to 28.12; the experiment is Sec. B.2]**. A negative result here therefore does not rule out a learned G_φ.

## 1. Can a low-frequency projection be defined for ln(Vs) on a fixed grid?

Yes. **[proposal]** Define the projection on a fixed depth grid:

- Sample ln(Vs) at fixed knots (for example, every 10 m to 200 m, giving 20 values).
- Downsample by averaging adjacent knots (factor 2 or 4), then upsample by linear interpolation.
- Dimension of the prior: 5 to 10 values, depending on the factor.

The paper uses area downsampling for D and bicubic upsampling for U **[paper, Alg. A.1 and App. D Remark; Sec. 4.2 defines T = U ∘ D without naming the interpolators]**. The linear 1D analogue is the closest match. The App. D Remark says area-then-bicubic "is not strictly orthogonal" and that the OT structure "is preserved empirically" on images (Fig. 3). Nearest-neighbor down and up is the case they prove is an orthogonal projection. The soil operator is a new domain, so the OT claim has to be checked on our data **[proposal; the per-domain warning is paper, Sec. 6]**.

The paper's factor of 4 is per spatial axis on images **[paper, Sec. 5.1 and App. A.1]**. CIFAR-10 goes from 32×32×3 to 8×8×3, which keeps 1/16 of the dimensions. That is not the same as keeping 1/4 of a 1D profile. Soil profiles need their own factor, chosen on validation.

## 2. Can the Transformer stay as the vector field?

The paper says the flow model needs no architectural change **[paper, Sec. 4.2: "no architectural changes"; abstract: "no modifications to the flow model itself"]**. That holds only if the prior and the data share the same representation. This is the main risk for our case.

The paper's identity coupling pairs x0 and x1 in one fixed space. Our data is a variable-length sequence of [travel time, depth] tokens, with a different length for each profile. The prior is on a fixed grid. Without a fix, the identity pairing is undefined.

**[proposal]** Change three points:

1. **Training loop:** build x0 for each profile from that profile's own coarse ln(Vs), add noise at α = 0.5 **[paper, Eq. 5; α = 0.5 is the operating point in Sec. 4.1]**, then convert it to breakpoints at the profile's own depths. Pair x0 with that profile's real breakpoints.
2. **Inference:** replace the N(0, I) draw with the Gaussian-trend sample, converted to breakpoints at the chosen breakpoint depths.
3. **Conditioning:** keep the existing sequence-stats conditioning vector, and optionally add the coarse profile.

The Transformer and all losses stay unchanged.

The training draw has to be the projection of the same profile. Eq. 5 is x0 = (1 − α) T(x1) + α ε, with T computed from that x1 **[paper, Sec. 4.2]**. An independent draw from the Gaussian trend, paired with whichever profile is in the batch, is the independent coupling the paper says produces crossing paths. The Gaussian trend stands in for G_φ, and G_φ is used at inference **[paper, Fig. 2 and Sec. 4.2]**. It is not the training target.

## 3. How is the coarse prior defined when breakpoint count varies?

**[proposal]** The prior is defined in ln(Vs) on the fixed grid. Each profile's breakpoint depths are then chosen (count from the training histogram, as now), and the prior's ln(Vs) is interpolated to those depths. Travel time is computed from the prior's velocities with the same Passeri rule used for the data, Δt = Δz / Vs.

Trade-off: the identity coupling holds in the ln(Vs) space, but the breakpoint representation adds a depth-dependent mapping. This mapping may break the exact identity pairing, so the OT-preservation test must be run in the fixed-grid space, not in breakpoint space.

## 4. Does the prior respect 100–5000 m/s?

Not automatically. **[proposal]** Sample the Gaussian trend with per-knot truncation to [ln 100, ln 5000], and record the fraction of samples that needed truncation. Truncation makes the prior slightly less Gaussian, which should be reported. The Gaussian baseline already has a bound rate of 0.945; the flagship's is 0.675. Any gain from the prior must not come with a bound loss.

## 5. What result would confirm the idea?

Written before the runs, consistent with the Stage 6 decision rule:

- At 5 to 10 Euler steps, the prior-based flow matches the 50-step flagship on vertical correlation at 1 and 2 m, within bootstrap intervals.
- Diversity (pairwise distance) matches or exceeds the flagship, against real profiles.
- Curvature under **[paper, App. A.3]** is lower than the flagship's at equal steps. The formula is C = E_t ||(x1 − x0) − v_θ(xt, t)||², on 10,000 generated samples and 128 Euler steps. This is the paper's own mechanism.
- Trend RMSE and bound rate do not get worse.

The target is the 1 m correlation gap: 0.029 for the flagship, against 0.001 for the Gaussian.

**OT-preservation test [proposal, following paper Sec. 4.1]:** for each noise level α, compute the Hungarian assignment between prior samples and their profiles in the fixed-grid ln(Vs) space, and report the fraction whose optimal match is the identity pairing. On images the paper reports 99.9% on CIFAR-10 pixel space and 100% on FFHQ latents, up to α ≈ 0.5, on 10,000 pairs **[paper, Fig. 3 and Sec. 4.1]**. The soil version may differ.

## 6. Would a fixed-grid model with the same prior do as well?

The paper does not test this, so it is an open question. **[proposal]** Replace the Transformer with a fixed-grid flow (for example, an MLP over the 20 knots) with the same prior and losses. If it matches the Transformer, the Transformer's variable-length handling is not contributing on 1D data. That would be the most useful result of Stage 6 for deciding where to spend the remaining budget.

## Recommendation

**Proceed with changes.** The paper's coupling idea is general, but its evidence is for images. The soil version needs:

- A fixed-grid representation of the prior, mapped to breakpoints at each profile's depths.
- Training pairs built from each profile's own projection, not from an independent Gaussian draw.
- An OT-preservation test in the fixed-grid space.
- Truncation to bounds, with the rate reported.
- The fixed-grid control from point 6, timed before it is given three seeds.

The Gaussian-trend stand-in is a valid first test of the coupling at inference. A learned G_φ is a separate, later experiment. G_φ adds about 5–10% on top of the main model's training time **[paper, App. A.2: CIFAR-10 1.5 h on 28 h, FFHQ 2.5 h on 25 h, MeanFlow ImageNet 24 h on 260 h]**, so it would be cheap to add if Stage 6 is promising.

## Open questions

1. Stage 6 currently uses 3 seeds of the Gaussian-trend prior. Should the fixed-grid control be added in Stage 6?
2. Is per-knot truncation acceptable for the prior, or should the prior be sampled by rejection?
3. The paper does not say whether xt in App. A.3 is the Euler state or the straight chord. Our reported curvature uses the Euler state (`run_corrected_eval.py` `paper_curvature_clean`, lines 163–208). Should that stay?

Answers, not yet applied to any run:

1. Not as three more Transformer runs. Three flagship seeds are 4.02 h, and 5.64 h remain, so a second set of flagship-sized runs does not fit. The control is the right comparison. Time one fixed-grid seed first, then decide whether three seeds fit.
2. Per-knot truncation, with the fraction reported. It is the proposal in section 4. Rejection of a whole profile would change which draw is paired at inference. The training projection should stay the real profile, including the catalog layers that fall outside [100, 5000].
3. Yes. App. A.3 evaluates 128 Euler steps of the learned trajectory, and the residual is against the net chord (x1 − x0). That is the Euler state, with x1 the Euler endpoint. `baseline_metrics.curvature` (lines 310–331) evaluates v_θ on the straight chord instead. That function is a different number and is not the Stage 3 curvature.
