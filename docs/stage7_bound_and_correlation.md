# Stage 7: bound violations and the correlation loss

No new training. Two checks from the Stage 6 closeout, both on the existing seed-42 split and the existing checkpoints. Script: `experiments/flow_matching_simplified/stage7_checks.py`. Numbers: `outputs/flow_matching_simplified/results/stage7_checks.json`.

## The 21 projection knots do not account for the bound rates

The Stage 6 hypothesis was that the unclipped training projection teaches the flow values it cannot reproduce, because the same 21 knots sit outside [100, 5000] m/s on every seed. That hypothesis is not supported.

The resample uses the Stage 6 trend-start generator: 200 profiles, Euler 50, the same seeds. The bound rates come back as 0.625, 0.680, and 0.595, the Stage 6 figures, so these are the same draws.

| Source | What is outside | Profiles | Layers or knots | Ending at or before 10 m |
|---|---|---|---|---|
| Training projection, unclipped | 21 knots | 7 / 1,895 | 21 knots | 10 knots, at 10 m or 20 m |
| Training endpoint x1, `Vs = Δz / Δt` | catalog layers | 142 / 1,895 (profile rate 0.925 inside) | 284 / 10,660 | 226 / 284 |
| Trend-start samples, seed 42 | generated layers | 75 / 200 | 131 / 1,102 | 98 / 131 |
| Trend-start samples, seed 43 | generated layers | 64 / 200 | 108 / 1,158 | 79 / 108 |
| Trend-start samples, seed 44 | generated layers | 81 / 200 | 133 / 1,188 | 90 / 133 |

The 21 knots are six shallow profiles with projected Vs from 24 m/s to 95 m/s at 10 m and 20 m, plus one profile (train row 1388) whose projected Vs runs from 5,352 m/s to 7,892 m/s between 100 m and 200 m. Three of the 21 sit on a raw layer that is itself inside [100, 5000]: row 1051 at 20 m (raw 187 m/s), row 1557 at 20 m (raw 184 m/s), and row 1388 at 100 m (raw 3,938 m/s). The factor-4 average created those three. The other 18 match raw layers that are already outside.

The sampler's violations are a different population. Of the outside layers, 129, 101, and 123 are below 100 m/s. The median outside Vs is 9.7, 2.0, and 4.4 m/s. The softest projected knot is 24 m/s. Layers that cover any of the 10 m–200 m knots are 5, 8, and 15 of the outside layers. The rest end by 10 m, above the first knot, so the projection grid does not contain them.

The regression target is x1, not the 21 knots. x1 already holds 284 outside layers on 142 profiles, and 226 of those layers end by 10 m. The flow is trained to match that catalog. The samples then put 64 to 81 of 200 profiles outside, against 142 of 1,895 training profiles. Copying the 21 knots does not produce that rate, those depths, or Vs values of a few metres per second.

The inference clip still removes 2, 1, and 5 knots before integration. That count is unchanged from Stage 6 and is too small for this layer count.

## The correlation weight does not train the velocity field

The flagship's 1 m gap is consistent with the vertical-correlation weight having no effect on `v_θ`.

The term is added in training only, on the real batch `u1`, with weight 0.025 and an extra factor 2 on lags of 1 m and 2 m (`train_with_depth_losses.py` lines 193–206). Validation calls `compute_flow_matching_loss` (`train.py` lines 209–217). The scheduler steps on that validation loss (`train_with_depth_losses.py` lines 584–598). The correlation term is not in the number that picks the learning rate.

Inside the term, the sequence is converted with `.cpu().numpy()` (`depth_dependent_stats/utils.py` line 63). A tensor that requires grad raises `Can't call numpy() on Tensor that requires grad`. The correlation is then wrapped in a new `torch.tensor` (`losses.py` lines 165–166), which is a leaf. On a 10-token training profile that actually has lag pairs, the returned loss has `requires_grad` false, `grad_fn` none, and value 4.87e-5. On a profile with no pairs the function returns `torch.tensor(0.0, requires_grad=True)` (`losses.py` line 176). That flag is a leaf too: `grad_fn` is none, and the value is 0. Neither object is a function of the network.

The stored targets, from `real_correlations_onewaytime.pkl`, are 0.0135 at 1 m, 0.0524 at 2 m, 0.0673 at 5 m, and 0.0360 at 10 m. Those come from the layer-center autocorrelation. The eval correlation on the real test profiles is 0.993 at 1 m, because the 0.5 m grid puts both points of a pair inside the same thick layer. A loss that matched 0.0135 would be matching a different statistic from the one in the decision table.

The depth-statistic term is the same kind of object: it is called on `u1` and rebuilt with `torch.from_numpy` (`losses.py` lines 91–92). It is not a second path into `v_θ`.

The prior was tested in Stage 6. This check covers the geostatistics term. The term does not move the velocity field. The conditional flow-matching loss is what trains it. Correction from Stage 8: the Vs penalty is computed on `ut`, which contains no network output, so it does not train `v_θ` either ([`stage8_generation_critique.md`](stage8_generation_critique.md)). The oracle result still stands: with a straight path from the true projection, two of three seeds keep a 1 m gap of 0.016 and 0.019. That gap is in the field the CFM loss produced.

## Open questions

- The Vs penalty on `ut`, weight 1e-2, was listed here as the remaining candidate for the bound-rate gap. Stage 8 shows its gradient with respect to the network is zero, so varying the weight cannot change the sampler.
- Making the correlation term a function of the sample, and of the eval estimator rather than the layer-center target, would be new training. It is not authorized.
- A learned G_φ is still a different model and is not authorized.

Stage 7 is done. GPU time for the resample was about 16 s. Budget used remains 6.35 h of 7.00 h.
