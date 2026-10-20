# Transformer Conditional Flow Matching for Vs Borehole Profiles

A comprehensive summary of the flagship model in [`experiments/flow_matching_simplified`](.).
Weights, parquet, and plots are **gitignored**; the durable local bundle lives next to the clone at `~/fm-vspdb-5k-export/` (see [Artifacts](#artifacts)).

---

## Idea

The goal is to **sample synthetic shear-wave velocity profiles** \(V_s(z)\) that look like real VSPDB boreholes, including variable layering and vertical spatial correlation — not to encode an arbitrary field borehole.

Fixed-length VAE/GAN baselines in this repo resample \(V_s(z)\) onto a grid and cannot change the number of layers. Here a profile is a **variable-length sequence of layer-boundary tokens** \([\mathrm{TTS},\, z]\). Velocity of layer \(i\) is reconstructed as

\[
V_{s,i} = \frac{\Delta z_i}{\Delta t_i}
\]

(one-way travel time in code; the README sometimes says “two-way,” but training and eval use \(V_s = \Delta z / \Delta t\)).

**Conditional flow matching (CFM)** learns a neural vector field \(v_\theta(u, t \mid c)\) that transports Gaussian noise (\(t=0\)) to real token sequences (\(t=1\)) along a straight path. Sampling is ODE integration. The condition \(c\) is **not** NEHRP class or \(H_{\max}\). It is:

1. per-profile scale \([\mu_t, \sigma_t, \mu_z, \sigma_z]\)
2. sequence length via the attention mask (number of breakpoints, hence number of layers once the origin \((0,0)\) is prepended)

A practitioner can therefore **pin layer count** and **borrow scale from real profiles in a depth window**, then filter on physics. That is a sampler of the VSPDB distribution, not a CAD tool for any soil.

---

## Model

### Representation

- Source: VSPDB profiles → breakpoint parquet (`profile_id`, `depth`, `tts`), 2,369 sequences after dropping the deterministic origin.
- Each sequence is z-scored **per profile**; the four moments are fed back so absolute scale can be recovered at denormalization.
- Pad / truncate to `max_length = 20`. Real layer counts: mean 5.63, median 5, range 1–21. Thickness median 5 m (floor 0.5 m). Real TTS is **100% strictly increasing** along depth.

### Architecture (`model.py`)

| Piece | Setting |
|---|---|
| Backbone | Transformer encoder, 4 layers, 8 heads, hidden 256 |
| Time | Sinusoidal embedding (128) → linear, added to every token |
| Scale condition | Linear projection of the 4-vector, broadcast to the sequence |
| Depth conv | 1D conv, kernel 3, residual (short-lag vertical coherence) |
| Output | Vector field in \(\mathbb{R}^2\) (TTS, depth) per token |
| Mask | Padding tokens zeroed at every ODE step |

### Losses

Training objective = CFM regression + physics + geostatistics:

- **Flow matching** MSE on the target velocity \(x_1 - x_0\) along the OT path.
- **Vs penalty** (\(w=0.01\)): implied \(V_s\) outside \([100, 5000]\) m/s.
- **Depth-bin \(\ln V_s\)** (\(w=0.01\)): match real \(\mu,\sigma\) in 10 m bins (mean weighted 2× std; bins with \(<50\) samples ignored).
- **Vertical correlation** (\(w=0.025\)): match empirical autocorrelation at lags 1, 2, 5, 10 m (short lags 2×).

### Sampling

Default `sample_sequences` draws **length and stats independently** from train. For controlled catalogs, pin the mask length and draw stats from the joint \((n_{\mathrm{layers}}, H)\) cell of real data (`bias_analysis.sample_conditional`, `stratified_hmax_test.py`). Inference uses **Euler**, 50 steps (same as the training eval path). Post-filters in `generation_profiles.py`: non-negative values, monotonic TTS, min thickness, Vs bounds, min \(\Delta t\).

---

## Training

| | |
|---|---|
| Hardware | Lambda A10, CUDA |
| Data split | 1,895 / 236 / 238 (train / val / test), seed 42 |
| Optim | Adam \(\beta=(0.6, 0.999)\), lr \(10^{-4}\), wd \(10^{-4}\), clip 1.0 |
| Schedule | ReduceLROnPlateau, patience 10, factor 0.5, min lr \(10^{-6}\) |
| Batch | 32 |
| Epochs | 1,000 then resume to **5,000** |

Checkpoints (not in git):

| File | Epoch | Val / test |
|---|---|---|
| `latest_ep1000.pt` | 1,000 | end of first run |
| `best.pt` | **3,150** | best val **0.130** |
| `latest.pt` | **5,000** | val 0.181, test **0.149** |

Test breakdown at epoch 5,000 (normalized token space): TTS MSE 0.021, depth MSE 0.019. Mean sequence length on the test loader: 5.68.

Use **`best.pt` for lowest val CFM loss**; use **`latest.pt` for depth-stat RMSE and the stratified / bias studies below**.

---

## Results

### Population (unconditional stats prior, 1,000 samples vs held-out test)

Feature space: z-scored \([\log V_{s30},\, \log H,\, n_{\mathrm{layers}},\, \log \bar V_s,\, \log\) contrast\(]\). Protocol: Naeem et al. 2020 PRDC; TreeFlow-style 1-NNA / COV.

| Checkpoint | Precision | Recall | Density | Coverage | 1-NNA (0.5 = match) | \(V_{s30}\) KS |
|---|---|---|---|---|---|---|
| `best.pt` | 0.88 | 1.00 | 0.70 | 0.88 | 0.38 | 0.046 |
| `latest.pt` | 0.89 | 0.96 | 0.77 | 0.85 | 0.43 | 0.066 |

NEHRP mix is copied (real E 59% vs generated 59%). Mean borehole depth is **not**: test \(\bar H = 92\) m vs generated \(\approx 70\) m (median both ~30 m — a missing deep tail).

Depth-binned \(\ln V_s\) (1,000 samples, CUDA eval):

| Checkpoint | Mean \(\ln V_s\) RMSE | Std RMSE | Corr. error (1 / 2 / 5 / 10 m) |
|---|---|---|---|
| `best.pt` | 1.061 | 0.974 | 0.015 / 0.001 / 0.016 / 0.027 |
| `latest.pt` | **0.772** | **0.796** | 0.004 / 0.015 / 0.024 / 0.040 |

Generated \(\ln V_s\) tracks real data through shallow/mid depths and **spikes near ~300 m and ~450–480 m**, where training mass is tiny.

Raw ODE physics (`latest.pt`, unfiltered): TTS non-decreasing **93.7%**, strictly increasing 91.6%, depth non-decreasing 96.5%. Real data: **100%** on all three. Failures are local \(\Delta t < 0\) (median \(|\min\Delta t| \approx 0.008\) s).

### Conditional fidelity (pin real test stats + length, `latest.pt`, \(n=238\))

| | |
|---|---|
| \(V_{s30}\) tracking | slope 0.97, \(r=0.97\), median AE 2.4 m/s, bias \(\approx 0\) |
| \(H\) tracking | slope 1.05, \(r=0.996\) |
| NEHRP self-rate | **96%** (E 98.5%, D 93%, C 93%; B 75% on \(n=4\)) |

This is partly **by construction**: \(V_{s30}\) and \(H\) are almost functions of the 4-vector. Freeze those moments and resample noise → \(V_{s30}\) CV drops to **2–4%** vs **18–25%** inside a real class. Layering still moves (relative std of \(V_s(z)\) ~23–35%). Independent medians of class-B stats **leave the class** (prototype stay-rate 0%, \(V_{s30}\approx 473\) m/s). Soft p10 tracks at \(r=0.998\); stiff p90 at \(r=0.59\).

### Stratified catalog: 3- and 4-layer, \(H\in[15,100]\) m

Mask length pinned; stats from real donors already in the cell (400 three-layer, 261 four-layer). Keep monotonic TTS, \(\Delta z\ge 0.5\) m, \(V_s\in[100,5000]\), \(H\) in window. 1,000 kept each.

| Check (raw ODE) | 3-layer | 4-layer |
|---|---|---|
| TTS strictly increasing | **97.4%** | **98.9%** |
| Depth increasing | 99.5% | 99.1% |
| \(H\) in 15–100 m | 99.3% | 99.0% |
| Thickness \(\ge 0.5\) m | 95.9% | 98.1% |
| All layer \(V_s\) in bounds | **60.6%** | **43.0%** |
| Kept | 60% | 43% |

Kept vs real donors (quarter-wavelength \(f_0 = 1/(4\,\mathrm{TTS}_H)\)):

| Metric | 3-layer real → gen | 4-layer real → gen |
|---|---|---|
| Mean \(H\) (m) | 28.2 → 27.1 (KS 0.15) | 32.9 → 32.3 (KS 0.14) |
| Median thickness (m) | 7.5 → 7.4 (KS 0.11) | 6.5 → 6.8 (KS 0.09) |
| Mean TTS (s) | 0.149 → **0.108** | 0.192 → **0.148** |
| Mean \(f_0\) (Hz) | 2.73 → **3.53** | 1.71 → **2.09** |
| Mean \(V_{s,\mathrm{eq}}\) (m/s) | 262 → **329** | 202 → **250** |
| Top-layer mean \(V_s\) (m/s) | 169 → **235** | 122 → **177** |

**Geometry (layer count, \(H\), thickness) is on target. Site period is biased short** (~20–40% high \(f_0\)) because every layer, especially the surface, is too fast, and impedance contrast is lower. Histograms: `outputs/figures/stratified_hmax_histograms.png` (gitignored) or `~/fm-vspdb-5k-export/stratified_hmax_test/histograms_H_f0_tts_Vs.png`.

---

## Limitations

1. **Not a representation of any borehole.** Piecewise-linear \(\le 20\) tokens; CPT and continuously varying \(V_s(z)\) lose fidelity. Real thickness floor is 0.5 m.
2. **Conditioning is scale + length, not site class or \(H_{\max}\).** Those engineering knobs work only by borrowing real joints or by rejection. Do not independently jitter the four moments.
3. **Tails are weak.** Deep \(H\), NEHRP B/rock (~1% of data), and stiff p90 are under-covered. Depth-stat RMSE is driven by empty deep bins.
4. **Physics is not guaranteed.** ~6–7% of unconstrained ODE samples violate monotonic TTS; stratified 3/4-layer draws do much better (~97–99%) but Vs-bound rejection is 39–57%.
5. **FM mean-seeking.** The CFM target is a density-weighted conditional mean (SubFlow-type bias). Majority class E is easy; rare rock lives on a thin stats manifold.
6. **\(f_0\) / \(V_s\) bias under a depth cap.** After a legal-Vs filter, kept 3/4-layer columns are stiffer than the donors. Do not use them as an unbiased site-period prior without a TTS or per-layer \(V_s\) correction.
7. **No spatial (x, y) correlation.** Vertical structure only. The model does not place profiles on a map.
8. **No apples-to-apples downstream bake-off** vs the published Conv1D VAE \(V_{s30}\) table yet (site response, hazard, etc.).
9. **Data gaps.** Japan borehole source and VSPDB API credentials were never on the training box. Flagship training used VSPDB breakpoints only.

---

## Artifacts

| Path | Contents |
|---|---|
| `~/fm-vspdb-5k-export/` | Checkpoints, eval plots, logs, samples, `breakpoints.parquet`, pickles |
| `data/vspdb_*.parquet` | Gitignored working copy of Vs / TTS profiles |
| `outputs/flow_matching_simplified/checkpoints/` | `best.pt`, `latest.pt`, `latest_ep1000.pt` |
| `python -m experiments.flow_matching_simplified.stratified_hmax_test --plot-only` | Rebuild H / \(f_0\) / TTS / \(V_{s,i}\) histograms |
| `python -m experiments.flow_matching_simplified.bias_analysis` | PRDC, 1-NNA, NEHRP tracking |

Reproduce training (CUDA):

```bash
python experiments/flow_matching_simplified/depth_dependent_stats/precompute_statistics.py
./experiments/flow_matching_simplified/depth_dependent_stats/train_model.sh
```

---

## Practical takeaway

The model is a **conditional sampler of VSPDB-like layered \(V_s(z)\)**. Pin \(n_{\mathrm{layers}}\) and draw stats from real profiles in the desired \(H\) window, then filter. Expect high TTS monotonicity and matched geometry; expect **too-high \(f_0\)** unless you correct stiffness. It is the right tool for ensembles that should look like this database. It is the wrong tool for “any soil borehole” or for an unbiased fundamental-frequency prior.
