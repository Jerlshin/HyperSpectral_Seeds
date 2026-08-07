# `SpectralQuadNet` — Architectural Audit and Refactoring Blueprint

> **Audit basis.** Documents `01`–`05` of the system suite, as committed. Every claim below is
> derived from a quantity stated in those documents; where a conclusion requires an assumption
> the documents do not pin (e.g. the empirical spacing of the 40 selected wavelengths, the
> number of source cubes), the assumption is stated explicitly and the finding is marked
> **[VERIFY]** with the one-line command or check that settles it.
>
> **Path convention.** The documents name paths under `data/`, `engine/`, `losses/`, `optim/`,
> `tracking/`, `utils/` and `configs/` but never name the module files that hold the model
> classes. Model paths below are written as `src/spectralquadnet/models/…` and marked
> *(inferred)* on first use; substitute the real path at implementation time.
>
> **Scoring convention.** Δ estimates are engineering priors with stated reasoning, not
> measurements. Two protocols are distinguished throughout:
> **P-cur** = the current patch-level split (the protocol that produced 0.8933 TTA macro-F1);
> **P-fix** = the grouped, session-disjoint split proposed in §3.1. Numbers are not comparable
> across the two, and §1.1 explains why that matters more than any other finding here.

**Notation.** $B$ batch, $C_\lambda = 40$ selected bands, $H = W = 64$, $N_g = 16$ grid cells,
$D = 256$ branch width, $K_{\text{cls}} = 90$ classes, $K = 3$ ArcFace sub-centres,
$f$ = foreground pixel fraction of a patch, $\lambda_i$ the $i$-th selected wavelength in nm,
$\tilde\lambda_i \in [0,1]$ its min–max normalisation, $m$ the ArcFace angular margin,
$\rho$ the SAM radius, $s = 48$ the ArcFace scale.

---

## 0 · Reading guide

| Tag | Meaning |
|---|---|
| **C-n** | Critical — invalidates a reported number, or costs ≳ 0.5 macro-F1 points, or makes a whole subsystem inert |
| **M-n** | Major — costs ≈ 0.1–0.5 points, or wastes ≳ 5 % of the parameter budget, or blocks a critical fix |
| **N-n** | Minor — correctness, reproducibility or hygiene; little direct metric impact |
| **[VERIFY]** | The finding depends on a repository fact the documents do not record; the check is given |
| **[FALSIFIABLE]** | A concrete, cheap experiment that will confirm or refute the diagnosis |

Sections map onto the requested output as: §1 taxonomy, §2 diagnostics, §3 redesign, §4 matrix.
Appendix A holds the derivations too long for §2; Appendix B is a consolidated register of the
falsifiable predictions, which is the part to run **first**.

---

# 1 · EXECUTIVE SUMMARY OF SYSTEM FLAWS

## 1.1 The three headline findings

**(i) The shipped model is the one that never used the metric-learning half of the system.**
`_pick_best_checkpoint` selects `best_stage1.pth` — a **linear-head, pre-ArcFace** model. Stages 2
and 3 together consume 270 epochs and contribute nothing to the reported result. The system's
three named contributions (C1 four-branch topology, C2 physical wavelength PE, C3 adaptive
sub-centre ArcFace) are reduced to two, because C3 is not in the shipped artifact. Any paper
built on `outputs/output_v12_spa40/` currently reports a number that its own headline method did
not produce. §2.4.6 and §2.5.9 explain *why* Stage 2 cannot recover Stage 1's level: the
Stage-1 → Stage-2 transition changes six things simultaneously, one of which (linear-with-bias on
unnormalised $\mathbf{e}$ → cosine on $\hat{\mathbf{e}}$ with $s = 48$) is a **representational
discontinuity that `init_from_linear` does not actually bridge**.

**(ii) "Stage 3 degrades" is, in part, a measurement artifact, and has never been tested.**
The comparison driving that conclusion is $0.8877$ (Stage 1) vs $0.8867$ (Stage 2) vs $0.8745$
(Stage 3) — all **validation** macro-F1, all read off a 1,294-patch split, and all obtained under
radically different numbers of selection opportunities: Stage 1 maximises over $600 \times 2 =
1{,}200$ live/EMA evaluations, Stage 2 over $\le 300$, Stage 3 over $15$ SWA evaluations. §2.1.3
shows the sampling standard error of macro-F1 on this split is $\approx 0.007$–$0.009$, so the
differential winner's-curse between "max of $\sim$1,200 correlated draws" and "max of 15" is
$+0.002$ to $+0.006$ — i.e. **roughly 20–45 % of the 0.0132 gap is selection bias, not
optimisation failure.** Worse: `final_eval` runs only the *selected* checkpoint, so
**Stage 3 has never been evaluated on the test split at all**. The single highest-value action in
this document is §4.1-A: run `final_eval` on all three checkpoints and publish the 3 × 2 table.

**(iii) The reported accuracy is measured under a split that almost certainly leaks acquisition
session.** `build_splits` performs a **patch-level** stratified partition. Patches are seeds
segmented out of shared cubes; a cube contains many seeds of one variety from one
`Data-VIS` session under one illumination, and the pipeline applies **dark-current subtraction
only — no white-reference division** (§2.2 of doc 02: $X = \max(R - \bar D, 0)$, and nothing
else). The stored values are therefore dark-corrected **radiance**, not reflectance, and carry
the illumination spectrum and sensor response of their session. Because session and variety are
confounded by construction, a patch-level split puts sibling seeds from the *same cube* into both
train and test, and the network can reach the label through a session-radiometric shortcut. This
does not make the architecture wrong; it makes **every number in doc 01 an upper bound of unknown
tightness**. §3.1 gives the fix (grouped split + SNV/continuum normalisation) and warns that it
will *lower* the reported figure — which is the point.

## 1.2 Taxonomy — ranked by impact on macro-F1

### Critical

| ID | Flaw | Class | Est. impact | § |
|---|---|---|---|---|
| **C-1** | Patch-level split leaks acquisition session; no white-reference normalisation, so values are radiance not reflectance | Protocol / physics | Invalidates the magnitude of every reported number | 2.1.1, 3.1 |
| **C-2** | Branches **A and D consume byte-identical inputs** ($16\times40$ grid spectra); together with B, **43.9 % of parameters see 640 numbers** while the only branch with access to the $163{,}840$-number cube gets 21.5 % and is dropped most often | Capacity allocation | −1.0 to −2.5 pts vs. a balanced design | 2.2.1–2.2.3, 3.3 |
| **C-3** | **No operator anywhere in the network models joint spectral–spatial structure at spatial resolution.** Branch C annihilates the spectral axis in its first $1\times1$; A/B/D see a $256{:}1$ spatially-averaged grid | Representation | −0.8 to −2.0 pts | 2.2.4, 3.3.3 |
| **C-4** | `SpectralStatsBranch` input is **rank $\le 2$** under the physically correct pixel model; 686 k parameters (8.7 %) act on $\approx 2$ effective dimensions, one of which duplicates Branch A | Information bottleneck | −0.3 to −0.8 pts; is the true cause of the $2\times$ aux hack | 2.2.5, 3.3.2 |
| **C-5** | Every `Conv1d` on the band axis is a **finite-difference stencil over a non-uniform grid**: mean selected spacing 15.98 nm, native floor 2.444 nm, so one shared kernel must serve stencils differing by $\ge 6.5\times$ in $\Delta\lambda$ | Mathematical | −0.4 to −1.2 pts; blocks all derivative features | 2.2.6, 3.3.1 |
| **C-6** | **Stage 3 is not SAM.** The ascent gradient is taken on $\mathcal{L}_{\text{focal}} + 0.02\mathcal{L}_{\text{sc}} + 0.10\mathcal{L}_{\text{aux}}$ and the descent gradient on $\mathcal{L}_{\text{focal}}$ alone, so the update penalises curvature **along the wrong direction** | Optimisation | Explains a large share of the Stage-3 gap | 2.5.1, 3.6.3 |
| **C-7** | Stage 3 compound defect: margin **discontinuity and inversion** at entry ($M(y)\!\in\![0.35,0.45]$ per class → uniform $0.30$, then annealed *down*), SWA averaging across **15 different objectives**, greedy filter that is a **provable no-op**, BN re-estimation under **active dropout and a re-weighted class prior**, and the EMA shadow **overwritten** by the SWA weights | Optimisation | −0.5 to −1.2 pts; recoverable in full | 2.5.2–2.5.7, 3.6 |
| **C-8** | Spectral TTA **destroys the background-zero invariant that four downstream modules depend on**; 4 of 12 views are off-manifold, and $p_{10}, p_{25}$ collapse to a constant while skew/kurt collapse to band-independent constants | Inference / domain shift | +0.2 to +0.6 pts recoverable | 2.6, 3.7 |
| **C-9** | The 1,294-patch validation split is used for **six** distinct purposes (margins, CDWS, P3 oversampling, early stop, checkpoint selection, SWA acceptance) — 270 real-valued parameters fitted on 14.4 samples/class, then re-used for model selection | Protocol | val→test gap of 1.07 pts is the visible symptom | 2.1.2–2.1.3, 3.1.4 |

### Major

| ID | Flaw | Class | Est. impact | § |
|---|---|---|---|---|
| **M-1** | **Fusion latent collapse.** Latents init at $\sigma = 0.02$ while the keys they query are LayerNormed to unit scale — a $50\times$ mismatch, so at init all four latents emit the same attention distribution and receive the same update | Fusion | Reduces fusion to a 1-latent bottleneck | 2.3.1 |
| **M-2** | Per-branch **LayerNorm destroys the cross-branch confidence signal** and amplifies a low-SNR branch to unit scale; the softmax modality gate is then **exclusive** where the task needs **conjunction** | Fusion | −0.2 to −0.5 pts | 2.3.2–2.3.3 |
| **M-3** | Fusion is **purely additive** — no multiplicative/bilinear term — in a problem whose own abstract calls it "fine-grained and near-degenerate" | Fusion | −0.2 to −0.5 pts | 2.3.4 |
| **M-4** | **2,190,916 parameters (27.8 %, the largest single component) attend over 4 tokens** — 547 k parameters per token, with $N_{\text{latent}} = N_{\text{modality}} = 4$ so the "Perceiver" compresses nothing | Capacity | 1.6–2.0 M reallocatable | 2.3.5, 3.4 |
| **M-5** | **Branch-drop probabilities are inverted**: effective rates are $P(\text{drop }C) = 0.225$, $P(\text{drop }D)=0.15$, $P(\text{drop }A)=P(\text{drop }B)=0$ — the redundant branches are protected, the unique one is suppressed | Regularisation | −0.2 to −0.4 pts | 2.2.7 |
| **M-6** | The adaptive margin $M(c) = 0.35 + 0.10(1 - F_1^{(c)})$ is a **positive-feedback loop with the wrong sign for low-recall classes**: a larger margin shrinks the class's decision region, lowering recall, raising the margin further | Head / metric | −0.1 to −0.4 pts on macro-F1 specifically | 2.4.1 |
| **M-7** | For the hardest classes the angular gradient **saturates and then decays**: $\partial\mathcal{L}/\partial\theta_y \propto \sin(\theta_y + m)$ peaks at $\pi/2$, and focal's $(1-p_y)^\gamma$ is already pinned at 1, so the margin increase buys no extra focus and costs gradient magnitude | Head / math | Compounds M-6 | 2.4.2 |
| **M-8** | **Sub-centre death.** $\max_k$ gives hard assignment; with $\sim67$ training samples per class, $K=3$ sub-centres and no load-balancing term, a sub-centre that stops winning can never recover | Head | Makes C3 partly inert | 2.4.3 |
| **M-9** | **Focal × label smoothing is mathematically inconsistent.** $p_t = e^{-\ell}$ with a smoothed $\ell$ has an entropy floor $H(q)$, so the focal modulator is bounded below by $0.396$ at $\varepsilon=0.10$ and $0.159$ at $\varepsilon=0.04$ — it cannot down-weight easy examples, which is its only purpose. This is active in the **winning** configuration (Stage 1 Phase 3) | Loss / math | −0.1 to −0.3 pts | 2.4.5 |
| **M-10** | **A single global `clip_grad_norm_(model.parameters(), 1.0)`** couples an $s{=}48$-amplified head to a 6-parameter front-end: on any step where the head saturates, the entire backbone's effective LR is scaled by the same factor | Optimisation | −0.1 to −0.4 pts | 2.5.8 |
| **M-11** | **`cv2.INTER_AREA` resize after masking** blends seed values with zeros at the boundary, so $\sim15\,\%$ of "foreground" pixels carry a partial fill factor $\alpha_p < 1$ at full mask weight — a shape- and size-dependent multiplicative bias on **every masked statistic** | Preprocessing | −0.1 to −0.3 pts; also drives C-4 | 2.2.8 |
| **M-12** | **Corner-cell dilution.** Branch A takes an *unweighted* mean over 16 grid cells, of which $\sim4$ are majority background and clamp toward $0$; the resulting gain depends on the seed's fill fraction — a nuisance variable | Branch A | −0.05 to −0.2 pts | 2.2.9 |
| **M-13** | **Morphometrics are computed and thrown away.** `regionprops` already yields area, eccentricity, solidity and axis lengths in segmentation; the resize to $64\times64$ then destroys absolute scale, which is a primary taxonomic feature of rice grain | Preprocessing | +0.2 to +0.6 pts recoverable, near-free | 2.2.10, 3.2.4 |
| **M-14** | Band selection was validated with **LDA / LinearSVC on spatially-averaged mean spectra** — a hypothesis class that discards everything the deployed model uses; and the recorded curve **terminates exactly at the chosen $k=40$**, so the "elbow" is unverifiable and may be vacuous | Feature selection | Unknown, possibly large | 2.2.11 |

### Minor

| ID | Flaw | § |
|---|---|---|
| **N-1** | Five dead control paths: `model.fusion_heads` (8 runs, not 4), `model.specf_drop` (0.10 runs, not 0.15), `model.wl_embed_dim` (unused), `model.branch_drop_prob` (hardcoded vector runs), Stage-3 `proto_weight` (module passed, term never applied) | 2.7 |
| **N-2** | `set_dropout` cannot reach `nn.MultiheadAttention`'s dropout, so fusion and SpecFormer attention stay at 0.10 for the entire run including Stage 3 | 2.7 |
| **N-3** | `spec_pos_embed` has capacity 12, uses 11 — one permanently dead row | 2.7 |
| **N-4** | ArcFace clamp at $1-10^{-6}$ admits $\left|\partial\sqrt{1-c^2}/\partial c\right| \le 707$; combined with $s=48$ a single aligned sample can dominate the clipped gradient direction | 2.4.4 |
| **N-5** | `nan_to_num` on eval logits converts numerical failure into a silently degraded metric with no counter | 2.7 |
| **N-6** | `use_amp = (supcon is None) ∧ (scaler is not None)` ties precision policy to loss composition, so the "contrastive off" ablation is not compute-matched | 2.7 |
| **N-7** | Both samplers draw from an unseeded `np.random.default_rng()`, so no single-seed ablation delta is interpretable | 2.7 |
| **N-8** | `stage2_arcface.py` reads `param_groups[0]`/`[2]` positionally | 2.7 |
| **N-9** | The fusion block applies pre-LN **only to the feed-forward sublayer**; both attention sublayers read an un-normalised, growing residual stream | 2.3.6 |
| **N-10** | `output_proj` and `EmbedNet` are two sequential $256\to256$ residual refinements doing the same job | 2.3.7 |
| **N-11** | Stage 3 logs no `val/f1_ema` channel, unlike Stages 1–2, so its EMA shadow is never scored before being overwritten | 2.5.7 |
| **N-12** | The BN re-estimation note is self-contradictory: grad is enabled *specifically so dropout applies* on MPS, yet the doc claims "forward values … identical either way". Both cannot hold — the pass is device-dependent | 2.5.6 |

## 1.3 Verdict on Stage 3

Stage 3 is not failing because SAM and SWA are the wrong tools. It is failing because **six
independent mechanisms all push the same way at once**, and because the yardstick used to judge
it is biased against it:

| # | Mechanism | Direction | § |
|---|---|---|---|
| 1 | Ascent and descent objectives differ → the update penalises curvature along $\hat{g}_A$, not $\hat{g}_D$ | Degrades | 2.5.1 |
| 2 | Margin jumps from per-class $[0.35, 0.45]$ to uniform $0.30$ and anneals *down* → Stage 2's per-class calibration is actively unlearned | Degrades | 2.5.2 |
| 3 | SWA averages 15 snapshots taken under 15 different margins → averaging a **translating** trajectory, not an oscillating one | Degrades | 2.5.3 |
| 4 | Greedy acceptance ($F_1 \ge 0.98 \max$, running max updated *before* the test) is a no-op — 15/15 accepted, as recorded | Removes the safeguard | 2.5.4 |
| 5 | Adam's second moment is reset at stage entry; its horizon is $1/(1-\beta_2) = 1000$ steps $= 21.3$ epochs, but the **first snapshot is taken at epoch 8**, deep in the transient | Degrades | 2.5.5 |
| 6 | BN re-estimation runs in `train()` with **dropout active** (variance-inflated $\sigma$) over a **CDWS-reweighted class prior**, and only branches C and D carry BatchNorm — i.e. exactly the two highest-capacity branches | Degrades | 2.5.6 |
| 7 | The SWA weights **overwrite the EMA shadow**, discarding the one averaging scheme that demonstrably worked in Stages 1–2 | Removes a gain | 2.5.7 |
| 8 | Stage 3 is judged by the max of 15 validation draws against Stage 1's max of $\sim$1,200 | Biases the comparison | 2.1.3 |

The diagnostic that clinches it: the Stage-2 → Stage-3 drop is **−0.0122 macro-F1 and −0.0116
accuracy** — almost exactly equal. A loss-reweighting or class-prior effect would show up as a
macro-F1 drop *larger* than the accuracy drop, because macro-F1 weights the rare/hard classes
equally. A uniform drop across both metrics is the signature of a **globally mis-calibrated set
of weights** — which is precisely what mechanisms 1, 3, 5 and 6 produce. Fix those four and
Stage 3 should at minimum match Stage 2; §3.6 argues it should exceed it.

---

# 2 · MATHEMATICAL & STRUCTURAL DIAGNOSTICS

## 2.1 Protocol-level defects

### 2.1.1 · C-1 — the split is not grouped, and the data are not reflectance

`data/loaders.py::build_splits` performs a two-step **stratified** partition on the 8,624 patch
indices with `random_state=42`. Stratification is on the **label only**. The generative structure
of the data is:

```
Zenodo 3241923
 └─ session  (path component "Data-VIS…")
      └─ scan  (file stem "<variety>-<n>")      ← one variety, one illumination, one dark ref
           └─ ~N seeds segmented from one cube  ← ALL share every acquisition nuisance factor
```

Each cube yields many patches that share: illumination spectrum, integration time, sensor
temperature, optical path, and the *same* dark reference $\bar D$. A patch-level random split
therefore places siblings from one cube on both sides of the train/test boundary.

That would be tolerable if the per-cube nuisance were removed radiometrically. It is not. The
correction implemented is dark subtraction only:

$$
X_{h,w,c} = \max\big(R_{h,w,c} - \bar D_{w,c},\, 0\big)
$$

There is **no white-reference division**. Writing $S_c$ for the illumination × sensor response of
the session and $\varrho_c$ for the true reflectance,

$$
X_{h,w,c} \;\approx\; a_{h,w}\, S_c\, \varrho_c \quad\Longrightarrow\quad
\text{the stored cube is radiance, and } S_c \text{ is a per-session multiplicative field.}
$$

Doc 01 calls the input a "VIS–NIR **reflectance** patch"; the pipeline produces dark-corrected
radiance. Since $S_c$ is constant within a cube and a cube contains exactly one variety, $S_c$ is
a **perfectly predictive shortcut for the label within any patch-level split**, and it is
available to the network in every branch.

Estimated leak channel capacity: with $C_\lambda = 40$ bands and even $1\,\%$ session-to-session
variation in $S_c$, the shortcut is trivially learnable by the depthwise $1\times1$ in Branch C's
`band_reduce` (one scalar per band — literally a learned $\hat S^{-1}$) or by the 6-parameter
`MaskedSpectralECA` gate.

**[VERIFY]** Count distinct `(session, scan)` groups and their variety multiplicity:
`python -c "from data.prep... ; print(df.groupby('variety').scan.nunique().describe())"`.
If any variety has only one scan, a session-disjoint split is *impossible* for that variety and
the dataset cannot support the claim being made — this must be reported.

**[FALSIFIABLE] F-1.** Train the identical pipeline on `StratifiedGroupKFold(groups=scan_id)`.
Prediction: macro-F1 falls by **5–20 points**. If it falls by $<2$ points, the leak is
immaterial and C-1 is downgraded to N-class. Either outcome is publishable; the current
ungrouped number is not.

### 2.1.2 · C-9 — one 1,294-patch split, six jobs

| # | Consumer of the validation split | What it fits |
|---|---|---|
| 1 | `update_margins_from_f1` | 90 per-class ArcFace margins |
| 2 | `build_cdws_weights` | 90 CDWS sampling weights |
| 3 | `HardClassOversampledSampler` | 90 Phase-3 oversampling weights |
| 4 | Early stopping | Stage-1 patience 160, Stage-2 patience 80 |
| 5 | `_pick_best_checkpoint` | which of 3 checkpoints ships |
| 6 | Greedy SWA acceptance | which of 15 snapshots enter the average |

Consumers 1–3 fit **270 real-valued parameters on 1,294 samples** = 14.4 samples per class, and
consumers 4–6 then use the *same* samples for model selection. The validation split is being
trained on, not held out. The observable symptom is already in the record:

$$
\underbrace{0.8877}_{\text{val, Stage 1}} \;-\; \underbrace{0.8770}_{\text{test, single view}}
\;=\; 1.07 \text{ points}
$$

on a split of identical size (1,294) drawn from the same pool by the same stratifier. Under an
honest protocol the two should agree to within sampling noise ($\pm 0.9$ pts, §2.1.3). The gap is
exactly one noise unit — consistent with, and fully explained by, selection on the validation set.

### 2.1.3 · The winner's curse, quantified — and why it dominates the Stage-3 comparison

Per-class $F_1$ on the validation split is estimated from $n_c = 1294/90 = 14.4$ samples. For a
class at $F_1 \approx 0.89$, the binomial standard error of recall is

$$
\mathrm{SE}(\hat R_c) = \sqrt{\frac{0.89 \times 0.11}{14.4}} = 0.0836 ,
$$

and precision is comparable. Macro-F1 averages 90 such quantities; treating the *idiosyncratic*
component (the part specific to these particular 1,294 patches) as weakly correlated across
classes gives

$$
\mathrm{SE}\big(\widehat{F_1^{\text{macro}}}\big) \;\approx\; \frac{0.07}{\sqrt{90}} \;\approx\; 0.0074
\quad\text{(range } 0.007\text{–}0.009\text{)} .
$$

Now compare the selection procedures. Let $n_{\text{eff}}$ be the number of *effectively
independent* validation readings a stage maximises over. For a maximum of $n$ draws from
$\mathcal{N}(\mu, \sigma^2)$, $\mathbb{E}[\max] \approx \mu + \sigma\,\Phi^{-1}\!\big(1 -
\tfrac{1}{n+1}\big)$:

| Stage | Raw evaluations | Correlation-adjusted $n_{\text{eff}}$ | $\mathbb{E}[\max] - \mu$ |
|---|---:|---:|---:|
| 1 | $600 \times 2 = 1200$ (live + EMA) | $\approx 30$–$100$ | $+0.0137$ to $+0.0172$ |
| 2 | $\le 300$ | $\approx 15$–$40$ | $+0.0114$ to $+0.0146$ |
| 3 | $15$ (SWA cycle ends only) | $\approx 15$ | $+0.0114$ |

Differential bias, Stage 1 over Stage 3: **$+0.0023$ to $+0.0058$**, against an observed gap of
$0.0132$. So roughly **20–45 % of the "Stage 3 is worse" signal is an artifact of unequal
selection opportunity**, before a single optimisation mechanism is invoked. (At the upper bound of
the noise estimate, $\sigma = 0.009$, the differential reaches $0.0071$ — over half the gap.)

And the comparison that would settle it does not exist: `engine/stages/final_eval.py` runs
**only** the checkpoint `_pick_best_checkpoint` returns. There is no test-split number for
Stage 2 or Stage 3 anywhere in `outputs/output_v12_spa40/`.

### 2.1.4 · Selection–evaluation mismatch (part of C-9)

Every stage saves on $\max\big(F_1^{\text{live}}, F_1^{\text{ema}}\big)$ and writes that maximum
to the sidecar as `val_f1`. `_pick_best_checkpoint` ranks by that field. But `final_eval`
**always evaluates the EMA shadow**. Therefore:

$$
\text{selected because } F_1^{\text{live}} = 0.8877 \;\Longrightarrow\;
\text{evaluated as } f_{\theta^{\text{EMA}}} \neq f_{\theta^{\text{live}}} .
$$

If the live model won the max at epoch 488, the shipped test number comes from a model that was
never the one that scored 0.8877. The sidecar does not record *which* of the two won, so this is
currently unfalsifiable from the artifacts — which is itself the defect.

---

## 2.2 Representation-level defects

### 2.2.1 · C-2 — parameters per unique input scalar

| Branch | Input tensor | Unique input scalars | Params | Params / scalar |
|---|---|---:|---:|---:|
| A · SpectralProfile | $(B,16,40)$ grid spectra | $640$ | 592,753 | 926 |
| B · SpectralStats | $(B,9,40)$ masked moments | $360$ nominal, **$\le 2$ effective** (§2.2.5) | 686,424 | 1,907 nominal / $\ge 3.4\times10^{5}$ effective |
| C · SpatialCNN | $(B,40,64,64)$ gated cube | $163{,}840$ | 1,694,158 | **10.3** |
| D · SpecFormer | $(B,16,40)$ — **byte-identical to A** | $0$ *new* | 2,180,866 | $\infty$ |
| Fusion | $4 \times 256$ | $1{,}024$ | 2,190,916 | 2,140 |

Two readings of this table, both damning:

1. **A, B and D share one 640-number view.** $592{,}753 + 686{,}424 + 2{,}180{,}866 =
   3{,}460{,}043$ parameters — **43.9 % of the budget** — are applied to the same
   $16\times40$ tensor. B is a deterministic, rank-deficient function of the same cube, so it adds
   no new measurement. The claim in C1 that the four branches are "four **disjoint** views" is
   false for A/D and near-false for B.
2. **Branch C has $330\times$ fewer parameters per input dimension than Branch D** while being the
   only branch that can see spatial texture — the very signal that distinguishes rice varieties
   with sub-percent reflectance differences (husk striation, groove geometry, chalkiness,
   pericarp mottling).

### 2.2.2 · A ⊆ D: Branch A is a strict functional subset of Branch D

Both consume $G \in \mathbb{R}^{B\times16\times40}$. Branch A computes

$$
\mathbf{b}_A = \tfrac{1}{16}\sum_{n=1}^{16} \psi_A(G_n), \qquad \psi_A:\mathbb{R}^{40}\to\mathbb{R}^{256}
$$

— a **fixed uniform** pooling over cells. Branch D computes a per-cell summary $\psi_D(G_n)$ and
then pools with **learned attention** (the `spatial_cls` token over 17 tokens). Uniform pooling is
a special case of attention pooling (constant logits). Hence $\mathbf{b}_A$ lies in the
representable set of Branch D. The only structural difference is $\psi_A$'s multi-scale
$\{3,5,7\}$ depthwise towers versus $\psi_D$'s $\{3,5,7\}$ strided tokeniser — the *same* kernel
family, one at stride 1 and one at stride 4.

**Gradient consequence.** In the fused path, cross-attention plus the softmax modality gate route
credit to whichever branch is most linearly predictive of the residual. Given $\mathbf{b}_A \in
\mathrm{range}(\text{Branch D})$ and $|\theta_D| = 3.68 |\theta_A|$, Branch D wins that
competition, and

$$
\frac{\partial \mathcal{L}_{\text{main}}}{\partial \theta_A} \longrightarrow 0 .
$$

This is precisely the failure the documents already report:

> *"gradient collapse in the spectral branches A/B is a real failure mode of this architecture —
> the same one the $2\times$ auxiliary weighting counteracts."* (doc 05, §5.2)

**The $2\times$ auxiliary weight is not a spectral-vs-spatial balance term. It is life support for
two branches whose inputs are duplicated.** Removing the duplication removes the need for the
hack — which is why §3.3 de-duplicates the inputs rather than re-tuning $\omega_b$.

### 2.2.3 · Where the 640 numbers came from: a 256:1 spatial annihilation

`extract_grid_spectra(·, 4)` reduces $(B,40,64,64)$ to $(B,16,40)$. Each grid cell averages
$16 \times 16 = 256$ pixels. Three of four branches therefore see the seed at an effective
spatial resolution of $4\times4$, i.e. a compression ratio of

$$
\frac{40 \times 64 \times 64}{16 \times 40} = 256:1 .
$$

At a nominal $64\times64$ patch covering a seed of $\sqrt{300}$–$\sqrt{800} \approx 17$–$28$ px in
the source cube, one grid cell spans roughly a quarter of the seed's short axis. Every
intra-seed structure — embryo vs. endosperm vs. husk, the dorsal groove, surface defects — is
averaged away before A, B and D ever see it.

### 2.2.4 · C-3 — the network contains no joint spectral–spatial operator

Trace the spectral axis through Branch C, the only branch with spatial access:

| Step | Operation | Spectral axis after |
|---|---|---|
| C.1 | depthwise $1\times1$, `groups=40` — one scalar per band | 40, unmixed |
| C.2 | dense $1\times1$, $40 \to 64$ | **gone** — replaced by 64 abstract channels via a *single global linear map shared across all $64^2$ pixels* |
| C.3–C.9 | four `ResBlock2D` + three `CBAM` | purely spatial |

So the only place where "band $c$ at pixel $(h,w)$" exists jointly is a $40\to64$ linear layer
with 2,560 weights, applied identically everywhere. Any hypothesis of the form *"variety X shows
a 940 nm water-band depression **specifically in the embryo region**"* is representable only as
$\langle w_j, x_{:,h,w}\rangle$ for a fixed $w_j$, with the spatial localisation deferred to the
2-D stack operating on already-mixed channels. That is a strictly weaker function class than a
$3\times3\times k_\lambda$ 3-D convolution, and it is the standard reason HSI classifiers
(HybridSN, SSRN, A²S²K-ResNet) use 3-D or factorised spectral–spatial kernels.

$$
\text{Current: } h_{j,h,w} = \sigma\Big(\textstyle\sum_c W_{jc}\, x_{c,h,w}\Big)
\qquad\text{vs.}\qquad
\text{Needed: } h_{j,h,w} = \sigma\Big(\textstyle\sum_{c,\delta h,\delta w} W_{j,c,\delta h,\delta w}\, x_{c,h+\delta h,w+\delta w}\Big)
$$

### 2.2.5 · C-4 — `SpectralStatsBranch` receives a rank-$\le 2$ matrix

Model a single segmented seed's pixel spectra. Within one seed of one variety, pixel-to-pixel
variation is dominated by **illumination geometry and surface orientation**, plus the resize fill
factor of §2.2.8 — all of which act as a **per-pixel scalar gain** on a common spectral shape:

$$
x_{c,p} \;=\; a_p \, r_c \;+\; \eta_{c,p}, \qquad \eta \text{ small},\; a_p > 0 .
$$

Substituting into the nine statistics of `masked_spectral_stats` (all computed over foreground
pixels only, all in `float32`):

| # | Statistic | Value under the rank-1 pixel model |
|---|---|---|
| 1 | mean $\mu_c$ | $\bar a\; r_c$ |
| 2 | std $\sigma_c$ | $\mathrm{sd}(a)\; r_c$ |
| 3 | max | $a_{\max}\; r_c$ |
| 6–9 | $p_{10},p_{25},p_{75},p_{90}$ | $q_{10}(a)\,r_c,\;\dots,\;q_{90}(a)\,r_c$ |
| 4 | skewness $= \mathbb{E}[\delta^3]/\sigma^3$ | $\dfrac{r_c^3\,\mathbb{E}[(a-\bar a)^3]}{r_c^3\,\mathrm{sd}(a)^3} = \mathrm{skew}(a)$ — **independent of $c$** |
| 5 | kurtosis $= \mathbb{E}[\delta^4]/\sigma^4$ | $\mathrm{kurt}(a)$ — **independent of $c$** |

Therefore the input matrix $S \in \mathbb{R}^{9\times40}$ decomposes as

$$
S \;=\; \underbrace{\mathbf{v}\,\mathbf{r}^{\top}}_{\text{7 rows, rank 1}}
\;+\;\underbrace{\mathbf{u}\,\mathbf{1}^{\top}}_{\text{2 rows, rank 1}} ,
\qquad
\mathbf{v} = \big(\bar a,\,\mathrm{sd}(a),\,a_{\max},\,0,0,\,q_{10},q_{25},q_{75},q_{90}\big)^{\top},
\quad \mathbf{u} = (0,\dots,\mathrm{skew}(a),\mathrm{kurt}(a),\dots)^{\top}
$$

$$
\boxed{\;\operatorname{rank}(S) \;\le\; 2\;}
$$

Consequences, in order of severity:

1. **686,424 parameters (8.7 % of the model) operate on $\le 2$ effective degrees of freedom.**
2. One of those two, $\mathbf{r}$, is the seed's mean spectral **shape** — *exactly* what Branch A
   receives as its 16 cell means. So Branch B's *unique* contribution is the pair
   $\big(\mathrm{skew}(a), \mathrm{kurt}(a)\big) \in \mathbb{R}^2$: two scalars describing the
   **pixel-intensity histogram shape**, i.e. the illumination/surface geometry of the seed — a
   *nuisance* variable, not a varietal one.
3. The `stat_attn` gate $S \leftarrow S \odot \sigma(W_2\mathrm{GELU}(W_1\mathrm{GAP}_\lambda(S)))$
   can only rescale the nine rows. It cannot raise the rank. The stated design goal — "the network
   learns *which moments matter*" — is unreachable, because seven of the nine moments are the same
   moment up to a scalar.
4. The $1\times1$ projection $9 \to 96$ then lifts a rank-2 matrix into 96 channels; the three
   `ResBlock1D` towers and the fusion operate on a manifold of dimension $\le 2 \cdot 40$.

**[FALSIFIABLE] F-2.** Compute the singular values of $S$ over the training set:
`np.linalg.svd(stats_batch)`; report $\sigma_3/\sigma_1$. Prediction: $\sigma_3/\sigma_1 < 0.05$
for $>90\,\%$ of seeds. If so, C-4 is confirmed and §3.3.2 applies.

### 2.2.6 · C-5 — every 1-D convolution is a finite difference on an irregular grid

The 40 selected bands are **sorted ascending, and the selection order is discarded** — only the
*set* survives (doc 02 §2.4). They span $383.22$–$1006.47$ nm, so

$$
\overline{\Delta\lambda} \;=\; \frac{1006.47 - 383.22}{39} \;=\; 15.98\ \text{nm},
\qquad
\Delta\lambda_{\min} \;\ge\; 2.444\ \text{nm (native spacing)} .
$$

SPA maximises geometric spread but the $\tau_{\text{corr}} = 0.995$ decorrelation pre-filter is
applied *first*, and mRMR seeds SPA at the highest-MI band — both of which produce clustering in
informative regions. So the realised spacing is **non-uniform with a dynamic range of at least
$6.5\times$ and plausibly $>20\times$**.

Now consider any $\mathrm{Conv1d}(k=3)$ in Branch A's stem, Branch B's `ResBlock1D` towers, or
Branch D's tokeniser. Its kernel $(w_{-1}, w_0, w_{+1})$ is **shared across all band positions**.
Applied at position $i$ it computes

$$
(\mathrm{Conv}\,F)_i = w_{-1}F_{i-1} + w_0 F_i + w_{+1} F_{i+1} .
$$

If the kernel learns a derivative stencil ($w_{-1} = -1, w_0 = 0, w_{+1} = +1$), the quantity it
actually produces is

$$
F_{i+1} - F_{i-1} \;\approx\; \frac{\partial F}{\partial\lambda}\Big|_{\lambda_i}\cdot(\lambda_{i+1} - \lambda_{i-1})
$$

i.e. the true spectral derivative **multiplied by a position-dependent, unknown scale factor
$\Delta\lambda_i$ that varies by $\ge 6.5\times$ across the axis.** One shared kernel cannot serve
both a 5 nm stencil and a 100 nm stencil. The network must instead spend capacity learning a
position-dependent correction — for which it has only an *additive* code.

**And the additive code cannot fix a multiplicative metric distortion.** `PhysicalWavelengthPE`
adds $E_{\mathrm{wl}}^{\top}$ to the feature map:

$$
F \leftarrow F + E_{\mathrm{wl}}^{\top}, \qquad E_{\mathrm{wl}}[i,j] = \sin(\tilde\lambda_i\omega_j) \;\|\; \cos(\tilde\lambda_i\omega_j) .
$$

This tells the *channel mixing* where it is in $\lambda$, which is genuinely valuable and is a
real contribution (C2). But convolution is linear in $F$, so

$$
\mathrm{Conv}\big(F + E\big) = \mathrm{Conv}(F) + \mathrm{Conv}(E)
$$

— the PE contributes an **additive, input-independent bias field**. It cannot rescale the stencil.
The correct object is a kernel whose *weights* are a function of $\Delta\lambda$:

$$
(\mathrm{Conv}_\lambda F)_i \;=\; \sum_{j \in \mathcal{N}(i)} \kappa_\phi\!\big(\lambda_j - \lambda_i\big)\, F_j
$$

with $\kappa_\phi$ a small MLP (a continuous / implicit kernel). §3.3.1 specifies this.

**Contradiction worth naming.** Doc 01 justifies C2 by observing that *"band selection removes 216
of 256 bands non-uniformly, index adjacency no longer implies spectral adjacency"*. That
observation is correct, and it is a direct argument that **index-based convolution is invalid** —
yet the architecture then applies index-based convolution in every 1-D branch and index-based
striding in Branch D. The PE addresses the symptom; the convolution retains the disease.

### 2.2.7 · M-5 — the branch-drop vector suppresses the wrong branch

Forward pass (training mode) uses a hardcoded $p = (0,\,0,\,0.30,\,0.20)$ with a safe-index rescue
$k \leftarrow \max(k, \operatorname{onehot}(s))$, $s\sim\mathcal{U}\{0,1,2,3\}$. Since
$k_A = k_B = 1$ always, the safe index never rescues a *needed* branch — it only occasionally
un-drops C or D. Effective rates:

$$
P(\text{drop } C) = 0.30\cdot\tfrac{3}{4} = 0.225,\qquad
P(\text{drop } D) = 0.20\cdot\tfrac{3}{4} = 0.150,\qquad
P(\text{drop } A) = P(\text{drop } B) = 0 .
$$

Cross-referencing §2.2.1: the two branches with **zero unique information** (A duplicates D's
input; B is rank-2 and its informative component duplicates A) are the two that are **never
dropped**, and the single branch holding all $163{,}840$ spatial-spectral numbers is dropped
$22.5\,\%$ of the time. The fusion is thus trained to be robust to the absence of the only
irreplaceable modality, which is exactly the pressure that teaches the modality gate to
down-weight it. `model.branch_drop_prob = 0.20` is stored and ignored (N-1).

### 2.2.8 · M-11 — `INTER_AREA` resize breaks the zero-background invariant it is supposed to serve

The extraction order is: **mask → centre-pad to $(S,S,256)$ → resize band-by-band to $64\times64$
with `cv2.INTER_AREA` → transpose**. `INTER_AREA` is an area-weighted box filter. A destination
pixel straddling the seed boundary averages source pixels of which a fraction $\alpha_p \in (0,1)$
are seed and $1-\alpha_p$ are the exact zeros the mask wrote:

$$
x^{\text{resized}}_{c,p} \;=\; \alpha_p\, \bar x^{\text{src}}_{c} , \qquad 0 < \alpha_p < 1 .
$$

Because $\alpha_p$ is **band-independent** (the mask is a single 2-D indicator), the *shape* of
each edge pixel's spectrum is preserved — but its *amplitude* is attenuated, and the downstream
mask $\mathbb{1}[\sum_c |x_{c,p}| > 10^{-5}]$ counts it at **full weight**. Order of magnitude: at
$f = 0.284$ the foreground is $\approx 1{,}163$ px with a perimeter of $\approx 130$–$190$ px, so
**roughly $11$–$16\,\%$ of "foreground" pixels are partial-coverage pixels**.

Consequences:

- $\mu_c \leftarrow \bar\alpha\, \mu_c$ with $\bar\alpha \approx 0.95$ — a **shape-dependent gain**
  (perimeter/area ratio), i.e. a nuisance correlated with grain elongation.
- $\sigma_c$ is inflated by the variance of $\alpha$, again shape-dependent.
- Crucially, this is a *second* source of the per-pixel scalar gain $a_p$ that drives the rank-1
  collapse of §2.2.5 — the two defects reinforce each other.

Correct order: **resize the mask and the cube separately, then re-apply a hard threshold**, or
resize with `INTER_NEAREST` for the mask and re-mask after `INTER_AREA` on the cube, then
normalise each surviving pixel by its own $\alpha_p$ (which the resized mask gives for free).

### 2.2.9 · M-12 — corner-cell dilution in the $4\times4$ grid

`extract_grid_spectra` clamps an all-background cell to $0$ rather than dividing by zero. A seed
padded to a square and inscribed in a $64\times64$ frame leaves the four corner cells majority
background; typically **4 of 16 cells are dominated by the clamp**. Branch A then takes an
*unweighted* mean:

$$
\mathbf{b}_A = \frac{1}{16}\sum_{n=1}^{16}\psi_A(G_n)
\;=\; \frac{n_{\text{fg}}}{16}\,\overline{\psi_A(G_{\text{fg}})} \;+\; \frac{16-n_{\text{fg}}}{16}\,\psi_A(\mathbf{0}) .
$$

The mixing coefficient $n_{\text{fg}}/16$ is a function of the seed's **shape and fill**, not its
variety. The denominator $\max(\mathrm{AvgPool}(m), 10^{-5})$ is already computed — it is the
natural cell weight and is currently discarded. §3.3.4 replaces the mean with a foreground-mass
weighted pool, at a cost of zero parameters.

### 2.2.10 · M-13 — morphometrics are computed, gated on, and then destroyed

`segment()` runs `regionprops` and gates on
$300 < \text{area} < 800 \wedge \text{eccentricity} > 0.6 \wedge \text{solidity} > 0.85$.
So area, eccentricity, solidity, major/minor axis length and equivalent diameter **exist** for
every retained region. None is written to disk. Then `build_patch_dataset` pads to
$S = \max(h,w)$ and resizes to $64\times64$, which:

- **preserves** aspect ratio (padding is symmetric, resize is isotropic), and
- **destroys absolute scale** — a $17\times17$ px seed and a $28\times28$ px seed become the same
  $64\times64$ tensor.

Grain length, width and thousand-grain weight are among the *defining* descriptors used to
distinguish rice cultivars. The pipeline gates on them and then throws them away. Recovering them
costs one extra `.npy` of shape $(8624, \sim6)$ and a small MLP into fusion.

### 2.2.11 · M-14 — the band subset was chosen for the wrong hypothesis class, and the elbow is unverifiable

Two independent problems.

**(a) Wrong estimator.** The selection criterion is 5-fold CV accuracy of LDA / `LinearSVC` on
**standardised spatially-averaged mean spectra**, i.e. on $\mathbb{R}^{40}$. Both are *linear
classifiers on the global mean*. They cannot express spatial structure, band ratios, derivative
features, or any interaction — precisely the function classes the 7.88 M-parameter network exists
to fit. The winner ($0.4755$ SVC) and the elbow are therefore optimal for a model that scores
$47.6\,\%$, not for one that scores $89\,\%$. Selecting features under estimator $g$ and deploying
under estimator $f$ is only valid when $\mathcal{H}_g \supseteq \mathcal{H}_f$; here
$\mathcal{H}_g \subset \mathcal{H}_f$ by a wide margin.

**(b) The elbow may be vacuous.** The rule is

$$
k^\star = \min\Big\{k : \mathrm{acc}(k) \ge 0.98 \max_{k'} \mathrm{acc}(k')\Big\} .
$$

The grid is declared as $k \in \{5,10,15,20,25,30,40,50,70,100\}$ *"restricted to
$k \le |\text{ordering}|$"*, yet the recorded curve in `band_selection_report.csv`
**terminates at exactly $k = 40$**, the chosen value, and $\mathrm{acc}$ is still **monotonically
increasing** there ($0.4416 \to 0.4755$ from $k{=}30$ to $k{=}40$, a $+7.7\,\%$ relative jump —
the largest in the table). If $|\mathcal{K}| = 40$ after the $\tau = 0.995$ decorrelation filter,
then $k=40$ is not an elbow at all — it is *the entire available set*, and the criterion is a
tautology.

**[VERIFY]** `wc -l dataset/band_selection_report.csv` and print $|\mathcal{K}|$ from
`band_selection.py` step 2. If $|\mathcal{K}| \le 50$, M-14(b) is confirmed and the "$84.4\,\%$
reduction, validated by an elbow" claim must be withdrawn or re-derived.

**[FALSIFIABLE] F-3.** Re-run selection with the *deployed* estimator in the loop: for
$k \in \{20,40,60,80,120,256\}$, train the actual network for a short fixed budget on a grouped
inner split. Prediction: the curve does **not** plateau at 40, because the network — unlike LDA —
can exploit narrow, individually-low-MI bands through derivative and ratio features.

---

## 2.3 Fusion — `CrossModalInteraction`

### 2.3.1 · M-1 — latent collapse from an initialisation-scale mismatch

The latent array $L \in \mathbb{R}^{4\times256}$ is initialised $\mathcal{N}(0, 0.02^2)$. The keys
and values it queries are the four branch embeddings after **independent LayerNorm**, i.e.
approximately unit-variance per dimension, $\|T_m\| \approx \sqrt{256} = 16$. Meanwhile
$\|L_n\| \approx 0.02\sqrt{256} = 0.32$. The scale ratio is

$$
\frac{\|T_m\|}{\|L_n\|} \;\approx\; \frac{16}{0.32} \;=\; 50 .
$$

Cross-attention logits are $\langle W_Q L_n, W_K T_m\rangle/\sqrt{d_h}$. With
$\|L_n\| \ll \|b_Q\|$ (the query bias), $W_Q L_n + b_Q \approx b_Q$ **for every $n$**, so all four
latents emit *the same* query, hence the same attention distribution $\alpha_{n\cdot}$, hence the
same update:

$$
L_n \leftarrow L_n + \sum_m \alpha_{nm} V(T_m), \qquad \alpha_{1\cdot} \approx \alpha_{2\cdot} \approx \alpha_{3\cdot} \approx \alpha_{4\cdot} .
$$

The latents are permutation-symmetric under the loss, and the only symmetry-breaking is their
$\sigma = 0.02$ initialisation — which is $50\times$ too small to break it. Two depth-2 blocks are
not enough for the residual stream to separate them. The subsequent pooling

$$
\mathbf{f} = \tfrac{1}{4}\sum_n L_n \;\approx\; L_1
$$

therefore averages four near-copies. **The fusion's effective latent capacity is 1, not 4.**

Two independent fixes, both structural: (i) initialise the latents at the scale of the keys
($\sigma = 1/\sqrt{d}$ or a LayerNorm on $L$ before the first cross-attention), and (ii) add a
learned per-latent code $c_n$ so that symmetry is broken by construction rather than by noise.
§3.4 does both and also questions whether $N_{\text{lat}} = 4$ is the right number at all.

**[FALSIFIABLE] F-4.** Log $\max_{n\ne n'}\cos(L_n, L_{n'})$ at the end of Stage 1.
Prediction: $> 0.95$.

### 2.3.2 · M-2a — independent LayerNorm destroys the cross-branch confidence signal

Each branch embedding is normalised *independently and per sample*:

$$
\hat{\mathbf{b}}_m = \gamma_m \odot \frac{\mathbf{b}_m - \mu(\mathbf{b}_m)}{\sigma(\mathbf{b}_m)} + \beta_m .
$$

The documents present this as *"preserving modality structure"*. It does preserve the **direction**
of each embedding. It destroys two things that matter more:

1. **Cross-branch magnitude.** After LN, $\|\hat{\mathbf{b}}_m\| \approx \sqrt{256}\,\|\gamma_m\|$
   for every $m$ regardless of how much evidence branch $m$ actually found. The gate
   $\mathbf{g} = \mathrm{softmax}(W_2\mathrm{GELU}(W_1\mathbf{f}))$ must therefore infer "is branch
   C confident on *this* seed?" from direction alone, having been denied the natural scalar.
2. **Per-sample noise amplification.** LN normalises by the *sample's own* $\sigma(\mathbf{b}_m)$.
   For a seed on which branch $m$ found nothing — a spectrally flat, low-contrast patch —
   $\sigma(\mathbf{b}_m)$ is small and LN **divides by it**, inflating that branch's noise to full
   scale before fusion. The weakest evidence is promoted to the loudest input.

This is the mechanism behind the user-suspected "suppression of weak but critical absorption
features", though the precise statement is the inverse: LN does not attenuate weak features
*within* a branch (relative structure survives), it **erases the ability of the gate to tell a
weak-but-real signal from noise across branches**.

Fix (§3.4): keep a normalisation for conditioning stability, but (a) use a **dataset-statistics**
norm (BatchNorm/RMSNorm with a running scale) rather than a per-sample one, and (b) feed the
pre-norm magnitudes $\big(\log\|\mathbf{b}_A\|,\dots,\log\|\mathbf{b}_D\|\big) \in \mathbb{R}^4$
into the gate as an explicit confidence vector.

### 2.3.3 · M-2b — a softmax gate is an exclusive operator on a conjunctive problem

$$
\mathbf{g} = \mathrm{softmax}(\cdot) \in \Delta^3, \qquad \textstyle\sum_m g_m = 1 .
$$

The simplex constraint means **raising the weight on one modality necessarily lowers the others**.
That is the correct inductive bias when modalities are *substitutes* (any one suffices). It is the
wrong bias when they are *complements*. For 90 varieties of one species discriminated by
sub-percent reflectance differences, the discriminative event is overwhelmingly a **conjunction**:
"a shallow 970 nm feature **and** a particular husk texture". A convex gate cannot express
$g_A \uparrow \wedge g_C \uparrow$; the model must instead route the conjunction through
cross-attention, which is also a convex (softmax) operator.

Replace $\mathrm{softmax}$ with an independent $\sigma$ gate (four free scalars in $(0,1)$), or —
better — with a gate over $2^4$ *interaction* terms via a low-rank bilinear form (§3.4.3).

### 2.3.4 · M-3 — the fusion is entirely first-order

Every operation in `CrossModalInteraction` is a weighted sum:

$$
L \leftarrow L + \mathrm{MHA}(L, T, T),\quad
L \leftarrow L + \mathrm{MHA}(L,L,L),\quad
L \leftarrow L + \mathrm{FF}(\mathrm{LN}(L)),\quad
\mathbf{f} \leftarrow \tfrac14\textstyle\sum_n L_n + \textstyle\sum_m g_m T_m .
$$

Attention weights are input-dependent, so the map is not *globally* linear — but the branch
tokens enter only through convex combinations, and there is **no term of the form
$\mathbf{b}_A \odot \mathbf{b}_C$ or $\mathbf{b}_A^{\top} U \mathbf{b}_C$**. The FF layers can
approximate products, but only after the modalities have already been *summed*, at which point
$\mathbf{b}_A$ and $\mathbf{b}_C$ are no longer separable. For fine-grained recognition, low-rank
bilinear pooling (MLB / MFB / compact bilinear) is the standard remedy and costs far less than the
2.19 M currently spent.

### 2.3.5 · M-4 — 2.19 M parameters to mix four vectors

$$
\underbrace{2 \times 4d^2}_{\text{cross-attn QKVO}} + \underbrace{2 \times 4d^2}_{\text{self-attn}}
+ \underbrace{2 \times 2 \cdot d \cdot 4d}_{\text{FF } 256\to1024\to256}
= 16d^2 + 16d^2 \;\Big|_{d=256} \;\approx\; 2.10\ \text{M}
$$

plus norms, gate and `output_proj` → the recorded **2,190,916**. That is **27.8 % of the entire
model, the single largest component, spent on a $4 \times 4$ mixing problem** — 547,729 parameters
per token. The Perceiver premise is compression ($N_{\text{lat}} \ll N_{\text{in}}$) or expansion
($N_{\text{lat}} \gg N_{\text{in}}$); here $N_{\text{lat}} = N_{\text{in}} = 4$, so it does
neither. Eight heads over four keys gives an attention distribution with entropy $\le \log 4 =
1.386$ nats and a $4\times4$ score matrix per head — 32-dimensional heads resolving a rank-$\le4$
interaction. The parameter mass is not doing work that a gated low-rank bilinear layer with
$<200$ k parameters could not do better (§3.4).

### 2.3.6 · N-9 — the fusion block is not a valid pre-LN transformer

As written:

$$
L \leftarrow L + \mathrm{MHA}(Q{=}L,K{=}T,V{=}T), \qquad
L \leftarrow L + \mathrm{MHA}(L,L,L), \qquad
L \leftarrow L + \mathrm{FF}(\mathrm{LN}(L)) .
$$

LayerNorm appears **only on the feed-forward sublayer**. Both attention sublayers read the raw,
growing residual stream. Over depth 2 the variance accumulates as
$\mathrm{Var}(L^{(2)}) \approx \mathrm{Var}(L^{(0)}) + \sum \mathrm{Var}(\text{sublayer})$ with no
renormalisation, so the second block's attention logits are computed at a different scale than the
first's. This is survivable at depth 2 but it (a) makes the softmax temperature depth-dependent
and (b) **inflates the loss-surface curvature**, which is exactly what Stage 3's SAM is trying to
measure. Two of the four Stage-3 mechanisms in §1.3 interact with this.

### 2.3.7 · N-10 — `output_proj` and `EmbedNet` are the same block twice

$$
\text{output\_proj}: \mathrm{Drop}\big(\mathrm{GELU}(W_o\,\mathrm{LN}(\mathbf{f}))\big) : 256\to256
$$
$$
\text{EmbedNet}: \mathrm{LN}_2\Big(\mathbf{u} + \mathrm{Drop}\big(\mathrm{MLP}_{256\to512\to256}(\mathrm{LN}_1(\mathbf{u}))\big)\Big)
$$

Two sequential $256\to256$ post-fusion refinements, $\approx 330$ k parameters combined, with no
change of width or role between them. One of them is redundant.

### 2.3.8 · A consequence nobody has measured: `EmbedNet`'s LayerNorm pre-empts ArcFace's normalisation

`EmbedNet` ends in $\mathrm{LN}_2$. At initialisation ($\gamma = 1$, $\beta = 0$), LayerNorm forces
zero mean and unit variance across the 256 dimensions, hence

$$
\|\mathbf{e}\|_2 = \sqrt{\textstyle\sum_j e_j^2} = \sqrt{256 \cdot 1} = 16 \quad\text{exactly.}
$$

So $\hat{\mathbf{e}} = \mathbf{e}/\|\mathbf{e}\|_2$ is, at init, a **constant rescaling by $1/16$**,
and remains near-constant thereafter (drifting only as $\gamma,\beta$ move). Two implications:

1. ArcFace's embedding normalisation — the step that is supposed to strip "confidence = magnitude"
   — removes almost nothing, because LayerNorm already removed it one layer earlier.
2. **Stage 1's linear head is already an (unnormalised) cosine classifier.**
   $z_c = \mathbf{w}_c^{\top}\mathbf{e} + b_c = 16\|\mathbf{w}_c\|\cos\theta_c + b_c$ — a cosine
   classifier with a learned *per-class scale* $\|\mathbf{w}_c\|$ and a bias. This partly explains
   why Stage 1 is so strong, and it is the key that makes the Stage-1/Stage-2 head unification of
   §3.5.1 cheap: the geometry is already almost there.

---

## 2.4 Classification heads and margins

### 2.4.1 · M-6 — the adaptive margin is a positive-feedback loop with the wrong sign

$$
M(c) = m_{\text{base}} + m_\Delta\big(1 - \min(F_1^{(c)},1)\big) \in [0.35,\,0.45] .
$$

Decompose class difficulty. A class can have low $F_1$ for two structurally opposite reasons:

| Failure mode | Symptom | Geometric cause | Correct margin response |
|---|---|---|---|
| **Low recall** | its own samples are assigned elsewhere | the class's angular region is *too small* / its prototype is crowded out | **decrease** $m_c$ — let it claim more sphere |
| **Low precision** | it absorbs other classes' samples | its region is *too large* | **increase** $m_c$ — push its boundary in |

$F_1$ is the harmonic mean of the two and **cannot distinguish them**. The rule raises $m_c$ in
both cases. For a low-recall class, raising the additive angular margin means requiring
$\cos(\theta_y + m)$ to beat every $\cos\theta_{c\ne y}$ — i.e. **shrinking the class's decision
region on the hypersphere**, which lowers recall further:

$$
F_1^{(c)}\downarrow \;\Longrightarrow\; M(c)\uparrow \;\Longrightarrow\; \text{region}(c)\downarrow
\;\Longrightarrow\; R_c\downarrow \;\Longrightarrow\; F_1^{(c)}\downarrow
$$

a closed positive-feedback loop, recomputed at **every checkpoint improvement**. For the five
recorded hardest classes (49: 0.519, 52: 0.533, 41: 0.538, 51: 0.629, 37: 0.640) the likely
mechanism is *pairwise confusion* — near-degenerate varieties overlapping in embedding space —
which is a recall failure for at least one member of each confused pair. The rule is therefore
actively harming the classes it is designed to help, and macro-F1 is the metric that feels it.

**[FALSIFIABLE] F-5.** Print the per-class precision/recall for classes 49, 52, 41, 51, 37 from
`test_preds_TTA.npy` and `test_targets.npy` (both on disk). Prediction: at least three of the five
show $R_c \ll P_c$. If so, replace the rule with the signed form of §3.5.3.

### 2.4.2 · M-7 — angular gradient saturation for exactly the classes the margin targets

For the target class, with $\phi_i = \cos(\theta_{i,y} + m_i)$ and logits $s\phi_i$:

$$
\frac{\partial\mathcal{L}}{\partial\phi_i} = s\,(p_{i,y} - 1),
\qquad
\frac{\partial\phi_i}{\partial\theta_{i,y}} = -\sin(\theta_{i,y} + m_i)
$$

$$
\boxed{\;\left|\frac{\partial\mathcal{L}}{\partial\theta_{i,y}}\right| = s\,\big(1 - p_{i,y}\big)\,\sin\!\big(\theta_{i,y} + m_i\big)\;}
$$

Two factors, both pathological at the hard end:

- $\big(1 - p_{i,y}\big)$: for a failing class $p_{i,y}\to 0$, so this factor **saturates at 1**.
  Focal's extra $(1-p_{i,y})^{\gamma}$ multiplier likewise saturates at 1. **There is no additional
  focus available to allocate** — the loss is already giving hard samples its maximum weight.
- $\sin(\theta_{i,y} + m_i)$: rises to a maximum at $\theta + m = \pi/2$, then **falls**.

So raising $m$ from $0.35$ to $0.45$ rad ($20.1° \to 25.8°$) for the hardest classes buys nothing
on the first factor and, once $\theta_y > \pi/2 - m$, strictly *reduces* the second:

| $\theta_y$ | $\theta_y + 0.35$ | $\sin$ | $\theta_y + 0.45$ | $\sin$ | Change |
|---:|---:|---:|---:|---:|---:|
| $1.0$ rad ($57°$) | $1.35$ | $0.976$ | $1.45$ | $0.993$ | $+1.7\,\%$ |
| $1.4$ rad ($80°$) | $1.75$ | $0.984$ | $1.85$ | $0.961$ | $-2.3\,\%$ |
| $1.8$ rad ($103°$) | $2.15$ | $0.837$ | $2.25$ | $0.778$ | $-7.0\,\%$ |
| $2.2$ rad ($126°$) | $2.55$ | $0.558$ | $2.65$ | $0.471$ | $-15.6\,\%$ |

Hard classes live in the bottom rows. The adaptive margin **reduces the gradient signal on the
worst samples by up to 15 %** while the easy-margin guard
($\phi \leftarrow c - m\sin(\pi - m)$ for $c \le \cos(\pi - m)$) only rescues the extreme tail past
$\theta + m > \pi$. Combined with M-6, the mechanism is: *the hardest classes get a larger margin,
which shrinks their region and weakens their gradient.*

### 2.4.3 · M-8 — sub-centre death under hard assignment

$$
\cos\theta_{i,c} = \max_{k\in[3]}\; \hat{\mathbf{e}}_i^{\top}\hat{\mathbf{W}}_{c,k}
$$

`max` is a hard router: for a given $(i,c)$ **exactly one** sub-centre receives gradient. This is
$k$-means with hard assignment on a hypersphere, run with

$$
n_{\text{train}}/\text{class} \approx 67, \qquad K = 3, \qquad \dim = 256 .
$$

i.e. $3 \times 256 = 768$ prototype parameters per class fitted from 67 samples. There is **no
load-balancing term, no usage entropy penalty, no sub-centre repulsion**. The classical failure
follows immediately: once a sub-centre stops winning the max for every sample of its class, its
gradient is identically zero forever — a **dead centroid** that still occupies $256$ parameters and
still participates in the max at inference (where it can only ever *raise* an off-target cosine,
i.e. contribute false positives).

The bootstrap makes this worse, not better. `init_from_linear` sets

$$
\mathbf{W}_{c,k} \leftarrow \hat{\mathbf{w}}^{\text{lin}}_c + 0.01\,k\,\boldsymbol\epsilon,\quad \boldsymbol\epsilon\sim\mathcal{N}(0,I_{256}) .
$$

$\|0.01\boldsymbol\epsilon\| \approx 0.01\sqrt{256} = 0.16$ and $\|0.02\boldsymbol\epsilon\| \approx
0.32$, giving angular tilts of $\arctan(0.16) = 9.1°$ and $\arctan(0.32) = 17.7°$ from $k=0$. The
tilt directions are **isotropic random**, not aligned with any real appearance mode of the class.
So $k=1,2$ start at a random $9$–$18°$ offset and win the max only for samples that happen to lie
in that random direction. With 67 samples in $S^{255}$, the probability that a random $17.7°$ cap
contains a meaningful subpopulation is negligible. **$K=3$ therefore initialises to one live
centroid and two randomly-oriented decoys.**

Fixes in §3.5.2: soft (log-sum-exp) assignment annealed toward max, plus a usage-balance term, plus
data-driven initialisation (per-class $k$-means on the Stage-1 embeddings, which are already
computed by `return_embed=True`).

### 2.4.4 · N-4 — the clamp admits a derivative of 707

$$
s_i = \sqrt{\max\!\big(1 - c_i^2,\,10^{-6}\big)}, \qquad
\frac{\partial s_i}{\partial c_i} = \frac{-c_i}{\sqrt{1-c_i^2}} .
$$

At the clamp $c_i = 1 - 10^{-6}$: $1 - c_i^2 \approx 2\times10^{-6}$, so
$|\partial s/\partial c| \approx 707$. Multiplied by $s = 48$, a single perfectly-aligned sample
can produce a head gradient $\sim 3.4\times10^4$ times the per-unit scale. The global
`clip_grad_norm_(·, 1.0)` then rescales the **entire model's** gradient by
$1/\|\mathbf{g}\|$ (M-10), so one saturated sample can dictate the update direction for every
parameter on that step. Loosening the clamp to $1 - 10^{-3}$ bounds the derivative at $22.4$ —
a $32\times$ reduction — at a cost of $\le 0.06°$ of angular resolution near $\theta = 0$, which is
immaterial.

### 2.4.5 · M-9 — focal loss cannot down-weight easy examples when label smoothing is on

`losses/focal.py` computes

$$
\ell_i = -\sum_c q_{i,c}\log p_{i,c},\qquad
\mathcal{L}_{\text{focal}} = \frac{1}{B}\sum_i \big(1 - e^{-\ell_i}\big)^{\gamma}\,\ell_i,
$$

with $p_t := e^{-\ell}$. When $\varepsilon = 0$ this is exact: $\ell = -\log p_y$ so $p_t = p_y$.
When $\varepsilon > 0$, $\ell$ is the cross-entropy to the *smoothed* target, whose minimum over
$p$ is attained at $p = q$ and equals the entropy of $q$:

$$
\ell \;\ge\; H(q) \;=\; -(1-\varepsilon)\log(1-\varepsilon) - \varepsilon\log\!\frac{\varepsilon}{C-1}
$$

so $p_t = e^{-\ell} \le e^{-H(q)}$ and the modulator is **bounded below**:

$$
\big(1 - p_t\big)^{\gamma} \;\ge\; \big(1 - e^{-H(q)}\big)^{\gamma} .
$$

Numerically, at $C = 90$, $\gamma = 1.5$:

| $\varepsilon$ | $H(q)$ | $\max p_t$ | **min focal modulator** |
|---:|---:|---:|---:|
| $0.10$ (Stage 1, epoch 1) | $0.7739$ | $0.4612$ | $\mathbf{0.3955}$ |
| $0.07$ (mid-stage) | $0.5686$ | $0.5663$ | $0.2857$ |
| $0.04$ (Stage 1, epoch 600) | $0.3475$ | $0.7064$ | $\mathbf{0.1591}$ |
| $0$ (Stage 2, Stage 3) | $0$ | $1.0$ | $0$ ✓ correct |

At $\varepsilon = 0.10$ the focal term's dynamic range is compressed from $[0,1]$ to
$[0.3955, 1]$ — a factor of $2.5$ instead of $\infty$. **Focal loss degenerates into a mild,
monotone rescaling of cross-entropy.** Worse, $p_t = e^{-\ell}$ is no longer the model's confidence
in the target class at all; it conflates confidence with the smoothing entropy floor, so the "hard
example mining" is partly mining *smoothing noise*.

This is active in **Stage 1 Phase 3 — the phase that produced the shipped checkpoint** (epoch 488
of 600, so $\varepsilon(488) = 0.10 - 0.06\cdot\frac{487}{599} = 0.0512$, giving a floor of
$\approx 0.20$). Stage 2 and Stage 3 pass $\varepsilon = 0$ and are unaffected.

Correct form: modulate on the **unsmoothed** target probability,
$\mathcal{L} = (1 - p_{i,y_i})^{\gamma}\,\ell_i^{\text{smoothed}}$ — one line, no new
hyperparameter, restores the full $[0,1]$ range.

### 2.4.6 · The Stage-1 → Stage-2 head discontinuity (drives finding 1.1(i))

`init_from_linear` copies $\hat{\mathbf{w}}^{\text{lin}}_c$ into the sub-centres and is described as
starting "near a known-good boundary". Compare the two decision rules:

$$
\text{Stage 1: } \arg\max_c \big[\mathbf{w}_c^{\top}\mathbf{e} + b_c\big]
\qquad\text{vs.}\qquad
\text{Stage 2: } \arg\max_c \big[48\cdot\hat{\mathbf{w}}_c^{\top}\hat{\mathbf{e}}\big]
$$

Three differences survive the copy:

1. **The bias $b_c$ is dropped.** An affine hyperplane arrangement becomes a spherical Voronoi
   partition. For a class whose $b_c$ was doing real work (rare or systematically-offset classes),
   the boundary moves discontinuously.
2. **The per-class scale $\|\mathbf{w}_c\|$ is normalised away.** Stage 1 learned a per-class
   *temperature*; Stage 2 replaces all of them with the single global $s = 48$.
3. **A margin is applied from epoch 1** ($m_0 = 0.18$), so even at the instant of transfer the
   objective is not the one Stage 1 converged on.

Net: the "warm start" transfers *directions* but not the *boundary*. Empirically Stage 2's best
($0.8867$, epoch 50 of 150) never regains Stage 1's $0.8877$ — consistent with paying a transition
cost it cannot amortise in 150 epochs at a backbone LR of $7\times10^{-5}$.

Because $\|\mathbf{e}\| \approx 16$ by construction (§2.3.8), the fix is nearly free: make
**Stage 1's head a margin-free cosine head** with the same $s$ and the same $K$ sub-centres, so
that Stages 1 → 2 → 3 differ *only* in $m$, sampler and optimiser. `use_arcface` and
`init_from_linear` then become unnecessary, and the curriculum becomes continuous in the objective
rather than piecewise. §3.5.1.

---

## 2.5 Optimisation — and the anatomy of the Stage-3 regression

### 2.5.1 · C-6 — Stage 3 does not implement SAM

The two Stage-3 objectives, verbatim from doc 04 §4.4:

$$
\textbf{ascent: } \mathcal{L}_A = \mathcal{L}_{\text{focal}}(\gamma{=}1) + 0.02\,\mathcal{L}_{\text{supcon}} + 0.10\,\mathcal{L}_{\text{aux}}
$$
$$
\textbf{descent: } \mathcal{L}_D = \mathcal{L}_{\text{focal}}(\gamma{=}1) \quad\text{only}
$$

SAM is defined as $\min_\theta \max_{\|\varepsilon\|\le\rho}\mathcal{L}(\theta+\varepsilon)$ for
**one** $\mathcal{L}$; its first-order realisation is

$$
\theta \leftarrow \theta - \eta\,\nabla\mathcal{L}\big(\theta + \rho\,\hat g\big),\qquad \hat g = \frac{\nabla\mathcal{L}(\theta)}{\|\nabla\mathcal{L}(\theta)\|} .
$$

What runs here is

$$
\theta \leftarrow \theta - \eta\,\nabla\mathcal{L}_D\big(\theta + \rho\,\hat g_A\big),\qquad \hat g_A = \frac{\nabla\mathcal{L}_A(\theta)}{\|\nabla\mathcal{L}_A(\theta)\|},\qquad \mathcal{L}_A \ne \mathcal{L}_D .
$$

Expanding to first order in $\rho$:

$$
\nabla\mathcal{L}_D(\theta + \rho\hat g_A) \;=\; \nabla\mathcal{L}_D(\theta) \;+\; \rho\, H_D\,\hat g_A \;+\; O(\rho^2)
$$

so the SAM correction term is $\rho\,H_D\hat g_A$ instead of $\rho\,H_D\hat g_D$. Decompose
$\hat g_A = \alpha\,\hat g_D + \beta\,u$ with $u \perp \hat g_D$, $\alpha^2 + \beta^2 = 1$:

$$
\rho\,H_D\hat g_A \;=\; \underbrace{\alpha\,\rho\,H_D\hat g_D}_{\text{genuine sharpness penalty}} \;+\; \underbrace{\beta\,\rho\,H_D u}_{\text{curvature-shaped noise}} .
$$

Only the $\alpha$-fraction regularises flatness. The $\beta$-fraction is an **anisotropic
perturbation with no descent guarantee** — it is amplified by whatever the Hessian's large
eigendirections happen to be, which is precisely the direction SAM was supposed to avoid.

How large is $\beta$? $\mathcal{L}_A$ contains $0.10\,\mathcal{L}_{\text{aux}}$, and
$\mathcal{L}_{\text{aux}} = \sum_b \omega_b\mathcal{L}_b$ with $\omega_A = \omega_B = 2$. The four
auxiliary heads are two-layer MLPs sitting **directly** on the branch embeddings — shallow paths
with large, well-conditioned gradients. Their contribution to $\nabla\mathcal{L}_A$ is
disproportionate to their $0.10$ weight because the main path's gradient must traverse fusion +
`EmbedNet` + the $s{=}48$ head. There is every reason to expect $\alpha \ll 1$.

**This alone can explain a large share of the Stage-3 regression**, and it is a five-line fix
(§3.6.3): use the identical objective in both steps.

### 2.5.2 · C-7a — the margin discontinuity at Stage-3 entry undoes Stage 2

| | Terminal Stage 2 | Entry Stage 3 |
|---|---|---|
| Margin source | per-class buffer $M(y_i)$ | scalar `global_m`, buffer **bypassed** |
| Value | $[0.35,\,0.45]$, hardest classes at $0.45$ | uniform $0.30$ |
| Direction | ramped **up** ($0.18 \to 0.35$), then recalibrated per class | annealed **down** ($0.30 \to 0.20$) |

Stage 2 spends 150 epochs building a per-class angular calibration; Stage 3 discards it in one step
and then relaxes it further. For a class that Stage 2 had pushed to $m = 0.45$, the effective
margin drops by $0.15$ rad ($8.6°$) instantaneously, then by another $0.10$ rad over the stage —
a total relaxation of $14.3°$ on exactly the classes macro-F1 weights most.

Note the asymmetry with the documented rationale: the stage describes the decreasing margin as *"a
deliberate late-training relaxation"*. Relaxing the margin during fine-tuning is defensible **if
the per-class structure is preserved** — e.g. $m_c(e) = M(c) \cdot \kappa(e)$ with $\kappa: 1 \to
0.7$. Replacing per-class margins with a uniform scalar is a different operation entirely, and it
is the one that runs.

### 2.5.3 · C-7b — SWA averages a translating trajectory, not an oscillating one

SWA's validity rests on the snapshots being approximately exchangeable samples from a *stationary*
distribution around one basin: $\theta^{(n)} = \theta^\star + \xi_n$ with $\mathbb{E}[\xi_n] = 0$.
Then $\bar\theta \to \theta^\star$ and the average sits nearer the basin's centre than any sample.

Here the objective itself moves between snapshots:

$$
m_{\text{S3}}(e) = 0.25 + 0.05\cos\!\Big(\frac{\pi e}{120}\Big) : \; 0.30 \to 0.20,
\qquad \text{snapshots at } e = 8, 16, \dots, 120 .
$$

So $\theta^{(n)} \approx \theta^\star\big(m(e_n)\big) + \xi_n$ and

$$
\bar\theta \;=\; \frac{1}{15}\sum_n \theta^\star\big(m(e_n)\big) + \bar\xi
\;\approx\; \theta^\star(\bar m) \;+\; \tfrac12\,\mathrm{Var}(m)\,\partial^2_m\theta^\star + \bar\xi ,
\qquad \bar m = 0.25 .
$$

Two errors:

1. The average corresponds to a model trained with $m = 0.25$, whereas the *last* iterate
   corresponds to $m = 0.20$ and neither is the geometry Stage 2 produced. Along a systematic
   drift, weight averaging is a **lag operator** — it sits behind the trajectory rather than at its
   centre.
2. Weight averaging across geometrically different embeddings is only meaningful if the parameter
   space is locally linear over the span. Across a $0.10$ rad margin sweep, class prototypes rotate
   by degrees and the whole hypersphere partition rearranges. That is exactly the regime where
   averaging weights is invalid.

### 2.5.4 · C-7c — the greedy filter is provably a no-op

$$
\text{accept} \iff F_1^{\text{live}}(e) \;\ge\; 0.98\cdot\max_{e'\le e}F_1^{\text{live}}(e')
$$

with the running max updated **before** the test. Two facts make this vacuous:

- If the current epoch sets a new max, the test becomes $F_1(e) \ge 0.98 F_1(e)$ — **always true**.
- Otherwise the band is $2\,\%$ *relative*. At $F_1^{\max}\approx 0.885$ the acceptance floor is
  $0.867$ — while the entire observed spread of the stage lies within roughly $[0.87, 0.89]$.

Recorded outcome: **15 accepted, 0 rejected**. The safeguard designed to keep bad snapshots out of
the average has a threshold $2\times$ looser than the phenomenon it is filtering.

Deeper defect: **the criterion tests the wrong object.** It evaluates the *candidate snapshot's*
$F_1$ and then blindly folds it into the average. A snapshot can be individually excellent and
still move the average away from the optimum (if it lies on the far side of a drift, §2.5.3).
Correct greedy ensemble selection evaluates the **candidate average**:

$$
\text{accept } \theta^{(n)} \iff F_1\Big(\tfrac{n-1}{n}\bar\theta_{n-1} + \tfrac1n\theta^{(n)}\Big) \;>\; F_1\big(\bar\theta_{n-1}\big) .
$$

### 2.5.5 · C-7d — the first SWA snapshot is taken inside Adam's warm-up transient

Stage 3 constructs a **new** `SAM(AdamW)` optimiser, so the second-moment accumulator $v$ starts at
zero. Its effective horizon is

$$
\frac{1}{1-\beta_2} = 1000 \text{ steps}, \qquad \frac{6036}{128} = 47 \text{ steps/epoch}
\;\;\Longrightarrow\;\; \approx 21.3 \text{ epochs to equilibrate.}
$$

The cycle length is **8 epochs**, so snapshots 1 and 2 (epochs 8 and 16) — and arguably 3 (epoch
24) — are taken while $\hat v$ is still biased and the per-parameter step sizes are mis-scaled.
Since the greedy filter accepts everything (§2.5.4), **3 of 15 snapshots come from a
non-stationary, badly-conditioned regime** and enter the average with weight $1/15$ each.

Also note the interaction with §2.5.3: the cycle length (8) and the margin period (240 epochs, of
which 120 are run) are incommensurate, so the LR cycle and the margin sweep never re-align. SWA's
usual justification — "snapshot at the *same phase* of a repeating cycle in a *fixed* landscape" —
holds for the LR but not for the objective.

### 2.5.6 · C-7e — BatchNorm re-estimation is wrong three ways

Only two layers in the model carry `BatchNorm1d`: `branch_c.proj` (C.11) and `branch_d.proj`
(D.12) — i.e. the output projections of the **two highest-capacity branches** (1.69 M and 2.18 M
parameters). `update_bn_stats` resets them and runs one pass over the training loader in
`train()` mode. Three defects:

**(i) Dropout is active during the statistics pass.** In `train()` mode every `nn.Dropout`
fires, plus the `nn.MultiheadAttention` internal dropout of $0.10$ that `set_dropout` cannot reach
(N-2). Branch D's `BatchNorm1d` sits downstream of **four** pre-LN transformer blocks with two
dropout sites each. Inverted dropout preserves the mean but inflates the variance:

$$
\mathrm{Var}\big(\mathrm{Drop}_p(a)\big) = \mathrm{Var}(a) + \frac{p}{1-p}\,\mathbb{E}[a^2]
\;\Big|_{p=0.1}\; = \mathrm{Var}(a) + 0.111\,\mathbb{E}[a^2] .
$$

BN therefore records a $\sigma$ that is systematically **larger** than the eval-time $\sigma$;
at eval the branch's activations are divided by that inflated $\sigma$ and come out attenuated —
**per channel, non-uniformly**, so the downstream LayerNorm in fusion cannot undo it. The correct
implementation puts BN in `train()` while forcing every dropout module to `eval()`.

**(ii) The class prior is wrong.** The pass runs over the *training* loader, which for Stage 3 is
`ClassBalancedBatchSampler` with Stage-2 CDWS weights ($16\times8$, $W_{\max}=3$). BN's $\mu,\sigma^2$
are thus estimated under a **re-weighted class prior** with hard classes over-represented up to
$3\times$, then applied at test under the natural prior. Since BN buffers are a *global* affine
correction, this is a systematic shift of both branch outputs.

**(iii) The pass is device-dependent, and the documentation admits it.** Doc 04 §4.3 states that on
Metal the pass keeps grad **enabled** *"because MPS routes attention through a fused inference
kernel with no dropout support when grad is off"*, and then claims *"forward values, and therefore
the estimated statistics, are identical either way."* These two statements are mutually exclusive:
if dropout is inert without grad and active with grad, the forward values differ by exactly the
dropout noise. Whichever is true, **the SWA checkpoint's BN buffers are not reproducible across
accelerators** — and they are the only stateful buffers the SWA average does not itself compute.

### 2.5.7 · C-7f — Stage 3 destroys its own EMA shadow

Telemetry per doc 05 §5.3:

| Stage | Validation keys logged |
|---|---|
| 1 | `val/{f1_live, acc_live, f1_ema, acc_ema, f1_best}` |
| 2 | `val/{…}` (same set) |
| 3 | `val/{f1_live, acc_live, f1_best}` — **no `f1_ema`** |

Stage 3 never scores its EMA shadow, and at the end *"the SWA weights are copied into the EMA
shadow"*. So the shadow — the averaging scheme that carried Stages 1 and 2, and the object
`final_eval` **always** evaluates — is overwritten by SWA without the two ever being compared. Two
averaging schemes were available; the stage picked one blind.

Note this also breaks the semantics of the checkpoint bundle: for Stages 1–2, `ema` means "EMA of
the trajectory"; for Stage 3 it means "SWA average". Any downstream code that assumes the former is
silently wrong.

### 2.5.8 · M-10 — one global gradient clip across a model with $10^3$-fold gradient scale disparity

`clip_grad_norm_(model.parameters(), 1.0)` computes one global norm and applies a **single scalar**
$c = \min\!\big(1, 1/\|\mathbf{g}\|_2\big)$ to every parameter. The model's gradient scales are not
remotely comparable:

- **ArcFace head:** $\partial\mathcal{L}/\partial\hat{\mathbf{W}}_{y,k^\star} \propto s\,(p_y - 1)\,\hat{\mathbf{e}}$
  with $s = 48$ and $\|\hat{\mathbf{e}}\| = 1$. Over a batch of 128 with several hard samples,
  $\|\mathbf{g}_{\text{head}}\|$ is readily $O(10)$–$O(10^2)$.
- **`MaskedSpectralECA`:** 6 parameters, gradient $O(10^{-2})$.
- **Branch A/B:** documented to collapse toward zero (§2.2.2).

The system's own diagnostic exists to observe exactly this — `branch_grad_norm_tensors` reports six
group norms *before* the clip. The consequence is that on any step where the head is large, the
**backbone's effective learning rate is divided by $\|\mathbf{g}\|$**. If $\|\mathbf{g}\| = 50$ on
a hard batch, the backbone receives $2\,\%$ of its gradient — on precisely the batches that carry
the most information. This is a coupling artefact, not regularisation.

In Stage 3 it is worse: the clip is applied **before both SAM steps**, so it also rescales the
ascent gradient, which changes $\hat g_A$'s magnitude but not its direction — harmless — while the
descent clip re-couples head and backbone as above.

### 2.5.9 · The Stage-2 → Stage-3 objective is not a refinement; it is a different problem

| Term | Stage 2 (effective) | Stage 3 | Ratio |
|---|---:|---:|---:|
| $\mathcal{L}_{\text{cls}}$ weight | $1 - 0.40 - 0.18 = 0.42$ | $1.00$ | $\times 2.38$ |
| Focal $\gamma$ | $1.5$ | $1.0$ | less hard-example focus |
| $\lambda_{\text{supcon}}$ | $0.40$ | $0.02$ | $\times 1/20$ |
| $\lambda_{\text{proto}}$ | $0.18$ | $0$ (module passed, term never applied — N-1) | $\times 0$ |
| $w_{\text{aux}}$ | scheduled $\to 0.25$ | fixed $0.10$ | $\times 0.4$ |
| Margin | per-class $[0.35, 0.45]$ | uniform $0.30 \to 0.20$ | see §2.5.2 |
| Dropout | $0.10$ | as set | — |

Every single knob moves in the direction of *less* embedding-geometry shaping and *more* raw
classification. Stage 3 is therefore not fine-tuning Stage 2's solution — it is optimising a
different objective starting from Stage 2's weights, and SWA faithfully averages along the
resulting drift.

**The decisive diagnostic.** If loss re-weighting or class-prior effects were the cause, macro-F1
would fall *more* than accuracy, because macro-F1 weights the rare/hard classes equally. Recorded:

$$
\Delta F_1^{\text{macro}} = 0.8867 \to 0.8745 = -0.0122, \qquad
\Delta \text{Acc} = 0.8864 \to 0.8748 = -0.0116 .
$$

The two fall by the **same** amount ($-1.22$ vs $-1.16$ points). That is the signature of a
**uniform, global mis-calibration of the evaluated weights**, not of a class-reweighting effect.
The mechanisms that produce a uniform shift are exactly §2.5.1 (wrong SAM direction), §2.5.3
(averaging a drift), §2.5.5 (averaging a transient) and §2.5.6 (wrong BN buffers). The mechanisms
that would produce a macro-heavy shift (§2.5.2 margin, §2.5.9 loss reweighting) are present but
apparently second-order. **Fix the four uniform mechanisms first.**

---

## 2.6 Test-time augmentation — C-8, derived in full

### 2.6.1 · The transform and the invariant it breaks

$$
\mu_{b,c} = \frac{1}{HW}\sum_{h,w}x_{b,c,h,w}, \qquad x^{(s)} = \mu + (x - \mu)\,s,
\qquad s \in \{0.95,\,0.98\overline{3},\,1.01\overline{6},\,1.05\} .
$$

The mean is taken over **all $HW = 4096$ pixels including the exact-zero background**. Writing
$f$ for the foreground fraction and $m_c$ for the true foreground mean:

$$
\mu_c = f\,m_c .
$$

A background pixel ($x = 0$) maps to

$$
x^{(s)}_{\text{bg},c} = \mu_c(1-s) = f\,m_c\,(1-s) .
$$

At $f = 0.284$ (the measured value for patch 0) and $s = 0.95$: $x_{\text{bg}} = 0.0142\,m_c$ —
**$1.4\,\%$ of the mean foreground reflectance, three orders of magnitude above the $10^{-5}$ mask
threshold.** Hence, as doc 05 already records, the foreground fraction reported by
$\mathbb{1}[\sum_c|x_c| > 10^{-5}]$ becomes $1.0$: the entire frame reads as seed.

The documents report this as a *"derived property, verified numerically"* and note the ensemble
still gains. What follows is what that property costs, module by module.

### 2.6.2 · Module-by-module corruption

**`MaskedSpectralECA`.** With $m \equiv 1$ and $n_b = 4096$:

$$
\mu^{\text{masked}}_c \;=\; \frac{1}{4096}\sum_p x^{(s)}_{c,p} \;=\; \mu_c + s\Big(\tfrac{1}{4096}\textstyle\sum_p x_{c,p} - \mu_c\Big) \;=\; \mu_c \;=\; f\,m_c
$$

— **independent of $s$, and a factor $f = 0.284$ smaller than the training-time value $m_c$.** The
first learned operation in the network (a 6-parameter $\mathrm{Conv1d}(2\to1,k{=}3)$ followed by a
sigmoid) therefore receives its first input row scaled by $1/3.52$. The gate
$g = \sigma(\mathrm{Conv1d}(\mu,\nu))$ shifts systematically, and since $x' = x + x\odot g$ the
entire cube is re-weighted by a band-gate the network never saw during training.

**`extract_grid_spectra`.** The denominator $\max(\mathrm{AvgPool}_{4\times4}(m), 10^{-5})$ becomes
$1$ everywhere, so

$$
G_{(i,j),c} \;=\; \mathrm{AvgPool}_{4\times4}\big(x^{(s)}\big)_{c,i,j}
$$

— **plain average pooling including background**. The four corner cells, which at training time
clamp to $\mathbf{0}$, now all return the *same* constant vector $f\,m_c(1-s)$. Branch A's
unweighted 16-cell mean and Branch D's 17-token spatial attention are both fed with up to four
identical background artifacts.

**`masked_spectral_stats`.** This is the severe one. With $n_b = 4096$ and no $+\infty$ fill,
$71.6\,\%$ of the sorted population is background at value $f m_c(1-s)$, while foreground values sit
near $\mu_c + s(m_c - \mu_c) \approx 3.4\,\mu_c$. Hence:

| # | Statistic | Training-time meaning | Under spectral TTA |
|---|---|---|---|
| 1 | mean | $m_c$ | $f m_c$ — scaled by $0.284$, $s$-independent |
| 2 | std | foreground dispersion | $\approx s\sqrt{f(1-f)}\,m_c = 0.451\,s\,m_c$ — dominated by the **background/foreground bimodality**, not by the seed |
| 3 | max | brightest seed pixel | unchanged in rank, shifted by $\mu(1-s)$ |
| 6 | $p_{10}$ | 10th pct of seed pixels | **$= f m_c(1-s)$ — a pure background constant** |
| 7 | $p_{25}$ | 25th pct of seed pixels | **$= f m_c(1-s)$ — identical to $p_{10}$** |
| 8 | $p_{75}$ | 75th pct of seed pixels | $\approx$ the **1st–15th** percentile of the seed |
| 9 | $p_{90}$ | 90th pct of seed pixels | $\approx$ the **65th** percentile of the seed |
| 4 | skewness | seed illumination asymmetry | two-point law: $\dfrac{1-2f}{\sqrt{f(1-f)}} = 0.958$ — **the same for every band** |
| 5 | kurtosis | seed illumination tail | two-point law: $3 + \dfrac{1-6f(1-f)}{f(1-f)} = 1.916$ — **the same for every band** |

So under the four spectral views, statistics 6 and 7 become *identical constants*, statistics 4 and
5 become *band-independent constants determined solely by $f$*, statistic 2 measures the mask
geometry rather than the seed, and statistics 8–9 measure the wrong tail. **Branch B — 686 k
parameters — is fed a tensor that shares almost nothing with its training distribution.**

### 2.6.3 · The transform is not in the training augmentation family

TTA views must be drawn from the transformation distribution the model was trained to be invariant
to. Compare:

| Train-time (doc 02 §2.6) | Test-time spectral view |
|---|---|
| multiplicative: $x \leftarrow x \odot (1 + 0.05 s\,\epsilon_c)\odot m$ — **explicitly re-masked** | $x^{(s)} = \mu + (x-\mu)s$ — **not masked** |
| spectral noise: $x \leftarrow x + \sigma\,m\odot\epsilon$ — **explicitly masked** | — |
| band drop / band cutout: zeroing, mask-preserving | — |
| spectral warp: $\lambda$-axis resample, mask-preserving | — |

Every train-time spectral augmentation preserves the background-zero invariant, by explicit
multiplication by $m$. The test-time one does not, and it applies a *contrast about the mean*
operation that appears nowhere in training. **The train and test augmentation families are
disjoint.**

The dihedral $D_4$ views are a different story: they are exact symmetries of the input lattice,
preserve the mask exactly, and match the train-time `spatial` augmentation (random flips + $k\cdot
90°$ rotations, applied unconditionally under every active profile). Those 8 views are sound.

**[FALSIFIABLE] F-6.** Run `final_eval` with `tta_spatial=8 tta_spectral=0` on the same
checkpoint. Prediction: macro-F1 $\ge 0.8933$, i.e. the 8 dihedral views alone match or beat the
12-view ensemble, because the 4 spectral views are averaging in logits from off-manifold inputs.
This is a **single command** and it either confirms C-8 or refutes it outright.

### 2.6.4 · A second, quieter TTA defect

*"TTA forward passes run under `autocast` at the device default — unlike `engine/evaluate.py`,
which forces fp32."* So the headline number ($0.8933$, TTA) and the reference number ($0.8770$,
no-TTA) are computed at **different numerical precision**. Part of the $+0.0163$ delta is a
precision difference, not an ensembling gain. `evaluate.py`'s justification for forcing fp32 — *"a
stage's reported F1 never depends on the AMP state it was called from"* — applies verbatim to
`tta.py` and is not applied there.

---

## 2.7 Dead control paths and hygiene

| ID | Path | Declared | Actually runs | Consequence |
|---|---|---|---|---|
| N-1a | `model.fusion_heads` | $4$ | module default **$8$** | The 2.19 M-parameter fusion runs at $d_h = 32$ instead of $64$; any ablation on head count is inert |
| N-1b | `model.specf_drop` | $0.15$ | call site hardcodes **$0.10$** | SpecFormer regularisation is $33\,\%$ weaker than configured |
| N-1c | `model.wl_embed_dim` | $16$ | accepted, unused | Suggests an intended learned $\lambda$-embedding that was never wired — relevant to §3.3.1 |
| N-1d | `model.branch_drop_prob` | $0.20$, set to $0$ by Stage 3 | forward uses hardcoded $(0,0,0.30,0.20)$ | **Stage 3 believes it has disabled branch dropout and has not.** Branch C is still dropped $22.5\,\%$ of the time during SAM steps *and* during the BN re-estimation pass |
| N-1e | Stage-3 `proto_weight = 0.01` | passed with a `proto` module | no ProtoNCE term in `train_one_epoch_sam` | Telemetry key `sched/proto_weight` reports a number that affects nothing |
| N-1f | `SpecFormerBranch(physical_wl, patch_size)` | accepted | unused | The wavelength axis is available to Branch D and deliberately discarded (§3.3.3) |
| N-2 | `set_dropout(p)` | "sets `p` on every `nn.Dropout`" | misses `nn.MultiheadAttention.dropout` (a float attribute) | Fusion and SpecFormer attention dropout is pinned at $0.10$ for all 870 epochs, including the Stage-3 BN pass (§2.5.6) |
| N-3 | `spec_pos_embed` capacity $\lfloor 40/4\rfloor + 2 = 12$; 11 used | — | 1 dead row (256 params) | Harmless, but signals the capacity formula and the token count disagree |
| N-5 | `nan_to_num` on eval logits | "non-finite logits are clamped rather than raised" | — | A numerically broken model scores $\approx 1/90$ instead of crashing; no counter is logged |
| N-6 | `use_amp = (supcon is None) ∧ (scaler is not None)` | — | — | Precision policy is a function of loss composition, so the documented "contrastive off" ablation changes precision *and* loss — it is not a clean ablation |
| N-7 | `np.random.default_rng()` per `__iter__` in both samplers | — | — | No single-seed ablation delta is interpretable; §5.5's own caveat says so but the ablation table does not carry error bars |
| N-8 | `stage2_arcface.py` reads `param_groups[0]`, `[2]` | — | — | Any reordering in `build_optimizer_s2` silently mislabels the logged LRs |

**Highest-severity item in this table is N-1d.** Stage 3 sets `branch_drop_prob = 0`, which is
documented as the stage's way of turning off stochastic branch masking for fine-tuning. It does not
take effect. Therefore during Stage 3: (a) each SAM ascent/descent pair is computed on a randomly
masked network, adding variance to $\hat g_A$ on top of §2.5.1; (b) the SWA snapshots are taken from
a model whose training-time forward is stochastic in a way the stage believes it disabled; and (c)
the BN re-estimation pass runs with branch masking live. This belongs in the Stage-3 causal chain
of §1.3 as mechanism 9.

---

# 3 · PROPOSED ARCHITECTURAL IMPROVEMENTS

## 3.0 Design principles

Five rules govern every change below. They are stated first so that each specification can be
checked against them rather than argued individually.

1. **Every branch must consume a view no other branch can reconstruct.** Duplicated inputs
   guarantee gradient collapse in the weaker branch and force compensatory hacks (§2.2.2).
2. **Physics enters as geometry, not as an additive hint.** Wavelength must parameterise the
   *kernels* and the *attention bias*, not merely be summed into the features (§2.2.6).
3. **Invariants declared in preprocessing must hold at inference.** The background-zero invariant
   is load-bearing for four modules; any test-time transform that breaks it is a bug regardless of
   its measured effect (§2.6).
4. **A curriculum must be continuous in its objective.** Stages should differ in *one* dimension at
   a time, so that a regression is attributable (§2.4.6, §2.5.9).
5. **Averaging schemes require stationarity.** Do not average weights across a moving objective, a
   transient optimiser state, or a re-estimated normalisation (§2.5.3–2.5.6).

---

## 3.1 Data contract and protocol (**P**-series)

### P-1 · Session-disjoint grouped splits *(fixes C-1)*

Replace the stratified patch-level split with a grouped one. Persist the group key at extraction
time — it is available in `patch_extraction` and currently discarded.

```python
# data/prep/patch_extraction.py  — write one extra array
groups[i] = scan_id            # int index of "<session>/<variety>-<n>"
np.save(root / "groups.npy", groups)          # (8624,) int64

# data/loaders.py::build_splits
from sklearn.model_selection import StratifiedGroupKFold
sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42)   # 70/15/15 via 5+1+1
```

Contract to add to `tests/regression/`:

$$
\big(\text{scan}(i) \in \text{train}\big) \;\Longrightarrow\; \big(\text{scan}(i) \notin \text{val} \cup \text{test}\big)
$$

If some varieties have a single scan, a session-disjoint split is impossible for them; report the
count explicitly and fall back to **leave-one-scan-out cross-validation** for those classes rather
than silently mixing.

### P-2 · Make the data reflectance, or make it scale-invariant *(fixes C-1's radiometric half)*

Two options, in order of preference.

**(a) White-reference division**, if a white panel exists in the archive:

$$
\varrho_{h,w,c} = \frac{R_{h,w,c} - \bar D_{w,c}}{\bar{W}_{w,c} - \bar D_{w,c} + \epsilon}
$$

**(b) Per-pixel Standard Normal Variate (SNV)** — the chemometric standard, applied along
$\lambda$ *after* masking, if no white panel exists:

$$
\tilde x_{c,p} \;=\; \frac{x_{c,p} - \bar x_{\cdot,p}}{\mathrm{sd}_c(x_{\cdot,p}) + \epsilon},
\qquad \bar x_{\cdot,p} = \tfrac{1}{C_\lambda}\textstyle\sum_c x_{c,p}
$$

SNV removes **exactly** the per-pixel scalar gain $a_p$ of §2.2.5 and the session gain $S_c$ of
§2.1.1 to first order. Store *both* the SNV cube and the per-pixel gain $\big(\bar x_{\cdot,p},
\mathrm{sd}_c\big)$ as two extra channels, so the network can still use brightness if it is
genuinely varietal — but must do so explicitly rather than by default.

> **Consequence to state in the paper.** P-1 and P-2 will *lower* the headline number. That is the
> correct direction. A grouped, SNV-normalised $0.75$ is a publishable claim about rice varieties;
> an ungrouped, radiance-domain $0.89$ is a claim about acquisition sessions.

### P-3 · Fix the resize order *(fixes M-11)*

```python
# data/prep/patch_extraction.py
mask_sq  = center_pad(mask.astype(np.float32), S)             # (S,S)
cube_sq  = center_pad(P, S)                                   # (S,S,256), already masked
mask_rs  = cv2.resize(mask_sq, (64,64), interpolation=cv2.INTER_AREA)   # = alpha map
cube_rs  = cv2.resize(cube_sq, (64,64), interpolation=cv2.INTER_AREA)
keep     = mask_rs > 0.5
cube_rs[~keep] = 0.0
cube_rs[keep] /= mask_rs[keep][..., None]      # undo partial-coverage attenuation
```

The resized mask **is** the fill map $\alpha_p$; dividing by it restores the unattenuated spectrum
on boundary pixels and re-establishes the exact-zero background. Persist `mask_rs` as a
$(8624, 64, 64)$ `uint8`/`float16` array so the network never has to re-derive the mask from a
$>10^{-5}$ threshold (which is what makes it fragile under TTA, §2.6).

### P-4 · Persist morphometrics *(fixes M-13)*

`segment()` already computes them. Write `morphology.npy` of shape $(8624, 8)$:

$$
\mathbf{s}_n = \big[\text{area},\ \text{major axis},\ \text{minor axis},\ \text{axis ratio},\
\text{eccentricity},\ \text{solidity},\ \text{equiv. diameter},\ \text{perimeter}/\sqrt{\text{area}}\big]
$$

standardised on the **training split only**. These are in physical pixel units — the absolute scale
that the $64\times64$ resize destroys — and grain length/width is a defining cultivar descriptor.
Consumed by a 5th fusion token (§3.4.4) at a cost of $\approx 3$ k parameters.

### P-5 · Split the validation split *(fixes C-9)*

| Purpose | New source |
|---|---|
| Per-class margins, CDWS, P3 oversampling weights | **`calib`** — an inner split carved from *train*, never used for gradients |
| Early stopping, checkpoint selection, SWA acceptance | `val` — untouched by any fitted parameter |
| Final metrics | `test` |

Suggested partition under P-1: $60\,\%$ train / $10\,\%$ calib / $15\,\%$ val / $15\,\%$ test, all
group-disjoint. This removes 270 fitted parameters from the selection split and should collapse
most of the observed $1.07$-point val→test gap.

---

## 3.2 Front-end (**FE**-series)

### FE-1 · A wavelength-aware, spacing-normalised band axis *(fixes C-5)*

The root problem is that the 40 selected $\lambda_i$ are non-uniform and every 1-D operator assumes
they are not. Two complementary mechanisms:

**(a) Continuous (implicit) kernels.** Replace `Conv1d(k)` on the band axis with a kernel whose
weights are generated from wavelength *offsets*:

$$
\big(\mathrm{Conv}_\lambda F\big)_{o,i} \;=\; \sum_{j \in \mathcal{N}_k(i)} \Big[\kappa_\phi\big(\lambda_j - \lambda_i\big)\Big]_{o,\cdot}\; F_{\cdot,j},
\qquad
\kappa_\phi:\mathbb{R}\to\mathbb{R}^{C_{\text{out}}\times C_{\text{in}}}
$$

with $\kappa_\phi$ a 2-layer MLP over a Fourier-featurised $\Delta\lambda$ (`SIREN`/`CKConv`
style). $\mathcal{N}_k(i)$ is the $k$ nearest bands **in $\lambda$**, not in index. Parameter cost
is *lower* than the current towers because the kernel is shared across widths; the
$\Delta\lambda$-dependence is learned once.

**(b) Explicit derivative channels.** Cheaper, and worth doing regardless of (a). Append two
Savitzky–Golay derivative bands computed on the **irregular grid** (a local weighted least-squares
polynomial fit, which handles non-uniform $\lambda$ natively):

$$
F \;\leftarrow\; \Big[\,F \;\Big\|\; \tfrac{\partial F}{\partial\lambda} \;\Big\|\; \tfrac{\partial^2 F}{\partial\lambda^2}\,\Big]
$$

First and second spectral derivatives are the workhorse features of VIS–NIR chemometrics; the
current architecture cannot compute them correctly and therefore does not have them.

**Retire `model.wl_embed_dim`.** The unused key (N-1c) is the natural home for $\kappa_\phi$'s
Fourier-feature width — wire it rather than delete it.

### FE-2 · Make the mask an input, not a threshold *(supports C-8, M-11)*

Every masked operator currently re-derives $m$ from $\mathbb{1}[\sum_c|x_c| > 10^{-5}]$. That
threshold is what breaks under TTA (§2.6.1) and what mis-counts partial pixels (§2.2.8). With P-3
the true mask is on disk; pass it explicitly:

```python
def forward(self, x, mask=None, ...):
    if mask is None:                       # backward-compatible fallback
        mask = (x.abs().sum(1, keepdim=True) > 1e-5).float()
```

and thread `mask` through `MaskedSpectralECA`, `extract_grid_spectra` and
`masked_spectral_stats`. This makes the four masked modules **immune** to any brightness/contrast
transform, at test time or otherwise, and is a prerequisite for the TTA fix of §3.7.

---

## 3.3 Branch redesign (**BR**-series)

The controlling constraint: **each branch must see something the others cannot reconstruct.** The
proposed allocation:

| Branch | New role | New input | Unique information |
|---|---|---|---|
| **A** | Continuum & derivative profile | per-cell SNV spectra + $\partial_\lambda$, $\partial^2_\lambda$ on the $\lambda$-metric | spectral *shape*, gain-free |
| **B** | Scale-invariant index bank + morphology | learned normalised-difference indices + $\mathbf{s}_n$ | ratios and physical size — neither derivable from A |
| **C** | Joint spectral–spatial | $(40,64,64)$ cube with 3-D stem | spatial texture × spectral position |
| **D** | $\lambda$-aware spectral transformer | grid spectra with **continuous $\lambda$** tokens and relative-$\lambda$ attention bias | long-range band interactions |

### BR-1 · Branch B: from rank-2 moments to a scale-invariant index bank *(fixes C-4)*

Delete the nine raw moments. Replace with three groups, none of which is rank-deficient under the
gain model $x_{c,p} = a_p r_c$:

**(i) Learned normalised-difference indices.** The classical scale-invariant spectral feature:

$$
\mathrm{NDI}_{ij} = \frac{r_i - r_j}{r_i + r_j + \epsilon} \quad\Longrightarrow\quad
\mathrm{NDI}_{ij}\big(a\,\mathbf{r}\big) = \mathrm{NDI}_{ij}(\mathbf{r}) \;\; \forall a > 0
$$

Rather than enumerating all $\binom{40}{2} = 780$ pairs, learn a **low-rank soft index bank**:

$$
u_k = \boldsymbol\pi_k^{+\top}\mathbf{r}, \quad v_k = \boldsymbol\pi_k^{-\top}\mathbf{r},
\qquad
z_k = \frac{u_k - v_k}{u_k + v_k + \epsilon}, \qquad k = 1..64
$$

with $\boldsymbol\pi_k^{\pm} = \mathrm{softmax}(\theta_k^{\pm}) \in \Delta^{39}$ — soft band
selectors, so each index is a differentiable, interpretable *pair of spectral regions*. Cost:
$2 \times 64 \times 40 = 5{,}120$ parameters. Output $\mathbf{z}\in\mathbb{R}^{64}$ is exactly
invariant to $a_p$ and to the session gain $S_c$ *if* $S$ is locally flat.

**(ii) Continuum-removed depths.** Hull-normalise $\mathbf{r}$ and read the depth at the $K$
deepest features — a physically meaningful, gain-free absorption descriptor.

**(iii) Morphology.** $\mathbf{s}_n \in \mathbb{R}^{8}$ from P-4.

New Branch B: $\big[\mathbf{z} \,\|\, \text{depths} \,\|\, \mathbf{s}_n\big] \to
\mathrm{MLP} \to \mathbb{R}^{256}$, **$\approx 90$ k parameters instead of 686 k**, on a genuinely
full-rank input. The freed $\approx 600$ k goes to Branch C (BR-3).

> If the moments must be retained for continuity, at minimum apply SNV first (P-2b). After SNV the
> per-pixel gain is removed and the nine statistics stop being collinear — the rank-2 proof of
> §2.2.5 no longer applies.

### BR-2 · Branch A: de-duplicate from D, and pool by foreground mass *(fixes C-2, M-12)*

Three changes, none of which adds parameters:

1. **Different input from D.** A consumes the *derivative* representation (FE-1b) — first and
   second $\lambda$-derivatives of the per-cell SNV spectrum. D keeps the raw grid spectra. These
   are related but not mutually reconstructible in a finite receptive field, and they emphasise
   opposite ends of the spectral frequency band.
2. **Mass-weighted cell pooling.** Replace the unweighted mean with the cell foreground mass
   $\omega_n = \mathrm{AvgPool}_{4\times4}(m)_n$, which is *already computed* as the normaliser in
   `extract_grid_spectra` and currently discarded:

$$
\mathbf{b}_A = \frac{\sum_{n=1}^{16}\omega_n\,\psi_A(G_n)}{\sum_{n=1}^{16}\omega_n + \epsilon}
$$

3. **Finer grid.** $4\times4$ is a $256{:}1$ compression (§2.2.3). Move to $8\times8$ ($N_g = 64$
   cells, $64{:}1$). Branch A processes cells independently so the parameter count is unchanged;
   only the flattened batch grows from $16B$ to $64B$. If throughput matters, keep $4\times4$ for D
   and use $8\times8$ for A — they are now different branches with different inputs anyway.

### BR-3 · Branch C: a genuine spectral–spatial stem *(fixes C-3)*

Replace the two-line `band_reduce` with a factorised 3-D stem that keeps the spectral axis alive
for at least two stages:

```
x  (B, 1, 40, 64, 64)                      # add a singleton feature axis
 ├─ Conv3d(1→16,  k=(7,3,3), stride=(2,1,1), pad=(3,1,1))   → (B,16,20,64,64)
 ├─ GroupNorm(4,16) + GELU
 ├─ Conv3d(16→32, k=(5,3,3), stride=(2,2,2), pad=(2,1,1))   → (B,32,10,32,32)
 ├─ GroupNorm(8,32) + GELU
 ├─ Conv3d(32→64, k=(5,3,3), stride=(2,2,2), pad=(2,1,1))   → (B,64, 5,16,16)
 └─ reshape (B, 64·5=320, 16, 16) → 1×1 → (B,192,16,16)     # spectral axis folded, not deleted
      └─ existing ResBlock2D / CBAM tail
```

Parameter cost of the 3-D stem: $\approx 16\cdot7\cdot9 + 16\cdot32\cdot5\cdot9 + 32\cdot64\cdot5\cdot9
\approx 116$ k — comfortably inside the $600$ k freed by BR-1. The key property is that

$$
h_{j,h,w} \;=\; \sigma\Big(\sum_{c,\delta h,\delta w} W_{j,c,\delta h,\delta w}\,x_{c,h+\delta h,w+\delta w}\Big)
$$

is now representable: *"this absorption feature, in this part of the seed."* Use the persisted mask
(FE-2) to zero padded regions after every stage so the CNN never learns the frame.

Also: **remove Branch C from the drop vector** (§2.2.7, M-5) and, if branch dropout is retained at
all, invert it to $p = (0.15,\,0.15,\,0,\,0.15)$ — drop the reconstructible branches, protect the
irreplaceable one. Wire `model.branch_drop_prob` so Stage 3 can actually disable it (N-1d).

### BR-4 · Branch D: give the transformer the wavelength axis *(fixes C-5 for D)*

`SpecFormerBranch` currently accepts `physical_wl` and ignores it, tokenises by index stride 4, and
adds a **learned positional embedding indexed by token position**. That embedding is
un-transferable: re-run band selection at a different $k$ and token 3 means a different wavelength
region. Three changes:

**(i) $\lambda$-derived token embedding.** Each token $t$ covers a set of bands
$\mathcal{S}_t$; define its centre $\bar\lambda_t = \frac{1}{|\mathcal{S}_t|}\sum_{i\in\mathcal{S}_t}\lambda_i$
and replace `spec_pos_embed[:11]` with

$$
\mathrm{PE}_t = \mathrm{PhysicalWavelengthPE}\big(\tilde{\bar\lambda}_t\big) \in \mathbb{R}^{256}
$$

reusing the **existing** module (it is already a shared instance across A and B — make it four-way
shared). The `spec_cls` token keeps its own learned code.

**(ii) Relative-$\lambda$ attention bias.** Add a scalar bias to every spectral-stage attention
logit:

$$
\mathrm{Attn}_{tu} = \mathrm{softmax}\!\left(\frac{q_t^{\top}k_u}{\sqrt{d_h}} \;+\; b_\psi\big(\bar\lambda_t - \bar\lambda_u\big)\right),
\qquad b_\psi:\mathbb{R}\to\mathbb{R}^{n_{\text{heads}}}
$$

with $b_\psi$ a tiny MLP (a few hundred parameters). This lets a head specialise on
*"bands 60 nm apart"* — the natural unit for absorption-feature pairs — instead of *"tokens 2
apart in an arbitrary index."*

**(iii) $\lambda$-uniform tokenisation.** Rather than `stride = specf_patch // 2` over the index,
partition $[\lambda_{\min}, \lambda_{\max}]$ into $L$ equal-width windows and pool the bands falling
in each. Token $t$ then always means the same spectral region regardless of the selected subset —
which also makes the branch transferable across band-count ablations (F-3).

Wire `model.specf_drop` (N-1b) and reduce the branch to $\approx 1.2$ M parameters by dropping
$d_{\text{model}}$ from 256 to 192 in the spectral stage; with $\lambda$-aware attention it needs
less brute capacity.

---

## 3.4 Fusion redesign (**FU**-series) — 2.19 M → ≈ 0.55 M

### FU-1 · Fix the latent scale, or delete the latents *(fixes M-1, M-4)*

Two viable designs. Prefer **(b)** unless a fifth+ modality is planned.

**(a) Keep the Perceiver, make it work.**
- Initialise $L \sim \mathcal{N}(0, d^{-1})$ so $\|L_n\| \approx 1$ against LayerNormed keys of the
  same scale (currently $50\times$ off), **or** apply `LayerNorm` to $L$ before the first
  cross-attention.
- Add a learned per-latent code $c_n \in \mathbb{R}^{256}$ so symmetry is broken by construction.
- Add a diversity penalty $\mathcal{L}_{\text{div}} = \sum_{n\ne n'}\big(\cos(L_n, L_{n'})\big)^2$
  with a small weight, and log $\max_{n\ne n'}\cos(L_n,L_{n'})$ as a first-class diagnostic (F-4).
- Replace the mean-over-latents with a **concatenate + project**: $\mathbb{R}^{4\times256} \to
  \mathbb{R}^{1024} \to \mathbb{R}^{256}$, so the four latents are not forced to be redundant.

**(b) Delete the latents.** With $N_{\text{mod}} = 4$ (5 with morphology), latent cross-attention
compresses nothing. Replace the whole module with:

$$
\text{gate: } \boldsymbol\gamma = \sigma\Big(W_g\big[\hat{\mathbf b}_A\|\cdots\|\hat{\mathbf b}_D\|\,\boldsymbol\nu\,\big]\Big) \in (0,1)^4
\qquad (\text{independent, not softmax — §3.4.2})
$$
$$
\text{first order: } \mathbf{f}_1 = \sum_m \gamma_m \hat{\mathbf{b}}_m
$$
$$
\text{second order: } \mathbf{f}_2 = \sum_{m<m'} \big(U_m\hat{\mathbf{b}}_m\big)\odot\big(U_{m'}\hat{\mathbf{b}}_{m'}\big),
\qquad U_\cdot \in \mathbb{R}^{r\times 256},\ r = 128
$$
$$
\mathbf{f} = W_o\big[\mathbf{f}_1 \,\|\, V\mathbf{f}_2\big]
$$

Parameters: $4\!\times\!256\!\times\!128\ (U) + 128\!\times\!256\ (V) + $ gate + $W_o$
$\approx 0.29$ M — a **$7.5\times$ reduction** on the largest component, with a strictly richer
function class (it has the multiplicative term the current design lacks).

### FU-2 · Replace the softmax gate and restore the confidence signal *(fixes M-2)*

$$
\boldsymbol\nu = \big(\log\|\mathbf b_A\|_2,\ \log\|\mathbf b_B\|_2,\ \log\|\mathbf b_C\|_2,\ \log\|\mathbf b_D\|_2\big) \in \mathbb{R}^4
$$

computed **before** normalisation and fed into the gate MLP. This returns to the gate the scalar
that per-sample LayerNorm removed (§2.3.2). Simultaneously:

- swap $\mathrm{softmax} \to \sigma$ (independent gates, so conjunctions are expressible);
- swap the per-sample `LayerNorm` for `BatchNorm1d` (or RMSNorm with a running scale), so the
  normaliser is a *dataset* statistic and a low-SNR sample is not amplified to unit scale.

### FU-3 · Make the block a valid pre-LN transformer *(fixes N-9)*

If FU-1(a) is chosen, apply LN before **all three** sublayers:

$$
L \leftarrow L + \mathrm{MHA}\big(\mathrm{LN}_1(L),\,\mathrm{LN}_2(T),\,\mathrm{LN}_2(T)\big),\quad
L \leftarrow L + \mathrm{MHA}\big(\mathrm{LN}_3(L)\big),\quad
L \leftarrow L + \mathrm{FF}\big(\mathrm{LN}_4(L)\big)
$$

This also flattens the loss surface the Stage-3 SAM is trying to measure — a prerequisite for
§3.6 to be interpretable.

### FU-4 · A fifth modality token: morphology *(uses P-4)*

$\mathbf{s}_n \in \mathbb{R}^8 \to \mathrm{MLP}_{8\to64\to256} \to T_5$. Cost $\approx 17$ k. The
attention/gate cost of a fifth modality is one extra token, as the documents already note. This is
the cheapest expected gain in the entire plan (§2.2.10).

### FU-5 · Collapse `output_proj` and `EmbedNet` *(fixes N-10)*

Keep one post-fusion residual MLP block (pre-LN, $256\to512\to256$, residual, final LN). Delete the
other. Saves $\approx 0.15$ M with no functional loss.

---

## 3.5 Heads and margins (**HD**-series)

### HD-1 · Unify the Stage-1 and Stage-2 heads *(fixes 1.1(i), §2.4.6)*

Delete `linear_head`, `use_arcface(flag)` and `init_from_linear`. Use **one** head for all three
stages:

$$
\text{logits}_{i,c} = s\Big(\mathbb{1}[c = y_i]\cos(\theta_{i,y_i} + m_i) + \mathbb{1}[c\ne y_i]\cos\theta_{i,c}\Big),
\qquad m_i = 0 \text{ in Stage 1.}
$$

At $m = 0$ this is a **cosine (NormFace) classifier**, which §2.3.8 shows is nearly what the linear
head already is, because `EmbedNet`'s terminal LayerNorm pins $\|\mathbf{e}\| \approx 16$. The
curriculum then differs across stages in exactly three controlled dimensions — margin, sampler,
optimiser — and the six-way discontinuity of §2.4.6 disappears.

Checkpoint-schema note: this changes the 14 pinned top-level attribute names (`linear_head` is
removed), which `tests/regression/test_state_dict_compatibility.py` enforces. Handle it with an
explicit, versioned migration (a `schema_version` field in the bundle and a `remap_state_dict`
shim) rather than by keeping a dead module alive.

### HD-2 · Fix sub-centre assignment *(fixes M-8)*

**(i) Soft-to-hard assignment.** Replace $\max_k$ with a temperature-annealed log-sum-exp:

$$
\cos\theta_{i,c} = \tau_k\,\log\sum_{k=1}^{K}\exp\!\Big(\frac{\hat{\mathbf e}_i^{\top}\hat{\mathbf W}_{c,k}}{\tau_k}\Big),
\qquad \tau_k: 0.20 \to 0.02
$$

At $\tau_k \to 0$ this recovers the current $\max$ exactly; at $\tau_k = 0.20$ every sub-centre
receives gradient, so none can die before it has seen data.

**(ii) Usage balance.** With $\pi_{c,k}$ the empirical win-rate of sub-centre $k$ within class $c$
over a batch,

$$
\mathcal{L}_{\text{bal}} = \sum_c \mathrm{KL}\Big(\pi_{c,\cdot} \,\Big\|\, \tfrac1K\mathbf{1}\Big),
\qquad \text{weight } 10^{-2}
$$

— the standard mixture-of-experts load-balancing term, which is exactly the mechanism missing here.

**(iii) Data-driven initialisation.** Replace $\hat{\mathbf w}^{\text{lin}}_c + 0.01k\boldsymbol\epsilon$
(random $9$–$18°$ decoys, §2.4.3) with spherical $k$-means on the Stage-1 embeddings of class $c$
— which `forward(..., return_embed=True)` already produces:

$$
\{\mathbf{W}_{c,k}\}_{k=1}^{K} \leftarrow \text{$k$-means}\Big(\{\hat{\mathbf e}_i : y_i = c\},\ K\Big)
$$

**(iv) Make $K$ data-dependent.** With $\sim67$ samples/class, $K = 3$ is unjustified for
unimodal classes. Set $K_c \in \{1,3\}$ by the silhouette or the eigengap of the class's embedding
Gram matrix; use $K_c = 1$ where the class is unimodal. Report the histogram of $K_c$ — it is a
publishable observation about the dataset.

### HD-3 · Replace the $F_1$-driven margin with a signed, precision/recall-aware rule *(fixes M-6, M-7)*

$F_1$ conflates the two failure modes that require **opposite** margin responses (§2.4.1), so it
must be replaced by a signal that separates them. Precision and recall do:

$$
\boxed{\;M(c) \;=\; \operatorname{clip}\Big(m_{\text{base}} + m_\Delta\big(R_c - P_c\big),\; 0.20,\; 0.50\Big)\;}
$$

with $P_c, R_c$ measured on the **calibration** split (P-5), $m_{\text{base}} = 0.35$,
$m_\Delta = 0.20$.

| Class state | $R_c - P_c$ | $M(c)$ | Geometric effect | Current rule |
|---|---:|---:|---|---|
| **Over-claims** — absorbs other classes' samples ($P_c$ low, $R_c$ high) | $> 0$ | raised | region shrinks — **correct** | raised (accidentally correct) |
| Balanced | $\approx 0$ | $m_{\text{base}}$ | unchanged | raised if $F_1$ is low |
| **Under-claims** — its own samples go elsewhere ($R_c$ low) | $< 0$ | lowered | region expands — **correct** | raised — **wrong direction** |

The sign is the whole point and it is easy to get backwards: an *additive angular margin shrinks
the margined class's decision region*, because it requires $\cos(\theta_y + m)$ to beat every
unmargined $\cos\theta_{c \ne y}$. A margin is therefore a **penalty on the class it is applied
to**, not a boost. Any rule that raises $m$ for a low-recall class is pushing that class further
out of contention — which is exactly the feedback loop of §2.4.1.

**Additionally, target the margin at the confusion, not at the class.** The five hardest classes
almost certainly form confusable *pairs*. An inter-class (pairwise) margin is the right instrument:

$$
\text{logit}_{i,c} = s\big(\cos\theta_{i,c} - \mathbb{1}[c \ne y_i]\cdot \delta\, \Omega_{y_i, c}\big)
$$

with $\Omega \in \mathbb{R}^{90\times90}$ the row-normalised confusion matrix from the calibration
split. This pushes class $y$ away *specifically from the classes it is confused with*, instead of
away from all 89 others uniformly — which is what an additive per-class margin does, and why it
shrinks the region.

**And cap $\theta + m$ below $\pi/2$** to stay on the monotone-gradient side of §2.4.2:
$m_i \leftarrow \min\big(m_i,\ \max(0,\ \tfrac{\pi}{2} - \theta_{i,y_i})\big)$ — a one-line clamp
that removes the gradient-decay regime entirely and subsumes the easy-margin guard.

### HD-4 · Loosen the cosine clamp *(fixes N-4)*

$$
c_i \leftarrow \operatorname{clamp}\big(\cdot,\ -1 + 10^{-3},\ 1 - 10^{-3}\big)
\quad\Longrightarrow\quad
\Big|\tfrac{\partial}{\partial c}\sqrt{1-c^2}\Big| \le 22.4 \;\;(\text{was } 707)
$$

Angular resolution lost: $\arccos(1-10^{-3}) = 2.6°$ vs $\arccos(1-10^{-6}) = 0.08°$ — irrelevant
next to margins of $20$–$26°$.

---

## 3.6 Curriculum, losses and the Stage-3 rewrite (**OP**-series)

### OP-1 · Fix focal × label smoothing *(fixes M-9)*

```python
# losses/focal.py
logp   = F.log_softmax(z, dim=-1)
ell    = -(q * logp).sum(-1)                 # smoothed CE, unchanged
p_true = logp.gather(-1, y[:, None]).squeeze(-1).exp()   # UNSMOOTHED p_y   <-- the fix
loss   = ((1.0 - p_true) ** gamma * ell).mean()
```

One line. Restores the modulator's range from $[0.396, 1]$ to $[0,1]$ at $\varepsilon = 0.10$, and
makes $p_t$ mean what its name says. Add a unit test asserting
$\big(1-p_t\big)^\gamma \to 0$ as $p_y \to 1$ for every $\varepsilon$ — the property that currently
fails.

### OP-2 · Rebalance auxiliary supervision by construction, not by a constant *(fixes the $2\times$ hack)*

Once BR-1/BR-2/BR-4 give A, B and D non-duplicated inputs, the structural cause of A/B gradient
collapse is gone and $\omega_A = \omega_B = 2$ has no justification. Replace the fixed vector with
**gradient-norm balancing** (GradNorm-style), which makes the balance a measured quantity rather
than a constant:

$$
\omega_b^{(t+1)} = \omega_b^{(t)}\cdot\Big(\frac{\bar g}{g_b}\Big)^{\alpha},
\qquad g_b = \big\|\nabla_{\theta_b}\,\omega_b\mathcal{L}_b\big\|_2,\quad \bar g = \tfrac14\textstyle\sum_b g_b,\quad \alpha = 0.5
$$

$g_b$ is **already computed and logged** by `branch_grad_norm_tensors` under
`grad_norm/branch_{a,b,c,d}` — the telemetry for this fix exists; only the feedback loop is
missing. Update $\omega$ once per epoch, log it, and keep the existing decay schedule
$w_{\text{aux}}(e): 0.65 \to 0.25$ as an outer multiplier.

### OP-3 · Per-group gradient clipping *(fixes M-10)*

Replace the single global clip with three:

```python
clip_grad_norm_(head_params,     max_norm=1.0)
clip_grad_norm_(fusion_params,   max_norm=1.0)
clip_grad_norm_(backbone_params, max_norm=1.0)
```

The parameter-group machinery already exists (`optim/param_groups.py` splits head/backbone for
Stage 2). This decouples the $s{=}48$-amplified head from the backbone so a saturated batch no
longer divides the whole model's effective LR (§2.5.8). Keep logging pre-clip norms.

### OP-4 · **Stage 3 rewrite** *(fixes C-6, C-7)*

The stage is repairable without changing its concept. Seven changes, ordered by expected effect:

| # | Change | Fixes | Rationale |
|---:|---|---|---|
| **1** | **Identical objective on both SAM steps.** Use $\mathcal{L}_A = \mathcal{L}_D = \mathcal{L}_{\text{focal}} + 0.02\mathcal{L}_{\text{sc}} + 0.10\mathcal{L}_{\text{aux}}$ | C-6 | Restores $\hat g_A = \hat g_D$, so the correction term is $\rho H\hat g$, the actual sharpness penalty (§2.5.1) |
| **2** | **Keep per-class margins.** Pass `arc_m=None`; anneal the *whole vector* multiplicatively: $m_c(e) = M(c)\cdot\kappa(e)$, $\kappa: 1.0 \to 0.85$ | C-7a | Preserves Stage 2's calibration while still relaxing late (§2.5.2) |
| **3** | **Freeze the objective across a cycle.** Update $\kappa$ only at cycle boundaries, not per epoch, so all 47 steps between two snapshots optimise one function | C-7b | Restores the stationarity SWA requires (§2.5.3) |
| **4** | **True greedy SWA — evaluate the average.** Accept $\theta^{(n)}$ iff $F_1\big(\tfrac{n-1}{n}\bar\theta + \tfrac1n\theta^{(n)}\big) > F_1(\bar\theta)$ | C-7c | Tests the object being built, not the candidate (§2.5.4) |
| **5** | **Discard the first $\lceil 1/(1-\beta_2)/N_{\text{steps}}\rceil = 3$ cycles** from SWA, or warm-start Adam's moments from Stage 2's optimiser state | C-7d | Keeps the transient out of the average (§2.5.5) |
| **6** | **BN pass: dropout off, natural prior.** Force every `nn.Dropout` (and MHA dropout) to eval during `update_bn_stats`; run it on a **shuffled, unweighted** loader; assert device-independence in a test | C-7e | Removes the variance inflation, the prior shift and the MPS/CUDA divergence (§2.5.6) |
| **7** | **Keep the EMA shadow.** Maintain the EMA *and* the SWA average; score both; write the better as `ema`, record which in the sidecar. Log `val/f1_ema` like Stages 1–2 | C-7f, N-11 | Stops discarding the scheme that worked (§2.5.7) |

Plus two supporting fixes: wire `model.branch_drop_prob` so Stage 3's `=0` actually disables branch
masking (N-1d), and either apply the `proto` term or delete the argument (N-1e).

### OP-5 · Use ASAM, or restrict $\rho$ to where it means something *(deepens C-6)*

SAM's $\rho$-ball is defined in raw parameter space and is **not scale-invariant**, while the
ArcFace head is scale-invariant by construction ($\mathcal{L}(cW) = \mathcal{L}(W)$). The
perturbation budget is allocated proportionally to gradient magnitude:

$$
\hat\varepsilon = \rho\,\frac{\mathbf g}{\|\mathbf g\|_2}
\quad\Longrightarrow\quad
\|\hat\varepsilon_{\text{group}}\| = \rho\,\frac{\|\mathbf g_{\text{group}}\|}{\|\mathbf g\|}
$$

Because the head's gradients are amplified by $s = 48$ (§2.5.8), the head consumes a
disproportionate share of a budget that, spread isotropically over $7.88$ M parameters, would be
only $\rho/\sqrt{n} = 0.015/2807 = 5.3\times10^{-6}$ per parameter. **SAM is currently flattening
the classifier, not the representation.** Use **ASAM** (element-wise normalisation by $|\theta|$),
which is scale-invariant and therefore well-defined on a normalised head:

$$
\hat\varepsilon = \rho\,\frac{T_\theta^2\,\mathbf g}{\|T_\theta\,\mathbf g\|_2},
\qquad T_\theta = \operatorname{diag}\big(|\theta|\big)
$$

Alternatively, restrict SAM to the backbone and exclude the head from the perturbation entirely —
a two-line change with the same intent.

### OP-6 · Angular-compatible regularisation to replace the CutMix gap

The documents note that mixup is mutually exclusive with ArcFace (soft targets vs. a single-label
angular margin) and that no CutMix operator exists. Four alternatives that **are** compatible with
an angular objective, none of which requires interpolated labels:

| Method | Operation | Why it is angular-compatible |
|---|---|---|
| **Manifold mixup on the *embedding*, with mixed ArcFace** | $\tilde{\mathbf e} = \lambda\hat{\mathbf e}_i + (1-\lambda)\hat{\mathbf e}_j$, renormalised; loss $= \lambda\,\mathcal{L}_{\text{arc}}(\tilde{\mathbf e}, y_i) + (1-\lambda)\mathcal{L}_{\text{arc}}(\tilde{\mathbf e}, y_j)$ with **each term using its own class's margin** | The margin is still indexed by a single label *within each term*; the raise-`ValueError` guard is over-broad |
| **Spectral CutMix** (`_band_cutout`'s missing sibling) | swap a contiguous $\lambda$-window between two seeds of the **same** class | Label is unchanged, so no soft target is needed at all |
| **Spatial CutMix, same class** | paste a $k\times k$ region from another seed of the same class | Same |
| **Virtual-prototype / sub-centre noise** | perturb $\hat{\mathbf W}_{c,k}$ tangentially by $\sigma$ during training | A pure metric-space regulariser; the standard companion to sub-centre ArcFace |

Same-class CutMix in particular is the natural fit here: it produces genuinely novel intra-class
variation (the thing 67 samples/class lack) without touching the label, and it composes with the
$16\times8$ balanced batch, which guarantees 8 same-class partners per anchor.

### OP-7 · Restore ProtoNCE to Stage 3 or delete it *(N-1e)*

Stage 3's compound objective drops $\lambda_{\text{pt}}$ from $0.18$ to $0$ while still constructing
the module and logging `sched/proto_weight`. Either apply it (recommended — it is the cheapest term
that maintains the embedding geometry Stage 2 built, at $O(B|\mathcal{C}_{\text{batch}}|)$) or
remove the argument and the telemetry key. Silently inert configuration is worse than either.

---

## 3.7 TTA rewrite (**TT**-series)

### TT-1 · Mask-preserving spectral views *(fixes C-8)*

```python
# engine/tta.py — replace the spectral view
m   = mask if mask is not None else (x.abs().sum(1, keepdim=True) > 1e-5).float()   # (B,1,H,W)
n   = m.sum((-2, -1), keepdim=True).clamp_min(1.0)
mu  = (x * m).sum((-2, -1), keepdim=True) / n          # FOREGROUND mean, per band
xs  = (mu + (x - mu) * s) * m                          # re-mask: background stays EXACTLY 0
```

Two corrections: the mean is over the **foreground** (so $\mu_c = m_c$, not $f m_c$), and the
result is **re-masked** (so the invariant survives). This makes the spectral view exactly the
test-time analogue of the train-time `multiplicative` augmentation, which is already written as
$x \odot (1 + 0.05s\epsilon_c)\odot m$ — mask multiplication included.

### TT-2 · Draw TT views from the training augmentation family

Replace the contrast transform, which appears nowhere in training, with the two transforms the
model was actually trained to be invariant to:

$$
\mathcal{T}_{\text{spectral}} = \Big\{\text{$\lambda$-warp by }\alpha \in \{0.98,\,1.00,\,1.02\}\Big\}
\;\cup\;
\Big\{\text{per-band gain } x\odot(1 + \varsigma\epsilon_c)\odot m,\ \varsigma \in \{0.02\}\Big\}
$$

matching `_spectral_warp` and `_multiplicative` in `data/datasets.py`. Under P-2's SNV the gain
views become near-no-ops, which is itself informative — it tells you the normalisation is working.

### TT-3 · Force fp32 in TTA *(fixes §2.6.4)*

`engine/tta.py` must wrap its forward passes in `autocast(enabled=False)` exactly as
`engine/evaluate.py` does, for the same stated reason. Until it does, the $+0.0163$ TTA gain is
confounded with a precision change.

### TT-4 · Report the ablation, not just the ensemble

`final_eval` should emit macro-F1 for $\{$1 view, 4 spatial, 8 spatial, 8 spatial + 4 spectral$\}$.
This is nearly free (the logits are already computed per view) and it turns the TTA section from a
single number into a curve — which is what a reviewer will ask for, and what settles F-6.

---

## 3.8 Revised parameter budget

| Component | Current | Proposed | Δ | Driver |
|---|---:|---:|---:|---|
| `se` (MaskedSpectralECA) | 6 | 6 | — | unchanged |
| `branch_a` — profile → derivative profile | 592,753 | ~600,000 | +7 k | FE-1b derivative channels |
| `branch_b` — moments → index bank + morphology | 686,424 | **~95,000** | **−591 k** | BR-1 (rank-2 collapse removed) |
| `branch_c` — 2-D CNN → 3-D spectral–spatial stem | 1,694,158 | **~2,300,000** | **+606 k** | BR-3 (the only branch with unique input) |
| `branch_d` — SpecFormer → $\lambda$-aware SpecFormer | 2,180,866 | **~1,250,000** | **−931 k** | BR-4 ($d_{\text{model}}$ 256→192, $\lambda$-attention needs less capacity) |
| `cross_interaction` — Perceiver → gated bilinear | 2,190,916 | **~550,000** | **−1,641 k** | FU-1(b), FU-5 |
| `morphology_embed` (new) | — | ~17,000 | +17 k | FU-4 |
| `aux_head_{a..d}` | 178,024 | 178,024 | — | unchanged |
| `embed_net` (merged with `output_proj`) | 263,936 | ~264,000 | — | FU-5 |
| `linear_head` | 23,130 | **0** | −23 k | HD-1 (unified head) |
| `arcface_head` | 69,120 | 69,120 | — | unchanged shape |
| **Total** | **7,879,333** | **≈ 5,323,000** | **−2.56 M (−32 %)** | |

The reallocation is the point: **−2.56 M parameters removed from modules operating on 640 numbers
or on 4 tokens, +0.6 M added to the only module that sees the full cube.** Params-per-input-scalar
for Branch C rises from $10.3$ to $\approx 14$, and the A∪B∪D share of the budget falls from
$43.9\,\%$ to $\approx 36\,\%$ of a much smaller model, on genuinely disjoint inputs.

---

# 4 · ACTIONABLE IMPLEMENTATION MATRIX

## 4.1 Phase 0 — measure before you modify (run these first, in this order)

None of these changes a line of the model. Together they cost a few GPU-hours and they determine
which of §3's proposals are worth the effort. **Do not begin Phase 1 before Phase 0 reports.**

| ID | Action | File / command | Answers | Cost |
|---|---|---|---|---|
| **0-A** | Run `final_eval` on **all three** checkpoints, no-TTA and TTA | `engine/stages/final_eval.py` — loop over `best_stage{1,2,3}.pth` | Is Stage 3 actually worse *on test*? (§2.1.3) | 6 forward passes over 1,294 patches |
| **0-B** | `tta_spatial=8 tta_spectral=0` on the selected checkpoint | one Hydra override | **F-6**: do the 4 spectral views help or hurt? (§2.6.3) | 1 eval |
| **0-C** | Bootstrap CIs on the val and test macro-F1 from the three saved `.npy` prediction arrays | new `scripts/bootstrap_ci.py` | Is the Stage-1/Stage-3 gap outside noise? (§2.1.3) | seconds — arrays are on disk |
| **0-D** | SVD of the $(9,40)$ statistics tensor over the train split | `np.linalg.svd`; report $\sigma_3/\sigma_1$ | **F-2**: confirm the rank-2 collapse (§2.2.5) | minutes |
| **0-E** | Print $\max_{n\ne n'}\cos(L_n, L_{n'})$ from `cross_interaction.latents` in each checkpoint | 3 lines | **F-4**: confirm latent collapse (§2.3.1) | seconds |
| **0-F** | Print $\Delta\lambda_i$ histogram from `wavelengths_spa_40b.csv`; report $\max/\min$ | 3 lines | Magnitude of C-5 (§2.2.6) | seconds |
| **0-G** | Per-class precision *and* recall for the 5 hardest classes from `test_preds_TTA.npy` / `test_targets.npy` | `sklearn.classification_report` | **F-5**: is the margin rule's sign wrong for them? (§2.4.1) | seconds |
| **0-H** | Count distinct `(session, scan)` groups; varieties-per-scan histogram | `data/prep/` walk | Is a grouped split feasible? (§2.1.1) | minutes |
| **0-I** | Count dead sub-centres: for each $(c,k)$, the win-rate of $\arg\max_k$ over the train split | 10 lines using `return_embed=True` | Confirm M-8 (§2.4.3) | 1 forward pass |
| **0-J** | Print `cross_interaction` head count and SpecFormer dropout from a loaded checkpoint's module | 2 lines | Confirm N-1a/N-1b empirically | seconds |

## 4.2 Full implementation matrix

Legend — **Risk**: L = local, no schema change; M = touches training dynamics; H = changes the
checkpoint schema or the reported protocol. **Δ** is on **P-cur** unless marked *(P-fix)*.

### Tier 1 — correctness fixes (do these regardless of anything else)

| ID | Target file | Change | Δ macro-F1 | Validation criterion | Risk |
|---|---|---|---:|---|---|
| **T1-1** | `engine/tta.py` | TT-1: foreground-only mean + re-mask (§3.7) | **+0.002 … +0.006** | Assert `(x_s * (1-m)).abs().max() == 0` for all 4 scales; foreground fraction unchanged from the un-augmented patch | L |
| **T1-2** | `engine/tta.py` | TT-3: wrap forwards in `autocast(enabled=False)` | ±0.001 (removes a confound) | TTA and no-TTA numbers now differ only by ensembling | L |
| **T1-3** | `losses/focal.py` | OP-1: modulate on unsmoothed $p_y$ (§3.6) | **+0.001 … +0.003** | New unit test: $(1-p_t)^\gamma \to 0$ as $p_y \to 1$ for $\varepsilon \in \{0, 0.04, 0.10\}$ | L |
| **T1-4** | `engine/stages/stage3_sam_swa.py` | OP-4.1: identical objective on ascent and descent (§3.6) | **+0.003 … +0.008** | Log $\cos(\hat g_A, \hat g_D)$; must be exactly $1.0$ after the fix | M |
| **T1-5** | `engine/stages/stage3_sam_swa.py` | OP-4.6: dropout forced to eval during `update_bn_stats`; shuffled unweighted loader | **+0.001 … +0.004** | New test: BN buffers after the pass are identical on CPU and the accelerator to $10^{-5}$ (kills N-12) | M |
| **T1-6** | `models/quadnet.py` *(inferred)* | N-1d: wire `model.branch_drop_prob` into the forward vector | +0.001 … +0.003 | Stage 3 with `branch_drop_prob=0` must produce a deterministic train-mode forward | L |
| **T1-7** | `models/heads.py` *(inferred)* | HD-4: clamp at $1-10^{-3}$ (§3.5) | ≈ 0 (stability) | Max observed head grad-norm falls by $\ge 10\times$ | L |
| **T1-8** | `engine/stages/final_eval.py` | Record which of live/EMA won selection; evaluate that one (§2.1.4) | ±0.003 (removes a mismatch) | Sidecar gains `best_source ∈ {live, ema}`; regression test pins it | M |
| **T1-9** | `engine/stages/stage3_sam_swa.py` | OP-4.7: keep and score the EMA shadow; write the better of EMA/SWA | **+0.002 … +0.006** | `val/f1_ema` appears in Stage-3 telemetry like Stages 1–2 | M |
| **T1-10** | `engine/stages/stage3_sam_swa.py` | OP-7: apply ProtoNCE or delete the argument | +0.000 … +0.002 | `sched/proto_weight` reflects a term that exists | L |

**Tier 1 subtotal: +0.010 to +0.032 macro-F1, no architecture change, ~1 day of work.**
If 0-A shows Stage 3 is not actually worse on test, T1-4/5/9 alone plausibly make Stage 3 the
selected checkpoint — which restores the metric-learning contribution to the paper.

### Tier 2 — optimisation and curriculum

| ID | Target file | Change | Δ macro-F1 | Validation criterion | Risk |
|---|---|---|---:|---|---|
| **T2-1** | `engine/stages/stage3_sam_swa.py` | OP-4.2/4.3: per-class margins retained, annealed multiplicatively, frozen within a cycle | **+0.003 … +0.008** | `arcface_head.margins` is non-constant throughout Stage 3; objective constant across each 8-epoch cycle | M |
| **T2-2** | `engine/stages/stage3_sam_swa.py` | OP-4.4: greedy SWA evaluates the **candidate average** | +0.002 … +0.005 | `swa/n_rejected > 0` on a normal run (currently provably 0) | M |
| **T2-3** | `engine/stages/stage3_sam_swa.py` | OP-4.5: discard the first 3 cycles, or warm-start Adam moments from Stage 2 | +0.001 … +0.004 | First accepted snapshot index $\ge 4$, or optimiser state loaded | M |
| **T2-4** | `optim/sam.py` *(inferred)* | OP-5: ASAM ($T_\theta = \mathrm{diag}|\theta|$), or exclude the head from the perturbation | +0.002 … +0.006 | Perturbation mass on `arcface_head` falls below its parameter share | M |
| **T2-5** | `optim/param_groups.py`, both epoch loops | OP-3: per-group gradient clipping | +0.001 … +0.004 | Backbone pre/post-clip norm ratio decouples from the head's | M |
| **T2-6** | `losses/auxiliary.py` | OP-2: GradNorm-style $\omega_b$ from the already-logged `grad_norm/branch_*` | +0.002 … +0.005 | $\omega$ becomes a logged time series; A/B norms converge toward C/D without the hardcoded $2\times$ | M |
| **T2-7** | `data/datasets.py`, `losses/` | OP-6: same-class spectral and spatial CutMix | **+0.004 … +0.012** | New augmentation is label-preserving, so the ArcFace `ValueError` guard is untouched | M |
| **T2-8** | `models/heads.py`, `engine/stages/stage2_arcface.py` | HD-3: signed $R_c - P_c$ margin rule + pairwise confusion margin | **+0.004 … +0.010** (macro-F1 specifically) | Per-class recall of the 5 hardest classes rises; $M(c)$ becomes non-monotone in $F_1$ | M |
| **T2-9** | `models/heads.py` | HD-2: soft-to-hard sub-centre assignment + balance term + $k$-means init | +0.003 … +0.008 | Dead-sub-centre count (0-I) drops to 0; $\pi_{c,k}$ entropy $> 0.9\log K$ | M |
| **T2-10** | `models/quadnet.py`, `configs/` | HD-1: unified cosine head across all three stages | **+0.005 … +0.015** | Stage 2's epoch-1 val F1 $\ge$ Stage 1's final, i.e. the transition costs nothing | **H** |

**Tier 2 subtotal: +0.027 to +0.077.** T2-10 is the single largest item and the one that recovers
the metric-learning contribution; it requires the schema migration of §3.5.1.

### Tier 3 — architectural redesign

| ID | Target file | Change | Δ macro-F1 | Validation criterion | Risk |
|---|---|---|---:|---|---|
| **T3-1** | `models/branches/stats.py` *(inferred)* | BR-1: replace 9 moments with a learned NDI bank + continuum depths + morphology | **+0.005 … +0.015**; **−591 k params** | 0-D repeated on the new input: $\sigma_3/\sigma_1 > 0.3$; `influence/branch_b` rises | H |
| **T3-2** | `models/branches/spatial.py` *(inferred)* | BR-3: 3-D spectral–spatial stem (§3.3) | **+0.010 … +0.030** | `influence/branch_c` rises materially; branch-C aux-head accuracy rises | H |
| **T3-3** | `models/branches/specformer.py` *(inferred)* | BR-4: $\lambda$-derived token embeddings, relative-$\lambda$ attention bias, $\lambda$-uniform tokenisation | **+0.005 … +0.015**; **−931 k params** | Branch D transfers across band counts without retraining the positional table (settles F-3) | H |
| **T3-4** | `models/fusion.py` *(inferred)* | FU-1(b) + FU-2 + FU-4: gated low-rank bilinear fusion, sigmoid gate, confidence vector, morphology token | **+0.005 … +0.015**; **−1,641 k params** | Gate entropy no longer pinned; $\sum_m g_m \ne 1$; 0-E collapse metric becomes moot | H |
| **T3-5** | `models/front_end.py` *(inferred)* | FE-1: continuous $\lambda$-kernels and/or explicit derivative channels | **+0.008 … +0.020** | A synthetic test with two bands $2.4$ nm and $100$ nm apart yields correctly-scaled derivatives | H |
| **T3-6** | `models/branches/profile.py` *(inferred)* | BR-2: derivative input, mass-weighted cell pooling, $8\times8$ grid | +0.003 … +0.010 | Branch A's input is no longer byte-identical to D's; `grad_norm/branch_a` rises without the $2\times$ weight | M |
| **T3-7** | all masked modules | FE-2: pass the persisted mask explicitly | +0.001 … +0.003 | Masked statistics become invariant to any global brightness transform | M |

**Tier 3 subtotal: +0.037 to +0.108, and −2.56 M parameters.**

### Tier 4 — protocol (changes what the number *means*)

| ID | Target file | Change | Δ macro-F1 | Validation criterion | Risk |
|---|---|---|---:|---|---|
| **T4-1** | `data/prep/patch_extraction.py`, `data/loaders.py` | P-1: persist `scan_id`; `StratifiedGroupKFold` | **−0.05 … −0.20** *(this is the correct direction)* | Regression test: no `scan_id` appears in more than one split | **H** |
| **T4-2** | `data/prep/{download,patch_extraction}.py` | P-2: white-reference division, else per-pixel SNV | **+0.02 … +0.08** *(P-fix)* | Session-held-out accuracy gap narrows; the leak channel of §2.1.1 closes | **H** |
| **T4-3** | `data/prep/patch_extraction.py` | P-3: resize the mask, re-mask, divide by $\alpha_p$ | +0.002 … +0.008 | Foreground pixel count matches the resized mask exactly; no partial-coverage pixels remain | M |
| **T4-4** | `data/prep/segmentation.py` | P-4: persist 8 morphometrics | +0.002 … +0.006 *(with FU-4)* | `morphology.npy` exists; standardised on train only | L |
| **T4-5** | `data/loaders.py` | P-5: carve a `calib` split for margins/CDWS/oversampling | +0.000 … +0.005; **removes the val→test gap** | 270 fitted parameters no longer touch the selection split; val−test gap $< 0.005$ | **H** |
| **T4-6** | `data/prep/band_selection.py` | Re-run selection with the deployed estimator (F-3); publish the full $k$-curve to $k = 256$ | unknown, possibly large | The recorded curve extends past the chosen $k$; the elbow is demonstrable, not asserted | M |

## 4.3 Regression tests to add

The existing gate suite is unusually good — golden forward pass, per-tensor SHA-256, all 600
scheduler values, config round-trip. It has **no test for any of the defects above**, because every
one of them is a property of the *design*, not of a numerical value. Add:

| Test | Asserts | Catches |
|---|---|---|
| `test_tta_preserves_background_mask` | $\forall s$: $\|x^{(s)}\odot(1-m)\|_\infty = 0$; foreground fraction unchanged | C-8 |
| `test_sam_ascent_descent_objectives_identical` | $\cos(\hat g_A, \hat g_D) = 1$ within $10^{-6}$ | C-6 |
| `test_focal_modulator_reaches_zero` | $(1-p_t)^\gamma < 10^{-3}$ at $p_y = 1-10^{-6}$, for every $\varepsilon$ used | M-9 |
| `test_splits_are_group_disjoint` | $\text{scan}(\text{train}) \cap \text{scan}(\text{val}\cup\text{test}) = \emptyset$ | C-1 |
| `test_branch_inputs_are_distinct` | hash of each branch's input tensor differs pairwise for a fixed batch | C-2 |
| `test_fusion_latents_are_diverse` | $\max_{n\ne n'}\cos(L_n, L_{n'}) < 0.9$ after training | M-1 |
| `test_no_dead_subcentres` | every $(c,k)$ wins the max for $\ge 1$ training sample | M-8 |
| `test_bn_stats_device_independent` | BN buffers after `update_bn_stats` match on CPU and accelerator | N-12 |
| `test_config_keys_are_wired` | every key in `configs/model/*.yaml` is read at least once during a traced forward | N-1a…f |
| `test_margin_rule_sign` | $M(c)$ decreases as $R_c$ falls at fixed $P_c$ | M-6 |

`test_config_keys_are_wired` is the highest-leverage of these: it would have caught all five dead
paths of §2.7 automatically, and the config round-trip gate that *does* exist explicitly does not
check wiring — only that all 81 keys "have a home."

## 4.4 Sequencing

```
Phase 0  ─ 0-A…0-J  (measure)                          ── 1 day, no code changes
   │
   ├─► Tier 1  T1-1 … T1-10   (correctness)            ── 1–2 days,  +0.010…0.032
   │      │
   │      └─► Tier 2  T2-1 … T2-9  (optimisation)      ── 1 week,    +0.022…0.062
   │             │
   │             └─► T2-10 unified head  [schema bump] ── 2 days,    +0.005…0.015
   │
   ├─► T4-4 morphometrics ──┐
   ├─► T4-3 resize order  ──┤
   ├─► T4-1 grouped split ──┼─► RE-BASELINE everything on P-fix  ◄── do this before Tier 3
   ├─► T4-2 SNV / white   ──┤
   └─► T4-5 calib split   ──┘
                             │
                             └─► Tier 3  T3-1 … T3-7  (redesign) ── 3–4 weeks, +0.037…0.108
```

Two ordering constraints are load-bearing:

1. **Tier 4 before Tier 3.** Redesigning branches against a leaky split optimises for the leak.
   Every architectural decision in §3.3 is a bet about which signal is real; measuring those bets
   on P-cur measures the wrong thing.
2. **Phase 0 before everything.** If 0-B shows the spectral TTA views *help* despite being
   off-manifold, that is a genuinely interesting result about robustness and C-8's remedy changes
   from "fix the transform" to "add the transform to training." If 0-A shows Stage 3 beats Stage 1
   on test, §1.1(ii) is confirmed and the entire Stage-3 narrative in the paper must be rewritten.

## 4.5 What to claim in the paper, after the work

| Current claim | Status | Replacement |
|---|---|---|
| "0.8933 macro-F1 with 12-view TTA" | Unsafe under C-1 | Grouped-split number, with the ungrouped number reported alongside as an explicit leakage upper bound — the *difference* is a contribution |
| "Four disjoint views of the same patch" | False for A/D (§2.2.2) | True after BR-1…BR-4; support it with a pairwise input-hash test and the `influence/branch_*` KL table |
| "Wavelength enters the network physically" | Partially true — additive PE only, and Branch D discards it | True after FE-1 and BR-4; demonstrate with the $\Delta\lambda$-scaled derivative test (T3-5) |
| "Adaptive sub-centre ArcFace" | Not in the shipped checkpoint (§1.1(i)); sub-centres partly dead (M-8) | True after HD-1/HD-2; report the $K_c$ histogram and sub-centre usage entropy |
| "Three-stage curriculum" | Stage 3 degrades; Stage 2 never recovers Stage 1 | Report the *per-stage test* table (0-A) — three rows, no cherry-picking |
| "$k^\star = 40$ by elbow" | Curve terminates at the chosen $k$ (M-14) | Publish the curve to $k = 256$ under the deployed estimator, or withdraw the elbow claim |
| "no external baseline has been executed" | Honest, and the right call | Keep the protocol table; add 2–3 HSI baselines (HybridSN, SSRN, a spectral-only 1-D CNN) run under the five stated conditions on the **grouped** split |

---

# Appendix A · Derivations

### A.1 · Rank of the masked-statistics tensor (§2.2.5)

Let a seed's foreground pixels satisfy the gain model $x_{c,p} = a_p r_c$, $a_p > 0$, $r_c > 0$,
$p = 1..n$. Write $\bar a, \mathrm{sd}(a), q_\alpha(a)$ for the moments of $\{a_p\}$.

$$
\mu_c = \tfrac1n\textstyle\sum_p a_p r_c = \bar a\,r_c
$$
$$
\delta_{c,p} = x_{c,p} - \mu_c = (a_p - \bar a)\,r_c
\;\Longrightarrow\;
\sigma_c = \Big(\tfrac1n\textstyle\sum_p (a_p-\bar a)^2\Big)^{1/2} r_c = \mathrm{sd}(a)\,r_c
$$
$$
\text{skew}_c = \frac{\tfrac1n\sum_p (a_p - \bar a)^3 r_c^3}{\mathrm{sd}(a)^3 r_c^3} = \text{skew}(a)
\qquad
\text{kurt}_c = \frac{\tfrac1n\sum_p (a_p - \bar a)^4 r_c^4}{\mathrm{sd}(a)^4 r_c^4} = \text{kurt}(a)
$$

Order statistics are equivariant under a positive scaling: $q_\alpha(\{a_p r_c\}) = q_\alpha(a)\,r_c$
for $r_c > 0$, and $\max_p a_p r_c = a_{\max} r_c$. Assembling $S \in \mathbb{R}^{9\times40}$ with
rows in the documented order:

$$
S = \begin{bmatrix} \bar a \\ \mathrm{sd}(a) \\ a_{\max} \\ 0 \\ 0 \\ q_{10} \\ q_{25} \\ q_{75} \\ q_{90}\end{bmatrix}\mathbf r^{\top}
\;+\;
\begin{bmatrix} 0 \\ 0 \\ 0 \\ \text{skew}(a) \\ \text{kurt}(a) \\ 0\\0\\0\\0\end{bmatrix}\mathbf 1^{\top}
\;\Longrightarrow\;
\operatorname{rank}(S) \le 2 . \qquad\blacksquare
$$

The model is exact when within-seed pixel variation is purely a scalar gain, and is a good
approximation whenever illumination geometry and the resize fill factor $\alpha_p$ (§2.2.8)
dominate — which is the regime a segmented, dark-corrected, non-white-referenced seed patch is in.

### A.2 · Focal modulator floor under label smoothing (§2.4.5)

$\ell = -\sum_c q_c \log p_c = H(q) + \mathrm{KL}(q\|p) \ge H(q)$, with equality iff $p = q$.
Since $p_t := e^{-\ell}$,

$$
p_t \le e^{-H(q)}, \qquad
H(q) = -(1-\varepsilon)\log(1-\varepsilon) - \varepsilon\log\frac{\varepsilon}{C-1} .
$$

At $C = 90$, $\varepsilon = 0.10$: $H = 0.9(0.10536) + 0.1(6.7912) = 0.7739$, $e^{-H} = 0.4612$,
$(1 - 0.4612)^{1.5} = 0.3955$. At $\varepsilon = 0.04$: $H = 0.3475$, $e^{-H} = 0.7064$,
$(0.2936)^{1.5} = 0.1591$. $\blacksquare$

### A.3 · SAM with mismatched objectives (§2.5.1)

$$
\theta^+ = \theta - \eta\nabla\mathcal{L}_D(\theta + \rho\hat g_A)
= \theta - \eta\Big[\nabla\mathcal{L}_D(\theta) + \rho H_D\hat g_A\Big] + O(\rho^2)
$$

Write $\hat g_A = \alpha\hat g_D + \beta u$, $u\perp\hat g_D$, $\alpha^2+\beta^2 = 1$. True SAM
would give $\rho H_D\hat g_D$. The realised correction is
$\alpha\rho H_D\hat g_D + \beta\rho H_D u$. The second term has no descent guarantee:
$\langle \nabla\mathcal{L}_D, H_D u\rangle$ has arbitrary sign, so for $\alpha$ small the update is
$-\eta\nabla\mathcal{L}_D$ plus a perturbation of magnitude $\eta\rho\|H_D u\| \le \eta\rho\lambda_{\max}(H_D)$
in an uncontrolled direction — noise scaled by the largest curvature, which is the opposite of
what SAM is for. $\blacksquare$

### A.4 · TTA background value and the two-point statistics (§2.6)

With $\mu_c = f m_c$ (all-pixel mean) and $x^{(s)} = \mu + (x-\mu)s$:

- background: $0 \mapsto \mu_c(1-s) = f m_c (1-s)$;
- all-pixel mean of $x^{(s)}$: $\mu + s(\mathbb{E}[x] - \mu) = \mu$, independent of $s$;
- the pixel population is two-point to leading order (background at $b$, foreground at $g$) with
  weights $(1-f, f)$, so
$$
\sigma = |g - b|\sqrt{f(1-f)},\qquad
\text{skew} = \frac{1-2f}{\sqrt{f(1-f)}},\qquad
\text{kurt}_{\text{raw}} = 3 + \frac{1-6f(1-f)}{f(1-f)} .
$$
  At $f = 0.284$: $\sqrt{f(1-f)} = 0.451$, skew $= 0.958$, kurt$_{\text{raw}} = 1.916$ — both
  **independent of $c$**, i.e. band-independent constants. $\blacksquare$
- percentiles: with $1-f = 0.716$ of the mass at $b$ and $b < g$ for both $s < 1$ and $s > 1$
  (since $b = f m_c(1-s)$ and $g \approx 3.4\,\mu_c$), the $10^{\text{th}}$ and $25^{\text{th}}$
  percentiles both fall in the background block, hence $p_{10} = p_{25} = b$.

### A.5 · Effective branch-drop rates (§2.2.7)

$k_b = \mathbb{1}[u_b > p_b]$ then $k \leftarrow \max(k, \text{onehot}(s))$, $s\sim\mathcal{U}\{0,1,2,3\}$.
Since $p_A = p_B = 0$, $k_A = k_B = 1$ always. For $b \in \{C, D\}$:

$$
P(\text{drop } b) = P(u_b \le p_b)\cdot P(s \ne b) = p_b \cdot \tfrac34 .
$$

$P(\text{drop }C) = 0.225$, $P(\text{drop }D) = 0.150$. The safe index therefore never protects
against total ablation (A and B guarantee that); it only randomly un-drops C or D. $\blacksquare$

---

# Appendix B · Falsifiable prediction register

| ID | Prediction | Refutes/confirms | Cost | Where |
|---|---|---|---|---|
| **F-1** | Grouped `StratifiedGroupKFold(groups=scan_id)` drops macro-F1 by 5–20 pts | C-1 | 1 full run | §2.1.1 |
| **F-2** | $\sigma_3/\sigma_1 < 0.05$ for $>90\,\%$ of seeds in the $(9,40)$ statistics tensor | C-4 | minutes | §2.2.5 |
| **F-3** | The band-count/accuracy curve does **not** plateau at $k=40$ under the deployed estimator | M-14 | 6 short runs | §2.2.11 |
| **F-4** | $\max_{n\ne n'}\cos(L_n,L_{n'}) > 0.95$ in every trained checkpoint | M-1 | seconds | §2.3.1 |
| **F-5** | $\ge 3$ of classes $\{49,52,41,51,37\}$ show $R_c \ll P_c$ | M-6 | seconds | §2.4.1 |
| **F-6** | `tta_spatial=8 tta_spectral=0` $\ge$ the 12-view result | C-8 | 1 eval | §2.6.3 |
| **F-7** | Stage 3 evaluated on **test** is within noise of Stage 1 | §1.1(ii) | 6 evals | §2.1.3 |
| **F-8** | $\ge 1$ sub-centre per class has zero win-rate over the training split | M-8 | 1 forward pass | §2.4.3 |
| **F-9** | `cos(ĝ_A, ĝ_D)` in Stage 3 is materially below 1 (predicted $< 0.8$) | C-6 | 1 epoch | §2.5.1 |
| **F-10** | BN buffers after `update_bn_stats` differ between CPU and accelerator | N-12 | minutes | §2.5.6 |

**F-6, F-7 and F-9 are the three to run today.** Between them they cost under an hour, require no
code changes beyond three print statements, and they determine whether the Stage-3 section of the
paper describes an optimisation failure or a measurement artifact.
