# Research Architecture & Training Revision

**Independent audit of `SpectralQuadNet` v4 / run `stratified_benchmark_rtx3060`**
Auditor: independent reviewer. Date: 2026-08-13.

**Evidence base actually available to this audit**

| Artifact | Status |
|---|---|
| `01`–`06` design documents + `config_rename_table.md` | Present. Written **before** any run: `01` §1.1 states "No training run has yet been recorded… `outputs/` is git-ignored and empty." |
| Console log, run `stratified_benchmark_rtx3060` | Present. Stage 1 (336 ep), Stage 2 (49 ep), Stage 3 **truncated at ep 87/120**. |
| 7 W&B panel screenshots | Present. **Stage 1 only** — see §10.1. |
| Test-set metrics, `test_preds_*.npy`, `classification_report` | **Do not exist.** `final_eval` never ran. |
| Any executed ablation | **None.** `05` §5.5 documents 21 levers; zero have been pulled. |
| External baselines | **None implemented, none run.** |

Every number in this report that describes model performance is a **validation** number, from a
**scan-leaky** split, **selected on the same split it is reported from**. There is no held-out
result of any kind in this project.

---

## 1. Executive Summary

**The engineering is unusually careful and the science is unusually weak.** The repository is
better instrumented, better tested and better documented than most research code I review. But
the run it produced cannot support any of the claims the architecture was built to make, and the
reason is not the architecture — it is the evaluation protocol and the fact that no comparison
has ever been executed.

Seven findings, ordered by how much they change the conclusions:

**F1 · The evaluation protocol measures acquisition-bundle recognition as well as variety
recognition, in an unknown proportion.** The source dataset (Zenodo 3241923) images each variety
as **two bundles of 48 kernels, each bundle a tray of one single variety**. The run used
`split_scheme=stratified`, which splits at the patch level: the log records *"180 of 180 scans
are in train and in val/test."* Every class's two — and only two — acquisition conditions are
present in training. A model that learns "this tray's residual radiometric signature ⇒ class X"
scores correctly. The 0.847 macro-F1 is a mixture of two quantities and the mixing ratio has
never been measured. **The repo already ships the fix** (`data=spa40_90class_pfix`, `grouped`);
it was not used.

**F2 · The 40-band subset was selected using the labels of the test patches.** `band_selection.py`
runs mRMR/SPA over the mean spectra of all 8,624 patches *with their labels*, before any split
exists. Feature selection outside the resampling loop is a known, quantified source of optimism
(Ambroise & McLachlan, PNAS 2002). This is genuine label leakage, independent of F1, and it
contaminates every protocol including `grouped`.

**F3 · The headline number is a maximum over ~944 correlated selections on a 1,294-sample split
that also carries every fitted parameter.** `calib_frac=0.0`, so `val` fits the per-class
margins, the CDWS weights *and* the Phase-3 oversampling weights, then selects the checkpoint,
then reports the number. With ~472 epochs × {live, EMA} selection events and an observed
epoch-to-epoch macro-F1 sd of ≈0.012, the expected upward bias of a running maximum is
**≈0.042 macro-F1** — an order of magnitude larger than everything Stages 2 and 3 produced.

**F4 · Stages 2 and 3 are not supported by the evidence.** They consumed **65% of the 18.7-hour
wall clock** and moved validation macro-F1 by **+0.005 = 6.5 of 1,294 samples**, against a ±0.020
sampling CI and the ≈0.042 selection bias above. Worse, Stage 2's best epoch is **19**, which is
*during* the global margin warm-up — the per-class signed-margin vector, the headline C3
contribution, hands over at epoch 21 and produced no improvement in the 30 epochs before early
stopping.

**F5 · The model collapsed to one branch, and the branch-dropout policy guaranteed it would.**
Leave-one-out influence moves from A:48/C:0.2 at epoch 1 to **A:6/B:3.5/C:87/D:3** by Stage 2.
Branch C is the only branch with drop probability **0.0** — the other three are dropped 15% of the
time. The fusion gate was structurally taught to rely on the always-present branch. C's dominance
is therefore *confounded* and cannot be read as "C is intrinsically best," but the practical
consequence is unambiguous: **three of four branches contribute ≈12% of the fused decision
between them, and Branch A alone costs 60% of the forward FLOPs.**

**F6 · The Stage-1 objective is dominated by heads that are thrown away at evaluation.** With
GradNorm driving the per-branch auxiliary weights into their clip ceiling of 4.0 (visible in the
`aux_weight/*` panels), the auxiliary term is **≈7.8×** the main classification term at epoch 20.
The fused head — the only path that produces an evaluation logit — is **≈11% of the gradient
signal** for the first third of training.

**F7 · The hard classes never moved, under any intervention.** Classes {49, 51, 52, 41, 70, 30,
42, 37, 38, 45} are the bottom-10 at epoch 46 of Stage 1 and are still the bottom-10 at Stage 3.
Class 49 goes 0.28 (Stage-1 end) → 0.24 (Stage-2 best) → ~0.29. Hard-class oversampling (7× cap),
CDWS (3× cap), signed per-class margins, and the pairwise confusion penalty were all aimed at
exactly these classes and all failed. **This is the strongest evidence in the run that the
residual error is representational or label-intrinsic, not an optimisation or class-balance
problem** — and therefore that the four mechanisms built to fix it should be removed rather than
tuned.

**Verdict.** The architecture is not *wrong*, it is *unfalsified and over-built*. I recommend
cutting the model from 5.19 M parameters / 4.06 GFLOP to ≈2.4 M / ≈1.4 GFLOP, collapsing three
stages into one, deleting nine loss/scheduling mechanisms, and spending the freed compute on the
two-fold scan-disjoint protocol and the ablation grid — none of which has ever been run.

**On the >95% target:** not scientifically defensible on this dataset under a bundle-disjoint
protocol, and I would not claim it. Under the leaky patch-level protocol it is plausibly
reachable — published work on this exact dataset reports 92.73–96.17% precision (Taheri et al.,
*J Ambient Intell Human Comput* 15:2883–2899, 2024) — but reaching it that way would be
reproducing the field's existing error, not correcting it. Full reasoning in §23 and Q11.

---

## 2. Current System Reconstruction

### 2.1 What is implemented vs. what has been executed

The prompt asks for this distinction explicitly. Applying its seven levels:

| Level | Components |
|---|---|
| **Merely proposed** | Every entry in `05` §5.5 (21 ablations); external baselines; the `white` radiometry path; the 256-band "86.9% TTA" figure quoted in a source comment with no artifact behind it |
| **Implemented** | All four branches, fusion, head, all losses, all schedules, DDP, TTA, `grouped` split, calibration split, `verify_elbow` |
| **Configured** | The shipped `output_v12_spa40` composition; the run's overrides (`masks_path`, `morphology_path`, `allow_tf32=True`, `amp_dtype=bf16`, `num_workers=0`, `compile=off`) |
| **Executed** | One run. Stage 1 ✓, Stage 2 ✓, Stage 3 **partial (87/120 epochs in the log)** |
| **Completed** | Stages 1 and 2. Stage 3 and `final_eval`: no evidence |
| **Measured** | Validation macro-F1 / accuracy per epoch; per-class F1; branch influence; per-branch grad norms and aux weights (Stage 1 only in W&B) |
| **Statistically supported** | **Nothing.** n=1 run, n=1 seed, n=1 split, no control arm, no test set, no ablation |

### 2.2 Data path (as executed)

```
Zenodo 3241923 (90 varieties × 2 bundles × 48 kernels, 256 bands, 385–1006 nm)
  → dark-frame subtraction (column-mean), crop to rows [0:600]
  → Otsu×0.4 over 450–700 nm, fill holes, clear border, remove <150 px
  → shape gate: 300<area<800, ecc>0.6, solidity>0.85          → 8,624 of 8,640 kernels survive
  → per-region crop, centre-pad to square, area-resize to 64×64,
    divide by fill fraction α where α>0.5 else 0               → exact-zero background invariant
  → per-pixel SNV across λ (radiometry="auto"→"snv")           → gain.npy persisted, NEVER LOADED
  → mRMR + SPA band selection on all 8,624 labelled spectra    → 40 bands  ⚠ LEAKAGE (F2)
  → patches_spa_40b.npy (8624,40,64,64) float32, 5.65 GB, mmap
  → stratified patch-level split 6,036 / 1,294 / 1,294         → ⚠ LEAKAGE (F1)
```

### 2.3 Model (measured, `03` §3.7)

| Component | Params | Share | **FLOP/sample (my estimate)** | **FLOP share** | End-state influence |
|---|---:|---:|---:|---:|---:|
| `se` MaskedSpectralECA | 6 | 0.0% | negligible | — | shared |
| Branch A · SpectralProfile | 603,089 | 11.6% | **2.44 G** | **60.1%** | **5.6%** |
| Branch B · Index Bank | 94,896 | 1.8% | 0.0002 G | 0.0% | 3.5% |
| Branch C · Spatial CNN (3-D) | 2,230,646 | 42.9% | 1.39 G | 34.2% | **87.4%** |
| Branch D · SpecFormer | 1,241,640 | 23.9% | 0.23 G | 5.7% | 3.1% |
| `morphology_embed` | 17,216 | 0.3% | — | — | (in fusion) |
| `cross_interaction` fusion | 496,005 | 9.6% | 0.001 G | 0.0% | — |
| 4× `aux_head` | 178,024 | 3.4% | — | — | train-only |
| `embed_net` + `arcface_head` | 333,056 | 6.4% | — | — | — |
| **Total** | **5,194,578** | 100% | **≈4.06 G** | 100% | — |

FLOPs are mine (2×MACs, forward, batch-1), computed from the shape matrix in `03` §3.6; the
ResBlock tail of Branch C is approximated. Not measured on hardware. The important and robust
part is the **ratio**: Branch A replicates a six-block 1-D ConvNeXt tower stack over **64 grid
cells per sample**, so its FLOPs scale with `grid_size_a²` while its parameters do not. It is the
single most expensive component in the network and ends at 5.6% influence.

### 2.4 Curriculum as executed

| | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|
| Budget / actual | 400 / **336** (early stop) | 150 / **49** (early stop) | 120 / **≥87 (log truncated)** |
| Wall clock | 6 h 35 m | 2 h 39 m | ≈9.5 h (extrapolated) |
| Objective | P1–2: CE+LS+mixup + 4×aux. P3: Focal+SupCon(0.35)+ProtoNCE(0.15)+aux | Focal + ArcFace(s=48) + SupCon(0.40) + ProtoNCE(0.18) + aux + balance | Focal(γ=1) + SupCon(0.02) + aux(0.10) + balance, under ASAM(ρ=0.015) |
| Sampler | shuffled → inverse-F1 oversampled (γ=0.65, cap 7×) | 16×8 class-balanced, CDWS (cap 3×) | same, Stage-2 CDWS |
| Margin | 0 (NormFace) | 0.18→0.35 global (ep 1–20), then per-class M(c) | κ·M(c), κ 1.00→0.85 per cycle |
| Best val macro-F1 | **0.842** (ep 286) | **0.844** (ep 19) | **0.847** (ep 63) |
| Precision | bf16 AMP (P1–2), **fp32 (P3)** | **fp32** | **fp32** |
| s/epoch | 39 (P1–2) → **190–405 (P3)** | 186–210 | 373–450 |

**Gradient flow.** One graph, no frozen parameters at any point, no stage boundary that changes
what receives gradient. Everything is trainable in all three stages; the only things that change
are the loss mixture, the sampler and the margin. There is no pretraining, no teacher/student, no
frozen backbone, no representation-learning phase in the SSL sense (see §7.5).

---

## 3. Dataset Audit

### 3.1 Ground truth from the primary source

**SOURCE-DERIVED FACT** (Zenodo record 3241923 / Strathclyde PURE, Vu, Tachtatzis, Murray, Harle,
Dao, Andonovic, Marshall, Fabiyi, 2020): 90 rice varieties; **96 kernels per variety**; 8,640
kernels total; each variety captured in **two imaging bundles of 48 kernels**; each bundle is
48 kernels of one variety laid out in an **8×6 matrix on a sheet of white paper** on a
translational stage; Specim V10E + Hamamatsu ORCA-05G, 256 bands ≈385–1000 nm; sessions named by
date (e.g. `Data-VIS-20170111-2-room-light-off`), and the source notes at least one bundle
(`NDC1-01`) was re-acquired in a different session after file corruption.

**MY INFERENCE, high confidence:** 90 varieties × 2 bundles = **180 acquisition units**, each
class-pure. This matches the log's measured `180 of 180 scans` exactly (8,624/180 = 47.9 patches
per scan ≈ 48 kernels per bundle).

**DOCUMENTATION DEFECT:** `01` §1.3 and `02` §2.8 both state the dataset has **107** capture
scans and that the leak was "measured 107/107". The executed run measures **180/180**. One of
these is wrong and it matters, because 107 is not divisible by 90 and would contradict the
"exactly two scans per class" statement made in the same paragraph. Fix the docs to 180.

### 3.2 Structure and what it implies

| Property | Value | Consequence |
|---|---|---|
| Samples | 8,624 patches (of 8,640 kernels) | 16 kernels lost to the shape gate — a silent, unaudited 0.19% class-dependent filter |
| Classes | 90 | balanced by construction |
| Samples/class | 91–96 | effectively balanced; **class imbalance is not a real problem here** |
| **Independent acquisition units/class** | **2** | the binding constraint on everything |
| One patch = one physical kernel | yes | **no same-specimen duplication** — a genuine strength, and better than most seed datasets |
| Near-duplicates | within-bundle neighbours share illumination, stage position, seed lot, handling | correlated, not duplicated |
| Resolution | 64×64 after area-resize of a variable-size crop | **absolute scale destroyed**; morphometrics restore it |
| Modalities | 40 SPA bands + fill map α + 8 morphometrics | RGB channel of the source dataset unused |
| Foreground | exact zero outside the kernel, enforced and unit-tested | clean; removes background as a nuisance |

**What the model actually needs to distinguish classes.** 90 varieties of *Oryza sativa* differ in
starch composition, protein, moisture, pericarp pigmentation and hull texture. The discriminative
signal is:

- **Global and chemometric** (absorption shape in 900–1000 nm water/starch overtones, visible
  pigmentation) — captured by the mean spectrum. Empirical anchor: LDA on 40-band mean spectra
  scores **0.5916** accuracy (`band_selection_report.csv`, patch-level 5-fold, therefore also
  leaky). So ~59 points are available from the global spectrum alone.
- **Local and spatial-spectral** (hull texture, awn/glume structure, spatial heterogeneity of
  pigment) — needs a joint operator. Empirical anchor: the full model reaches ~84.5% accuracy
  under the same leaky protocol, so ~25 points are attributable to everything beyond the mean
  spectrum, of which the fusion gate assigns 87% to Branch C.

This is the single most useful pair of numbers in the whole project, and it justifies keeping
exactly two things: a joint spectral–spatial operator, and the mean spectrum.

### 3.3 Nuisance variables, ranked

1. **Bundle identity** — perfectly confounded with class (each bundle is one variety). Residual
   after per-pixel SNV: spectral *shape* drift (lamp colour temperature, detector response),
   per-column dark-current residual (the dark frame is column-averaged), focus/height, and the
   seed-lot/handling effects shared by 48 kernels processed together. **Magnitude unknown.**
2. **Stage position within the 8×6 tray** — pushbroom column index maps to sensor column; the
   dark correction is per-column, so residuals are position-dependent.
3. **Session date** — `Data-VIS-2017xxxx`; at least one variety spans two sessions.
4. **Orientation** — genuinely nuisance, correctly handled by the D₄ augmentation and D₄ TTA.
5. **Size** — destroyed by the resize, partially restored by morphometrics.

**Credit where due:** the per-pixel SNV at prep time removes the *scalar* per-pixel and
per-session gain — the largest single bundle artifact — and the persisted `gain.npy` is never fed
back to the model. This meaningfully *reduces* leakage. It does not eliminate it, and no
experiment in this project measures what remains.

---

## 4. Data Leakage / Evaluation Protocol Audit

The prompt asks for four categories to be distinguished. Applying them:

| Category | Present? | Where |
|---|---|---|
| **Label leakage** | **YES** | Band selection (§4.1). Test-patch labels entered the choice of the 40 input features. |
| **Sample / data-distribution exposure** | **YES, severe** | Stratified split: all 180 class-pure bundles appear in train *and* eval (§4.2). |
| **Transductive / self-supervised pretraining** | **N/A** | There is no SSL stage and no unlabelled pool. Training is fully supervised. |
| **Strict inductive evaluation** | **NO** | Not achieved by either shipped protocol; `grouped` gets closest. |

### 4.1 Band selection is fitted on the whole labelled dataset (HIGH)

`02` §2.4: mRMR relevance is `mutual_info_classif(x_k, y)` over **X ∈ ℝ^{8624×256}** — every
patch, including the 1,294 that will become test. The 40 selected bands are a hyperparameter of
the input representation, chosen with test labels in scope.

**What claim this supports:** "given a band set chosen with knowledge of the full labelled
corpus, this model achieves X." **What it does not support:** any statement about deploying the
pipeline on unseen varieties or unseen data. The optimism from selecting features outside the
resampling loop is well documented (Ambroise & McLachlan, PNAS 99(10):6562–6566, 2002 —
**SOURCE-DERIVED**: they show cross-validated error estimates become severely optimistic when
gene selection precedes CV, and near-zero apparent error can be obtained on random labels).

**Aggravating detail, self-documented:** `02` §2.4 already concedes the shipped
`band_selection_report.csv` **cannot demonstrate its own elbow** — the curve terminates at k=40,
its own chosen value, so the 98%-of-peak criterion is satisfied vacuously. There is a unit test
pinning exactly this. So the band count is both leaky *and* unjustified by the artifact that
selected it.

**Fix:** run band selection inside each training fold (train patches only), or fix k=40 a priori
and report the selection as part of the pipeline being cross-validated.

### 4.2 The stratified split leaks every acquisition bundle (CRITICAL)

The log is explicit:

```
Split: stratified (fold 0)  train: 6,036 (70%)  val: 1,294 (15%)  test: 1,294 (15%)
  ⚠ patch-level split — 180 of 180 scans are in train and in val/test (C-1).
```

With 2 class-pure bundles per class and ~48 patches each, a class's ~67 training patches and its
~14 val patches are drawn from **the same two trays**. Any residual tray signature is a perfect
class predictor on the eval split.

I want to be careful about how strong this claim is. It is **not** established that the model
*is* exploiting bundle identity — only that it *can*, that nothing prevents it, and that no
measurement excludes it. Two observations bear on it:

- **Mitigating:** per-pixel SNV removes the dominant scalar gain, and `gain.npy` is not an input.
- **Aggravating:** the model reached 96–98% *training* accuracy in Phase 3 on 6,036 samples with
  5.19 M parameters. A network with that much fitting capacity, given a feature that is both
  present and perfectly predictive, will use it.

**The decisive experiment costs one config flag and is already implemented** (§20, A1).

### 4.3 `grouped` is the honest protocol, and it is nearly saturated

`02` §2.8 is admirably candid: with exactly two groups per class, three-way group-disjointness is
*mathematically impossible*; `split_eval_frac=0.30` realises close to a **50/50 one-scan-out
split**, with val and test being two halves of the same held-out bundle. Sweeping
`split_fold ∈ {0,1}` is the **complete** leave-one-bundle-out cross-validation the dataset
supports — a 2-fold CV, and there is no third fold to be had.

Under `grouped`, training sees **one** bundle per class, so the training set contains **zero
within-class acquisition variance**. The model cannot learn what varies across acquisitions
because it never sees two. This is a hard ceiling imposed by the data collection, not by the
method, and it must be stated in the paper.

Note also that under `grouped`, val and test are halves of the same held-out bundle, so **they are
not independent of each other**. Model selection on val is still partially self-fulfilling on
test. The correct reporting is: select on `calib` (carved from train, `calib_frac=0.15`, which
`spa40_90class_pfix.yaml` already ships), and treat val∪test as one held-out bundle scored once.

### 4.4 `val` is used three times over (HIGH)

The run's own banner: `Fitted on: val (1294 patches) | Selected on: val (1294 patches)`, because
`calib_frac=0.0`. On that one 1,294-patch split the system fits:

1. per-class ArcFace margins M(c) (from R−P measured on val),
2. the row-normalised confusion matrix Ω for the pairwise penalty,
3. CDWS class-sampling weights,
4. Phase-3 inverse-F1 oversampling weights,

then **selects the checkpoint on the same split**, then **reports the number from it**. Anything
these mechanisms achieve is achieved partly by fitting the noise of the split they are scored on.
`spa40_90class_pfix.yaml` already ships `calib_frac: 0.15` and fixes this; it was not used.

### 4.5 Quantified selection bias

Epoch-to-epoch val macro-F1 in the Stage-1 plateau (ep 200–330) oscillates roughly 0.79–0.83 →
sd ≈ 0.012. Selections: 336 + 49 + 87 ≈ 472 epochs, each taking `max(F1_live, F1_ema)` ⇒ ~944
draws. For approximately-independent draws, `E[max] − mean ≈ σ√(2 ln n)`:

| σ | n = 472 | n = 944 |
|---|---|---|
| 0.008 | +0.028 | +0.030 |
| **0.012 (observed)** | **+0.042** | **+0.044** |
| 0.016 | +0.056 | +0.059 |

Draws are correlated so this over-estimates, but even at a third of the value the bias is
**≈0.015**, i.e. **three times the entire Stage-2 + Stage-3 gain of +0.005**. Sampling noise on
1,294 samples adds ±0.020 (95%). Conclusion: **0.847 is not a 0.847-level estimate of anything.**

### 4.6 Is the test set untouched?

Yes — trivially, because it was **never evaluated**. The log ends mid-Stage-3 and `final_eval`
never ran. That is currently the project's only clean asset. It must stay clean: the test split
should be scored **once**, after all design decisions are frozen.

---

## 5. Architecture Audit

### 5.1 Branch-by-branch

**Branch A — SpectralProfile (603 k params, ≈2.44 GFLOP, 60% of compute, 5.6% end influence)**

*Purpose:* gain-free spectral shape + exact λ-derivatives, on an 8×8 grid, with wavelength-generated
kernels (`LambdaConv1d`) so parameter cost is band-count independent.

*Assessment:* the `LambdaConv1d` idea is elegant and the band-count-transfer property is real and
tested. But three things undercut the branch as built:

1. **It runs the entire six-block tower stack 64 times per sample.** Parameters are shared across
   cells; *compute is not*. This is where 60% of the FLOPs go.
2. **Its gain-invariance is largely redundant.** The cube has already had per-pixel SNV applied at
   prep time (`02` §2.3). Branch A then applies SNV again. It is being made invariant to a
   variation that was removed before it saw the data.
3. **The evidence says it is not used.** Influence falls monotonically 78% → 5.6% over 300 epochs.
   Its aux weight is driven to the GradNorm floor of 0.25 within ~10 epochs because its aux loss
   falls fastest — consistent with Branch A carrying the *easy* (mean-spectrum-shape) part of the
   signal, the part a linear model already extracts 59 points from.

**Verdict: REMOVE the branch.** Keep the SNV + Savitzky–Golay derivative operators as *fixed,
zero-parameter features* fed to the cheap spectral head. Cost ≈0; the physics is preserved.

**Branch B — Index Bank (95 k params, 0.0002 GFLOP, 3.5% influence)**

*Purpose:* 64 learned soft normalised-difference indices + 16 continuum-removed absorption depths
+ 8 morphometrics, from the foreground mean spectrum.

*Assessment:* by far the best cost/benefit ratio in the network — **1.8% of parameters, 0.005% of
compute**, and it carries the chemometric signal that decades of NIR seed literature is built on.
Its 3.5% fused influence is not evidence of uselessness; it is evidence that the fusion gate
prefers C, which the dropout policy taught it to do. Two flaws: the gain-invariance is redundant
(as above), and morphometrics enter the model **twice** (concatenated here *and* as the fifth
fusion modality).

**Verdict: RETAIN, in simplified form.** It is nearly free. Feed it morphometrics once.

**Branch C — Spatial CNN (2.23 M params, 1.39 GFLOP, 87.4% influence)**

*Purpose:* the only joint spectral–spatial operator; three `Conv3d` stages fold λ into channels,
then four `ResBlock2D` + `CBAM`.

*Assessment:* this is the model. It is the only component whose contribution is supported by
convergent evidence — rising influence, the largest parameter share, and the fact that it is the
only branch that can see texture × spectral position. The 3-D stem is band-count-agnostic by
construction and there is a synthetic band/space-swap test confirming it is genuinely joint.

**Caveat that must be stated:** its 87% influence is **confounded** by never being dropped (§5.2).
The correct experiment is a C-only model versus the full model (§20, A3).

**Verdict: RETAIN as the backbone.**

**Branch D — SpecFormer (1.24 M params, 0.23 GFLOP, 3.1% influence)**

*Purpose:* a 4-layer, 8-head, d=192 transformer over λ-uniform tokens with a relative-λ attention
bias.

*Assessment:* **1.24 M parameters — 24% of the model — to run attention over 10 tokens** derived
from a 40-band spectrum, on a dataset with ~67 training samples per class. The λ-uniform
tokenisation and the strict-load band-count transfer property are genuinely nice engineering. But
the capacity is disproportionate to a 10-token sequence, transformers are the architecture family
most dependent on data scale, and the measured influence is 3.1%. There is also a live defect: the
`stride` argument computed as `specf_patch // 2` is passed to the constructor and never read
(`03` §3.8), and `set_dropout` cannot reach `nn.MultiheadAttention`'s internal dropout, which
therefore stays at 0.15 for the entire run regardless of the stage dropout schedule.

**Verdict: REMOVE.** UNJUSTIFIED — highest parameter-per-unit-influence in the network, and the
mechanism it implements (long-range band interaction over 10 tokens) is within reach of a 1-D
convolution stack a fraction of the size.

### 5.2 The branch-dropout policy is a confound, not just a regulariser

`p = 0.20 × (0.75, 0.75, 0.0, 0.75) = (0.15, 0.15, 0.0, 0.15)` for A/B/C/D. Branch C is **never**
dropped; the other three are absent from the fused path 15% of the time.

**OBSERVATION** → Branch C influence rises 0.2% → 87.4% while A falls 78% → 5.6%.
**INFERENCE** → The fusion gate learned to route around the branches that intermittently vanish
and onto the one that never does. Dropping a modality's fused path teaches the gate that the
modality is unreliable, which is exactly what modality dropout is *for* in multimodal
learning — but here it was applied asymmetrically, so it did not regularise the gate, it *biased*
it.
**EVIDENCE** → the drop-rate vector; the monotone influence trajectories in `influence/branch_*`.
**CONFIDENCE** → Medium-high for the mechanism; the direction is unambiguous, the magnitude is not.
**ALTERNATIVE** → Branch C genuinely carries most of the discriminative information (it is the only
branch with spatial texture) and would dominate under a symmetric policy too. The alternative is
plausible and is **not excluded by anything in this run.** Resolving it needs A3 (§20).

Either way the design rationale — "protect the branch nothing else can reconstruct, drop the ones
that can be" — is self-defeating: if A, B and D *can* be reconstructed from C, they are redundant
by the project's own argument and should not be in the model.

### 5.3 Fusion

`CrossModalInteraction`: 5 BatchNorm1d + a 5-way sigmoid gate over a 1,285-d input + a rank-128
bilinear pool over all 10 modality pairs + a 512→256 output. 496 k parameters, 9.6% of the model,
for a fusion in which **three of five modalities carry ≤6% influence each**. A rank-128 second-order
interaction across ten pairs is a large hypothesis class to fit from 6,036 samples in service of
combining what is effectively one strong signal and one weak one.

**Verdict: REPLACE with concatenation + a 2-layer MLP**, or at most retain the sigmoid gate. The
bilinear term is a candidate for an ablation (§20, A5), not a default.

### 5.4 Classification head

- **Sub-centres K=3.** The log records, at Stage-2 entry: `Seeded 90 classes; worst within-class
  sub-centre cosine 0.987`. Under either reading of "worst" (max similarity in the worst class, or
  min over classes), **the sub-centres of a class are near-collinear at initialisation** — spherical
  k-means over real embeddings could not find three separated modes. K=3 therefore triples the head's
  parameters and adds a KL load-balancing term whose job is to keep degenerate structure alive.
  With ~67 training samples per class there are ~22 samples per sub-centre. **Verdict: set K=1.**
- **Signed per-class margin M(c) = clip(0.35 + 0.20(R−P), 0.20, 0.50).** The *rule* is genuinely
  better-reasoned than the usual F1-driven one — an additive angular margin shrinks a class's
  decision region, so over-claiming classes should get a larger margin. I agree with the sign
  argument. But the run shows the realised spread is `mean=0.348, min=0.264, max=0.412` — a total
  range of 0.15 around a base of 0.35 — and, decisively, **Stage 2's best checkpoint is epoch 19,
  two epochs before the per-class vector takes over at epoch 21.** The mechanism never contributed
  to the selected model. **Verdict: UNCERTAIN, currently UNJUSTIFIED. Remove from the default; keep
  the code and test it in isolation (§20, A7).**
- **Pairwise confusion penalty** (δ=0.10 × row-normalised Ω, fitted on val at Stage-2 entry).
  Fitted on the selection split (§4.4), never ablated, and aimed at the hard classes that never
  moved (F7). **Verdict: REMOVE from the default.**
- **s = 48.** High for a 256-d embedding at 90 classes. Combined with per-group clipping this is
  partly self-correcting, but it is untuned.

### 5.5 Capacity vs. independent information

- 5.19 M parameters / 6,036 training patches = **860 parameters per training sample**.
- 5.19 M / 90 independent training bundles under `grouped` = **57,700 parameters per independent
  acquisition unit**.

The prompt rightly warns against "the model is large" as an argument. So here is the *evidence*
that capacity is being spent unproductively rather than merely being present:

**OBSERVATION** → At the Phase 2→3 boundary (epoch 273) training accuracy jumps **42% → 96.6% in a
single epoch**, while validation macro-F1 moves 0.816 → 0.816. Over the remaining 63 epochs train
accuracy reaches 98.3% and val gains +0.026 (much of it selection bias, §4.5).
**INFERENCE** → The network was never capacity-limited. Mixup, which is switched off at that
boundary, was the *only* thing holding training accuracy down. Once removed, the model memorised
the training split almost immediately. A ~14-point train/val gap opened and never closed.
**EVIDENCE** → `train/acc` panel (step ~273 discontinuity); the log's `Tr:` column; `02` §2.6's
augmentation table, where even the `heavy` profile has trigger probabilities summing to 0.26, so
**~77% of "heavily augmented" samples receive no spectral augmentation at all** — only the D₄
spatial transform.
**CONFIDENCE** → High. The discontinuity is exactly one epoch wide and coincides with the documented
mixup switch-off.
**ALTERNATIVE** → The jump is also the augmentation profile changing (`medium`→`very_light`) and the
sampler changing (shuffled→oversampled) at the same epoch. All three change together, so the
attribution to mixup specifically is an inference. **However**, the augmentation profiles differ by
only a few percentage points of trigger probability, and oversampling makes the training
distribution *harder*, not easier — so mixup is by far the most plausible cause.

**Conclusion on capacity:** the network is not too large *per se* — it is too large **for the
regularisation actually in force**. The right response is not "shrink blindly" but "remove the
components with no demonstrated benefit (which happens to remove 55% of the parameters and 66% of
the FLOPs) and keep a real regulariser switched on for the whole run."

---

## 6. Training Curriculum Audit

Classification per the prompt's scheme. "Demonstrated" means *demonstrated by evidence in this
project*, not by the literature.

| # | Component | Purpose | Demonstrated? | Class |
|---|---|---|---|---|
| 1 | Supervised CE/Focal on the fused head | learn the task | Yes — this is what learns | **ESSENTIAL** |
| 2 | Mixup α=0.35, Phases 1–2 | regularise | Yes — the only thing holding train acc at 42% | **ESSENTIAL** |
| 3 | Label-smoothing decay 0.10→0.04 | calibration | No isolated evidence; cheap, standard | LIKELY USEFUL |
| 4 | EMA (d=0.999) | variance reduction | Yes — `f1_ema` is smoother and usually higher than `f1_live` | **ESSENTIAL** |
| 5 | D₄ spatial augmentation | pose invariance | Yes, physically correct (no canonical seed pose) | **ESSENTIAL** |
| 6 | 4× auxiliary heads + deep supervision | keep A/B alive | Partly — but at ~7.8× the main loss it inverts the objective | **HARMFUL as weighted** |
| 7 | GradNorm aux reweighting (α=0.5) | balance branch gradients | No — saturates at clip bounds (§10.3) | **UNJUSTIFIED** |
| 8 | 3-phase progressive augmentation | curriculum | No — profiles differ by ~2–4 pp of trigger probability; the only real transition is mixup off | **REDUNDANT** (it is a mixup switch) |
| 9 | Phase-3 SupCon (0.35) + ProtoNCE (0.15) | metric structure | No — confounded with aug change + oversampling; costs 5–10× per epoch | **UNCERTAIN** |
| 10 | Phase-3 hard-class oversampling (7× cap) | fix hard classes | **No — hard classes did not move (F7)** | **UNJUSTIFIED** |
| 11 | Phase-aware LR (3 regimes + 30-ep restarts) | optimisation | No — largely nullified by clipping (§8.1) | **UNCERTAIN** |
| 12 | EMA re-init at phase boundaries | clean averaging | Not isolated; plausible | UNCERTAIN |
| 13 | Sub-centre ArcFace K=3 + τ anneal + KL balance | multi-modal classes | **No — sub-centres near-collinear at seeding (0.987)** | **UNJUSTIFIED** |
| 14 | Global margin warm-up 0.18→0.35 | stability | Weak — best epoch (19) is inside the warm-up | UNCERTAIN |
| 15 | Signed per-class margin M(c) | asymmetric errors | **No — never active at the selected checkpoint** | **UNJUSTIFIED** |
| 16 | Pairwise confusion penalty δ=0.10 | targeted separation | No; fitted on the selection split | **UNJUSTIFIED** |
| 17 | CDWS class-balanced sampling (3× cap) | hard classes | No — classes are already balanced (91–96/class) | **REDUNDANT** |
| 18 | SGDR (T₀=25, ×2) | escape minima | No — restart at ep 28 was followed by 21 epochs of no improvement, then early stop | **UNJUSTIFIED** |
| 19 | Whole of Stage 2 | metric learning | **+0.002 in 2.65 h** | **REDUNDANT** |
| 20 | ASAM (ρ=0.015) | flat minima | No isolated evidence; doubles cost/step | **UNCERTAIN** |
| 21 | Greedy SWA (8-ep cycles, 3 warm-up) | averaging | Weak — 4 accepted / 3 rejected, net +0.003 | **UNCERTAIN** |
| 22 | Margin anneal κ 1.0→0.85 | late relaxation | No | **UNJUSTIFIED** |
| 23 | Whole of Stage 3 | refinement | **+0.003 in ~9.5 h** | **REDUNDANT** |
| 24 | 12-view TTA (8 D₄ + 4 spectral gain) | inference | Never executed | UNCERTAIN (cheap, keep) |
| 25 | Same-class CutMix (p≈0.06–0.10) | intra-class variety | Not isolated; label-preserving, cheap | LIKELY USEFUL |
| 26 | Branch dropout (0.15,0.15,0,0.15) | fusion robustness | Actively biased the gate (§5.2) | **HARMFUL as configured** |

**The pattern.** Eleven mechanisms are UNJUSTIFIED or HARMFUL, and eight of those eleven exist to
solve the same problem: *the hard classes*. Oversampling, CDWS, signed margins, pairwise confusion,
sub-centres, SupCon, ProtoNCE and focal γ are all, in one way or another, hard-class machinery.
**None of them moved the hard classes** (F7). This is a coherent, load-bearing negative result and
it should drive the redesign: stop attacking the hard classes with the loss function, and find out
instead *what they are* (§20, A9).

---

## 7. Loss / Objectives Audit

### 7.1 The Stage-1 objective is inverted

$$
\mathcal{L}_{\text{S1}} = \underbrace{\lambda\mathcal{L}_{CE}(z,y)+(1-\lambda)\mathcal{L}_{CE}(z,y_\pi)}_{\text{the head that is evaluated}}
+ \underbrace{w_{\text{aux}}(e)\sum_{b}\omega_b\,\mathcal{L}_{CE}(z_b,\cdot)}_{\text{four heads discarded at eval}}
$$

Reading $\omega$ from the `aux_weight/branch_*` panels and $w_{\text{aux}}$ from the log's `auxW`
column:

| Epoch | ω_A | ω_B | ω_C | ω_D | Σω | w_aux | aux : main | main share of gradient |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ~20 | 0.25 | 3.9 | 4.0 | 4.0 | 12.15 | 0.64 | **7.8 : 1** | **11%** |
| ~200 | 0.7 | 3.8 | 3.6 | 1.8 | 9.90 | 0.42 | 4.2 : 1 | 19% |
| ~330 | 0.55 | 2.5 | 4.0 | 1.0 | 8.05 | 0.27 | 2.2 : 1 | 32% |

**This explains the loss curve.** `train/loss` *rises* from 22 to 34 over the first ~20 epochs
before falling — an anomaly that would otherwise look like divergence. It is not: it is
$\sum_b \omega_b$ ramping into the clip ceiling while the underlying CEs are already falling. The
total loss is not a comparable quantity across epochs, because its own weights are non-stationary.

**Consequence:** for the first third of Stage 1 the network optimises four linear probes on
unmasked branch embeddings roughly 8× harder than it optimises the fused representation the
evaluation actually uses. **This is a genuine objective mismatch**, and it is the mechanism most
likely to explain why the fused path took ~120 epochs to overtake single-branch behaviour.

*Alternative explanation I considered and reject:* the aux heads might be a deliberate warm-start
that is later annealed away. But $w_{\text{aux}}$ only floors at 0.25 and $\sum\omega$ stays ≈8, so
the aux term is still 2.2× the main loss at epoch 330. It is never annealed away.

### 7.2 SupCon and ProtoNCE are near-redundant, and one is dead

- **SupCon** (τ=0.10) on a 16×8 balanced batch: 8 positives per anchor, 120 negatives. Well-posed
  for this batch shape.
- **ProtoNCE** (τ=0.10) uses **in-batch** class means μ_c as prototypes. With 8 samples per class,
  μ_c is an 8-sample estimate of a 256-d unit vector — high-variance, and it is *a summary of the
  same positives SupCon is already pulling together*. Two losses, one signal, at the same
  temperature, on the same normalised embedding. Estimator-variance-wise ProtoNCE is the worse of
  the two (its gradient depends on a noisy prototype), and it is the cheaper only in FLOPs
  (O(B|C_batch|) vs O(B²)), which is irrelevant at B=128.
- **ProtoNCE is inactive in Stage 3 despite its weight being logged.** `04` §4.4 states the module
  is passed with weight 0.01 but the Stage-3 loop applies no ProtoNCE term. `sched/proto_weight`
  therefore logs a constant for a term that does not exist. Dead code with live telemetry is worse
  than dead code.

**Verdict: REMOVE ProtoNCE entirely. Keep SupCon only if A6 (§20) earns it.**

### 7.3 The AMP interaction is the real cost of the contrastive terms

`use_amp = (supcon is None) ∧ (scaler is not None)`. Passing a SupCon module — which happens for
all of Phase 3 and all of Stage 2 — **disables autocast for the entire epoch**, not just for the
contrastive term. Combined with the memory warning below, this is why Phase 3 epochs cost
190–405 s against Phases 1–2's 39 s.

This is a fixable implementation choice, not a property of SupCon. SupCon needs fp32 for the
`exp(·/τ)` reduction at τ=0.10; it does not need the *backbone forward* in fp32. The correct
pattern is bf16 autocast with an explicit `.float()` cast on the normalised embeddings before the
similarity matrix.

### 7.4 Focal loss on a balanced dataset

γ=1.5 with classes at 91–96 samples each. Focal loss addresses foreground/background imbalance at
ratios of 1000:1; here the imbalance is 96:91. It is doing down-weighting of easy examples, not
imbalance correction. Harmless, unjustified, one more untested hyperparameter. Combined with
label smoothing it is also slightly self-defeating — the implementation is careful about this
(`04` §4.4: the modulator reads the *unsmoothed* p_y so it can still reach 0), which is a nice
detail, but the right question is whether γ>0 helps at all here. Untested.

### 7.5 On the prompt's DINO / self-distillation questions

The prompt asks about teacher/student asymmetry, teacher centring, teacher/student temperatures,
Sinkhorn normalisation and iterations, prototype count and utilisation, KoLeo, global/local crop
semantics, and whether the SSL objective supervises the downstream representation.

**None of these exist in this system.** There is no self-supervised stage, no teacher/student pair,
no DINO head, no Sinkhorn, no centring, no local crops, no unlabelled pool. Training is fully
supervised from epoch 1. I will not manufacture an analysis of components that are not present.
The nearest structural analogues, and what the evidence says about them:

| DINO concept | Nearest thing here | Status |
|---|---|---|
| EMA teacher | `ModelEMA` shadow (d_max=0.999) | **Not a teacher** — takes no gradient, produces no target. It is an evaluation/averaging device only. It is used correctly and it works. |
| Prototypes (65,536 in DINOv2) | ArcFace sub-centres, K=3 per class (270 total) | Degenerate at seeding (cos 0.987, §5.4) |
| Prototype utilisation / entropy | `L_balance` = Σ_c KL(π_c ‖ Uniform_K) | Implemented but never logged as a curve; utilisation is unobservable in this run |
| Collapse | Not applicable — a supervised CE floor prevents representation collapse | No collapse observed |
| Local/global crops | 8×8 (A) and 4×4 (D) grid pooling | Multi-scale *within* a sample, not a multi-crop objective |
| KoLeo diversity | none | absent |

**A defensible reason to consider adding SSL exists, and I flag it as speculative (§25/D):** with
2 acquisition bundles per class, a self-supervised objective whose positives are *the same variety
across different bundles* would directly attack the nuisance this dataset is built around. But
with only 2 bundles that objective has almost no data to work with. **I do not recommend adding
SSL.** It is the wrong tool for 8,624 labelled samples with 90 balanced classes.

### 7.6 Numerical conditioning — a genuine strength

Credit where due: the cosine clamp at ε=10⁻³, the arccos guard capping θ+m at π/2, the
`nan_to_num` on eval logits, forcing fp32 evaluation via `autocast(enabled=False)`, the explicit
`GradScaler(device=device.type)` binding, and the documented fp16→bf16 migration after non-finite
losses are all correct and better than typical. The log shows `train/skipped_batches = 0` for the
entire run. **No numerical instability was observed and none should be expected.**

---

## 8. Optimisation Audit

### 8.1 Gradient clipping nullifies the learning-rate schedule (HIGH)

`grad_clip = 1.0`, applied **per group** (head / fusion / backbone). Pre-clip norms from the
`grad_norm/*` panels:

| Group | Pre-clip L2 (typical) | vs clip=1.0 | Effective scaling |
|---|---:|---|---|
| `preclip_backbone` | 25–50 | **always clipped** | ×0.02–0.04 |
| `branch_a` | 10–25 | **always clipped** | ×0.04–0.10 |
| `branch_b` | 10–20 | **always clipped** | ×0.05–0.10 |
| `branch_c` | 10–20 | **always clipped** | ×0.05–0.10 |
| `branch_d` | 8–25 | **always clipped** | ×0.04–0.13 |
| `preclip_fusion` | 6.9 → ~1.0 | clipped early, marginal later | — |
| `preclip_head` | 2.3 → 0.3 | clipped early only | — |

**OBSERVATION** → The backbone's pre-clip norm is 25–50× the clip threshold for the entire run.
**INFERENCE** → The backbone is doing **normalised-gradient descent at a fixed step size**, not
Adam-with-a-schedule. Once the norm is renormalised to 1.0 every step, the elaborate phase-aware
LR schedule (warm-up → 0.6× → 0.2× → 30-epoch cosine restarts) controls step size but the *shape*
of the schedule is interacting with a direction-only update. The three-regime schedule that `04`
§4.1 pins bit-exact in unit tests is, in magnitude terms, doing far less than it appears to.
**EVIDENCE** → `grad_norm/preclip_backbone` panel; `grad_clip=1.0` in the config dump.
**CONFIDENCE** → High that clipping binds on essentially every step. Medium on the practical
consequence, because AdamW's per-parameter second-moment normalisation already makes the update
partly scale-free, which softens the effect.
**ALTERNATIVE** → With AdamW, a global rescale of g partly cancels in `g/√v`, so a constant clip
factor is less damaging than it would be under SGD. This is a real mitigation and it is why the
run trains at all. It does not make the situation intentional.

**Fix:** either raise `grad_clip` to ~5–10 (so it clips outliers, which is its job) or make it
adaptive (e.g. a running 90th percentile). Then re-tune the LR. **Do not do both at once.**

### 8.2 GradNorm is a bang-bang controller, not a balancer

$\omega_b \leftarrow \text{clip}(\omega_b (\bar g/g_b)^{0.5},\, 0.25,\, 4.0)$, once per epoch.

This is *not* GradNorm (Chen et al., ICML 2018), which optimises the task weights by gradient
descent against a target based on relative inverse training rates. It is a multiplicative
feedback loop with no restoring force except the clips. Its only fixed point is $g_b = \bar g$
exactly; any persistent deviation compounds geometrically until a bound is hit.

**The panels confirm it.** `aux_weight/branch_c` and `branch_b` sit pinned at **4.0** for most of
training; `branch_a` collapses to the **0.25** floor within ~10 epochs. So the "adaptive" weights
are, in practice, **the clip bounds** — a fixed vector (0.25, 4, 4, ~1–4) chosen by
`clip_min`/`clip_max`, not by any balance criterion. The documented default `(2, 2, 1, 1)` never
stands.

**Verdict: set `aux_gradnorm_alpha = 0` and use a fixed, small, explicit weight vector.** If
per-branch weighting is wanted later, implement actual GradNorm with a target and a learning rate
on ω, and log the fixed point.

### 8.3 Batch size, effective batch and estimator variance

- Stage 1: 128, shuffled, `accum=1` → 47 steps/epoch. Fine.
- Stages 2–3: 16 classes × 8 samples = 128, 47 batches/epoch. For SupCon this is the *right*
  shape (8 positives/anchor). But it means each epoch touches 47×16 = 752 class-slots over 90
  classes ≈ 8.4 visits/class, with classes drawn *without replacement within a batch* from a
  CDWS-weighted categorical — so per-class visit counts are themselves random. Combined with the
  unseeded per-epoch RNG (`02` §2.7), **the sampler is a documented source of run-to-run variance
  that has never been quantified**, and `05` §5.5 warns explicitly: *"A single-seed ablation delta
  smaller than run-to-run variance is not evidence."* Every delta in this run is smaller than
  run-to-run variance has ever been shown to be, because it has never been measured.

### 8.4 SAM/ASAM

ρ=0.015 with ASAM (perturbation ∝ |θ|). Doubles forward+backward per step; Stage 3 costs ~380–450
s/epoch. `sam/grad_cos` is computed and logged — the single most informative diagnostic for
whether SAM is doing anything (if ascent and descent gradients are nearly parallel, ρ is too small
to matter) — **and it is absent from the uploaded panels**, because of the W&B step collision
(§10.1). So the one number that would justify Stage 3 was computed and lost.

### 8.5 Precision and runtime settings contradict the project's own invariant

`06`'s opening invariant: *"every field under `cfg.runtime` is a throughput knob, and changing one
must never change a reported metric,"* and `allow_tf32` defaults to **false** specifically because
it cuts matmul mantissas from 24 to 11 bits. **The run set `allow_tf32=True`.** It also set
`num_workers=0` (auto would give 8 on CUDA) and `compile=off` (auto would give on for CUDA) —
i.e. the two knobs that would have made it *faster* were disabled and the one that changes
numerics was enabled. This is not a correctness bug but it is an experimental-hygiene failure: the
run is not comparable to a run at the shipped defaults.

---

## 9. Compute & Memory Audit

### 9.1 Where the compute goes

| Component | Params | FLOP/sample | FLOP share | End influence | **FLOPs per influence point** |
|---|---:|---:|---:|---:|---:|
| Branch A | 0.60 M | 2.44 G | **60.1%** | 5.6% | **436 MFLOP** |
| Branch C | 2.23 M | 1.39 G | 34.2% | 87.4% | **16 MFLOP** |
| Branch D | 1.24 M | 0.23 G | 5.7% | 3.1% | 74 MFLOP |
| Branch B | 0.09 M | 0.0002 G | 0.0% | 3.5% | **0.06 MFLOP** |

Branch A is **27× less compute-efficient than Branch C** per unit of fused influence; Branch B is
**270× more** efficient than C. The ranking is robust to large errors in my FLOP estimates.

The mechanism is structural: Branch A's parameters are shared across the 8×8 grid, so its
*parameter* cost is modest, but its *compute* scales as `grid_size_a² = 64`. Reducing
`grid_size_a` from 8 to 4 would cut its cost 4× at a stroke — a cheaper intervention than removal,
and a good ablation (§20, A4) if the branch is kept.

### 9.2 Memory — the run was paging

```
[MEM] peak 11.7 GB of a 12.9 GB working set (91%) on cuda. The allocator pages rather than
failing here, so if this run is slower per sample than a smaller stage*.batch, this is why…
```

This warning fired at epoch 1 and is the best explanation for the Phase-3 slowdown:

**OBSERVATION** → Epoch time goes 39 s (P1–2, bf16) → 190–405 s (P3, fp32), a 5–10× jump, on
identical data with a sampler that draws the same number of indices.
**INFERENCE** → fp32 roughly doubles activation memory on a working set already at 91% of an 11 GB
RTX 3060; the allocator starts paging against a host page cache that is simultaneously holding the
5.65 GB mmapped cube. The slowdown is memory thrash, not arithmetic.
**EVIDENCE** → The `[MEM]` warning text; `train/epoch_s` shows a single-epoch step from ~39 to
~200 at step 273; AMP is disabled exactly there (`use_amp = supcon is None`); Stage 3 (also fp32,
plus SAM's second pass) sits at 373–450 s.
**CONFIDENCE** → High.
**ALTERNATIVE** → SAM's double backward explains Stage 3's 2× over Stage 2, but not Phase 3's 5–10×
over Phase 2, since Phase 3 has no double backward. SupCon's own O(B²D) cost is ~4 MFLOP —
negligible. Neither alternative accounts for the magnitude.

**Fixes, in order of value:** (1) keep bf16 autocast through the contrastive phases with an fp32
cast only on the similarity matrix; (2) `num_workers=4–8`, `persistent_workers`, `prefetch_factor`
(the host cost is 1.86 ms/sample, 1.41 ms of it mmap page-in — at `num_workers=0` that is ~11 s per
epoch fully serialised); (3) drop batch to 64 in fp32 phases; (4) `compile=auto` on CUDA.

### 9.3 Compute return on investment

| Stage | Wall clock | Δ val macro-F1 | Δ in eval samples | Hours per 0.001 F1 |
|---|---:|---:|---:|---:|
| Stage 1 | 6.6 h | (baseline 0.842) | — | — |
| Stage 2 | 2.7 h | **+0.002** | **2.6 / 1294** | 1,325 |
| Stage 3 | ~9.5 h | **+0.003** | **3.9 / 1294** | 3,167 |

**65% of the run's wall clock bought 6.5 samples of a 1,294-sample split**, against a ±26-sample
95% CI. Under any reasonable accounting, Stages 2 and 3 as configured have negative expected value.

### 9.4 A cheap win nobody took

Stage-1 Phases 1–2 run at 39 s/epoch. The proposed single-stage curriculum (§17) at ~150 epochs
with the trimmed model (≈1.4 GFLOP vs 4.06) and working data loading should complete in **well
under an hour**. That converts today's 19-hour single run into a budget of ~20 runs — enough for
the 2-fold × 3-seed protocol *and* the full ablation grid, on the same GPU, in the same wall clock.
**This is the single largest practical gain available in the whole audit.**

---

## 10. Log and Figure Forensics

### 10.1 The figures are Stage 1 only — Stages 2 and 3 have no telemetry

**OBSERVATION** → Every Stage-2 and Stage-3 logging call in the log is followed by
`wandb: WARNING Tried to log to step N that is less than the current step 336… this data will be
ignored.` All seven uploaded panels terminate at step ≈336.
**INFERENCE** → Stage 1 ended at epoch 336, leaving W&B's global step at 336. Stages 2 and 3 restart
their step counter at 1, so **every scalar they logged was silently discarded**. The uploaded
figures are a complete record of Stage 1 and contain **zero** information about Stages 2 and 3.
**EVIDENCE** → ~200 identical warnings in the log; the x-axis of every panel.
**CONFIDENCE** → Certain.
**IMPACT** → `sam/grad_cos`, the Stage-2 margin curves, `swa/*`, and every Stage-2/3 branch
diagnostic are lost. Any figure-based claim about Stages 2 or 3 is unsupportable.
**FIX** → pass a monotone global step (`global_step = cumulative_epoch`) or use
`wandb.define_metric` with a per-stage step key. One line.

### 10.2 The loss/branch_* panels do not measure what they appear to

`05` §5.2: `_compute_aux_loss(return_components=True)` returns the **weighted** terms
$\omega_b\mathcal{L}_b$. Since $\omega_b$ is itself non-stationary and pinned at the clip bounds
(§8.2), `loss/branch_a` crashing from 11 to ~1 at epoch ~10 is **the weight collapsing to 0.25**,
not Branch A suddenly solving the task. Cross-check: at epoch 10 overall training accuracy is
15.9%; an aux CE of 1.0 would imply that branch alone is at ~70% train accuracy. The two are
irreconcilable, and the weighted-quantity reading resolves it.

**This is an observability defect with real consequences** — it is the panel a reader would use to
judge branch health, and it is confounded with the controller acting on it. **Log both**
$\mathcal{L}_b$ and $\omega_b\mathcal{L}_b$.

### 10.3 Aux weights saturate at the clip bounds

`aux_weight/branch_c` and `branch_b` sit at the 4.0 ceiling for hundreds of epochs;
`branch_a` sits at the 0.25 floor from ~epoch 10 to ~60 before creeping to ~0.7. A controller whose
output spends most of its life on a bound is not controlling. See §8.2.

### 10.4 The branch-influence transition, and its two candidate mechanisms

**OBSERVATION** → `influence/branch_a` 48 → 78% (ep 3–6) → monotone decline → 5.6%.
`influence/branch_c` 0.2% → 30% (ep 32) → 65% (ep 50) → plateau ~70% → **step to 87% at ep 276**.
**INFERENCE** → Two regimes. (a) Epochs 1–50: the network solves the task with the cheapest
available signal — the spectral profile — then discovers spatial-spectral structure and reallocates.
That is healthy and is exactly what a fusion gate should do. (b) Epoch 273–276: the *discontinuous*
jump to 87% coincides exactly with the Phase-3 boundary, where mixup switches off and training
accuracy jumps to 96.6%. As the model begins to memorise, it concentrates on the highest-capacity
pathway.
**EVIDENCE** → `influence/*` panels; the epoch-276 log block (`A: 6.0 B: 3.6 C: 87.0 D: 3.4`) against
epoch-267's (`A: 14.0 B: 10.9 C: 70.3 D: 4.9`).
**CONFIDENCE** → High that the two regimes exist; Medium on the memorisation reading of regime (b).
**ALTERNATIVE** → The Phase-3 switch also changes augmentation and sampler; and the influence metric
is a KL on the *fused* path, which the asymmetric branch dropout biases (§5.2). Both alternatives
are live. What is not in doubt: after epoch 276 the fused decision is a Branch-C decision.

### 10.5 The hard classes are a stable structure, not a training artifact

Bottom-10 class IDs across the run:

| Checkpoint | Bottom-5 |
|---|---|
| S1 ep 46 | 30, 49, 66, 41, 52 |
| S1 ep 141 | 49, 52, 51, 41, 30 |
| S1 ep 248 | 52, 49, 51, 70, 41 |
| S1 ep 285 | 49, 52, 70, 51, 41 |
| P2→3 boundary | 49, 52, 51, 41, 70 |
| S2 ep 19 (best) | 52, 51, 49, 70, 41 |

**INFERENCE** → {41, 49, 51, 52, 70} form a persistent confusable cluster, invariant to 470 epochs
of training, three loss regimes, two samplers and four difficulty-targeted mechanisms. The
precision/recall table logged at Stage-2 entry sharpens it: `c49: R=0.36 P=0.33`, `c41: R=0.40
P=0.43`, `c51: R=0.43 P=0.46`, `c52: R=0.47 P=0.30`, `c42: R=0.50 P=0.70` — **recall and precision
are both low for most of them**, which is the signature of *mutual confusion within a cluster*, not
of a threshold being in the wrong place. c52 (R=0.47, P=0.30) is over-claiming; c42 (R=0.50,
P=0.70) is under-claiming. They are trading predictions with each other.
**CONFIDENCE** → High.
**ALTERNATIVE** → These could be classes with poor segmentation (broken/small kernels failing the
`300<area<800` gate asymmetrically) rather than spectrally similar varieties. **This alternative is
cheap to test and has never been tested** (§20, A9) — and it would change the recommendation
completely, because a segmentation problem is fixable and a genetic-similarity problem is not.

### 10.6 What the curves do *not* show

No collapse. No divergence. No oscillation beyond normal epoch noise. No gradient explosion
(`skipped_batches = 0` throughout). No teacher/student drift (no teacher). The optimisation is
healthy and well-behaved. **The problem is not that training is going wrong. The problem is that
training going right is not being measured against anything.**

---

## 11. What Is Working

1. **Engineering discipline.** Schema-versioned checkpoints, both-files-required auto-resume,
   golden-forward-pass regression tests with per-tensor SHA-256, scheduler values pinned bit-exact
   across the full epoch range, a config round-trip checker, an RNG-ordering invariant stated at
   the call site. This is better than most published research code and it is what makes the audit
   possible at all.
2. **Preprocessing invariants.** The exact-zero-background guarantee — mask *after* resize, divide
   by fill fraction — is correct, non-obvious, and unit-tested. Every downstream masked statistic
   depends on it.
3. **Honest self-documentation.** The docs state the scan leak, the two-scan-per-class limitation,
   the vacuous elbow, the inactive Stage-3 ProtoNCE, the unread `stride` argument, and the
   `set_dropout` gap that leaves `nn.MultiheadAttention` at 0.15. A project that documents its own
   defects is one you can audit. Several of my findings are elaborations of admissions the authors
   already made.
4. **The signed-margin *argument*.** M(c) = clip(m + m_Δ(R−P), ·) is genuinely better reasoned than
   the usual 1−F1 rule. I disagree with keeping it (untested, inactive at the selected checkpoint),
   not with the reasoning.
5. **Numerical hygiene.** fp32-forced evaluation, arccos capping, device-bound GradScaler, the
   documented fp16→bf16 migration. Zero skipped batches in 19 hours.
6. **Branch C.** The only component with convergent evidence of contribution.
7. **Mixup + EMA + D₄.** Three cheap, standard, effective things that are demonstrably load-bearing.
8. **The `grouped` split and calibration split already exist**, correctly implemented, with a
   `SplitReport` that names which classes leak. The fix for the largest problem is a config flag.

## 12. What Is Not Working

1. **The evaluation protocol** (§4). Everything else is downstream of this.
2. **Nothing has been compared to anything.** One run, one seed, one split, no control arm, no
   baseline, no ablation, no test evaluation. `05` §5.5 lists 21 levers; zero pulled.
3. **Stage 2 and Stage 3** deliver +0.005 for 65% of the compute (§9.3).
4. **The hard-class machinery** — 8 mechanisms, 0 effect (§10.5).
5. **The objective is dominated by discarded heads** (§7.1).
6. **The adaptive weight controller saturates at its bounds** (§8.2).
7. **Clipping nullifies the LR schedule** (§8.1).
8. **Branch dropout biased the fusion gate** (§5.2).
9. **Stage-2/3 telemetry does not exist** (§10.1).
10. **The run diverges from the shipped defaults** on three runtime knobs, one of which changes
    numerics (§8.5).
11. **Phase 3 pays 5–10× per epoch for an avoidable memory/precision interaction** (§9.2).
12. **The run did not finish.** Stage 3 log ends at 87/120; `final_eval` never ran.

## 13. Components With Little / No Demonstrated Benefit

Ordered by (parameters + compute) freed per unit of evidential loss:

| Rank | Component | Cost | Evidence for it |
|---|---|---|---|
| 1 | **Branch D (SpecFormer)** | 1.24 M params (23.9%) | 3.1% influence |
| 2 | **Branch A (SpectralProfile)** | 0.60 M params, **60% of FLOPs** | 5.6% influence, monotone decline |
| 3 | **Stage 3 entire** | ~9.5 h | +0.003, inside noise |
| 4 | **Stage 2 entire** | ~2.7 h | +0.002, inside noise; best epoch precedes its own key mechanism |
| 5 | **Bilinear fusion (rank 128, 10 pairs)** | 0.50 M params | none isolated |
| 6 | **Sub-centre K=3 + KL balance** | 46 k params + a loss term | sub-centres collinear at seeding (0.987) |
| 7 | **Signed per-class margin + Ω penalty** | 2 buffers, 4 hyperparameters | inactive at the selected checkpoint |
| 8 | **GradNorm aux reweighting** | 1 hyperparameter | saturates at clip bounds |
| 9 | **Phase-3 hard-class oversampling** | 4 hyperparameters | hard classes unmoved |
| 10 | **CDWS** | 2 hyperparameters | dataset is already balanced |
| 11 | **ProtoNCE** | 2 hyperparameters | redundant with SupCon; inactive in Stage 3 |
| 12 | **SGDR in Stage 2** | 3 hyperparameters | restart at ep 28 → 21 stale epochs → early stop |
| 13 | **3-phase augmentation curriculum** | 2 hyperparameters | profiles differ by 2–4 pp; the real transition is mixup |
| 14 | **Margin anneal κ→0.85** | 1 hyperparameter | none |

**Hyperparameter accounting.** The three stage configs expose **69 fields**. My count of those with
*any* evidence in this run: **six** (`epochs`, `batch`, `max_lr`, `mixup`, `dropout`, `patience`).
The remaining ~63 are unconstrained by evidence, and each one is a researcher degree of freedom
that a reviewer is entitled to ask about.

## 14. Components That Should Be Removed

**Remove now (no experiment needed — the evidence is already in, or the mechanism is defective):**

- Branch D entirely
- Branch A as a branch (keep SNV + Savitzky–Golay derivatives as fixed features into the spectral head)
- Stage 3 entirely
- Stage 2 entirely (its objective folds into Stage 1 — see §17)
- ProtoNCE
- GradNorm aux reweighting (`aux_gradnorm_alpha=0`)
- Sub-centres (`subcenter_K=1`) and the KL balance term
- Signed per-class margins and the pairwise confusion penalty (from the default; keep the code)
- CDWS and hard-class oversampling
- The 3-phase augmentation split (replace with a single profile + a mixup schedule)
- Asymmetric branch dropout
- The rank-128 bilinear fusion (replace with concat + MLP)
- SGDR, margin annealing, `swa_*`, `sam_*` (with Stage 3)

**Net effect:** 5.19 M → **≈2.4 M** parameters, 4.06 → **≈1.4 GFLOP/sample**, three stages → one,
69 stage hyperparameters → ~12, 19 h → **<1 h per run**.

## 15. Components That Should Be Retained

| Component | Why |
|---|---|
| Branch C (3-D stem + ResBlock/CBAM tail) | The only evidence-supported discriminative component |
| Branch B (index bank + continuum depths) | 1.8% of params, 0.005% of compute, carries the chemometric signal |
| Morphometrics (8, once) | Restores the physical scale the 64×64 resize destroys |
| `MaskedSpectralECA` | 6 parameters. Free. Keep. |
| Exact-zero background + masked pooling ops | Correctness-critical invariants |
| Mixup | The only demonstrably load-bearing regulariser |
| EMA (d=0.999) | Demonstrably reduces evaluation variance |
| D₄ augmentation and D₄ TTA | Physically correct — a segmented seed has no canonical pose |
| Same-class CutMix | Label-preserving, cheap, composes with a margin |
| Label smoothing | Cheap, standard |
| Cosine / single-margin ArcFace head | Keep the head; drop the elaborations |
| Per-group gradient clipping (retuned) | The right *idea* — the ArcFace head's s=48 gradient genuinely should not divide everyone else's LR |
| fp32-forced evaluation, `nan_to_num`, arccos cap | Numerical hygiene |
| The whole test/regression harness | It is what makes a redesign safe |
| `grouped` split + calibration split + `SplitReport` | Already correct — start using them |

---

## 16. Proposed Architecture

### 16.1 Design principle

Prefer the simplest architecture that can plausibly reach the achievable ceiling, then earn every
addition with an ablation. The evidence supports exactly two information pathways: a **joint
spectral–spatial operator** (worth ~25 points over the mean spectrum) and the **global mean
spectrum** (worth ~59 points on its own). Everything else in the current model is a variation on
one of those two, competing for the same gradient.

### 16.2 `SpectralSeedNet` (proposed)

```
x (B,40,64,64), mask α (B,64,64), morph (B,8)
  │
  ├─ MaskedSpectralECA (6 params)                            → x' (B,40,64,64)
  │
  ├─ SPATIAL PATH  ────────────────────────────────────────────────────────────
  │    SpectralSpatialStem3D  (B,1,40,64,64) → (B,192,16,16)          178 k
  │    4× ResBlock2D + 3× CBAM → (B,256,4,4)                        1,921 k
  │    pn-mean ⊕ pn-max, ℓ2, Linear(512→256)                          132 k
  │                                                     → b_S (B,256)
  │
  └─ SPECTRAL PATH  ───────────────────────────────────────────────────────────
       masked_mean_spectrum → x̄ (B,40)
       [ SoftIndexBank(64) ‖ ContinuumDepths(16)
         ‖ snv(x̄)(40) ‖ D₁snv(x̄)(40) ‖ D₂snv(x̄)(40)                 ← from Branch A, free
         ‖ morph(8) ]                                → (B,208)
       LayerNorm → MLP 208→256→256                                     120 k
                                                     → b_T (B,256)

  concat [b_S ‖ b_T] (B,512) → Dropout(0.1) → Linear(512→256) → LN     131 k
  → EmbedNet (pre-LN residual MLP 256→512→256)                          264 k
  → ℓ2-normalise → ArcFace head, K=1, single global margin (90×256)      23 k
                                                     → logits (B,90)

  + 1 auxiliary head on b_S, weight 0.2, fixed                           35 k
```

**≈2.80 M parameters, ≈1.40 GFLOP/sample** — 54% of the parameters and 34% of the compute of the
current model. (Dropping the spatial ResBlock tail's width by 25% would reach ~2.2 M if the
capacity ablation A10 says to.)

### 16.3 What changed and why

| Change | Rationale | Evidence class |
|---|---|---|
| Branch A → fixed SNV + λ-derivative **features** into the spectral MLP | Keeps the physics (exact Savitzky–Golay operators on the irregular grid, zero parameters) and discards the 64-cell tower replication that costs 60% of FLOPs for 5.6% influence | Evidence-backed (A) |
| Branch D removed | 23.9% of parameters, 3.1% influence, 10-token sequence | Evidence-backed (A) |
| 5-modality gated bilinear pool → 2-way concat + MLP | Rank-128 second-order interactions over 10 pairs cannot be justified from 6,036 samples when 3 of 5 modalities are ≤6% | Theoretical (B) |
| Branch dropout removed | Only two pathways remain and the asymmetric policy biased the gate | Evidence-backed (A) |
| 4 aux heads → 1, fixed weight 0.2 | The spatial path needs a direct gradient early; four heads at 8× the main loss inverts the objective | Evidence-backed (A) |
| Sub-centres K=3 → K=1 | Seeded sub-centres are collinear (cos 0.987) | Evidence-backed (A) |
| Morphometrics enter once | Currently double-counted | Correctness |
| `MaskedSpectralECA` kept | 6 parameters | Free |

### 16.4 Architectures I considered and rejected

- **Shrink to a "Tiny" model (<0.5 M).** Rejected. The 59% LDA baseline and the 84.5% full-model
  number bracket a real ~25-point gain from spatial-spectral structure, and that gain needs a
  reasonable convolutional stack. The prompt warns against blindly shrinking, correctly.
- **ImageNet-pretrained 2-D backbone on band-triplets.** Rejected. The input is a 40-band
  reflectance cube whose discriminative content is spectral shape, not RGB texture statistics.
  ImageNet initialisation on a spectral cube is a domain mismatch, and the seed images are
  low-resolution 64×64 crops of a near-uniform object.
- **Adding iBOT / MIM / patch-level objectives.** Rejected. 8,624 labelled samples with 90 balanced
  classes is a supervised problem. Masked-image-modelling pretraining is a data-scale technique.
- **Self-supervised pretraining with cross-bundle positives.** Interesting and on-target for the
  actual nuisance (§7.5), but there are only 2 bundles per class. Rejected as speculative (D).
- **Mixture-of-Experts.** Rejected. The current model already *is* an implicit MoE (four experts +
  a gate) and it collapsed to one expert. Adding formal MoE machinery would repeat the experiment.
- **Keeping all four branches with symmetric dropout.** This is the honest alternative to removal
  and it is what ablation A3 tests. If A3 shows the 4-branch model beats a C+B model by more than
  run-to-run variance under the grouped protocol, **reverse this recommendation.**

---

## 17. Proposed Training Curriculum

**One stage. One objective. One schedule.**

```
Epochs 1–150 (early stop, patience 25 on calib macro-F1)

  Loss:      CE with label smoothing 0.10 → 0.04 (linear)
             + 0.2 × aux CE on the spatial path (fixed weight, no GradNorm)
             + mixup(α=0.35) for epochs 1–110, off for 111–150
  Head:      ArcFace, K=1, margin 0 → 0.30 cosine warm-up over epochs 111–130
             (margin only after mixup is off — they are mutually exclusive by construction)
  Sampler:   plain shuffled, batch 128    (classes are already balanced 91–96)
  Aug:       single `medium` profile throughout + D₄ + same-class CutMix
  Optimiser: AdamW, lr 5e-4, wd 2e-4, cosine to 5e-6, 5-epoch warm-up
  Clipping:  per-group, threshold 5.0  (clips outliers, does not renormalise every step)
  Precision: bf16 autocast throughout, fp32 cast on normalised embeddings only
  EMA:       d_max 0.999, no re-init
  Select:    max(F1_live, F1_ema) on the CALIBRATION split, not val
  Report:    val∪test scored ONCE at the end, with and without 12-view TTA
```

**Why one stage.** The three-stage structure exists to separate representation learning, metric
learning and flat-minimum refinement. In the measured run, stages 2 and 3 moved the metric by
0.005 combined. The *only* thing Stage 2 does that Stage 1 does not is introduce a non-zero
margin — and a margin is incompatible with mixup, which is why they were separated. Turning mixup
off at epoch 110 and warming a single margin in over epochs 111–130 achieves the same
transition **inside one stage, with one optimiser state, one schedule and no EMA re-initialisation.**

**Why the margin at all.** ArcFace at m=0 is NormFace, which is what Stage 1 already ran, and it
reached 0.842. The margin is worth *testing* (A7) because angular margins are well-supported in
fine-grained recognition generally — but it must be tested, not assumed.

**Optional Phase B, only if A6 earns it.** If SupCon at 16×8 balanced batches beats plain CE by
more than run-to-run variance under the grouped protocol, append 30 epochs of
`CE + 0.3 × SupCon` with a class-balanced sampler — in bf16, with the fp32 cast confined to the
similarity matrix.

**Expected wall clock:** ≈40–55 min/run on the same RTX 3060 (1.4 GFLOP/sample, working data
loading, bf16 throughout, no paging). Against 19 h today.

---

## 18. Proposed Hyperparameters

| Group | Key | Current | **Proposed** | Justification |
|---|---|---|---|---|
| data | `split_scheme` | `stratified` | **`grouped`** | §4.2 — the only protocol supporting a variety claim |
| data | `split_fold` | 0 | **sweep {0,1}** | The complete leave-one-bundle-out CV this dataset supports |
| data | `calib_frac` | 0.0 | **0.15** | Separates fitting/selection from reporting (§4.4) |
| data | `masks_path`, `morphology_path` | set | set | Keep — real information |
| model | `grid_size_a` | 8 | **n/a (branch removed)** | §9.1 |
| model | `specf_*` (7 keys) | — | **n/a (branch removed)** | §5.1 |
| model | `subcenter_K` | 3 | **1** | Collinear at seeding (§5.4) |
| model | `subcenter_tau_*`, `subcenter_balance_weight` | — | **n/a** | Follow K=1 |
| model | `fusion_rank`, `fusion_gate_hidden` | 128, 128 | **n/a (concat+MLP)** | §5.3 |
| model | `branch_drop_prob` | 0.20 | **0.0** | §5.2 |
| model | `index_bank_size` | 64 | 64 | Cheap; ablate later |
| stage | `epochs` | 400/150/120 | **150 (single)** | §17 |
| stage | `batch` | 128 | 128 | Unchanged |
| stage | `max_lr` | 5e-4 | **5e-4, then re-tune after the clip change** | §8.1 — change one thing at a time |
| stage | `grad_clip` | 1.0 | **5.0** | §8.1 — clip outliers, not every step |
| stage | `mixup` | 0.35 (P1–2) | **0.35, epochs 1–110** | The load-bearing regulariser |
| stage | `arcface_m` | 0.0 → 0.35 → κ·M(c) | **0 → 0.30, epochs 111–130, global scalar** | §5.4 |
| stage | `arcface_s` | 48.0 | **32.0** | 48 is high for d=256/C=90; ablate |
| stage | `focal_gamma` | 1.5 | **0.0 (plain CE)** | Balanced classes (§7.4); ablate before re-adding |
| stage | `aux_loss_weight` | 0.65 → 0.25, ×Σω≈8–12 | **0.2 fixed, single head** | §7.1 |
| root | `aux_gradnorm_alpha` | 0.5 | **0.0** | §8.2 |
| stage | `dropout` | 0.15 / 0.25 / 0.10 | **0.15 throughout** | Remove an untested schedule |
| stage | `patience` | 50 / 30 | **25** | On calib; fewer selection events (§4.5) |
| root | `ema_decay` | 0.999 | 0.999 | Working |
| root | `weight_decay` | 2e-4 | 2e-4 | Unchanged |
| root | `tta_spatial` / `tta_spectral` | 8 / 4 | 8 / 4 | Cheap; report separately |
| root | `seed` | 42 | **sweep {0,1,2}** | §8.3 — no delta is interpretable without this |
| runtime | `allow_tf32` | **True** | **False** | §8.5 — restore the project's own invariant |
| runtime | `num_workers` | 0 | **-1 (auto → 8)** | §9.2 |
| runtime | `compile` | off | **auto** | §9.2 |
| runtime | `amp_dtype` | bf16 | bf16 | Correct |

**Stage hyperparameter count: 69 → ~14.** Every one of the 14 either has evidence in this run or
is a standard default I am prepared to defend in review.

---

## 19. Required Data-Split Protocol

**This section is the paper's methods section. Nothing else in the audit matters if this is wrong.**

### 19.1 Primary protocol — leave-one-bundle-out, 2 folds

```
for fold in {0, 1}:
    data.split_scheme = grouped
    data.split_fold   = fold          # holds out one of the two bundles per class
    data.calib_frac   = 0.15          # carved from TRAIN, by group
    single_group_policy = "error"     # refuse to silently accept a leak
    for seed in {0, 1, 2}:
        train → select on calib → score val∪test ONCE
report mean ± range over 2 folds × 3 seeds  (6 runs, ≈5 h total)
```

Constraints that must be stated in the paper, not buried:

1. Training sees **one** acquisition bundle per class ⇒ **zero within-class acquisition variance in
   training**. The model cannot learn acquisition invariance because it never observes two
   acquisitions of the same class. This is a data-collection ceiling.
2. `val` and `test` are two halves of the **same** held-out bundle and are therefore **not mutually
   independent**. They must be treated as one held-out set, scored once. Selection happens on
   `calib`.
3. Two folds is the maximum; there is no third bundle. Report both, never their max.

### 19.2 Secondary protocol — the leaky one, reported as a contrast

Also run `stratified` at 3 seeds and report it **beside** the grouped number. The gap
`F1_stratified − F1_grouped` is not an embarrassment — it is a **result**, and given that the
published literature on this dataset reports 92.73–96.17% without stating a bundle-disjoint
protocol, it is arguably the most valuable number this project can produce.

### 19.3 Band selection must move inside the fold

Re-run `scripts/select_bands.py` **per fold, on training patches only**, or fix k=40 a priori and
declare it as a fixed design choice made on external grounds. Additionally re-run with the wider
`n_candidates` default so the elbow is demonstrable rather than vacuous (`02` §2.4 already flags
this and a unit test already pins it).

### 19.4 Reporting rules

- Macro-F1 primary; balanced accuracy, per-class recall, and the full 90×90 confusion matrix
  alongside.
- Every number as mean ± range over 3 seeds. **A single-seed delta is not a result** — the project's
  own `05` §5.5 says so.
- Test scored exactly once, after freezing.
- State that no model selection used test.
- Report the LDA-on-mean-spectra baseline (0.5916 leaky) recomputed under the grouped protocol.
  **This is the paper's most important baseline** and it costs seconds.

---

## 20. Required Ablation Studies

Ordered by information gained per GPU-hour. At ~50 min/run these are ~2 days total on one 3060.

| # | Question | Design | Runs | Decision rule |
|---|---|---|---|---|
| **A1** | **How much of the score is bundle recognition?** | Identical model/config, `stratified` vs `grouped`, 3 seeds each | 6 | **Blocks every other claim.** The gap is the headline result. |
| **A2** | Does band selection leak materially? | k=40 selected on all data vs selected within-fold, grouped | 6 | If gap > 2σ, all published numbers on this dataset need the caveat |
| **A3** | Is the 4-branch design justified? | {A,B,C,D} vs {B,C} vs {C} vs {B}, symmetric branch dropout, grouped | 12 | If {A,B,C,D} − {B,C} < 2σ → removal confirmed |
| **A4** | Is Branch A's 64-cell cost necessary? | `grid_size_a` ∈ {8,4,2} (only if A3 keeps A) | 9 | Cheapest capacity/compute trade in the model |
| **A5** | Is bilinear fusion worth 0.5 M params? | bilinear+gate vs gate-only vs concat+MLP | 9 | Runs only if A3 keeps ≥3 modalities |
| **A6** | Does SupCon help? | CE vs CE+SupCon(0.3), balanced sampler both arms | 6 | Must control the sampler — the current design confounds them |
| **A7** | Does any margin machinery help? | m=0 / global m=0.30 / +per-class M(c) / +Ω penalty | 12 | Four arms, one variable each |
| **A8** | Do Stages 2 and 3 add anything at all? | S1-only vs S1+S2 vs S1+S2+S3, grouped, 3 seeds | 9 | The falsification test for 65% of current compute |
| **A9** | **What are classes {41,49,51,52,70}?** | Not a training run: t-SNE of embeddings, mean-spectrum overlays, 90×90 confusion, segmentation-quality audit (area/eccentricity/solidity distributions, patches lost to the gate) for those 5 vs 5 easy classes | 0 | Distinguishes *spectrally inseparable varieties* from *segmentation failure*. **Changes the whole research direction.** |
| **A10** | Is capacity actually harmful? | Spatial-path width × {0.5, 0.75, 1.0, 1.5} | 12 | Answers "is 5.19 M too big" with data instead of intuition |
| **A11** | Is mixup the load-bearing regulariser? | mixup on/off × aug profile {heavy, medium, none} | 18 | Tests §5.5 directly |
| **A12** | Run-to-run variance | Identical config, 5 seeds | 5 | **Prerequisite for interpreting every row above.** Run this first. |

**Run A12 first.** Until σ is known, no delta in the table means anything. Then A1, then A9, then
A3/A8. A9 is the one with the largest potential to change the research question rather than the
model.

---

## 21. Experimental Validation Plan

**Phase 0 — instrument (½ day, no GPU).** Fix the W&B step collision; log unweighted *and* weighted
per-branch aux losses; log `sam/grad_cos` if SAM survives; log post-clip as well as pre-clip norms;
log sub-centre utilisation; add a `SplitReport` assertion that fails the run if group-disjointness
is violated when `grouped` is requested.

**Phase 1 — establish the baseline (1 day).** A12 (variance), then the LDA/SVC mean-spectrum
baseline under `grouped`, then A1. **Deliverable: σ, the honest floor, and the leakage gap.**

**Phase 2 — falsify the current system (1 day).** A8 and A3 on the *existing* architecture under
`grouped`. If Stages 2–3 or Branches A/D survive, this audit is wrong about them and the
recommendations in §14 should be reversed for those items. **Design this to be able to prove me
wrong.**

**Phase 3 — build the replacement (1 day).** Train `SpectralSeedNet` (§16) under §17 and §19.
Compare to the current 5.19 M model under identical conditions.

**Phase 4 — earn the additions (2 days).** A6, A7, A5, A10. Add back only what clears 2σ.

**Phase 5 — understand the errors (1 day).** A9. Then decide whether the remaining error is
addressable at all.

**Phase 6 — freeze and report (½ day).** Score test once, with and without TTA, both protocols.

**Success criterion for the whole programme:** not a higher number. It is being able to state, with
a confidence interval, **how much of rice-seed HSI classification performance on this dataset is
variety recognition and how much is acquisition recognition.** No published work on this dataset
that I could access answers that.

---

## 22. Expected Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **The grouped number is much lower — perhaps 0.55–0.70** | **High** | A "worse" headline number | Frame correctly: the number is not worse, the previous number was measuring something else. Report both. This is the paper's contribution. |
| Reviewers expect >95% because the literature reports 92–96% | High | Rejection risk | Pre-empt in the methods: state the split protocol explicitly and note that comparable published protocols are not stated to be bundle-disjoint |
| Removing branches loses real accuracy | Medium | Weaker model | A3 gates the removal. Reverse the recommendation if it fails. |
| Removing Stage 2/3 loses real accuracy | Low | Weaker model | A8 gates it. Current evidence: +0.005 for 65% of compute. |
| Raising `grad_clip` to 5.0 destabilises training | Medium | Divergence | Change it *alone*, watch `preclip_*`, keep the fallback. Do not co-tune LR in the same run. |
| Two folds are too few to bound variance | **Certain** | Wide error bars | 3 seeds per fold; report range not just mean; state the limitation |
| The hard classes are genuinely inseparable | Medium-high | A performance ceiling | A9 determines this. If genetic, say so — a well-characterised ceiling is a publishable result |
| A9 finds a segmentation bug | Medium | Rework | Would be *good* news: fixable, and it would explain F7 |
| Removing complexity reads as "less novel" | Medium | Reviewer perception | The novelty becomes the protocol result, which is stronger and more durable |

---

## 23. Scientific Defensibility

### 23.1 What can be claimed today

**Nothing about model performance.** There is no test evaluation, one seed, one leaky split, no
baseline, no ablation, and the headline is a maximum over ~944 selections on the split that also
carries every fitted parameter.

### 23.2 What could be claimed after §19–§21

- "Under leave-one-acquisition-bundle-out validation on the 90-variety Strathclyde rice HSI
  dataset, a `X`-parameter spectral–spatial CNN attains `A ± B` macro-F1 (2 folds × 3 seeds),
  against `C ± D` for LDA on 40-band mean spectra."
- "The same model attains `E ± F` under a patch-level stratified split in which all 180 class-pure
  acquisition bundles appear in both training and evaluation. The `E − A` gap quantifies the
  contribution of acquisition-session recognition to reported performance on this dataset."
- "Branches X and Y and training stages 2–3 do not improve macro-F1 beyond run-to-run variance
  (Δ = …, σ = …)." *— A negative result, properly powered, is a contribution.*

### 23.3 Three things that will be asked in review, and their current answers

1. *"Are your splits specimen-independent?"* Today: **no**, and the paper would have to say so.
2. *"Where does band selection sit relative to the split?"* Today: **outside it, with test labels
   in scope.**
3. *"How many seeds?"* Today: **one.**

Each has a cheap fix. None can be fixed after submission.

### 23.4 The honesty asset

The design documents already disclose the scan leak, the two-scan limitation, the vacuous elbow,
and several dead code paths. That candour is the project's strongest scientific asset. **Do not
lose it by reporting the stratified number as the headline.** A paper that measures its own leakage
and publishes the gap will outlive a paper that reports 96%.

---

## 24. Implementation Changes

Ordered by dependency. Format as specified.

---

**IC-1 · Fix the W&B step collision**
**FILE** `src/spectralquadnet/tracking/wandb_tracker.py`, `engine/stages/stage{1,2,3}_*.py`
**CURRENT** Each stage logs with `step=epoch`, restarting at 1. W&B rejects any step below the
running maximum, so all Stage-2 and Stage-3 scalars are discarded (~200 warnings in the log).
**NEW** Thread a monotone `global_step = stage_offset + epoch`; or call `wandb.define_metric` with a
per-stage step key.
**RATIONALE** §10.1. Stage-2/3 telemetry does not currently exist.
**DEPENDENCIES** None. Do this first — every later experiment depends on observability.
**VALIDATION** A 3-epoch run over all three stages emits zero `Tried to log to step` warnings and
`val/f1_best` is monotone across stage boundaries in the W&B UI.

---

**IC-2 · Log unweighted per-branch aux losses**
**FILE** `src/spectralquadnet/engine/train_epoch.py` (`_compute_aux_loss`)
**CURRENT** `loss/branch_{a,b,c,d}` carry $\omega_b\mathcal{L}_b$; the weight is non-stationary and
pinned at the clip bounds, so the curve tracks the controller, not the branch.
**NEW** Emit both `loss/branch_b_raw` ($\mathcal{L}_b$) and `loss/branch_b_weighted`.
**RATIONALE** §10.2 — the panel a reader uses to judge branch health is confounded with the
controller acting on it.
**DEPENDENCIES** IC-1.
**VALIDATION** Assert `weighted ≈ raw × aux_weight` to 1e-6 for all four branches, and that the
existing bit-identical-total test in `test_diagnostics.py` still passes.

---

**IC-3 · Switch the default split protocol**
**FILE** `configs/experiment/output_v12_spa40.yaml`
**CURRENT** Composes `data/spa40_90class` — `split_scheme=stratified`, `calib_frac=0.0`.
**NEW** Compose `data/spa40_90class_pfix` — `split_scheme=grouped`, `calib_frac=0.15`,
`single_group_policy="error"`. Keep `stratified` reachable as an explicit override for the contrast
arm.
**RATIONALE** §4.2, §4.4, §19. Largest single correctness change in the audit.
**DEPENDENCIES** `dataset/groups.npy` must exist (it does — `02` §2.3).
**VALIDATION** `SplitReport` shows train/eval group-disjointness for all 90 classes; the run banner
prints `Fitted on: calib` and `Selected on: calib` with *different* sizes; the "180 of 180 scans"
warning does not appear.

---

**IC-4 · Move band selection inside the fold**
**FILE** `src/spectralquadnet/data/prep/band_selection.py`, `scripts/select_bands.py`
**CURRENT** mRMR relevance = `mutual_info_classif(X_all, y_all)` over all 8,624 patches including
test.
**NEW** Accept a `train_idx` argument and restrict every step (decorrelation, FDR, mRMR, SPA, the
5-fold CV curve) to those rows. Emit one band file per fold. Also widen `n_candidates` past k=40 so
the elbow is demonstrable.
**RATIONALE** §4.1. Feature selection outside the resampling loop is a known optimism source
(Ambroise & McLachlan, PNAS 2002).
**DEPENDENCIES** IC-3 (needs the fold definition).
**VALIDATION** `test_the_shipped_curve_cannot_demonstrate_its_elbow` **flips to demonstrable**; a
new test asserts no test-split row index appears in the selector's input.

---

**IC-5 · Disable GradNorm; fix the aux weight**
**FILE** `configs/experiment/*.yaml` (`aux_gradnorm_alpha`), `engine/train_epoch.py`
**CURRENT** `aux_gradnorm_alpha=0.5`; ω saturates at the (0.25, 4.0) clip bounds; aux:main ≈ 7.8:1.
**NEW** `aux_gradnorm_alpha=0.0`; single aux head on the spatial path at fixed weight 0.2.
**RATIONALE** §7.1, §8.2. The controller is bang-bang and the objective is inverted.
**DEPENDENCIES** IC-2 (to confirm the change did what is intended).
**VALIDATION** `aux_weight/*` is constant for the whole run; total loss at epoch 1 is within 2× of
the main CE (≈4.5 at 90 classes) rather than ≈22.

---

**IC-6 · Retune gradient clipping**
**FILE** `configs/experiment/*.yaml` (`grad_clip`), `optim/clipping.py`
**CURRENT** 1.0 per group, against backbone pre-clip norms of 25–50 → clipped every step.
**NEW** 5.0. Additionally log post-clip norms and the per-epoch fraction of steps clipped.
**RATIONALE** §8.1. At the current threshold the backbone does normalised-gradient descent and the
LR schedule is largely decorative.
**DEPENDENCIES** IC-1. **Change this alone** — do not co-tune LR in the same run.
**VALIDATION** New metric `grad_norm/clip_fraction` drops from ≈1.0 to <0.2; a 20-epoch run remains
finite with `skipped_batches=0`.

---

**IC-7 · Keep AMP on through the contrastive phases**
**FILE** `src/spectralquadnet/engine/train_epoch.py`
**CURRENT** `use_amp = (supcon is None) ∧ (scaler is not None)` — passing a SupCon module disables
autocast for the whole epoch, driving Phase 3 to 190–405 s/epoch against Phase 2's 39 s.
**NEW** Keep bf16 autocast; wrap only the similarity-matrix computation in
`autocast(enabled=False)` after casting the ℓ2-normalised embeddings to fp32.
**RATIONALE** §7.3, §9.2. SupCon needs fp32 for `exp(·/0.1)`; the backbone forward does not.
**DEPENDENCIES** None.
**VALIDATION** SupCon loss agrees with the fp32 reference to <1e-4; epoch time in a SupCon phase
comes within 1.5× of a non-SupCon phase; `skipped_batches=0`.

---

**IC-8 · Restore the runtime invariant**
**FILE** the run's override set / `configs/experiment/*.yaml`
**CURRENT** `allow_tf32=True` (doc default False, "a precision change"), `num_workers=0` (auto → 8),
`compile=off` (auto → on).
**NEW** `allow_tf32=False`, `num_workers=-1`, `compile=auto`.
**RATIONALE** §8.5, §9.2. The one knob that changes numerics was on; the two that only change speed
were off.
**DEPENDENCIES** None.
**VALIDATION** Startup banner reads `TF32=False`; epoch time falls; eval macro-F1 on a fixed
checkpoint is unchanged to 1e-6 by the worker-count change.

---

**IC-9 · Simplify the head**
**FILE** `src/spectralquadnet/models/heads/arcface.py`, `configs/model/*.yaml`
**CURRENT** K=3 sub-centres (seeded collinear, cos 0.987), τ anneal, KL balance, signed per-class
M(c), pairwise Ω penalty.
**NEW** `subcenter_K=1`; drop the balance term; single global margin 0→0.30 warmed over epochs
111–130; per-class margins and Ω retained in code, off by default, behind A7.
**RATIONALE** §5.4. The sub-centres are degenerate; the per-class margin was never active at the
selected checkpoint.
**DEPENDENCIES** IC-3 (margins must be fitted on `calib`, not `val`).
**VALIDATION** `test_unified_head.py` still passes at K=1; `test_margin_rule.py` still pins the sign
property on the now-optional path; the seeding log line reports a within-class sub-centre cosine
for K=1 as trivially 1.0 and the check is removed rather than left misleading.

---

**IC-10 · Implement `SpectralSeedNet`**
**FILE** new `src/spectralquadnet/models/spectral_seed_net.py`; extend `models/__init__.py`
**CURRENT** `SpectralQuadNet`, 5.19 M params, 4 branches + morphology, gated bilinear fusion,
asymmetric branch dropout.
**NEW** §16.2: spatial path (Branch C verbatim) + spectral path (index bank + continuum depths +
SNV/D₁/D₂ of the mean spectrum + morphometrics) → concat → MLP → EmbedNet → K=1 ArcFace, one aux
head. `SCHEMA_VERSION = 4`. **Keep `SpectralQuadNet` intact** — the ablations need it.
**RATIONALE** §16.
**DEPENDENCIES** IC-9. Must not be merged before A3 and A8 report.
**VALIDATION** New golden-forward-pass test with per-tensor SHA-256; a parameter-count test asserting
≈2.8 M; a test that `SpectralQuadNet` checkpoints still load under `SCHEMA_VERSION=3` and raise
`SchemaTooOldError` against the new class.

---

**IC-11 · Collapse the curriculum**
**FILE** `train.py`, new `engine/stages/single_stage.py`, `configs/stage/single.yaml`
**CURRENT** Three stage modules, 69 config fields, ~19 h.
**NEW** One stage per §17. Keep `stage{1,2,3}_*.py` for the A8 ablation arms.
**RATIONALE** §6, §9.3.
**DEPENDENCIES** IC-3, IC-5, IC-6, IC-7, IC-10, and **A8 must have reported**.
**VALIDATION** `test_schedulers.py` extended to pin every LR/margin/mixup value across the 150-epoch
range; auto-resume and `_pick_best_checkpoint` still pass with one stage present.

---

**IC-12 · Multi-seed, multi-fold driver**
**FILE** new `scripts/run_protocol.py`
**CURRENT** Single runs by hand; Hydra multirun exists but no aggregation.
**NEW** Sweep `split_fold ∈ {0,1} × seed ∈ {0,1,2}`, aggregate to mean ± range, emit a
publication-ready table and the pooled 90×90 confusion matrix.
**RATIONALE** §19, §8.3. `05` §5.5 already warns that single-seed deltas below run-to-run variance
are not evidence.
**DEPENDENCIES** IC-3.
**VALIDATION** On a 2-epoch smoke config it produces 6 run directories and one aggregate JSON whose
mean matches a manual recomputation.

---

**IC-13 · Fix the documented scan count**
**FILE** `01_ABSTRACT_AND_OVERVIEW.md` §1.3, `02_DATASET_AND_PREPROCESSING.md` §2.8
**CURRENT** "107 capture scans (measured 107/107)".
**NEW** 180 acquisition bundles (90 varieties × 2), matching the executed run and the Zenodo record.
**RATIONALE** §3.1. 107 is not divisible by 90 and contradicts the "exactly two scans per class"
statement in the same paragraph.
**DEPENDENCIES** None.
**VALIDATION** A test asserting `len(np.unique(groups)) == 180` and that the docs' figure matches.

---

**IC-14 · Remove dead paths**
**FILE** `models/branches/specformer.py` (unread `stride` arg), `engine/stages/stage3_sam_swa.py`
(ProtoNCE passed but never applied), `tracking` (`sched/proto_weight` logged for an inactive term),
`data/mmap_store.py` (`gain.npy` written but never wired to any config key).
**CURRENT** Four documented dead paths, one of which emits live telemetry for a loss that does not
run.
**NEW** Delete or wire up. For `gain.npy`: either add a `gain_path` key and test whether brightness
helps (it is also the strongest bundle-identity signal, so this is an A1-adjacent experiment), or
stop writing it and remove the claim in `02` §2.3 that brightness "stays available as an explicit
input."
**RATIONALE** §7.2, §5.1. Dead code with live telemetry is worse than dead code.
**DEPENDENCIES** None.
**VALIDATION** `test_config_wiring.py` extended to assert every persisted dataset artifact is either
consumed by a config key or documented as unused.

---

## 25. Priority Roadmap

**P0 — Blocking. Nothing published without these.**

| | Action | Effort |
|---|---|---|
| 1 | IC-1, IC-2 — restore observability | 0.5 d |
| 2 | **A12** — measure run-to-run variance (5 seeds) | 0.5 d |
| 3 | IC-3 — grouped split + calibration split | 0.5 d |
| 4 | **A1** — stratified vs grouped, 3 seeds each | 0.5 d |
| 5 | IC-4 — band selection inside the fold; **A2** | 1 d |
| 6 | LDA/SVC mean-spectrum baseline under grouped | 0.1 d |

*Exit criterion: σ is known, the leakage gap is measured, an honest floor exists.*

**P1 — Falsification. Try to prove this audit wrong.**

| | Action | Effort |
|---|---|---|
| 7 | **A8** — S1 vs S1+S2 vs S1+S2+S3 under grouped | 0.5 d |
| 8 | **A3** — branch ablation with *symmetric* dropout | 1 d |
| 9 | **A9** — diagnose classes {41,49,51,52,70} | 1 d |

*Exit criterion: each removal in §14 is either confirmed or reversed by data.*

**P2 — Rebuild.**

| | Action | Effort |
|---|---|---|
| 10 | IC-5 … IC-9 — objective, clipping, AMP, runtime, head | 1 d |
| 11 | IC-10, IC-11 — `SpectralSeedNet` + single stage | 1.5 d |
| 12 | IC-12 — protocol driver; full 2×3 grid | 0.5 d |

**P3 — Earn the additions back.**

| | Action | Effort |
|---|---|---|
| 13 | A6 (SupCon), A7 (margins), A5 (fusion), A10 (capacity), A11 (mixup) | 2 d |
| 14 | A4 (`grid_size_a`), if A3 kept Branch A | 0.5 d |

**P4 — Freeze and write.**

| | Action | Effort |
|---|---|---|
| 15 | IC-13, IC-14 — docs and dead paths | 0.5 d |
| 16 | Test scored **once**, both protocols, ±TTA | 0.2 d |
| 17 | Write up, leading with the protocol result | — |

**Total ≈ 12 working days**, most of it GPU-idle. The current single run is 19 hours; the proposed
programme is ~40 runs at <1 h each.

---

# Final Research Judgment

**1 · Is the current architecture scientifically justified?**
**No.** Not because it is wrong, but because it is unfalsified. Four branches, three stages, five
fusion modalities, three sub-centres and eleven auxiliary mechanisms were built on hypotheses, and
in one executed run **not one of those hypotheses was tested**. `05` §5.5 documents 21 ablation
levers; zero have been pulled. Meanwhile the measured behaviour contradicts several design
rationales directly: sub-centres are collinear at seeding, the adaptive weight controller saturates
at its bounds, the per-class margin was inactive at the selected checkpoint, and three branches
carry ≈12% of the fused decision between them.

**2 · Which components demonstrably contribute to learning?**
Branch C (rising influence to 87%, largest parameter share, the only joint spectral–spatial
operator). Mixup (removing it moved training accuracy 42% → 96.6% in one epoch). EMA (`f1_ema`
consistently smoother and usually above `f1_live`). The D₄ augmentation (physically correct). The
supervised CE/Focal objective itself. Branch B is *not* demonstrated but is nearly free and carries
the signal the entire NIR-chemometrics literature is built on — I keep it on those grounds and flag
the reasoning as theoretical, not evidential.

**3 · Which components have little or no demonstrated value?**
Branch D (23.9% of parameters, 3.1% influence). Branch A as a branch (60% of FLOPs, 5.6%
influence). Stages 2 and 3 (+0.005 for 65% of compute). Sub-centres. Signed per-class margins. The
pairwise confusion penalty. CDWS. Hard-class oversampling. ProtoNCE. GradNorm reweighting. SGDR.
Margin annealing. The bilinear fusion term. The three-phase augmentation split. Fourteen items, and
eight of them were built to fix the hard classes, which never moved.

**4 · Which stages should be removed or shortened?**
Remove Stage 3 entirely. Remove Stage 2 entirely; fold its only distinct ingredient — a non-zero
angular margin — into a single stage by switching mixup off at epoch ~110 and warming one global
margin over epochs 111–130. Shorten Stage 1 from 400 to 150 epochs (it early-stopped at 336 and
gained +0.026 over its last 63 epochs, most of which is selection bias). **Three stages → one.**

**5 · Smallest architecture likely to retain the necessary representational capacity?**
≈2.4–2.8 M parameters: the Branch-C spectral–spatial stack (2.23 M) plus a ~0.12 M spectral MLP over
the mean spectrum, its SNV/λ-derivatives, learned NDIs, continuum depths and 8 morphometrics, plus
fusion/embed/head. I would **not** go below ~1.5 M without evidence: the 25-point gap between LDA on
mean spectra (0.5916) and the full model (~0.845) under the same leaky protocol is real
spatial-spectral signal, and a tiny model will not capture it. A10 settles the exact width; do not
guess it.

**6 · Most promising architecture?**
`SpectralSeedNet` (§16.2). Two pathways — joint spectral–spatial CNN, and global spectral
chemometrics — concatenated, one embedding block, one K=1 cosine/ArcFace head, one auxiliary head.
It is the current model with everything that did not earn its place removed, which is the correct
direction of travel when nothing has been ablated.

**7 · Most promising training curriculum?**
One stage, 150 epochs (§17): CE + label smoothing + one fixed-weight aux head, mixup for epochs
1–110, margin warmed over 111–130, plain shuffled batches, AdamW + cosine, bf16 throughout, EMA,
early stop on a **calibration** split, D₄ TTA at the end. ~45 min/run instead of 19 h — which is the
real prize, because it converts a project that can afford one run into one that can afford forty.

**8 · Primary source of current underperformance?**
There is no established underperformance, because there is no valid measurement. The primary source
of **uninterpretability** is the evaluation protocol (§4). If forced to rank the causes of the
error that does exist: **(i)** evaluation-protocol invalidity, which makes the number
uninterpretable in both directions; **(ii)** data — 2 acquisition bundles per class and ~67 training
patches per class, which bounds what any architecture can learn about acquisition invariance;
**(iii)** an intrinsic confusable cluster {41, 49, 51, 52, 70} unmoved by 470 epochs and eight
targeted mechanisms; **(iv)** optimisation friction (inverted objective, clipping, saturated
controller). Architecture is *not* on this list, and that is the most useful thing I can tell you:
**the model is not the bottleneck.**

**9 · Mandatory experiments before claiming improvement?**
A12 (variance — without σ nothing is interpretable), A1 (stratified vs grouped), A2 (band-selection
leakage), A8 (do Stages 2–3 do anything), A3 (branch ablation with symmetric dropout), plus the
LDA-on-mean-spectra baseline recomputed under the grouped protocol. **A9** is not strictly required
for a claim but is the highest-value experiment in the list, because it may change the research
question.

**10 · What would constitute convincing evidence that the revised system is better?**
Under `grouped`, both folds, ≥3 seeds, with band selection inside the fold, checkpoint selection on
`calib`, and test scored once: a macro-F1 improvement **exceeding 2σ of the measured run-to-run
variance**, on the *same* splits, with the confusion matrix and per-class recall published, and
with an ablation table showing which component contributed what. A +0.005 delta from a single seed
on a leaky split — the current situation — is not evidence of anything.

**11 · Is >95% realistically achievable?**

**Under a bundle-disjoint protocol: almost certainly not, and I would not pursue it.** Reasons:
(i) training sees exactly one acquisition per class, so there is zero within-class acquisition
variance to learn from; (ii) ~67 training samples per class across 90 fine-grained classes of one
species; (iii) a persistent mutually-confusable cluster that eight targeted mechanisms could not
move; (iv) LDA on mean spectra reaches only 0.59 *with* leakage.

**Under the leaky patch-level protocol: plausibly yes.** **SOURCE-DERIVED:** Taheri, Ebrahimnezhad
& Sedaaghi (*J Ambient Intell Human Comput* 15:2883–2899, 2024) report 92.73–96.17% overall
precision on this exact 90-class dataset using ensemble deep learning over 15 selected bands plus
RGB. **MY INFERENCE (medium confidence):** their abstract does not state a bundle-disjoint
protocol, and no accessible paper on this dataset does; given that the dataset ships as
class-pure bundles, a patch-level split is the likely default. If so, published 92–96% figures are
comparable to this project's *stratified* arm, not its grouped arm — and this project's 84.7% is
*below* them, which is worth noting plainly.

**The honest formulation for the paper:** ">95% is reachable on this dataset under patch-level
splitting, and we report that number for comparability; under leave-one-acquisition-bundle-out
validation the same system reaches X%, and the gap is the contribution of acquisition recognition
to reported performance." **Chasing >95% by keeping the leaky split would be reproducing the
field's error rather than correcting it.** The target should be reclassified from an objective to a
diagnostic.

**12 · What would make the final paper scientifically defensible?**

1. **Lead with the protocol.** Report grouped as the headline and stratified as the contrast. Make
   the gap a numbered result, not a footnote.
2. **Band selection inside the fold**, or declared as a fixed a-priori choice.
3. **≥3 seeds × 2 folds**, mean ± range, everywhere. Never a max.
4. **Test scored once**, after freezing; state it.
5. **Selection on a calibration split**, never on the reported split.
6. **A real baseline table**: LDA and LinearSVC on mean spectra under the identical protocol, plus
   at least one published architecture re-run on the identical splits.
7. **An ablation table** in which every retained component has a measured Δ with an error bar, and
   every removed component has a measured Δ showing why.
8. **Publish the negative results.** "Stages 2–3 contribute Δ = 0.005 ± σ" and "eight class-difficulty
   mechanisms did not move the hard classes" are, on a small fine-grained dataset, more useful to the
   field than another 96%.
9. **Characterise the hard classes** (A9) and state whether the residual error is genetic,
   acquisition-limited or segmentation-induced.
10. **Keep the candour.** The design documents already disclose the scan leak, the two-scan
    limitation and the vacuous elbow. That is the most scientifically valuable thing in this
    repository. Build the paper around it.

---

*Prepared from the six design documents, `config_rename_table.md`, the console log of run
`stratified_benchmark_rtx3060`, seven W&B panels (Stage 1 only — §10.1), and the cited primary
sources. Every performance figure quoted is a validation figure from a scan-leaky split selected on
the split it is reported from. No test-set result exists in this project. FLOP figures are my own
estimates from the published tensor-shape matrix, not hardware measurements, and are used only as
ratios.*
