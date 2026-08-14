# 1 · Abstract and System Overview

> ## The one-paragraph version
>
> This repository trains a hyperspectral classifier on the **complete 256-band VIS–NIR
> cube** of 8,624 single rice kernels across 90 varieties, under a
> **leave-one-acquisition-bundle-out** protocol. There is no band selection and no
> dimensionality reduction on the primary path. The headline artifact is not an accuracy
> but a **gap** — `F1_stratified − F1_grouped` — which measures how much of rice-seed HSI
> classification performance on this dataset is variety recognition and how much is
> acquisition recognition. Band selection is retained in full as a separate, opt-in
> research pathway (§1.6, `07_BAND_SELECTION_PATHWAY.md`).

> ## ⚠ Read `CHANGES.md` for the audit this revision implements
>
> Its central finding was not about the architecture:
>
> The source dataset images each variety as **two class-pure bundles of 48 kernels**, and
> the audited run split at the **patch** level — so all **180** acquisition bundles
> appeared in training *and* in evaluation. A model that learns "this tray's residual
> radiometric signature ⇒ class X" scores correctly, and the reported 0.847 macro-F1 is a
> mixture of variety recognition and acquisition-bundle recognition whose ratio was never
> measured.
>
> | | Audited | Now |
> |---|---|---|
> | Input | 40 SPA-selected bands | **all 256 acquired bands**, no selection |
> | Split | `stratified` (patch-level) | **`grouped`** — leave-one-acquisition-bundle-out |
> | Selection | on `val`, which also carried 270+ fitted parameters and produced the headline | on **`calib`**, carved from train by group |
> | Reported | `val`, as a maximum over ~944 correlated selection events | **`val ∪ test`, scored once**, with a bootstrap CI |
> | Seeds / folds | 1 / 1 | **3 seeds × 2 folds**, mean ± range, never a maximum |
> | Model | `SpectralQuadNet`, 5.19 M | **`SpectralSeedNet`, 3.05 M** at 256 bands (the four-branch model is retained as the control arm) |
> | Curriculum | 3 stages, ~19 h | **1 stage** |
> | Ablations executed | **0 of 21** | a declared grid with pre-registered decision rules |
>
> Expect the grouped number to be **lower**. That is the correct direction: the previous
> number was measuring something else. See `README.md` for how to run any of it.

> **Scope of this suite.** Every number, equation, default and contract in these seven
> documents is derived from `src/spectralquadnet/`, `configs/`, `train.py` and `tests/`.
> No training run has yet been recorded in this repository against the architecture
> described here — `outputs/` is git-ignored and empty in this tree — so no performance
> figures are reported below. Where these documents describe an evaluation artifact or
> metric, they describe the mechanism that produces it, not a specific recorded number.

| Document | Contents |
|---|---|
| `01_ABSTRACT_AND_OVERVIEW.md` | Objective, study design, contributions, evaluation framework |
| `02_DATASET_AND_PREPROCESSING.md` | Acquisition → segmentation → patches → data store → split protocols |
| `03_MODEL_ARCHITECTURE.md` | Both architectures, the 256-band native design, full tensor-shape matrix |
| `04_CURRICULUM_AND_LOSSES.md` | The single-stage curriculum, the retained three-stage one, schedules, objectives |
| `05_EXPERIMENTS_AND_ABLATIONS.md` | TTA inference, diagnostics, telemetry, checkpointing, regression gates, ablation surface |
| `06_EXECUTION_AND_HARDWARE.md` | Entrypoint orchestration, runtime performance knobs, distributed (DDP) training |
| `07_BAND_SELECTION_PATHWAY.md` | The retained band-selection research pathway |

---

## 1.1 Abstract

`SpectralSeedNet` assigns one of $C = 90$ rice-seed varieties to a single segmented
kernel, observed as a $256 \times 64 \times 64$ VIS–NIR reflectance patch spanning
383.2–1006.5 nm. The source cubes are the 256-band VIS–NIR scans of the *RGB and VIS-NIR
HSI Data for 90 Rice Seed Varieties* collection (Zenodo record `3241923`, referenced in
`data/prep/config.py::DATA_URL`); an offline pipeline applies per-pixel radiometric
normalisation, segments individual seeds, and extracts $64 \times 64$ patches **at the
full acquired band count**.

The classification problem is *fine-grained and near-degenerate*: 90 varieties of the same
species, $\sim$96 patches per class, discriminated by sub-percent differences in
reflectance shape rather than by morphology.

### Why the full cube

The audited pipeline reduced $256 \to 40$ bands by an mRMR/SPA selection before training.
That reduction is **removed from the primary path** for a reason that is methodological
rather than performance-driven:

- Both shipped reductions record `"demonstrable": false` in their own elbow files. Each
  cross-validated accuracy curve terminates at its own chosen $k$, so the 98%-of-peak
  elbow criterion is satisfied *vacuously* — the peak it is measured against is the peak
  of a truncated curve (`CHANGES.md` M-14). Neither $k$ was chosen by an experiment that
  could have returned a different answer.
- The mRMR relevance term was `mutual_info_classif(X, y)` over **every** patch, including
  the ones that become test. Feature selection outside the resampling loop is a known and
  quantified source of optimism (Ambroise & McLachlan, *PNAS* 99(10):6562–6566, 2002, who
  obtain near-zero apparent error on *random labels* that way). That is genuine label
  leakage, independent of the split protocol: it contaminates `grouped` too.
- A study whose stated question is *what rice-variety information VIS–NIR hyperspectral
  imaging carries* cannot begin by discarding 84% of the spectrum on an undemonstrated
  elbow.

So the primary path reads every acquired band, and "how few bands would do?" becomes a
measured question with its own experiment (§1.6) rather than an inherited assumption.

### The architecture, in four claims

1. **Two pathways, each seeing something the other structurally cannot reconstruct.** A
   joint spectral–spatial 3-D operator over the full cube (the only module in the network
   that can express "this absorption feature, in this part of the seed"), and a global
   chemometric descriptor over the foreground-masked mean spectrum. They are combined by
   concatenation and one MLP, not by attention and not by a second-order pool.
2. **Every width is a function of `data.num_bands`.** The 3-D stem's spectral strides are
   *derived* from the band count so the full cube folds at a bounded depth (§1.4); the
   continuum-hull operator is an exact $O(C^2)$ suffix maximum rather than an $O(C^3)$
   chord enumeration. Both reduce to the audited 40-band behaviour exactly, which is what
   keeps the reduced ablation arms comparable with the primary path.
3. **Wavelength is a first-class axis, not a band index.** The Savitzky–Golay derivative
   operators are fitted on the *actual* nanometre offsets, so they are exact on
   polynomials of the fitted degree regardless of band spacing. Branch B's
   normalised-difference indices and continuum-removed depths are constructed to be
   exactly invariant to a per-pixel or per-session gain, and therefore carry no
   wavelength-positional information at all — gain-invariance and wavelength-awareness are
   different, sometimes competing properties, assigned to different components rather than
   asking one to have both.
4. **One classification head.** An adaptive sub-centre ArcFace head at $K = 1$, whose
   angular margin is the only thing that changes across the curriculum: zero (a plain
   cosine/NormFace classifier) while mixup is active, then warmed to a single global
   scalar once mixup stops. Per-class margins and the pairwise confusion penalty remain
   in the code, off by default, measured by ablation A7.

`SpectralSeedNet` has **3,052,682 parameters**, all trainable. Full derivation in
`03_MODEL_ARCHITECTURE.md` §3.8.

### Recorded performance

No checkpoint has yet been produced or evaluated in this repository. `outputs/` is
git-ignored and the working tree contains no `.pth` artifacts, so there is no macro-F1,
accuracy or per-class figure to report. A run is produced by

```bash
python train.py
```

which trains the single-stage curriculum, auto-resuming from a completed stage on disk.
Final numbers are produced by `engine/stages/final_eval.py`, twice — once with a single
forward pass and once with the 12-view TTA of `05_EXPERIMENTS_AND_ABLATIONS.md` §5.1 —
writing predictions and targets to `cfg.output_dir`, so any reported metric is
recomputable from disk without re-running inference.

---

## 1.2 The study's two questions

### Q1 — the mixing ratio (the headline)

> *How much of rice-seed HSI classification performance on this dataset is variety
> recognition, and how much is acquisition recognition?*

Operationalised as `F1_stratified − F1_grouped` over matched arms that differ in the
**split scheme and nothing else** — same 256-band input, same calibration split, same
folds, same seeds, same augmentation geometry. Produced by `scripts/run_protocol.py`,
written to `outputs/experiments/protocol/leakage_gap.md`, and declared as ablation **A1**.

### Q2 — what the acquired spectrum supports

Q1 is a question about the protocol. Q2 is a question about the representation, and
answering it honestly requires not having discarded most of the representation first —
which is what §1.1 is about.

Its complement, *how few bands would suffice*, is the retained band-selection pathway
(§1.6) and ablation **A2**, whose reference arm is the primary path.

### Three constraints that belong in the paper

- Training sees **one** acquisition bundle per class, so there is **zero within-class
  acquisition variance in training**. The model cannot learn acquisition invariance
  because it never observes two acquisitions of one class. This is a ceiling imposed by
  the data collection, not by the method.
- **Two folds is the maximum.** There is no third bundle, so there is no third fold.
- `val` and `test` are two halves of the same held-out bundle and are therefore *not*
  independent of each other. They are scored together, once.

These are returned as data by `experiments/protocol.py::constraints()` so the run banner,
the README and the generated report cannot drift into three different phrasings.

---

## 1.3 Core contributions

### C1 · Two pathways over the full cube, concatenated

A single shared spectral-attention block (`MaskedSpectralECA`, 10 parameters at 256
bands) conditions the cube once, before either pathway sees it.

```
x (B,256,64,64) + mask (B,1,64,64) + morph (B,8)
  └─ MaskedSpectralECA ─────────────────────────────► x' (B,256,64,64)
        ├── x' ⊙ m, full cube      ─► SpatialPath   ─► (B,256)
        └── foreground mean x̄ (B,256) ─► SpectralPath ─► (B,256)
                                            │
                     concat (B,512) → Dropout → Linear → LayerNorm
                                            │
                              EmbedNet ─► ê (B,256)
                                            │
                     arcface_head (K=1, margin varies) ─► (B,90)
```

The spectral path is background-masked, so padded pixels never dilute the mean spectrum;
the spatial path is the only component that ever sees the raw spatial cube. One auxiliary
head provides deep supervision on the spatial path, at a **fixed** weight of 0.2 — four
heads under a saturating GradNorm controller made the auxiliary term ≈7.8× the main
classification loss at epoch 20, so the fused head, the only path that produces an
evaluation logit, carried ≈11% of the gradient for the first third of training
(`CHANGES.md` §7.1).

### C2 · Native at 256 bands, and provably identical at 40

Two components would otherwise make the full cube either mis-shaped or unaffordable.
Both are solved rather than parameterised around.

**(a) Derived spectral strides.** `spectral_stride_schedule(C, folded_depth)` returns the
smallest power-of-two reduction with $\lceil C/\text{total}\rceil \le$ `folded_depth`,
spending the remainder in stage 1 where one input channel makes it cheapest:

| bands | strides | kernel depths | folded depth | fold input |
|---:|---|---|---:|---:|
| 8 | (1, 1, 1) | (7, 5, 5) | 8 | 512 |
| 40 | **(2, 2, 2)** | **(7, 5, 5)** | 5 | 320 |
| 100 | (4, 2, 2) | (7, 5, 5) | 7 | 448 |
| **256** | **(8, 2, 2)** | **(15, 5, 5)** | **8** | **512** |

The 40-band row is the audited stem tensor for tensor. Three hardcoded halvings — what the
stem used to do — would fold 256 bands at depth 32, i.e. a `Conv2d(2048 → 192)` fold and a
stage-2 cube 6.4× deeper than the design was measured on.

The kernel depth widens with the stride (`kernel_depth`) so **every band reaches at least
one tap**. The bound is $2s - 1$, not $s$: under symmetric $\lfloor k/2 \rfloor$ padding
the last output position reads input indices only up to $C - s + (k-1)/2$, so at $s = 8$ a
9-tap kernel would drop bands 253–255 of the acquired cube — silently, since every shape
still agrees. `tests/unit/test_branch_c_stem.py` asserts the coverage band by band rather
than trusting the arithmetic.

**(b) An exact $O(C^2)$ continuum hull.** The upper concave envelope is the pointwise
maximum over chords, which is $O(C^3)$ written directly: 64,000 chords at $C = 40$ and
**16.8 million** at $C = 256$, with four dense $(C,C,C)$ buffers (570 MB resident) and a
268 MB activation per chunk at batch 128. Because the chord is affine in the right
endpoint's *slope* with a non-negative coefficient for $a \le i$,

$$
\mathrm{chord}(a,b,i) = r_a + (\lambda_i - \lambda_a)\,\frac{r_b - r_a}{\lambda_b - \lambda_a},
$$

maximising over $b \ge i$ is maximising the slope over $b \ge i$ — one reversed
`cummax` — and the envelope is a maximum over the left endpoint alone. The value is then
evaluated from the *selected* endpoints in the original $(1-t)r_a + t r_b$
parameterisation, so the result is **bit-identical** to the chord enumeration, which
`tests/unit/test_masked_ops.py` asserts against a literal transcription of the $O(C^3)$
form.

Verified end to end: `python scripts/capture_golden.py --verify` reports
`v3/logits match (max |Δ| = 0.000e+00)` and `v3/init digests match (306 tensors)`.

### C3 · Wavelength as two mechanisms, and one deliberate absence

`front_end.py` gives the λ-aware components two independent, physically-motivated ways of
using $\lambda$, and the gain-invariant ones none at all:

- **Exact λ-derivatives** — `SpectralDerivatives` builds dense $(C, C)$ first- and
  second-derivative operators from a local weighted least-squares polynomial fit
  (Savitzky–Golay) whose design matrix carries the *actual* wavelength offsets. The
  estimator is exact for polynomials up to the fitted degree no matter how uneven the
  grid is — which is precisely what a `Conv1d` on the index axis cannot be. Zero
  parameters; the operators are persistent buffers, so they are the record of *which*
  wavelength grid a checkpoint was trained on.
- **Continuous-λ convolution kernels** — `LambdaConv1d` (Branch A of the control arm)
  *generates* the weight for the pair (band $i$, neighbour $j$) from a Fourier
  featurisation of $\lambda_j - \lambda_i$ via a small MLP, with the neighbourhood taken
  as the $k$ nearest bands **in** $\lambda$ rather than in index. One kernel function is
  learned once and shared by every band, so its parameter cost is independent of the band
  count.
- **λ-uniform tokenisation** — Branch D pools bands into `model.specf_tokens` equal-width
  *wavelength* windows and biases its attention by the difference in window centre
  wavelengths. The token count is configured **directly**; deriving it from the band count
  (as `num_bands // (specf_patch // 2)` did) made a window's width a function of $k$ —
  15 nm at $k = 40$ and 2.4 nm at $k = 256$ — so "token 3" denoted a different spectral
  region in every arm, contradicting the property λ-uniform tokenisation exists to provide.
- **Deliberate absence** — `SoftIndexBank`'s normalised-difference indices and
  `ContinuumDepths`' hull-removed depths are constructed to be exactly invariant to a
  per-pixel or per-session gain, so they carry **no** wavelength-positional information.

### C4 · One stage, one objective, one schedule

Stages 2 and 3 of the audited three-stage curriculum consumed 65% of an 18.7-hour wall
clock and moved validation macro-F1 by +0.005 — 6.5 samples of a 1,294-sample split,
against a ±0.020 sampling CI. The *only* thing Stage 2 did that Stage 1 did not was
introduce a non-zero angular margin, and a margin is incompatible with mixup, which is why
they were separated in the first place.

The single stage achieves the same transition internally: mixup runs to
`single.mixup_epochs`, then a single global margin warms in over
`margin_warmup_start`…`margin_warmup_end`. One optimiser state, one schedule, no EMA
re-initialisation. Stage hyperparameter count: 69 → 14.

**The three-stage modules are not deleted.** `pipeline=three_stage` still reaches them,
because A8 is the experiment that decides whether the collapse was correct.

---

## 1.4 Evaluation framework

### Split protocol

`data/loaders.py::build_split_bundle` builds one of two protocols, selected by
`cfg.data.split_scheme`, with a module-level seed deliberately decoupled from `cfg.seed`
so overriding the run seed cannot silently re-partition the data.

- **`grouped`** (`configs/data/hsi256_grouped.yaml`, the default) holds out whole
  acquisition bundles. Per class: order the class's groups deterministically, rotate by
  `split_fold`, take $\max(1, \mathrm{round}(m \cdot \texttt{eval\_frac}))$ of them (never
  all) as the class's eval groups, split those into val/test by group when there are ≥ 2
  and by patch when there is 1, then carve `calib_frac` out of the remaining train pool.
  This guarantees the contract regardless of what the later steps can manage: an eval
  group is never a train group.
- **`stratified`** (`configs/data/hsi256_stratified.yaml`) is the patch-level contrast
  arm. `groups.npy` is still loaded — not to *build* the split but to **measure** it: the
  banner reports how many of the 180 bundles cross the train/eval boundary. Expect
  `180 of 180`.

This dataset's two-scans-per-class structure makes full three-way group disjointness
mathematically unreachable — a class with two groups can appear in at most two of three
splits — and `SplitReport` says so explicitly rather than silently approximating.
`assert_protocol_holds` **fails the run** when `grouped` is requested and not realised.

A **calibration split** (`data.calib_frac = 0.15`) additionally separates the split that
fits per-class margins, CDWS weights and oversampling weights from the split that selects
the checkpoint. Full mechanics in `02_DATASET_AND_PREPROCESSING.md` §2.8.

### Band-geometry contract

Before the model is built, `data/mmap_store.py::band_geometry` checks that four
descriptions of the spectral axis are one description — the cube's axis 1, the optional
`band_indices_path`, the wavelength CSV's row count, and `data.num_bands` — and raises
`BandGeometryError` naming the disagreement otherwise. The run then prints which regime it
is in:

```
Spectral: 256 bands — the full acquired cube, no band selection (primary methodology)
Spectral: 40 of 256 bands — REDUCED arm via data.band_indices_path. …
```

and records `band_geometry` and `band_selection` in `results/run.json`, so "this number is
from the full cube" is an artifact rather than a claim.

### Metric rule

**Macro-F1 is the primary metric and the only one that gates a checkpoint save.** Accuracy
is reported alongside but never decides. Every per-epoch evaluation scores *both* the live
model and its EMA shadow and takes $\max(F_1^{\text{live}}, F_1^{\text{ema}})$; evaluation
is forced to fp32 via `autocast(enabled=False)` so a stage's reported F1 never depends on
the AMP state it was called from, and non-finite logits are `nan_to_num`-clamped rather
than raised.

### Reporting rules, enforced by code

1. **Selection never happens on the reported split.** `calib` selects; `val ∪ test` is
   scored once. The run banner prints both.
2. **`val` and `test` are scored together**, being two halves of one held-out bundle.
3. **Mean ± range over folds × seeds. Never a maximum.** A running maximum over ~944
   correlated selection events was worth an estimated +0.042 macro-F1 in the audited run.
4. **Every reported number carries an interval** — a 2,000-resample bootstrap. Sampling
   noise on a ~1,300-patch split is ±0.020 at 95%.
5. **A delta whose interval crosses zero has not been shown to do anything**, and is
   reported that way, in grey, on the forest plot.

### Inference protocol

Final numbers are produced by `engine/stages/final_eval.py`, twice — once with a single
forward pass and once with the 12-view TTA of `05_EXPERIMENTS_AND_ABLATIONS.md` §5.1 —
from whichever weight set (live or EMA) actually won the checkpoint comparison. The run
writes `test_preds_noTTA.npy`, `test_preds_TTA.npy` and `test_targets.npy`.

### Comparative benchmark structure

The repository contains **no external baseline model implementations**, and no external
baseline has been executed. The table below is a *protocol specification with empty result
cells*.

| Model | Input representation | Params | Test Macro F1 | Test Acc | TTA Macro F1 |
|---|---|---|---|---|---|
| SpectralSeedNet | $(256,64,64)$ patch + 8 morphometrics | 3.05 M | *not yet run* | *not yet run* | *not yet run* |
| SpectralQuadNet (control) | $(256,64,64)$ patch + 8 morphometrics | 5.26 M | *not yet run* | *not yet run* | *not yet run* |
| *external baseline #1* | — | — | *not run* | *not run* | *not run* |

Conditions any entry must satisfy to be comparable:

1. **Identical split indices** — the arrays `build_split_bundle` returns under a fixed
   `split_scheme`/`split_fold`, not a re-drawn split.
2. **Identical inputs** — `dataset/patches.npy`, all 256 bands. A model consuming a
   selected subset is a *different benchmark row*, not a baseline for this one, and its
   row must state $k$ and the selection method.
3. **Macro-F1 on `val ∪ test`** as the headline number, with accuracy and weighted-F1
   alongside.
4. **TTA reported separately**, never merged into the single-view number.
5. **Checkpoint selection by `calib` macro-F1**, matching `_pick_best_checkpoint`.

Two internal reference points *do* exist and must not be mistaken for baselines:

- `dataset/band_selection_report.csv` records 5-fold `StratifiedKFold` accuracy of LDA and
  LinearSVC on **spatially-averaged mean spectra** at several band counts — at $k = 40$,
  LDA $0.5916$ / SVC $0.4755$ (SPA ordering). These exist only to choose a band count, on
  the leaky patch-level partition, and the checked-in curve is not itself a demonstrated
  elbow.
- `data/prep/band_selection.py`'s closing console message refers to a *"256-band baseline
  (86.9 % TTA)"*. No artifact in this repository reproduces that figure; it is an in-source
  claim, not a verified result, reported here only for provenance.

---

## 1.5 Verification gates

| Gate | What it pins |
|---|---|
| `tests/unit/test_branch_c_stem.py` | The derived stride schedule at 8/40/100/256 bands; that 40 bands still yields `(2,2,2)`; that **every band reaches at least one stage-1 tap**, checked band by band with a one-hot spectrum |
| `tests/unit/test_masked_ops.py` | That the $O(C^2)$ continuum hull is **bit-identical** to a literal transcription of the $O(C^3)$ chord enumeration; that it runs at batch 64 × 256 bands; gain invariance at the primary band count |
| `tests/unit/test_mmap_store.py` | The band-geometry contract, including the wavelength-vector mismatch no other check catches |
| `tests/unit/test_cutmix.py` | That every shipped data config's augmentation widths equal `band_augmentation_widths(num_bands)` |
| `tests/unit/test_spectral_seed_net.py` | That the primary config *is* 256 bands with no index file; the parameter budget; that both pathways influence the output |
| `tests/regression/test_golden_forward_pass.py` | Eval-mode logits, per-tensor SHA-256 of the initialised state dict (306 tensors), exact Stage-1 epoch-1 loss and post-step weight hashes |
| `tests/regression/test_state_dict_compatibility.py` | The top-level attribute names; the checkpoint bundle schema |
| `tests/regression/test_resume_and_final_eval.py` | Auto-resume detection, sidecar schema, `val_f1`-based selection |
| `tests/unit/test_schedulers.py` | Every LR multiplier and margin value across the full epoch range of every stage |
| `tests/unit/test_config_wiring.py` | Every `cfg.model.*` key is forward-observable, dropout-observable, or has a named reason it cannot be |
| `scripts/check_config_roundtrip.py` | Every `configs/` key resolves to exactly one field, generating `docs/config_migration_table.md` via `--emit-markdown` |

---

## 1.6 The band-selection pathway, in one paragraph

Band selection is retained in full — `spectralquadnet.bandstudy` (12 methods including
evenly-spaced and random nulls × 20 budgets up to the full 256 × 3 proxy families),
`data/prep/band_selection.py` (mRMR + SPA + elbow → a materialised reduced cube),
`scripts/select_bands.py`, `configs/data/ablation/`, and the `data.band_indices_path`
mechanism that slices a k-band arm off the mmap without materialising a reduced cube at
all. None of it runs during a default `python train.py`. It is retained rather than
deleted for the same reason `SpectralQuadNet` is: "how few bands would do?" is a real
question with a real deployment consequence, and deleting the machinery that can answer it
would make the primary path's refusal to reduce an assumption rather than a measured
choice. Ablation **A2** is the gateway, and it takes the primary path as its *reference*
arm. See `07_BAND_SELECTION_PATHWAY.md`.

---

## 1.7 Known non-determinism

A full training run is **not** bit-reproducible, by deliberate choice.
`utils/seed.py::set_seed` leaves `cudnn.benchmark=True` (fast autotuned kernels,
non-deterministic algorithm selection). On a single process, both custom samplers draw
from an unseeded `np.random.default_rng()` per epoch; under DDP,
`ClassBalancedBatchSampler` is given an explicit seed (so every rank composes the identical
global batch before it is sharded — `06_EXECUTION_AND_HARDWARE.md` §6.4), making that
specific configuration deterministic per `(seed, epoch)`.

**The realised augmentation draws are also a function of the worker count.**
`RiceSeedDataset` builds every sample on the host rather than on the training device,
which is what makes `runtime.num_workers > 0` possible at all — a worker process cannot
hand a CUDA tensor back through the queue. The augmentation call order, every profile
probability and every augmentation's distribution are fixed, but the noise tensors come
from the host RNG rather than the accelerator's, and at `num_workers > 0` each worker
seeds its own `torch`/`numpy`/`random` streams from `torch.initial_seed()`
(`loaders.py::seed_worker`). A run is reproducible at a fixed `cfg.seed` **and a fixed
worker count**; it does not reproduce the realised draws of a `num_workers=0` run. Set
`runtime.num_workers=0` if that specific stream is what you need.

What *is* pinned and tested regardless of configuration: weight initialisation, every
scheduler and margin value, and one fixed-seed forward + backward step.
`enable_deterministic_algorithms()` exists for test harnesses and is never used in
training.
