# 1 · Abstract and System Overview

> **Scope of this suite.** Every number, equation, default and contract in these six documents
> is derived from `src/spectralquadnet/`, `configs/`, `train.py` and `tests/`. The one exception
> is the recorded performance of the `outputs/output_v12_spa40/` run: `outputs/` is git-ignored
> and that directory is not in this tree, so those figures are a transcribed record rather than
> something a reader can re-derive from a checkout (§5.4). Where a quantity is *not* reproducible
> from this repository, it is marked as such rather than reported.

| Document | Contents |
|---|---|
| `01_ABSTRACT_AND_OVERVIEW.md` | Problem, contributions, evaluation protocol, recorded results |
| `02_DATASET_AND_PREPROCESSING.md` | Acquisition → segmentation → patches → band selection → data store → split protocols |
| `03_MODEL_ARCHITECTURE.md` | Branch formulations, gated bilinear fusion, heads, full tensor-shape matrix |
| `04_CURRICULUM_AND_LOSSES.md` | 3-stage curriculum over one unified head, schedules, loss objectives, optimisation rules |
| `05_EXPERIMENTS_AND_ABLATIONS.md` | TTA inference, diagnostics, telemetry, checkpointing, regression gates, ablation surface |
| `06_EXECUTION_AND_HARDWARE.md` | Entrypoint orchestration, runtime performance knobs, distributed (DDP) training |

---

## 1.1 Abstract

`SpectralQuadNet` is a hyperspectral classifier that assigns one of $C=90$ rice-seed varieties to
a single segmented seed, observed as a $40\times64\times64$ VIS–NIR reflectance patch. The
source cubes are the 256-band VIS–NIR scans of the *RGB and VIS-NIR HSI Data for 90 Rice Seed
Varieties* collection (Zenodo record `3241923`, referenced in `data/prep/config.py::DATA_URL`);
an offline pipeline segments individual seeds, extracts $64\times64$ patches, and reduces
$256\to40$ spectral bands by a validated mRMR/SPA selection.

The classification problem is *fine-grained and near-degenerate*: 90 varieties of the same
species, $\sim$96 patches per class, discriminated by sub-percent differences in reflectance
shape rather than by morphology. The current (Tier-3) design responds to that in four ways:

1. **Four branches, each seeing something the others structurally cannot reconstruct** —
   gain-free SNV spectra and their exact λ-derivatives (Branch A); a learned, gain-invariant
   normalised-difference index bank plus continuum-removed absorption depths (Branch B); the
   only joint spectral–spatial operator in the network, a 3-D stem over the full cube (Branch C);
   and a wavelength-uniform spectral transformer that reads long-range band interactions
   (Branch D) — plus a fifth, non-spectral token carrying eight persisted morphometrics. The five
   are fused by a **gated low-rank bilinear pool**, not concatenation and not attention: an
   independent sigmoid gate per modality (fed each branch's pre-normalisation confidence) plus a
   rank-128 second-order term over every modality pair.
2. **Wavelength enters two branches through two different, physically-motivated mechanisms**,
   and is deliberately absent from a third. Branch A's convolution kernels are *generated* from
   continuous $\Delta\lambda$ by a small Fourier-feature MLP rather than learned per band index,
   so their parameter cost is independent of band count; Branch D tokenises the spectrum into
   equal-width wavelength windows (not equal-count index strides) and biases its attention by the
   *difference* in window centre wavelengths. Branch B is built to be exactly invariant to the
   gain that a per-session illumination or a per-pixel geometric factor would otherwise impose,
   so it deliberately carries no wavelength-positional information at all.
3. **A single classification head, three margins.** Where a two-head design would create a
   discontinuity at every stage boundary, one adaptive sub-centre ArcFace head is shared across
   all three curriculum stages; only the angular margin passed at each call changes — zero
   (a plain cosine classifier) in Stage 1, a warmed global scalar handing over to a per-class
   vector in Stage 2, and a multiplicatively-annealed version of that same vector in Stage 3. The
   per-class margin itself is calibrated from the **sign** of each class's precision/recall gap,
   not from a single F1 score, so classes failing for opposite reasons are pushed in opposite
   directions.
4. **A three-stage curriculum** separating representation learning (progressive augmentation +
   deep supervision), metric learning (class-balanced sampling, contrastive auxiliaries,
   margin calibration), and flat-minimum refinement (SAM/ASAM + greedy, transient-rejecting SWA),
   rather than training one objective end to end.

The model has **5,194,578 parameters** (5.19 M, all trainable) — a $-2.68$ M reduction from the
pre-Tier-3 architecture, dominated by replacing Perceiver-style latent cross-attention fusion
with the gated bilinear pool ($-1.64$ M) and dropping a provably rank-2 statistics branch
($-0.59$ M), partially reinvested in Branch C's new 3-D stem ($+0.54$ M) and the new morphology
token ($+17$ k). Full derivation in `03_MODEL_ARCHITECTURE.md` §3.7.

### Recorded performance

> **Provenance caveat.** The figures below come from the `outputs/output_v12_spa40/` run's
> artifacts (not in this tree — see above), produced under the **pre-Tier-3** architecture: a
> 4-branch model without the morphology token, index bank, 3-D stem or λ-uniform tokenisation
> described above and in `03_MODEL_ARCHITECTURE.md`. Those checkpoints carry a schema
> (`schema_version ≤ 2`) that the current model's `load_state_dict(strict=True)` **refuses to
> load** — Tier 3 changed what three of the four branches *consume*, not merely their parameter
> counts, so no tensor-level remap exists (§3.8). They are reported here as the last complete,
> reproducible three-stage run on record, not as a claim about the current architecture's
> performance. A fresh number requires a new three-stage training run.

The evaluated network is always the checkpoint that `_pick_best_checkpoint` ranks highest by
recorded validation macro-F1 — on these artifacts, **Stage 1** (`best_stage1.pth`, epoch 488).

| Split | Model / protocol | Macro F1 | Weighted F1 | Accuracy |
|---|---|---|---|---|
| **Test** (1,294 patches) | single view, no TTA | 0.8770 | 0.8776 | 0.8779 |
| **Test** (1,294 patches) | 12-view TTA (current, corrected spectral transform) | **0.8889** | — | — |
| Validation (1,294 patches) | Stage 1 checkpoint (`stage1_meta.json`) | 0.8877 | — | 0.8872 |
| Validation | Stage 2 checkpoint (`stage2_meta.json`) | 0.8867 | — | 0.8864 |
| Validation | Stage 3 SWA checkpoint (`stage3_meta.json`) | 0.8745 | — | 0.8748 |

> **The ~87.5 % figure is a validation number, not a test number.** `0.8745 / 0.8748` is the
> Stage-3 SWA model's *validation* macro-F1/accuracy, written by `run_stage3_swa`'s own
> `save_ckpt` call. The held-out **test** result of the shipped pipeline is `0.8889` macro-F1
> with TTA (§5.1 documents the spectral-view fix behind this number; the run's original console
> log reported `0.8933` under a since-corrected transform). Stage 3 did not beat Stage 2 on this
> run, which its sidecar records verbatim: `"val_f1 did not beat Stage 2; Stage 2 ckpt preferred
> for eval"`.

On the test split with TTA, no class falls below $F_1=0.50$, 23 of 90 classes reach $F_1=1.00$,
and the five hardest are classes 49 (0.519), 52 (0.533), 41 (0.538), 51 (0.629) and 37 (0.640) —
per-class figures as originally logged, not re-scored through the corrected TTA transform
(§5.1).

---

## 1.2 Core contributions

### C1 · Four branches plus a morphology token, fused by a gated bilinear pool

A single shared spectral-attention block (`MaskedSpectralECA`) conditions the cube once, before
any branch sees it. Each branch then sees a structurally different, non-reconstructible
projection of the same seed:

```
x (B,40,64,64)
  └─ MaskedSpectralECA ──────────────────────────► x' (B,40,64,64)
        ├── 8×8 grid, SNV + ∂λ,∂²λ    ─► (64B,3,40) ─► Branch A  SpectralProfile  ─► (B,256)
        ├── foreground mean spectrum  ─► (B,40) ───────► Branch B  Index Bank      ─► (B,256)
        ├── x' + mask, full cube      ─► (B,40,64,64) ─► Branch C  SpatialCNN      ─► (B,256)
        └── 4×4 grid, raw spectra     ─► (16B,40) ──────► Branch D  SpecFormer     ─► (B,256)
                                                              morphometrics (B,8) ─► MorphologyEmbed ─► (B,256)
                                                                         │
                          gated low-rank bilinear pool (5 modalities) ─► (B,256)
                                                                         │
                                                       EmbedNet ─► ê (B,256)
                                                                         │
                                              arcface_head (all 3 stages, margin varies) ─► (B,90)
```

Branch B works on the single foreground mean spectrum — background-masked, so padded pixels
never dilute it — while Branch C is the only branch that ever sees the raw spatial cube. Four
auxiliary heads provide deep supervision, one per branch, called only in training mode and
always on the **unmasked** branch embedding, so Tier 3's branch-dropout regularisation
(inverted from the pre-Tier-3 profile — protect the branch nothing else can reconstruct, drop
the ones that can be) never starves a branch of gradient.

### C2 · Wavelength as two mechanisms, and one deliberate absence

`front_end.py` (FE-1) gives Branches A and D two independent, physically-motivated ways of using
$\lambda$, and Branch B none at all:

- **Branch A** — a continuous-wavelength convolution kernel generator $\kappa_\phi$: rather than
  learning one weight per band index, a small Fourier-feature MLP maps the signed offset
  $\lambda_j-\lambda_i$ to a convolution weight, generated once per forward and shared across the
  batch. Cost is independent of band count. Exact λ-derivatives ($\partial_\lambda,
  \partial^2_\lambda$) are additionally computed once via local Savitzky–Golay operators fit on
  the irregular band grid.
- **Branch D** — tokens are pooled into **equal-width wavelength windows**, not equal-count
  index strides, so token $t$ means the same physical spectral region regardless of how many
  bands were selected; attention is additionally biased by the *difference* in window centre
  wavelengths via a learned per-head function, zero for the CLS tokens.
- **Branch B** is built the opposite way on purpose: its normalised-difference indices and
  continuum-removed depths are constructed to be exactly invariant to a per-pixel or per-session
  gain, so it carries **no** wavelength-positional information — gain-invariance and
  wavelength-awareness are different, sometimes competing properties, and Tier 3 assigns each to
  a different branch rather than asking one branch to have both.

### C3 · Adaptive sub-centre ArcFace, signed-margin calibration

`AdaptiveSubcenterArcFaceHead` gives every class $K=3$ sub-centres, pooled by a temperature that
anneals from a soft log-sum-exp to the exact hard maximum within each stage (so no sub-centre can
die before it sees data), plus a KL load-balancing term against a uniform sub-centre assignment:

$$
\cos\theta_{i,c} = \tau\log\sum_k\exp(\cos\theta_{i,c,k}/\tau) \;\xrightarrow{\tau\to0}\; \max_k\cos\theta_{i,c,k}
$$

The per-class additive margin is recalibrated once per stage from the **signed** gap between
recall and precision, not from F1:

$$
M(c) = \operatorname{clip}\big(m_{\text{base}}+m_\Delta(R_c-P_c),\; m_{\min},\; m_{\max}\big)
$$

so an over-claiming class ($R_c>P_c$) gets a larger margin and an under-claiming one gets a
smaller one — two classes tied on F1 can land on opposite sides of $m_{\text{base}}$, which the
F1-driven rule it replaced could never express. A row-normalised confusion matrix additionally
penalises each class's non-target logits toward the classes it is *actually* confused with.
Full formulation in `03_MODEL_ARCHITECTURE.md` §3.5 and `04_CURRICULUM_AND_LOSSES.md` §4.2.

### C4 · Three-stage curriculum over one shared head

| Stage | Objective family | Margin | Sampler | Optimiser |
|---|---|---|---|---|
| 1 | CE/Focal + deep supervision + (Phase 3) SupCon/ProtoNCE | $0$ (cosine classifier) | shuffled → hard-class oversampled | AdamW + phase-aware LR |
| 2 | Sub-centre ArcFace (Focal) + SupCon + ProtoNCE | warmed global → per-class | class-balanced, CDWS-weighted | AdamW + SGDR, split head/backbone LR |
| 3 | Focal + SupCon under SAM/ASAM, greedy SWA | per-class vector, annealed | class-balanced, CDWS-weighted | SAM/ASAM(AdamW) + cyclic LR |

Each stage writes `best_stage{n}.pth` and a JSON sidecar `stage{n}_meta.json`; `train.py`
auto-resumes by probing $3\to2\to1$ and treats a stage as complete only when **both** files
exist. The checkpoint used for final evaluation is chosen by recorded validation macro-F1, *not*
by stage order.

---

## 1.3 Evaluation framework

### Split protocol

`data/loaders.py::build_split_bundle` builds one of two protocols, selected by
`cfg.data.split_scheme`, with a module-level seed deliberately decoupled from `cfg.seed`.
`stratified` (the default, and the protocol every archived checkpoint was trained and selected
on) is a patch-level split that puts every one of the dataset's 107 capture scans in train *and*
in val/test — part of any reported number is therefore scan recognition, not purely variety
recognition. `grouped` holds out whole scans instead, at the cost that this dataset's
two-scans-per-class structure makes full three-way group-disjointness unreachable, which the
split's own report says explicitly rather than silently approximating. A **calibration split**
(`data.calib_frac`) additionally separates the split that fits per-class margins/CDWS
weights/oversampling weights from the split that selects the checkpoint. Full mechanics,
including the archive's specific two-scans-per-class limitation and the two shipped configs
(`spa40_90class.yaml` vs. `spa40_90class_pfix.yaml`), are in `02_DATASET_AND_PREPROCESSING.md`
§2.8.

### Metric rule

**Macro-F1 is the primary metric and the only one that gates a checkpoint save**, in all three
stages. Accuracy is reported alongside but never decides. Every per-epoch evaluation scores
*both* the live model and its EMA shadow and takes $\max(F_1^{\text{live}}, F_1^{\text{ema}})$;
evaluation is forced to fp32 via `autocast(enabled=False)` so a stage's reported F1 never
depends on the AMP state it was called from, and non-finite logits are `nan_to_num`-clamped
rather than raised.

### Inference protocol

Final test-set numbers are produced by `engine/stages/final_eval.py`, twice — once with a single
forward pass and once with the 12-view TTA of `05_EXPERIMENTS_AND_ABLATIONS.md` §5.1 — from
whichever weight set (live or EMA) actually won that stage's checkpoint comparison. The run
writes `test_preds_noTTA.npy`, `test_preds_TTA.npy` and `test_targets.npy`, so any reported
metric is recomputable from disk without re-running inference.

### Comparative benchmark structure

The repository contains **no external baseline model implementations**, and no external
baseline has been executed. The comparison table below is therefore given as a *protocol
specification with empty result cells* — populating it requires running each baseline under the
identical conditions listed.

| Model | Input representation | Params | Test Macro F1 | Test Acc | TTA Macro F1 |
|---|---|---|---|---|---|
| SpectralQuadNet (Tier 3, current) | $(40,64,64)$ patch + 8 morphometrics | 5.19 M | *not yet run* | *not yet run* | *not yet run* |
| SpectralQuadNet (pre-Tier-3, archived) | $(40,64,64)$ patch | 7.88 M | 0.8770 | 0.8779 | 0.8889 |
| *external baseline #1* | — | — | *not run* | *not run* | *not run* |
| *external baseline #2* | — | — | *not run* | *not run* | *not run* |

Conditions any entry must satisfy to be comparable:

1. **Identical split indices** — the arrays `build_split_bundle` returns under a fixed
   `split_scheme`/`split_fold`, not a re-drawn split.
2. **Identical inputs** — `dataset/patches_spa_40b.npy`, the same 40 SPA bands; a model
   consuming all 256 bands or a different subset is a *different benchmark row*, not a baseline
   for this one.
3. **Macro-F1 on the 1,294-patch test split** as the headline number, with accuracy and
   weighted-F1 reported alongside.
4. **TTA reported separately**, never merged into the single-view number.
5. **Checkpoint selection by validation macro-F1**, matching `_pick_best_checkpoint`.

Two internal reference points *do* exist and must not be mistaken for baselines:

- `dataset/band_selection_report.csv` records 5-fold `StratifiedKFold` accuracy of LDA and
  LinearSVC on **spatially-averaged mean spectra** at several band counts — at $k=40$, LDA
  $0.5916$ / SVC $0.4755$ (SPA ordering). These exist only to choose the band count, and — per
  `02_DATASET_AND_PREPROCESSING.md` §2.4 — the checked-in curve is not itself a demonstrated
  elbow; it terminates at its own chosen $k$.
- `data/prep/band_selection.py`'s closing console message refers to a *"256-band baseline
  (86.9 % TTA)"*. No artifact in this repository reproduces that figure; it is an in-source
  claim, not a verified result, reported here only for provenance.

### Verification gates

| Gate | What it pins |
|---|---|
| `tests/regression/test_golden_forward_pass.py` | eval-mode logits, per-tensor SHA-256 of the initialised state dict (306 tensors, `golden/v3/`), exact Stage-1 epoch-1 loss and post-step weight hashes, against the schema-v3 architecture |
| `tests/regression/test_state_dict_compatibility.py` | the 14 top-level attribute names; the schema-v3 bundle schema; the v1/v2 → v3 refusal (`SchemaTooOldError`) |
| `tests/regression/test_resume_and_final_eval.py` | auto-resume detection, sidecar schema, `val_f1`-based selection |
| `tests/unit/test_schedulers.py` | every LR multiplier and margin value across the full epoch range of all three stages |
| `tests/unit/test_config_wiring.py` | every `cfg.model.*` key is either forward-observable, dropout-observable, or has a named reason it cannot be — no known dead model config key |
| `scripts/check_config_roundtrip.py` | all 81 pre-refactor `CONFIG` keys map 1:1 onto `configs/`, with identical values except those recorded in `INTENDED_VALUE_CHANGES` (`docs/config_rename_table.md`, generated by `--emit-markdown`) |

**Two of these gates are currently red, and both for the same reason.** Three Stage-1/2 budget
knobs were retuned in the shipped configs — `stage1.epochs` $600\to400$, `stage1.patience`
$160\to50$, `stage2.patience` $80\to30$ — without the two artifacts that bake those values in
being updated with them:

| Red gate | Symptom | Cause |
|---|---|---|
| `scripts/check_config_roundtrip.py` | exits non-zero on three "value drift" lines | the three knobs are not declared in the script's `INTENDED_VALUE_CHANGES`. The mapping itself is unaffected: each key still resolves to exactly one field, which is why `--emit-markdown` still regenerates `docs/config_rename_table.md` cleanly — the table reports the *current* composed values. |
| `test_golden_forward_pass.py::test_stage1_epoch_{loss,weights}_matches_golden` | epoch-1 loss $23.0805\to23.0653$; post-step weight hash differs | `golden/v3/stage1_epoch1_loss_seed42.json` was captured at `stage1.epochs = 600`, and `_aux_loss_weight` reads `progress = ep / total_ep`, so epoch 1's auxiliary weight is a function of the stage budget. Re-running the identical capture with `stage1.epochs=600` reproduces the golden loss and both hashes **bit-exactly**, so nothing about the model's numerics moved — only a schedule input did. |

The other two halves of the golden gate — eval-mode logits and the initialised state-dict hashes
— are unaffected and pass, since neither depends on the stage budget. The retuned values are what
`04_CURRICULUM_AND_LOSSES.md` documents, because they are what the shipped configs run.

### Known non-determinism

A full training run is **not** bit-reproducible, by deliberate choice.
`utils/seed.py::set_seed` leaves `cudnn.benchmark=True` (fast autotuned kernels,
non-deterministic algorithm selection). On a single process, both custom samplers draw from an
unseeded `np.random.default_rng()` per epoch; under DDP, `ClassBalancedBatchSampler` is given an
explicit seed (so every rank composes the identical global batch before it is sharded —
`06_EXECUTION_AND_HARDWARE.md` §6.4), making that specific configuration deterministic per
`(seed, epoch)`.

**The realised augmentation draws are also a function of the worker count.** `RiceSeedDataset`
builds every sample on the host rather than on the training device — the change that removed the
per-sample H2D copy and made `runtime.num_workers > 0` possible at all, since a worker process
cannot hand a CUDA tensor back through the queue. The augmentation call order, every profile
probability and every augmentation's distribution are unchanged, but the noise tensors now come
from the host RNG rather than the accelerator's, and at `num_workers > 0` each worker seeds its
own `torch`/`numpy`/`random` streams from `torch.initial_seed()` (`loaders.py::seed_worker`). A
run is reproducible at a fixed `cfg.seed` **and a fixed worker count**; it does not reproduce the
realised draws of a `num_workers=0` run. Set `runtime.num_workers=0` if that specific stream is
what you need.

What *is* pinned and tested regardless of configuration: weight initialisation, every scheduler
and margin value, and one fixed-seed forward + backward step.
`enable_deterministic_algorithms()` exists for test harnesses and is never used in training.
