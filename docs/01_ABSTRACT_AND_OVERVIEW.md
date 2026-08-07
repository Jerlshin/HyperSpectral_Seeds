# 1 · Abstract and System Overview

> **Scope of this suite.** Every number, equation, default and contract in these five
> documents is derived from `src/spectralquadnet/`, `configs/`, `train.py`, `tests/` and the
> committed run artifacts in `outputs/output_v12_spa40/`. Where a quantity is *not*
> reproducible from this repository, it is marked as such rather than reported.

| Document | Contents |
|---|---|
| `01_ABSTRACT_AND_OVERVIEW.md` | Problem, contributions, evaluation protocol, recorded results |
| `02_DATASET_AND_PREPROCESSING.md` | Acquisition → segmentation → patches → band selection → data store |
| `03_MODEL_ARCHITECTURE.md` | Branch formulations, fusion, heads, full tensor-shape matrix |
| `04_CURRICULUM_AND_LOSSES.md` | 3-stage curriculum, schedules, loss objectives, optimisation rules |
| `05_EXPERIMENTS_AND_ABLATIONS.md` | TTA inference, diagnostics, telemetry, regression gates |

---

## 1.1 Abstract

`SpectralQuadNet` is a four-branch hyperspectral classifier that assigns one of
$C = 90$ rice-seed varieties to a single segmented seed, observed as a
$40 \times 64 \times 64$ VIS–NIR reflectance patch. The source cubes are the 256-band
VIS–NIR scans of the *RGB and VIS-NIR HSI Data for 90 Rice Seed Varieties* collection
(Zenodo record `3241923`, referenced in `data/prep/config.py::DATA_URL`); an offline
pipeline segments individual seeds, extracts $64\times64$ patches, and reduces
$256 \to 40$ spectral bands by a validated mRMR/SPA selection.

The classification problem is *fine-grained and near-degenerate*: 90 varieties of the same
species, ~96 patches per class, discriminated by sub-percent differences in reflectance
shape rather than by morphology. The design responds to that in three ways:

1. **Four disjoint views of the same patch** — raw regional spectra, background-masked
   per-band statistics, 2-D spatial texture, and a spectral transformer — fused by latent
   cross-attention rather than concatenated, so no single view can dominate.
2. **Wavelength enters the network physically**, not positionally: the encoding is keyed on
   nanometre values, so bands adjacent in $\lambda$ receive similar codes even after band
   selection has made them non-adjacent in index order.
3. **A three-stage curriculum** that separates representation learning (progressive
   augmentation + deep supervision), metric learning (adaptive sub-centre ArcFace), and
   flat-minimum refinement (SAM + greedy SWA), rather than training one objective end to end.

The model has **7,879,333 parameters** (7.88 M, all trainable).

### Recorded performance

All figures below are regenerated from committed artifacts by
`tests/regression/test_resume_and_final_eval.py`. The evaluated network is always the **EMA
shadow** of the checkpoint that `_pick_best_checkpoint` ranks highest by validation macro-F1
— on these artifacts, **Stage 1** (`best_stage1.pth`, epoch 488).

| Split | Model / protocol | Macro F1 | Weighted F1 | Accuracy |
|---|---|---|---|---|
| **Test** (1,294 patches) | 12-view TTA | **0.8933** | 0.8939 | **0.8941** |
| **Test** (1,294 patches) | single view, no TTA | 0.8770 | 0.8776 | 0.8779 |
| Validation (1,294 patches) | Stage 1 checkpoint (`stage1_meta.json`) | 0.8877 | — | 0.8872 |
| Validation | Stage 2 checkpoint (`stage2_meta.json`) | 0.8867 | — | 0.8864 |
| Validation | Stage 3 SWA checkpoint (`stage3_meta.json`) | 0.8745 | — | 0.8748 |

> **The ~87.5 % figure is a validation number, not a test number.** `0.8745 / 0.8748` is the
> Stage-3 SWA model's *validation* macro-F1/accuracy, written by `run_stage3_swa`'s own
> `save_ckpt` call and quoted in `configs/experiment/output_v12_spa40.yaml`. The held-out
> **test** result of the shipped pipeline is `0.8933` macro-F1 with TTA. These are different
> splits *and* different checkpoints; `tests/regression/test_resume_and_final_eval.py`
> documents and pins the distinction. Stage 3 did not beat Stage 2 on this run, which its
> sidecar records verbatim: `"val_f1 did not beat Stage 2; Stage 2 ckpt preferred for eval"`.

On the test split with TTA, no class falls below $F_1 = 0.50$, 23 of 90 classes reach
$F_1 = 1.00$, and the five hardest are classes 49 (0.519), 52 (0.533), 41 (0.538),
51 (0.629) and 37 (0.640).

---

## 1.2 Core contributions

### C1 · Four-branch topology with a shared masked front-end

A single shared spectral-attention block conditions the cube, after which two *pure*
extractors (no parameters, no state) derive the branch inputs; each branch then sees a
structurally different projection of the same seed.

```
x (B,40,64,64)
  └─ MaskedSpectralECA ──────────────────────────► x' (B,40,64,64)
        ├── extract_grid_spectra(4×4) ─► (B,16,40) ─┬─► Branch A  SpectralProfile  ─► (B,256)
        │                                           └─► Branch D  SpecFormer       ─► (B,256)
        ├── masked_spectral_stats     ─► 9×(B,40) ───► Branch B  SpectralStats     ─► (B,256)
        └── x' itself                 ─► (B,40,64,64) ► Branch C  SpatialCNN       ─► (B,256)
                                                            │
                          CrossModalInteraction (Perceiver-style latents) ─► (B,256)
                                                            │
                                        EmbedNet ─► e (B,256)
                                                            │
                        linear_head (Stage 1)  ──or──  arcface_head (Stage 2+)  ─► (B,90)
```

Branches A and D operate on a $4\times4$ grid of **background-masked regional mean spectra**,
so padded pixels never dilute the signal; branch B consumes nine masked per-band statistics;
branch C is the only branch that sees the raw spatial cube. Four auxiliary heads provide deep
supervision, one per branch, with the two spectral branches weighted $2\times$ so their
gradients do not collapse under the spatial branches (§4.3).

### C2 · Physical wavelength positional encoding

`PhysicalWavelengthPE` builds a sinusoidal code whose argument is the **min–max normalised
physical wavelength** $\tilde\lambda_i \in [0,1]$, not the band index $i$:

$$
E_{\mathrm{wl}}[i, j] = \sin(\tilde\lambda_i\,\omega_j), \qquad
E_{\mathrm{wl}}[i, j + d/2] = \cos(\tilde\lambda_i\,\omega_j), \qquad
\omega_j = \exp\!\left(-\,j\,\frac{\ln 10^4}{d/2 - 1}\right)
$$

with $d = 96$ (the 1-D tower width) and $j = 0,\dots,47$. Because band selection removes
$216$ of $256$ bands non-uniformly, index adjacency no longer implies spectral adjacency;
keying on $\tilde\lambda$ restores that correspondence for branches A and B. The table is a
registered **buffer**, so it travels in the checkpoint (`wl_pe_cnn.pe`,
`branch_{a,b}.wl_pe_module.pe`, each $(40, 96)$).

### C3 · Adaptive sub-centre ArcFace

`AdaptiveSubcenterArcFaceHead` gives every class $K = 3$ sub-centres and takes the maximum
cosine over them, so a class with several appearance modes needs no single prototype:

$$
\cos\theta_{i,c} \;=\; \max_{k \in [K]} \; \operatorname{clamp}\!\left(
\hat{\mathbf{e}}_i^{\top}\hat{\mathbf{W}}_{c,k},\; -1{+}10^{-6},\; 1{-}10^{-6}\right)
$$

The additive angular margin is **per class**, recalibrated from validation $F_1$:

$$
M(y_i) \;=\; m_{\text{base}} + m_\Delta\big(1 - \min(F_1^{(y_i)}, 1)\big),
\qquad m_{\text{base}} = 0.35,\; m_\Delta = 0.10
$$

so a class currently at $F_1 = 1.0$ keeps $0.35$ while a fully-failed class is pushed out to
$0.45$. Full formulation, warm-up schedule and the linear-head bootstrap are in §3.5 and §4.2.

### C4 · Three-stage curriculum optimisation

| Stage | Objective family | Head | Sampler | Optimiser |
|---|---|---|---|---|
| 1 | CE/Focal + deep supervision + (P3) SupCon/ProtoNCE | linear | shuffled → hard-class oversampled | AdamW + phase-aware LR |
| 2 | Sub-centre ArcFace (Focal) + SupCon + ProtoNCE | ArcFace | class-balanced, CDWS-weighted | AdamW + SGDR, split head/backbone LR |
| 3 | Focal + SupCon under SAM, greedy SWA | ArcFace | class-balanced, CDWS-weighted | SAM(AdamW) + cyclic LR |

Each stage writes `best_stage{n}.pth` and a JSON sidecar `stage{n}_meta.json`; `train.py`
auto-resumes by probing $3 \to 2 \to 1$ and treats a stage as complete only when **both**
files exist. The checkpoint used for final evaluation is chosen by recorded validation
macro-F1, *not* by stage order.

---

## 1.3 Evaluation framework

### Split protocol

`data/loaders.py::build_splits` performs a two-step stratified partition with
`random_state=42` **hardcoded**, deliberately decoupled from `cfg.seed`, so overriding the
run seed can never silently re-partition the data:

$$
8{,}624 \;\longrightarrow\; \underbrace{6{,}036}_{\text{train, }70\%} \;+\;
\underbrace{1{,}294}_{\text{val, }15\%} \;+\; \underbrace{1{,}294}_{\text{test, }15\%}
$$

Approximately 67 training patches per class; 13–15 test patches per class.

### Metric rule

**Macro-F1 is the primary metric and the only one that gates a checkpoint save**, in all
three stages (`engine/evaluate.py`). Accuracy is reported alongside but never decides. Every
per-epoch evaluation scores *both* the live model and its EMA shadow and takes
$\max(F_1^{\text{live}}, F_1^{\text{ema}})$; evaluation is forced to fp32 via
`autocast(enabled=False)` so a stage's reported F1 never depends on the AMP state it was
called from, and non-finite logits are `nan_to_num`-clamped rather than raised.

### Inference protocol

Final test-set numbers are produced by `engine/stages/final_eval.py`, twice — once with a
single forward pass and once with the 12-view TTA of §5.1 — from the EMA shadow of the
selected checkpoint. The run writes `test_preds_noTTA.npy`, `test_preds_TTA.npy` and
`test_targets.npy`, so any reported metric is recomputable from disk without re-running
inference.

### Comparative benchmark structure

The repository contains **no external baseline model implementations**, and no external
baseline has been executed. The comparison table below is therefore given as a *protocol
specification with empty result cells* — populating it requires running each baseline
under the identical conditions listed.

| Model | Input representation | Params | Test Macro F1 | Test Acc | TTA Macro F1 |
|---|---|---|---|---|---|
| SpectralQuadNet (this work) | $(40,64,64)$ patch | 7.88 M | 0.8770 | 0.8779 | **0.8933** |
| *external baseline #1* | — | — | *not run* | *not run* | *not run* |
| *external baseline #2* | — | — | *not run* | *not run* | *not run* |

Conditions any entry must satisfy to be comparable:

1. **Identical split indices** — the arrays returned by `build_splits` (`random_state=42`),
   not a re-drawn stratified split.
2. **Identical inputs** — `dataset/patches_spa_40b.npy`, the same 40 SPA bands; a model
   consuming all 256 bands or a different subset is a *different benchmark row*, not a
   baseline for this one.
3. **Macro-F1 on the 1,294-patch test split** as the headline number, with accuracy and
   weighted-F1 reported alongside.
4. **TTA reported separately**, never merged into the single-view number.
5. **Checkpoint selection by validation macro-F1**, matching `_pick_best_checkpoint`.

Two internal reference points *do* exist and must not be mistaken for baselines:

- `dataset/band_selection_report.csv` records 5-fold `StratifiedKFold` accuracy of LDA and
  LinearSVC on **spatially-averaged mean spectra** at several band counts — at $k = 40$,
  LDA $0.5916$ / SVC $0.4755$ (SPA ordering). These are cross-validated over the *whole*
  dataset on a 40-dimensional feature vector, not held-out-test numbers on $(40,64,64)$
  patches, and exist only to choose the band count (§2.4).
- `data/prep/band_selection.py`'s closing console message refers to a *"256-band baseline
  (86.9 % TTA)"*. No artifact in this repository reproduces that figure; it is an
  in-source claim, not a verified result, and is reported here only for provenance.

### Verification gates

| Gate | What it pins |
|---|---|
| `tests/regression/test_golden_forward_pass.py` | eval-mode logits ($\text{atol}=10^{-6}$), per-tensor SHA-256 of the initialised state dict, exact Stage-1 epoch-1 loss and post-step weight hashes |
| `tests/regression/test_state_dict_compatibility.py` | all three real checkpoints load `strict=True`; the 14 pinned top-level attribute names; the bundle schema |
| `tests/regression/test_resume_and_final_eval.py` | auto-resume detection, sidecar schema, `val_f1`-based selection, and reproduction of the recorded test metrics |
| `tests/unit/test_schedulers.py` | every LR multiplier and margin value across the full epoch range of all three stages |
| `scripts/check_config_roundtrip.py` | all 81 pre-refactor `CONFIG` keys map 1:1 onto `configs/` with identical values (`docs/config_rename_table.md`) |

### Known non-determinism

A full training run is **not** bit-reproducible, by deliberate choice.
`utils/seed.py::set_seed` leaves `cudnn.benchmark=True` (fast autotuned kernels,
non-deterministic algorithm selection), and both custom samplers draw from an unseeded
`np.random.default_rng()` per epoch. What *is* pinned and tested: weight initialisation,
every scheduler and margin value, and one fixed-seed forward + backward step.
`enable_deterministic_algorithms()` exists for test harnesses and is never used in training.
