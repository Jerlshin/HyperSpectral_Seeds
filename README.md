# SpectralSeedNet — acquisition-aware HSI and RGB rice-variety research

90 rice varieties · 8,624 paired kernels · 180 scans · acquired 256 bands, common valid 215 · HSI 64×64 and foreground RGB 224×224.
Source: [Zenodo 3241923](https://zenodo.org/records/3241923) (Vu et al., Strathclyde).

```bash
pip install -e ".[tracking,figures]"
python train.py
```

Training on Kaggle's two T4s: [§10 · Kaggle — GPU T4 x2](#kaggle--gpu-t4-x2).
**Current reference — SeedNet v5** (S16, D35): the lean network under R1 — grouped 0.571, stratified 0.745 macro-F1 at
3 seeds. Until the default switch lands (S17 part 1, D36) it is selected with the overrides
`single.mixup_epochs=30 single.arcface_m=0.0 single.margin_warmup_start=31 single.margin_warmup_end=31 grad_clip=50.0
single.epochs=200 single.patience=40 model.spectral_descriptor=snv_morph 'model.spatial_tail_strides=[2,2,2,1]'
model.cbam_min_hw=3`.
**Completed — [S22](docs/research/studies/S22_complementary_v5/results.md):** seed 0 on both
corrected folds under explicit compute amendments. v5 F1 .558909; equal RGB fusion
.601595, with gains on both folds but no supported cross-session gain.
**Completed learned development — [S23–S25](docs/research/studies/S25_head_seed_screen/results.md):**
a 23,514-parameter additive head over frozen HSI/RGB, mean F1 .604054 across three
head seeds on encoder0. Its gain over matched single-view fusion is .010522;
advantage over stronger TTA fusion remains uncertain. S24 branch controls did not
qualify for simplification. [S26](docs/research/studies/S26_tta_anchor/results.md)
rejected a fixed TTA-anchor substitution on calibration without new test scores.
**Next proposed — [S27](docs/research/studies/S27_tta_trained_head/README.md):** train
the same head against its intended TTA anchor, seed0 both corrected folds first.
No experiment is running; final independent encoder-seed confirmation remains conditional.
S17/S18 remain reserved historical work.

**Completed RGB phase:** [S20](docs/research/studies/S20_rgb_pathway/README.md) establishes
all 8,624 paired crops/masks and compact full spectra; [S21](docs/research/studies/S21_complementary_rgb/README.md)
runs 24 CPU arms on exhaustive complementary folds. RGB probe F1 **.451**, HSI32 **.535**,
equal fusion **.592**. RGB is informative; reliable cross-session gains over v5 are
still unproven. A coverage audit found the historical two grouped folds repeated some
scans; old results are preserved, new studies use explicit corrected row plans.

Start/resume at [RESEARCH_PROGRESS.md](docs/research/RESEARCH_PROGRESS.md).
[Data/cache reproduction](docs/research/studies/S20_rgb_pathway/reproduce.md) documents
`prepare_multimodal.py`, offline DINO caching and immutable CPU runners. Install the
`prep` extra for preprocessing. Raw data, model weights and large derived arrays stay
local and ignored; compact scientific evidence, predictions and figures are tracked.

The engineering narrative below includes the project's earlier no-band-selection
rationale. Current measured inputs are white-tile reflectance (215 common valid
bands); v5 uses uniform430 k32 and the multimodal builder retains full compact
measurements. The research log governs current claims and experiment priority.

---

## 1 · Research objective and study design

Two questions, in order. The second is only answerable once the first is settled.

**Q1 — How much of rice-seed HSI classification performance on this dataset is
variety recognition, and how much is acquisition recognition?**

The source dataset images each variety as **two class-pure bundles of 48 kernels**,
each bundle a tray of one single variety. An earlier run on this repository split at
the **patch** level, so all **180** acquisition bundles appeared in training *and* in
evaluation. A model that learns "this tray's residual radiometric signature ⇒ class X"
scores correctly. The 0.847 macro-F1 that produced was a mixture of variety recognition
and acquisition-bundle recognition, and the mixing ratio had never been measured. No
published work on this dataset that the accompanying audit ([`CHANGES.md`](CHANGES.md))
could access measures it either.

The headline artifact of this project is therefore a **gap**, not an accuracy:
`F1_stratified − F1_grouped`, produced by `scripts/run_protocol.py`.

**Q2 — What does the acquired spectrum support?**

Q1 is a question about the *protocol*. Q2 is a question about the *representation*, and
answering it requires not having quietly discarded most of the representation first.
This is why the primary pipeline trains on the **complete 256-band cube**.

### Expect these consequences

- The **grouped** (leave-one-acquisition-bundle-out) number will be **lower** than the
  patch-level one, probably substantially. That is the correct direction — the previous
  number was measuring something else.
- Published work on this exact dataset reports 92.73–96.17% precision
  ([Taheri et al. 2024](https://doi.org/10.1007/s12652-023-04716-4)) without stating a
  bundle-disjoint protocol. Those figures are comparable to this project's **stratified**
  arm, not its grouped arm.
- Chasing >95% by keeping the patch-level split would reproduce the field's error rather
  than correct it. See `CHANGES.md` Q11.

### Three constraints that belong in the paper, not a footnote

- Training sees **one** acquisition bundle per class, so there is **zero within-class
  acquisition variance in training**. The model cannot learn acquisition invariance
  because it never observes two acquisitions of one class. A data-collection ceiling,
  not a method limitation.
- **Two folds is the maximum.** There is no third bundle.
- `val` and `test` are two halves of the same held-out bundle and are therefore *not*
  independent of each other. They are scored together, once.

---

## 2 · The no-band-selection methodology

**The primary pipeline reads every band the instrument acquired. There is no band
selection, no PCA, and no dimensionality reduction of any kind between
`dataset/patches.npy` and the model's first parameter.**

`scripts/prepare_dataset.py` writes the 256-band cube; `python train.py` trains on it.
Nothing runs in between.

### Why the reduction was removed from the default path

Both previously-shipped reductions — the 40-band SPA subset and the per-fold 100-band
mRMR one — record `"demonstrable": false` in their own elbow files. Each accuracy curve
**terminates at its own chosen k**, so the "98% of peak" elbow criterion is satisfied
vacuously: the peak it is measured against is the peak of a truncated curve
(`CHANGES.md` M-14). Neither k was ever chosen by an experiment that could have returned
a different answer.

A study whose stated question is *what rice-variety information VIS-NIR hyperspectral
imaging carries* cannot begin by discarding 84% of the spectrum on an undemonstrated
elbow. So the default reads all 256 bands, and band reduction becomes a measured
question (§5) instead of an inherited assumption.

Every run states which it is, on its own first screen:

```
Spectral: 256 bands — the full acquired cube, no band selection (primary methodology)
```

and the same fact is written into `results/run.json` as `band_geometry` and
`band_selection`, so "this number is from the full cube" is an artifact rather than a
claim. `spectralquadnet.data.band_geometry` fails the run at startup if the cube, the
wavelength CSV and `data.num_bands` ever disagree.

---

## 3 · The architecture on the 215-band reflectance cube

`SpectralSeedNet` — **3,003,412 parameters** on the 215-band primary input (3,052,682 at 256), two pathways over the full cube.

```
x (B, 256, 64, 64)  +  mask (B, 1, 64, 64)  +  morph (B, 8)
 │
 └─ MaskedSpectralECA(256) ──────────────────────────────► x' (B, 256, 64, 64)
      │                                                        [10 parameters]
      ├── x' ⊙ m ─────────────────► SpatialPath ────────────► (B, 256)
      │      3-D stem: (15,3,3)/s8 → (5,3,3)/s2 → (5,3,3)/s2
      │      256 → 32 → 16 → 8 spectral, 64×64 → 64×64 → 32×32 → 16×16 spatial
      │      fold 64·8 = 512 → 192 channels, then ResBlock2D/CBAM ×4
      │
      └── masked mean spectrum x̄ (B, 256) ──► SpectralPath ─► (B, 256)
             SoftIndexBank(64) ‖ ContinuumDepths(16)
             ‖ snv(x̄)(256) ‖ D₁(256) ‖ D₂(256) ‖ morph(8)   = 856
             → LayerNorm(856) → MLP(856 → 256 → 256)
                                    │
              concat (B, 512) → Dropout → Linear → LayerNorm → EmbedNet → ê (B, 256)
                                    │
                    AdaptiveSubcenterArcFace(K=1) ──────────► (B, 90)
                    aux head on the spatial path (weight 0.2, fixed)
```

| Component | Parameters | Share |
|---|---:|---:|
| `spatial` — 3-D stem + CBAM/ResBlock tail | 2,268,662 | 74.32% |
| `spectral` — index bank, hull, SNV/∂λ, morphometrics | 320,688 | 10.5% |
| `embed_net` | 263,936 | 8.6% |
| `fuse` | 131,840 | 4.3% |
| `aux_head_spatial` | 44,506 | 1.5% |
| `arcface_head` (K=1, 90 classes) | 23,040 | 0.8% |
| `se` — `MaskedSpectralECA` | 10 | — |

### What "native at 256" means concretely

Two components would otherwise make a full cube either mis-shaped or unaffordable.
Both are **solved**, not parameterised around — and both reduce to the previously-audited
40-band behaviour exactly, which is what keeps the reduced arms comparable.

**(a) The 3-D stem's spectral strides are derived from the band count.**
`spectral_stride_schedule(C, folded_depth)` returns the smallest power-of-two reduction
with `ceil(C / total) ≤ folded_depth`, spending the remainder in stage 1 — one input
channel, sixteen output channels, so it is the cheapest place to spend it.

| bands | strides | folded depth | fold input |
|---:|---|---:|---:|
| 8 | (1, 1, 1) | 8 | 512 |
| 40 | **(2, 2, 2)** | 5 | 320 |
| 100 | (4, 2, 2) | 7 | 448 |
| **256** | **(8, 2, 2)** | **8** | **512** |

Three hardcoded halvings — what the stem used to do — folds 256 bands at depth 32, i.e.
a `Conv2d(2048 → 192)` fold and a stage-2 cube 6.4× deeper than the design was measured
on. That is a reduced-band stem carrying a full cube.

The kernel depth widens with the stride (`kernel_depth`) so **every band reaches at least
one tap**. The bound is `2s − 1`, not `s`: under symmetric `k // 2` padding the last
output position reads input indices only up to `C − s + (k−1)/2`, so at `s = 8` a 9-tap
kernel drops bands 253–255 of the acquired cube — silently, since every shape still
agrees. `tests/unit/test_branch_c_stem.py` asserts the coverage band by band.

**(b) The continuum hull is an exact O(C²) suffix maximum.**
The upper concave envelope is the pointwise maximum over chords, which is O(C³) written
directly: 64,000 chords at C = 40 and **16.8 million** at C = 256, with four dense
`(C, C, C)` buffers — 570 MB resident — and a 268 MB activation per chunk at batch 128.
Because the chord is affine in the right endpoint's slope with a non-negative
coefficient, maximising over `b ≥ i` is one reversed `cummax`:

```
hull(r)ᵢ = max( rᵢ , max_{a ≤ i} chord(a, b*(a,i), i) ),   b*(a,i) = argmax_{b ≥ i} slope(a,b)
```

The value is then evaluated from the *selected* endpoints in the original
`(1−t)·r_a + t·r_b` parameterisation, so the result is **bit-identical** to the chord
enumeration — asserted in `tests/unit/test_masked_ops.py` against a literal transcription
of the O(C³) form, not against a refactor of itself.

### What every branch sees, and why

- **Spatial path** — the only joint spectral–spatial operator in the network. Its first
  kernel spans 15 bands × 3×3 pixels at once, so "this absorption feature, in this part
  of the seed" is in its hypothesis class. The persisted fill map re-zeros the padded
  region after every stage, so a padded pixel stays exactly zero however deep the stack
  goes and the CNN can never learn the frame instead of the seed.
- **Spectral path** — the foreground-masked mean spectrum at full resolution, plus:
  64 learned normalised-difference indices (exactly invariant to a per-pixel or
  per-session gain), the 16 deepest continuum-removed absorption depths (gain-free for
  the same reason), the SNV spectrum and its exact first and second λ-derivatives from
  Savitzky–Golay operators fitted on the **irregular** band grid, and the eight
  morphometrics — the only non-spectral evidence in the model, entering **once**.
- **Wavelength** is a first-class axis, not a band index. The derivative operators are
  built from the actual nanometre offsets, so they are exact on polynomials of the fitted
  degree regardless of band spacing; a `Conv1d` over the index axis cannot be.

### The retained four-branch control

`SpectralQuadNet` (5,242,710 parameters at 215 bands) adds Branch A (a
continuous-λ-kernel spectral profile tower over an 8×8 grid), Branch D (a λ-uniform
SpecFormer, window count set directly by `model.specf_tokens`), a rank-128 gated bilinear
fusion over five modalities, K=3 sub-centres and four auxiliary heads. It is **kept
unchanged** and is the control arm for A3/A4/A5/A8 — deleting the thing an ablation
exists to falsify would reproduce the exact defect this revision corrects.

Full derivation: [`docs/03_MODEL_ARCHITECTURE.md`](docs/03_MODEL_ARCHITECTURE.md).

---

## 4 · Data and preprocessing pipeline

```
Zenodo 3241923 (ENVI cubes + RGB)
   │  scripts/prepare_dataset.py           [prep extra; 17 GB zip, ~10 min, ~30 GB]
   ├─ download + extract
   ├─ radiometry            reflectance: each scan ÷ its in-scene Spectralon tile;
   │                        608–706 nm dropped (the tile clips there in sessions 0–7)
   ├─ segmentation          Otsu + connected components → one component per kernel
   ├─ morphometrics         8 descriptors per kernel
   └─ patch extraction      64×64 crops, background exactly zero
   │
   ├─ dataset/patches.npy      (8624, 215, 64, 64) float32   30.4 GB   ← the model's input
   ├─ dataset/labels.npy       (8624,)             int64     variety index 0…89
   ├─ dataset/groups.npy       (8624,)             int64     acquisition-bundle id
   ├─ dataset/masks.npy        (8624, 64, 64)      float16   fill-map alpha ∈ [0,1]
   ├─ dataset/morphology.npy   (8624, 8)           float32   size/shape, unstandardised
   ├─ dataset/gain.npy         (8624, 2, 64, 64)   float32   per-pixel (mean, sd) along λ
   ├─ dataset/wavelengths.csv  215 rows            383.2 … 1006.5 nm, 608.0–705.8 absent
   └─ dataset/radiometry.json, white_tiles.csv, white_spectra.npz   tile QC
   │
   │  train.py  →  DataStore (mmap, read-only, process-wide singleton)
   │               band_geometry() — cube ⇔ wavelengths ⇔ num_bands must agree
   │               build_split_bundle() — grouped, fold 0, calib 0.15
   │               RiceSeedDataset — per-sample augmentation, on the host
   └─ model
```

**`gain.npy` is never a model input.** That is deliberate rather than an oversight: it is
the per-pixel brightness the SNV divided out, which is also the strongest single carrier
of acquisition-bundle identity. `spectralquadnet.experiments.leakage` reads it to
*measure* how much bundle identity the pipeline could exploit.

**Zero-RAM.** The cube is `np.load(..., mmap_mode="r")` and only `__getitem__` touches it,
copying exactly one `(256, 64, 64)` patch. Workers re-open the mapping rather than
receiving a pickled (materialising) copy.

### Augmentation, and why its widths are fractions

Two augmentations are expressed in *bands*, and a band is not a fixed quantity of
spectrum. `band_augmentation_widths(C)` derives both from a fixed share of the axis, so
the primary path and every band-selection arm run the *same* augmentation:

| | fraction | at 256 bands | at 40 bands |
|---|---:|---:|---:|
| `cutmix_bands` — same-class spectral CutMix window | 0.20 | **51** | 8 |
| `max_cutout_bands` — spectral cutout | 0.075 | **19** | 3 |

Same-class CutMix (spectral window, or a 24×24 spatial paste from another seed of the
same class) is **label-preserving**, so unlike mixup it composes with the angular-margin
objective. `noise_std` and the D₄ dihedral transform are band-count independent.

### Split protocol

`data/loaders.py::build_split_bundle`, seeded by a module-level literal deliberately
decoupled from `cfg.seed`.

- **`grouped`** (default) — per class: order the class's groups deterministically, rotate
  by `split_fold`, hold out `max(1, round(m · eval_frac))` of them (never all), split
  those into val/test by group when there are ≥ 2 and by patch when there is 1, then
  carve `calib_frac` out of the remaining train pool.
- **`stratified`** — the patch-level contrast arm. `groups.npy` is still loaded, not to
  build the split but to **measure** it: the banner reports how many bundles cross the
  boundary. Expect `180 of 180`.

`SplitReport` names every place group disjointness could not be achieved rather than
silently approximating, and `assert_protocol_holds` **fails the run** if `grouped` was
requested and not realised.

Full mechanics: [`docs/02_DATASET_AND_PREPROCESSING.md`](docs/02_DATASET_AND_PREPROCESSING.md).

---

## 5 · The band-selection pathway — retained, separate, off by default

Band selection is a real research question with a real deployment consequence: a
multispectral instrument costs a fraction of a hyperspectral one. So the machinery is
**kept and kept working** — it is simply not on the primary path, because deleting it
would make the primary path's refusal to reduce an assumption rather than a measured
choice.

Everything band-selection-related is reachable only by explicit opt-in:

| | What it is |
|---|---|
| `src/spectralquadnet/bandstudy/` | The experiment: 12 methods (including **evenly-spaced** and **random** nulls) × 20 budgets **up to the full 256** × 3 proxy families, under the same grouped protocol and the same split builder the training runs use. |
| `src/spectralquadnet/data/prep/band_selection.py` | The build step: mRMR + SPA + cross-validated elbow → a materialised reduced cube. |
| `scripts/select_bands.py` | Its CLI. Optional; a default `python train.py` never needs it. |
| `configs/data/ablation/` | `spa40_grouped`, `spa40_stratified`, and the frozen `spa40_audited` replica. |
| `data.band_indices_path` | The cheap mechanism: a `.npy` of band indices sliced off the mmap as each patch is read, so a k-band arm costs a config change instead of a 14 GB reduced cube. |
| **Ablation A2** | The gateway experiment, with the primary path as its **reference** arm. |

```bash
python -m spectralquadnet.bandstudy.cli list      # the plan and its cost; runs nothing
python -m spectralquadnet.bandstudy.cli all       # the complete analysis, ~2-3 h, resumable
python -m spectralquadnet.bandstudy.cli neural    # print the confirmation arms, spend nothing
python -m spectralquadnet.bandstudy.cli neural --execute
```

Selectors see training rows only; every decision reads `calib`; `val ∪ test` is reachable
only from the opt-in `confirm` stage, which refuses to run before a recommendation
exists. Produces `outputs/band_study/REPORT.md`.

Run a reduced arm directly:

```bash
python train.py data=ablation/spa40_grouped                 # the shipped 40-band subset
python train.py \
  data.band_indices_path=outputs/band_study/select/fold0/mrmr_k64_bands.npy \
  data.wavelength_path=outputs/band_study/select/fold0/mrmr_k64_wavelengths.csv \
  data.num_bands=64 data.cutmix_bands=13 data.max_cutout_bands=5
```

Either prints `REDUCED arm via data.band_indices_path` at startup and records
`"band_selection": true` in its results JSON.

Details: [`docs/07_BAND_SELECTION_PATHWAY.md`](docs/07_BAND_SELECTION_PATHWAY.md).

---

## 6 · Training, losses, evaluation, checkpointing, reproducibility

### The default curriculum — one stage

`configs/single/one_stage.yaml`. Stages 2 and 3 of the audited three-stage curriculum
consumed 65% of a 18.7-hour wall clock and moved validation macro-F1 by +0.005 — 6.5
samples of a 1,294-sample split, against a ±0.020 sampling CI.

| | |
|---|---|
| Epochs | 150, early stop patience 25 on **calib** macro-F1 |
| Loss | CE + label smoothing 0.10 → 0.04 (linear) + w_aux(t) × aux CE on the spatial path — **applied 0.65 → 0.25** (`single.aux_weight_schedule=legacy`, the default and what every run so far trained with), not the fixed 0.2 earlier versions of this table stated (S10 F54); `=fixed` applies `model.aux_head_weight` |
| Mixup | α = 0.35, epochs 1–110 |
| Head | ArcFace K=1, margin 0 → 0.30 warmed over epochs 111–130 — *after* mixup stops, because a margin and mixup are mutually exclusive by construction |
| Sampler | plain shuffled, batch 128 (classes are already 91–96 each) |
| Augmentation | one `medium` profile throughout + D₄ + same-class CutMix |
| Optimiser | AdamW, lr 5e-4 → 5e-6 cosine, 5-epoch warm-up, wd 2e-4 |
| Clipping | per-parameter-group, threshold 5.0, with clip-fraction telemetry; groups per `clip_partition` (default `legacy` — on this model `fuse` clips with the backbone; S11 D22) |
| Precision | bf16 autocast (fp16 + GradScaler under `runtime=kaggle_t4x2`); **TF32 off**; evaluation forced to fp32 |
| Fit telemetry | clean fit on a fixed 1,000-kernel training subset (eval mode, no aug, margin 0, plain CE), live + EMA → `fit/*` and `clean_fit.json`; `train/acc_dominant`, `train/acc_plain`, `sched/aux_weight_applied` (S10 P0.2–P0.3) |
| EMA | d_max 0.999, no re-initialisation |

`pipeline=three_stage` still reaches the audited Stage 1 → 2 → 3 curriculum, because A8
is the experiment that decides whether the collapse was right.

### Evaluation protocol — enforced by code, not convention

1. **Selection never happens on the reported split.** `calib` selects; `val ∪ test` is
   scored once. The run banner prints both.
2. **`val` and `test` are scored together** — two halves of one held-out bundle.
3. **Mean ± range over folds × seeds. Never a maximum.** A running maximum over ~944
   correlated selection events was worth an estimated +0.042 macro-F1 in the audited run.
4. **Every reported number carries an interval** (2,000-resample bootstrap).
5. **A delta whose interval crosses zero has not been shown to do anything** — and is
   reported that way, in grey, on the forest plot.

Macro-F1 is the primary metric and the only one that gates a checkpoint save. Every
per-epoch evaluation scores both the live model and its EMA shadow and takes
`max(F1_live, F1_ema)`. Final numbers are produced twice — once single-pass and once with
the 12-view TTA (8 dihedral + 4 spectral-gain views about the foreground mean) — and both
are written to disk with their predictions **and float16 logits** (also for `calib`), so any
reported metric, calibration or margin analysis is recomputable without re-running inference.
Under DDP every evaluation is **de-duplicated**, so each kernel is scored exactly once.
`results/run.json` records the git commit (+ dirty flag), the environment and the training
regime as applied (`docs/05` §5.1).

### Checkpointing and resume

Each stage writes `best_stage{n}.pth` plus a JSON sidecar `stage{n}_meta.json`. A stage
counts as complete only when **both** exist; the pipeline auto-resumes by probing 3 → 2 →
1. The single-stage curriculum also writes its **full training state** — weights, EMA and
its update count, optimiser, scheduler, loss scaler, early-stopping counters, RNG — to
`last_stage1.pth` every `runtime.checkpoint_every` epochs (default 1), on a background
thread, with a `last_stage1.json` sidecar marked `finished: false` from the moment training
starts. A run cut off mid-stage therefore **resumes at the next epoch** instead of being
taken for a finished one and reported half-trained (`engine/resume.py`). Every bundle records its `arch` and `schema_version`, and `load_ckpt` **refuses** a
cross-architecture load rather than matching two-thirds of the tensors by coincidence of
naming. The checkpoint used for final evaluation is chosen by recorded validation
macro-F1, *not* by stage order. Since S13 every best checkpoint and its sidecar are written
**atomically** (temporary file + `os.replace`) and every stage ends with a **barrier**, so under
DDP no rank reloads a checkpoint before rank 0 has finished writing it — the race that left one
S11 cell unscored when its best epoch was its last (S12 F71a, S13 F72).

### Reproducibility, stated exactly

Pinned regardless of configuration: weight initialisation, every scheduler and margin
value across the full epoch range, and one fixed-seed forward + backward step.

Not bit-reproducible, by deliberate choice: `cudnn.benchmark=True` autotunes kernels, and
the samplers draw from an unseeded per-epoch RNG on a single process. **The realised
augmentation draws are also a function of the worker count** — a run reproduces at a
fixed `cfg.seed` *and* a fixed worker count. Set `runtime.num_workers=0` if you need the
single-stream draws. `cfg.runtime` holds every throughput knob and none of them may
change a reported number; the two that would (`allow_tf32`, `channels_last`) are off by
default, and `amp_dtype` is printed in the startup banner precisely because it *is* part
of what a number means.

Details: [`docs/04_CURRICULUM_AND_LOSSES.md`](docs/04_CURRICULUM_AND_LOSSES.md),
[`docs/05_EXPERIMENTS_AND_ABLATIONS.md`](docs/05_EXPERIMENTS_AND_ABLATIONS.md).

---

## 7 · Module and package structure

```
configs/                      Hydra composition
  data/                       refl215_grouped (PRIMARY) | refl215_stratified
    ablation/                 spa40_{grouped,stratified,audited}, u430k32_{grouped,stratified}
                              (pre-sliced 32-band cube) — reduced arms only
  model/                      seed_net (primary) | quadnet_v4_audited (control)
  single/ stage{1,2,3}/       the collapsed curriculum | the audited three-stage one
  evaluation/                 held_out_once (primary) | audited_replica
  runtime/                    default (empty) | kaggle_t4x2 — hardware profiles
  tracking/                   console | console_jsonl | jsonl | wandb | tensorboard | none
  experiment/                 seednet_full256 (DEFAULT) | quadnet_full256 (control)
                              | quadnet_audited (frozen historical replica)

src/spectralquadnet/
  config/       schema.py typed dataclasses; compose.py programmatic composition
  data/         mmap_store (+ band_geometry, page-cache pre-warm), loaders (splits),
                datasets (augmentation), samplers, morphometrics
    prep/       offline: download → radiometry → segmentation → patch extraction
                preslice.py — a band subset as a small, checksummed, uploadable dataset
                band_selection.py — ABLATION PATHWAY ONLY
  models/       registry (arch → network), spectral_seed_net (primary),
                spectral_quadnet (control), front_end (λ operators), fusion, heads,
                ema, control, stats_ops
    branches/   spatial_cnn (the 3-D stem), spectral_stats (index bank + hull),
                spectral_profile, specformer
    blocks/     attention (ECA/CBAM), conv_blocks, positional
  engine/       pipelines/ (context + single | three_stage dispatch)
                stages/ (single_stage, stage1/2/3, final_eval)
                train_epoch, evaluate, tta, checkpoint, resume, diagnostics, batch,
                clean_fit (fixed-subset clean-fit probe, S11)
  losses/       focal, contrastive (SupCon/ProtoNCE), cdws, mixup, auxiliary
  optim/        param_groups, schedulers, sam
  reporting/    metrics + CIs, results tree, figures, tables
  experiments/  registry (the ablation grid), runner, protocol, baselines,
                leakage, aggregate, analysis, cli,
                frozen (S11: X1/X2/X4 expanded from the hashed pre-registrations)
  bandstudy/    ABLATION PATHWAY ONLY — how many bands, which, which method
  tracking/     console | jsonl | wandb | tensorboard | multi
  utils/        device (runtime plan), distributed (DDP), seed, warning_filters,
                provenance (code revision, environment, applied regime → run.json)

scripts/        thin CLI wrappers + validation gates
tests/          unit/ (fast) · regression/ (goldens) · smoke/ (end-to-end)
```

### Key interfaces

| Interface | Contract |
|---|---|
| `build_model(cfg, physical_wl) → nn.Module` | The only place `cfg.model.arch` becomes a network. |
| model `forward` | dict with `main` + `aux_*` (+ `emb`, `balance`; `main_plain` — telemetry only — under a margin) in `.train()`; bare logits in `.eval()`. Everything downstream is written against this shape, never against a class. |
| model declarations | optional `pathway_labels()`, `grad_groups()`, `clip_groups()` — what the influence probe, per-module gradient norms and `clip_partition=model` read (SeedNet declares all three; S11). |
| `RunContext` | Hardware, plan, store, splits, model, EMA, tracker, clock, calib loader, band geometry — built once, in the one order that is correct (`set_seed → DataStore → build_model → ModelEMA`). |
| `Splits` / `SplitReport` | Four index arrays plus what the split *achieved*, not what was requested. |
| `DataStore` | Process-wide singleton over read-only mmaps; `band_geometry(cfg, store)` is its contract check. |
| `band_augmentation_widths(C)` | The single source of the two band-count-dependent augmentation widths. |
| `spectral_stride_schedule(C, d)` | The single source of the stem's spectral reduction. |
| `Ablation` / `Arm` | An experiment as a *value*: a question, arms, and a pre-registered decision rule. |

---

## 8 · Configuration

Standard Hydra. Every value-carrying field is `omegaconf.MISSING` in the schema, so a key
missing from `configs/` fails at composition time rather than silently defaulting.
`TrackingConfig`, `RuntimeConfig` and `EvaluationConfig` carry real defaults, because a
config that never mentions them is not under-specified.

| Group | Owns |
|---|---|
| `data` | paths, `num_bands`, `num_classes`, augmentation widths, the split protocol, `band_indices_path` |
| `model` | `arch`, both architectures' widths, `stem_folded_depth`, `specf_tokens`, head elaborations |
| `single` | the primary curriculum: 14 hyperparameters, plus `aux_weight_schedule` (which aux weight is applied) and `clean_fit_*` (measurement) |
| `stage1/2/3` | the audited curriculum, retained for A8 |
| `evaluation` | `select_split`, `report_split`, `tta`, `bootstrap_samples`, `save_logits` |
| `runtime` | throughput only — nothing here may change a reported number (a profile such as `runtime=kaggle_t4x2` also picks the AMP dtype, and says so) |
| `tracking` | backend selection and diagnostics verbosity; `jsonl` writes `output_dir/metrics.jsonl` |
| root | `pipeline`, `seed`, `device`, `weight_decay`, `grad_clip`, `clip_partition`, `ema_decay`, TTA counts |

Two config keys carry the 256-band design and are worth knowing:

- **`model.stem_folded_depth`** (8) — the spectral depth the 3-D stem folds into channels.
  The three spectral strides are *derived* from this and `data.num_bands`.
- **`model.specf_tokens`** (10 audited / 32 on the full cube) — Branch D's λ-window count,
  set **directly**. It used to be `num_bands // (specf_patch // 2)`, which made a window's
  width a function of k — 15 nm at k = 40 and 2.4 nm at k = 256 — so "token 3" denoted a
  different spectral region in every arm, contradicting the property λ-uniform
  tokenisation exists to provide.

Overriding anything:

```bash
python train.py data.split_fold=1 seed=1
python train.py data=refl215_stratified                       # the contrast arm
python train.py --config-name experiment/quadnet_full256     # the four-branch control
python train.py --config-name experiment/quadnet_audited     # the frozen replica
python train.py single.max_lr=1e-4 single.epochs=80
python train.py -m seed=0,1,2                                # Hydra multirun
python train.py tracking.backend=wandb
python train.py tracking=console_jsonl                        # + metrics.jsonl
torchrun --standalone --nproc_per_node=2 train.py runtime=kaggle_t4x2
```

[`docs/config_reference.md`](docs/config_reference.md) documents every group's keys, their
shipped values and what they mean. [`docs/config_migration_table.md`](docs/config_migration_table.md)
is **generated** by `scripts/check_config_roundtrip.py --emit-markdown` and maps every key of
the pre-refactor monolith to its single home; the same script enforces the mapping.

---

## 9 · Testing and validation

```bash
pytest                             # fast tier — unit tests only (~10 s)
pytest --run-all                   # + regression, slow and dataset-dependent tiers
pytest --run-slow tests/smoke/     # end-to-end runs on a synthetic dataset (~2 min)

ruff check . && black --check . && mypy       # lint / format / types
python scripts/check_config_roundtrip.py      # every config key has exactly one home
python scripts/capture_golden.py --verify     # numerical gates, writes nothing
```

The smoke tier builds a miniature dataset with the real one's load-bearing structure —
**two class-pure bundles per class** — and runs the actual `train.py` composition end to
end, because both of the audited run's most consequential defects were integration
failures that no component test could have caught.

| Gate | What it pins |
|---|---|
| `tests/unit/test_branch_c_stem.py` | The stride schedule at 8/40/100/256 bands; that the 40-band schedule is still `(2,2,2)`; **that every band reaches at least one stage-1 tap**, band by band |
| `tests/unit/test_masked_ops.py` | That the O(C²) continuum hull is **bit-identical** to a literal transcription of the O(C³) chord enumeration; that it runs at batch 64 × 256 bands; gain invariance |
| `tests/unit/test_mmap_store.py` | The band-geometry contract, including the wavelength-vector mismatch no other check catches |
| `tests/unit/test_cutmix.py` | That every shipped data config's augmentation widths equal `band_augmentation_widths(num_bands)` |
| `tests/unit/test_spectral_seed_net.py` | The primary config *is* 256 bands with no index file; the parameter budget; both pathways influence the output |
| `tests/unit/test_config_wiring.py` | Every `cfg.model.*` key is forward- or dropout-observable, or has a named reason it cannot be |
| `tests/unit/test_schedulers.py` | Every LR multiplier and margin value across the full epoch range of every stage |
| `tests/regression/test_golden_forward_pass.py` | Eval-mode logits, per-tensor SHA-256 of 306 initialised tensors, the Stage-1 epoch-1 loss and post-step weight digests |
| `tests/regression/test_state_dict_compatibility.py` | Top-level attribute names; the checkpoint bundle schema |
| `tests/regression/test_resume_and_final_eval.py` | Auto-resume detection, sidecar schema, `val_f1`-based selection |
| `tests/unit/test_presliced_dataset.py` | The pre-sliced cube equals its source bands within the float16 bound (bit-exact at float32); a corrupted file fails `--verify`; a pre-sliced cube reports itself as reduced; the loader widens float16 to float32 |
| `tests/unit/test_resume.py` | Training-state snapshot/restore continues exactly; interruption vs completion; the background writer; the DDP sampler reshuffles per epoch; the T4 profile composes |
| `tests/smoke/test_kaggle_path.py` | Pre-slice → train → `metrics.jsonl` → simulated interruption → resume at the next epoch, and a two-rank `torchrun` run to the final evaluation |

### Recorded result of the S15 validation (2026-10-03, macOS CPU, torch 2.13)

```
pytest                787 passed, 403 skipped, 2 xfailed, 0 failed     (S13 + S15 plan tests 28/28)
run_s15.py --check    hashes S12/S13/S14 verify; code identity OK (129 files = aed5257); 10/10 cells compose
run_s15.py --cfg-job  10/10 compose through train.py
2-rank torchrun       R1, Y3 (seed 1), Y5 desc_only, Y5 spatial_repair finish; each held-out kernel once
```

S15 changes no model, training, data or config code (content digest equal to `aed5257`'s). Details:
[`docs/research/studies/S15_y3_replication`](docs/research/studies/S15_y3_replication/README.md).

### Recorded result of the S13 validation (2026-10-02, macOS CPU, torch 2.13)

```
pytest --run-all      1100 passed, 4 failed, 37 errors, 3 xfailed   (413a11e: 1050 / 4 / 37 / 3 — identical failing set)
G-neutral gate        PASS vs 413a11e — identical step losses, 222/222 tensors, predictions, 3 regimes;
                      Y2 (mixstyle) and Y3 (lean) detected as sensitivity controls
2-rank torchrun       every S13 arm finishes; each held-out kernel scored once; κ on every training row once
race regression       413a11e: rank 1 EOFError (the S11 failure); S13: both ranks reload the final epoch
```

The S13 arm keys default to the shipped network; the failing set is the pre-existing one described
below. Details: [`docs/research/studies/S13_representation_screening`](docs/research/studies/S13_representation_screening/README.md).

### Recorded result of the S11 validation (2026-10-01, macOS CPU, torch 2.14)

```
pytest --run-all      1034 passed, 6 failed, 37 errors, 3 xfailed   (pre-S11 tree: 964 / 6 / 37)
pytest                 711 passed, 1 failed                          (pre-S11: 645 / 1)
ruff check .          13 findings, all pre-existing     mypy --strict   12 errors, the pre-existing set
G-neutral gate        PASS — identical step losses, 222/222 tensors, predictions, 3 regimes
```

The failing set is **identical before and after S11** and none is new: 37 tests (and
`capture_golden.py --verify`, `check_config_roundtrip.py`) need the pinned pre-refactor commit
`886560fe…`, which is not in this checkout's history (FW-26); 2 are the Stage-1 golden drift below
(FW-11); 2 smoke tests build a 256-band synthetic cube against the 215-band primary config (stale
since S07); 1 is the W&B step-collision smoke test; 1 expects an fp16 NaN that torch 2.14 no longer
produces. The 3 xfails are S10 F48's structural defect, `strict=True` until X5. Details:
[`docs/research/studies/S11_frozen_arms_execution`](docs/research/studies/S11_frozen_arms_execution/README.md).

### Recorded result of the earlier revision's validation

```
pytest --run-all      918 passed, 2 failed, 29 skipped   (25:52)
ruff check .          All checks passed
mypy                  Success: no issues found in 116 source files
check_config_roundtrip.py   ✓ all 81 keys map 1:1 with identical values
```

`black --check .` reports 16 files, **all of them pre-existing** and none touched by this
revision — the repository was formatted by an older `black` than the one installed here.
The two test failures are the two documented below.

The 256-band re-architecture was verified non-regressive on the control arm:
`python scripts/capture_golden.py --verify` reports **`v3/logits match (max |Δ| = 0.000e+00)`**
and **`v3/init digests match (306 tensors)`** — the stem's derived schedule and the
rewritten continuum hull reproduce the audited 40-band model tensor for tensor.

`test_stage1_epoch_loss_matches_golden` and `test_stage1_epoch_weights_match_golden` are
the **two pre-existing failures**, on the Stage-1 epoch loss and weight digests (loss
`23.06525230407715` vs a golden `23.080477237701416`). They reproduce identically before
and after this revision, so they are environment drift — a torch/BLAS version difference
against the machine that captured the goldens — not a regression. Re-capture with
`python scripts/capture_golden.py` on the target environment to clear them.

---

## 10 · Exact commands

### The primary pipeline, from nothing

```bash
# 1 — install (Python >= 3.10)
pip install -e ".[tracking,figures]"

# 2 — build the dataset: download → radiometry → segment → extract
#     17 GB download, ~10 min, ~30 GB. Writes patches/labels/groups/masks/morphology/gain
#     + wavelengths + tile QC. Delete dataset/rice_hsi.zip once it has finished.
pip install -e ".[prep]"
python scripts/prepare_dataset.py

# 3 — train the primary experiment. No step goes between 2 and 3.
python train.py
```

`python train.py` with no arguments runs:

| | |
|---|---|
| Input | **256 bands**, `dataset/patches.npy`, no selection |
| Architecture | `SpectralSeedNet` — 3,003,412 parameters (215 bands), two pathways |
| Curriculum | one stage, 150 epochs, early stop on `calib` |
| Split | `grouped` — leave-one-acquisition-bundle-out, fold 0 |
| Selection | on `calib` (carved from train, by group) |
| Reported | `val ∪ test`, scored **once**, ±TTA, with a bootstrap CI |
| Runtime | bf16, TF32 **off**, workers auto, compile auto |
| Output | `outputs/seednet_full256_f0_s42/` |

### The headline result (Q1)

```bash
python scripts/run_protocol.py --dry-run          # prints every per-cell command; costs nothing
python scripts/run_protocol.py                    # 2 folds × 3 seeds grouped + matched stratified
python scripts/run_protocol.py --baseline         # + LDA/LinearSVC on mean spectra
python scripts/run_protocol.py --include-audited   # + the four-branch control, same protocol
```

Produces under `outputs/experiments/protocol/`: `protocol.md`/`.csv` (mean ± range per
arm, never a maximum), **`leakage_gap.md`** (`F1_stratified − F1_grouped` — the headline),
`per_cell.md` (every individual run, so the means are auditable), `protocol.png`.

### Ablations

```bash
python -m spectralquadnet.experiments.cli list          # the grid, its cost, its ordering
python -m spectralquadnet.experiments.cli ablate A12    # ← RUN THIS FIRST
python -m spectralquadnet.experiments.cli ablate A1
python -m spectralquadnet.experiments.cli ablate A3 --arms abcd bc --dry-run
```

| | Question | Runs |
|---|---|---:|
| **A12** | What is run-to-run variance? **Run first** — until σ is known, no delta means anything | 10 |
| **A1** | How much of the score is bundle recognition? **Blocks every other claim** | 12 |
| **A2** | What does band selection cost, and does selecting outside the fold leak? | 18 |
| **A3** | Is the four-branch design justified? (symmetric dropout) | 24 |
| **A4** | Is Branch A's 64-cell replication necessary? | 18 |
| **A5** | Is the rank-128 bilinear fusion worth 0.5 M parameters? | 18 |
| **A6** | Does SupCon help, with the sampler controlled? | 12 |
| **A7** | Does any margin machinery help? | 24 |
| **A8** | Do Stages 2 and 3 add anything at all? | 24 |
| **A10** | Is capacity actually harmful? | 24 |
| **A11** | Is mixup the load-bearing regulariser? | 36 |

Each carries a **pre-registered decision rule**, printed by `list`, because an ablation
without one is an invitation to read whichever number is convenient afterwards. A2 is the
gateway to the band-selection pathway and takes the primary path as its reference arm.
A3/A4/A5/A8 run `experiment/quadnet_full256`, so an arm differs from the primary
experiment in exactly the thing it varies.

### A9 — what *are* the hard classes? (no training run)

Classes {41, 49, 51, 52, 70} were the bottom-5 at Stage-1 epoch 46 and still the bottom-5
at Stage 3, invariant to 470 epochs, three loss regimes, two samplers and four
difficulty-targeted mechanisms. A9 asks whether that is **spectrally inseparable
varieties** (a ceiling worth publishing) or **segmentation failure** (a fixable bug that
would explain the whole thing).

```bash
python -m spectralquadnet.experiments.cli analyse --run outputs/seednet_full256_f0_s42
```

### Baselines, the leakage probe, and the report

```bash
python -m spectralquadnet.experiments.cli baseline    # LDA/LinearSVC on mean spectra
python -m spectralquadnet.experiments.cli leakage     # bundle identity from residual brightness
python -m spectralquadnet.experiments.cli aggregate   # rebuild every table, no GPU
python -m spectralquadnet.experiments.cli report      # → outputs/experiments/REPORT.md
```

The LDA-on-mean-spectra baseline costs seconds and is *the paper's most important
baseline*: it reaches 0.5916 under the leaky protocol at k = 40, so ~59 points are
available with no spatial information at all. The leakage probe fits a 10-feature linear
model on **residual brightness alone** and reports how well it recovers the acquisition
bundle — a model-free measurement of the nuisance.

### The S11 frozen diagnostic arms — X1, X2, X4

Pre-registered in S09 (`preregistration_next.json`, X1/X2) and S10 (`preregistration_s10.json`,
X4); `scripts/run_frozen.py` builds the 23 cells **from those files after checking their
SHA-256**, gives each its own directory under `outputs/experiments_u430k32/s11/`, and writes
`frozen_cell.json` (source, hashes, exact command) before it runs. See
[`docs/research/studies/S11_frozen_arms_execution`](docs/research/studies/S11_frozen_arms_execution/README.md)
for what each arm decides and the exact Kaggle cells.

```bash
python scripts/run_frozen.py --list          # cells, overrides, frozen source — free
python scripts/run_frozen.py --check         # compose every cell, assert its regime — free
python scripts/run_frozen.py --cfg-job       # `train.py --cfg job` for every cell — free
python scripts/run_frozen.py --arms X1 X4 --nproc-per-node 2 --stream   # 11 runs, T4 x2
python scripts/run_frozen.py --arms X2    --nproc-per-node 2 --stream   # 12 runs, T4 x2
```

### S13 — the route-A arms Y1–Y4, as a single-seed screen

Frozen in S12 (`preregistration_s12.json`, 30 runs) and **amended before any run to one seed** (seed 0;
`preregistration_s13.json`, D28): every arm, control, protocol contrast and grouped fold is kept — **10 GPU runs**
(Y1 decoupled pathways 4, Y2 masked MixStyle 2, Y3 lean architecture 3, Y4 the 80/20 tier 1) plus Y1's two fused cells
(CPU). `scripts/run_s13.py` builds the cells from **both** hashed files and refuses to run if either moved. Outcomes are
screening verdicts; an arm that passes is replicated at seeds 1–2 before it becomes a reference or a claim. See
[`docs/research/studies/S13_representation_screening`](docs/research/studies/S13_representation_screening/README.md).

```bash
python scripts/run_s13.py --list                          # 10 GPU + 2 fused cells, their frozen sources — free
python scripts/run_s13.py --check                         # compose every cell, assert R1 + its arm — free
python scripts/run_s13.py --nproc-per-node 2 --dry-run    # the exact torchrun commands — free
python scripts/run_s13.py --nproc-per-node 2 --stream     # everything: 10 runs, fusion, summary — T4 x2, ≈ 4.4 h
python scripts/run_s13.py --fuse-only                     # re-fuse Y1 from finished cells — CPU
python scripts/run_s13.py --summary                       # what is scored so far
```

The arms' model keys default to the shipped network (bit-identical, S13's G-neutral gate):
`model.spatial_mixstyle` (Y2), `model.spectral_descriptor`, `model.spatial_tail_strides`, `model.cbam_min_hw` (Y3).
Every single-stage run now also reports the training-rows session κ (`evaluation.session_probe`, D26).

### S15 — replicate the lean network (Y3) and dissect it (Y5)

S14 read the S13 screen: only Y3, the lean network (`model.spectral_descriptor=snv_morph
model.spatial_tail_strides=[2,2,2,1] model.cbam_min_hw=3`), passed — grouped 0.562, stratified 0.746, above every X1 run.
S14 froze the next round in `docs/research/evidence/S14_screen_reading/preregistration_s14.json` (`9e182670…`):
**exactly 10 GPU runs and nothing else** — Y3 at seeds 1 and 2 (grouped folds 0/1 + stratified, 6 runs; S13 already
ran seed 0) and Y5, Y3's changes split in two at seed 0 (`desc_only`, `spatial_repair`; grouped folds 0/1, 4 runs).
`scripts/run_s15.py` builds the cells from the three hashed files (S12 → R1, S13, S14 → cells with their own seeds),
refuses to run if any moved, refuses if the model/training/data/config code differs from `aed5257` (the code the S13
cells ran — a content digest, so it works in Kaggle's shallow clone), and never re-runs an S13 cell. Replication cells
run first. See [`docs/research/studies/S15_y3_replication`](docs/research/studies/S15_y3_replication/README.md).

```bash
python scripts/run_s15.py --list                          # the 10 cells, their seeds and frozen sources — free
python scripts/run_s15.py --check                         # hashes, code identity, R1, every cell's composition — free
python scripts/run_s15.py --cfg-job                       # `train.py --cfg job` for every cell — free
python scripts/run_s15.py --nproc-per-node 2 --dry-run    # the exact torchrun commands — free
python scripts/run_s15.py --nproc-per-node 2 --stream     # all 10 runs — T4 x2, ≈ 4.4 h (≤ 4.8 h)
python scripts/run_s15.py --summary                       # what is scored so far
python scripts/run_s15.py --cells Y5/desc_only__f0_s0 --nproc-per-node 2 --stream   # one cell
```

No new model key: S15 uses only keys S13 added. The cells write to
`outputs/experiments_u430k32/s15/{Y3,Y5}/<variant>__f<fold>_s<seed>/`; the reading (H21a–H21e, H22a/H22b) belongs to
the next analysis study (S16).

### Multi-GPU

```bash
torchrun --standalone --nproc_per_node=2 train.py
```

DDP with `SyncBatchNorm`, so two GPUs compute the same function as one: the batch is
split, but the normalisation statistics are all-reduced back to the global batch's and
the gradient average over equal shards is the global-batch gradient. The distributed
sampler is re-seeded every epoch (`set_epoch`), so each epoch draws a new shard order.

### Kaggle — GPU T4 x2

The full 215-band cube is 30.4 GB. What goes to Kaggle is the **pre-sliced 32-band
finalist** (`uniform430_k32`, 432–1006 nm, float16): **2.33 GB**, with labels, groups, fill
maps, morphometrics, wavelengths, the session table, the radiometry record, a provenance
file and a checksummed manifest. Runs on it are the **k = 32 reduced arm**, not the primary
215-band methodology, and say so: `band_axis.json` makes every run print
`Spectral: 32 of 215 bands — REDUCED arm via a pre-sliced cube` and record
`"band_selection": true` in `results/run.json`.

**1 · On your machine — build and upload the dataset (once).** Needs `./dataset/` and
`outputs/band_finalists/`, and the [Kaggle CLI](https://github.com/Kaggle/kaggle-api) with an
API token in `~/.kaggle/kaggle.json`.

```bash
python scripts/build_presliced_dataset.py --kaggle-id <kaggle-username>/rice-hsi-u430k32
python scripts/build_presliced_dataset.py --verify dataset_u430k32   # re-hashes every file
kaggle datasets create -p dataset_u430k32                            # private by default
```

The build takes about a minute. The float16 cast is measured, not assumed, and recorded in
`dataset_u430k32/band_axis.json`: max relative error 4.88e-4 over normal-range values, and
16 of 2.9 × 10⁸ non-zero values (all ≤ 4.9e-16) flushed to zero — no pixel changes
foreground status. `--dtype float32` writes a bit-identical 4.5 GB subset instead;
`--set all` writes all 215 bands as float16 (15.2 GB) if you want the primary cube on Kaggle.

**2 · Notebook settings.** *Accelerator*: **GPU T4 x2**. *Internet*: **On**.
*Add Input*: the dataset from step 1.

**3 · Cell 1 — code, dependencies, data link, sanity check.**

```bash
%%bash
set -euo pipefail
cd /kaggle/working
[ -d HyperSpectral_Seeds ] || git clone --depth 1 https://github.com/Jerlshin/HyperSpectral_Seeds.git
cd HyperSpectral_Seeds
pip install -q -e .                     # torch is Kaggle's own; this adds hydra/omegaconf
ln -sfn /kaggle/input/rice-hsi-u430k32 dataset_u430k32   # <- your dataset's slug
python scripts/build_presliced_dataset.py --verify dataset_u430k32 --quick
nvidia-smi --query-gpu=index,name,memory.total --format=csv
```

**4 · Cell 2 — one short run first** (both GPUs, 2 epochs, ~minutes; catches a path or
driver problem before the long sweep):

```bash
!cd /kaggle/working/HyperSpectral_Seeds && torchrun --standalone --nproc_per_node=2 train.py \
    data=ablation/u430k32_grouped runtime=kaggle_t4x2 tracking=console_jsonl \
    single.epochs=2 output_dir=outputs/kaggle_smoke
```

Look for `[DDP]  world_size=2`, `Precis : AMP=fp16 + GradScaler`, the `REDUCED arm` line and
`[DATA] ✓ Page cache warmed` in the banner.

**5 · Cell 3 — the training matrix.** The primary protocol on the 32-band cube: grouped
2 folds × 3 seeds + the matched stratified contrast (12 cells), each on both GPUs, then the
aggregate tables and the leakage gap. `--dry-run` prints every command first.

```bash
!cd /kaggle/working/HyperSpectral_Seeds && python scripts/run_protocol.py \
    --data ablation/u430k32 --nproc-per-node 2 \
    --override runtime=kaggle_t4x2 --override tracking=console_jsonl \
    --output-root outputs/experiments_u430k32 --baseline --stream
```

Other grids take the same flags, e.g. a single cell or the four-branch control:

```bash
!cd /kaggle/working/HyperSpectral_Seeds && torchrun --standalone --nproc_per_node=2 train.py \
    data=ablation/u430k32_grouped data.split_fold=1 seed=2 runtime=kaggle_t4x2 tracking=console_jsonl
!cd /kaggle/working/HyperSpectral_Seeds && torchrun --standalone --nproc_per_node=2 train.py \
    --config-name experiment/quadnet_full256 data=ablation/u430k32_grouped \
    runtime=kaggle_t4x2 tracking=console_jsonl
```

**What `runtime=kaggle_t4x2` does** (`configs/runtime/kaggle_t4x2.yaml`):

| | |
|---|---|
| Topology | DDP over both T4s with `SyncBatchNorm`; global batch unchanged (128 → 64 per GPU); `multi_gpu: ddp` fails fast if `torchrun` is missing |
| Precision | **fp16 autocast + GradScaler** — the T4's Tensor Core path. The shipped `bf16` is emulated on Turing and slower than fp32. The dtype is part of what a number means, so it is printed and recorded; non-finite batches are skipped and counted in `train/skipped_batches` — if that climbs, rerun with `runtime.amp_dtype=bf16` |
| Input | 2 loader workers + 2 eval workers per rank (4 vCPUs), pinned memory, persistent workers; the mmapped cube is **read once into the page cache** at startup (local rank 0), so every random patch read afterwards is a memory hit |
| Kernels | `torch.compile` (inductor), cuDNN autotune, fused AdamW; TF32 off (Turing has none) |
| Checkpoints | full training state every epoch, written on a background thread (below) |

**Outputs, per cell** (`outputs/experiments_u430k32/protocol/<arm>__f<fold>_s<seed>/`):

| File | What |
|---|---|
| `metrics.jsonl` | every scalar, epoch row, table (per-class F1, session breakdown), banner and notice as one JSON object per line |
| `training.log` | the console lines |
| `best_stage1.pth` + `stage1_meta.json` | the checkpoint selected on `calib` |
| `last_stage1.pth` + `last_stage1.json` | the full training state after the latest epoch |
| `results/` | `run.json` (metrics + CIs, both no-TTA and TTA), predictions, confusion, per-class and session tables |

```python
import pandas as pd
m = pd.read_json("outputs/experiments_u430k32/protocol/grouped__f0_s0/metrics.jsonl", lines=True)
curves = pd.json_normalize(m[m.event == "scalars"].to_dict("records"))   # metrics.val/f1_best, …
```

**Sessions end; the sweep does not.** A Kaggle session has a run-time cap (12 h) and GPU
time is a weekly quota.
A finished cell is skipped on rerun (its `results/run.json` exists), and a cell that was
cut off resumes from `last_stage1.pth` at the next epoch — it is never mistaken for a
finished one. To continue in a new session, run the notebook as *Save Version → Save & Run
All* so `/kaggle/working` is kept as the version's output, add that output as an input to
the next version, and restore it before Cell 3:

```bash
!cp -rn /kaggle/input/<this-notebook-slug>/HyperSpectral_Seeds/outputs /kaggle/working/HyperSpectral_Seeds/
```

**S15 in one session** (done 2026-10-03, 10/10 cells; read in S16 — kept as the template for S17's cells; after Cell 1, the 2-epoch smoke run is optional). The code must be on GitHub
`main` (Cell 2 prints the commit). Only the 10 frozen S15 cells run; nothing from S11/S13 is repeated and no S11/S13
output needs to be attached.

```bash
# Cell 2 — the S15 code, its frozen plan, the code-identity guard (must print "code identity: OK" and "10/10 cells OK")
!cd /kaggle/working/HyperSpectral_Seeds && git log --oneline -1 && python scripts/run_s15.py --check
# Cell 3 — the 10 S15 runs on both GPUs: Y3 seeds 1–2 first (6), then the Y5 dissection (4), then a summary (≈ 4.4 h)
!cd /kaggle/working/HyperSpectral_Seeds && python scripts/run_s15.py --nproc-per-node 2 --stream
# Cell 4 — one archive to download (≈ 0.25 GB with checkpoints), for the S16 reading on your machine
!cd /kaggle/working/HyperSpectral_Seeds && tar czf /kaggle/working/s15_outputs.tar.gz outputs/experiments_u430k32/s15 && ls -lh /kaggle/working/s15_outputs.tar.gz
```

Back on your machine, unpack it at the repository root (`tar xzf s15_outputs.tar.gz`) so the cells land in
`outputs/experiments_u430k32/s15/`. If the session is cut short, restore `outputs/` as below and re-run Cell 3: finished
cells are skipped and an interrupted one resumes from its last epoch.

*S13 (done, 2026-10-03):* `python scripts/run_s13.py --nproc-per-node 2 --stream` ran the 10 screening cells and Y1's
fusion; it skips finished cells if re-run.

When every cell is done (or to re-tabulate what is there):

```bash
!cd /kaggle/working/HyperSpectral_Seeds && python scripts/run_protocol.py \
    --data ablation/u430k32 --output-root outputs/experiments_u430k32 --aggregate-only
```

If `torch.compile` misbehaves on the image's torch version, add `runtime.compile=off`; if
the loader cannot keep up (GPU utilisation well under 100 % in `nvidia-smi`), try
`runtime.num_workers=6`.

---

## 11 · Hardware

Trains on CPU, CUDA and Apple Metal; device selection is automatic (`device=auto`).

The 3-D stem is where the band count is paid for, and the derived stride schedule is what
bounds it. Multiply-accumulates per sample at 64×64 spatial, `stem_channels=192`:

| bands | folded depth | stem MACs | vs. 40 bands |
|---:|---:|---:|---:|
| 40 | 5 | 452 M | 1.00× |
| 100 | 7 | 597 M | 1.32× |
| **256** | **8** | **874 M** | **1.93×** |
| 256, three hardcoded halvings | 32 | 2,894 M | 6.40× |

So a 6.4× wider input costs **1.93×** in the stem, and the last row is what the schedule
avoids.

Wall-clock and activation-memory figures in `docs/06_EXECUTION_AND_HARDWARE.md` were all
measured on the **40-band** audited configuration, which is the only one a full run has been
executed on; no 256-band timing has been measured on this hardware and none is claimed. Two
Metal-only execution paths matter at this size and are auto-enabled there:
`runtime.decompose_conv3d` (identical arithmetic in a different summation order; **2.12×
on the whole training step**) and `runtime.checkpoint_branch_a`. Neither changes a
reported number.

---

## 12 · Documentation

| | |
|---|---|
| [`docs/research/`](docs/research/README.md) | **The research log**: every study, finding and decision so far, with evidence and figures, why each step was taken, and the backlog. Start here for *what we have learned*. |
| [`CHANGES.md`](CHANGES.md) | The audit. The authoritative specification this revision implements. |
| [`docs/01_ABSTRACT_AND_OVERVIEW.md`](docs/01_ABSTRACT_AND_OVERVIEW.md) | Objective, contributions, evaluation framework |
| [`docs/02_DATASET_AND_PREPROCESSING.md`](docs/02_DATASET_AND_PREPROCESSING.md) | Acquisition → segmentation → patches → store → splits |
| [`docs/03_MODEL_ARCHITECTURE.md`](docs/03_MODEL_ARCHITECTURE.md) | The 256-band architecture, branch by branch, with the tensor-shape matrix |
| [`docs/04_CURRICULUM_AND_LOSSES.md`](docs/04_CURRICULUM_AND_LOSSES.md) | Objectives, schedules, optimisation rules |
| [`docs/05_EXPERIMENTS_AND_ABLATIONS.md`](docs/05_EXPERIMENTS_AND_ABLATIONS.md) | TTA, diagnostics, telemetry, checkpointing, the ablation surface |
| [`docs/06_EXECUTION_AND_HARDWARE.md`](docs/06_EXECUTION_AND_HARDWARE.md) | Entrypoint orchestration, runtime knobs, DDP, profiling |
| [`docs/07_BAND_SELECTION_PATHWAY.md`](docs/07_BAND_SELECTION_PATHWAY.md) | The retained band-selection research pathway |
| [`docs/config_reference.md`](docs/config_reference.md) | Configuration key reference — every group, every shipped value, what it means |
| [`docs/config_migration_table.md`](docs/config_migration_table.md) | Generated: the pre-refactor monolith's `CONFIG` keys → their single home in `configs/` |
