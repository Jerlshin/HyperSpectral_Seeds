# REFACTOR_PLAN.md
### SpectralQuadNet — From Monolithic Script to Modular Research Framework

**Author:** Senior Principal AI Architect review
**Scope:** Planning only. No source files are modified, moved, or deleted by this document.
**Subject system:** 4-branch hyperspectral CNN/Transformer (`SpectralQuadNet`) classifying 90 rice
seed varieties from 40-band VIS-NIR (385–1000 nm) patches, achieving **~87.5% Macro F1 / 87.5% Acc**
on held-out validation (`stage3_meta.json`) via a 3-stage curriculum (progressive augmentation →
sub-center ArcFace → SAM+SWA), evaluated with 12-view TTA.

---

## 0. Executive Summary

The entire model, training loop, loss functions, samplers, checkpointing, and CLI entrypoint live in
one 2,857-line file: `HSI_modality_training/hsi_training.py`. It is dense, correct, and already
achieves strong results — this is **not** a rewrite. The goal is a **mechanical decomposition**:
move existing, working code into well-scoped modules with zero behavioral drift, then layer on
config management, experiment tracking, and diagnostics as strictly additive capabilities.

**Guiding principles for the eventual implementation work:**

1. **Move, don't rewrite.** Every class/function body is relocated verbatim (only import paths and
   `CONFIG[...]` lookups change to config-object attribute access).
2. **State-dict keys are sacred.** Three real trained checkpoints exist
   (`best_stage1.pth`, `best_stage2.pth`, `best_stage3.pth`, ~63 MB each, in
   `HSI_modality_training/output_v12_SPA40/`). They store `model.state_dict()` tensors, not pickled
   objects, so as long as `nn.Module` attribute names (`self.branch_a`, `self.se`, `self.arcface_head`,
   …) are unchanged, file relocation is checkpoint-safe by construction. This is verified explicitly
   in §3.1, not assumed.
3. **RNG call order is the real hazard, not file layout.** `set_seed()` currently runs as an
   **import-time side effect** and every subsequent `nn.init.*` call in `_init_weights()` consumes
   the same global RNG stream. Reordering when configs load vs. when the model is built is the
   single highest-risk change in this migration — see §3.6.
4. **Config replaces the `CONFIG` dict 1:1, not more, not less.** Every one of the ~70 keys in the
   current dict must have exactly one new home; nothing is silently dropped, renamed, or
   re-defaulted during the mechanical migration.
5. **Diagnostics and tracking are additive.** They must be implementable by wrapping existing
   `print(...)` call sites and existing return values (`compute_branch_influence`, per-branch aux
   losses, `evaluate_per_class`) — none of that logic needs to change to support them.

---

## 1. Current State Audit

### 1.1 Repository inventory

| Path | Role | Status |
|---|---|---|
| `HSI_modality_training/hsi_training.py` | Monolith: config, data, model, losses, optim, engine, 3 stages, CLI | Active, target of this refactor |
| `HSI_modality_training/output_v12_SPA40/` | Run artifacts: 3 checkpoints + 3 meta JSONs + preds/targets + log | **Must remain loadable** — treated as ground truth |
| `band_selection.py` (root) | Standalone mRMR+SPA band-selection pipeline (256→40 bands), own `CONFIG` dict | Active, decoupled from training |
| `data_setup_v3.py` (root) | Raw Zenodo zip → Otsu segmentation → 64×64 patch extraction → `patches.npy`/`labels.npy` | Active, decoupled from training |
| `installation.py` (root) | Compiles `causal-conv1d` + `mamba-ssm` against the active conda env | **Dead relative to current model** — `hsi_training.py` has zero `mamba`/`causal_conv1d` imports; this is a forward-looking or legacy dependency for a SpecMamba branch that isn't wired in |
| `dataset/` | `patches.npy` (36 GB, 256-band), `patches_spa_40b.npy` (5.6 GB, 40-band, **the file actually used**), `labels.npy`, `wavelengths*.csv`, `rice_hsi.zip` (17 GB), `band_selection_report.csv` | Binary data, not code |
| `figures/`, `*.png` (root) | Static plots for papers/README | Not code |
| `*.ipynb` (root) | `embedding_overview`, `pipeline_overview`, `VIS_NIR_hyperspectral_modality` — EDA/visualization notebooks | Not part of the training path |
| `README.md` | One line (`# HSI_RGB_Seeds`) | Effectively empty |
| — | No `requirements.txt`, `pyproject.toml`, or `environment.yml` anywhere | **Confirmed gap** — dependency set is only inferable from `import` statements |

`git status` shows ~30 prior training-script iterations (`hs_train.py`, `hsi_trainingv2.py` … `v4`,
`final_hsi_train_base.py`, `refined_hs_training.py`, a `src/` package, etc.) already staged as
deleted, alongside a modified `hsi_training.py` and several new untracked artifacts
(`band_selection.py`, `dataset/`, `output_v12_SPA40/`, notebooks). **This plan does not act on that
staged state** — it documents the target for the *next* structural change, to be applied on top of
whatever the user decides to do with those pending deletions.

### 1.2 `hsi_training.py` symbol inventory (by line number, current file)

| Lines | Symbol(s) | Category |
|---|---|---|
| 34–143 | `CONFIG` dict (~70 keys) | Config |
| 163–187 | `_load_data_mmap`, `_load_wavelengths_to_gpu` (+ module globals `_GPU_PATCHES`, `_GLOBAL_LABELS`, `_PHYSICAL_WL`) | Data / global state |
| 194–200 | `set_seed` (called at import time, line 200) | Reproducibility |
| 207–246 | `ModelEMA` | Model support |
| 253–356 | `RiceSeedDataset` (+ 6 augmentation primitives, 3 phase profiles) | Data |
| 362–446 | `ClassBalancedBatchSampler`, `HardClassOversampledSampler` | Data |
| 453–461 | `build_cdws_weights` (Class-Difficulty-Weighted Sampling) | Losses/weighting |
| 468–487 | `_mixup`, `mixed_aug`, `mixed_loss` | Batch augmentation |
| 494–559 | `FocalLoss`, `SupConLoss`, `ProtoNCELoss` | Losses |
| 566–610 | `SAM` (Sharpness-Aware Minimization optimizer) | Optim |
| 617–678 | `AdaptiveSubcenterArcFaceHead` (dynamic per-class margin) | Model head |
| 684–718 | `MaskedSpectralECA` | Model block |
| 720–812 | `SEBlock1D`, `ResBlock1D`, `CBAM`, `ResBlock2D`, `PhysicalWavelengthPE` | Model blocks |
| 817–843 | `LargeKernelBlock1D` | Model block |
| 846–908 | `SpectralProfileBranch` (**Branch A**) | Model branch |
| 914–988 | `SpectralStatsBranch` (**Branch B**) | Model branch |
| 995–1024 | `SpatialCNNBranch` (**Branch C**) | Model branch |
| 1030–1164 | `MultiScaleSpectralTokenizer`, `_PreLNBlock`, `SpecFormerBranch` (**Branch D**) | Model branch |
| 1171–1283 | `CrossModalInteraction` (Perceiver-style latent fusion) | Model fusion |
| 1285–1304 | `EmbedNet` | Model fusion |
| 1310–1326 | `AuxiliaryHead` | Model head (deep supervision) |
| 1334–1421 | `compute_branch_influence`, `extract_grid_spectra`, `masked_spectral_stats` | Diagnostics / feature ops |
| 1429–1631 | `SpectralQuadNet` (composition root, forward pass, branch dropout) | Model |
| 1638–1667 | `tta_predict` (8 spatial + 4 spectral views) | Inference |
| 1674–1743 | `build_splits`, `build_loaders`, `build_phase3_loader` | Data |
| 1750–1801 | `_wd_groups`, `build_optimizer_s1/s2/s3`, `sgdr_scheduler`, `arcface_margin` | Optim |
| 1807–1852 | `_aux_loss_weight`, `_compute_aux_loss` | Losses |
| 1859–1961 | `train_one_epoch` (AdamW path, stages 1–2) | Engine |
| 1967–2027 | `train_one_epoch_sam` (SAM path, stage 3) | Engine |
| 2034–2063 | `_run_eval`, `evaluate`, `evaluate_per_class` | Engine |
| 2070–2150 | `stage_ckpt_path/meta_path/exists`, `latest_completed_stage`, `save_ckpt`, `load_ckpt`, `load_stage_meta`, `update_bn_stats`, `_is_json_serialisable` | Checkpointing |
| 2157–2183 | `compute_class_difficulty` | Diagnostics |
| 2190–2388 | `run_stage1` — 3-phase progressive augmentation + custom phase-aware LR | Stage |
| 2395–2501 | `run_stage2` — Sub-center ArcFace + SGDR + SupCon + ProtoNCE + CDWS | Stage |
| 2508–2617 | `run_stage3_swa` — SAM + greedy SWA snapshotting | Stage |
| 2624–2668 | `final_evaluation` — TTA-based test evaluation | Stage |
| 2675–2694 | `_pick_best_checkpoint` | Checkpointing |
| 2701–2837 | `main` — auto-resume orchestration across all 3 stages | CLI |
| 2843–2858 | `if __name__ == "__main__"` — logging setup + top-level exception handler | CLI |

### 1.3 Notable tech debt observed (documented, **not fixed** by this refactor)

These are flagged so the migration preserves them faithfully (a mechanical move must not "helpfully"
fix them mid-flight — that's a separate, reviewable change):

- `CONFIG["output_dir"]` (line 39) is a **hardcoded absolute path**
  (`/Users/jerlshin/.../HSI_modality_training/output_v12_SPA40`) — machine-specific, breaks on any
  other checkout.
- `wl_embed_dim` is accepted by `SpectralQuadNet.__init__` (line 1457) and by `CONFIG` (line 132) but
  is **never used** in the constructor body — dead parameter.
- `SpecFormerBranch.__init__`'s `patch_size` argument (line 1087) is explicitly commented "kept for
  API compatibility, handled by MultiScale" — dead parameter.
- Module-level side effects at import time: `print(CONFIG)` (145), `Path(...).mkdir()` (147),
  `torch.cuda.empty_cache()` (148), and `set_seed(CONFIG["seed"])` (200) all execute merely by
  `import hsi_training` — anti-pattern for a library, and the exact reason config loading order
  becomes safety-critical during migration (§3.6).
- `ClassBalancedBatchSampler.__iter__` and `HardClassOversampledSampler.__iter__` each draw from an
  **unseeded** `np.random.default_rng()` / `torch.multinomial` (no `generator=` argument) — so
  Stage 2/3 batch composition is already non-reproducible across runs, independent of `set_seed()`.
- `set_seed()` sets `torch.backends.cudnn.deterministic = False` and `benchmark = True` — conv
  kernel selection is explicitly non-deterministic. Full bit-exact reproducibility of a training run
  was **never a property of the current code**; the refactor cannot regress a guarantee that doesn't
  exist (see §3.6 for what *is* verifiable).
- No `requirements.txt`/`pyproject.toml` — dependency versions are only known implicitly
  (`torch==2.13.0`, `numpy==2.1.3`, etc., per the active environment at audit time).

---

## 2. Architectural Blueprint & Target Directory Tree

```
HSI_RGB_seeds/Code/                              (repo root)
├── pyproject.toml                   # package metadata + deps + ruff/black/mypy/pytest config
├── environment.yml                  # optional conda spec (CUDA toolchain for mamba/causal-conv1d extras)
├── .pre-commit-config.yaml
├── README.md
├── REFACTOR_PLAN.md
│
├── configs/                         # Hydra structured configs (YAML), one concern per file
│   ├── data/
│   │   └── spa40_90class.yaml       #   patches_data, labels_path, wavelength_path, num_bands, num_classes
│   ├── model/
│   │   └── spectral_quadnet_v4.yaml #   branch_drop_prob, subcenter_K, specf_*, aux_head_hidden, wl_embed_dim
│   ├── stage1/
│   │   └── progressive_3phase.yaml  #   s1_* keys (epochs, phase fracs, LR, mixup, oversampling, aux weights)
│   ├── stage2/
│   │   └── arcface_supcon.yaml      #   s2_* keys, arcface_*, supcon/proto weights, bal_n_cls/spc
│   ├── stage3/
│   │   └── sam_swa.yaml             #   s3_* keys (SAM rho, cycle length, greedy SWA)
│   ├── tracking/
│   │   ├── none.yaml
│   │   ├── console.yaml             #   Rich-only (default; zero external deps)
│   │   ├── wandb.yaml
│   │   └── tensorboard.yaml
│   └── experiment/
│       └── output_v12_spa40.yaml    #   composes the above via Hydra `defaults:`, sets seed + output_dir + run_name
│
├── src/
│   └── spectralquadnet/
│       ├── __init__.py
│       ├── config/
│       │   ├── __init__.py
│       │   └── schema.py            # dataclasses: DataConfig, ModelConfig, Stage1/2/3Config, TrackingConfig, ExperimentConfig
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── mmap_store.py        # DataStore: replaces module globals _GPU_PATCHES/_GLOBAL_LABELS/_PHYSICAL_WL
│       │   ├── datasets.py          # RiceSeedDataset + augmentation primitives
│       │   ├── samplers.py          # ClassBalancedBatchSampler, HardClassOversampledSampler
│       │   ├── loaders.py           # build_splits, build_loaders, build_phase3_loader
│       │   └── prep/                # offline data-prep, importable + testable (was root-level scripts)
│       │       ├── __init__.py
│       │       ├── download.py      # download() + DATA_URL  (from data_setup_v3.py)
│       │       ├── segmentation.py  # load_hsi, preprocess_raw, segment
│       │       ├── patch_extraction.py  # pad_to_square, resize_patch, 3-pass patch writer
│       │       └── band_selection.py    # extract_mean_spectra … save_outputs (from band_selection.py)
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── ema.py               # ModelEMA
│       │   ├── blocks/
│       │   │   ├── __init__.py
│       │   │   ├── attention.py     # MaskedSpectralECA, SEBlock1D, CBAM
│       │   │   ├── conv_blocks.py   # ResBlock1D, ResBlock2D, LargeKernelBlock1D
│       │   │   └── positional.py    # PhysicalWavelengthPE
│       │   ├── branches/
│       │   │   ├── __init__.py
│       │   │   ├── spectral_profile.py  # Branch A
│       │   │   ├── spectral_stats.py    # Branch B
│       │   │   ├── spatial_cnn.py       # Branch C
│       │   │   └── specformer.py        # MultiScaleSpectralTokenizer, _PreLNBlock, Branch D
│       │   ├── fusion.py            # CrossModalInteraction, EmbedNet
│       │   ├── heads.py             # AdaptiveSubcenterArcFaceHead, AuxiliaryHead
│       │   ├── stats_ops.py         # masked_spectral_stats, extract_grid_spectra
│       │   └── spectral_quadnet.py  # SpectralQuadNet — composition root only, no math of its own
│       │
│       ├── losses/
│       │   ├── __init__.py
│       │   ├── focal.py             # FocalLoss
│       │   ├── contrastive.py       # SupConLoss, ProtoNCELoss
│       │   ├── mixup.py             # _mixup, mixed_aug, mixed_loss
│       │   └── cdws.py              # build_cdws_weights
│       │
│       ├── optim/
│       │   ├── __init__.py
│       │   ├── sam.py               # SAM
│       │   ├── param_groups.py      # _wd_groups, build_optimizer_s1/s2/s3
│       │   └── schedulers.py        # sgdr_scheduler, arcface_margin, phase_aware_lr factory
│       │
│       ├── engine/
│       │   ├── __init__.py
│       │   ├── train_epoch.py       # train_one_epoch, train_one_epoch_sam
│       │   ├── evaluate.py          # _run_eval, evaluate, evaluate_per_class
│       │   ├── tta.py               # tta_predict
│       │   ├── checkpoint.py        # stage_*_path, stage_exists, latest_completed_stage,
│       │   │                        # save_ckpt, load_ckpt, load_stage_meta, _pick_best_checkpoint
│       │   ├── diagnostics.py       # compute_branch_influence, compute_class_difficulty,
│       │   │                        # + NEW: grad-norm-per-branch, hardest-class report
│       │   └── stages/
│       │       ├── __init__.py
│       │       ├── stage1_progressive.py  # run_stage1
│       │       ├── stage2_arcface.py      # run_stage2
│       │       ├── stage3_sam_swa.py      # run_stage3_swa
│       │       └── final_eval.py          # final_evaluation
│       │
│       ├── tracking/                # NEW — additive, see §4.1
│       │   ├── __init__.py
│       │   ├── base.py              # ExperimentTracker Protocol
│       │   ├── console_tracker.py   # Rich progress bars / tables — default backend
│       │   ├── wandb_tracker.py
│       │   ├── tensorboard_tracker.py
│       │   └── multi_tracker.py     # fan-out composite (log to N backends at once)
│       │
│       └── utils/
│           ├── __init__.py
│           ├── seed.py              # set_seed
│           └── device.py            # device resolution
│
├── scripts/                         # thin CLIs — argument parsing + config loading only
│   ├── prepare_dataset.py           # wraps spectralquadnet.data.prep.*  (was data_setup_v3.py)
│   ├── select_bands.py              # wraps spectralquadnet.data.prep.band_selection (was band_selection.py)
│   └── install_mamba_kernels.py     # renamed installation.py — flagged as optional/unused by current model
│
├── train.py                         # root CLI entrypoint, @hydra.main — replaces the `if __name__` block
│
├── tests/
│   ├── conftest.py                  # tiny synthetic-patch fixtures (no real dataset needed)
│   ├── unit/
│   │   ├── test_blocks.py           # shape/dtype tests for every block in models/blocks + branches
│   │   ├── test_losses.py           # FocalLoss/SupCon/ProtoNCE known-input → known-output
│   │   ├── test_samplers.py         # class balance invariants
│   │   ├── test_schedulers.py       # arcface_margin / sgdr_scheduler / phase_aware_lr — exact-value tests
│   │   ├── test_cdws.py             # build_cdws_weights known-input → known-output
│   │   └── test_seed_determinism.py # two `set_seed(42)` calls → identical weight init tensors
│   └── regression/                  # see §3.2
│       ├── test_golden_forward_pass.py
│       ├── test_state_dict_compatibility.py
│       └── golden/
│           ├── forward_logits_seed42.npy
│           ├── stage1_epoch1_loss_seed42.json
│           └── README.md            # capture procedure + source git SHA
│
├── notebooks/
│   ├── embedding_overview.ipynb
│   ├── pipeline_overview.ipynb
│   └── VIS_NIR_hyperspectral_modality.ipynb
│
├── dataset/                          # unchanged; large binaries stay gitignored
├── figures/                          # unchanged
└── outputs/
    └── output_v12_spa40/              # `git mv` of HSI_modality_training/output_v12_SPA40, bytes untouched
        ├── best_stage1.pth
        ├── best_stage2.pth
        ├── best_stage3.pth
        ├── stage1_meta.json
        ├── stage2_meta.json
        ├── stage3_meta.json
        ├── test_preds_noTTA.npy
        ├── test_preds_TTA.npy
        ├── test_targets.npy
        └── training.log
```

### Design rationale

- **`src/` layout** (not a flat top-level package) so `pip install -e .` gives a real installable
  package, `spectralquadnet`, importable from tests/notebooks/scripts without `sys.path` hacks.
- **Package boundaries mirror the file's own section banners** (`# ══ BRANCH A ══`, `# ══ SAM ══`,
  etc.) — the original author already organized the monolith by concern; the refactor formalizes
  those existing boundaries rather than inventing new ones.
- **`engine/stages/` mirrors the 3-stage curriculum 1:1** — `run_stage1`/`run_stage2`/`run_stage3_swa`
  become `stage1_progressive.py`/`stage2_arcface.py`/`stage3_sam_swa.py`. This keeps the curriculum's
  narrative structure legible instead of flattening it into a generic `trainer.py`.
  These modules are **orchestration only** — they call into `engine/train_epoch.py`,
  `engine/evaluate.py`, `engine/checkpoint.py`, `losses/`, `optim/` for their actual work, matching
  how the current functions already delegate to `train_one_epoch`, `evaluate`, `save_ckpt`, etc.
- **`data/prep/` absorbs `band_selection.py` and `data_setup_v3.py`** as importable, testable
  modules rather than leaving them as disconnected root scripts — `scripts/prepare_dataset.py` and
  `scripts/select_bands.py` become thin CLI wrappers, matching the pattern used for `train.py`.
- **`tracking/` is new** and has no equivalent in the current file (today: bare `print(...)`) — it is
  additive, not a relocation.

### 2.1 Old → New symbol migration (complete map)

| Old location (`hsi_training.py` line) | Symbol | New location |
|---|---|---|
| 34–143 | `CONFIG` | `configs/**/*.yaml` + `src/spectralquadnet/config/schema.py` |
| 163–170 | `_load_data_mmap`, `_GPU_PATCHES`, `_GLOBAL_LABELS` | `data/mmap_store.py::DataStore` |
| 173–187 | `_load_wavelengths_to_gpu`, `_PHYSICAL_WL` | `data/mmap_store.py::DataStore` |
| 194–200 | `set_seed` | `utils/seed.py` |
| 207–246 | `ModelEMA` | `models/ema.py` |
| 253–356 | `RiceSeedDataset` | `data/datasets.py` |
| 362–446 | `ClassBalancedBatchSampler`, `HardClassOversampledSampler` | `data/samplers.py` |
| 453–461 | `build_cdws_weights` | `losses/cdws.py` |
| 468–487 | `_mixup`, `mixed_aug`, `mixed_loss` | `losses/mixup.py` |
| 494–515 | `FocalLoss` | `losses/focal.py` |
| 518–559 | `SupConLoss`, `ProtoNCELoss` | `losses/contrastive.py` |
| 566–610 | `SAM` | `optim/sam.py` |
| 617–678 | `AdaptiveSubcenterArcFaceHead` | `models/heads.py` |
| 684–718 | `MaskedSpectralECA` | `models/blocks/attention.py` |
| 720–734 | `SEBlock1D` | `models/blocks/attention.py` |
| 738–754 | `ResBlock1D` | `models/blocks/conv_blocks.py` |
| 756–772 | `CBAM` | `models/blocks/attention.py` |
| 774–793 | `ResBlock2D` | `models/blocks/conv_blocks.py` |
| 796–811 | `PhysicalWavelengthPE` | `models/blocks/positional.py` |
| 817–843 | `LargeKernelBlock1D` | `models/blocks/conv_blocks.py` |
| 846–908 | `SpectralProfileBranch` | `models/branches/spectral_profile.py` |
| 914–988 | `SpectralStatsBranch` | `models/branches/spectral_stats.py` |
| 995–1024 | `SpatialCNNBranch` | `models/branches/spatial_cnn.py` |
| 1030–1164 | `MultiScaleSpectralTokenizer`, `_PreLNBlock`, `SpecFormerBranch` | `models/branches/specformer.py` |
| 1171–1283 | `CrossModalInteraction` | `models/fusion.py` |
| 1285–1304 | `EmbedNet` | `models/fusion.py` |
| 1310–1326 | `AuxiliaryHead` | `models/heads.py` |
| 1334–1365 | `compute_branch_influence` | `engine/diagnostics.py` |
| 1367–1421 | `extract_grid_spectra`, `masked_spectral_stats` | `models/stats_ops.py` |
| 1429–1631 | `SpectralQuadNet` | `models/spectral_quadnet.py` |
| 1638–1667 | `tta_predict` | `engine/tta.py` |
| 1674–1743 | `build_splits`, `build_loaders`, `build_phase3_loader` | `data/loaders.py` |
| 1750–1801 | `_wd_groups`, `build_optimizer_s1/s2/s3`, `sgdr_scheduler`, `arcface_margin` | `optim/param_groups.py`, `optim/schedulers.py` |
| 1807–1852 | `_aux_loss_weight`, `_compute_aux_loss` | `losses/mixup.py` or new `losses/auxiliary.py` |
| 1859–1961 | `train_one_epoch` | `engine/train_epoch.py` |
| 1967–2027 | `train_one_epoch_sam` | `engine/train_epoch.py` |
| 2034–2063 | `_run_eval`, `evaluate`, `evaluate_per_class` | `engine/evaluate.py` |
| 2070–2150 | checkpoint helpers | `engine/checkpoint.py` |
| 2157–2183 | `compute_class_difficulty` | `engine/diagnostics.py` |
| 2190–2388 | `run_stage1` | `engine/stages/stage1_progressive.py` |
| 2395–2501 | `run_stage2` | `engine/stages/stage2_arcface.py` |
| 2508–2617 | `run_stage3_swa` | `engine/stages/stage3_sam_swa.py` |
| 2624–2668 | `final_evaluation` | `engine/stages/final_eval.py` |
| 2675–2694 | `_pick_best_checkpoint` | `engine/checkpoint.py` |
| 2701–2858 | `main`, `if __name__` | `train.py` (root CLI) |
| — (root) | `band_selection.py` | `data/prep/band_selection.py` + `scripts/select_bands.py` |
| — (root) | `data_setup_v3.py` | `data/prep/{download,segmentation,patch_extraction}.py` + `scripts/prepare_dataset.py` |
| — (root) | `installation.py` | `scripts/install_mamba_kernels.py` (unchanged behavior, relocated only) |

---

## 3. Zero-Regression Guarantees

### 3.1 Checkpoint / state-dict compatibility (highest priority — real trained weights exist)

`save_ckpt`/`load_ckpt` call `model.state_dict()` / `model.load_state_dict()` — they persist **plain
tensors keyed by attribute path** (e.g. `"branch_a.stem.0.weight"`), never a pickled `nn.Module`
object. This means:

- **File relocation carries zero pickling risk.** There is no `torch.save(model)` anywhere in the
  codebase to break via changed import paths.
- **The only thing that can break compatibility is renaming an `nn.Module` attribute** inside
  `SpectralQuadNet.__init__` (`self.branch_a`, `self.branch_b`, `self.branch_c`, `self.branch_d`,
  `self.se`, `self.cross_interaction`, `self.embed_net`, `self.linear_head`, `self.arcface_head`,
  `self.aux_head_{a,b,c,d}`, `self.wl_pe_cnn`). The migration must keep every one of these attribute
  names byte-identical even as their *class definitions* move to new files.
- **Validation gate (Phase 5):** `tests/regression/test_state_dict_compatibility.py` loads all three
  real checkpoints (`outputs/output_v12_spa40/best_stage{1,2,3}.pth`) into a freshly constructed
  refactored `SpectralQuadNet` via `load_state_dict(strict=True)` and asserts zero missing/unexpected
  keys. This is a **hard gate** — Phase 2 (model migration) is not considered complete until it passes
  against the real artifacts, not synthetic ones.

### 3.2 Numerical identity validation

Three complementary techniques, ordered from cheapest/strongest to most expensive/weakest:

1. **AST-level "no-op move" diff.** Because this migration is a mechanical relocation, write a
   one-off verification script (used during Phase 2/3, not kept as a permanent test) that extracts
   each class/function body from the pre-refactor `hsi_training.py` (via `git show <pre-refactor-sha>`)
   and from its new file, parses both with `ast.parse`, and diffs `ast.dump(node, annotate_fields=False)`
   after stripping only the enclosing module qualifiers. Any change beyond whitespace/import
   rewiring fails the check. This catches accidental logic drift (e.g. an operator typo during
   copy-paste) that a shape-only test would miss.
2. **Golden forward-pass regression test** (`tests/regression/test_golden_forward_pass.py`, kept
   permanently). Procedure:
   - On the **pre-refactor** code, with `torch.manual_seed(42)` and `set_seed(42)` called immediately
     before model construction, build `SpectralQuadNet` with a frozen reference config, feed a fixed
     synthetic input (`torch.randn(4, 40, 64, 64)`, same seed), run one forward pass in `.eval()` mode,
     and save `out.detach().numpy()` to `tests/regression/golden/forward_logits_seed42.npy`, plus the
     scalar loss from one `train_one_epoch`-equivalent step on 32 synthetic samples to
     `stage1_epoch1_loss_seed42.json`. Record the exact git SHA these were captured from in
     `golden/README.md`.
   - On the **post-refactor** code, repeat the identical procedure and assert `np.allclose(new, golden,
     atol=1e-6)` for logits (float32 matmul order is unchanged since no math changed) and exact
     equality for the loss scalar.
   - This test is the **primary regression gate** for Phases 2–3 and must pass before Phase 4 begins.
3. **Pure-function exact-equality tests** (`tests/unit/test_schedulers.py`, `test_cdws.py`). Several
   components are already pure functions of primitives with no I/O or model dependency —
   `arcface_margin(ep, m0, m_target, warmup_ep)`, the `phase_aware_lr` closure inside `run_stage1`,
   `sgdr_scheduler`'s `_l(ep)`, and `build_cdws_weights`. These are the cheapest possible regression
   tests: call the old and new implementations across their full valid epoch range (e.g.
   `arcface_margin` for `ep in range(0, 20)` since `s2_margin_warmup_ep=20`) and assert bit-exact
   equality. This directly satisfies the user's explicit ask for **"dynamic margin schedules …
   remain 100% identical."**

### 3.3 Config round-trip validation

Every one of the ~70 `CONFIG` keys (§1.2, lines 34–143) must resolve to exactly one field somewhere
in `configs/**/*.yaml` — no key silently dropped, renamed, or given a new default during the
mechanical migration. Validation: a one-off script that loads the composed Hydra config for the
`output_v12_spa40` experiment, flattens it to a dict, and diffs its key set (after documented
renames — e.g. `s1_max_lr` → `stage1.max_lr` — captured in a rename table) against the original
`CONFIG` dict's key set from the pre-refactor commit. Any key present in one but not the other fails
the check.

### 3.4 mmap zero-RAM data loading — preserved exactly

Current implementation (`_load_data_mmap`, line 163–170; `RiceSeedDataset.__getitem__`, line 335–356):
`np.load(patches_path, mmap_mode='r')` creates a memory-mapped array; only `RiceSeedDataset.__getitem__`
ever touches it, and only via `np.array(self.patches[ri])` — a per-item copy that pages in exactly one
`(40, 64, 64)` patch from disk, never materializing the full 5.6 GB array in RAM.

The refactor replaces the three module-level globals (`_GPU_PATCHES`, `_GLOBAL_LABELS`, `_PHYSICAL_WL`)
with a `DataStore` singleton in `data/mmap_store.py` that `RiceSeedDataset` receives via constructor
injection (or a module-level lazy singleton, matching today's lazy-init guard at line 165:
`if _GPU_PATCHES is not None: return`). **Non-negotiable invariants to preserve:**
- `mmap_mode='r'` flag unchanged (read-only mapping — no accidental `mmap_mode=None` full load).
- The per-item `np.array(...)` copy pattern unchanged (this is what keeps RAM bounded — do not
  introduce a batched `.copy()` over the full array anywhere in the loader path).
- Load-once guard semantics unchanged (re-importing/re-instantiating must not re-mmap or duplicate
  the file handle).
- Validated via a test that mocks a small `.npy` file, instantiates `DataStore` twice, and asserts
  the second call is a no-op (same object identity), plus a memory-usage smoke check (process RSS
  stays within a small bound after touching a few thousand random indices) run manually against the
  real 5.6 GB file in Phase 5 (not part of CI, since it needs the real dataset).

### 3.5 Multi-stage checkpointing — filenames, schema, and auto-resume logic preserved exactly

`stage_ckpt_path`/`stage_meta_path` (lines 2070–2076) produce `best_stage{1,2,3}.pth` and
`stage{1,2,3}_meta.json` under `CONFIG["output_dir"]`. The refactored `engine/checkpoint.py` must:
- Keep these exact filename templates (only `output_dir`'s *value* moves from a hardcoded absolute
  path to a config field — see tech-debt note in §1.3 — the templates themselves do not change).
- Keep the `save_ckpt` bundle schema unchanged: `{"epoch", "stage", "model", "ema", "val_f1",
  "val_acc", "use_arcface", **metadata}` plus the JSON sidecar that strips `"model"`/`"ema"` and
  keeps only JSON-serializable metadata (`_is_json_serialisable`, line 2109).
- Keep `latest_completed_stage()`'s resume logic (checks stage 3→2→1 in that order via
  `stage_exists`, which requires *both* the `.pth` and `.json` to exist) so that pointing a new
  config's `output_dir` at the existing `outputs/output_v12_spa40/` directory correctly reports
  "Stages 1–2 done" / resumes into Stage 3, exactly as `main()` does today (lines 2701–2713).
- Keep `_pick_best_checkpoint`'s val_f1-based selection logic (lines 2675–2694) unchanged.
- **Validation:** point a refactored `train.py` run's config at the *existing*
  `outputs/output_v12_spa40/` directory (after the `git mv` in Phase 1) with all 3 stages already
  complete, and assert it correctly detects `done_stage == 3`, skips straight to `final_evaluation`,
  and reproduces the same Macro F1 reported in `stage3_meta.json` (0.8745) on the test split — this
  is the strongest possible end-to-end regression signal because it requires every one of §3.1–3.5 to
  be correct simultaneously.

### 3.6 Random seed behavior — the highest-risk item in this migration

**The hazard:** `set_seed(CONFIG["seed"])` currently executes as a **module import-time side effect**
(line 200, immediately after the `CONFIG` dict is defined). Every `nn.init.kaiming_normal_` /
`trunc_normal_` / `xavier_uniform_` call inside every module's `_init_weights()` — and there are
several, one per branch plus the top-level `SpectralQuadNet._init_weights()` — consumes the **same
global PyTorch RNG stream**, in the exact order those modules are constructed inside
`SpectralQuadNet.__init__` (branch A → B → C → D → fusion → heads, per the constructor body at
lines 1466–1519).

If the refactored `train.py` calls `set_seed()` at a different point relative to model construction
than the original import-time call did — e.g. after Hydra has already done some other RNG-consuming
work, or after `DataStore` initialization touches `torch.from_numpy` in a way that consumes global
state — **initial weights will differ from the original checkpoints' training lineage**, even though
loading the *existing* checkpoints (§3.1) is unaffected. This only matters for **new training runs**
compared byte-for-byte against a fresh reference run, not for resuming/evaluating existing ones.

**Migration requirement:** `train.py`'s call order must be, explicitly and in this sequence:
`load config → set_seed(cfg.seed) → construct DataStore → construct SpectralQuadNet → construct
ModelEMA → ...`, mirroring the effective order the import-time side effects produced today
(`set_seed` at line 200 runs before `main()` — and hence before model construction at line 2726 — is
ever called, since it's at module scope). Document this ordering as a code comment at the call site,
not just in this plan.

**Honest scope of "100% identical":** Per §1.3, the current code already contains two
non-deterministic islands independent of `set_seed()` — `cudnn.benchmark=True` (non-deterministic
conv algorithm selection) and the unseeded `np.random.default_rng()` calls inside both custom
samplers. **Bit-exact reproducibility of a full training run was never a property of the pre-refactor
code**, so this is not a regression the refactor can introduce. The verifiable, testable guarantee is
narrower and is exactly what §3.2's golden test checks: **identical weight initialization** (seed → model
construction, zero non-determinism), **identical scheduler/margin values** (pure functions, zero
non-determinism), and **identical loss on one fixed-seed, fixed-data forward+backward step** (the only
non-determinism source — cudnn kernel selection — is disabled for the regression test via
`torch.use_deterministic_algorithms(True)` / `cudnn.deterministic=True`, which the *original* script
never sets but which is a legitimate, additive setting for a *test harness* rather than for training
itself).

---

## 4. Modern Research & Observability Enhancements

### 4.1 Experiment tracking abstraction

```python
# tracking/base.py
class ExperimentTracker(Protocol):
    def log_scalar(self, tag: str, value: float, step: int) -> None: ...
    def log_scalars(self, tags: dict[str, float], step: int) -> None: ...
    def log_table(self, tag: str, rows: list[dict], step: int) -> None: ...   # e.g. per-class F1
    def log_hyperparams(self, cfg: dict) -> None: ...
    def watch(self, model: nn.Module) -> None: ...                            # optional grad histograms
    def close(self) -> None: ...
```

- `console_tracker.py` wraps every existing `print(f"Ep {ep:03d}/{ep_total} │ Loss {tl:.4f} ...")`
  call site (e.g. line 2378, 2491, 2585) with a `rich.progress`/`rich.table` renderer — same
  information density, readable in a terminal, zero external service dependency, and the **default**
  backend so the framework works offline out of the box.
- `wandb_tracker.py` / `tensorboard_tracker.py` implement the same protocol; selected via
  `configs/tracking/{wandb,tensorboard}.yaml`.
- `multi_tracker.py` fans out to N backends simultaneously (e.g. console + W&B in the same run) —
  useful during the migration itself, to visually diff old console output against new structured
  logs for the same run.
- Every stage function (`stage1_progressive.py` etc.) receives a `tracker: ExperimentTracker`
  instead of calling `print` directly; this is the **only** behavioral change to the stage functions'
  bodies beyond import-path updates, and it's purely additive (the values logged already exist as
  local variables in the current code — `tl, ta, f1_live, acc_live, f1_ema, acc_ema, lr_now`, etc.).

### 4.2 Deep inspection & diagnostics

All four requested capabilities build directly on existing return values — none require new math:

- **Per-branch loss contribution.** `_compute_aux_loss` (line 1826) already computes
  `aux_a/aux_b/aux_c/aux_d` losses individually before summing them with branch weights
  (`{"aux_a": 2.0, "aux_b": 2.0, "aux_c": 1.0, "aux_d": 1.0}`, line 1843). Change: return the
  per-branch dict alongside the summed total instead of only the total, and log each component via
  `tracker.log_scalars({"loss/branch_a": ..., ...}, step=ep)`.
- **Gradient norm flow.** `train_one_epoch`/`train_one_epoch_sam` already call
  `nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])` (lines 1946, 2006, 2019), which
  returns the *total* pre-clip norm. Add a diagnostics helper in `engine/diagnostics.py` that,
  reusing the exact same attribute-prefix filtering pattern `_wd_groups` already uses (line
  1750–1759, matching on `n.startswith(...)`), computes the norm restricted to each branch's named
  parameters (`branch_a.`, `branch_b.`, `branch_c.`, `branch_d.`, `cross_interaction.`,
  `arcface_head.`) before the clip call, and logs them — a direct, cheap generalization of code that
  already exists.
- **Branch influence percentage.** `compute_branch_influence` (line 1334) already computes this via
  leave-one-branch-out KL divergence ablation and is already called from `compute_class_difficulty`
  (line 2174) at every stage's checkpoint-improvement event. Change: route its returned
  `{"A": ..., "B": ..., "C": ..., "D": ...}` dict to `tracker.log_scalars` instead of only the
  existing `print` (line 2176–2182) — the computation itself is untouched.
- **Per-class failure analysis / hardest classes.** `evaluate_per_class` (line 2058) and the existing
  hard-class threshold logic already scattered across `HardClassOversampledSampler` (line 426–438,
  `hard_f1_thresh`) and `build_cdws_weights` give per-class F1 and derived weights. New (additive)
  `engine/diagnostics.py::hardest_classes_report(class_f1, k=10)` sorts and formats the bottom-K
  classes into a table for `tracker.log_table`, called once per stage after `final_evaluation`/each
  checkpoint-improvement event — a thin wrapper, not new statistics.

### 4.3 Configuration management (Hydra)

- `src/spectralquadnet/config/schema.py` defines `@dataclass` schemas (`DataConfig`, `ModelConfig`,
  `Stage1Config`, `Stage2Config`, `Stage3Config`, `TrackingConfig`, `ExperimentConfig`) registered
  with Hydra's `ConfigStore` — giving static type-checking on every field mypy can verify, plus
  runtime validation on load (catches typos in YAML that a plain `dict` would silently swallow, unlike
  today's `CONFIG[...]` lookups which raise `KeyError` only when the missing key is first accessed,
  possibly deep into a multi-hour Stage 1 run).
- `train.py` becomes `@hydra.main(config_path="configs", config_name="experiment/output_v12_spa40")`,
  supporting CLI overrides (`python train.py stage1.max_lr=1e-4`) and multirun sweeps
  (`python train.py -m stage1.max_lr=1e-4,5e-4,1e-3`) for free — directly useful given how many
  hand-tuned constants already exist in the curriculum (phase fractions, margin deltas, oversample
  power, etc.).
- `output_dir` becomes a config field resolved from a `data_root`/`run_name` pair rather than a
  hardcoded absolute path (fixes the tech-debt item in §1.3 — this *is* an intended behavior change,
  called out explicitly rather than silently folded into the "mechanical move").

### 4.4 Developer quality

- **Typing:** the current file already type-hints most signatures (`from __future__ import
  annotations` at line 2, extensive `Optional`/`Tuple`/`Dict` usage) — the refactor's job is to
  preserve and complete this coverage (a few helpers like `EmbedNet.forward`, line 1301, currently
  have untyped `x`) and add `mypy --strict` to CI.
- **Docstrings:** convert existing prose docstrings (e.g. `RiceSeedDataset`, `SAM`-adjacent classes
  are currently under-documented; `AdaptiveSubcenterArcFaceHead` has none) to Google-style with
  `Args:`/`Returns:`/`Shape:` sections, prioritizing the four branches and `SpectralQuadNet.forward`
  since those are what new contributors will read first.
- **Lint/format:** `ruff` (lint + import sorting, replacing the need for a separate isort config),
  `black` (formatting), `mypy` (types), wired into `pyproject.toml` and `.pre-commit-config.yaml`, run
  in CI on every PR.

---

## 5. Step-by-Step Migration Roadmap

Each phase ends with a **validation gate** that must pass before the next phase starts. All work
happens on a branch; `main` is untouched until Phase 5 signs off.

### Phase 1 — Structure & Configs
- [ ] Create `src/spectralquadnet/` package skeleton (empty `__init__.py` files per §2 tree) and
      register it in a new `pyproject.toml` (`pip install -e .` must succeed with no other changes).
- [ ] Transcribe every `CONFIG` key (lines 34–143) into `configs/**/*.yaml`, grouped exactly as shown
      in §2; write `config/schema.py` dataclasses matching them field-for-field.
- [ ] Write and run the config round-trip check (§3.3); record the key-rename table it depends on.
- [ ] `git mv HSI_modality_training/output_v12_SPA40 outputs/output_v12_spa40` (byte-identical
      contents — verify with `md5sum` before/after on all `.pth`/`.json` files).
- [ ] `git mv` the three notebooks into `notebooks/`.
- **Gate:** `pip install -e .` succeeds; config round-trip check passes; `md5sum` diff on moved
      checkpoint files is empty; no `.py` files touched yet.

### Phase 2 — Data & Models
- [ ] Relocate `data/prep/*` from `data_setup_v3.py` and `band_selection.py` (verbatim function
      bodies; only `CONFIG` dict access becomes config-object field access).
- [ ] Relocate `data/mmap_store.py`, `data/datasets.py`, `data/samplers.py`, `data/loaders.py`
      (§2.1 rows for lines 163–446, 1674–1743).
- [ ] Relocate every model file under `models/` (§2.1 rows for lines 207–246, 617–1631) — this is the
      largest single chunk of the migration; do it class-by-class, one file per commit, to keep the
      AST-diff check (§3.2.1) reviewable per-commit rather than as one giant diff.
- [ ] Run the AST-level "no-op move" diff (§3.2.1) against the pre-refactor commit for every relocated
      class.
- [ ] Capture golden values (§3.2.2) from the **pre-refactor** code if not already captured in Phase 0
      of implementation; then run the same procedure against the **post-refactor** model construction
      + forward pass and assert match.
- [ ] Run `test_state_dict_compatibility.py` (§3.1) against all three real checkpoints in
      `outputs/output_v12_spa40/` — **hard gate, must be `strict=True` clean**.
- **Gate:** AST-diff clean; golden forward-pass test passes (`atol=1e-6`); all 3 real checkpoints load
      with zero missing/unexpected keys.

### Phase 3 — Engine & Losses
- [ ] Relocate `losses/*` (§2.1 rows for lines 453–559) and `optim/*` (566–610, 1750–1801).
- [ ] Write `test_schedulers.py` and `test_cdws.py` (§3.2.3) — pure-function exact-equality tests,
      run against both pre- and post-refactor implementations during development.
- [ ] Relocate `engine/train_epoch.py`, `engine/evaluate.py`, `engine/tta.py`,
      `engine/checkpoint.py`, `engine/diagnostics.py` (§2.1 rows for lines 1334–1421, 1638–1667,
      1807–2183).
- [ ] Relocate the three stage orchestrators and `final_eval.py` (§2.1 rows for lines 2190–2668),
      wiring them to the relocated `engine/`, `losses/`, `optim/` modules — **no logic changes**, only
      import updates and `CONFIG[...]` → `cfg.stage1.xxx` style attribute access.
- [ ] Golden loss-value test (§3.2.2's second artifact, `stage1_epoch1_loss_seed42.json`): run one
      real epoch-equivalent step (32 synthetic samples, fixed seed) through the relocated
      `train_one_epoch` and assert scalar loss matches the pre-refactor capture exactly.
- **Gate:** scheduler/CDWS pure-function tests pass with bit-exact equality; golden loss-value test
      passes; `mypy` clean on all relocated engine/loss/optim modules.

### Phase 4 — CLI & Logging
- [ ] Implement `tracking/` (base protocol + console/wandb/tensorboard/multi backends, §4.1).
- [ ] Wire `tracker` parameter through the three stage orchestrators, replacing `print(...)` call
      sites one-for-one (§4.1's explicit scope note: this is the only intentional behavioral touch to
      stage function bodies, and it's additive/observability-only).
- [ ] Implement the diagnostics additions from §4.2 (per-branch loss logging, per-branch grad norm,
      hardest-classes report) as thin wrappers around existing computations.
- [ ] Build `train.py` as the Hydra entrypoint (§4.3), replacing `main()` + `if __name__` (lines
      2701–2858) with identical auto-resume orchestration logic, now reading from `cfg` instead of
      `CONFIG`.
- [ ] Rebuild `set_seed()` call-site ordering exactly per §3.6's required sequence; add the
      call-order comment at the `train.py` call site.
- [ ] Build `scripts/prepare_dataset.py`, `scripts/select_bands.py`,
      `scripts/install_mamba_kernels.py` as thin CLI wrappers around `data/prep/*`.
- **Gate:** `python train.py` run against `outputs/output_v12_spa40/` (all 3 stages already complete)
      correctly detects `done_stage == 3`, skips training, and reproduces `stage3_meta.json`'s
      Macro F1 (0.8745) on `final_evaluation` — this is the §3.5 end-to-end validation and the
      strongest signal in the whole plan, since it exercises config loading, data loading, model
      construction, checkpoint loading, and evaluation together against real artifacts.

### Phase 5 — Verification & Testing
- [ ] Full `pytest tests/` run (unit + regression) green in CI.
- [ ] `ruff`, `black --check`, `mypy --strict` clean across `src/`, `scripts/`, `train.py`.
- [ ] Manual smoke test: run a **short** fresh training (e.g. 3 epochs of Stage 1 on a config with
      `s1_epochs: 3`) end-to-end through `train.py`, confirm no crashes, confirm checkpoint/meta files
      are written with the exact expected schema (§3.5), confirm tracker output (console at minimum)
      is legible.
- [ ] Real-dataset mmap memory check (§3.4): run the above smoke test under a memory profiler,
      confirm RSS stays bounded (not proportional to the 5.6 GB dataset size).
- [ ] Delete `HSI_modality_training/hsi_training.py` (and the empty `HSI_modality_training/`
      directory) **only after** every above gate is green — this is the only step in the entire plan
      that removes code, and it happens last, once every guarantee in §3 has been independently
      re-verified against the new tree.
- [ ] Update `README.md` (currently one line) with the new structure, install instructions, and
      `python train.py` usage.
- **Gate:** all of the above green; PR reviewed and merged.

---

## 6. Non-Goals (explicitly out of scope for this migration)

- No architecture changes to any branch, fusion module, or head.
- No hyperparameter changes anywhere in `configs/` relative to the current `CONFIG` dict's values.
- No fixing of the dead `wl_embed_dim`/`patch_size` parameters (§1.3) — noted for a separate,
  reviewable follow-up cleanup PR after Phase 5 sign-off, not bundled into this migration.
- No dependency version upgrades beyond the net-new packages this plan introduces (`hydra-core`,
  `omegaconf`, `rich`, optionally `wandb`/`tensorboard`, `ruff`/`black`/`mypy`/`pytest`/`pre-commit`).
- No action taken on the ~30 already-deleted legacy training-script iterations currently staged in
  `git status` — this plan documents a target structure to build once those pending deletions are
  resolved by the user, not a mandate to resolve them.
- No changes to `dataset/`, `figures/`, or any binary/notebook content.

---

## 7. Open Questions for the User

1. **Package/import name** — is `spectralquadnet` acceptable, or is there a preferred name (e.g. tied
   to a paper title or lab convention)?
2. **Hydra vs. lighter-weight OmegaConf-only config** — Hydra gives multirun sweeps and structured
   configs "for free" but adds a dependency and a `@hydra.main` decorator convention some users find
   heavier than plain `argparse` + `OmegaConf.load`. §4.3 assumes Hydra; flag if a lighter approach is
   preferred.
3. **`installation.py`/Mamba dependency** — confirmed unused by the current `SpectralQuadNet` (no
   `mamba_ssm`/`causal_conv1d` imports exist). Keep as `scripts/install_mamba_kernels.py` for a future
   SpecMamba branch, or drop it entirely from this repo?
4. **Timing relative to the pending `git status` deletions** — should this migration be implemented
   as a fresh commit on top of the current working tree (deletions + modification + new files all
   staged together), or should those be resolved/committed first as a separate, prior commit?
