# SpectralQuadNet

A four-branch hyperspectral CNN/Transformer that classifies **90 rice seed varieties** from
40-band VIS-NIR (385–1000 nm) 64×64 patches, trained with a three-stage curriculum and evaluated
with 12-view test-time augmentation.

| Metric (held-out test split, 1,294 patches) | Score |
|---|---|
| Macro F1 — 12-view TTA | **0.8933** |
| Macro F1 — no TTA | 0.8770 |
| Best validation Macro F1 (Stage 1 checkpoint) | 0.8877 |

Reference artifacts live in `outputs/output_v12_spa40/` (three checkpoints + metadata sidecars +
recorded predictions). The numbers above are regenerated from them by
`tests/regression/test_resume_and_final_eval.py`.

---

## Architecture

`SpectralQuadNet` (7.88M parameters) fuses four views of the same patch:

| Branch | Module | Sees |
|---|---|---|
| **A** — spectral profile | `models/branches/spectral_profile.py` | per-region mean spectra on a 4×4 grid |
| **B** — spectral statistics | `models/branches/spectral_stats.py` | 9 background-masked per-band statistics (mean, std, max, skew, kurtosis, p10/p25/p75/p90) |
| **C** — spatial CNN | `models/branches/spatial_cnn.py` | 2-D texture, band-reduced |
| **D** — SpecFormer | `models/branches/specformer.py` | multi-scale spectral tokens through a pre-LN transformer |

The branches are fused by a Perceiver-style latent cross-attention block
(`models/fusion.py::CrossModalInteraction`), then classified by either a linear head or an
adaptive sub-centre ArcFace head (`models/heads.py`). Every branch also carries its own auxiliary
head for deep supervision. Wavelengths enter the model physically, not positionally, via
`models/blocks/positional.py::PhysicalWavelengthPE`.

### Training curriculum

| Stage | Module | What it does |
|---|---|---|
| 1 | `engine/stages/stage1_progressive.py` | 3-phase progressive augmentation (heavy → medium → very light), phase-aware LR, mixup, hard-class oversampling and contrastive losses in phase 3 |
| 2 | `engine/stages/stage2_arcface.py` | sub-centre ArcFace with a warmed-up adaptive per-class margin, SGDR, SupCon + ProtoNCE, class-difficulty-weighted sampling |
| 3 | `engine/stages/stage3_sam_swa.py` | Sharpness-Aware Minimisation with greedy SWA snapshotting |
| — | `engine/stages/final_eval.py` | 8 spatial + 4 spectral TTA views, per-class report |

---

## Install

Requires Python ≥ 3.10.

```bash
git clone <this-repo> && cd Code
pip install -e .                 # training + evaluation
pip install -e ".[dev]"          # + pytest, ruff, black, mypy
pip install -e ".[tracking]"     # + wandb, tensorboard backends
pip install -e ".[prep]"         # + opencv, scikit-image, spectral, scipy (offline data prep only)
```

The core install deliberately excludes the data-prep stack: `spectralquadnet.data.prep.*` is the
only thing that imports it, and training never touches that path.

### Data

Training reads three files, all configured in `configs/data/spa40_90class.yaml`:

```
dataset/patches_spa_40b.npy      # (8624, 40, 64, 64) float32 — 5.65 GB, memory-mapped
dataset/labels.npy               # (8624,)
dataset/wavelengths_spa_40b.csv  # 40 band centres in nm
```

To rebuild them from the raw Zenodo archive:

```bash
python scripts/prepare_dataset.py   # download → Otsu segmentation → 64×64 patch extraction (256 bands)
python scripts/select_bands.py      # mRMR + SPA + validated elbow → the 40-band subset
```

The patch cube is **never loaded into RAM**. `data/mmap_store.py` opens it with `mmap_mode="r"` and
`RiceSeedDataset.__getitem__` copies exactly one `(40, 64, 64)` patch per item, so resident memory
stays bounded rather than scaling with the cube. Measured over a full 3-stage run against the real
5.65 GB file: **peak RSS 1.39 GB, median 0.65 GB**, with mean usage *falling* from 0.71 GB in the
first half of the run to 0.55 GB in the second — clean mapped pages being reclaimed, not a dataset
accumulating in memory.

---

## Usage

```bash
python train.py                                   # the reference experiment
python train.py stage1.max_lr=1e-4                # single override
python train.py -m stage1.max_lr=1e-4,5e-4,1e-3   # Hydra sweep
python train.py run_name=my_run stage1.epochs=20  # short run into outputs/my_run
```

### Hardware and mixed precision

`device: auto` selects the fastest local accelerator — **Metal (MPS) → CUDA → CPU** — so an Apple
Silicon machine trains on its GPU. Override with `device=cuda`, `device=cpu` or `device=mps`; an
explicit choice is never overridden. On an M-series Air this is worth roughly a **12× speedup** over
the CPU fallback (a 7-epoch 3-stage run: ~18 min on Metal vs. ~3.5 h on CPU).

Stage 1 trains under `torch.amp.autocast` with a `GradScaler` bound to the active device. Stages 2
and 3 run in fp32 by the existing `use_amp = (supcon is None) and (scaler is not None)` rule — the
contrastive losses and SAM's two-step gradient are why, and that is unchanged from the original
script.

Two Metal-specific details are handled in `utils/device.py`:

- **`GradScaler(device=...)`** — the original bare `GradScaler()` binds to CUDA, so on any other
  accelerator it disables itself and AMP silently becomes a no-op.
- **`update_bn_stats` keeps grad enabled on Metal.** It is the only place the model runs in
  `train()` mode under `no_grad`, and Metal routes attention through a fused inference kernel that
  raises `scaled_dot_product_attention for MPS does not support dropout`. Grad mode selects the math
  path; forward values, and so the BatchNorm statistics being estimated, are identical.

Configuration is Hydra-composed from `configs/`, with dataclass schemas in
`config/schema.py` giving startup-time validation — a typo in a YAML field fails immediately
instead of hours into Stage 1.

```
configs/
├── data/spa40_90class.yaml        paths, num_bands, num_classes
├── model/spectral_quadnet_v4.yaml branch/fusion/head hyperparameters
├── stage{1,2,3}/*.yaml            one file per curriculum stage
├── tracking/*.yaml                none | console | wandb | tensorboard
└── experiment/output_v12_spa40.yaml   composes the above; sets seed and output_dir
```

### Auto-resume

`train.py` probes for completed stages (3 → 2 → 1; a stage counts as done only when **both** its
`best_stage{n}.pth` and `stage{n}_meta.json` exist) and loads rather than retrains them. Pointing
`output_dir` at a directory with all three present skips straight to final evaluation:

```bash
python train.py output_dir=outputs/output_v12_spa40
```

The checkpoint the final evaluation runs on is chosen by validation F1, not by stage order.

### Experiment tracking

The default `console` backend renders the same information the original script printed, via `rich`,
with no external service. `wandb` and `tensorboard` implement the same `ExperimentTracker` protocol
and are machine-channel only, so pair them with the console renderer to keep terminal output:

```bash
python train.py tracking.backend=multi tracking.backends=[console,wandb]
```

Beyond scalar losses and metrics, the trackers receive per-branch auxiliary losses, per-branch
gradient norms sampled before clipping, leave-one-branch-out influence percentages, and a
bottom-K hardest-classes table.

---

## Repository layout

```
├── train.py                      Hydra entrypoint (auto-resume orchestration)
├── configs/                      Hydra config groups
├── src/spectralquadnet/
│   ├── config/                   dataclass schemas + programmatic composition
│   ├── data/                     DataStore (mmap), dataset, samplers, loaders
│   │   └── prep/                 offline: download, segmentation, patches, band selection
│   ├── models/                   blocks/, branches/, fusion, heads, stats_ops, SpectralQuadNet
│   ├── losses/                   focal, contrastive, mixup, cdws, auxiliary
│   ├── optim/                    SAM, param groups, schedulers
│   ├── engine/                   train_epoch, evaluate, tta, checkpoint, diagnostics
│   │   └── stages/               one module per curriculum stage + final_eval
│   ├── tracking/                 ExperimentTracker protocol + 4 backends
│   └── utils/                    seed, device
├── scripts/                      thin CLIs + migration verification tooling
├── tests/                        unit/ + regression/ (+ golden/ captures)
├── docs/config_rename_table.md   every pre-refactor CONFIG key → its new home
└── REFACTOR_PLAN.md              the migration spec this structure was built to
```

---

## Development

```bash
pytest tests/            # unit + regression
ruff check src scripts train.py tests
black --check src scripts train.py tests
mypy                     # --strict, configured in pyproject.toml
```

This package was mechanically decomposed from a single 2,857-line script (`hsi_training.py`, plus
three root-level data scripts), all of which Phase 5 deleted. The decomposition is still
machine-verifiable: every check reads the originals from git at SHA `886560f` via `git show`, never
from the working tree, so they keep working with the files gone. Three checks enforce that the move
introduced no behavioural drift:

| Check | Guarantees |
|---|---|
| `pytest tests/regression/` | the three real checkpoints load `strict=True`; a fixed-seed forward pass matches its pre-refactor golden logits; the recorded metrics regenerate |
| `python scripts/check_ast_no_op_move.py` | every relocated class/method is AST-identical to the pre-refactor original, or carries a written deviation reason |
| `python scripts/check_config_roundtrip.py` | all 81 keys of the original `CONFIG` dict map 1:1 onto `configs/`, with identical values |

Current status: **107 tests passing**, AST check **133 identical / 41 declared / 3 new / 0 drift**,
config round-trip **81/81**, and `ruff` / `black --check` / `mypy --strict` clean.

### Known non-determinism (pre-existing, deliberately preserved)

A full training run was never bit-reproducible, and the refactor did not change that:
`set_seed()` sets `cudnn.benchmark=True`, and both custom samplers draw from an unseeded RNG. What
*is* pinned and tested: weight initialisation, every scheduler and margin value, and the loss of one
fixed-seed forward+backward step.

---

## References

The band-selection pipeline implements mRMR (Peng et al., *IEEE TPAMI* 2005) and SPA (Araújo et al.,
*Chemom. Intell. Lab. Syst.* 2001); see `src/spectralquadnet/data/prep/band_selection.py` for the
full rationale.
