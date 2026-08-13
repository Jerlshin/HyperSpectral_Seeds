# SpectralSeedNet — rice-variety classification from VIS-NIR hyperspectral seed images

90 rice varieties, 8,624 single-kernel patches, 40 SPA-selected bands, 64×64 spatial.
Source: [Zenodo 3241923](https://zenodo.org/records/3241923) (Vu et al., Strathclyde).

---

## Read this first

This repository was restructured in response to an independent audit
([`CHANGES.md`](CHANGES.md)). The audit's central finding was not about the architecture:

> The source dataset images each variety as **two bundles of 48 kernels, each bundle a tray of
> one single variety**. The original run split at the **patch** level, so all 180 acquisition
> bundles appeared in training *and* in evaluation. A model that learns "this tray's residual
> radiometric signature ⇒ class X" scores correctly. The reported 0.847 macro-F1 was a mixture
> of variety recognition and acquisition-bundle recognition, and the mixing ratio had never been
> measured.

So the headline number this project aims to produce is **not** an accuracy. It is:

> *How much of rice-seed HSI classification performance on this dataset is variety recognition,
> and how much is acquisition recognition?*

No published work on this dataset that the audit could access answers that. Everything below is
organised around making that question answerable.

**Consequences you should expect:**

- The **grouped** (leave-one-acquisition-bundle-out) number will be **lower** than the
  patch-level one, probably substantially. That is the correct direction — the previous number
  was measuring something else.
- Published work on this exact dataset reports 92.73–96.17% precision
  ([Taheri et al. 2024](https://doi.org/10.1007/s12652-023-04716-4)) without stating a
  bundle-disjoint protocol. Those figures are comparable to this project's **stratified** arm,
  not its grouped arm.
- Chasing >95% by keeping the patch-level split would reproduce the field's error rather than
  correct it. See `CHANGES.md` Q11.

---

## Quick start

```bash
# 1. Install (Python ≥ 3.10)
pip install -e ".[tracking,figures]"

# 2. Build the dataset (~hours, ~36 GB; writes patches/labels/groups/masks/morphology/gain)
pip install -e ".[prep]"
python scripts/prepare_dataset.py

# 3. Reduce 256 bands to 40, INSIDE each fold (see "Band selection" below)
python scripts/select_bands.py --per-fold

# 4. Train the default experiment
python train.py
```

`python train.py` with no arguments runs the **default composition**:

| | |
|---|---|
| Architecture | `SpectralSeedNet` — 2.82 M parameters, two pathways |
| Curriculum | one stage, ~150 epochs, early stop on `calib` |
| Split | `grouped` — leave-one-acquisition-bundle-out, fold 0 |
| Selection | on `calib` (carved from train, by group) |
| Reported | `val ∪ test`, scored **once**, ±TTA, with a bootstrap CI |
| Runtime | bf16, TF32 **off**, workers auto, compile auto |

---

## Repository layout

```
configs/                    Hydra composition
  data/                     spa40_90class{,_pfix,_stratified}
  model/                    seed_net | quadnet_v4_audited
  single/                   one_stage            ← the collapsed curriculum
  stage{1,2,3}/             the audited three-stage curriculum (kept for A8)
  evaluation/               held_out_once | audited_replica
  experiment/               seednet_grouped (default) | quadnet_audited (control)

src/spectralquadnet/
  config/                   typed schema + programmatic composition
  data/                     mmap store, splits, samplers, augmentation
    prep/                   offline: download → segment → extract → band-select
  models/
    registry.py             cfg.model.arch  →  a network
    spectral_seed_net.py    the proposed model (CHANGES §16.2)
    spectral_quadnet.py     the audited model — retained as the control arm
    branches/, blocks/      shared components
  engine/
    pipelines/              context + single | three_stage dispatch
    stages/                 single_stage, stage1/2/3, final_eval
    train_epoch.py          the AdamW and SAM epoch loops
  losses/, optim/           objectives, param groups, schedules
  reporting/                metrics + CIs, results tree, figures, tables
  experiments/              the ablation grid, protocol sweep, baselines, A9
  tracking/                 console | wandb | tensorboard | multi

scripts/                    thin CLI wrappers over the package
tests/unit/ regression/ smoke/
```

---

## Running experiments

Everything is driven by one CLI. **Start with `--dry-run`** — it costs nothing and prints the
exact per-cell command, which is the artifact a reviewer should see before a GPU-day is spent.

```bash
python -m spectralquadnet.experiments.cli list          # the grid, its cost, its ordering
```

### The primary protocol (CHANGES §19)

2 folds × 3 seeds under `grouped`, plus a matched `stratified` contrast arm:

```bash
python scripts/run_protocol.py --dry-run
python scripts/run_protocol.py --baseline          # + LDA/LinearSVC on mean spectra
python scripts/run_protocol.py --include-audited   # + the 5.19 M model, same protocol
```

Produces, under `outputs/experiments/protocol/`:

- `protocol.md` / `.csv` — mean ± range per arm, **never a maximum**
- `leakage_gap.md` — `F1_stratified − F1_grouped`, the headline result
- `per_cell.md` — every individual run, so the means are auditable
- `protocol.png` — the comparison figure

### Ablations (CHANGES §20)

```bash
python -m spectralquadnet.experiments.cli ablate A12   # ← RUN THIS FIRST
python -m spectralquadnet.experiments.cli ablate A1
python -m spectralquadnet.experiments.cli ablate A3 --arms abcd bc --dry-run
```

| | Question | Runs |
|---|---|---:|
| **A12** | What is run-to-run variance? **Run first** — until σ is known, no delta means anything | 10 |
| **A1** | How much of the score is bundle recognition? **Blocks every other claim** | 12 |
| **A2** | Does band selection outside the fold leak materially? | 12 |
| **A3** | Is the four-branch design justified? (symmetric dropout) | 24 |
| **A4** | Is Branch A's 64-cell replication necessary? | 18 |
| **A5** | Is the rank-128 bilinear fusion worth 0.5 M parameters? | 18 |
| **A6** | Does SupCon help, with the sampler controlled? | 12 |
| **A7** | Does any margin machinery help? | 24 |
| **A8** | Do Stages 2 and 3 add anything at all? | 24 |
| **A10** | Is capacity actually harmful? | 24 |
| **A11** | Is mixup the load-bearing regulariser? | 36 |

Each ablation carries a **pre-registered decision rule** — printed by `list` — because an
ablation without one is an invitation to read whichever number is convenient afterwards.

### A9 — what *are* the hard classes? (no training run)

Classes {41, 49, 51, 52, 70} were the bottom-5 at Stage-1 epoch 46 and still the bottom-5 at
Stage 3, invariant to 470 epochs, three loss regimes, two samplers and four difficulty-targeted
mechanisms. A9 asks whether that is **spectrally inseparable varieties** (a ceiling worth
publishing) or **segmentation failure** (a fixable bug that would explain the whole thing).

```bash
python -m spectralquadnet.experiments.cli analyse --run outputs/seednet_grouped_f0_s42
```

### Baselines and the leakage probe

```bash
python -m spectralquadnet.experiments.cli baseline --data spa40_90class_pfix
python -m spectralquadnet.experiments.cli leakage
```

The LDA-on-mean-spectra baseline costs seconds and is *the paper's most important baseline*: it
reaches 0.5916 under the leaky protocol, so ~59 points are available with no spatial information
at all. The leakage probe fits a 10-feature linear model on **residual brightness alone** and
reports how well it recovers the acquisition bundle — a model-free measurement of the nuisance.

### Assembling the report

```bash
python -m spectralquadnet.experiments.cli aggregate   # rebuild every table, no GPU
python -m spectralquadnet.experiments.cli report      # → outputs/experiments/REPORT.md
```

---

## Overriding anything

Standard Hydra. The composition groups are the ones in `configs/`:

```bash
python train.py data.split_fold=1 seed=1
python train.py model=quadnet_v4_audited pipeline=three_stage
python train.py data=spa40_90class_stratified          # the contrast arm
python train.py single.max_lr=1e-4 single.epochs=80
python train.py -m seed=0,1,2                          # Hydra multirun
python train.py --config-name=experiment/quadnet_audited   # the full control arm
```

Multi-GPU (DDP with `SyncBatchNorm`, so two GPUs compute the same function as one):

```bash
torchrun --standalone --nproc_per_node=2 train.py
```

### Experiment tracking

```bash
python train.py tracking.backend=wandb
python train.py tracking.backend=multi tracking.backends=[console,wandb]
```

W&B receives a **monotone global step** across every stage (CHANGES IC-1). In the audited run,
Stage 2 and Stage 3 restarted their step counters at 1 and W&B discarded every scalar they
logged — ~200 warnings, and seven panels that stop at Stage 1. Series now include
`progress/stage` and `progress/stage_epoch` so the flattened axis can be split back apart.

Logged throughout: `train/*`, `val/*`, `sched/*`, `loss/branch_*_{raw,weighted}`,
`grad_norm/{preclip,postclip,clipped}_*`, `grad_norm/clip_fraction`, `influence/branch_*`,
per-class tables, the confusion matrix and the final metrics with their intervals.

---

## What changed, and why

| | Change | Reason |
|---|---|---|
| IC-1 | Monotone cross-stage W&B step | Stage 2/3 telemetry did not exist |
| IC-2 | Log raw **and** weighted per-branch aux losses | The panel tracked the controller, not the branch |
| IC-3 | Default to `grouped` + `calib_frac=0.15` + `single_group_policy=error` | The largest correctness change in the audit |
| IC-4 | Band selection restricted to training rows, per fold | Label leakage independent of the split |
| IC-5 | `aux_gradnorm_alpha=0`, one aux head at a fixed 0.2 | Aux term was ≈7.8× the main loss; controller saturated at its clip bounds |
| IC-6 | `grad_clip` 1.0 → 5.0, + clip-fraction telemetry | The clip fired every step; the LR schedule was decorative |
| IC-7 | bf16 kept through the contrastive phases | Passing SupCon dropped the whole epoch to fp32 → 5–10× slower |
| IC-8 | `allow_tf32=False`, `num_workers=-1`, `compile=auto` | The one knob that changes numerics was on; the two that only change speed were off |
| IC-9 | `subcenter_K=1`, balance/per-class-margin/Ω off by default | Sub-centres were collinear at seeding (cos 0.987) |
| IC-10 | `SpectralSeedNet` (2.82 M) alongside `SpectralQuadNet` (5.19 M) | Branch D: 23.9% of params, 3.1% influence. Branch A: 60% of FLOPs, 5.6% influence |
| IC-11 | One stage replaces three | Stages 2+3: +0.005 macro-F1 for 65% of the wall clock |
| IC-12 | Ablation registry + protocol driver + aggregation | 21 levers documented, zero pulled |
| IC-13 | 107 → **180** acquisition bundles in the docs | 107 is not divisible by 90 |
| IC-14 | Dead paths removed or wired | `stride`, Stage-3 ProtoNCE, `sched/proto_weight`, `gain.npy` |

`SpectralQuadNet` and the three-stage curriculum are **kept, unmodified**. A3 and A8 are the
experiments that decide whether the removals above were right, and deleting the thing an
ablation exists to falsify would reproduce the exact defect this revision corrects.

---

## Reporting rules

These are enforced by the code, not by convention:

1. **Selection never happens on the reported split.** `calib` selects; `val ∪ test` is scored
   once. The run banner prints both.
2. **`val` and `test` are two halves of the same held-out bundle** and are therefore *not*
   independent of each other. They are scored together.
3. **Mean ± range over folds × seeds. Never a maximum.** A running maximum over ~944 correlated
   selection events was worth an estimated +0.042 macro-F1 in the audited run.
4. **Every reported number carries an interval.** Sampling noise on ~1,300 patches is ±0.020 at
   95%; the audited run's entire Stage-2 + Stage-3 gain was +0.005.
5. **A delta whose interval crosses zero has not been shown to do anything** — and is reported
   that way, in grey, on the forest plot.

Three constraints belong in the paper rather than a footnote:

- Training sees **one** acquisition bundle per class, so there is **zero within-class acquisition
  variance in training**. The model cannot learn acquisition invariance because it never observes
  two acquisitions of one class. This is a data-collection ceiling, not a method limitation.
- Two folds is the maximum. There is no third bundle.
- Band selection must be inside the fold, or declared as a fixed a-priori choice.

---

## Testing

```bash
pytest                       # fast tier — unit tests only (~10 s)
pytest --run-all             # + regression, slow and dataset-dependent tiers
pytest --run-slow tests/smoke/   # end-to-end runs on a synthetic dataset (~2 min)
```

The smoke tier builds a miniature dataset with the real one's load-bearing structure — **two
class-pure bundles per class** — and runs the actual `train.py` composition end to end, because
both of the audited run's most consequential defects were integration failures that no
component test could have caught.

```bash
ruff check . && black --check . && mypy       # lint / format / types
python scripts/check_config_roundtrip.py      # every config key has a home
```

### Known state

- `tests/regression/test_golden_forward_pass.py` has **two pre-existing failures** on the
  Stage-1 epoch loss and weight digests. They reproduce identically before and after this
  revision (loss `23.06525230407715` vs a golden `23.080477237701416`), so they are environment
  drift — a torch/BLAS version difference against the machine that captured the goldens — not a
  regression introduced here. Re-capture with `python scripts/capture_golden.py` on the target
  environment to clear them.

---

## Hardware

Trains on CPU, CUDA and Apple Metal. Device selection is automatic (`device=auto`).
`cfg.runtime` holds every throughput knob and none of them may change a reported number — the
two that would (`allow_tf32`, `channels_last`) are off by default, and `amp_dtype` is recorded in
the startup banner precisely because it *is* part of what a number means.

Expected: ≈45 min/run for the default single-stage configuration on an RTX 3060, against the
audited 19 h. That arithmetic is the point — it converts a project that could afford one run into
one that can afford the whole grid.

---

## Documentation

| | |
|---|---|
| [`CHANGES.md`](CHANGES.md) | The audit. The authoritative specification for this revision. |
| [`docs/01_ABSTRACT_AND_OVERVIEW.md`](docs/01_ABSTRACT_AND_OVERVIEW.md) | System overview |
| [`docs/02_DATASET_AND_PREPROCESSING.md`](docs/02_DATASET_AND_PREPROCESSING.md) | Dataset, segmentation, splits |
| [`docs/03_MODEL_ARCHITECTURE.md`](docs/03_MODEL_ARCHITECTURE.md) | Branch-by-branch architecture |
| [`docs/04_CURRICULUM_AND_LOSSES.md`](docs/04_CURRICULUM_AND_LOSSES.md) | Objectives and schedules |
| [`docs/05_EXPERIMENTS_AND_ABLATIONS.md`](docs/05_EXPERIMENTS_AND_ABLATIONS.md) | Diagnostics and logging |
| [`docs/06_EXECUTION_AND_HARDWARE.md`](docs/06_EXECUTION_AND_HARDWARE.md) | Runtime, DDP, profiling |
| [`docs/config_rename_table.md`](docs/config_rename_table.md) | Config field reference |
