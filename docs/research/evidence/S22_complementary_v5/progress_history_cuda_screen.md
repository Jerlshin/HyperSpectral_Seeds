# Research progress and exact resume state

Updated 2026-10-05. **S20/S21 complete; amended S22 CUDA screen running; S23 implemented, conditional.**
Read the [master plan](MASTER_RESEARCH_PLAN.md), [S22 compute amendment](studies/S22_complementary_v5/amendment02.md),
and [current CUDA contract](studies/S22_complementary_v5/amendment05.md).

## Current execution and allocation

S22 now means exactly **seed0 on both corrected acquisition folds**, two complete
v5 fits rather than six. R1, k32, 2,725,700-parameter v5, batch 128, maximum 200 epochs,
patience 40, calib-only live/EMA selection and historical TTA are retained. Both folds
hold all 8,624 rows out exactly once and both directions of all 17 bridges.

The original S22 six-cell JSON/evidence remain immutable. Amendment02 records the
user-authorized screening policy;03 records a local compatibility attempt;05 restores
the originally planned two-T4 DDP/fp16 runtime for both complete cells. Final training
plan: `configs/research/s22_screening_amendment05.json`, SHA256
`1dd50dae26f356ac963b4713ae5820a5476ede2129f8c554178a09abc7ea7f95`.
A separate `s22_analysis_code05.json` seals the scorer without changing its rules.

The configured research account runs a **new private** Kaggle kernel:
[CUDA screen](https://www.kaggle.com/code/jgfreak/s22-corrected-fold-single-seed-development-screen), version1.
The prior HSI notebook already attaches `jerlshinjg/dataset-u430k32`; all ten files
must pass the frozen hashes before training. Credentials were used only locally,
never included in the uploaded source bundle. Dispatch metadata and per-file source
hashes are tracked in `evidence/S22_complementary_v5/kaggle_dispatch_manifest.json`.
`outputs/s22_cloud_execution.json` records job state. No neural held-out score is
quoted until both complete cells and guarded analysis have returned.

A collector process waits for the cloud job, retrieves `s22_cuda_outputs.tar.gz`,
refuses overwrite/unsafe extraction, validates the CUDA completion plan/cell list,
and invokes `scripts/run_s22_development.py`. That coordinator skips matching complete
cells, analyzes/archives/draws S22, then freezes/runs S23 only if H40-screen passes.
State/logs live in `outputs/s22_development_execution.json` and
`outputs/s22_development_coordinator.log` once collection reaches that phase.
If a process is absent or fails, inspect these files before replay; do not overwrite
a completed output or launch a duplicate GPU cell.

## Exactly how to resume after cloud completion

1. Retrieve the private job's saved output archive to a new local download directory.
   Preserve `cuda_launch_receipt.json`, both entire `f*_s0` folders, logs/checkpoints,
   and `CUDA_COMPLETE.json`. Check the marker against the active05 hash and cells.
2. `PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python scripts/run_s22_development.py` performs
   the remaining guarded sequence when both complete folders are already local and
   no coordinator state/output exists. Otherwise inspect current state first.
3. Standalone analysis is `python scripts/analyze_s22_cuda_screening.py`; archive via
   `docs/research/evidence/S22_complementary_v5/code/read_screen.py`. Both refuse
   existing output. Scientific figures follow saved compact results.
4. Read S22 fold results, acquisition directions, class intervals, error rescues and
   H40-screen. Then integrate S23 outcome (if it ran), registers, study pages and this
   handoff. Class intervals do not estimate seed or independent-session uncertainty.

## Evidence already established

- All180 scans and8,624 retained kernels are uniquely paired by grid; exact historical
  float16 k32, masks and morphology parity. Native RGB crops/masks, descriptors,
  full215 compact spectra/regions exist; strict214 excludes a pooled-white wavelength.
- S21 corrected folds: RGB F1 .450520, HSI32 linear .534850, equal fusion .591820.
  Per-fold fusion gains are +.062663/+.051278; cross gains +.056451/+.043382.
- S20 historical v5 fusion gains averaged over both legacy folds are +.046404,
  +.039623,+.036660 across seeds0/1/2; descriptive paired-delta SD .004995. This
  supports efficient screening, not a corrected-fold seed-variance estimate.
- Appearance matters beyond silhouette. Full spectra/concatenation improve aggregate
  F1 but lose cross recall in these probes. Correct pairing does not beat the scan
  shuffle; do not claim individual-kernel interactions or justify attention from it.
- Legacy folds repeat 1,772 rows and omit 1,776; S21 repairs exhaustive coverage. Old
  matched comparisons remain on their actual rows; old networks cannot be reused on
  corrected training scans. All17 bridges still touch session8; new crossed sessions
  and lots are required for broader transfer claims.

## Small learned successor and confirmation policy

[S23](studies/S23_frozen_multimodal/README.md) has 23,514 trainable parameters:
independent256-D HSI/384-D RGB →32-D additive branches, shared residual classifier,
and an equal calibrated probability anchor. Encoders and RGB probe stay frozen;
zero-initialized readout equals the anchor. Train-only feature scales/gradients,
calib-only selection, epoch0 eligible, one head seed across both folds. No sweep.
The gate also compares against S22's stronger TTA reference, including cross recall.
The runner refuses freeze/execute before complete verified S22 passes.

Confirm the frozen **final learned system and matched HSI/fixed-fusion controls**
at seeds0/1/2 on both corrected folds. Sample HSI and head initialization, not only
heads on one encoder. RGB's fixed deterministic probe does not need three identical
refits; a trainable RGB control does. Confirm mechanism ablations when required for
a paper claim or a meaningful borderline decision. Rejected intermediate candidates
receive no automatic seed expansion. Seed confirmation cannot create new sessions.
The [confirmation queue](studies/S22_complementary_v5/confirmation_queue.md) gives the
conditional fit counts and the first useful learned-head branch ablations.

## Preserved attempts and assets

| Repository-relative path | State |
|---|---|
| `dataset_u430k32/`, `dataset_rgb_hsi_v3/` | unchanged reference and validated paired assets |
| `outputs/s20_rgb_features_v3/` | frozen DINO views and source/checkpoint provenance |
| `outputs/s20_rgb_study/`, `outputs/s21_rgb_study/` | immutable prior results/probe exports |
| `outputs/s22_aborted_amendment02/f0_s0/` | terminated loader at epoch0; no checkpoint/held-out score |
| `outputs/s22_partial_amendment03/f0_s0/` | interrupted local MPS attempt; six complete epochs, no held-out score |
| `outputs/s22_kaggle_push/` | private CUDA bootstrap and metadata, no credentials |
| `outputs/s22_complementary_v5/f*_s0/` | reserved for complete matching CUDA cells after return |
| `outputs/s22_fusion_analysis/`, `outputs/s23_frozen_multimodal/` | pending guarded outcomes |

The interruption request observed epoch5; asynchronous GPU work completed epoch6
before termination. The actual marker and correction are recorded under
`amendment03_partial/interruption_receipt.json`. Original sealed motivation text is
not silently edited. Full files are ignored local assets; compact receipts are tracked.

## Verification and historical continuity

Targeted RGB/split/row/resume/scorer/head tests: **22 passed**. Strict mypy: current
CUDA scorer, launcher/coordinator/candidate and head pass; Ruff passes. Training-only
head interface probe: finite 256-D features/gradients, zero encoder gradients, initial
anchor error 2.24e-8, zero held-out scoring. S21 replay validates all 48 arm/fold
prediction arithmetic, all 8,624 identities/assets, immutable plan/results and 40
unchanged historical files. Receipts are under S22/S23 evidence.

S17/S18 remain reserved; no historical source guard was weakened. Earlier full-suite
results and handoffs remain in
[evidence/S22 progress history](evidence/S22_complementary_v5/progress_history_amendments_20261005.md)
and [S20 history](evidence/S20_rgb_pathway/progress_history.md). The old six-cell
instructions are historical and superseded by explicit amendments, never a new
confirmation claim.
