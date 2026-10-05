# Research progress and exact resume state

Updated 2026-10-05. **S20/S21 complete; S22 amended two-cell screen running.**
The authorized RGB phase is implemented and executed locally. This file is the current
handoff; the earlier running log is preserved as
[evidence/S20_rgb_pathway/progress_history.md](evidence/S20_rgb_pathway/progress_history.md).
Read the [master plan](MASTER_RESEARCH_PLAN.md) and
[S22 runbook](studies/S22_complementary_v5/README.md) before starting another run.

## Completed and learned

- **Data:** verified original 17.3-GB archive, checksum and 584 member CRCs. All 180 RGB
  scans yield 48 objects; all 8,624 retained HSI identities are uniquely paired by grid,
  with 16 historical exclusions explained. Exact equality of every float16 k32 value,
  masks and morphology. Native RGB masks/crops, descriptors, full 215-band compact spectra
  and 4×4 regions are complete. Strict214 excludes a pooled-white wavelength.
- **CPU:** four frozen DINOv2-S/14 views extracted; S20 runs 30 arms × two legacy folds,
  including fusion with three saved v5 seeds. S21 runs 24 arms × two corrected folds.
  Predictions, per-class metrics, intervals, numeric probe exports, hashes and figures
  are saved. No GPU training/fine-tuning has run.
- **Information:** S21 RGB F1 **.450520**, HSI32 **.534850**, equal fusion **.591820**;
  fusion cross recall **.127858** vs HSI32 **.077941**. Appearance contributes beyond
  silhouette. Historical v5 fusion gains **.040896 F1**, but cross delta **.009490**
  has CI[−.039544,.058211]. Useful complementarity is measured; reliable v5 transfer
  improvement is unproven.
- **Rejected/deferred:** full bands and feature concatenation gain aggregate F1 while
  losing transfer. Matching kernels does not beat the within-scan shuffle. Keep k32
  and simple late fusion/unimodal controls; defer attention, gating and broad tuning.
  Preprocessing v1/v2 were rejected before predictor scoring; v3 passed reviewed QC.
- **Protocol correction:** legacy grouped folds repeat 1,772 rows and omit 1,776, while
  remaining group-disjoint within each fold. S21 fixes keyed RNG and holds all 8,624
  rows out once. Old plans/studies remain intact. Never reuse old v5 networks under
  new partitions. Both studies reuse historical acquisitions, not independent data.

[Complete S20 report](studies/S20_rgb_pathway/README.md) ·
[S21 report](studies/S21_complementary_rgb/README.md) ·
[Findings F97–F103](FINDINGS.md) · [Decisions D41–D44](DECISIONS.md).

## Local assets and immutable outputs

| Path from repository root | State / purpose |
|---|---|
| `dataset/rice_hsi.zip` | Raw 17.3-GB archive, read-only; SHA256 `92af3258ba72950e301ee7f4030f59cc88922e21f34d00a06d68debb6b046c14` |
| `dataset_u430k32/` | Historical reference, unchanged; required on S22 GPU host |
| `dataset_rgb_hsi_v3/` | Final complete paired assets, native mask shards/QC, summary/region arrays and manifest |
| `dataset_rgb_hsi_v1/`, `dataset_rgb_hsi_v2/` | Rejected label-free preparation, never scored; not current inputs |
| `outputs/s20_rgb_features_v3/` | Four384-D frozen CPU feature caches and checkpoint/source provenance |
| `outputs/s20_rgb_study/` | Complete30-arm legacy-fold screen;40 fitted probe exports plus fusion/logit arms |
| `outputs/s21_rgb_study/` | Complete24-arm complementary screen;40 fitted probe exports; S22 analysis uses its two probability NPZs |
| `outputs/s22_complementary_v5/cpu_profile/`, `outputs/s22_profile_final/` | CPU engineering profiles only; no held-out predictions |
| `docs/research/evidence/S20_rgb_pathway/`, `S21_complementary_rgb/` | Tracked compact predictions, metrics, audits, plans, code, figures and validation receipts |
| `configs/research/s22_complementary_v5.json` | Final executable GPU plan; evidence copy and pre-GPU amendment preserved |

Large arrays/checkpoints/raw data remain local and ignored. To move machines, transfer
these artifacts separately and verify hashes; a source checkout alone is insufficient.
The cached DINO checkout/checkpoint live in the user's local torch hub cache; exact
hashes are in extraction provenance. S22 training does not require RGB assets.
Builders/runners refuse completed output overwrite. Engineering replays use new
output directories and are not counted as new scientific evidence.

## Frozen plans

| Study | SHA256 | Status |
|---|---|---|
| S20 | `c2d4f18394bd031530ba438b4ef2cad04a5b56a97f2eee2cad6a96b67951607a` | executed |
| S21 | `e4d1e4a3d469f98c2c4c7e69b023a9835d56a12cda4159c108d4ea2504bf63ca` | executed |
| S22 | `f44c0d2e22f2849816cf2a8f2cdad8ae90084c695b06e4c181e7fc9b000f6f73` | prepared/unrun, pre-GPU amendment01 |

S22's amendment fixes distributed output ownership and exact-cell analysis guards;
scientific design and thresholds are unchanged, and the parent plan/source are saved.
CPU smoke profiles precede final sealing and name their draft/parent hash honestly.

## Current amendment02 execution

The user authorized efficient architecture development. The executable child is
`configs/research/s22_screening_amendment02.json`, SHA256
`207db0236ecb95e387f451e6ade0bb45779f981a6354b53f543d150b91e05bb0`.
The original S22 file and evidence remain unchanged. Both folds at seed0 only;
MPS/fp32 is the declared local runtime. Training-only hardware probe passed, and
five split/resume tests passed before launch. Fold0 is in progress; no neural
held-out result exists yet. [Full amendment](studies/S22_complementary_v5/amendment02.md).

The prior six-cell handoff below is retained as historical context and superseded
by amendment02. New default launcher/analyzer use the child. Original drivers are
archived in `evidence/S22_complementary_v5/amendment_02_parent/code/`.

## Original next steps (superseded by amendment02)

1. Transfer this repository and unchanged `dataset_u430k32/` to the two-T4 GPU host.
   Run the **six explicit commands** in [S22](studies/S22_complementary_v5/README.md):
   folds 0/1 × seeds 0/1/2, exact v5 R1 with S21's frozen partitions. `--resume` is only
   for matching unfinished cells. Do not broaden bands or architectures.
2. Return all six complete `outputs/s22_complementary_v5/f*_s*/` folders. Validate
   provenance/coverage; run `PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python
   scripts/analyze_complementary_v5.py` once, with unchanged local S21 probabilities.
3. Archive S22 compact results/figures; resolve H40; update registers and this handoff.
   Positive cross CI is required to claim a transfer gain. A higher aggregate F1 alone
   cannot justify that claim. Failure retains unimodal HSI.
4. Decide a bounded adaptation/geometry study only from that evidence. For broad
   robustness claims, obtain crossed independent sessions/lots with common class
   support and a locked test; all 17 current bridges touch session8.

S17/S18 remain reserved historical work. Their frozen designs and S00–S19 study
texts have not been rewritten to accommodate the new evidence. A 40-file initial
historical hash snapshot is verified by the current validators.

## Validation and remaining limitations

Final new pathway/split/resume tests: **18 passed**. Broader affected checks earlier:
**55 passed**. Strict mypy:12 production files pass, with amended S22 drivers checked
again; Ruff passes. Full unit run: **779 passed, 333 skipped, 2 xfailed, 8 failed**.
Five failures reproduce at the starting HEAD (Torch autocast, sandbox DDP sockets,
ANSI styling); three are historical S15 source guards correctly rejecting changed
source. No historical guard was weakened. [Receipts](evidence/S20_rgb_pathway/validation_summary.json).

All108 arm/fold prediction metrics, immutable result copies and frozen inputs can be
validated without retraining; both fold-coverage audits are reproducible. Scientific
figures were visually checked. CPU profile: two real training rows, finite gradients,
2,725,700 parameters, zero held-out scoring. GPU execution is still unverified.
NumPy matrix-product warnings remain documented; all coefficients/probabilities are
finite/normalized and train/calib explicit sums agree, without refitting outcomes.
Masks lack pixel annotation, pretraining overlap is unauditable, and class-bootstrap
intervals do not estimate new-session uncertainty.

Safe local checks (from repository root):

```sh
PYTHONPATH=src python docs/research/evidence/S20_rgb_pathway/code/validate_study.py --assets
PYTHONPATH=src python docs/research/evidence/S21_complementary_rgb/code/validate_study.py --assets
PYTHONPATH=src python docs/research/evidence/S20_rgb_pathway/code/audit_fold_coverage.py
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 pytest -q -o addopts='' tests/unit/test_rgb_pathway.py tests/unit/test_complementary_split.py tests/unit/test_frozen_rows.py tests/unit/test_s22_preflight.py
```

## Runtime compatibility follow-up03 (2026-10-05)

Attempt02 ended before epoch1 with a terminated DataLoader worker, no checkpoint
and no held-out predictions. Its logs/provenance are preserved. Amendment03 freezes
zero train/eval workers; both seeds/folds, MPS/fp32 and scientific recipe are unchanged.
Current hash: `1808076bcabd9189c55f78abc621e7fdd27b938834bea9f4c1f47263b82b8e76`.
Use `--plan configs/research/s22_screening_amendment03.json` explicitly for both
training cells and analysis; add `--resume` only for an unfinished matching cell.
Fold0 retry has written its first checkpoint. No neural result is quoted until both
fits and final analysis complete. [Compatibility note](studies/S22_complementary_v5/amendment03.md).

S23's23,514-parameter additive frozen-feature residual head is implemented and
unit-tested, pending the S22 development gate. `scripts/run_frozen_multimodal.py
freeze` refuses until complete verified S22 evidence passes; source/assets/checkpoints
are pinned before scoring that candidate. [Prospective specification](studies/S23_frozen_multimodal/README.md).

## Final CUDA execution amendment05 (2026-10-05)

The current scientific execution contract is **amendment05**, SHA256
`1dd50dae26f356ac963b4713ae5820a5476ede2129f8c554178a09abc7ea7f95`:
seed0 on both corrected folds, original two-T4 DDP/fp16 runtime. The partial MPS
fit stopped after six complete epochs and no held-out evaluation; its full folder
and compact receipts are preserved. The local coordinator was cancelled before
launching another cell. Current data are already attached to the prior private
Kaggle HSI notebook and configured account has30 GPU hours available.

New private CUDA dispatch: `jgfreak/s22-corrected-fold-single-seed-development-screen`.
The bundle metadata/source hashes are in
`evidence/S22_complementary_v5/kaggle_dispatch_manifest.json`; no credentials were
uploaded. Wait for the job, retrieve its `s22_cuda_outputs.tar.gz`, verify both
complete cells and input identity, then run `scripts/analyze_s22_cuda_screening.py`.
`run_s22_development.py` can analyze/archive/draw and conditionally freeze/run S23
once both CUDA folders exist locally. It uses05 and refuses existing state/output.
S23 remains conditional and unrun; no corrected-fold neural score is available yet.
[Final CUDA amendment](studies/S22_complementary_v5/amendment05.md).
