# S22 · v5 rebaseline and fixed RGB fusion on complementary folds

2026-10-04 · **Prepared and frozen; six GPU fits unrun.** CPU engineering profile
completed on two training rows with finite gradients, 2,725,700 parameters and zero
held-out predictions. No learning curve, GPU timing or neural result is claimed.

[S20](../S20_rgb_pathway/README.md) found non-complementary historical folds;
[S21](../S21_complementary_rgb/README.md) repaired coverage and found useful RGB
fusion against linear HSI. Old v5 checkpoints were trained on different scans and
cannot serve as the neural control on S21 rows. This study closes that specific gap.

## Frozen contract

The executable [configuration](../../../../configs/research/s22_complementary_v5.json)
and [evidence copy](../../evidence/S22_complementary_v5/preregistration.json) have
separate SHA-256 files. Input hashes pin training source, YAML configuration,
historical k32 arrays and exact S21 row plan. The launcher validates every frozen
input. An opt-in `split_override` checks dataset identity, exhaustive/disjoint row
partitions, class support and group separation before fitted preprocessing/loaders.
Legacy default splitting remains unchanged. Seed means model initialization; the
same frozen partitions are used for all three seeds.

Six cells only: **folds 0/1 × seeds 0/1/2**, v5 R1 with k32 reflectance,
`snv_morph`, spatial strides `[2,2,2,1]`, CBAM minimum size 3, 200 maximum epochs,
patience 40 and the historical R1 optimization recipe. No architecture or band sweep.

**H40:** equal calibrated v5 + fixed S21 RGB improves three-seed/two-fold mean
macro-F1 by ≥.02, with paired class interval above zero and nonnegative cross-recall
delta. A transfer-improvement claim additionally requires a positive cross-recall
interval. Three HSI seeds share the deterministic RGB probe; this does not replicate
RGB pretraining or create new acquisitions. Freeze the final candidate before a new
crossed-session/lot test. Failure retains unimodal HSI and triggers a documented stop.

A pre-GPU amendment fixes rank-zero ownership of output preflight (preventing a
late rank from mistaking a newly created directory for an old run), and checks GPU
cell identity in postprocessing. The prior sealed draft and exact edits are preserved
under `evidence/S22_complementary_v5/pre_gpu_amendment_01`; no results or scientific
thresholds existed or changed. The original CPU profile predates this amendment.

## Run on the GPU host

Transfer the repository and **`dataset_u430k32/`** with byte-identical files. The
training phase needs no RGB images/checkpoint/features. Install the repository's
runtime and prep dependencies (hash checks import the RGB probe module). Use a
compatible CUDA/PyTorch environment and the existing two-T4 runtime. Source hashes
are portable; environment differences must be retained in the run artifacts. If a
compatibility edit becomes necessary, make a new explicit amendment before training.
Do not disable hash guards to run a different implementation.

From the repository root, execute these six jobs sequentially:

```sh
PYTHONPATH=src torchrun --standalone --nproc_per_node=2 scripts/run_complementary_v5.py train --fold 0 --seed 0
PYTHONPATH=src torchrun --standalone --nproc_per_node=2 scripts/run_complementary_v5.py train --fold 0 --seed 1
PYTHONPATH=src torchrun --standalone --nproc_per_node=2 scripts/run_complementary_v5.py train --fold 0 --seed 2
PYTHONPATH=src torchrun --standalone --nproc_per_node=2 scripts/run_complementary_v5.py train --fold 1 --seed 0
PYTHONPATH=src torchrun --standalone --nproc_per_node=2 scripts/run_complementary_v5.py train --fold 1 --seed 1
PYTHONPATH=src torchrun --standalone --nproc_per_node=2 scripts/run_complementary_v5.py train --fold 1 --seed 2
```

Outputs are `outputs/s22_complementary_v5/f{fold}_s{seed}/`. Preserve the complete
folders, including provenance, configuration, training logs, checkpoints, split
reports and `results/`. Repeat an interrupted identical cell with `--resume`; only
unfinished output with matching plan/fold/seed is accepted. Completed cells refuse
overwrite. The underlying training pipeline provides checkpoint resumption. No GPU
availability was present locally, so distributed GPU execution remains unverified.

Optional CPU smoke replay (not a model experiment; use a new output directory):

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python scripts/run_complementary_v5.py profile --output outputs/s22_profile_replay
```

## Analyze exactly once after all six cells complete

Copy all six complete output folders back to this repository. Keep the S21 saved
probability files in `outputs/s21_rgb_study/`; their hashes are pinned separately in
`analysis_inputs`, so the training host does not need them. Run:

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python scripts/analyze_complementary_v5.py
```

The analysis refuses missing/incomplete/wrong-plan/wrong-cell runs, changed inputs
or an existing analysis directory. It aligns calibration and held-out logits by exact
row/target identity, fits temperatures only on calibration, averages with S21 RGB at
fixed .5 weight, and reports all seeds/folds plus 2,000 paired variety-resample
intervals. Original saved v5 predictions define the baseline; float16 logit argmax
disagreements are reported. The result folder is `outputs/s22_fusion_analysis/`.

Next session: verify provenance and coverage first; archive compact outcomes and
figures in this study, resolve H40, update the registers/plan/progress, and decide
whether bounded RGB adaptation is justified. Do not merge S22 estimates with legacy
fold results or call this new-session replication. S17/S18 remain historical reserved
work and are not silently amended by S22.
