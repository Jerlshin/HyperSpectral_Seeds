# S11 · Executing the frozen diagnostics — instrumentation, the X2 switch, and a ready-to-run X1/X2/X4

| | |
|---|---|
| **Status** | **running** — part 1 (code, instrumentation, validation) complete; part 2 (the GPU arms X1, X2, X4) not yet run. No held-out row has been read by anything on this page |
| **Dates** | 2026-10-01 → |
| **Commits** | base `8050ba2` (S09). The S11 code, the S10 analysis it implements and this page are **uncommitted at the time of writing**; every S11 run records its own commit and dirty flag in `results/run.json → run.code`, which is the authoritative record. Commit and push before a Kaggle session clones `main` |
| **Data** | none for part 1 beyond the S08 sweep's checkpoints and *training* rows (`clean_fit_reproduction.py`) and synthetic cubes. Part 2: `dataset_u430k32` (refl-215 axis, uniform430 k = 32), grouped folds 0/1 and stratified |
| **Code** | `src/spectralquadnet/`: `engine/clean_fit.py`, `experiments/frozen.py`, `utils/provenance.py` (new); `engine/train_epoch.py`, `engine/stages/single_stage.py`, `engine/stages/final_eval.py`, `engine/evaluate.py`, `engine/diagnostics.py`, `engine/pipelines/{single,three_stage,context}.py`, `losses/auxiliary.py`, `optim/param_groups.py`, `models/spectral_seed_net.py`, `data/loaders.py`, `reporting/artifacts.py`, `config/schema.py`; `scripts/run_frozen.py` (new); configs `single/one_stage`, `model/seed_net`, `experiment/seednet_full256`, `evaluation/held_out_once` |
| **Raw outputs** | part 1: none kept (synthetic runs in temp dirs). Part 2: `outputs/experiments_u430k32/s11/{X1,X2,X4}/<variant>__f<fold>_s<seed>/` |
| **Evidence** | [`evidence/S11_frozen_arms_execution/`](../../evidence/S11_frozen_arms_execution/) |
| **Findings** | F57, F58 · **Decisions** D22; D18, D20, D21 annotated · **Hypotheses** none new; H12a–H16 reading map §6 |

## 1 · Question
Can the frozen S09/S10 diagnostics (X1 fit-first, X2 attribution, X4 fit ceiling) be run on code that measures what
they need — clean fit, the aux weight actually applied, per-module gradients, logits, provenance, each kernel scored
once — **without** changing the regime they are compared against? And what do they then show?

Part 1 (this page, now) answers the first half. Part 2 will answer the second with the frozen decision rules. Nothing
here re-opens D16 (protocol) or the frozen plans; it implements S10 §6 P0 and §9 steps 1–6.

## 2 · Why
S10 §9 makes the P0 instrumentation a gate before ≈ 8 GPU-pair-hours are spent: S09's fit hypothesis H13 and S10's
H16 are unmeasurable without a clean-fit probe; the applied aux weight was not the documented one (F54); per-module
gradient norms, logits and the code revision were not recorded; and X2 needs a pathway switch that did not exist
(S09 §10). D18 and D20 list the items; D19 forbids any architectural change before X1 reports.

## 3 · What changed — old → new

| S10 item | Old behaviour | New behaviour | Where |
|---|---|---|---|
| **P0.1** DDP de-dup | `DistributedSampler` padding survived the gather: every grouped S08 run wrote 4,312 / 4,310 predictions for 4,311 / 4,309 held-out kernels and counted the duplicate in `run.json` (F58) | every evaluation pass (per-epoch calib selection, clean fit, final scoring, logits) is de-duplicated into dataset order: each kernel exactly once on any world size | `data/loaders.py::{gathered_positions, dedup_index}`, `engine/evaluate.py`, `engine/stages/final_eval.py` |
| P0.1 provenance | `run.json` had no code revision | `run.json → run.code` = `{commit, dirty, source}`; `run.environment` (python, torch, CUDA, GPU, world size); `run.regime` — the training regime **as applied** (incl. the aux weight's first/last/mean, pathways, whether morphometrics reached the model); `run_name`, `output_dir` | `utils/provenance.py`, `engine/pipelines/context.py` |
| P0.1 logits | argmax predictions only | float16 `results/logits_<split>.npz` (logits, targets, rows) for the reported split **and calib**, ±TTA; listed in `run.json → logits` (`evaluation.save_logits`) | `final_eval.py`, `reporting/artifacts.py` |
| P0.1 pathway labels | SeedNet declared none → influence logged as `A`–`D` with `C = D = 0` | `pathway_labels() → ("SPATIAL", "SPECTRAL")` → `influence/branch_{spatial,spectral}` | `models/spectral_seed_net.py` |
| **P0.2** clean fit | no clean training accuracy (S09/S10 read proxies or measured offline) | fixed class-stratified 1,000-kernel training subset (private RNG, `single.clean_fit_seed`), eval mode, no augmentation, margin 0, plain CE; live **and** EMA; on every diagnostics stride, every new best and the final epoch (incl. an early stop) → `fit/clean_train_{acc,ce}_{live,ema}`, `output_dir/clean_fit.json` (subset rows, final, at-best, history), `stage1_meta.json → clean_fit`; survives resume | `engine/clean_fit.py`, `single_stage.py`, `pipelines/single.py` |
| **P0.3** telemetry | `sched/aux_weight` logged the configured 0.2; `train/acc` scored against one mixup label / margined logits; banner said "aux w=0.2 (fixed)" | `sched/aux_weight_applied` (logged by the loop that applies it) + `sched/aux_weight_configured`; `train/loss_main`, `train/acc_dominant` (label with λ ≥ 0.5), `train/acc_plain` (unpenalised logits, the model's detached `main_plain`); the banner states the applied schedule. `sched/aux_weight` is gone from the single stage | `engine/train_epoch.py`, `single_stage.py`, `losses/auxiliary.py` |
| **P0.4** aux schedule | hidden: always `_aux_loss_weight` = 0.65 → 0.25 | `single.aux_weight_schedule ∈ {legacy, fixed}`, **default `legacy`** (bit-identical); `fixed` = `model.aux_head_weight` | `losses/auxiliary.py::single_stage_aux_weight`, schema, `one_stage.yaml` |
| **P0.5** gradient groups | `grad_norm/*` by QuadNet prefixes (only `arcface_head` matched SeedNet); epoch means `inf`/`nan` after a GradScaler overflow | model-declared `grad_groups()` → `grad_norm/{gate,stem,tail,proj,spectral,fuse,embed,head,aux}`; every `grad_norm/*` mean over finite steps + `grad_norm/nonfinite_steps` | `engine/diagnostics.py`, `train_epoch.py`, `spectral_seed_net.py` |
| P0.5 clip partition | hard-coded `CLIP_GROUPS` (`fuse` and the aux head in `backbone` on SeedNet) | `clip_partition ∈ {legacy, model}`, **default `legacy`** (D22 deviation 1); `model` uses `clip_groups()` (`fuse` → `fusion`) | `optim/param_groups.py`, schema, `seednet_full256.yaml` |
| **P0.6** tests | — | `test_s11_instrumentation.py` (46), `test_structural_defects.py` (F48 `xfail(strict=True)` ×3; F50 ×5), `test_frozen_arms.py` (8), `tests/smoke/test_s11_arms.py` (5); `test_protocol_guard.py`, `test_config_wiring.py`, `test_branch_drop.py`, `test_spectral_seed_net.py` updated | `tests/` |
| **P0.7** docs | `docs/03` gave the tail as (B, 256, 4, 4) and a 32→16→8→4 geometry; `docs/04` described a fixed 0.2 aux weight, clip 1.0 and SupCon-disabled AMP; `docs/05` A–D influence labels and a stale results table | corrected throughout; k = 32 parameter table with the 327,680 dead parameters; README, config reference | `docs/03–05`, `README.md`, `docs/config_reference.md` |
| §9.4 X2 switch | — | `model.pathways ∈ {[spatial, spectral], [spectral], [spatial]}`: the disabled pathway's output is zero in train **and** eval, its forward is skipped, its parameters frozen (no optimiser state, no DDP unused-parameter error); `[spectral]` drops the aux term (its head reads the spatial output), `[spatial]` drops the morphometrics; initial weights of the live pathway unchanged | `spectral_seed_net.py` |
| §9.5 frozen arms | the frozen commands could not run as written (no launcher, no output dir) | `scripts/run_frozen.py` + `experiments/frozen.py`: the 23 cells built **from the hashed files** (refuses if a hash moved), `torchrun`, one directory per cell, `frozen_cell.json`; `--list`, `--check`, `--cfg-job`, `--dry-run` (D22 deviation 2) | `scripts/`, `experiments/` |

**Not changed, by decision.** The architecture (D19: X5 tail stride and X6 level block wait for X1), every training
default (mixup, margin, clip 5.0, aux `legacy`, partition `legacy`, LR, epochs), the protocol, the frozen files.

## 4 · Method (validation of part 1)
1. **Gate G-neutral** (`code/g_neutral.py`): the same miniature run (8 classes × 2 bundles × 12 kernels, 8 bands,
   16 × 16, CPU, single RNG stream, 3 epochs) through a `git worktree` of `8050ba2` and through the S11 tree with all
   new telemetry on (clean fit every epoch, per-module norms). Regimes: `shipped` (mixup → margin ramp, clip 5),
   `clip_binds` (clip 0.05, so the partition is exercised), `x1_like`. Compared: every step's loss, the epoch
   pairs, SHA-256 of all 222 checkpoint tensors (live + EMA), held-out predictions ±TTA. **Sensitivity controls**:
   the S11 tree with `clip_partition=model` (under `clip_binds`) and with `aux_weight_schedule=fixed` must be
   *detected* as different.
2. **Clean-fit reproduction** (`code/clean_fit_reproduction.py`): the probe, built exactly as the pipeline builds it,
   on the selected checkpoint of all 12 S08 runs, live and EMA, over (a) the 1,000-kernel subset and (b) the full
   training set for two runs; compared with S10's offline `ckpt_fit.csv`. Train rows only.
3. **DDP** (`code/ddp_dedup.py`): two gloo ranks, an odd-sized held-out split (91 kernels), through both trees; plus
   each X2 arm under two ranks.
4. **Reference** (`code/reference_dedup.py`): the S08 grouped runs' saved predictions re-scored with the duplicate
   removed.
5. **Composition** (`code/frozen_plan_snapshot.py`, `scripts/run_frozen.py --check --cfg-job`): every cell composed
   and checked against its frozen regime; X1 cells against the config their literal frozen commands compose.
6. **Test tiers, lint, types**: `pytest` fast and `--run-all` on both trees; `ruff`, `mypy --strict` on both trees.

## 5 · Results (part 1)

| Check | Result | Evidence |
|---|---|---|
| G-neutral, 3 regimes | **identical**: 30/30 step losses (max \|Δ\| = 0), 222/222 tensors, held-out predictions ±TTA; every new series present and `clean_fit.json` written | `g_neutral.json` |
| G-neutral sensitivity | `clip_partition=model` under a binding clip: step loss Δ up to 1.17, **213/222 tensors differ**; `aux_weight_schedule=fixed`: Δ up to 1.35, 210/222 — both detected | `g_neutral.json` |
| Clean fit vs S10 (full training set) | equal to `ckpt_fit.csv` to < 4 × 10⁻⁷ (acc) and 2 × 10⁻⁷ (CE) (e.g. grouped f0 s0 live: 0.872387 / CE 0.375196 vs 0.872387 / 0.375196) | `clean_fit_reproduction.csv` |
| Clean fit, 1,000-kernel subset vs full | max \|Δacc\| **0.017**, mean 0.008 over 24 run × weight pairs — sampling error at n = 1,000 (SE ≈ 0.01 at acc 0.9) | `clean_fit_reproduction.csv` |
| DDP, 91 held-out kernels on 2 ranks | pre-S11: **92** predictions, per-class support sums to 92; S11: **91**, logits rows 91 unique, calib 14 unique | `ddp_dedup.json` |
| X2 arms under 2-rank DDP | `spectral_only`, `spatial_only`, `no_morph` all train and score each kernel once | §4.3 run (not snapshotted) |
| S08 reference without the duplicate | per-run \|Δ macro-F1\| ≤ 1.7 × 10⁻⁴; grouped TTA mean 0.530068 (`run.json`) → **0.530033 = the frozen reference** | `reference_dedup.csv` |
| Frozen cells | 23/23 compose to their frozen regime; 9/9 X1 cells equal their literal frozen command's config; 23/23 pass `train.py --cfg job` | `frozen_plan.json` |
| Test tiers | `--run-all`: pre-S11 964 passed / 6 failed / 37 errors; **S11 1,034 passed / 6 failed / 37 errors + 3 strict xfails (F48)** — the failing sets are identical, all pre-existing (37 need the absent pinned ref, FW-26; 2 golden drift, FW-11; 2 a stale 256-band smoke fixture; 1 W&B-collision smoke; 1 an fp16 NaN expectation under torch 2.14). Fast tier 645 → 711 passed | `test_tiers.json` |
| Lint / types | ruff: 13 findings, all pre-existing (14 before; one in a touched file fixed); mypy --strict: the same 12 pre-existing errors before and after, none new | — |

## 6 · Findings
Recorded in [FINDINGS](../../FINDINGS.md).

| ID | Finding | Strength |
|---|---|---|
| F57 | The S11 instrumentation moves no training number (G-neutral, 3 regimes), the gate detects a real change, and the clean-fit probe reproduces S10's offline clean fit exactly on the full training set and within sampling error on its 1,000-kernel subset | E4 |
| F58 | Before S11 a DDP evaluation scored the padding kernel twice; every grouped S08 run counted one extra prediction (Δ ≤ 1.7 × 10⁻⁴). The frozen S09 reference (0.530) is the once-per-kernel value, so the de-duplicated S11 arms compare with it directly | E4 |

### 6.1 Where each frozen hypothesis will be read (recorded before any arm runs)
No threshold, arm or rule is changed; this maps each frozen quantity to the file that will hold it, so the reading is
fixed before the data exist.

| Hypothesis (frozen in) | Quantity | Read from |
|---|---|---|
| H12a (S09) | X1 stratified macro-F1, TTA, mean of seeds 0–2, vs 0.712 | `s11/X1/stratified__f0_s*/results/run.json → results.tta.macro_f1` |
| H12b (S09) | X1 grouped macro-F1, TTA, mean of 2 folds × 3 seeds, vs 0.530; and Δgrouped/Δstratified | `s11/X1/grouped__f*_s*/results/run.json → results.tta.macro_f1` |
| H13 (S09; D21 guard 1) | X1 clean training accuracy on the D18 subset. The frozen text names neither the epoch nor the protocol; its reference (F45, 0.90 / 0.91) was measured on the **selected checkpoint**, so it is read there, per protocol: `at_best_checkpoint[<best_source>].acc`, with the final-epoch live value reported beside | `s11/X1/*/clean_fit.json`, `run.json → checkpoint.best_source` |
| H14a (S09) | X2 `no_morph` cross-session recall, TTA, mean of 4 runs, ≤ 0.07 | `s11/X2/no_morph__*/results/run.json → results.tta.session.cross_session.macro_recall` |
| H14b (S09) | full − `spectral_only` grouped macro-F1 ≤ +0.02 | `s11/X2/spectral_only__*/…/macro_f1` against the S08 runs (see risk 6) |
| H16 (S10) | X4 grouped fold 0, clean training accuracy, live weights, final epoch, ≥ 0.98 | `s11/X4/grouped__f0_s0/clean_fit.json → final.live.acc` |
| D22 reversal | any X1/X4 epoch with `grad_norm/clip_fraction > 0.01` | `metrics.jsonl` |
| D21 guard 4 | X1 runs stopping before epoch 160 | `last_stage1.json → epoch`, `run.json → checkpoint.epoch` |

## 7 · Decisions
- **[D22](../../DECISIONS.md) · P0 as implemented, three deviations.** (1) `clip_partition=legacy` by default — the
  model-declared partition is opt-in because it is not neutral where the 5.0 clip binds (F57's sensitivity control),
  which would break G-neutral and X2's comparability; (2) the frozen commands run under `torchrun` with one directory
  per cell, the overrides read out of the hashed files; (3) X5/X6 not implemented (D19).
- D18 and D20 marked implemented; D21 guard 3 now enforced by the default and `test_protocol_guard.py`.

## 8 · Remaining risks and threats to validity
1. **The partition at clip 50.** X1/X4 assume clip 50 never binds (S10 B5). If `grad_norm/clip_fraction > 0`, the
   run is a `legacy`-partition result (D22's reversal trigger) — report it, do not re-run silently.
2. **Clean-fit cost.** The probe runs live + EMA over 1,000 kernels on every new best (frequent early), every 25th
   epoch and at the end: ≈ 17 s per 1,000-kernel pass (each of live and EMA) on 4 CPU threads; on T4 × 2 an estimated few seconds per measured epoch,
   ≈ 5–10 % of a run. A 200-epoch X1 cell is ≈ 25–30 min; X1 + X4 ≈ 5 h, X2 ≈ 4 h — inside two Kaggle sessions, and
   an interrupted cell resumes at the next epoch (the probe's history is in the training-state snapshot).
3. **Code must be on GitHub `main` before a Kaggle session.** The notebook clones `--depth 1`; uncommitted S11 code
   would silently run the old regime. `run.json → run.code.commit` must name the S11 commit and `dirty: false`.
4. **fp16 on T4.** Overflow steps are now counted (`grad_norm/nonfinite_steps`) rather than poisoning epoch means;
   `train/skipped_batches` > 0 still warrants `runtime.amp_dtype=bf16` per README.
5. **Analysis scripts must read the new keys.** `sched/aux_weight` no longer exists in single-stage runs; pre-S11
   runs logged SeedNet influence as `branch_a`/`branch_b` (= spatial/spectral).
6. **H14b's "full".** The frozen text compares X2's `spectral_only` (4 runs: folds 0,1 × seeds 0,1) with "full"
   without saying whether "full" is the matching four S08 cells or the six-run mean. Both will be reported; the
   choice must be recorded before the X2 numbers are read.
7. **Pinned-reference gates unavailable.** `capture_golden.py --verify` and `check_config_roundtrip.py` read commit
   `886560fe…`, which is not in this repository's history; both fail identically before and after S11 (FW-26). The
   committed-golden pytest gates ran instead.
8. **The miniature G-neutral run is not the production run.** It proves the code path is neutral (same operations,
   same RNG stream) on CPU; on CUDA, `cudnn.benchmark` and fused AdamW are non-deterministic run to run anyway, so
   bit-identity there is neither expected nor tested.

## 9 · What would change these conclusions
- A `clip_fraction > 0` in X1/X4 → the clip-partition choice is part of X1's regime (D22 reversal).
- `run.json → run.code.dirty: true` or a commit other than S11's in any cell → that cell is not an S11 result.
- Clean fit measured on the subset deviating from the full-set value by more than ≈ 0.03 in a new run → the subset is
  not representative there; report the full-set value from the saved checkpoint (the reproduction script does it).

## 10 · Run — part 2 (GPU), exact commands

**Pre-flight, locally (free):**
```bash
shasum -a 256 docs/research/evidence/S09_post_sweep_forensics/preregistration_next.json      # f896d0e5…
shasum -a 256 docs/research/evidence/S10_training_architecture_review/preregistration_s10.json # 1c8ae693…
python scripts/run_frozen.py --check        # 23/23 cells compose to their frozen regime
python scripts/run_frozen.py --cfg-job      # 23/23 compose through train.py itself
git add -A && git commit && git push origin main   # Kaggle clones main --depth 1
```

**Kaggle notebook** (GPU T4 × 2, Internet on, dataset `rice-hsi-u430k32` attached). Cell 1 as in README §10
(clone, `pip install -e .`, link `dataset_u430k32`, `--verify`). Then:

```bash
# Cell 2 — confirm the session runs the S11 code (prints the commit; must not be 8050ba2)
!cd /kaggle/working/HyperSpectral_Seeds && git log --oneline -1 && python scripts/run_frozen.py --check

# Cell 3 — X1 (9 runs) + X4 (2 runs), each on both GPUs; ≈ 5 h
!cd /kaggle/working/HyperSpectral_Seeds && python scripts/run_frozen.py --arms X1 X4 \
    --nproc-per-node 2 --output-root outputs/experiments_u430k32 --stream

# Cell 4 — X2 (12 runs, shipped regime); ≈ 4 h — may share the session or run in the next one
!cd /kaggle/working/HyperSpectral_Seeds && python scripts/run_frozen.py --arms X2 \
    --nproc-per-node 2 --output-root outputs/experiments_u430k32 --stream
```

A finished cell is skipped on rerun; an interrupted one resumes from `last_stage1.pth`. Across sessions, restore
`outputs/` from the previous version's output as README §10 describes. Any single cell can be re-run by hand with the
exact command `python scripts/run_frozen.py --nproc-per-node 2 --dry-run` prints, e.g.

```bash
torchrun --standalone --nproc_per_node=2 train.py --config-name=experiment/seednet_full256 \
  data=ablation/u430k32_grouped runtime=kaggle_t4x2 tracking=console_jsonl \
  single.mixup_epochs=30 single.arcface_m=0.0 single.margin_warmup_start=31 single.margin_warmup_end=31 \
  grad_clip=50.0 single.epochs=200 single.patience=40 \
  run_name=s11/X1/grouped__f0_s0 output_dir=outputs/experiments_u430k32/s11/X1/grouped__f0_s0 \
  data.split_fold=0 seed=0
```

(the script also writes `frozen_cell.json`; a hand-run cell does not).

**Telemetry to confirm on the first X1 cell** (S10 §9 step 7): `grad_norm/clip_fraction` ≈ 0;
`sched/aux_weight_applied` = 0.648 at epoch 1 → 0.25 from epoch 176; `train/skipped_batches` = 0; `fit/*` present;
`run.json → run.code.dirty` false. For X4: `sched/aux_weight_applied` = 0 throughout.

**Analysis** (part 2): the frozen decision rules (S09 `preregistration_next.json`, S10 `preregistration_s10.json`),
the D21 guards and §6.1 above. Held-out scored once per arm; nothing selected on it.

## 11 · Reproduce part 1
From the repository root (CPU; the second and fourth need `outputs/experiments_u430k32/` and `dataset_u430k32/`):
```bash
git worktree add /tmp/s09_tree 8050ba2
python docs/research/evidence/S11_frozen_arms_execution/code/g_neutral.py --old /tmp/s09_tree --new . \
    --out docs/research/evidence/S11_frozen_arms_execution/g_neutral.json                          # ≈ 1 min
python docs/research/evidence/S11_frozen_arms_execution/code/clean_fit_reproduction.py           # ≈ 15 min
python docs/research/evidence/S11_frozen_arms_execution/code/ddp_dedup.py --repo . --repo /tmp/s09_tree \
    --out docs/research/evidence/S11_frozen_arms_execution/ddp_dedup.json                          # ≈ 2 min
python docs/research/evidence/S11_frozen_arms_execution/code/reference_dedup.py                  # seconds
python docs/research/evidence/S11_frozen_arms_execution/code/frozen_plan_snapshot.py             # seconds
pytest --run-all                                                                                   # ≈ 25 min
```

## 12 · Provenance
Synthetic cubes are regenerated by the scripts (seeded); the pre-S11 tree is `git worktree add … 8050ba2`. The S08
checkpoints and predictions read by `clean_fit_reproduction.py` and `reference_dedup.py` are
`outputs/experiments_u430k32/protocol/*/{best_stage1.pth, results/}` (not snapshotted; 69 MB per run). Every table on
this page is a file in the evidence folder.
