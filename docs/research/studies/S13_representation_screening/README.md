# S13 · Route-A representation arms Y1–Y4 as a single-seed screen — P0, implementation and the Kaggle run

| | |
|---|---|
| **Status** | **complete.** Part 1 (P0, Y2/Y3 code, the one-seed amendment, validation — §5) 2026-10-02; **part 2 (10 GPU cells + 2 fused cells) ran on Kaggle 2026-10-03 at `aed5257`, all scored, and is read in [S14](../S14_screen_reading/README.md)** (Y3 passes; Y1, Y2 rejected; H15 supported). The P0.3 re-score was skipped (S11 output not attached) |
| **Dates** | 2026-10-02 → |
| **Commits** | base `413a11e` (S11 code; the S12 analysis is uncommitted). The S13 code, the amendment and this page are **uncommitted at the time of writing**; the **S13 commit** is the one that lands them, and every cell records its commit and dirty flag in `results/run.json → run.code`. Commit and push before the Kaggle session — it clones `main --depth 1` |
| **Data** | part 1: synthetic cubes; the S11 cells' `metrics.jsonl` for timing only. Part 2: `dataset_u430k32` (refl-215 axis, uniform430 k = 32), grouped folds 0, 1 and the stratified contrast, **seed 0** |
| **Code** | `engine/checkpoint.py`, `engine/stages/single_stage.py`, `engine/pipelines/{single,three_stage,context}.py`, `engine/stages/final_eval.py`, `reporting/session_probe.py` (new), `experiments/fusion.py` (new), `experiments/s13.py` (new), `models/branches/spatial_cnn.py`, `models/spectral_seed_net.py`, `models/registry.py`, `config/schema.py`, `utils/provenance.py`; `scripts/run_s13.py` (new); `configs/model/seed_net.yaml`, `configs/evaluation/held_out_once.yaml`; `.gitignore` |
| **Raw outputs** | part 2: `outputs/experiments_u430k32/s13/<arm>/<variant>__f<fold>_s0/` |
| **Evidence** | [`evidence/S13_representation_screening/`](../../evidence/S13_representation_screening/) |
| **Findings** | F72 · **Decisions** D28, D29; D24, D26, D27 annotated · **Hypotheses** H19a–H21b and H15, as frozen, read as screening verdicts (D28) |

## 1 · Question
Can the route-A arms frozen in S12 (Y1 decoupled pathways, Y2 masked MixStyle, Y3 lean architecture, Y4 the 80/20 tier)
be run as a **fast single-seed screen** — every arm, control, protocol contrast and grouped fold kept, seed replication
removed — on code that fixes S12's infrastructure defects (F71) **without** changing what the arms are compared against?

Part 1 (this page, now) answers that. Part 2 runs the 10 cells and is read in the next study. Nothing here re-opens
D16, D23 or D24, or changes a frozen threshold.

## 2 · Why
- **The design.** `preregistration_s12.json` (`88b377c5…`) froze 30 runs ≈ 12.5 GPU-pair-hours (two Kaggle sessions).
  Before any of it ran, the PI asked for a screen: *"a fast screening study using one seed only … successful arms can be
  replicated across multiple seeds later"*. The parent file stays frozen; the change is a hashed amendment (D28, §3).
- **P0 (FW-34, D27).** One S11 cell was lost to a final-epoch checkpoint race (F71a); every Kaggle run recorded
  `dirty: true` because of the dataset symlink (F71b); Y1 needs a late-fusion scorer; D26 asks for a training-rows
  session κ in every report.
- **The arms' code.** Y2 and Y3 need model changes; S12 asked for them behind default-off keys, gated by G-neutral.

## 3 · The single-seed screen (D28)

**The amendment.** [`preregistration_s13.json`](../../evidence/S13_representation_screening/preregistration_s13.json),
SHA-256 `ef5982131df48c2ba8105c751fcf21f527a01ea2c0addf4ad59380f1a0989460` (also in the `.sha256` beside it), names the
parent's hash, the one approved change, why, and the exact cells. The runner refuses to start if either file's hash has
moved.

| arm | change (parent, unchanged) | parent runs | **S13 runs (seed 0)** | hypothesis (unchanged) |
|---|---|---:|---:|---|
| **Y1** decoupled pathways | `model.pathways=[spectral]` and `[spatial]`, log-prob fusion, w on calib | 12 | **4** + 2 fused (CPU) | H19a, H19b |
| **Y2** masked MixStyle | `model.spatial_mixstyle=true` | 6 | **2** | H20 |
| **Y3** lean architecture | `snv_morph` + tail `[2,2,2,1]` + `cbam_min_hw=3`; grouped + stratified | 9 | **3** | H21a, H21b |
| **Y4** 80/20 tier | `data=ablation/u430k32_stratified data.split_eval_frac=0.2` | 3 | **1** | H15 |
| | | **30** | **10** | |

**Unchanged:** R1 (read from the parent's `regime.overrides`), every arm's change, both grouped folds and the stratified
contrast, the reference (6-run X1 mean) and every margin and threshold, the decision-rule logic, the guards, the
held-out rule, P0, and what is not in this round.

**Why seed 0.** It is the first seed of every frozen design (X1, X2, X4, Y1–Y4), fixed before any run and chosen on
nothing. Every S13 cell therefore has an already-scored, fold- and seed-matched R1 cell (X1 seed 0), and Y2 and Y1's
spatial network start from exactly X1's initial weights at that seed — the matched delta removes initialisation
variance. These matched deltas are *reported beside* and decide nothing.

**What one seed costs.** The margins are 2·max(run-level sd, floor) and do not move. A 2-run grouped mean has
SE ≈ 0.0099/√2 ≈ 0.007 (≈ 0.004 at 6 runs), so the 0.020 margins sit ≈ 2.9 SE from an arm's estimate instead of ≈ 5;
a single stratified run has SE ≈ 0.006 against the 0.018 margin. WORKFLOW gate G3 (an improvement needs Δ > 2σ over
folds × seeds) cannot be met by a screen. Hence:

> **Every S13 outcome is a screening verdict.** Where the frozen decision rule says *adopt*, S13 reads *passes the
> screen*: the arm is re-run at seeds 1 and 2 under a new frozen file (FW-35) and may serve as the provisional base of
> later screening arms; it does not become the reference form, SeedNet v5, or a paper claim until replicated. A
> screening rejection is not replicated unless a later frozen file gives a reason.

## 4 · What changed — old → new

| Item | Old behaviour | New behaviour | Where |
|---|---|---|---|
| **P0.1** checkpoint race (F71a) | `save_ckpt` wrote `best_stage1.pth` and its sidecar with a plain `torch.save`/`open` on rank 0; every rank reloaded it as soon as the stage returned | both files written to a temporary name and `os.replace`d (`.pth` before `.json`); the single stage ends with a **barrier**, and the three-stage pipeline has one after each stage — no rank reads a checkpoint before rank 0 has finished writing it | `engine/checkpoint.py`, `engine/stages/single_stage.py`, `engine/pipelines/three_stage.py` |
| **P0.2** provenance (F71b) | `.gitignore` had `dataset_*/` (directories only), so the Kaggle dataset **symlink** was untracked → `dirty: true` on every run | `dataset_*` (no slash): a scratch repository with the symlink now reports an empty `git status --porcelain` | `.gitignore` |
| **P0.3** unscored X2 cell | — | re-scored by re-running the S11 cell through `scripts/run_frozen.py` (finished stage detected → final evaluation only); in the Kaggle session (§8) | — |
| **P0.4** late-fusion scorer | — | reads two cells' float16 logits, refuses unless both scored the same kernels in the same order, picks w ∈ {0, 0.05, …, 1} by **calib** TTA macro-F1 (ties → closest to 0.5), scores held-out once; writes a model-shaped `run.json` (`results.{tta,no_tta}` with CI and session breakdown, `fusion.{weight, calib table, equal_weight}`) | `experiments/fusion.py` |
| **P0.5** session κ in the report (D26) | — | after the held-out scoring, the selected weights' embedding and each live pathway's output on **every training row** (eval mode, no augmentation, DDP-gathered and de-duplicated) → class-disjoint session κ (S12's probe, verbatim) → `results/session_probe.json`, `run.json → session_probe`, `session_probe/kappa_*`. `evaluation.session_probe` (default on); reads no held-out row | `reporting/session_probe.py`, `engine/stages/final_eval.py`, `engine/pipelines/single.py` |
| **Y3** descriptor | `[index bank 64 ‖ continuum 16 ‖ SNV ‖ D₁ ‖ D₂ ‖ morph]` (184 at k = 32) | `model.spectral_descriptor ∈ {full, snv_morph}`; `snv_morph` = `[SNV ‖ morph]` (40); the inert blocks are **not built** | `models/spectral_seed_net.py` |
| **Y3** tail | four stride-2 ResBlocks: 16 → 8 → 4 → 2 → **1**; 327,680 untrainable parameters | `model.spatial_tail_strides` (default `[2,2,2,2]`); `[2,2,2,1]` ends at **2 × 2** — no structurally dead tap (tested) | `models/branches/spatial_cnn.py` |
| **Y3** CBAM | a CBAM after blocks 1–3, including on the 2 × 2 map (40 of 49 spatial-gate taps on padding) | `model.cbam_min_hw` (default 0 = all kept): a gate is kept only after a block whose output side ≥ the value, else `nn.Identity` (no parameters, same module indices). The placement is computed for the cube's real patch side (`build_model(…, input_side=…)`, passed by the pipeline); a mismatched input is refused | `spatial_cnn.py`, `models/registry.py`, `engine/pipelines/context.py` |
| **Y2** masked MixStyle | — | `model.spatial_mixstyle` (default off: no module, no random draw). After 3-D stem blocks 1 and 2, training only, p 0.5 per call: each kernel's per-(channel, spectral-slice) mean/sd over its **foreground** (weights = the area-pooled mask) is replaced by λ·own + (1−λ)·partner's, λ ~ Beta(0.1, 0.1), random pairing; statistics detached (Zhou et al. 2021); background re-zeroed exactly; a kernel without foreground is passed through; excluded from `torch.compile` | `spatial_cnn.py::MaskedMixStyle` |
| provenance | `run.json → run.regime` had no architecture keys | + `spectral_descriptor`, `spatial_tail_strides`, `cbam_min_hw`, `spatial_mixstyle`, `split_eval_frac` — an S13 arm is identifiable from `run.json` alone | `utils/provenance.py` |
| runner | — | `scripts/run_s13.py` + `experiments/s13.py`: cells built **from the two hashed files** (R1 from the parent, cells and seed from the amendment), each checked against an independent statement of its arm's intent; `torchrun`, one directory per cell, `frozen_cell.json`; then Y1's fusion and a summary table. `--list`, `--check`, `--cfg-job`, `--dry-run`, `--fuse-only`, `--summary` | `scripts/`, `experiments/` |

**Not changed, by decision.** Every training default (the shipped regime stays the default — D29 deviation 1), the
frozen files, the S11 runner and its cells, the architecture at its defaults.

## 5 · Validation (part 1)

| Check | Result | Evidence |
|---|---|---|
| **G-neutral** vs `413a11e`, 3 regimes (shipped, clip binds, R1-like), new tree with all telemetry and the κ probe on | **identical**: 30/30 step losses (max \|Δ\| = 0), 222/222 checkpoint tensors, held-out predictions ±TTA | `g_neutral.json` |
| G-neutral sensitivity | S11's two controls still detected (213 / 210 tensors differ); **Y2 on: 212 tensors, Δloss up to 2.21; Y3 on: 214 tensors, Δloss 2.24** — the gate sees both arms | `g_neutral.json` |
| Race regression (2 gloo ranks, best epoch = last, rank 0's write slowed 1.5 s) | **S13 tree:** both ranks reload epoch 3, no temporary file left. **`413a11e`:** rank 1 fails with `EOFError: Ran out of input` — the Kaggle failure reproduced. Sensitivity control (barrier removed, atomic write kept): rank 1 reloads the *previous* epoch's checkpoint — the barrier is necessary, atomicity alone is not | `tests/unit/test_checkpoint_race.py` |
| Atomic write | a write that dies half-way leaves the previous `.pth` and sidecar intact (fails on `413a11e`) | same file |
| Dataset symlink | `?? dataset_u430k32` before; empty `git status --porcelain` after | §4 P0.2 |
| Every arm on a real 2-rank `torchrun` job | R1, Y1 ×2, Y2, Y3, Y4 all finish; every held-out kernel scored once (91/91; Y4 37/37); κ on every training row once; regime block names the arm; Y1 fuses | `ddp_arms.json` |
| Composition | 10/10 cells compose to R1 + their arm's intent; R1 read from the parent equals X1's overrides key for key | `frozen_plan.json`, `scripts/run_s13.py --check`, `--cfg-job` |
| Y2 unit behaviour | no initial weight changes; eval logits bit-identical to off; foreground moments become exactly λ·own + (1−λ)·partner's; background exactly 0 and never read | `tests/unit/test_s13_model.py` |
| Y2 under `torch.compile` (inductor, CPU) | trains and stays finite over coin flips. A first version failed to compile: the excluded MixStyle forward creates a graph break, after which the stem's sizes are symbolic, and the mask pooling done *outside* it could not be lowered. All MixStyle work (pooling included) now runs inside the excluded forward; with Y2 off the stem's traced graph is unchanged | `test_s13_model.py::test_mixstyle_trains_under_torch_compile` (slow) |
| Y3 unit behaviour | default network = S10's 2,849,478 parameters, tail (8, 4, 2, 1); lean: descriptor 40, tail (8, 4, 2, 2), gate on the 2 × 2 map only removed, **no dead tap**; stem/embedding unchanged | same |
| Fusion and κ | weight never moves with held-out logits; ties → 0.5; misaligned cells refused; a shared session direction gives κ > 0.9, class identity alone \|κ\| < 0.2 | `test_late_fusion.py`, `test_session_probe.py` |
| End to end (`train.py`, synthetic cube) | Y1 pair → fused cell with session breakdown; Y2 and Y3 trained and recorded | `tests/smoke/test_s13_arms.py` |
| Test tiers (`pytest --run-all`, macOS CPU) | **S13: 1,100 passed, 4 failed, 37 errors, 3 xfailed; `413a11e`: 1,050 / 4 / 37 / 3 — failing sets identical, no regression**; the 50 new tests pass. The failures are the pre-existing set (37 need the absent pinned ref `886560fe`, FW-26; 2 Stage-1 golden drift, FW-11; 2 stale 256-band smoke fixtures); the 3 xfails are F48 on the *default* tail, which S13 leaves unchanged. ruff: no finding in a touched file; mypy (project config): the same 5 pre-existing errors in both trees. (A first run concurrent with an edit of the amendment was discarded and repeated.) | `test_tiers.json` |

## 6 · Findings
Recorded in [FINDINGS](../../FINDINGS.md).

| ID | Finding | Strength |
|---|---|---|
| F72 | The S13 code at its defaults moves no training number (G-neutral vs `413a11e`, 3 regimes) and the gate detects both new arms; the F71a race reproduces as `EOFError` on `413a11e` and is gone with atomic writes + an end-of-stage barrier — the barrier is necessary (without it rank 1 silently reloads the previous epoch) | E4 |

## 7 · Decisions
- **[D28](../../DECISIONS.md) · S13 runs as a single-seed screen.** Seeds 3 → 1 (seed 0), everything else as frozen;
  recorded as a hashed amendment; outcomes are screening verdicts; passing arms are replicated (FW-35).
- **[D29](../../DECISIONS.md) · P0 and the S13 arms as implemented — two deviations.** (1) D24's switch of the config
  default to R1 is **deferred**: S13 cells carry R1 explicitly from the frozen file, so no S13 number depends on the
  default, while switching it now would re-compose S11's X2 cells (which inherit the shipped regime) — including the
  P0 re-score — and void G-neutral's shipped regime. (2) The barrier is also added after each stage of the three-stage
  pipeline (same defect, A8's path). Choices the frozen text leaves open are recorded there (MixStyle weighting and
  compile exclusion, CBAM placement from the real patch side, κ default-on).
- D27 implemented (P0.3 executes in the Kaggle session); D26 implemented; D24 annotated (deviation 1).

## 8 · Run — part 2 (GPU), exact commands

**Pre-flight, locally (free):**
```bash
shasum -a 256 docs/research/evidence/S12_frozen_arms_reading/preregistration_s12.json        # 88b377c5…
shasum -a 256 docs/research/evidence/S13_representation_screening/preregistration_s13.json   # ef598213…
python scripts/run_s13.py --check        # 10/10 cells compose to R1 + their arm
python scripts/run_s13.py --cfg-job      # 10/10 compose through train.py itself
git add -A && git commit && git push origin main   # Kaggle clones main --depth 1
```

**Kaggle notebook** — GPU T4 × 2, Internet on, dataset `rice-hsi-u430k32` attached. Cell 1 as README §10 (clone,
`pip install -e .`, link `dataset_u430k32`, `--verify`). Then:

```bash
# Cell 2 — confirm the session runs the S13 code and the frozen plan
!cd /kaggle/working/HyperSpectral_Seeds && git log --oneline -1 && python scripts/run_s13.py --check

# Cell 3 — THE production command: 10 GPU cells (each on both GPUs), then Y1's 2 fused cells, then a summary.
#          ≈ 4.4 h (≤ 4.8 h if every cell runs all 200 epochs) — one session.
!cd /kaggle/working/HyperSpectral_Seeds && python scripts/run_s13.py --nproc-per-node 2 --stream
```

**P0.3 — re-score `X2/spatial_only__f1_s0`** (no training; ≈ 3 min). It needs the S11 cell's directory, i.e. the
S11 notebook's output added as an input and restored as README §10 describes; then

```bash
%%bash
cd /kaggle/working/HyperSpectral_Seeds
cp -rn /kaggle/input/<s11-notebook-output>/HyperSpectral_Seeds/outputs . 2>/dev/null || true
if [ -f outputs/experiments_u430k32/s11/X2/spatial_only__f1_s0/best_stage1.pth ]; then
  python scripts/run_frozen.py --cells X2/spatial_only__f1_s0 --nproc-per-node 2 --stream
else
  echo "S11 outputs not attached — skip; the re-score decides nothing (S12 §5.1)"
fi
```

Any single cell can be re-run by hand with the command `python scripts/run_s13.py --nproc-per-node 2 --dry-run`
prints (also in `frozen_plan.json`), e.g.

```bash
torchrun --standalone --nproc_per_node=2 train.py --config-name=experiment/seednet_full256 \
  data=ablation/u430k32_grouped runtime=kaggle_t4x2 tracking=console_jsonl \
  single.mixup_epochs=30 single.arcface_m=0.0 single.margin_warmup_start=31 single.margin_warmup_end=31 \
  grad_clip=50.0 single.epochs=200 single.patience=40 model.spatial_mixstyle=true \
  run_name=s13/Y2/mixstyle__f0_s0 output_dir=outputs/experiments_u430k32/s13/Y2/mixstyle__f0_s0 \
  data.split_fold=0 seed=0
```
(a hand-run cell gets no `frozen_cell.json`). A finished cell is skipped on rerun; an interrupted one resumes from
`last_stage1.pth`; `python scripts/run_s13.py --fuse-only` re-fuses Y1 from finished cells on CPU.

**Runtime** (from the S11 cells' own wall clock on the same runtime; `runtime_estimate.py`):

| cells | runs | analogue (S11, T4 × 2) | min / run (typical · cap) | total |
|---|---:|---|---:|---:|
| Y1 spectral-only | 2 | X2 spectral-only, 6.3 s/epoch | 20.4 · 22.0 | 41 |
| Y1 spatial-only | 2 | X2 spatial-only, 8.0 s/epoch | 25.9 · 28.0 | 52 |
| Y2 mixstyle | 2 | X1 grouped ×1.03, 8.2 s/epoch | 26.5 · 28.6 | 53 |
| Y3 lean grouped | 2 | X1 grouped, 7.9 s/epoch | 25.7 · 27.8 | 51 |
| Y3 lean stratified | 1 | X1 stratified, 10.1 s/epoch | 32.1 · 34.7 | 32 |
| Y4 80/20 | 1 | X1 stratified ×1.14 (more training kernels) | 36.5 · 39.4 | 37 |
| **10 GPU runs** | | typical = X1's mean 184.5 epochs; cap = 200 | | **≈ 266 min ≈ 4.4 h (≤ 4.8 h)** |

Plus ≈ 30 s for the fusion and ≈ 3 min for the P0.3 re-score. The parent design costs ≈ 13.3 h by the same method.

**Telemetry to confirm on the first cell:** `run.json → run.code.commit` = the S13 commit and `dirty: false` (the
symlink no longer counts); `run.json → session_probe.representations` holds `embedding` and the live pathway(s);
`grad_norm/clip_fraction` ≈ 0 outside epoch 1; `train/skipped_batches` = 0; for Y2 `run.regime.spatial_mixstyle` true;
for Y3 `run.parameters` < 2,849,478.

## 9 · Reading map (recorded before any cell runs)
No threshold, reference or rule is changed. Every quantity is TTA, from `outputs/experiments_u430k32/s13/…/results/run.json`.

| Hypothesis | Quantity | Read from |
|---|---|---|
| H19a | mean of the 2 fused cells' macro-F1 ≥ 0.5108 | `Y1/fused__f{0,1}_s0 → results.tta.macro_f1` |
| H19b | mean cross-session recall ≥ 0.1666 **and** mean cross-session attraction ≤ 0.4240 | `… → results.tta.session.cross_session.macro_recall`, `… → results.tta.session.attraction.cross` |
| H20 | Y2: mean cross ≥ 0.1666 **and** mean same-session recall ≥ 0.6285 | `Y2/mixstyle__f{0,1}_s0 → … .cross_session.macro_recall`, `… .same_session.macro_recall` |
| H21a | Y3 grouped: mean macro-F1 ≥ 0.5108 | `Y3/lean_grouped__f{0,1}_s0 → results.tta.macro_f1` |
| H21b | Y3 stratified macro-F1 ≥ 0.7090 (one run) | `Y3/lean_stratified__f0_s0 → results.tta.macro_f1` |
| H15 | Y4 macro-F1 − 0.726984 ≤ +0.03 (one run) | `Y4/within_8020__f0_s0 → results.tta.macro_f1` |
| guards | dirty / commit; `grad_norm/clip_fraction > 0.01` outside epoch 1 and overflow epochs; stop epoch < 160 | `run.json → run.code`, `metrics.jsonl`, `run.json → checkpoint.epoch` |
| beside (decide nothing) | per-fold values; deltas vs X1 seed 0 (grouped f0 0.5353 / f1 0.5354; stratified 0.7238); equal-weight fusion; no-TTA; session κ | `fusion.equal_weight`, `results.no_tta`, `session_probe` |

`python scripts/run_s13.py --summary` prints these columns per cell; the reading — hypotheses, guards, matched deltas,
verdicts — belongs to the analysis study, under D28's screening semantics.

## 10 · Risks and threats to validity
1. **One seed.** A screen can pass an arm that seed variance flattered or fail one it hid (§3: margins ≈ 2.9 SE). This is
   the accepted cost; FW-35 is the remedy for passes.
2. **H21b and H15 rest on one run each**, where the parent design had three.
3. **Matched deltas are only partly paired.** Y2 and Y1-spatial share X1's initial weights at seed 0, but the training
   trajectories diverge with the first random draw; Y3's initial weights differ (different modules). Report, don't decide.
4. **MixStyle placement detail.** The frozen text says "after the GN→GELU→×α"; the module runs between GELU and the mask
   multiplication with the mask as pixel weights, which is algebraically the same affine map applied after ×α with the
   background held at zero (§4). On a 2-pixel-wide edge the weights are fractional (stem block 2) — the foreground
   moments are area-weighted there.
5. **Y2 on CUDA is compiled for the first time on Kaggle.** Validated under inductor on CPU only; if the first Y2 cell
   fails in compilation, re-run just Y2 with `python scripts/run_s13.py --arms Y2 --nproc-per-node 2 --stream
   --override runtime.compile=off` (a runtime knob, recorded in `frozen_cell.json`; report it).
6. **CBAM placement depends on the patch side.** Computed from the cube (64 → gates kept after the 8 × 8 and 4 × 4 maps);
   a model rebuilt offline must be given the same side (default 64).
7. **The κ probe runs on every run by default.** It costs < 1 min and reads only training rows, but it is a new block in
   every `run.json`; analysis scripts written for S11 ignore it.
8. **Code must be on GitHub `main` before the session**, or the notebook trains the S11 code (commit check in Cell 2).

## 11 · What would change these conclusions
- G-neutral failing on a later change → that change is not neutral; S13 cells are then comparable to X1 only if they
  ran before it.
- A Kaggle cell with `dirty: true` → something other than the dataset link is untracked in the checkout; that cell is
  not an S13 result (guard).
- Any rank failing to reload after a final-epoch best → the barrier is not where the reload happens on that path.

## 12 · Reproduce part 1
From the repository root (CPU):
```bash
git worktree add /tmp/s11_tree 413a11e
python docs/research/evidence/S13_representation_screening/code/g_neutral_s13.py --old /tmp/s11_tree --new . \
    --out docs/research/evidence/S13_representation_screening/g_neutral.json             # ≈ 1 min
python docs/research/evidence/S13_representation_screening/code/ddp_arms.py \
    --out docs/research/evidence/S13_representation_screening/ddp_arms.json              # ≈ 30 s
python docs/research/evidence/S13_representation_screening/code/plan_snapshot.py         # seconds
python docs/research/evidence/S13_representation_screening/code/runtime_estimate.py      # seconds; needs outputs/…/s11
pytest tests/unit/test_checkpoint_race.py --run-slow                                     # ≈ 20 s
pytest --run-all                                                                         # ≈ 30 min
```

## 13 · Provenance
Synthetic cubes are regenerated by the scripts (seeded); the pre-S13 tree is `git worktree add … 413a11e`. The timing
source for the runtime estimate is `outputs/experiments_u430k32/s11/*/metrics.jsonl` (not snapshotted). Every table on
this page is a file in the evidence folder or a named test.
