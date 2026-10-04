# S15 · Replicating the lean network (Y3) and dissecting it (Y5) — implementation and the Kaggle run

| | |
|---|---|
| **Status** | **complete** — part 1: the runner, its tests and the validation (§4); part 2: the 10 GPU cells ran on Kaggle on 2026-10-03 (10/10 scored, 265 min) and were read in [S16](../S16_replication_reading/README.md) (F83–F89, D35). No held-out row was read on this page |
| **Dates** | 2026-10-03 → |
| **Commits** | design frozen in S14 (`fbb0927`); the S15 runner is the commit that lands this page. Every cell records its commit and dirty flag in `results/run.json → run.code` and the training-code digest in `frozen_cell.json → code_identity` |
| **Data** | part 2: `dataset_u430k32` (refl-215 axis, uniform430 k = 32), grouped folds 0, 1 and the stratified contrast; seeds per cell (Y3: 1, 2; Y5: 0) |
| **Code** | `src/spectralquadnet/experiments/s15.py` (new), `scripts/run_s15.py` (new), `experiments/s13.py` (two helpers shared: `base_expected_values`, `check_cell(…, expected=)`); `tests/unit/test_s15_plan.py` (new). **No model, training, data or config code** |
| **Raw outputs** | part 2: `outputs/experiments_u430k32/s15/{Y3,Y5}/<variant>__f<fold>_s<seed>/` |
| **Evidence** | [`evidence/S15_y3_replication/`](../../evidence/S15_y3_replication/) |
| **Findings** | F82 · **Decisions** D34 · **Hypotheses** H21a–H21e, H22a, H22b (frozen in S14, read in S16) |

## 1 · Question
Can the round S14 froze (`preregistration_s14.json`, `9e182670…`; S14 §9, D33) be run on Kaggle **exactly as frozen
and nothing else** — 10 cells, each at its frozen seed, on the code the S13 cells ran — with every guard checkable in
Kaggle's shallow clone?

The design, its hypotheses and its reasons are S14's and are not restated here: [S14 §9](../S14_screen_reading/README.md#9--next-steps).

## 2 · The cells

| arm | cells (frozen order) | seed | role | hypotheses (read in S16) |
|---|---|---|---|---|
| **Y3** replication | `lean_grouped` f0, f1 × s1, s2; `lean_stratified` f0 × s1, s2 | 1, 2 | confirmatory | H21a/H21b on seeds 0–2 (S13's seed-0 cells + these); H21c–H21e on these fresh seeds only |
| **Y5** dissection | `desc_only` f0, f1; `spatial_repair` f0, f1 | 0 | screen (D28) | H22a, H22b |

Replication cells run first: a session cut short leaves the confirmatory cells done. Nothing S11 or S13 already scored
is in the plan (checked at load time); Y3's seed-0 cells and the X1 references are read from `outputs/…/s13` and
`outputs/…/s11` in S16.

## 3 · What changed — old → new

| Item | Old | New | Where |
|---|---|---|---|
| Runner | `run_s13.py` reads one seed for every cell (`deviation.seed`) and writes under `s13/` | `run_s15.py` reads each cell's own seed from S14's `cells.gpu[*].seed`, writes under `s15/`, verifies **three** hashes (S12 parent → R1, S13 amendment, S14 design) | `experiments/s15.py`, `scripts/run_s15.py` |
| Composition gate | S13's arms only | every S15 cell = R1 + the shipped keys + its arm's intent, restated independently of the override strings (`ARM_INTENT`); Y5's two intents must partition Y3's (tested) | `experiments/s15.py`, `experiments/s13.py` (shared helpers) |
| Code-identity guard | — | the frozen guard ("`git diff aed5257 … -- models engine data config` empty, or G-neutral") as a **content digest**: SHA-256 over the path and content of every training-relevant file — `src/spectralquadnet/**/*.{py,yaml}` except `experiments/`, `configs/**/*.yaml`, `train.py` (129 files) — pinned at `aed5257` (`fade41e5…`) and recomputed before any cell runs. Kaggle clones `--depth 1`, so `aed5257` is not in its history; the digest needs none. Its scope is wider than the frozen guard's (stricter) | `experiments/s15.py::check_code_identity` |
| Provenance | — | `frozen_cell.json`: study, role (replication / dissection), seed, sources, the three hashes, the code digest and whether it matches | `experiments/s15.py::provenance` |
| README | S13 Kaggle cells | S15 Kaggle cells (check, run, archive the outputs for download) | `README.md` §10 |

## 4 · Validation (part 1)

| Check | Result | Evidence |
|---|---|---|
| Hashes | S12 `88b377c5…`, S13 `ef598213…`, S14 `9e182670…` verify; a tampered S14 seed is refused | `test_s15_plan.py` |
| Plan | 10 cells, layout and seeds as frozen; replication first; **no cell repeats an S13 cell** | `frozen_plan.json`, `test_s15_plan.py` |
| Composition | 10/10 compose to R1 + their arm (`--check`); 10/10 compose through `train.py --cfg job`; a dissection arm given the other half's key is caught | `scripts/run_s15.py --check / --cfg-job` |
| Code identity | digest at `aed5257` (from `git archive`) = pinned = working tree, 129 files; `git diff --stat aed5257` on the scope: empty. The guard flags an edit to a model file and ignores a runner edit | `code_identity.json`, `test_s15_plan.py` |
| The arms' models | each builds from its composed config and trains one step (finite gradients) at the real shape (32 bands, 64 × 64); parameters: Y3 2,725,700 (= S13's `run.json`), `desc_only` 2,808,230 (shipped − 41,248), `spatial_repair` 2,766,948 (shipped − 82,530) — **the two halves remove exactly what Y3 removes** | `test_s15_plan.py` |
| 2-rank `torchrun` (gloo, CPU, S13's harness) | R1, Y3 (seed 1), `desc_only`, `spatial_repair`: all finish on both ranks; every held-out kernel scored once (91/91); κ on every training row once (77/77); the regime block names the arm; partition holds on the miniature network too | `ddp_arms.json` |
| Test tier (`pytest`, macOS CPU, torch 2.13) | **787 passed, 403 skipped, 2 xfailed, 0 failed**; S13 + S15 plan tests 28/28; ruff clean on every touched file; mypy clean on `experiments/s13.py`, `s15.py` | — |

## 5 · Findings
Recorded in [FINDINGS](../../FINDINGS.md).

| ID | Finding | Strength |
|---|---|---|
| F82 | The S15 runner composes exactly the 10 frozen cells (none repeats S13), on training code byte-identical to `aed5257` (129 files, digest `fade41e5…`); the two dissection arms partition Y3's parameter reduction exactly (41,248 + 82,530); every arm finishes a 2-rank job with each held-out kernel scored once | E4 |

## 6 · Decisions
- **[D34](../../DECISIONS.md) · S15 as implemented — no deviation from the frozen file.** The code-identity guard is a
  content digest (wider scope, works without git history); replication cells first; P0.3 (the old X2 re-score) is not
  part of the session (it decides nothing and needs the S11 output attached); a fourth Kaggle cell archives the outputs.

## 7 · Run — part 2 (GPU), exact commands

**Pre-flight, locally (free):** `python scripts/run_s15.py --check` (code identity OK, 10/10 cells OK), then commit and
push — Kaggle clones `main --depth 1`.

**Kaggle notebook** — GPU T4 × 2, Internet on, dataset `rice-hsi-u430k32` attached. Cell 1 as README §10 (clone,
`pip install -e .`, link `dataset_u430k32`, `--verify`). Then:

```bash
# Cell 2 — must print "code identity: OK" and "check: 10/10 cells OK"
!cd /kaggle/working/HyperSpectral_Seeds && git log --oneline -1 && python scripts/run_s15.py --check
# Cell 3 — THE production command: the 10 S15 cells, each on both GPUs, then a summary (≈ 4.4 h, ≤ 4.8 h)
!cd /kaggle/working/HyperSpectral_Seeds && python scripts/run_s15.py --nproc-per-node 2 --stream
# Cell 4 — one archive to download for the S16 reading
!cd /kaggle/working/HyperSpectral_Seeds && tar czf /kaggle/working/s15_outputs.tar.gz outputs/experiments_u430k32/s15
```

Unpack at the repository root (`tar xzf s15_outputs.tar.gz`). A finished cell is skipped on rerun; an interrupted one
resumes from `last_stage1.pth`. `--cells <name>` runs one cell; `--dry-run` prints every command;
`--override runtime.compile=off` is the only kind of override allowed (runtime knobs; recorded in `frozen_cell.json`).

**Runtime** (S13 cells' own wall clock on the same runtime; `runtime_estimate.json`): Y3 grouped 25.3 min × 4, Y3
stratified 31.7 × 2, `desc_only` ≈ 25.7 × 2 (the shipped tail, as X1), `spatial_repair` 25.3 × 2 → **≈ 266 min ≈ 4.4 h**
(≤ 4.8 h if every cell runs 200 epochs).

**Telemetry to confirm on the first cell:** `run.json → run.code.commit` = the S15 commit and `dirty: false`;
`frozen_cell.json → code_identity.identical: true`; `run.json → run.seed` = 1 for Y3, 0 for Y5; `run.parameters` 2,725,700
(Y3), 2,808,230 (`desc_only`), 2,766,948 (`spatial_repair`).

## 8 · Reading map (recorded before any cell runs; read in S16)
No threshold, reference or rule is changed — all are in `preregistration_s14.json`. Every quantity is TTA, from
`outputs/experiments_u430k32/{s13,s15}/…/results/run.json`; the template is S14's `hypotheses.py`.

| Hypothesis | Quantity | Cells |
|---|---|---|
| H21a | mean grouped macro-F1 ≥ 0.5108 | `s13/Y3/lean_grouped__f{0,1}_s0` + `s15/Y3/lean_grouped__f{0,1}_s{1,2}` (6) |
| H21b | mean stratified macro-F1 ≥ 0.7090 | `s13/Y3/lean_stratified__f0_s0` + `s15/Y3/lean_stratified__f0_s{1,2}` (3) |
| H21c | mean − X1 mean ≥ +0.020 and hierarchical-bootstrap CI excludes 0 | `s15/Y3/lean_grouped__f{0,1}_s{1,2}` vs `s11/X1/grouped__f{0,1}_s{1,2}` (0.528496) |
| H21d | cross-session recall ≥ 0.1666 and attraction ≤ 0.4240 | `s15/Y3/lean_grouped__f{0,1}_s{1,2}` |
| H21e | stratified mean − X1 mean ≥ +0.018 | `s15/Y3/lean_stratified__f0_s{1,2}` vs `s11/X1/stratified__f0_s{1,2}` (0.728593) |
| H22a / H22b | mean grouped macro-F1 ≥ 0.5508 | `s15/Y5/desc_only__f{0,1}_s0` / `s15/Y5/spatial_repair__f{0,1}_s0` |
| guards | dirty / commit / code identity; clip fraction > 0.01 outside epoch 1 and overflow epochs; stop epoch < 160 | `run.json`, `frozen_cell.json`, `metrics.jsonl` |

## 9 · Risks
1. **A different Kaggle image.** S11/S13 ran torch 2.10.0+cu128; a newer image changes numerics slightly. The
   frozen guard asks for the environment to be reported; S16 reads it from `run.json → run.environment`.
2. **The code digest is strict.** Any edit to a model, engine, data, config or YAML file makes `run_s15.py` refuse —
   by design. A runner-only fix (under `experiments/` or `scripts/`) does not.
3. **H21a/H21b include the seed-0 cells from another session.** Same code (digest) and the same runtime are required;
   a different environment is reported (preregistration guard 3).

## 10 · Reproduce part 1
```bash
python scripts/run_s15.py --list && python scripts/run_s15.py --check && python scripts/run_s15.py --cfg-job
pytest tests/unit/test_s15_plan.py tests/unit/test_s13_plan.py                        # ≈ 30 s
python docs/research/evidence/S15_y3_replication/code/plan_snapshot.py                 # seconds; needs git history
python docs/research/evidence/S15_y3_replication/code/ddp_arms.py \
    --out docs/research/evidence/S15_y3_replication/ddp_arms.json                      # ≈ 20 s
```
