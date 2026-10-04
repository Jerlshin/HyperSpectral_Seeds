# Next-generation research progress and resume state

Updated: **2026-10-04**. Master plan: [MASTER_RESEARCH_PLAN.md](MASTER_RESEARCH_PLAN.md).
**P0–P6 complete: strategy delivered.** Proposed predictive experiments remain unrun.
Study: [S19_next_generation_strategy](studies/S19_next_generation_strategy/README.md).
S17 execution and S18 reading remain reserved by S16; do not reuse those identifiers.
Literature search/verification cutoff is2026-10-03; final integration concluded2026-10-04.

## What was completed

- P0: created the master plan first and reported it; inventoried the working tree.
- P1: reviewed the S00–S16 scientific chain, registers, source/configuration, data
  processing, splits, models and saved results. Persisted the evidence/history review.
- P2: descriptive CPU audit produced acquisition support, split/rank, band geometry,
  calibration-source, raw archive/RGB pairing, array, run and frozen-plan inventories.
  Two figures regenerate from the saved evidence through the shared asset builder.
- P3:39 primary-source records, bibliographic IDs/URLs/dates, access/protocol limits,
  applicability judgments, search log and working BibTeX. Contemporary coverage
  includes2025–September2026; no assertion of exhaustive systematic-review coverage.
- P4: wrote the proposed wavelength-aware HSI + pretrained RGB system, mathematical
  support/limits, controls, conditional mechanisms and rejected/deferred routes.
- P5: wrote staged experiments, compute policy, crossed acquisition design, statistics,
  prior-art replication requirements, claim ledger and conditional journal route.
- P6: integrated F90–F96, D38–D40, draft H26–H31, FW-39–43, study index and timeline.
  Validated local links/JSON/Python,39 bibliography entries, band/calibration/support
  invariants, all eight frozen-plan hashes and17 unchanged historical study documents.
  Visually inspected both figures and rebuilt S19 through `build_assets.py`.

## Research conclusion to carry forward

v5 is the replicated reference (k32 refl; grouped TTA F1 .570816; same/cross recall
.682841/.199401). More training/capacity did not solve generalisation. Prioritize
complementary information, not a long incremental SeedNet sweep. All180 raw RGB scan
partners are present; kernel pairing is not validated. Full215 extracted cube is
absent.32/64 overlap at only eight bands;215 adds20 below430nm; three retained white
reference values are session-filled in64/215 but none in32. All17 cross-session
bridges touch session8, and each grouped training class has one session. These are
specific controls/limitations, not a numeric Bayes ceiling.

The proposal is untrained. Its novelty/performance must be earned by controlled
experiments and independent acquisition evidence. Generic RGB–HSI/gated fusion is
already prior art. Published high scores use different or unverified protocols;
do not claim either superiority or proven leakage without reconstruction.

## Exact next work for an execution session

1. Read S19 README, architecture and experiment plan, then WORKFLOW and the active
   frozen S16 parent. Preserve all historical study evidence and existing changes.
2. Complete FW-39: raw checksum/payload verification, reproducible RGB/HSI physical
   kernel IDs, missing-component review and full-band rebuild with k32 parity.
   Persist overlays, exclusions, hashes and calibration fit/apply rules.
3. Pin v5 explicitly or execute D36's neutral default migration. The bounded S17
   round remains unchanged: Z1 three80/20 runs; Z3 two end-map runs; Z2 two k64 runs.
   Read it in reserved S18. Do not silently modify the parent JSON.
4. Implement train/calib-only E0/E1 feature probes and simple fusion controls. Check
   checkpoint licensing, input wavelengths,90-class support and overlap before
   choosing a pretrained option. Profile compute before fixing a GPU-hour budget.
5. Freeze a separate next-study manifest before any new confirmation: exact paired
   data/splits, band indices, arms, seeds, metric, thresholds, selection and stopping
   rules. H26–H31 are **draft** and must not be reported as preregistered.
6. Pursue crossed acquisitions in parallel with model development: at least two
   training sessions plus development and locked test sessions with common class
   support; record/cross biological lots and keep physical kernel identities disjoint.
   Existing benchmark data has informed many rounds; new seeds are not a fresh test.

## Reproduction and validation

```sh
python docs/research/evidence/S19_next_generation_strategy/code/audit_repository.py
python docs/research/evidence/S19_next_generation_strategy/code/build_literature.py
python docs/research/evidence/S19_next_generation_strategy/code/draw_figures.py
python docs/research/evidence/S19_next_generation_strategy/code/validate_study.py
```

Audit uses local metadata and archive headers, not model predictions; ~3s after
font cache on the current machine. Drawing needs only saved evidence. Literature
script materializes manually verified entries offline; rerunning does not reverify
the web. `validation.json` records checks; it is not model-performance validation.

## Working-tree provenance and boundaries

Starting HEAD:`52fba4f4698e0a8948b00a4a83f3893ae793e55b`. The repository already had
uncommitted edits to root README, research registers/index/timeline, S15 README and
build_assets.py, plus untracked S16 study/evidence/figures. Those were preserved.
S19 added plan/progress, its study/evidence/figures, appended registers/backlog/timeline,
rewrote the current research-index summary and added only the S19 figure hook to the
already-modified asset builder. Root README and S15 were not edited by S19.

No training/model/data configuration code changed; no old study page or frozen plan
changed. No new model training/scoring, git commit, remote publication, external
message or scheduled automation was performed. No AGENTS.md was found in the initial
repository/ancestor inspection. All new research artifacts remain under docs/research/.

## S20 execution started — 2026-10-04

User authorized substantive RGB execution. S20 will establish physical grid identity,
masked RGB crops, full-spectrum compact measurements with k32 parity checks, and
CPU RGB/HSI/fusion controls. S17/S18 remain reserved. Starting state and historical
hashes are saved in `evidence/S20_rgb_pathway/starting_state.json`. No GPU is available.
Frozen predictive design will be saved after label-free data validation and before
held-out feature scoring. Existing historical documents are preserved.

### S20 checkpoint — assets validated, refined masks rebuilding

Initial v1: all 180 scans and 8,624 identities passed, exact masks/morphology/k32
parity, full archive checksum/CRC valid. Visual inspection rejected threshold-only
RGB masks because of truncated dark tips. Final v2 uses bounded seeded GrabCut;
scan 2 (bright plate) and scan 27 (dark seed tip) reviewed. No predictive scores
have been read. v1 feature extraction completed but is rejected, not a result.
Next: finish v2 scan shards/assembly, audit, extract four DINOv2 views, freeze the
30-arm CPU study, execute once, analyze and integrate. New tests pass; broad suite
has five starting-HEAD/environment failures and three intentional S15 source-digest
refusals caused by additive preprocessing files. Historical guards remain intact.

### S20 mask QA correction before scoring

The v2 all-scan audit found one scan-150 crop containing plate background. No
classifier had been run. v2 extraction was interrupted, provenance retained, and
v3 makes the outside of the spatial refinement support certain background.
Full-size review passed scans 2, 27 and 150; v3 is rebuilding all scans with the
same rule. The predictive design is still unfrozen/unscored. Twelve targeted tests
now pass, including RGB transformation controls.

### S20 predictive freeze and execution

Final v3 assets passed all 180 scan checks and visual QA; exact k32 parity holds.
Four DINOv2 feature caches completed on CPU (262–277 seconds per view in two
concurrent two-thread workers). Thirteen targeted tests, Ruff and strict mypy on
six new production files pass. The 30-arm, two-fold manifest was frozen before
predictor fitting/scoring at SHA-256
`c2d4f18394bd031530ba438b4ef2cad04a5b56a97f2eee2cad6a96b67951607a`.
The immutable run is now `outputs/s20_rgb_study`; do not refit or tune from its
held-out results. Next: descriptive saved-prediction analysis and research integration.
