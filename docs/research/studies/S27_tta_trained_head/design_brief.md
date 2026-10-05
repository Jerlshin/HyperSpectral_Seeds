# S27 · Train the small correction against its intended TTA anchor

> Preserved verbatim as the pre-freeze design brief (2026-10-05). The executed design is in [README.md](README.md); results in [results.md](results.md).

2026-10-05 · **Next proposed experiment; not frozen or executed.**
This page is a resumable design brief. Choose and record the frozen-inference runtime
and seal exact sources/inputs before producing new model outcomes.

## Evidence and architecture

S22 equal fusion improves corrected-fold F1 by .042686 over neural HSI. S23/S25's
23,514-parameter additive correction adds .010522 over matched single-view fusion,
stable over three head initializations, but only .002459 over stronger TTA fusion.
S24's branch removals do not qualify for simplification. S26's post-training TTA
anchor substitution fails calibration on both folds and is rejected without test
scoring. The correction should be trained against its intended deployment anchor.

Retain the S23 model: independent HSI256/RGB384 →32-D GELU terms and additive
readout, initialized at log equal calibrated probabilities. Keep frozen v5/DINO
encoders and the RGB probe; no new attention, encoder tuning, band expansion,
latent-width or optimizer sweep. The learned architecture already accepts an
arbitrary probability anchor. Change its training/calibration anchor to equal TTA
fusion, while retaining the cached single-view feature inputs. Do not claim learned
TTA feature averaging or an individual-kernel interaction.

## Minimum compute and data contract

Initial **head seed 0 on both corrected folds**, two cheap head fits. Reuse existing
S22 seed-0 encoders, S24 feature/scaling caches, S21 RGB probe and S22 saved
calibration/held-out TTA logits. Generate **outer-training-only** frozen HSI TTA
logits once and cache them. R1 TTA has 8 spatial and 4 deterministic foreground
spectral gain views, evaluated in fp32; logits are averaged. Pin exact rows,
masks/morphology, selected live weights, wavelengths, inference source/runtime and
cache hashes. No new encoder fit is required.

Assess CPU versus an available GPU with a training-only inference timing probe
before freezing the runtime. Twelve views make this inference work material;
do not label it free or invent a runtime estimate. The closed private S22 host is
available as execution provenance, not an active new job. Credentials remain local.

Retain S23's fixed head recipe, train-only feature scales/gradients and calib-only
checkpoint selection, including epoch 0. Any temperature/fusion recipe is fixed or
calib-fitted before held-out scoring. Record TTA-anchor and single-view controls
on exact rows, acquisition directions and error rescues/harms. If all corrections
are rejected by calibration, stop without a new held-out predictor.

## Proposed development gate and later confirmation

Before execution, seal H45-screen: learned minus matched **equal TTA** mean F1 ≥.01,
paired variety interval >0, positive F1 each fold and nonnegative mean cross delta.
Require positive cross interval for transfer support. No automatic three-seed
allocation for this intermediate. A clear practical improvement earns selective
head sensitivity using the new caches; a selected final recipe earns a matched
encoder-seed check and eventual seeds 0/1/2 across both folds.

Final learned system and HSI/fixed-fusion controls share encoder fits. Relevant
modality/necessity controls need matched confirmation if they support a paper
claim. Deterministic RGB refits do not add seed evidence. Crossed new sessions/lots
and a locked external evaluation remain necessary; all current bridges touch
session 8 and all existing test outcomes are reused development evidence.
