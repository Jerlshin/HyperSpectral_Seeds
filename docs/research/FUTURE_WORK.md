# Future work — the prioritised backlog

What to run next and why, ordered by **how much the answer would change the project per unit of
cost**. Each item names the findings it builds on, the decisions it could reverse, and what a
positive and a negative result would each mean — so whoever picks it up knows what they are
deciding before they spend the compute.

When an item is started it becomes a study (`S##`, see WORKFLOW §5); mark it here with the study
ID and leave the row in place.

---

## Priority 1 — decides what the project is about

### FW-01 · Does reflectance calibration restore cross-session recall?
- **Builds on** F23 (cross-session recall ≈ 0), F26 (fingerprint at the lamp peak), D12 (the reflectance cube was built to fix this).
- **Design.** Re-run the S05 linear and CNN proxies, unchanged, on the 215-band reflectance cube:
  the same folds, the same pre-registered arm families re-cut on the 215 axis (D13), and the same
  same/cross-session breakdown. Pre-register first (H8: *reflectance raises held-out cross-session
  recall above 0.05 for at least one proxy*). Also run the SNV arm on identical rows — the two cubes
  are row-aligned, so the comparison is kernel for kernel. The S05 scripts in
  `evidence/S05_band_research/code/` show exactly what to repeat; `extract.py` must be pointed at
  the new cube.
- **Why it is a sharp test.** All 17 cross-session varieties pair session 8 (no tile saturation,
  i.e. a different illumination level) with one other session (F22). If the confound is
  illumination, reflectance should remove it precisely for that contrast.
- **Cost.** CPU minutes for LDA; ~1 h for the CNN proxy. No network training.
- **If positive:** the session confound was largely illumination shape; reflectance is the
  correct input; cross-session recall becomes the honest variety metric (FW-05).
- **If negative:** the fingerprint is not (only) illumination — detector state, focus, handling or
  seed lot. Variety recognition across sessions is then not measurable from this dataset, and
  that *is* the paper's result. Revisit D12's cost (41 dropped bands).

### FW-02 · Run-to-run σ of SpectralSeedNet (ablation A12)
- **Builds on** D03. Nothing about the network can be compared until σ is known.
- **Design.** Identical config, grouped fold 0, 5 seeds, on Kaggle T4 × 2 (D15). Report σ of
  held-out macro-F1 and of cross-session recall.
- **Cost.** 5 runs. Do it as the first use of the Kaggle setup.

### FW-03 · Neural confirmation of the band budget (S08)
- **Builds on** F17 (fewer bands better for the CNN proxy), challenges D04.
- **Design.** SpectralSeedNet on `uniform430` k ∈ {16, 24, 32, 48, 64} vs the full 215-band cube,
  grouped, 2 folds × 3 seeds, with session reporting (D14). Pre-register non-inferiority at −0.01
  beyond 2σ from FW-02.
- **If k ≤ 64 is non-inferior:** reverse D04 — the default becomes a 430-nm-floored evenly spaced
  set, a 7–13× cheaper input and a deployable multispectral design.
- **If the full cube wins:** the CNN proxy's preference was a property of the proxy; D04 stands.

### FW-04 · The leakage gap for the network (ablation A1)
- **Builds on** D01, F02. `F1_stratified − F1_grouped`, same model, 3 seeds each. Now also report
  the *session* decomposition, which makes it a three-level gap: patch-level → bundle-held-out →
  cross-session.

### FW-05 · Make cross-session recall a first-class metric
- **Builds on** F22, F23, D14.
- **Design question.** Under grouped, 73 classes are evaluated within-session and 17 across. Options
  to evaluate: (a) report cross-session macro-recall as a co-headline beside macro-F1; (b) a
  leave-one-session-out protocol for the 17 cross-session varieties only; (c) restrict claims of
  "variety recognition" to the cross-session subset. Decide this *before* FW-03/FW-04 report, so
  the decision is not shaped by the numbers.

## Priority 2 — tests decisions taken on thin evidence

### FW-06 · What are the hard classes? (ablation A9)
- **Builds on** F09. No training: mean-spectrum overlays, segmentation-quality audit (area,
  eccentricity, solidity distributions; kernels lost to the shape gate), embeddings, 90 × 90
  confusion. New angle from S06: check each hard class's session pattern (class 70 is
  cross-session).

### FW-07 · Falsify D05 and D06 (ablations A3, A8)
- Both were adopted on one leaky run. Run A3 (branches, *symmetric* dropout) and A8 (stages) under
  grouped once σ is known. Either can reverse a decision.

### FW-08 · The paper's baseline table on the reflectance cube
- LDA and LinearSVC on mean spectra under grouped, full cube and uniform430 k32, with session
  breakdown. Seconds of CPU; it is the floor every network number is read against (CHANGES §19.4).
  Already wired: `python scripts/run_protocol.py --baseline`.

### FW-09 · Record why k = 32 (D15)
- The frozen rule gave k\* = 24. Either write the reason for 32 into D15, or include k = 24 in FW-03.

## Priority 3 — housekeeping and longer-term

### FW-10 · Session invariance, only if FW-01 is negative
Speculative: per-session normalisation against the tile spectrum; augmenting illumination shape
during training; a session-adversarial head. Each needs a cross-session test set to be judged,
which only 17 classes provide — so expect wide intervals.

### FW-11 · Re-capture the Stage-1 golden digests
`test_stage1_epoch_loss_matches_golden` / `…_weights_…` fail from torch/BLAS drift (loss 23.0653
vs 23.0805). Re-capture on the target machine with `python scripts/capture_golden.py`.

### FW-12 · More acquisition units per class
The binding constraint (two bundles per class, mostly in one session) is in the data, not the
method. A third bundle per variety, imaged in a *different* session, would allow a real
three-way split and a cross-session test for every class. Worth raising with the dataset authors
or planning as new acquisition.

### FW-13 · A2 as a curve, with the network
Band selection on all data vs within-fold at several k, neural, grouped — the CHANGES §19.3 test,
using the per-fold band files the band study already writes.

### FW-14 · Write-up
Lead with the protocol: stratified → grouped → cross-session, as a three-level gap with intervals;
then the band-budget result; publish the negative results (F05, F09, F18, F25).

---

## Done / moved to a study

| Item | Became |
|---|---|
| Band study inside the fold (CHANGES §19.3 / IC-4) | S03, S05 |
| White-tile reflectance (raised in S06) | S07 |
| Kaggle infrastructure for neural runs | S04 (Oct 2026 part), S08 |
