# Hypotheses register

Every hypothesis that was written down *before* its test, with the test, the outcome and the
evidence. Planned-but-unrun ablations are listed too, so it is visible which questions the
project has posed and not yet answered.

**Outcome values:** `supported` · `rejected` · `mixed` (holds for one proxy or region, not
another) · `undetermined` (the saved evidence cannot decide it) · `not run`.

---

## 1 · Pre-registered held-out hypotheses (S05, S06)

Frozen in `evidence/S05_band_research/preregistration.frozen.json` (SHA-256 `5ae90caf…`,
2026-09-29 22:39 CEST, repo commit `78eea0d`) and `preregistration2.json` (`2077efc9…`,
22:46 CEST). Protocol: grouped, fit on train ∪ calib (the fold's training bundle), score val ∪ test
(the other bundle) once, 2 folds; LDA with shrinkage 1e-3 on SNV mean spectra; CNN proxy 2 folds ×
2 seeds. All numbers below are means over folds (and seeds) and are recomputable from
`evidence/S05_band_research/heldout_lda_summary.csv` and `cnn_summary.csv`.

| ID | Hypothesis (as frozen) | Outcome | Evidence |
|---|---|---|---|
| **H1** | Arms containing 383–430 nm bands lose their within-bundle advantage over their ≥ 430 nm counterparts on held-out bundles | **mixed** — holds for the CNN, rejected for LDA | LDA uniform − uniform430: +0.038 / +0.059 / +0.059 / +0.053 at k16/32/48/64; glw − glw430 +0.046…+0.071. CNN uniform − uniform430: −0.002 / +0.004 / +0.010 at k16/32/64; glw − glw430 −0.002 / +0.004. The frozen switch condition ("falsified on **both** proxies, ≥ 0.02") is **not** met, so the R1 track stood. |
| **H2** | Held-out uniform430 peaks at k ≤ 64 and is non-inferior (≥ −0.01) to the full 256-band cube somewhere in k ∈ [24, 64] | **mixed** — supported for CNN, rejected for LDA | CNN uniform430 peaks at k48 (0.453); k32/48/64 all exceed full-cube 0.437. LDA uniform430 peaks at k128 (0.361); best in [24, 64] is 0.344 vs full 0.419 (−0.075). |
| **H3** | No supervised selector beats uniform430 by > 0.01 (mean of 2 folds) at k ≥ 32 on held-out | **supported** for CNN; one LDA cell exceeds | CNN glw430 − uniform430: −0.006 (k32), −0.004 (k64). LDA: +0.007, **+0.013**, +0.005, +0.005, −0.004 (k32…128). → F18 |
| **H4** | The within-bundle gain from log μ, log sd (gstat3) over −μ/sd alone (gstat1) shrinks on held-out | **undetermined** | Held-out gstat3 − gstat1 = +0.062 / +0.053 / +0.053 / +0.045 / +0.046 (k16…128). The within-bundle value (a6) was printed, not saved, so "shrinks" cannot be checked. Gain stats are strongly session-informative (F26). |
| **H5** | Low-pass DCT (m = 48–64) ≥ every selected band set of equal size on held-out | **rejected** | DCT m48 0.391 / m64 0.394 vs glw k48 0.408 / k64 0.408. DCT does beat uniform and uniform430 at equal size. → F20 |
| **H6** | Session-clean arms (bands with session F < 2.2) raise held-out cross-session recall above 0.05 | **rejected** | All 12 clean arms: 0.0006–0.0012. → F25 |
| **H7** | Session-clean arms lose macro-F1 vs unconstrained arms of equal k; the loss estimates the session-recognition share | **supported as stated; interpretation fails** | clean_uniform_k64 0.318 vs uniform_k64 0.397 (−0.078); clean_glw_k64 0.322 vs glw_k64 0.408 (−0.086). Because H6 failed, the clean arms did not remove session recognition, so the loss is lost variety signal, not a session share. → F25 |

**Pre-registered decision rule and what happened to it** — see [D11](DECISIONS.md#d11--finalists-are-uniform430--deviation-recorded).

## 2 · Pre-registered decision rules of the band study (S03)

Set in `BandStudyConfig` before the run; each could return "no".

| Rule | Threshold | Outcome |
|---|---|---|
| A curve **plateaus** at the smallest k within 0.01 of its own peak, *demonstrable* only if the curve extends past it | 0.010 macro-F1 | 44 / 72 plateaus demonstrable; 28 are the curve's endpoint |
| A selection is **stable** at mean replicate Jaccard ≥ 0.50 | 0.50 | every method except `random` stable at k ≤ 40 |
| A method is **effective** only if it beats a random subset of equal size by ≥ 0.01 somewhere | 0.010 | `variance` and `random` never effective; `uniform` (a null) ranked first |

## 3 · Audit hypotheses and the ablation plan (S01 → S02)

Defined in CHANGES.md §20 and registered in `experiments/registry.py`. A1 and (partly) A12 were executed
by the S08 sweep on 2026-10-01 and analysed in S09; the rest have not run. Each row says which decision it
would confirm or reverse.

| ID | Question | Decision rule | Gates | Status |
|---|---|---|---|---|
| A12 | Run-to-run σ (identical config, 5 seeds) | prerequisite for reading every other delta | all | **measured (S09, F30)** — σ = 0.009 grouped (6 runs, 2 folds), 0.033 stratified (6 seeds) |
| A1 | Leakage gap: same model, `stratified` vs `grouped`, 3 seeds each | the gap *is* the Q1 headline | D01 | **run (S09)** — +0.182; confounded with +39 % training data (F38 bounds that at ≈ 0.01, linear) |
| A2 | Band selection on all data vs within-fold, grouped | gap > 2σ ⇒ published numbers on this dataset need the caveat | D04, D08 | **not run** (proxy evidence: S03, S05) |
| A3 | {A,B,C,D} vs {B,C} vs {C} vs {B}, symmetric branch dropout | 4-branch − {B,C} < 2σ ⇒ removal confirmed | **D05** | **not run** — FW-07; for SpectralSeedNet's two pathways X2 is the analogue: spectral-only −0.099, spatial-only −0.081, late fusion ≈ joint (S12, F64, F65) |
| A4 | Branch A `grid_size_a` ∈ {8, 4, 2} | only if A3 keeps A | D05 | not run |
| A5 | bilinear+gate vs gate-only vs concat+MLP | only if A3 keeps ≥ 3 modalities | D05 | not run |
| A6 | CE vs CE + SupCon, balanced sampler both arms | SupCon earns a Phase B only if > 2σ | D06 | not run |
| A7 | margin 0 / global 0.30 / + per-class / + Ω penalty | one variable per arm | D07 | not run — S10 (F47): only with ≥ 25 % of the LR budget and ε = 0 (S10 P1.4); S12: m = 0 is non-inferior (X1, F59), margin dropped with R1 (D24) |
| A8 | S1 vs S1+S2 vs S1+S2+S3, grouped, 3 seeds | falsification test for the single stage | **D06** | **not run** — FW-07 |
| A9 | What are classes {41, 49, 51, 52, 70}? (no training: embeddings, overlays, confusion, segmentation audit) | genetic vs segmentation vs acquisition | — | **partly answered (S09, F42)** — 70 is acquisition; {0, 30, 41, 49, 51, 52} a mutual-confusion cluster; segmentation audit not run |
| A10 | Spatial-path width × {0.5, 0.75, 1, 1.5} | is capacity harmful? | D05 | not run — S12: H16 supported (capacity ample); width is not a lever under D23 |
| A11 | mixup on/off × augmentation profile | is mixup the load-bearing regulariser? | D07 | **answered in part (S12)** — mixup 110 → 30 with no margin is neutral on held-out (F59); all softeners off costs −0.06…−0.17 (F61); the single-factor split is still unmeasured |

## 4 · S08 drafts (never frozen) — what the sweep showed, and the frozen S09 → S10 set

**S08 drafts, observed in S09.** These were written in the S08 page but not frozen before the sweep, so
their outcomes are *observations*, not confirmations.

| ID | Draft | Observed | Evidence |
|---|---|---|---|
| H8 | Reflectance raises held-out cross-session recall above 0.05 for at least one proxy | **borderline** — LDA spectrum-only 0.039 (k32), 0.053 (215); SNV on identical rows not run | `c3_representation.csv` |
| H9 | uniform430 at some k ∈ {24…64} is non-inferior to the full 215-band cube (network) | **not run** — the sweep had only k32; LDA prefers 215 (+0.04 grouped) | — |
| H10 | The network's cross-session recall exceeds 0.05 on reflectance | **observed 0.152** — but morphometrics alone give 0.124 (F33), so the spectrum's share is unknown → H14a | `runs.csv` |
| H11 | `F1_stratified − F1_grouped` > 2σ | **observed +0.182** (≫ 2σ under either σ) — size-confounded (F38) | `summary.json` |

**Frozen for the next study.** `evidence/S09_post_sweep_forensics/preregistration_next.json`,
SHA-256 `f896d0e569e0c071f85fb1c7e2fbce23cd3266da1ef317204fa329af9bb542e7`, frozen 2026-10-01 before any
of these arms ran. Reference values: grouped 0.530 (σ 0.009), stratified 0.712 (σ 0.033), TTA.

| ID | Hypothesis (as frozen) | Experiment | Decision it drives | Status |
|---|---|---|---|---|
| H12a | Fit-first raises stratified macro-F1 by ≥ +0.05 | X1 (FW-15) | is the regime the limit? | **rejected** (S12) — 0.727, Δ +0.015 (CI −0.012…+0.041) |
| H12b | …and grouped by ≥ +0.02 with Δgrouped / Δstratified ≥ 0.37 | X1 | **route A vs B (D17)** | **rejected** (S12) — 0.531, Δ +0.001 (CI −0.010…+0.011); ratio 0.05 |
| H13 | Fit-first reaches clean training accuracy ≥ 0.95 | X1 | did the regime cause F35? | **supported** (S12) — 0.980 grouped / 0.992 stratified at the selected checkpoint; 8/9 cells ≥ 0.95 |
| H14a | Morphometrics-zeroed cross-session recall ≤ 0.07 | X2 (FW-16) | is cross-session recall shape-driven? (D12, D14) | **rejected** (S12) — 0.144 |
| H14b | full − spectral-only grouped ≤ +0.02 | X2 | is the spatial pathway dead weight? (D05) | **rejected** (S12) — +0.099 (CI +0.087…+0.112) under either definition of "full" |
| H15 | 80/20 − 59 %-stratified (same regime) ≤ +0.03 | X3 (FW-17) | is training share the literature gap? (D16) | not run — scheduled as S13 Y4 under R1 |

**S10 interpretation guards (D21), recorded before X1 runs; the frozen file is unchanged.** H13's reference,
measured as H13 defines it (eval mode, no augmentation, margin 0), is 0.90 grouped / 0.91 stratified (F45), not
0.76–0.87 — reaching 0.95 is +0.05. An X1 effect is credited to mixup duration, margin and epochs jointly (clip 50 is
near-neutral under AdamW, F53). The reference regime applied aux 0.65 → 0.25, not 0.2 (F54); X1 inherits the same
schedule as a function of progress only while the code default stays `legacy`.

## 5 · S10 → S11: frozen arms

`evidence/S10_training_architecture_review/preregistration_s10.json`, SHA-256
`1c8ae6937796cd7867f58ae901c1630671869f949ef8067cfefd72b2d4922739`,
frozen 2026-10-01 before any of these arms exist. Reference = the same-regime grouped mean (X1's grouped runs if
H12a is supported, else the S09 sweep, 0.530); σ_ref = max(observed sd, 0.009).

| ID | Hypothesis (as frozen) | Experiment | Decision it drives | Status |
|---|---|---|---|---|
| H16 | With every softener off (mixup 0, ε 0, aux 0, dropout 0, no augmentation) under X1's schedule, clean training accuracy ≥ 0.98 at the final epoch (grouped fold 0) | X4 (FW-20) | capacity vs regime (D19's reversal trigger) | **supported** (S12) — 1.000 (stratified companion 1.000) |
| H17 | Last spatial-tail block at stride 1 (no untrainable parameters) is non-inferior: grouped ≥ reference − max(2σ_ref, 0.018) | X5 (FW-21) | adopt the stride fix as default | not run — the change joins S13 Y3 (D25); **S16:** its closest arm, `spatial_repair` (X5 + no CBAM on 2 × 2), gained +0.035 F1 vs X1 s0 at one seed (F87) |
| H18a | Log-reflectance level block in the spectral descriptor raises grouped by ≥ max(2σ_ref, 0.02) | X6 (FW-22) | adopt level (with H18b) | not run — deferred (D25; linear proxy F70) |
| H18b | …without losing > 0.03 cross-session recall or adding > 0.05 cross-session attraction | X6 | the gain is not session recognition | not run — deferred (D25) |

**S11 (part 1).** The measurement for every hypothesis above now exists, and where each will be read from — file
and key, fixed before any arm runs — is S11 §6.1. One reading question is flagged there rather than resolved: H14b's
"full" (the four matching S08 cells or the six-run mean). Statuses stay `not run`.

**S12 (reading).** All six run hypotheses read from saved evidence (`evidence/S12_frozen_arms_reading/hypotheses.json`).
Routes as frozen: S09 → **route A** (D23); S10 → capacity ample, X5/X6 not run standalone (D25). H14b's "full" was not
fixed before X2 was read (S11 risk 6); both definitions give +0.099.

## 6 · S12 → S13: frozen arms

`evidence/S12_frozen_arms_reading/preregistration_s12.json`, SHA-256
`88b377c5bc32f31a9ccb95356a88eb0be35e8cb51216b035c88f1761919ae7f4`, frozen 2026-10-02 before any of these arms or their
code exist. Reference = X1 (R1, D24): grouped 0.530786 (σ 0.0099), same-session 0.648 (σ 0.014), cross-session 0.147
(σ 0.0098), attraction 0.464 (σ 0.020), stratified 0.727 (σ 0.0058). Margins are 2·max(sd, 0.009).

**Amended before any arm ran (D28):** S13 runs every cell at seed 0 only —
`evidence/S13_representation_screening/preregistration_s13.json` (hashed; the parent file is unchanged). Thresholds,
reference and decision logic are as frozen; outcomes are **screening verdicts** — an arm that passes is replicated at
seeds 1–2 (FW-35) before its decision rule is read as confirmatory.

| ID | Hypothesis (as frozen) | Experiment | Decision it drives | Status |
|---|---|---|---|---|
| H19a | Decoupled pathways fused on calib: grouped F1 ≥ 0.5108 | Y1 (FW-31) | retire joint fusion (with H19b) | **supported (screen)** — 0.545 (folds 0.556 / 0.535), clear (F76) |
| H19b | …and cross-session recall ≥ 0.1666 and attraction ≤ 0.4240 | Y1 | decoupling buys session robustness | **rejected (screen)** — cross 0.134, attraction 0.490; every fold fails both (F76–F78) → joint network stays |
| H20 | Masked MixStyle in the 3-D stem: cross-session ≥ 0.1666 and same-session ≥ 0.6285 | Y2 (FW-32) | a move off the frontier, not along it | **rejected (screen)** — cross 0.143 (same 0.641 met); not a frontier move (F79) |
| H21a | Lean architecture (SNV + morph descriptor, tail [2,2,2,1], no CBAM on ≤ 2 × 2): grouped ≥ 0.5108 | Y3 (FW-33) | adopt SeedNet v5 (with H21b) | **supported (screen)** — 0.562 (0.565 / 0.559), clear; above every X1 run (F74) → replicate (§7) |
| H21b | …and stratified ≥ 0.7090 | Y3 | non-inferior within acquisition | **supported (screen)** — 0.746, one run (F74) |
| H15 | (S09, unchanged) 80/20 − R1 stratified ≤ +0.03 | Y4 | D16 tier 1 | **supported (screen)** — +0.001 (0.728), one run (F80) |

**S14 (reading).** Read exactly as frozen, as screening verdicts (`evidence/S14_screen_reading/hypotheses.json`); each
verdict carries a *screen interval* (kernel bootstrap combined with X1's run-level sd / √n). No arm's two folds straddle
its threshold, so D28's "replicate before reading" clause does not apply. Routes: Y3 passes → §7; Y1, Y2 screening
rejections (D30).

## 7 · S14 → S15: Y3 replication and dissection, frozen

`evidence/S14_screen_reading/preregistration_s14.json`, SHA-256
`9e182670755e13a6ead29a761123841094a9b9fb6fae18392553893d65c1f3da`, frozen 2026-10-03 before any S15 cell or runner
exists. H21a/H21b are the parent's and are read on seeds 0–2 as frozen there. **H21c–H21e and H22a/H22b are motivated by
Y3's seed-0 results** (F74, F75); H21c–H21e are read on the fresh seeds (1, 2) only, against X1's cells at the same folds
and seeds (grouped 0.528496, stratified 0.728593). **Implemented in S15 part 1** (`scripts/run_s15.py`, D34); the
reading map is S15 §8.

| ID | Hypothesis (as frozen) | Experiment | Decision it drives | Status |
|---|---|---|---|---|
| H21a | (parent) Y3 grouped F1, 6 runs (seeds 0–2) ≥ 0.5108 | Y3 replication (FW-35) | adopt SeedNet v5 (with H21b): defaults → R1 + lean keys | **supported** — 0.571 (CI 0.559…0.579), clear (F84) → **v5 adopted (D35)** |
| H21b | (parent) Y3 stratified F1, 3 runs ≥ 0.7090 | Y3 replication | as above | **supported** — 0.745 (CI 0.724…0.760), clear (F84) |
| H21c | Y3 − X1 grouped F1 on seeds 1–2 ≥ +0.020 and its hierarchical-bootstrap CI excludes 0 | Y3 replication | the paper may claim a grouped gain (G3) | **supported** — +0.047 (CI +0.034…+0.060), clear (F84) |
| H21d | Y3 grouped cross-session recall ≥ 0.1666 and attraction ≤ 0.4240 on seeds 1–2 | Y3 replication | the paper may claim cross-session robustness | **supported** — cross 0.206 (clear), attraction 0.410 (CI 0.384…0.437, marginal) (F85) |
| H21e | Y3 − X1 stratified F1 on seeds 1–2 ≥ +0.018 | Y3 replication | within-acquisition gain | **rejected** — +0.016 (CI −0.005…+0.037) → not claimed (F86) |
| H22a | `desc_only` (SNV + morph descriptor alone) grouped F1, seed 0 ≥ 0.5508 | Y5 dissection (FW-36) | which removal carries the gain (only if H21a ∧ H21b) | **supported (screen, marginal)** — 0.553 (folds 0.568 / 0.538 straddle within the margin; D28 clause not fired) (F87) |
| H22b | `spatial_repair` (tail [2,2,2,1] + no CBAM on 2 × 2 alone) grouped F1, seed 0 ≥ 0.5508 | Y5 dissection | as above | **supported (screen)** — 0.570, clear → H22a ∧ H22b: *redundant, both kept*; robustness tracks the spatial repair (F87) |

**S16 (reading).** Read exactly as frozen (`evidence/S16_replication_reading/hypotheses.json`): H21a–H21d supported, H21e
rejected, H22a (marginal) and H22b supported. Seed 0 was Y3's lowest seed — the fresh-seed reading H21c–H21e guarded
against a winner's curse that did not occur. Routes: SeedNet v5 adopted (D35).

## 8 · S16 → S17: v5's tier-1 row and two screens, frozen

`evidence/S16_replication_reading/preregistration_s16.json`, SHA-256
`3b623c45c559962c36b59383ddf6a636da9a087e41523d8d5c455e81954b9d3e`, frozen 2026-10-03 before any S17 cell or runner
exists. Reference = v5 at seeds 0–2 (D35): grouped 0.570816 (σ 0.0102), cross 0.199401 (σ 0.0138), attraction 0.412773
(σ 0.0121), stratified 0.744788 (σ 0.0100); margins 2·max(σ, 0.009). Z2 and Z3 are screens (D28): a pass is replicated
at seeds 1–2 before it can replace v5. **Z3 is motivated by S16's post-hoc dissection profile (F87).**

| ID | Hypothesis (as frozen) | Experiment | Decision it drives | Status |
|---|---|---|---|---|
| H23 | v5 80/20 stratified F1 (3 runs) − v5 70/30 (0.744788) ≤ +0.030 | Z1 (FW-37) | the D16 tier-1 row; does v5, unlike X1 (F80), gain from same-acquisition data? | not run |
| H24a | v5 at uniform430 k64, grouped F1 (seed 0, 2 runs) ≥ 0.550404 | Z2 (FW-03) | k64 non-inferior → k32 stays, FW-03 closed for k ≤ 64 | not run |
| H24b | …≥ 0.591228 | Z2 | superior → with H24c: replicate k64, screen the 215-band cube | not run |
| H24c | …cross-session recall ≥ 0.171824 and attraction ≤ 0.436934 | Z2 | a gain without session cost (else F70's pattern: k32 stays) | not run |
| H25a | v5 with a 4 × 4 end map (tail `[2,2,1,1]`), grouped F1 (seed 0, 2 runs) ≥ 0.550404 | Z3 (FW-38) | non-inferior | not run |
| H25b | …cross-session recall ≥ 0.226978 and attraction ≤ 0.388612 | Z3 | with H25a: end-map extent is a robustness lever (replicate); else the repair was one step (v5 stays) | not run |

## 9 · Adding a hypothesis

Write it here *before* the run, with: the exact claim, the metric and split, the threshold, what
outcome supports it, and which decision or finding it would change. If held-out data will be
touched, also freeze it (WORKFLOW §3). Fill *Outcome* only from saved evidence, and say
`undetermined` when the evidence was not saved — as H4 shows, an unsaved number is a lost test.

## 10 · S19 next-generation hypotheses — draft, not frozen or run

These hypotheses are informed by historical confirmation data and current literature. They are proposals, not
preregistered outcomes. Exact manifests, model variants, thresholds and selection rules must be frozen in the next
execution study before confirmation. Planning margins appear in S19 `experiments_and_paper.md`.

| ID | Question / hypothesis | Decisive control | Decision / status |
|---|---|---|---|
| H26 | High-resolution foreground RGB supplies information complementary to HSI under acquisition shift | RGB-only, HSI-only, logit fusion; silhouette/grayscale and resolution controls on matched kernels | D38; untested; require replicated F1/transfer gain, not only within-acquisition improvement |
| H27 | Fuller measured spectra add discriminative information beyond32 after capacity, axis coverage and calibration are controlled | Historical and nested32/64, 195≥430,full 215; fixed-width spectral model and uniform/random controls | D39; untested; choose smallest noninferior budget |
| H28 | Physical wavelength/gap handling improves transfer across band sets compared with index-only encoding | Matched parameter/training budget; identical spectra, physical axis versus index encoding | D38/D39; untested; otherwise retain simpler encoding |
| H29 | Local arrangement adds robust information beyond the within-kernel spectral distribution | Quantile baseline, shared pixel bag, same encoder with occupied spatial regions | D38; untested; isolate spatial information rather than extra parameters |
| H30 | A nuisance family estimated from independent/training standards supports class-preserving correction/consistency | No correction, measured-nuisance intervention, generic matched-strength jitter; class-detail checks | D38/D40; conditional on identifiable measurement evidence; untested |
| H31 | Externally pretrained RGB/HSI features improve data efficiency beyond classical summaries and scratch ERM | Frozen probe, limited tuning, matched simple baselines; provenance/overlap checks | D38; untested; transfer is not presumed from remote-sensing/natural-image results |

## S20 · RGB execution hypotheses (2026-10-04, before execution)

- **H32:** Grid-constrained correspondence can uniquely pair retained HSI kernels to RGB; fail closed on ambiguous cells/residuals and report exclusions.
- **H33:** Masked RGB appearance improves grouped macro-F1 over RGB morphology by ≥.02, with paired class-bootstrap interval excluding zero. A CPU screen, not a broad robustness claim.
- **H34:** Simple RGB–HSI fusion improves grouped macro-F1 over matched HSI by ≥.02 without reducing observed cross-session recall. Retain unimodal alternatives if this fails.
- **H35:** Expanded spectral coverage improves matched low-capacity probes by ≥.01 macro-F1 without reducing cross-session recall. Otherwise retain k32 provisionally; this cannot reject all nonlinear full-spectrum models.

Predictive arm manifest and scoring guard will be frozen after label-free asset validation, before held-out scoring.

S20 pre-freeze clarification: **H34v5** also compares equal calibrated RGB fusion
against each of the three existing v5 TTA seeds, reusing saved calibration/held-out
logits with exact row/target checks. The same ≥.02 F1 / nonnegative cross-recall rule
applies; shared frozen RGB features do not constitute three RGB replications. No
new GPU training is implied. These arms were added before any S20 scoring.

Pre-freeze operational definition: H33's decisive shape control is the same frozen
DINOv2 encoder on a binary silhouette, isolating appearance at matched encoder
capacity. Eight explicit morphology values are an additional baseline. H35a uses
nested 32/64 quantiles; H35b compares mean214-own versus mean32. The three-seed v5
aggregate must also pass the same H34 threshold and paired interval rule; a shared
RGB branch does not create independent RGB replications.

## S21 · Complementary-fold follow-up (2026-10-04, before S21 execution)

S20 motivated a separately frozen 24-arm screen using independent group/patch RNG
streams. This is a declared adaptive follow-up on old acquisitions, not fresh evidence.
H36: RGB versus silhouette ≥.02 F1 and paired CI >0. H37: equal RGB/HSI32 fusion versus
HSI32 ≥.02 F1, CI >0, nonnegative cross-recall delta. H38: correctly paired versus
within-scan shuffled fusion ≥.01 F1 and CI >0; otherwise no individual-kernel
interaction claim. H39a/b: nested32/64 quantiles and mean32/214-own respectively,
≥.01 F1, CI >0 and nonnegative cross-recall delta. No historical v5 reuse because
its training acquisitions differ from the repaired partitions.

## S20/S21 outcomes and S22 continuation (2026-10-04)

| Hypothesis | Outcome | Interpretation |
|---|---|---|
| H32 | pass | 8,624 unique pairs, exact k32 parity, all16 original exclusions preserved; mask annotation remains absent |
| H33 | pass | S20 RGB vs silhouette F1 +.277696 [.242646,.310566] |
| H34 / H34v5 | pass bounded gates | S20 HSI32 fusion +.052257; three-seed v5 fusion +.040896 [.026310,.055735]; v5 cross-recall CI spans zero, no established transfer gain |
| H35a/b | fail | nested64 and full214 mean improve F1 but reduce observed cross recall |
| H36 | pass | S21 RGB vs silhouette +.281974 [.244211,.316018] |
| H37 | pass | S21 equal fusion +.056970 [.039397,.075737], cross +.049917 [.003676,.111698] |
| H38 | fail | correctly paired minus shuffled F1 −.014112 [−.020092,−.007799] |
| H39a/b | fail | S21 band-expansion transfer deltas negative |
| H40 | frozen, GPU unrun | S22 three-seed/two-fold equal v5+RGB F1 gain ≥.02, paired CI >0, cross delta ≥0; positive cross CI additionally required for transfer claim |

The S20 plan's “both directions” wording was corrected after a coverage audit;
its actual rows/plans remain immutable. S21 exhaustively covers both directions.
H26/H31 now have positive bounded CPU evidence, not complete neural/external validation;
H27's simple-probe transfer tests are negative. H28–H30 remain untested. These screens
reuse historical acquisitions and do not establish independent replication.
[S20 results](studies/S20_rgb_pathway/results.md) ·
[S21 results](studies/S21_complementary_rgb/results.md) ·
[S22 preregistration/runbook](studies/S22_complementary_v5/README.md).

## S22 amendment02 (2026-10-05, before training)

Original H40 three-seed confirmation is superseded for initial development, not
reported as evaluated. **H40-screen:** at seed0 on both corrected acquisition
folds, equal v5/RGB gains mean F1 ≥.02, paired variety interval >0, each fold F1
positive, and mean cross-recall delta ≥0. Positive cross interval is required for
transfer support. These class intervals do not include initialization uncertainty.
A clear pass advances a minimal learned multimodal candidate; seed confirmation
is reserved for the final frozen model and matched controls. See S22 amendment02.

## S23 prospective gate (2026-10-05, before S22 result)

**H41-screen:** one32-D additive residual head over frozen HSI256/RGB384 features,
seed0 and both corrected folds. Against matched equal single-view fusion: mean
F1 gain ≥.01, paired variety interval >0, each fold gain positive, mean cross recall
delta ≥0. Practical adoption additionally requires mean F1 and cross recall at least S22's equal-TTA
fusion. Epoch0 is eligible for calib checkpoint selection. No attention, encoder
fine-tuning or hyperparameter sweep. Run only if H40-screen passes; otherwise the
prospective code is retained unexecuted. [S23](studies/S23_frozen_multimodal/README.md).

## S22 observed screen (2026-10-05)

**H40-screen passes:** seed 0, both corrected folds; F1 gains .045339/.040033,
mean +.042686 [.027554,.057266]. Cross gain +.000912 [−.047802,.049617] is not
supported. Original confirmatory H40 is not evaluated. S23's source/input plan
was sealed after this recorded pass and before its head fitting/evaluation;
H41 remains subject to the prospective recipe and practical TTA comparator above.
[S22 results](studies/S22_complementary_v5/results.md).

## S23 observed screen and prospective S24 (2026-10-05)

H41-screen and practical point gates pass. Matched F1 gain +.010780
[.002308,.019329]; advantage over TTA only +.002717 [−.005946,.011384].
Cross-recall intervals span zero against both comparators. [S23 results](studies/S23_frozen_multimodal/results.md).

**H42-screen**, frozen before S24 outcomes: both learned feature branches are
needed only if S23 beats each branch-removal control by mean F1 ≥.005, paired
interval >0, positive F1 each fold and nonnegative cross delta. Each control retains
both modalities in the fixed probability anchor and the same head recipe, but fewer
active parameters. The separate provisional simplification rule requires H41/practical
gates plus F1 within .005 and cross recall within .01 of S23. One seed, both folds,
four cheap heads; no new encoders or automatic seed expansion. [S24](studies/S24_branch_multimodal/README.md).

## S24–S26 observed outcomes and proposed S27 (2026-10-05)

| Gate | Outcome | Evidence and limits |
|---|---|---|
| H42-screen | fail | S23 versus HSI-only correction +.004767 < .005; versus RGB-only +.008664. Positive class intervals do not meet both practical thresholds. Neither smaller candidate qualifies for simplification. |
| H43-screen | pass | Full-head matched gain +.010522 [.001851,.018941] across head seeds0/1/2, positive both folds at every seed. Fixed encoder0 only; descriptive seed-delta SD .000279. |
| S26 calib gate | fail both folds | Fixed TTA-anchor substitution reduces calib F1 .002395/.005613; no new test scores or fits. |
| H44-screen | not evaluated | Calib gate never opens held-out or conditional head-seed evaluation. |
| H45-screen | proposed, not frozen | S27 head trained against intended equal-TTA anchor; mean matched F1 gain≥.01, paired variety CI>0, positive each fold, nonnegative cross delta; positive cross CI additionally required for transfer support. |

H43's exact plan was frozen before its four additional head fits; seed0 was reused,
not retrained. S26's plan was frozen before its fixed calibration intervention.
H45 is a design brief, not a preregistration or outcome. Freeze source/runtime,
train-TTA cache and exact gates before S27 execution. All class intervals are
conditional on reused acquisitions and omit encoder/new-session uncertainty.
[S24](studies/S24_branch_multimodal/results.md),
[S25](studies/S25_head_seed_screen/results.md),
[S26](studies/S26_tta_anchor/results.md), [S27](studies/S27_tta_trained_head/README.md).

## S27–S29 observed outcomes and proposed S30 (2026-10-05)

| Gate | Frozen plan | Outcome | Evidence and limits |
|---|---|---|---|
| H45-screen | `s27_tta_trained_head.json` `780d564e…` | **pass** | TTA-trained head − equal TTA +.013124 [.004438, .021629], folds +.0125/+.0137, cross +.0274 [−.0028, .0629]: transfer unsupported |
| H46-screen | `s28_tta_head_seeds.json` `b0396c94…` | **pass** | head seeds 0/1/2 +.013942 [.006019, .021463], seed SD .000871, every seed positive both folds; encoder seed fixed |
| H47-screen | `s29_rgb_backbone_screen.json` `ee51803a…` | **pass** | calib-selected ViT-L head − S27 ViT-S head +.012431 [.005679, .019846], folds +.0132/+.0117; cross +.0111 [−.0074, .0331] |
| H44-screen | S26 | still not evaluated | superseded as a question by H45 (train against the anchor rather than substitute it) |

H45 was frozen after a train-only TTA cache and before any head fit. H46 after S27's outcome, as
S27's pre-declared pass path. H47 after S28; its first unfrozen draft (conditional on S27
failing) was replaced before freezing, as disclosed in S29 §2.

**Proposed for S30 (not frozen):** C1 equal ViT-L − HSI TTA ≥ .02; **C2 head ViT-L − equal
ViT-L ≥ .01** (decides whether the final system keeps a learned fusion component); C3 equal
ViT-L − equal ViT-S ≥ .01; C4 head ViT-S − equal ViT-S ≥ .01. All are means over encoder seeds
0/1/2 × both folds, with paired CI > 0, positive in each fold, and Δ > 2 × seed SD.
Transfer claims additionally need a gain in the away-from-session-8 direction (F113).
[S30 brief](studies/S30_final_confirmation/README.md).

## S31–S32: how much RGB is left unused? (registered 2026-10-05, before any outcome)

| ID | Study | Statement and gate | Recorded prediction |
|---|---|---|---|
| H48-screen | S31 frozen ViT-L readouts | The calib-selected readout (class + foreground-mean token, last-4 layers, and/or 4-view orientation averaging) beats the class-token probe on RGB alone: ΔF1 ≥ .02, paired variety CI > 0, positive each fold, cross Δ ≥ 0 | modest pass or near miss (+.01–.03); readout changes are mostly within-session |
| H49 (descriptive) | S31 | The same readout gain survives equal fusion with v5 TTA; metric morphometrics add to RGB alone but not to the system (v5 already consumes morphometrics, F33/F63) | system gain ≤ half the RGB gain; morph adds ≈ 0 at system level |
| H50-screen | S32 trained RGB branch | Calib-selected fine-tuned DINOv2 (ViT-B or ViT-L, foreground tokens, 4-view TTA) beats the S31 frozen ViT-L reference on RGB alone: ΔF1 ≥ .02, CI > 0, both folds, cross Δ ≥ 0 | pass on F1 (+.05 or more, mostly same-session); cross Δ near zero with session attraction rising |
| H51-screen | S32 | Equal fusion of v5 TTA with the trained branch beats equal fusion with the frozen reference: ΔF1 ≥ .01, CI > 0, both folds, cross Δ ≥ 0 | pass, but well below the RGB-only gain (shared errors with HSI) |

Gates follow the project's screening rule (`screen_metrics.gate`). These are development screens
on reused acquisitions under [D53](DECISIONS.md); intervals describe variety heterogeneity, not
new-session uncertainty.

**S31 outcomes (2026-10-05, plan `29ef87f3…`):**
- **H48-screen: fail, on the cross clause.** `last4_tta` − `cls` RGB-only +.1258 F1 [.1068, .1447],
  both folds positive, but cross −.0070 [−.0259, .0123]. The prediction (+.01–.03, within-session)
  underestimated the F1 gain fourfold; its within-session character was right.
- **H49 (descriptive): confirmed.** The system gain (+.0497) is under half the RGB gain (+.1258).
  Morphometrics add RGB transfer (+.0148 cross, CI > 0) but ≈ 0 at system level (equal morph vs
  HSI −.0047; + morph at system +.0106 F1, cross n.s.), as predicted.

## S33–S35: measured nuisance, strong-RGB fusion, class-conditional rendering (registered 2026-10-05, before scoring)

| ID | Study / plan | Statement and gate | Recorded prediction |
|---|---|---|---|
| H52-screen | S33 `2873190a…` | ViT-B trained with measured-nuisance blur (σ ~ U(0,1), p = .5) and scored on σ = 0.5 rendered crops vs S32 ViT-B: from-session-8 bridge recall Δ ≥ .03 (cell-bootstrap CI > 0) and F1 Δ ≥ −.01 | partial: from-s8 rises a little (+.01–.03), F1 falls slightly; likely fail on the CI |
| H53-screen | S33 | same contrast after equal fusion with v5 TTA; F1 Δ ≥ −.005 | fail (HSI dominates the from-s8 direction) |
| H54-screen | S34 `cb8063a7…` | calib-selected strong-RGB fixed fusion (equal or tri) − S29 learned system: F1 ≥ .01, CI > 0, both folds, cross ≥ 0 | pass on F1; cross is the risk |
| H55-screen | S35 `37da5f94…` | CCAR on the S33 branch vs the same branch unrendered: from-s8 Δ ≥ .03 (CI > 0) and F1 Δ ≥ −.005 | the most likely of the three to move from-s8; F1 cost ≈ 0 by construction |
| H56-screen | S35 | same after equal fusion | uncertain; HSI's from-s8 errors are not rendered |

S35 was frozen before S33's held-out scoring; S34 before S32's. Each resolves its inputs from sealed
outputs, so no design choice depends on a held-out outcome.

**S32 / S34 outcomes (2026-10-05):**
- **H50-screen: pass** (plan `251465cc…`). Fine-tuned ViT-B TTA − frozen ViT-L probe:
  +.1692 [.1511, .1868], folds +.1739 / +.1644, cross +.1053 [.0549, .1539]. The prediction
  (≥ +.05, cross ≈ 0, attraction rising) was right on F1 and **wrong on transfer**: cross rose
  and session attraction is .479, below HSI's .549.
- **H51-screen: pass.** Equal fusion +.0773 [.0646, .0898], cross +.0619 [.0221, .0999].
  The prediction "well below the RGB gain" was right (.077 vs .169).
- **H54-screen: pass** (plan `cb8063a7…`). Calib chose `equal_trained` (.8511 vs `tri` .8408).
  Versus the S29 learned system: +.0706 [.0556, .0847], folds +.0691 / +.0720, cross +.0473
  [.0008, .0938]. Predicted pass on F1 with cross as the risk; cross passed narrowly.

| ID | Study / plan | Statement and gate | Recorded prediction |
|---|---|---|---|
| H57-screen | S37 `96f31458…` (frozen before S33 scoring) | Trained ViT-B with multi-layer readout + metric morphometrics vs S32 ViT-B: RGB F1 ≥ .01, CI > 0, both folds, cross ≥ 0 | modest pass (+.01–.02); morphometrics should help cross (F117) |
| H58 (descriptive) | S37 | the same contrast after equal fusion with v5 | ≈ half the RGB gain; v5 already holds morphometrics |
| H59-screen | S38 (frozen before scoring) | Role specialization: equal(trained RGB, HSI spectral shape `snvmean214_own`) − equal(trained RGB, v5): cross Δ ≥ .02 (CI > 0) and F1 Δ ≥ −.01 | likely fail: v5 alone transfers better (.209) than spectral shape alone (.150); F1 cost probably > .01 |

**S33 / S35 / S38 outcomes (2026-10-05):**
- **H52-screen: fail** (`2873190a…`). RGB from-s8 −.0307 [−.0539, −.0062], F1 −.0072. Predicted a
  small rise; the sign was wrong.
- **H53-screen: fail.** System from-s8 −.0282 [−.0392, −.0172]. Predicted fail.
- **H55-screen: fail** (`37da5f94…`). RGB from-s8 −.0025 [−.0086, .0037]; F1 +.0001. Predicted the
  likeliest to move from-s8; it did not move.
- **H56-screen: fail.** System from-s8 +.0037 [−.0037, .0110].
- **H59-screen: fail** (`e5ea3571…`). Cross −.0368 [−.0674, −.0061], F1 −.0179. Predicted fail.
- **H57-screen: fail** (`96f31458…`). +.0029 [−.0053, .0108], folds −.0008 / +.0065, cross −.0059.
  Predicted a modest pass. Wrong: training absorbs both frozen-readout gains.
- **H58 (descriptive):** system +.0010 [−.0053, .0075].
