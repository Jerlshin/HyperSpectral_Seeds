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
| H17 | Last spatial-tail block at stride 1 (no untrainable parameters) is non-inferior: grouped ≥ reference − max(2σ_ref, 0.018) | X5 (FW-21) | adopt the stride fix as default | not run — the change joins S13 Y3 (D25) |
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
| H21a | (parent) Y3 grouped F1, 6 runs (seeds 0–2) ≥ 0.5108 | Y3 replication (FW-35) | adopt SeedNet v5 (with H21b): defaults → R1 + lean keys | not run |
| H21b | (parent) Y3 stratified F1, 3 runs ≥ 0.7090 | Y3 replication | as above | not run |
| H21c | Y3 − X1 grouped F1 on seeds 1–2 ≥ +0.020 and its hierarchical-bootstrap CI excludes 0 | Y3 replication | the paper may claim a grouped gain (G3) | not run |
| H21d | Y3 grouped cross-session recall ≥ 0.1666 and attraction ≤ 0.4240 on seeds 1–2 | Y3 replication | the paper may claim cross-session robustness | not run |
| H21e | Y3 − X1 stratified F1 on seeds 1–2 ≥ +0.018 | Y3 replication | within-acquisition gain | not run |
| H22a | `desc_only` (SNV + morph descriptor alone) grouped F1, seed 0 ≥ 0.5508 | Y5 dissection (FW-36) | which removal carries the gain (only if H21a ∧ H21b) | not run |
| H22b | `spatial_repair` (tail [2,2,2,1] + no CBAM on 2 × 2 alone) grouped F1, seed 0 ≥ 0.5508 | Y5 dissection | as above | not run |

## 8 · Adding a hypothesis

Write it here *before* the run, with: the exact claim, the metric and split, the threshold, what
outcome supports it, and which decision or finding it would change. If held-out data will be
touched, also freeze it (WORKFLOW §3). Fill *Outcome* only from saved evidence, and say
`undetermined` when the evidence was not saved — as H4 shows, an unsaved number is a lost test.
