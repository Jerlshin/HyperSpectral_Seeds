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

Defined in CHANGES.md §20 and registered in `experiments/registry.py`. **None has been executed**
(`outputs/` holds no experiment run directories as of 2026-10-01). Each row says which decision it
would confirm or reverse.

| ID | Question | Decision rule | Gates | Status |
|---|---|---|---|---|
| A12 | Run-to-run σ (identical config, 5 seeds) | prerequisite for reading every other delta | all | **not run** — FW-02 |
| A1 | Leakage gap: same model, `stratified` vs `grouped`, 3 seeds each | the gap *is* the Q1 headline | D01 | **not run** — FW-04 |
| A2 | Band selection on all data vs within-fold, grouped | gap > 2σ ⇒ published numbers on this dataset need the caveat | D04, D08 | **not run** (proxy evidence: S03, S05) |
| A3 | {A,B,C,D} vs {B,C} vs {C} vs {B}, symmetric branch dropout | 4-branch − {B,C} < 2σ ⇒ removal confirmed | **D05** | **not run** — FW-07 |
| A4 | Branch A `grid_size_a` ∈ {8, 4, 2} | only if A3 keeps A | D05 | not run |
| A5 | bilinear+gate vs gate-only vs concat+MLP | only if A3 keeps ≥ 3 modalities | D05 | not run |
| A6 | CE vs CE + SupCon, balanced sampler both arms | SupCon earns a Phase B only if > 2σ | D06 | not run |
| A7 | margin 0 / global 0.30 / + per-class / + Ω penalty | one variable per arm | D07 | not run |
| A8 | S1 vs S1+S2 vs S1+S2+S3, grouped, 3 seeds | falsification test for the single stage | **D06** | **not run** — FW-07 |
| A9 | What are classes {41, 49, 51, 52, 70}? (no training: embeddings, overlays, confusion, segmentation audit) | genetic vs segmentation vs acquisition | — | **not run** — FW-06 |
| A10 | Spatial-path width × {0.5, 0.75, 1, 1.5} | is capacity harmful? | D05 | not run |
| A11 | mixup on/off × augmentation profile | is mixup the load-bearing regulariser? | D07 | not run |

## 4 · Adding a hypothesis

Write it here *before* the run, with: the exact claim, the metric and split, the threshold, what
outcome supports it, and which decision or finding it would change. If held-out data will be
touched, also freeze it (WORKFLOW §3). Fill *Outcome* only from saved evidence, and say
`undetermined` when the evidence was not saved — as H4 shows, an unsaved number is a lost test.
