# Findings register

Every claim the project makes, in one place. A study page *explains* a finding; this register
is what other pages *cite*. Strength levels (E0–E4) are defined in [`README.md` §4](README.md#4--conventions).

**Status values:** `standing` (current, no contrary evidence) · `challenged` (newer evidence
pulls against it — see note) · `superseded by Fxx` · `retracted`.

**Split shorthand:** `strat` = patch-level stratified (leaky) · `calib` = carved from the
training bundle · `held-out` = the other acquisition bundle (`val ∪ test`), scored once.

---

## Summary table

| ID | Finding (one line) | Study | Split | Strength | Status |
|---|---|---|---|---|---|
| F01 | Pre-refactor model reported 87.8 % test accuracy, 89.4 % with 12-view TTA | S00 | strat | E0 | superseded by F02–F04 |
| F02 | The stratified split puts all 180 acquisition bundles in both train and eval | S01 | strat | E4 | standing |
| F03 | The shipped 40-band set was selected with test labels in scope; both shipped elbows are vacuous | S01 | — | E4 | standing |
| F04 | The 0.847 headline was a max over ~944 selections on the split that also fitted 4 mechanisms (bias ≈ +0.015 – 0.042) | S01 | strat | E1 | standing |
| F05 | Curriculum stages 2–3 used 65 % of wall clock for +0.005 macro-F1 | S01 | strat | E1 | standing |
| F06 | The 4-branch fusion collapsed onto Branch C (87 % influence); Branch A was 60 % of FLOPs for 5.6 % | S01 | strat | E1 | standing (confounded) |
| F07 | The Stage-1 loss was dominated 7.8 : 1 by auxiliary heads discarded at evaluation | S01 | — | E1 | standing |
| F08 | Gradient clipping at 1.0 bound on every step (backbone pre-clip norm 25–50) | S01 | — | E1 | standing |
| F09 | Hard classes {41, 49, 51, 52, 70} did not move under 8 mechanisms aimed at them | S01 | strat | E1 | standing |
| F10 | All Stage-2/3 telemetry was silently dropped by a W&B step collision | S01 | — | E4 | standing (fixed) |
| F11 | On mean-spectrum proxies, even spacing is the best band-selection method at k ≤ 40 | S03 | calib | E2 | standing; confirmed held-out by F18 |
| F12 | Mean-spectrum proxy curves plateau at 192–224 bands | S03 | calib | E2 | **challenged** by F17 |
| F13 | Supervised selectors pick different bands depending on which bundle is held out | S03 | calib | E2 | standing |
| F14 | Below ~430 nm pixel SNR < 10, and those bands' SNV values are 85–91 % one global statistic | S05 | train | E2 | standing |
| F15 | The spectrum is highly redundant: 6 PCs hold 99 % of variance; ~7 directions hold 90 % of discriminant power | S05 | train | E2 | standing |
| F16 | Inside the training bundle, greedy-LDA selection (glw) leads even spacing; mRMR and SPA trail it | S05 | train | E2 | standing; **did not transfer** (F18) |
| F17 | A spatial-spectral CNN scores higher with 24–64 evenly spaced bands than with the full cube | S05 | calib + held-out | E4 | standing |
| F18 | No supervised selector beats evenly spaced ≥ 430 nm bands on held-out bundles by > 0.01 (CNN) | S05 | held-out | E4 | standing (LDA: one cell at +0.013) |
| F19 | For the linear proxy, < 430 nm bands (+0.05) and brightness statistics (+0.05) still add held-out score | S05 | held-out | E4 | standing — interpretation: session-linked (F26) |
| F20 | Low-pass DCT features do not beat the best selected band set of equal size (H5 rejected) | S05 | held-out | E4 | standing |
| F21 | Honest held-out level ≈ 0.42 (LDA, full cube) and 0.44–0.46 (CNN); calib overstates held-out by ≈ 0.15 | S05 | calib vs held-out | E4 | standing |
| F22 | 73 of 90 varieties had both bundles imaged in one session; 17 span two | S06 | — | E4 | standing |
| F23 | Every model scores ≈ 0 held-out recall on the 17 cross-session varieties | S06 | held-out | E4 | standing — **the project's central finding** |
| F24 | Cross-session kernels are predicted as a class trained in their own session 71–78 % of the time (chance ≈ 15 %) | S06 | held-out | E3 | standing |
| F25 | Dropping session-informative bands does not restore cross-session recall (H6) and costs ≈ 0.08 macro-F1 (H7) | S06 | held-out | E4 | standing |
| F26 | The session fingerprint is strongest near the lamp peak (~710 nm, F ≈ 60) and below 450 nm | S06 | train | E2 | standing |
| F27 | The in-scene white tile is in 180/180 scans but clips at 608–706 nm in sessions 0–7, leaving 215 measurable bands | S07 | — | E4 | standing |
| F28 | Metal: decomposing Conv3d gives 2.12× per step; Branch-A recompute makes batch 128 fit (15.7×); compile is 2.25× slower | S04 | — | E3 | standing |
| F29 | The 256-band-native re-architecture reproduces the audited 40-band control model bit-exactly | S02 | — | E4 | standing |

---

## Detail

Each entry: **claim → evidence → caveats**. Figures and raw files are linked from the study page.

### F01 · Pre-refactor reported accuracy — *claim only*
87.8 % test accuracy, 89.4 % with 12-view TTA, mean per-class F1 0.89, three classes (49, 52, 41)
below 0.6. **Evidence:** `figures/spectralquadnet_results.png`, `figures/result_figure_90_classes.png`
(May 2026). No run directory, prediction file or config in the repository reproduces it. Same
leaky split and leaky band selection as F02/F03. A source comment also quotes "86.9 % TTA" for a
256-band run with no artifact. → [S00](studies/S00_pre_refactor_baseline/README.md)

### F02 · Every acquisition bundle leaks under `stratified`
Each variety is two class-pure trays of 48 kernels. A patch-level split put all 180 trays in both
train and eval (run log: `180 of 180 scans are in train and in val/test`). A model that recognises
a tray recognises its class. **Structural**, verified against `groups.npy` by `SplitReport`.
→ [S01](studies/S01_independent_audit/README.md)

### F03 · Band selection saw the test labels; neither elbow is demonstrable
mRMR relevance was `mutual_info_classif` over all 8,624 patches before any split. Both shipped
elbow files record `"demonstrable": false` — each curve ends at its own chosen k.
→ [S01](studies/S01_independent_audit/README.md)

### F04 · The headline number was a selected maximum
`calib_frac = 0`: val fitted per-class margins, the confusion penalty, CDWS and oversampling
weights, then selected the checkpoint, then reported. With epoch-to-epoch sd ≈ 0.012 over ~944
selection events, `E[max] − mean ≈ σ√(2 ln n) ≈ 0.042` (≈ 0.015 allowing for correlation).
**Caveat:** σ read from a log not in the repo. → [S01](studies/S01_independent_audit/README.md)

### F05 · Stages 2–3 bought nothing measurable
Best val macro-F1 0.842 → 0.844 → 0.847 for 6.6 h → +2.7 h → +9.5 h. +0.005 is 6.5 of 1,294
samples, inside a ±0.020 CI and below F04's bias. Stage 2's best epoch (19) preceded its own key
mechanism (per-class margin, from epoch 21). **n = 1 run.** → [S01](studies/S01_independent_audit/README.md)

### F06 · Fusion collapsed to the spatial branch
End influence A 5.6 / B 3.5 / C 87.4 / D 3.1 %; FLOP share A 60.1 / C 34.2 / D 5.7 %. **Confounded:**
Branch C was the only branch never dropped (drop p = 0.15 for A, B, D), so the gate was taught to
rely on it. Ablation A3 would separate the two readings; it has not run.
→ [S01](studies/S01_independent_audit/README.md)

### F07 · The objective optimised heads that are thrown away
GradNorm drove aux weights to their clip bounds (0.25 / 4.0); aux : main ≈ 7.8 : 1 at epoch 20,
so the evaluated head carried ≈ 11 % of the gradient. Explains the rising early loss curve.
→ [S01](studies/S01_independent_audit/README.md)

### F08 · Clipping turned Adam into fixed-step normalised descent
Per-group clip 1.0 against backbone pre-clip norms of 25–50 on every step: the LR schedule's
shape was largely decorative. → [S01](studies/S01_independent_audit/README.md)

### F09 · The hard classes are a stable structure
{41, 49, 51, 52, 70} were bottom-5 from Stage-1 epoch 46 to Stage-2 best, unmoved by oversampling,
CDWS, signed margins, confusion penalty, sub-centres, SupCon, ProtoNCE and focal loss. Low
precision *and* recall → mutual confusion. Whether this is genetic similarity or segmentation
(ablation A9) is **open**. Observation added 2026-10-01: class 70 is one of the 17 cross-session
varieties (F22); 41, 49, 51, 52 are not. → [S01](studies/S01_independent_audit/README.md)

### F10 · Stage-2/3 telemetry did not exist
Each stage restarted the W&B step at 1, below the running max of 336; ~200 warnings, every
scalar discarded. Fixed in S02 (monotone global step). → [S01](studies/S01_independent_audit/README.md)

### F11 · Even spacing is the method to beat
12 methods × 20 budgets × 3 proxies × 2 folds, decided on calib: `uniform` has the best mean
macro-F1 at k ≤ 40 (0.211); fdr, mi, pca_loading, pls_vip, spa, tree_importance and variance
average *below a random subset*. **Caveat:** mean spectra discard all spatial structure.
→ [S03](studies/S03_band_study_proxy/README.md)

### F12 · Proxy plateau at 192–224 bands — *challenged*
Per-proxy plateaus: extratrees 192, lda 224, linsvc 224 (calib). The study itself warned that a
proxy plateau is a *lower* bound for the network. F17 shows the opposite direction for a
spatial-spectral proxy: it peaks at 24–64 and *declines* toward the full cube.
→ [S03](studies/S03_band_study_proxy/README.md)

### F13 · Selected bands depend on the held-out bundle
34 (method, budget) pairs at k ≤ 40 have cross-fold Jaccard < 0.4. Supervised selection inherits
the acquisition confound one level up. → [S03](studies/S03_band_study_proxy/README.md)

### F14 · The blue end is noise
Median pixel SNR (spectral second-difference noise) crosses 10 at ~430 nm and falls to ~1.4 at
383 nm; dark-clipped pixels; SNV values there are 85–91 % explained by the per-kernel `−μ/sd`
statistic, i.e. by brightness, not spectral shape. → [S05](studies/S05_band_research/README.md)

### F15 · Few independent spectral directions
Median adjacent-band r = 0.9985; correlation length 3 bands at r 0.99, 15 at r 0.9; 6 principal
components for 99 % of variance (108 for 99.99 %); Fisher discriminant eigen-spectrum needs 7
directions for 90 %, 15 for 95 %, ~50 for 99 %. Explains why even spacing is hard to beat.
→ [S05](studies/S05_band_research/README.md)

### F16 · glw wins inside the bundle — and only there
Within-training-bundle 5-fold CV (nested selection): glw − uniform = +0.084 at k = 4, +0.027 at
k = 32; glw430 − uniform430 averaged +0.022 over k ∈ {24…64}. mRMR −0.08 and SPA −0.15 below
uniform at k = 32. On held-out bundles the glw430 advantage vanished for the CNN (F18).
→ [S05](studies/S05_band_research/README.md)

### F17 · Fewer bands score higher for a spatial-spectral model
Pre-registered CNN proxy (2 folds × 2 seeds): held-out macro-F1 uniform k32 0.455, k64 0.458 vs
full 256 0.437; uniform430 k48 0.453. Calib shows the same shape (peak 0.602 at k24, 0.549 at
k256). **Caveat:** a 2-D CNN on a 16 × 16 pooled cube, not SpectralSeedNet — S08 tests the
network. → [S05](studies/S05_band_research/README.md)

### F18 · Supervised selection does not transfer (H3)
Held-out at k ≥ 32, glw430 − uniform430: CNN −0.006 (k32), −0.004 (k64); LDA +0.007, **+0.013**,
+0.005, +0.005, −0.004 (k32…128). Repository selectors (mRMR k40 0.332, SPA k40 0.282) sit at or
below uniform. → [S05](studies/S05_band_research/README.md)

### F19 · The suspect features still score — for the linear model (H1, H4)
LDA held-out: uniform − uniform430 = +0.038…+0.059 (k16–64); glw − glw430 = +0.046…+0.071. Adding
log μ, log sd and −μ/sd to uniform430 (gstat3) adds +0.045…+0.062. For the CNN the < 430 nm
advantage is +0.004 on average (H1 holds there). Given F14 and F26, these gains are better read as
brightness/session information than as variety chemistry. → [S05](studies/S05_band_research/README.md)

### F20 · Spectral low-pass is not a shortcut (H5)
LDA held-out: DCT m48 0.391 / m64 0.394 vs glw k48 0.408 / k64 0.408. DCT does beat uniform and
uniform430 at equal size. → [S05](studies/S05_band_research/README.md)

### F21 · The honest level, and how optimistic calib is
Held-out macro-F1, scored once: LDA full 256 = 0.419 (fold 0/1 bootstrap CIs ≈ 0.40–0.43); CNN
0.437 (k256) to 0.458 (k64). The same CNN on calib: 0.549–0.602. **Calib overstates held-out by
≈ 0.15** because calib is a patch-level carve-out of the *training* bundle. Calib may rank options;
it must never be quoted as performance. → [S05](studies/S05_band_research/README.md)

### F22 · The session structure
`scan_table.csv`: 9 sessions (2017-01-11 → 2017-02-03), 5–25 varieties each. 73 varieties have
both bundles in one session; 17 (labels 4, 7, 12, 22, 38, 45, 59, 60, 61, 63, 64, 65, 66, 70, 71,
78, 79) span two — and **every one of the 17 has one bundle in session 8**, the only session where
the white tile never saturates (F27). Under `grouped`, a same-session variety's held-out bundle
shares lamp, detector state and room conditions with its training bundle. → [S06](studies/S06_session_confound/README.md)

### F23 · Cross-session recall is zero — *central finding*
Held-out macro-recall on cross-session varieties: LDA every arm 0.000–0.013 (full cube 0.000);
CNN 0.002–0.006. Same-session varieties: 0.41–0.60. Every held-out point either proxy earns comes
from classes whose two bundles share a session. So the grouped protocol, while honest about
*bundles*, still largely measures *session* recognition. → [S06](studies/S06_session_confound/README.md)

### F24 · Predictions collapse onto the acquisition session
74–78 % (LDA) and 71 % (CNN) of cross-session kernels are predicted as a variety whose training
bundle came from the kernel's own session, against ≈ 15 % for a random prediction. CNN session
prediction entropy 1.24 bits vs oracle 0.66. **Strength E3:** figures quoted in
`reporting/session.py`; the computation script's output was not persisted.
→ [S06](studies/S06_session_confound/README.md)

### F25 · The fingerprint is not in a removable set of bands (H6, H7)
Second pre-registration: keep only bands with session F < 2.2 on the fold's training rows.
Cross-session recall stayed 0.0006–0.0012 (H6 rejected). Macro-F1 fell ≈ 0.08 against
unconstrained arms of equal k (H7 supported) — but because cross-session recall did not move,
that loss cannot be read as "the session share" as H7 intended; it is variety information lost
with the bands. → [S06](studies/S06_session_confound/README.md)

### F26 · Where the session lives in the spectrum
Per-band F-ratio (between-session variance of class means over within-session), training rows,
mean of 2 folds: ≈ 60 at ~710 nm, ≈ 10 at ~775 nm, ≈ 6 at ~440 nm; below 1 at 470–570 and
860–930 nm. Gain statistics: log μ 13.4, log sd 17.4, −μ/sd 3.3. The ~710 nm peak sits at the
lamp maximum — the same region where the white tile saturates (F27). → [S06](studies/S06_session_confound/README.md)

### F27 · What the white tile can and cannot measure
Found in 180/180 scans (full-width strip, ~26,000 core pixels, CV ≤ 0.07). In sessions 0–7 the lamp
peak clips the tile at 4,095 DN in the same bands every scan, so 147 of 180 scans have bands that
no tile in their session measured — 41 bands at the union (608.0–705.8 nm). Session 8 never clips. Result: 8,624 × 215 × 64 ×
64 reflectance, seed reflectance 0.10 at 432 nm → 0.57 at 999 nm. → [S07](studies/S07_reflectance_calibration/README.md)

### F28 · Engineering measurements
M5 / Metal at batch 32: decomposed Conv3d 2,103 → 994 ms per step; Branch-A recompute 4,054 →
1,901 MB activations (+4.8 % time); at batch 128 recompute is 49.6 vs 780.1 ms/sample (paging);
inductor 983 vs eager 437 ms/forward. Host cost 1.86 ms/sample, 1.41 ms of it mmap page-in.
→ [S04](studies/S04_compute_engineering/README.md)

### F29 · The re-architecture is non-regressive on the control arm
`scripts/capture_golden.py --verify`: `v3/logits match (max |Δ| = 0.000e+00)`, 306 init tensor
digests match. 918 tests pass; the 2 failures are pre-existing environment drift in Stage-1 golden
loss digests. → [S02](studies/S02_architecture_protocol_revision/README.md)
