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
| F08 | Gradient clipping at 1.0 bound on every step (backbone pre-clip norm 25–50) | S01 | — | E1 | standing — **recurs at clip 5.0 (F36)** |
| F09 | Hard classes {41, 49, 51, 52, 70} did not move under 8 mechanisms aimed at them | S01 | strat | E1 | standing — refined by F42 (70 is a session failure) |
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
| F23 | Every model scores ≈ 0 held-out recall on the 17 cross-session varieties | S06 | held-out | E4 | standing for SNV proxies — **refined by F33**: the network on reflectance reaches 0.15, about what shape alone gives |
| F24 | Cross-session kernels are predicted as a class trained in their own session 71–78 % of the time (chance ≈ 15 %) | S06 | held-out | E3 | standing |
| F25 | Dropping session-informative bands does not restore cross-session recall (H6) and costs ≈ 0.08 macro-F1 (H7) | S06 | held-out | E4 | standing |
| F26 | The session fingerprint is strongest near the lamp peak (~710 nm, F ≈ 60) and below 450 nm | S06 | train | E2 | standing |
| F27 | The in-scene white tile is in 180/180 scans but clips at 608–706 nm in sessions 0–7, leaving 215 measurable bands | S07 | — | E4 | standing |
| F28 | Metal: decomposing Conv3d gives 2.12× per step; Branch-A recompute makes batch 128 fit (15.7×); compile is 2.25× slower | S04 | — | E3 | standing |
| F29 | The 256-band-native re-architecture reproduces the audited 40-band control model bit-exactly | S02 | — | E4 | standing |
| F30 | SpectralSeedNet (u430k32): grouped 0.530 ± 0.009, stratified 0.712 ± 0.033 macro-F1; seed σ 4× larger within-acquisition | S09 | held-out / strat | E4 | standing |
| F31 | 63 % of the grouped macro-recall shortfall is already lost in-distribution; cross-session varieties cap grouped at 0.84 | S09 | held-out vs strat | E3 | standing |
| F32 | The 2.85 M-param network adds only +0.035–0.054 macro-F1 over LDA on mean spectrum + 8 morphometrics | S09 | held-out / strat | E3 | standing |
| F33 | Morphometrics are acquisition-invariant (0.167 grouped vs 0.174 strat) and alone reach cross-session recall 0.124 (network 0.152) | S09 | held-out | E3 | standing — **attribution refined by F63** (the network's 0.15 does not come from the morph scalars) |
| F34 | Within-acquisition, run-level F1 tracks training fit (r = −0.98); across bundles no link is detectable (r = −0.32, n = 6) | S09 | strat / held-out | E3 | **challenged by F60** — a correlation across seeds, not a lever |
| F35 | The network under-fits: clean training accuracy 0.76–0.87; the margin phase drives training loss above ln 90 in 10/12 runs | S09 | train | E4 | standing — **refined by F45, F47** (eval-mode clean fit 0.87–0.95; the loss rise is the margin, not chance) |
| F36 | The backbone gradient is clipped on 100 % of steps at clip 5.0 (pre-clip median 8.9, up to ≈ 44) | S09 | — | E4 | standing — **refined by F53** (≥ 82.5 % of steps per epoch after epoch 30; under Adam a re-weighting, not a step-size cut) |
| F37 | Grouped ≈ 0.005 + 0.73 × stratified macro-F1 across 18 model/input pairs (r = 0.94); the network sits on the line | S09 | held-out vs strat | E3 | standing |
| F38 | The grouped–stratified gap is acquisition coverage, not training-set size (LDA: size ≈ 0.01 of 0.15; 24 test-bundle kernels +0.16) | S09 | held-out | E3 | standing (linear proxy) |
| F39 | Cross-session recall is directional (older → session 8: 0.22–0.25; reverse 0.07–0.08) and carried by 4 varieties | S09 | held-out | E3 | standing |
| F40 | Reflectance moved the session fingerprint from the lamp peak (F ≈ 60) to a broad NIR offset (F ≈ 7–9, 715–1000 nm) | S09 | train | E2 | standing |
| F41 | Errors are systematic: 80 % shared by all seeds; 3-seed vote +0.01; TTA +0.011 mean and negative in 1/6 grouped runs | S09 | held-out | E4 | standing |
| F42 | A hard in-distribution cluster {0, 30, 41, 49, 51, 52} confuses mutually under both protocols (stratified F1 0.17–0.39) | S09 | strat | E3 | standing — refines F09 |
| F43 | Prior 78–96 % results on this dataset are within-acquisition (random kernel splits) and rely on high-resolution RGB morphology | S09 | — | E1 | standing |
| F44 | The clean-label phase (epochs 111–150) gets 3.6 % of the cumulative LR; switching mixup off lifts calib F1 in 12/12 runs at 18 % of peak LR | S10 | train / calib | E4 | standing |
| F45 | Clean fit (D18 definition: eval mode, no augmentation, margin 0) of the selected checkpoints is 0.87–0.95 (mean 0.90 grouped, 0.91 stratified); train–calib gap ≈ 0.19 in every run | S10 | train / calib | E4 | standing — refines F35 |
| F46 | Within acquisition, held-out F1 rises ≈ 1 : 1 with clean training accuracy (r = 0.99, slope 1.01, n = 6); across bundles r = 0.36 | S10 | train vs strat / held-out | E3 | **challenged by F60** — under intervention (X1) the realised slope is 0.18 |
| F47 | The margin phase's loss is the margin, not chance: on the same training kernels plain-cosine CE 0.15–0.39 vs margin-penalised 2.9–6.7; 0–52 % satisfy m = 0.30; LS at s = 32 fixes the cosine gap at 0.21–0.24 | S10 | train | E4 | standing — refines F35 |
| F48 | The spatial tail reduces the kernel to 1 × 1; the last 3 × 3 stride-2 conv sees a 2 × 2 map, so 327,680 parameters (11.5 %) never receive a gradient | S10 | — | E4 | standing |
| F49 | The spectral pathway is in effect SNV(32) + morphometrics: the learned index bank never left uniform; index, continuum, D₁, D₂ carry ≈ 1 % of the descriptor's normalised variance | S10 | train / calib | E4 | standing |
| F50 | Reflectance level reaches the learned layers only through the 6-parameter ECA gate; a ±20–25 % gain flips 8–18 % of calib predictions | S10 | calib | E4 | standing |
| F51 | Eval-time on calib: morphometrics at the train mean cost 0.22 macro-F1; spatial path off → 0.19–0.21; spectral path off → 0.42–0.48 | S10 | calib | E2 | standing — retrained (X2): −0.040 / −0.081 / −0.099 held-out (F63, F64) |
| F52 | Linear probes (train → calib): fused embedding ≈ network (0.70 / 0.72); spatial output 0.54 / 0.56; spectral output 0.42 / 0.44, below its own input (0.62 / 0.59); the aux head fits 0.53–0.81 of training kernels | S10 | train / calib | E2 | standing |
| F53 | The backbone clip-group norm is set by two small modules — the stem's first Conv3d (1,008 params) and the spectral MLP — not the 1.9 M-param tail; the margin triples every main-path gradient; under AdamW the clip re-weights batches, it does not shrink steps | S10 | train | E2 | standing |
| F54 | The applied auxiliary weight is 0.65 → 0.25 (`stage1.aux_loss_weight_*`), not the configured and logged 0.2, in every SpectralSeedNet run | S10 | — | E4 | standing |
| F55 | Weight decay is inert: AdamW's total shrink over a run is 2 × 10⁻⁴ | S10 | — | E4 | standing |
| F56 | EMA leads live by +0.03–0.04 calib F1 under mixup and ≈ 0 after epoch 130; EMA and Adam memory span 25–36 epochs | S10 | calib | E2 | standing |
| F57 | The S11 instrumentation moves no training number (identical per-step losses, 222/222 checkpoint tensors and held-out predictions before/after, 3 regimes); the same gate detects a real change; the clean-fit probe equals S10's offline clean fit on the full training set and is within 0.017 of it on its 1,000-kernel subset | S11 | train (synthetic + S08 checkpoints) | E4 | standing |
| F58 | Before S11 a 2-rank evaluation scored the sampler's padding kernel twice: every grouped S08 run wrote one extra held-out prediction (Δ macro-F1 ≤ 1.7 × 10⁻⁴); the frozen S09 reference (0.530) is already the once-per-kernel value | S11 | held-out (re-scored, no new inference) | E4 | standing |
| F59 | Fit-first (X1) fits 0.98 / 0.99 of training kernels (H13) but moves held-out by +0.001 grouped (CI −0.010…+0.011) and +0.015 stratified (CI −0.012…+0.041): H12a, H12b rejected | S12 | train / held-out | E4 | standing |
| F60 | The extra fit is memorisation: calib accuracy at the selected checkpoint is unchanged (0.714 → 0.714 grouped); F46's slope predicted stratified 0.794, X1 gave 0.727 (realised slope 0.18; grouped 0.01); X1's stratified seed sd is 0.006 (S08 0.033) | S12 | train / calib / held-out | E4 | standing |
| F61 | Capacity is ample — X4 fits 1.000 (H16) — and the softeners are load-bearing: without them −0.090 grouped / −0.174 stratified (TTA; −0.064 / −0.113 no-TTA, fold 0, one seed); TTA lowers an augmentation-free network's score | S12 | train / held-out | E3 | standing |
| F62 | Errors stay systematic under X1: a 3-seed ensemble adds +0.013 grouped / +0.015 stratified; 80 % of grouped errors are shared by all seeds; per-class change vs S08 has sd 0.04 | S12 | held-out | E4 | standing |
| F63 | Morphometric scalars add same-session accuracy (−0.040 F1 without them, CI −0.056…−0.025) but not cross-session recall (0.144 vs 0.152): H14a rejected | S12 | held-out | E4 | standing — refines F33 |
| F64 | The spatial 3-D pathway adds +0.10 F1 (H14b rejected) and is the session channel: spectral-only reaches cross-session recall 0.214 (+0.054, CI +0.025…+0.082) with attraction 0.265 vs 0.47 | S12 | held-out | E4 (F1) / E3 (attraction) | standing |
| F65 | Joint training adds nothing over late fusion of separately trained pathways (F1 +0.006 for fusion, CI 0.000…+0.012); fusion keeps cross-session recall (+0.018) and lowers attraction (0.46 → 0.39), which a 2-seed full ensemble does not; the joint network follows its spatial pathway on cross-session kernels | S12 | held-out (post hoc) | E3 | **challenged** (S14): regime-specific — under R1 fusion has *less* cross-session recall than the joint network (F76–F78) |
| F66 | The spectral-only network behaves as shrinkage LDA on SNV + morph (0.431 / 0.214 / 0.265 vs 0.416 / 0.202 / 0.252); the same/cross-session trade-off follows how much a model trusts low-variance directions | S12 | held-out (linear controls) | E3 | standing |
| F67 | Shrinkage LDA on within-kernel pixel quantiles + morph (104 numbers) matches the network on grouped (0.518 vs 0.519–0.523 no-TTA) and calib (0.70); spread statistics carry the session (attraction 0.44–0.65); network − linear is +0.09 stratified but +0.01 grouped | S12 | calib / held-out | E3 | standing for X1 — the lean network (Y3) is +0.033 grouped (no-TTA) over it (F74) |
| F68 | Detector noise steps +25–30 % at session 5 and is the most session-decodable within-kernel statistic (κ 0.35), but equalising the noise floor leaves the session information in within-kernel spread intact (κ 0.369 → 0.358) | S12 | train / held-out | E4 (step) / E3 | standing |
| F69 | A training-rows class-disjoint session κ ranks held-out attraction across 13 linear representations (ρ 0.93) and separates spectral-only networks (κ 0.13–0.18) from spatial-pathway networks (0.31–0.36; r 0.80, ρ 0.23 within); inside joint networks the spatial output is 2–4× more session-decodable than the spectral output | S12 | train (validated on held-out) | E3 | standing — refined by F81 (no ranking among 16 spatial networks; seed noise ≈ 0.05) |
| F70 | Reflectance level added to a level-blind representation buys +0.02–0.04 F1 and costs 0.02–0.08 cross-session recall and +0.05–0.13 attraction (linear, both LDA estimators) — the H18a ∧ ¬H18b pattern | S12 | held-out (linear controls) | E3 | standing |
| F71 | Infrastructure: a best checkpoint saved at the final epoch races the other rank's reload (non-atomic save, no barrier) → one X2 cell unscored; `dirty: true` on Kaggle is the dataset symlink; clip 50 bound on ≤ 18 group-steps per X1/X4 run | S12 | — | E4 | standing — (a), (b) fixed in S13 (F72) |
| F72 | The S13 code at its defaults moves no training number (G-neutral vs `413a11e`: 3 regimes, 30/30 step losses, 222/222 tensors, held-out ±TTA identical) and the gate detects both new arms (Y2 212, Y3 214 tensors differ); F71a reproduces on `413a11e` as rank 1's `EOFError` and is gone with atomic writes + an end-of-stage barrier — without the barrier rank 1 silently reloads the previous epoch | S13 | synthetic | E4 | standing |
| F73 | The S13 cells are clean: 12/12 scored on `aed5257` with `dirty: false` (the symlink fix works), no regime deviation, every held-out kernel once; the clip guard fires only on single clipped group-steps (≤ 0.15 %); the in-pipeline κ reproduces S12's offline probe (|Δ| ≤ 0.021); the P0.3 re-score was not run | S14 | — | E4 | standing |
| F74 | The lean network (Y3) passes its screen and beats X1 beyond seed spread: grouped 0.562 (+0.026 vs X1 s0, CI +0.018…+0.035), stratified 0.746 (+0.022); above every X1 run in all three cells (p = 1/64 under exchangeability) | S14 | held-out | E3 (screen, 1 seed) | standing — replication frozen (S15, H21a–H21e) |
| F75 | Y3's gain is broad and off S12's same/cross frontier (same +0.026, cross +0.036, attraction −0.038, ECE 0.12–0.15 → 0.10–0.11), grows from calib (+0.013) to held-out, and comes with *more* spatial reliance (influence 62 → 75 %) and a spectral output half as session-decodable; which of its three removals carries it is unknown | S14 | calib / held-out | E3 (screen) | standing — dissection frozen (Y5, H22) |
| F76 | Under R1, decoupled pathways fused on calib keep F1 (0.545, H19a supported) but not robustness (cross 0.134, attraction 0.490; H19b rejected); equal weight gives X1's robustness (0.148 / 0.463), not more | S14 | held-out | E3 (screen) | standing |
| F77 | The regime moves single-pathway networks along the frontier: under R1 spectral-only fits 0.95 (shipped 0.74) and its cross-session recall falls 0.214 → 0.164 (matched −0.037 / −0.031); spatial-only gains +0.056 F1, all same-session, and loses 0.042 cross with attraction 0.40 → 0.53; the joint network does not move (F59) | S14 | train / held-out | E3 | standing |
| F78 | Calib-chosen fusion weights favour the session-carrying pathway (w_spectral 0.30 under R1, ≈ 0.48 shipped); on the held-out grid (post hoc) cross-session recall rises with the spectral weight and only w = 0.75 meets the H19 bars, by ≤ 0.0012 | S14 | calib / held-out (post hoc) | E3 | standing |
| F79 | Masked MixStyle lowers the spatial pathway's usefulness (influence 62 → 43 %, calib F1 −0.03) without lowering its session content (κ 0.31–0.32 unchanged): H20 rejected, no frontier move | S14 | train / held-out | E3 (screen) | standing |
| F80 | The 80/20 within-acquisition tier equals the 70/30 stratified level (0.728 F1 / 0.730 acc vs 0.727; H15 supported): 14 % more same-acquisition training kernels add nothing | S14 | strat | E3 (screen) | standing |
| F81 | The training-rows session κ is reproducible (|Δ| ≤ 0.021) but varies by up to 0.049 between seeds and does not rank held-out attraction among 16 spatial-pathway networks (ρ 0.05) | S14 | train (validated on held-out) | E3 | standing |

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

### F30 · The network's first honest numbers
SpectralSeedNet on `uniform430_k32` reflectance, single stage, calib-selected, TTA. Grouped (2 folds ×
3 seeds): **0.530 ± 0.009** macro-F1 (range 0.518–0.539; no TTA 0.519); stratified (6 seeds, fixed
split): **0.712 ± 0.033** (0.670–0.753). Same-session recall 0.648, cross-session 0.152. Fold means
0.529 / 0.531. **Caveat:** one band budget (k = 32, D15); stratified trains on 1.39× the kernels of
grouped. → [S09](studies/S09_post_sweep_forensics/README.md) · `evidence/S09_post_sweep_forensics/runs.csv`

### F31 · Most of the grouped shortfall exists before any shift
Grouped macro-recall shortfall 0.445 = same-session in-distribution 0.219 + same-session bundle shift
0.067 + cross-session in-distribution 0.064 + cross-session session shift 0.097. "In-distribution" is
1 − stratified recall of the same classes. If the 73 same-session varieties were perfect, grouped
macro-recall would be 0.84. **Caveat:** the stratified arm's extra data makes the shift terms slight
over-estimates (≈ 0.01, F38). → [S09](studies/S09_post_sweep_forensics/README.md) · `decomposition.json`

### F32 · The network barely beats a linear model on its own scalars
LDA (svd, untuned) on the 32-band mean spectrum + the 8 morphometrics — inputs the network also
receives — scores 0.485 grouped / 0.659 stratified against the network's 0.530 / 0.712 (TTA; 0.519 /
0.700 without). LDA beats logistic, RBF-SVM, boosting and an MLP at fixed hyperparameters. The spatial
pathway (79.6 % of parameters) plus the deep head buy ≈ 0.04–0.05. **Caveat:** non-linear tabular
models were not tuned; LDA was not either. → `network_margin.csv`, `c5_tabular.csv`

### F33 · Shape is the acquisition-invariant cue
Morphometrics-only LDA: 0.167 grouped vs 0.174 stratified (no gap); cross-session recall 0.124; per-feature
session F 0.8–1.6. Spectrum-only LDA cross-session recall 0.039 (k32) / 0.053 (215). Logistic regression on spectrum + morphometrics: 0.170. The network's
0.152 cross-session recall therefore cannot be attributed to the spectrum until a morphometrics-zeroed
arm is run (X2). → `c5_tabular.csv`, `c4_session_F_morphometrics.csv`

### F34 · Fit predicts score within the acquisition
Stratified, 6 seeds on one fixed split: held-out F1 vs final training loss r = −0.98 (95 % CI
−1.00 … −0.81), vs clean training accuracy r = 0.96, vs calib F1 r = 0.90. Grouped: r = −0.32 / 0.19 /
0.37, CIs spanning zero. The 0.08 stratified seed spread is an optimisation outcome. **Caveat:** n = 6;
"no link across bundles" is weak evidence. → `fit_link.json`

### F35 · Under-fitting under the shipped regime
At epoch 111 (first epoch without mixup and margin) training accuracy on augmented batches is
0.76–0.87; calib accuracy 0.66–0.73. Ramping the ArcFace margin to 0.3 (s = 32) raises training loss
from ≈ 1.6 to 3.7–6.1; 10 of 12 runs end above ln 90 = 4.50. Calib F1 changes by 0 to +0.03 across
that phase while the LR decays. → `curves.csv`
**Refined (S10).** Measured as D18 defines clean fit, the selected checkpoints fit 0.87–0.95 of their training
kernels (F45); the loss above ln 90 coexists with that accuracy and is the margin penalty at s = 32 (F47), so
"above chance" describes the loss, not the predictions. The under-fit stands; its size is smaller.

### F36 · Clipping binds on every step, again
`grad_norm/clipped_backbone` = 1.0 in every epoch of every run; backbone pre-clip norm median 8.9
against threshold 5.0, ≈ 26 during the margin ramp and ≈ 44 after it. D07 raised the clip from 1.0 to
5.0 so that it would clip outliers only; it still clips everything. Effect size unknown (Adam is
largely invariant to a constant gradient scale). → `curves.csv`
**Refined (S10, F53).** Per-epoch clip fractions after epoch 30 are ≥ 0.825 (1.0 in 55–98 % of epochs), not 1.0
everywhere; the `head` and `embed_net` groups are never clipped. Under AdamW a per-step clip re-weights batches
inside Adam's ≈ 1,000-step memory rather than shrinking the step.

### F37 · The transfer line
Across 18 (model, input) pairs measured on both protocols — the network ± TTA, 8 LDA representations,
8 non-linear tabular controls — grouped = 0.005 + 0.726 × stratified, r = 0.94. The network's residual
is +0.008: no measured model buys acquisition robustness beyond its in-distribution score. Observational,
not an intervention. → `transfer.csv`, `transfer_fit.json`

### F38 · Coverage, not quantity (linear proxy)
LDA at a matched 40 kernels/class: 0.484 stratified vs 0.332 grouped (gap 0.152 vs 0.162 at native
sizes). The grouped curve is flat past 30/class. Swapping 24 training kernels for 24 from the test
bundle (n fixed) lifts the rest of that bundle from 0.336 to 0.497; 80/20 5-fold scores 0.510 (k32).
**Caveat:** demonstrated for LDA only. → `c1_learning_curve.csv`, `c2_acquisition_mix.csv`

### F39 · Cross-session recall has a direction
Trained on the older session, tested on session 8: 0.22–0.25; trained on session 8, tested on the older
one: 0.07–0.08 (folds 0/1). DT66 (78) 0.91; 60, 64, 22 at 0.4–0.55; 7, 12, 38, 45, 79 at 0 in every run.
→ `session_direction.csv`

### F40 · Where the fingerprint went
Per-band session F on 215-band reflectance (S06 method, training rows): median 7–9 across 715–1000 nm,
≈ 0.7 at 430–500 nm. SNV-256 (F26) peaked at ≈ 60 at the lamp maximum and fell below 1 at 860–930 nm.
SNV of reflectance: median 1.3–2.6 in the NIR, peak 10.9 at 801 nm. → `c4_session_F_reflectance.csv`

### F41 · Systematic errors, small test-time gains
Kernels wrong in all 3 seeds: 0.36 of held-out (grouped) against a single-model error of 0.45 — 80 % of
errors are shared (stratified 72 %). 3-seed vote: +0.015 / +0.009 grouped, +0.012 stratified. TTA:
+0.011 grouped, +0.013 stratified; grouped f1_s2 −0.0095 (paired CI −0.016 … −0.002).
→ `kernel_consistency.csv`, `tta_delta.csv`

### F42 · The hard cluster is varietal, not session
NBP (30), TB13 (41), KB16 (49), NBK (51), NPT1 (52) and BC15 (0) are each other's top confusers
in-distribution (stratified F1 0.17–0.39) and remain hard under grouped. Class 70 (F09) is instead a
cross-session failure (stratified 0.46, grouped 0.04). → `confusion_pairs.csv`, `per_class.csv`

### F43 · What the literature numbers measure
Fabiyi et al. 2020 (the dataset authors): random forest on high-resolution RGB shape features + LDA of
256 bands, random 4:1 kernel split, 90-variety average F1 78.27 % (maximum over LDA component counts).
Taheri et al. 2024: 92.73–96.17 % precision with 15 bands + RGB, split protocol not stated. **E1:** read
from the papers, not reproduced. → `literature.csv`

### F44 · The clean objective gets the last 3.6 % of the learning rate
Under the shipped single stage, epochs 1–110 (mixup) receive 96.4 % of the cumulative LR, the margin ramp
(111–130) 2.9 % and the held margin (131–150) 0.6 %. When mixup stops (epoch 111, LR 0.18 × peak), logged
training accuracy jumps 0.30–0.45 → 0.76–0.87 and calib F1 (live) rises in all 12 runs (+0.001…+0.042; ≈ 12× the
0.0016/epoch trend of epochs 90–110); the best calib F1 after epoch 110 exceeds the best before by +0.016…+0.044.
**Caveat:** the step coincides with the margin ramp starting; the margin can only have pulled the other way.
→ [S10](studies/S10_training_architecture_review/README.md) · `schedule_budget.csv`, `transitions.csv`

### F45 · How well the selected checkpoints actually fit
Eval mode, no augmentation, margin 0, on each run's own training rows: accuracy 0.872–0.920 grouped (mean 0.900),
0.872–0.951 stratified (0.911); plain-cosine CE (s = 32, no smoothing) 0.15–0.39; calib accuracy 0.690–0.758;
train − calib 0.17–0.21 in every run. S09's 0.76–0.87 (F35) was train mode on augmented batches at epoch 111.
**Caveat:** these are the calib-selected checkpoints (epochs 121–150), not converged ones. → `ckpt_fit.csv`

### F46 · Fit and held-out move together, one for one, within the acquisition
Stratified (6 seeds, one split): S09's held-out macro-F1 (TTA) against F45's clean training accuracy: r = 0.99
(95 % CI 0.92–1.00), slope 1.01; the train–calib gap does not widen as fit improves. Grouped: r = 0.36 (CI −0.64…0.91).
**Caveat:** observational, over 0.87–0.95 only; no held-out row was re-read (S09's scores were paired with a
train-only measurement). → `ckpt_fit.csv`, S09 `runs.csv`

### F47 · The margin phase in angular terms
On the same clean training kernels: margin-penalised CE (m 0.30, ε 0.04) 2.9–6.7 — reproducing the logged loss
without augmentation or dropout — against plain-cosine CE 0.15–0.39 and 0.87–0.95 accuracy. Median target angle
53–72°, median target-vs-rival cosine gap 0.10–0.27; 0–52 % of kernels satisfy cos(θ_y + 0.30) > max_{j≠y} cos θ_j
(0–7 % in the three runs selected during the ramp). Label smoothing's optimum at s = 32 is a cosine gap of
0.21–0.24 (ε 0.10–0.04); a 0.30 rad margin needs ≈ 0.45–0.48 at θ_y = 45–60°. → `ckpt_fit.csv`,
`objective_geometry.csv`

### F48 · A quarter of the spatial tail's last block cannot learn
Stem spatial strides (1, 2, 2) take 64 → 16; four stride-2 ResBlocks take 16 → 8 → 4 → 2 → 1. The last block's
3 × 3, stride-2, padding-1 conv on a 2 × 2 map uses kernel rows/cols 1–2 only: 5 of 9 taps (327,680 parameters,
11.5 % of 2,849,478) get exactly zero gradient (backward-pass check) and end all 12 runs at init × 0.99980 /
0.99971 (the weight-decay factor). Mean and max pooling coincide at 1 × 1. `docs/03` states (B, 256, 4, 4).
→ `structure.csv`, `ckpt_dead_taps.json`

### F49 · The chemometric half of the spectral pathway is inert
Index bank: softmax entropy 0.9998 of ln 32 (31.98 effective bands), max weight 0.033–0.038 vs uniform 0.031,
cos(π⁺, π⁻) 0.998, |z| ≈ 0.008 with sd 0.001 across kernels. In the shared LayerNorm(184): SNV 79 % of the
variance, morph 20 %, D₁ 0.7 %, D₂ 0.09 %, continuum 0.07 %, index bank 0.08 %. The blocks are not
information-free (shrinkage LDA alone: index 0.20–0.22, continuum 0.06 calib F1; chance 0.011), but they reach
the MLP at ≈ 1 % amplitude; D₁/D₂ are fixed linear maps of SNV. → `ckpt_index_bank.csv`,
`ckpt_spectral_scale.csv`, `descriptor_probe.csv`

### F50 · The network sees reflectance level only through the ECA gate
Bias-free Conv3d → per-sample GroupNorm makes the stem invariant to x → a·x; SNV, D₁/D₂ of SNV, normalised
differences and hull-ratio depths are scale-invariant. Gain after the gate: spatial output ≤ 0.09 %, spectral
descriptor 0.000 %. Gain before it (×0.8 / ×1.25): spatial 6.6–8.4 %, descriptor 1.2–1.9 %, spectral output
12–22 %, 8–18 % of calib predictions change. For a linear model on the same rows, level is worth +0.04 calib F1
(raw 0.35 vs SNV 0.31) and +0.085 grouped held-out (S09 C3). → `level_channel.csv`, `ckpt_gain.csv`

### F51 · What the trained network leans on (eval-time knock-outs)
Calib macro-F1 0.70 / 0.72 (grouped / stratified) → morphometrics at the train mean 0.48 / 0.50; spatial output
zeroed 0.19 / 0.21; spectral output zeroed 0.42 / 0.48. **Caveat:** reliance, not ablation — a network trained
without the input would adapt (X2). → `ckpt_probes.csv`

### F52 · Linear probes of every representation
Shrinkage LDA fit on train, scored on calib (macro-F1, grouped / stratified): network 0.70 / 0.72; fused embedding
0.70 / 0.72; spatial output 0.54 / 0.56; spectral output 0.42 / 0.44; spectral descriptor 0.62 / 0.59 (without
morph 0.41 / 0.36); raw mean spectrum + morph 0.52 / 0.54; raw mean spectrum 0.35 / 0.34; SNV 0.31 / 0.30. The aux
head (spatial path) classifies 0.53–0.81 of training and 0.44–0.65 of calib kernels. → `ckpt_probes.csv`,
`ckpt_fit.csv`

### F53 · Where the gradient norm comes from, and what the clip does
Per-module gradient norms on each run's selected checkpoint (train mode, 4 unaugmented batches of 64, the applied
aux weight): under the clean objective (margin 0, ε 0.056, aux 0.313) the backbone group's norm is 16.6, of which the
stem's first Conv3d (1,008 parameters) contributes 11.2 and the spectral MLP 9.2; every one of the seven tail blocks
contributes ≤ 2.3 and the index bank 0.03. The auxiliary term adds 3.1 (in the spatial path only). Under the margin
objective (m 0.30, ε 0.04, aux 0.25) every main-path norm triples (backbone 49.2, stem stage 1 34.2, spectral MLP 26.0)
while the aux contribution is unchanged — the logged 25 → 45 rise is the margin, not the network. The signed-√ pooling
amplifies little (4 % of pooled activations below 0.01; median derivative 1.22, p99 9–15). Per-epoch clip fractions after
epoch 30 are ≥ 0.825 (1.0 in 55–98 % of epochs); the head and `embed_net` groups are never clipped. Because AdamW
normalises each parameter by its own second moment, a group-wide clip factor changes how batches are weighted within
Adam's ≈ 1,000-step memory, not the step size; its harm is likely small (exact for a constant scale, approximate
otherwise). **Caveat:** CPU, 4 batches per run; not the DDP-averaged gradient the run clipped. → `grad_modules.csv`, `grad_pn.csv`, `dynamics_epochs.csv`

### F54 · The auxiliary weight was never 0.2
`engine/train_epoch.py` computes `aux_w = _aux_loss_weight(cfg, ep, T)` on every call, which reads the
three-stage curriculum's `stage1.aux_loss_weight_init/final` = 0.65 / 0.25: 0.647 at epoch 1, 0.316 at 110, 0.25
from 132 (mean 0.424). `model.aux_head_weight` and `single.aux_loss_weight` (both 0.2) are never read; the banner
and `sched/aux_weight` print 0.2. Every SpectralSeedNet run is affected; D07's "fixed 0.2" was never executed.
→ [S10 B7](studies/S10_training_architecture_review/README.md)

### F55 · Weight decay does nothing at these settings
AdamW multiplies weights by (1 − lr·wd) per step; Σ lr·wd ≈ 2 × 10⁻⁴ over 4,200–6,000 steps. Measured on the
never-trained taps: × 0.99980 (grouped), × 0.99971 (stratified). → `ckpt_dead_taps.json`

### F56 · EMA helps only while the LR is high
EMA − live calib F1: +0.032…+0.041 per run over epochs 11–110, −0.005…+0.008 over 131–150; EMA selected in 5/12
runs. EMA decay 0.999 and Adam β₂ 0.999 both average over ≈ 1,000 steps = 36 epochs grouped, 25 stratified.
→ `transitions.csv`, `horizons.csv`

### F57 · The instrumentation is neutral, and the clean-fit probe measures what S10 measured
Gate G-neutral (S11 §4.1): the same miniature run through the code before (`8050ba2`) and after S11, with every new
probe switched on, in three regimes (shipped mixup → margin, a binding clip, X1-like): every one of 30 step losses
identical (max |Δ| = 0), all 222 live + EMA checkpoint tensors bit-identical, held-out predictions ±TTA identical. The
gate is not blind: `clip_partition=model` under a binding clip changes 213/222 tensors (step-loss Δ up to 1.17) and
`aux_weight_schedule=fixed` 210/222 (Δ up to 1.35) — the first is D22's evidence. On the 12 S08 selected checkpoints
the probe, built exactly as the pipeline builds it, reproduces `ckpt_fit.csv` (S10) on the full training set to
< 10⁻⁶ and differs from it by at most 0.017 (mean 0.008) on the 1,000-kernel subset — binomial sampling
error. **Caveat:** neutrality is shown on CPU with a single RNG stream; CUDA runs are not bit-reproducible anyway
(`cudnn.benchmark`, fused AdamW). → `evidence/S11_frozen_arms_execution/g_neutral.json`,
`clean_fit_reproduction.csv`

### F58 · DDP used to score one held-out kernel twice — the frozen reference did not
`DistributedSampler` pads a split to a multiple of the world size with its own first entries; before S11 the gathered
predictions kept the pad. A 2-rank run on a 91-kernel held-out split wrote 92 predictions (support 92) before S11 and
91 after. All six grouped S08 runs (4,311 / 4,309 held-out kernels) wrote one extra prediction; stratified held-out
(2,588) and calib (630, 906) are even, so checkpoint selection was unaffected. Re-scored without the duplicate, each
run moves by ≤ 1.7 × 10⁻⁴; the grouped TTA mean goes 0.530068 (`run.json`) → 0.530033 — which is the value S09 froze
(its extraction counted each kernel once). The S11 arms, de-duplicated natively, therefore compare with the frozen
reference without correction. → `ddp_dedup.json`, `reference_dedup.csv`

### F59 · Fitting the training set no longer moves held-out
X1 (mixup 30 epochs, no margin, clip 50, 200 epochs) classifies 0.980 (grouped) / 0.992 (stratified) of its fixed
training subset at the selected checkpoint (0.992 / 0.998 at the last epoch; H13 supported) — against 0.90 / 0.91 for
the shipped regime (F45). Held-out: grouped 0.531 vs 0.530 (Δ +0.001, hierarchical-bootstrap CI −0.010…+0.011),
stratified 0.727 vs 0.712 (Δ +0.015, CI −0.012…+0.041); H12a and H12b rejected. **Caveat:** one grouped cell stopped at
epoch 131 (D21 guard 4); without it the grouped mean is 0.533. → `evidence/S12_frozen_arms_reading/hypotheses.json`,
`cells.csv`

### F60 · The extra fit is memorisation; F46 was a correlation, not a lever
Calib accuracy at the selected checkpoint — unseen kernels of the training bundle — is 0.714 for S08 and 0.714 for X1
(stratified 0.720 / 0.723). The S08 seed-level line (slope 1.01, r 0.99) predicted X1 stratified 0.794; X1 delivered
0.727: realised slope 0.18 (grouped 0.01). X1's stratified seed sd is 0.006 against S08's 0.033 — the S08 spread was
optimisation noise, and the seeds that fitted better were the ones that had also learnt more transferable features.
→ `ladder_summary.csv`, `fit_transfer.json`

### F61 · Capacity is ample; the regularisers are what generalise
X4 (no mixup, label smoothing, aux term, dropout or augmentation) fits 1.000 of its training kernels (H16 supported)
and scores −0.090 grouped (CI −0.107…−0.072) and −0.174 stratified below X1 on fold 0 (TTA); without TTA −0.064 /
−0.113, because test-time views hurt a network never trained on augmented inputs. Its calib F1 peaks at epochs
36–46, then declines. **Caveat:** one seed per protocol. → `arm_summary.csv`, `calib_saturation.csv`

### F62 · What remains is bias, not variance
Averaging the logits of three X1 seeds adds +0.013 grouped and +0.015 stratified (TTA; +0.021 / +0.029 no-TTA); 80 % of
grouped errors are made by all three seeds; X1 − S08 per-class recall has sd 0.04 with 4 of 90 classes beyond ±0.10.
Held-out probabilities are over-confident under the bundle shift (ECE 0.15 grouped, 0.04 stratified).
→ `ensembles.csv`, `calibration.csv`, `per_class_x1_vs_s08.csv`

### F63 · The morphometric scalars are not where cross-session recall comes from
Retrained without them, the network loses 0.040 F1 (CI −0.056…−0.025) and keeps its cross-session recall (0.144 vs
0.152 / 0.160 matched; Δ −0.016, CI −0.039…+0.007): H14a rejected. The scalars buy same-session accuracy. → `hypotheses.json`

### F64 · The spatial pathway carries variety information and the session
Spectral-only scores 0.099 below the matched full network (CI 0.087…0.112; H14b rejected under either definition of
"full"). Spatial-only scores 0.081 below. But spectral-only has cross-session recall 0.214 (+0.054 over full, CI
+0.025…+0.082) and attraction 0.265 vs 0.467 — the highest cross-session recall measured on this dataset — and rescues
kernels imaged in sessions 0 and 1 that no full network ever classifies (0.34 / 0.17 vs 0.00).
→ `arm_summary.csv`, `cross_per_class.csv`, `cross_direction.csv`

### F65 · Joint fusion buys nothing late fusion does not, and loses robustness
On the three fold/seed cells where both single-pathway networks were scored, equal-weight log-probability fusion gives
0.533 vs the joint network's 0.527 (+0.006, paired kernel bootstrap CI 0.000…+0.012), cross-session recall +0.018
(+0.008…+0.028) and attraction 0.391 vs 0.464; a calib-chosen weight (≈ 0.48 spectral) gives the same. Ensembling two
full networks gives +0.009 F1, +0.003 cross-session and no attraction change. On cross-session kernels only the
spectral network gets right, the joint network is right 27 % of the time; on those only the spatial network gets right,
66 %. **Caveat:** post hoc on existing predictions (no tuning on held-out), 3 cells, shipped regime — re-tested by Y1.
→ `pathway_fusion.csv`, `pathway_fusion_delta.json`, `pathway_complementarity.csv`, `ensemble_session.csv`
**Challenged (S14):** under R1 the same fusion has *less* cross-session recall than the joint network (−0.015, CI
−0.028…−0.001) and more attraction; the shipped-regime advantage came from under-fitted single pathways (F77) and does
not survive calib-chosen weights (F78).

### F66 · The spectral pathway is a regularised linear discriminant
Spectral-only network: F1 0.431, same 0.501, cross 0.214, attraction 0.265. Shrinkage LDA on SNV + morph: 0.416, 0.483,
0.202, 0.252. Un-shrunk LDA on the same 40 features: 0.449, 0.561, 0.073, 0.480. With one session per class in
training, session differences look discriminative in low-variance directions; shrinkage discounts them.
→ `linear_controls.csv`, `frontier.csv`

### F67 · Within-kernel statistics hold the network's extra power — and the session
Shrinkage LDA, fit on train: mean + morph 0.436 grouped / calib 0.518; + per-band sd 0.507 / 0.660; per-band quantiles
(10/50/90) + morph 0.518 / 0.700; all pixel statistics (824 numbers) 0.513 / 0.735. The network: 0.519–0.523 no-TTA,
calib 0.70–0.72. Every spread family raises attraction (0.44–0.65 vs 0.32) and session κ (0.37–0.45 vs 0.25).
Network − quantile-LDA: +0.09 stratified (0.727 vs 0.639), +0.01 grouped. → `pixel_controls.csv`, `session_probe.csv`
**S14 note:** the lean network (Y3, F74) is 0.551 no-TTA grouped, +0.033 over quantile-LDA — the X1 network's +0.01 is
not a ceiling for architectures.

### F68 · Detector noise is a session fingerprint, but not the one that matters most
Median per-kernel high-frequency residual (3 × 3, bands 8–23): 0.0146–0.0166 in sessions 0–4, 0.0198–0.0206 in
sessions 5–8. As a feature it has the highest class-disjoint session κ of any within-kernel family (0.35) and transfers
nothing (cross 0.039, attraction 0.53). Topping every kernel's noise up to a common floor (99th percentile of training
kernels) drives its κ to 0.005 but leaves mean + sd (0.369 → 0.358) and quantiles (0.378 → 0.373) as session-laden as
before, at a calib cost of ≈ 0.015. → `noise_channel.csv`, `noise_by_session.csv`, `noise_equalisation_pretest.csv`

### F69 · Session reliance can be measured on training rows
Predict the acquisition session from a representation with a classifier fitted on some classes and scored on others
(GroupKFold by class, shrinkage LDA, κ). Across 13 linear representations κ ranks held-out cross-session attraction with
Spearman 0.93 (Pearson 0.88). On 11 fold-0 checkpoints, embeddings of spectral-only networks score κ 0.13–0.18 and every
network with the spatial pathway 0.31–0.36 (Pearson 0.80 with attraction; Spearman 0.23 — no ranking within that
group). Inside the joint networks, spatial-pathway outputs score 0.26–0.32 and spectral-pathway outputs 0.08–0.15.
**Caveat:** session and variety relatedness are partly confounded in the probe; one fold for networks.
→ `session_probe_validation.json`, `embed_probe.csv`
**S14 (F81):** with 16 spatial-pathway networks over both folds the within-group rank correlation is 0.05, and two seeds of
one arm differ by up to 0.049 — a coarse guard only.

### F70 · Level is session information as much as variety information
On the same rows and estimators: SNV + morph → raw + morph (per-band level) and SNV + morph → SNV + log-level + morph
both raise F1 by 0.02–0.04 and lower cross-session recall by 0.02–0.08 with attraction +0.05–0.13. A linear proxy for
X6, not X6. → `linear_controls.csv`

### F71 · Three infrastructure defects behind the S11 cells
(a) `save_ckpt` writes `best_stage1.pth` on rank 0 synchronously and non-atomically; after the last epoch rank 1 reloads
it with no barrier. When the best epoch is the last, rank 1 reads a truncated file (`EOFError`) and rank 0's next
collective pairs with rank 1's error-path barrier (received size `0x3F80000000000000`). `X2/spatial_only__f1_s0` was
never scored. (b) `run.code.dirty` is true on every Kaggle run because the README's `ln -sfn … dataset_u430k32` creates
an untracked symlink that `.gitignore`'s `dataset_*/` does not match. (c) Clip 50 bound on 5–18 of 11,004–24,000
group-steps per X1/X4 run (epoch 1 and fp16-overflow epochs) — D22's trigger fires literally; the effect is
negligible. → `integrity.json`, `cells.csv`

### F72 · The S13 code is neutral at its defaults, and the checkpoint race is fixed — by the barrier, not by atomicity
S11's G-neutral gate, run between `413a11e` and the S13 tree (all telemetry and the κ probe on): identical per-step
losses (30/30, max |Δ| = 0), checkpoint tensors (222/222) and held-out predictions ±TTA in the shipped, clip-binds and
R1-like regimes; S11's two sensitivity controls and two new ones are detected (`model.spatial_mixstyle=true`: 212
tensors, Δloss ≤ 2.21; the Y3 keys: 214 tensors, Δloss ≤ 2.24). The race, as a 2-rank gloo test of `run_single_stage`
with the best epoch last and rank 0's write slowed: on `413a11e` rank 1 fails with `EOFError: Ran out of input` (the
Kaggle failure); on S13 both ranks reload the final epoch; with the barrier removed but atomic writes kept, rank 1
reloads the **previous** epoch's checkpoint without error — atomicity turns a crash into a silent wrong-weights read, the
barrier is what fixes it. Every arm also runs on a real 2-rank `torchrun` job with each held-out kernel scored once.
→ `evidence/S13_representation_screening/g_neutral.json`, `ddp_arms.json`, `tests/unit/test_checkpoint_race.py`

### F73 · The S13 cells are what they claim
All 10 GPU cells and both fused cells were scored on `aed5257` with `dirty: false` — the `.gitignore` fix (F71b) works on
Kaggle. The regime as applied matches R1 plus each arm's intent, stated independently of the runner, with no deviation.
Every held-out kernel was scored once (4,311 / 4,309 grouped, 2,588 stratified, 1,725 for 80/20); attraction re-derived
from saved predictions equals `run.json`. The clip guard fires literally in Y1 spatial-only (10 and 5 epochs) and Y2 f0
(1 epoch), each time one or two clipped group-steps of 84; 0 skipped batches, 4–9 fp16-overflow steps per run; no stop
before epoch 160; `torch.compile` auto-disabled on T4 as in S11. No cell had its best epoch last, so the race fix (F72)
was not exercised. The in-pipeline κ matches S12's offline probe within 0.021. The P0.3 re-score of
`X2/spatial_only__f1_s0` was skipped (S11 output not attached). Training wall clock 274 min (estimate 266).
→ `evidence/S14_screen_reading/integrity.json`, `cells.csv`, `session_kappa_validation.json`

### F74 · The lean network beats X1 beyond its seed spread (screen)
Y3 (descriptor SNV + morph, tail [2,2,2,1], no CBAM on 2 × 2) at seed 0: grouped 0.565 / 0.559 (mean 0.562; H21a's bar
0.511), stratified 0.746 (H21b's bar 0.709). Against the seed-matched X1 cell on the same kernels: +0.029 / +0.024
grouped, +0.022 stratified. Against all X1 runs of the same fold/protocol: above every one in all three cells (X1 f0
{0.535, 0.518, 0.546}, f1 {0.535, 0.526, 0.524}, stratified {0.724, 0.734, 0.724}); under exchangeability p = 1/4 per cell,
1/64 jointly; z +3.1…+3.2 against X1's sd. Y2, X1's network plus MixStyle in the same session, scored at or below X1 —
the environment does not inflate. **Caveat:** one seed; a screening pass (D28) until S15 replicates it.
→ `arm_summary.csv`, `matched_deltas.csv`, `lean_rank.csv`, `hypotheses.json`

### F75 · What the lean network changed
Same-session +0.026, cross-session +0.036 (CI +0.024…+0.050), attraction −0.038 (0.419) against X1 s0: both recalls
rise, which no S12 model did (F66). 7 classes gain > 0.10, 1 loses; cross-session gains sit in sessions 8 and 2 (and 5),
while kernels from sessions 0, 1, 3, 4, 7 stay at ≈ 0. Clean fit 0.999, calib F1 +0.014 / +0.013, held-out +0.029 /
+0.024 — the gain grows away from the training bundle. Errors overlap X1's less than X1 seeds overlap each other
(Jaccard 0.70–0.73 vs 0.75–0.78); held-out ECE 0.098 / 0.111 vs 0.124 / 0.151 (stratified 0.019 vs 0.032). Leave-one-out
influence of the spatial pathway 75 / 74 % (X1 62 / 64 %); spectral-output κ 0.075 / 0.063 (X1 0.133 / 0.131), spatial
0.361 / 0.347 (X1 0.317 / 0.334). Three changes at once: attribution needs the dissection (Y5).
→ `lean_ladder.csv`, `lean_per_class.csv`, `lean_errors.csv`, `lean.json`

### F76 · Decoupled fusion under R1: F1 kept, robustness not bought
Y1 fused at the calib-chosen w = 0.30 (both folds): F1 0.556 / 0.535 (mean 0.545, H19a supported; +0.010 vs X1 s0),
cross-session 0.140 / 0.129 (0.134), attraction 0.464 / 0.517 (0.490): H19b rejected on both parts, every fold on the
wrong side. Equal weight: 0.550 / 0.148 / 0.463. → `hypotheses.json`, `cells.csv`

### F77 · The regime decides where a single pathway sits on the frontier
Seed- and fold-matched, R1 (Y1) vs shipped (X2): spectral-only clean fit 0.961 / 0.944 vs 0.754 / 0.744, F1 +0.006 /
−0.002, cross-session 0.218 → 0.181 and 0.178 → 0.147, attraction 0.29 → 0.31, κ 0.175 → 0.190 (f0); spatial-only (f0;
the f1 shipped cell was never scored) F1 0.438 → 0.494, same 0.542 → 0.613, cross 0.137 → 0.096, attraction 0.40 → 0.53,
κ unchanged. The joint network did not move between regimes (F59). The shipped regime's under-fitting acted like the
shrinkage of F66 on single pathways. **Caveat:** 2 + 1 matched cells. → `pathway_regime.csv`

### F78 · Calib rewards the session-carrying pathway
Calib macro-F1 along the frozen fusion grid peaks at w_spectral = 0.30 in both folds under R1 (shipped cells 0.35–0.55,
mean 0.48), because the R1 spatial network gained on calib (0.60 → 0.67) far more than the spectral one (0.52 → 0.55) and
calib shares the training bundle's session. A held-out grid, computed as a diagnostic that chose nothing: cross-session
recall rises monotonically toward the spectral end and first reaches 0.1666 at w = 0.75 (F1 0.512, attraction 0.390) —
the only grid point meeting all three H19 bars, F1 and cross by ≤ 0.0012. On cross-session kernels only the spectral
network gets right (10–13 %), the fused model is right 24–28 % of the time, the joint X1 network 37 %.
→ `fusion_grid.csv`, `complementarity_r1.csv`, `pathway_regime.json`

### F79 · Masked MixStyle weakens the spatial pathway without removing its session
Y2 vs X1 s0: F1 −0.009, same −0.011, cross −0.006 (CI −0.019…+0.006), attraction +0.029; spatial-output κ 0.313 / 0.321
vs 0.317 / 0.334; spatial influence 42–44 % vs 62–64 %; calib F1 0.662 / 0.701 vs 0.698 / 0.724. Consistent with F68: the
session in within-kernel statistics is low-frequency spatial structure, which per-channel moment mixing does not touch.
→ `cells.csv`, `matched_deltas.csv`

### F80 · The within-acquisition tier
80/20 stratified (Y4, seed 0, X1 architecture, R1): macro-F1 0.728 (kernel CI 0.704…0.743), accuracy 0.730, 1,725
held-out kernels, 5,864 training kernels (70/30: 5,130) — equal to X1's 70/30 stratified 0.727. H15 supported.
→ `hypotheses.json`, `cells.csv`

### F81 · The training-rows κ: reproducible, noisy, blind among spatial networks
In-pipeline vs offline: |Δκ| 0.001–0.021 (Y3 f0, Y1 spectral-only f1). Two seeds of the same arm differ by up to 0.049
(S12 `embed_probe.csv`). Across 21 grouped networks κ vs held-out attraction: Pearson 0.84, Spearman 0.57 (the spectral-only
vs spatial split); among the 16 with a spatial pathway: Spearman 0.05, Pearson 0.04. Fold-1 references added: X1 f1 s0
0.358, X2 spectral-only f1 s0 0.197, X2 spatial-only f1 s0 0.298. Y3 f0 raised κ (+0.036) while lowering attraction
(−0.025) — opposite directions, each within its noise (D26's trigger does not fire).
→ `session_kappa.csv`, `session_kappa_offline.csv`, `session_kappa_validation.json`

