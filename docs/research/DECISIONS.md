# Decision register

Why the code and the protocol are the way they are. Each decision names the evidence it rests on
(finding IDs), the alternatives that were rejected, and — most importantly — **the result that
would reverse it**. A decision without a reversal trigger is an assumption.

**Status values:** `active` · `active — under review` (newer evidence pulls against it) ·
`superseded by Dxx` · `reversed`.

**Deviations are recorded, not hidden.** Where what shipped differs from what a pre-registered
rule or an earlier plan said, the entry says so under *Deviation*.

---

## Summary table

| ID | Decision | Date | Rests on | Status |
|---|---|---|---|---|
| D01 | Leave-one-acquisition-bundle-out (`grouped`, 2 folds) is the primary protocol; `stratified` is a contrast arm | 2026-08-13 | F02 | active — **see D14**: bundles are not the only acquisition unit |
| D02 | Fit and select on a calibration split carved from train; score `val ∪ test` once | 2026-08-13 | F04, F21 | active |
| D03 | Report mean ± range over 2 folds × 3 seeds with bootstrap CIs; never a maximum | 2026-08-13 | F04 | active |
| D04 | The default input is the full acquired cube — no band selection on the primary path | 2026-08-14 | F03, F12 | active — **under review** (F17) |
| D05 | SpectralSeedNet (two pathways, 3.0 M) replaces SpectralQuadNet (four branches, 5.2 M) as primary | 2026-08-13 | F06, F07 | active — A3 not yet run |
| D06 | One training stage replaces three | 2026-08-13 | F05 | active — **under review** (F35: the margin phase pushes loss above chance) |
| D07 | Objective & optimiser fixes: one aux head at fixed 0.2, GradNorm off, clip 5.0, K = 1, AMP kept on, TF32 off | 2026-08-13 | F07, F08 | active — **under review** (F36: clip 5.0 binds on every step) |
| D08 | Band studies select inside the fold, decide on calib, include null methods and the full budget | 2026-08-13 | F03, F11 | active |
| D09 | Held-out confirmations are pre-registered and frozen by SHA-256 before any held-out row is scored | 2026-09-29 | F21 | active |
| D10 | Bands below 430 nm are excluded from candidate band sets | 2026-09-29 | F14, F19 | active |
| D11 | Finalist band sets are `uniform430` (label-free even spacing), k ∈ {16, 24, 32, 48, 64} | 2026-09-30 | F17, F18 | active — **deviation recorded** |
| D12 | The dataset is white-tile reflectance; the 41 unmeasurable bands are dropped, not filled | 2026-09-30 | F26, F27 | active — effect partly measured (F33, F40) |
| D13 | Band indices belong to one wavelength axis; sets are regenerated per axis, never transferred | 2026-09-30 | F27 | active |
| D14 | Every final evaluation reports same- vs cross-session scores, session attraction and entropy | 2026-09-30 | F22–F24 | active |
| D15 | Neural runs go to Kaggle T4 × 2 on a pre-sliced `uniform430_k32` float16 cube | 2026-10-01 | F17, F28 | active — **deviation recorded** |
| D16 | Three reporting tiers: within-acquisition (80/20) · cross-bundle (grouped, primary) · cross-session; 80/20 is never the headline | 2026-10-01 | F31, F38, F43 | active |
| D17 | Next investment: two diagnostic GPU experiments (fit-first, attribution) before any new architecture or tuning | 2026-10-01 | F32, F34, F35, F37 | active |
| D18 | Instrumentation fixes before the next sweep: logits, git commit, DDP de-duplication, pathway labels, clean train accuracy | 2026-10-01 | F35, F36, S09 §8 | active — not yet implemented |

---

## Detail

### D01 · Grouped is the primary protocol
**Context.** F02: the stratified split cannot separate variety from tray recognition.
**Decision.** `configs/data/refl215_grouped.yaml` is the default: per class, one bundle trains,
the other is held out; `split_fold ∈ {0, 1}` swaps them. `assert_protocol_holds` fails a run that
requests `grouped` and does not realise it. `stratified` stays runnable for ablation A1, whose
*gap* is the headline Q1 measurement.
**Alternatives rejected.** Keeping stratified as headline (reproduces the field's error; CHANGES Q11).
**Limits stated up front.** Two folds is the maximum (no third bundle); training sees zero
within-class acquisition variance; val and test are halves of one bundle.
**Reverse if** — never for bundles. **Extended by D14**: F23 shows the *session* leaks even when
bundles do not, so grouped is necessary but not sufficient.

### D02 · Calibration split
**Context.** F04. **Decision.** `data.calib_frac = 0.15` carved from train by group; per-class
margins, CDWS, oversampling weights and checkpoint choice are all fitted on calib.
**Caveat discovered later (F21).** Under grouped, calib comes from the *training bundle*, so it is
~0.15 macro-F1 optimistic. It is fit for *ranking* options, never for quoting performance.
**Reverse if** a third acquisition unit per class becomes available — then calib should be a bundle.

### D03 · Mean ± range, never max
Bootstrap CI (2,000 resamples) on every reported number; a delta whose interval crosses zero is
reported as no effect. **Reverse if** — not reversible; it is a reporting rule.

### D04 · Full cube by default — *under review*
**Context.** F03: neither shipped reduction was chosen by an experiment that could return a
different answer, and one leaked labels. A study about what the spectrum carries should not start
by discarding 84 % of it.
**Decision.** `train.py` reads every band; band selection is an opt-in research pathway
(`docs/07`), and A2 takes the full cube as its reference arm. Commit `2a459ef` (2026-08-14)
switched the primary configs from the 40-band SPA cube to the full 256-band cube.
**Why under review.** F17 (pre-registered, held-out): a spatial-spectral proxy scores *higher* with
24–64 evenly spaced bands than with all 256. F14: the < 430 nm bands are noise.
**Reverse if** S08 shows SpectralSeedNet on `uniform430` at some k in 24–64 is non-inferior
(≥ −0.01 macro-F1, beyond 2σ run-to-run) to the full 215-band reflectance cube under grouped.

### D05 · SpectralSeedNet as the primary model
**Context.** F06 (Branch C carried 87 % of the decision), F07, and the cost data in S01.
**Decision.** Two pathways — a 3-D spectral-spatial stem + 2-D ResNet tail, and a spectral MLP
over the masked mean spectrum with SNV/Savitzky–Golay derivatives, learned indices and continuum
depths — concatenated; one K = 1 ArcFace head; one aux head. 3,003,412 parameters at 215 bands.
SpectralQuadNet is kept intact as the control arm.
**Deviation.** The audit (CHANGES IC-10) said the new model *must not be merged before A3 and A8
report*. It was added and made primary in commit `a4883c1` (2026-08-13) and made 256-band-native
in `2a459ef` ("branch A and branch D drop due to analysis", 2026-08-14), on the strength of the
audit's single-run evidence. A3 has not run.
**Reverse if** A3, run under grouped with *symmetric* branch dropout, shows the four-branch model
beats {B, C} by more than 2σ.

### D06 · Single-stage curriculum
**Context.** F05. Stage 2's only distinct ingredient was a non-zero angular margin, which is
incompatible with mixup. **Decision.** One stage: mixup to `single.mixup_epochs`, then a single
global margin warmed in. 69 stage hyperparameters → 14. Stage 2/3 code kept for A8.
**Deviation.** As D05 — adopted before A8 ran. **Reverse if** A8 shows S1+S2(+S3) beats S1 by > 2σ.
**Under review (S09, F35).** The single stage's margin ramp (0 → 0.3, s = 32) lifts training loss from ≈ 1.6 to 3.7–6.1 — above ln 90 in 10/12 runs — while calib F1 moves by 0 to +0.03. X1 (FW-15) removes the margin; if it does not cost calib or held-out F1 beyond 2σ, the margin leaves the default.

### D07 · Objective and optimiser fixes
GradNorm off and one aux head at fixed weight 0.2 (F07); clip threshold 5.0 so it clips outliers
(F08); sub-centres K = 1 (seeded sub-centres were collinear, cos 0.987); bf16 autocast kept on
through contrastive terms with an fp32 cast only on the similarity matrix; `allow_tf32 = False`.
**Reverse if** an A-series ablation shows any removed mechanism adds > 2σ.
**Under review (S09, F36).** At clip 5.0 the backbone is still clipped on 100 % of steps (pre-clip median 8.9, ≈ 44 late) — the intent "clip outliers" is not realised. X1 raises the threshold to 50 as part of the fit-first arm; a clip-only dissection is run only if X1 moves.

### D08 · Band-study discipline
Selectors see `train` only; budget and method are decided on `calib`; `val ∪ test` is reachable
only through a logged reveal. The full budget is mandatory in every sweep (so an elbow is
falsifiable) and `uniform` and `random` are mandatory nulls (F11 shows why).

### D09 · Pre-registration with a hash
**Decision.** Before held-out rows are scored, the arm list, hypotheses and decision rule are
written to a JSON file and its SHA-256 is stored beside it; the confirmation script refuses to run
if the hash differs. Used twice in S05/S06 (`preregistration.frozen.json` → `5ae90caf…`,
`preregistration2.json` → `2077efc9…`); both still verify. See [`WORKFLOW.md` §3](WORKFLOW.md).

### D10 · The 430 nm floor
F14: pixel SNR < 10, dark clipping, and SNV values there dominated by one brightness statistic.
F19 shows the linear proxy still gains from them on held-out data — but F26 shows that region is
session-informative. A real k-band instrument would not see a "brightness" feature computed over
bands it does not have. **Reverse if** a session-clean evaluation (FW-05) shows < 430 nm bands add
cross-session recall.

### D11 · Finalists are uniform430 — *deviation recorded*
**Pre-registered rule (`decision_rule.json`, frozen 2026-09-29).** Track R1: choose `glw430` if its
within-training-bundle advantage over `uniform430` across k ∈ {24…64} is ≥ 0.01 — it was +0.022, so
**the frozen rule selected glw430**. Budget: smallest k within 1 SE of the CNN calib peak → **k\* = 24**
(`r1_budget.json`).
**What shipped.** `uniform430` at k ∈ {16, 24, 32, 48, 64} (`bandstudy/finalists.py`).
**Justification.** The held-out confirmation then showed glw430's advantage does not transfer to the
CNN (F18: −0.006 at k32, −0.004 at k64) and is ≤ 0.013 for LDA, while uniform430 is label-free,
fold-independent and needs no nested selection. This is a decision informed by held-out data,
which the frozen rule said must not pick an arm (*"No arm is picked by its held-out score"*). It is
recorded here so a reader can weigh it; the honest summary is that the method choice between
glw430 and uniform430 is a tie on held-out data and uniform430 was preferred for simplicity.
**Reverse if** a pre-registered run on the reflectance cube shows a supervised set beats uniform430
by > 0.01 held-out on the network.

### D12 · Reflectance from the in-scene white tile
**Context.** SNV removes a scalar gain per pixel, not the illumination's spectral *shape*; F26 shows
that shape is a session fingerprint, and F23 shows what the fingerprint does. Every scene contains
a Spectralon tile (F27).
**Decision.** `--radiometry tile --tile-drop-unresolved` is the default extraction. Each scene is
divided by its own tile; bands no tile in a session measured are dropped from every scan
(215 kept). Dropping, not filling: a fill across a window no tile measured would be a model of the
lamp, not a measurement. The 256-band SNV cube in `./dataset` was replaced (it is rebuildable with
`--radiometry snv --root …`).
**Not yet known.** Whether this helps — FW-01. **Reverse if** FW-01 shows reflectance does not move
cross-session recall *and* costs same-session performance relative to SNV.
**Partly measured (S09).** Spectrum-only LDA cross-session recall rose from 0.000 (SNV-256, S06) to 0.039 (k32) / 0.053 (215) — above chance, small (F33). The lamp-peak fingerprint is gone; a broad NIR offset remains (F40). The network's 0.15 is not yet attributable to the spectrum (X2). No SNV arm on identical rows has been run, so D12 is neither confirmed nor reversed.

### D13 · Band indices are axis-specific
The 256-axis index 128 and the 215-axis index 128 are different wavelengths. Finalist sets are cut
from `dataset/wavelengths.csv` and must be regenerated whenever the dataset is rebuilt; S03/S05
band files address the old axis only. `band_geometry` checks four descriptions of the axis agree.

### D14 · Session reporting is mandatory
`reporting/session.py` runs inside final evaluation: same- vs cross-session macro-recall, macro-F1,
kernel accuracy; attraction rate beside its chance level; `H(Ŝ | S)` beside oracle and chance.
A headline macro-F1 alone averages two populations that behave completely differently (F23).
**Extended (S09).** Cross-session recall must also be *attributed*: morphometrics alone reach 0.124 (F33), so a cross-session number is reported beside a shape-only control and, for networks, a morphometrics-zeroed arm.

### D15 · Kaggle T4 × 2 with a pre-sliced 32-band cube — *deviation recorded*
**Decision.** `runtime=kaggle_t4x2` (DDP + SyncBN, fp16 + GradScaler since Turing has no bf16 Tensor
Cores) reading `./dataset_u430k32` — `uniform430_k32` as float16, 2.33 GB instead of 30.4 GB, with
the cast error recorded in `band_axis.json`.
**Deviation.** The frozen budget rule gave k\* = 24 (D11); the pre-sliced cube is k = 32. Both are
within 1 SE of the calib peak, and k = 32 had the higher held-out CNN score (0.450 vs no held-out
k = 24 arm), but **the reason for 32 is not written down anywhere in the repo**. FW-09 asks for it to
be recorded, or for k = 24 to be run alongside.

### D16 · Three reporting tiers; 80/20 is a tier, not the headline
**Context.** The user's question: literature reports 92 %+ with larger training shares — should 80/20
be the main protocol? F38: on this data the grouped–stratified gap is acquisition *coverage*, not
training-set size (LDA: size ≈ 0.01 of 0.15; 59 → 80 % train ≈ +0.02). An 80/20 split places ≈ 38
kernels of every test bundle in training, and 24 already add +0.16 at equal n (C2). F43: the published
numbers are within-acquisition and rely on high-resolution RGB morphology.
**Decision.** Every results table carries three tiers, with the protocol as a column:
1. **within-acquisition** — patch-level 80/20 (`data.split_eval_frac=0.2`, calib carved from the 80 %,
   ≥ 3 seeds). Supports: *"given the same acquisition, the representation separates 90 varieties to X"*;
   used for literature comparability and as the capacity diagnostic.
2. **cross-bundle** — grouped, 2 folds × ≥ 3 seeds. **Primary**: the only tier that supports *"identifies
   the variety of a kernel from an unseen acquisition bundle"*.
3. **cross-session** — recall on the 17 cross-session varieties, with a shape-only control (D14).
   Supports claims about session invariance, with wide intervals.
The existing 59/11/30 stratified arm is kept only as A1's contrast; it is confounded with training size
and matches neither the literature nor grouped.
**Alternatives rejected.** 80/20 as primary — the score would rise by ≈ +0.18 for reasons the paper's
own evidence attributes to acquisition recognition (F02, F38). Dropping 80/20 entirely — loses the only
comparison with prior work and the tier where the model's measured headroom lives (F34).
**Reverse if** a third acquisition per class becomes available (FW-12): then a bundle-disjoint split with
two training acquisitions replaces tier 1 as the comparability number.

### D17 · Diagnose before designing
**Context.** F32: the network adds ≈ 0.05 over LDA on its own scalars. F34, F35: within-acquisition it is
fit-limited and under-fits. F37: in-distribution gains have carried ≈ 0.73× to grouped across all models.
F33: its cross-session recall is matched by shape alone. A novel architecture designed now would be
measured through the same under-fitting regime and against the same thin margin.
**Decision.** Spend the next ≈ 8 GPU-pair-hours on two pre-registered diagnostics, in order: **X1
fit-first** (FW-15: does better fitting raise stratified, and does it transfer to grouped?) and **X2
attribution** (FW-16: what do the spatial pathway, the spectral pathway and the morphometrics each
contribute, and which carry cross-session recall?). No hyperparameter sweeps, no new branches, no 80/20
headline until both report.
**Routing.** X1 positive and transferring → invest in training/architecture (B). X1 positive in-distribution
but not on grouped → invest in data/representation (A: FW-12, FW-18). X1 null → the regime is not the
limit; X2 decides whether capacity sits in the wrong pathway.
**Reverse if** X1 and X2 both return within 2σ of the current model — then the representation, not the
learner, is binding, and work moves to inputs (band budget FW-03, RGB morphology FW-18).

### D18 · Instrumentation before the next sweep
**Context.** S09 §8: the reported metrics include one DDP-padded duplicate (Δ < 1e-4); `run.json` has no
git commit; no logits are saved (probability calibration and margin analysis impossible); SpectralSeedNet
lacks `pathway_labels()` so influence is logged as A–D; the only "train accuracy" is under mixup or
margin, so the fit that F34 depends on had to be read off one epoch (111).
**Decision.** Before X1: de-duplicate eval rows before scoring; write `git rev-parse HEAD` into `run.json`;
save held-out and calib logits (float16); add `pathway_labels() → ("SPATIAL", "SPECTRAL")`; log a clean
training accuracy (eval-mode, no augmentation, no margin) on a fixed 1,000-kernel training subset every
diagnostics interval. None of these changes a metric.
