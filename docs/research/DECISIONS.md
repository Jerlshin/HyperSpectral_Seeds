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
| D05 | SpectralSeedNet (two pathways, 3.0 M) replaces SpectralQuadNet (four branches, 5.2 M) as primary | 2026-08-13 | F06, F07 | active — A3 not yet run; S10: 11.5 % of params untrainable (F48), spectral blocks inert (F49); **S12: the spatial pathway adds +0.10 and is the session channel (F64); joint fusion ≈ late fusion (F65); lean version frozen as Y3**; **S14: the lean version passes its screen (+0.026 grouped, F74) — replication frozen (S15); joint fusion stays (F76)** |
| D06 | One training stage replaces three | 2026-08-13 | F05 | active — **under review** (F35, F47: the margin is met by 0–52 % of training kernels at < 4 % of the LR); **S12: margin removed with R1 (D24)** |
| D07 | Objective & optimiser fixes: one aux head at fixed 0.2, GradNorm off, clip 5.0, K = 1, AMP kept on, TF32 off | 2026-08-13 | F07, F08 | active — **under review** (F36, F53); **not executed as written: aux weight was 0.65 → 0.25, not 0.2 (F54)** |
| D08 | Band studies select inside the fold, decide on calib, include null methods and the full budget | 2026-08-13 | F03, F11 | active |
| D09 | Held-out confirmations are pre-registered and frozen by SHA-256 before any held-out row is scored | 2026-09-29 | F21 | active |
| D10 | Bands below 430 nm are excluded from candidate band sets | 2026-09-29 | F14, F19 | active |
| D11 | Finalist band sets are `uniform430` (label-free even spacing), k ∈ {16, 24, 32, 48, 64} | 2026-09-30 | F17, F18 | active — **deviation recorded** |
| D12 | The dataset is white-tile reflectance; the 41 unmeasurable bands are dropped, not filled | 2026-09-30 | F26, F27 | active — effect partly measured (F33, F40); the network sees level only via the ECA gate (F50) |
| D13 | Band indices belong to one wavelength axis; sets are regenerated per axis, never transferred | 2026-09-30 | F27 | active |
| D14 | Every final evaluation reports same- vs cross-session scores, session attraction and entropy | 2026-09-30 | F22–F24 | active |
| D15 | Neural runs go to Kaggle T4 × 2 on a pre-sliced `uniform430_k32` float16 cube | 2026-10-01 | F17, F28 | active — **deviation recorded** |
| D16 | Three reporting tiers: within-acquisition (80/20) · cross-bundle (grouped, primary) · cross-session; 80/20 is never the headline | 2026-10-01 | F31, F38, F43 | active; S12: Y4 runs the 80/20 tier under R1 |
| D17 | Next investment: two diagnostic GPU experiments (fit-first, attribution) before any new architecture or tuning | 2026-10-01 | F32, F34, F35, F37 | active |
| D18 | Instrumentation fixes before the next sweep: logits, git commit, DDP de-duplication, pathway labels, clean train accuracy | 2026-10-01 | F35, F36, S09 §8 | active — **implemented in S11** (D22); extended by D20 |
| D19 | Repair training before architecture: X1, X2, X4 run on the unchanged SpectralSeedNet; component repairs (X5 tail stride, X6 reflectance level) are separate pre-registered arms after X1; no simplification or redesign before X1/X2 report | 2026-10-01 | F44–F50, F54 | active; **S12: X1/X2 reported — architecture work opens under route A (D23); X5/X6 not run standalone (D25)** |
| D20 | Instrumentation scope extended (S10 P0): clean-fit telemetry, applied-aux-weight logging with a `legacy` default, model-declared gradient/clip groups, structural regression tests, docs | 2026-10-01 | F48, F53, F54, S10 B14 | active — **implemented in S11** with one deviation (D22: the model-declared *clip* partition is opt-in) |
| D21 | Interpretation guards on S09's frozen plan (file unchanged): H13's baseline is 0.90/0.91 under its own definition; an X1 gain is credited to mixup/margin/epochs, not the clip; the aux schedule stays `legacy` for X1/X2 | 2026-10-01 | F45, F53, F54 | active — guard 3 enforced by the code default and a test (S11) |
| D22 | S10's P0 implemented as specified except: the model-declared clip partition is opt-in (`clip_partition=legacy` default); the frozen commands run under `torchrun` with one output directory per cell; X5/X6 not implemented (D19) | 2026-10-01 | F57, F58, S10 §6, §9 | active; **S12: reversal trigger fired literally (clip 50 bound on ≤ 18 group-steps per run, F71c) — X1/X4 read as legacy-partition results, effect negligible** |
| D23 | Route A, per the frozen S09 rule (H12a rejected, H14b rejected): no further capacity or regime work; next = representation (session-robust, sample-efficient) and data/protocol | 2026-10-02 | F59–F67 | active |
| D24 | R1 (= X1: mixup 30 epochs, margin 0, clip 50, 200 epochs, patience 40) is the reference regime for new arms; credited with parsimony and lower variance, not with a gain | 2026-10-02 | F59, F60, F47 | **active — under review** (S14, D31): its trigger fired on the cross-session axis (fusion vs joint flips between regimes, F77/F78); F1 ranks unchanged; R1 stays the reference. S13: config-default switch deferred (D29) |
| D25 | X5 and X6 are not run as standalone frozen arms: X5's correctness change joins the lean arm (Y3); X6 is deferred (linear proxy shows the H18a ∧ ¬H18b pattern) | 2026-10-02 | F61, F70 | active — **deviation recorded** |
| D26 | Every run reports a training-rows session κ (embedding and each pathway output) beside the held-out session metrics; it may guard a design choice, never replace held-out confirmation | 2026-10-02 | F69 | active — **implemented in S13** (`evaluation.session_probe`); **S14 (D32): reported only — no ranking power among spatial-pathway networks (F81)** |
| D27 | Before the next Kaggle session: fix the final-epoch checkpoint race, ignore the dataset symlink in the dirty flag, re-score the unscored X2 cell; the next GPU round is S13, frozen in `preregistration_s12.json` | 2026-10-02 | F71 | **implemented in S13** (F72) except the X2 re-score — **not run** (S11 output not attached; decides nothing; F73) |
| D28 | S13 runs as a **single-seed screen**: seeds 3 → 1 (seed 0), every arm, control, protocol contrast and grouped fold kept; recorded as a hashed amendment (`preregistration_s13.json`); outcomes are screening verdicts and passing arms are replicated at seeds 1–2 before any claim | 2026-10-02 | F60, S12 §9 | active — **deviation recorded** (PI-approved, before any S13 run); **S14: read — 1 pass (Y3), 2 rejections (Y1, Y2), H15 supported; no fold straddled a threshold** |
| D29 | P0 and the S13 arms as implemented: atomic checkpoint writes + end-of-stage barriers, `dataset_*`, late-fusion scorer, κ default-on, Y2/Y3 behind default-off keys (G-neutral); D24's default switch deferred | 2026-10-02 | F72 | active — **deviation recorded** |
| D30 | The S13 screen read as frozen: Y3 passes → replicated (S15) and the provisional base of later *screening* arms only; Y1 and Y2 are screening rejections (not replicated); the joint network stays; Y4's 0.728 is the provisional tier-1 number for the X1 architecture | 2026-10-03 | F74–F80 | active |
| D31 | D24's trigger fired on the cross-session axis only; R1 stays the reference regime; every session-robustness claim names its regime | 2026-10-03 | F77, F78 | active |
| D32 | The training-rows κ is reported, never used to select among networks that share the spatial pathway; it guards only against large moves (≥ 0.1) | 2026-10-03 | F81 | active |
| D33 | S15 frozen (`preregistration_s14.json`, `9e182670…`): Y3 at seeds 1–2 (H21a/b as parent; H21c–e on fresh seeds) + a 4-run dissection (Y5, H22a/b); runner-only code change; one Kaggle session | 2026-10-03 | F74, F75 | active |

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
**Annotated (S10).** As trained, the two pathways are not what D05 describes: 327,680 of the spatial path's
parameters never receive a gradient (F48), and the spectral path's chemometric blocks — the index bank and
continuum depths kept "on theoretical grounds" — are inert, leaving SNV + morphometrics (F49). D05 stands; X2 and
X5 test it, and S10 P3.1 repairs the descriptor only if X2 shows the spectral path matters.

### D06 · Single-stage curriculum
**Context.** F05. Stage 2's only distinct ingredient was a non-zero angular margin, which is
incompatible with mixup. **Decision.** One stage: mixup to `single.mixup_epochs`, then a single
global margin warmed in. 69 stage hyperparameters → 14. Stage 2/3 code kept for A8.
**Deviation.** As D05 — adopted before A8 ran. **Reverse if** A8 shows S1+S2(+S3) beats S1 by > 2σ.
**Under review (S09, F35).** The single stage's margin ramp (0 → 0.3, s = 32) lifts training loss from ≈ 1.6 to 3.7–6.1 — above ln 90 in 10/12 runs — while calib F1 moves by 0 to +0.03. X1 (FW-15) removes the margin; if it does not cost calib or held-out F1 beyond 2σ, the margin leaves the default.
**S10 (F47).** On clean training kernels the selected checkpoints are 87–95 % accurate while their margin-penalised
loss is 2.9–6.7: the margin was satisfied by 0–52 % of kernels, introduced at < 4 % of the cumulative LR. A margin
returns only through A7 with ≥ 25 % of the LR budget and no label smoothing (S10 P1.4).

### D07 · Objective and optimiser fixes
GradNorm off and one aux head at fixed weight 0.2 (F07); clip threshold 5.0 so it clips outliers
(F08); sub-centres K = 1 (seeded sub-centres were collinear, cos 0.987); bf16 autocast kept on
through contrastive terms with an fp32 cast only on the similarity matrix; `allow_tf32 = False`.
**Reverse if** an A-series ablation shows any removed mechanism adds > 2σ.
**Under review (S09, F36).** At clip 5.0 the backbone is still clipped on 100 % of steps (pre-clip median 8.9, ≈ 44 late) — the intent "clip outliers" is not realised. X1 raises the threshold to 50 as part of the fit-first arm; a clip-only dissection is run only if X1 moves.
**S10 — the decision was not executed as written.** (i) The "one aux head at fixed 0.2" never ran: the loop applies
`stage1.aux_loss_weight` 0.65 → 0.25 (F54). (ii) Clipping at 5.0 binds, but under AdamW it re-weights batches rather
than shrinking steps (F53); its harm is likely small. (iii) wd 2e-4 is inert (F55). (iv) The clip groups are
SpectralQuadNet's: `fuse` and the aux head are clipped with the backbone. D20 makes all four visible; the applied aux
schedule is deliberately left at `legacy` until X1/X2 have run (D21).

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
**S10 (F50).** The network cannot use reflectance level as a feature: its stem and every spectral feature except
morph are scale-invariant by construction (a design carried over from the SNV cube), and level enters only through the
6-parameter ECA gate. X6 tests giving the spectral path the level, with a session guard (H18b).

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
**Extended by D20 (S10).** **Implemented in S11 (D22):** de-duplicated DDP evaluation everywhere (F58),
`run.json → run.code` (commit + dirty flag) with the environment and the regime as applied, float16 logits for the
reported split and calib ±TTA, `pathway_labels()`, and the clean-fit probe (`fit/*`, `clean_fit.json`), which
reproduces S10's offline measurement (F57).

### D19 · Repair training before architecture
**Context.** F44: the clean-label objective gets 3.6 % of the cumulative LR; F47: most of that is spent on a margin
the network cannot meet. F45/F46: fit is 0.87–0.95 and, within the acquisition, held-out moves with it one for one.
F48–F50: two component defects (a 1 × 1 spatial tail with 327,680 untrainable parameters; a level-blind input
path) and an inert chemometric block — none of which explains an under-fit of ≥ 2.5 M trainable parameters on 3,683
kernels. F54: the regime that ran is not the documented one.
**Decision.** Order of work: S10 P0 (D20) → X1, X2 (S09, frozen) and X4 (fit ceiling, S10) on the unchanged
architecture → under the regime X1 selects, X5 (tail stride) and X6 (reflectance level), each one arm, frozen in
`evidence/S10_training_architecture_review/preregistration_s10.json` → only then simplification (P3.1, P3.6) or
redesign. Every architecture change lands behind a config key whose default reproduces today's model bit-for-bit.
**Alternatives rejected.** Fixing the architecture first (every arm would be measured through a regime that cannot
fit); a wholesale redesign (no evidence that the two-pathway idea fails — only that it is untrained, mis-weighted and
mis-wired in two places); simplifying now (pre-empts H14b).
**Reverse if** X4 cannot fit its training bundle with every softener off (H16 rejected) — then capacity/optimisation
is the limit and P2.1/P3.6 move ahead of regime work.

### D20 · Instrumentation scope, extended
**Context.** D18 plus S10 B14: telemetry that misreports fit (train/acc under mixup has a 0.506 ceiling; under a
margin it is penalised), the aux weight (F54), the clip groups (F53), and non-finite epoch means.
**Decision.** S10 §6 P0.1–P0.7: clean-fit telemetry on a fixed subset (live and EMA); applied-aux-weight logging
with `single.aux_weight_schedule ∈ {legacy, fixed}`, **default `legacy`**; model-declared gradient and clip groups
(`fuse` joins `fusion`; neutral while no clip binds); structural tests for zero-gradient parameters, tail resolution
and gain response (xfail until X5/X6); dry-run composition of every frozen command; docs fixes. Gate: a 2-epoch CPU
run is bit-identical before and after.
**Reverse if** — not reversible; it is measurement.
**Implemented in S11 (D22).** P0.1–P0.4, P0.6, P0.7 as specified; P0.5's per-module gradient groups as specified,
its clip partition as an opt-in (see D22). Gate G-neutral passed: identical per-step losses, checkpoint tensors and
held-out predictions before and after, in three regimes (F57).

### D21 · Interpretation guards on the frozen S09 plan
**Context.** S10 found facts that change how three frozen outcomes read, without making the plan invalid.
**Decision.** `preregistration_next.json` is not modified. (1) H13 is evaluated as frozen (≥ 0.95), but against a
reference of 0.90 / 0.91 measured the same way (F45), and X4 (H16) — not H13 — answers whether capacity limits fit.
(2) An X1 effect is credited to mixup duration, the margin and the extra epochs jointly; clip 50 is near-neutral
under AdamW (F53). (3) The reference ran aux 0.65 → 0.25 (F54); X1 runs the same schedule as a function of
progress (mean 0.425 vs 0.424) only if P0 keeps `legacy` as the default — so it must. (4) An X1 run that early-stops
before epoch 160 is reported with that fact.
**Deviation.** None from the frozen file; these are readings, recorded before X1 runs.
**Reverse if** — superseded by S11's analysis.

### D22 · S10's P0 as implemented — three recorded deviations (S11)
**Context.** D18/D20 specify the instrumentation (S10 §6 P0.1–P0.7) and S10 §9 the checklist that precedes X1, X2
and X4. Implementing it exposed one internal inconsistency in the plan and one gap in the frozen commands.
**Decision.** Everything in P0.1–P0.4, P0.6, P0.7 and §9 steps 4–5 is implemented as written (S11 §4). Three
deviations, each recorded in the code, the configs and the S11 page:
1. **The model-declared clip partition is opt-in.** P0.5 asked for clip groups `head / fusion = fuse + embed_net /
   backbone` and called the change neutral "where no clip binds". But at the shipped `grad_clip=5.0` the backbone group
   binds on ≥ 82 % of steps (F36, F53), so moving `fuse` changes training — contradicting the same plan's gate
   G-neutral (bit-identical before/after) and putting a second variable into X2, which runs the shipped regime and is
   read against the S09 sweep. Measured: in a miniature run with a binding clip, `clip_partition=model` moves 213 of
   222 checkpoint tensors (F57). So `clip_partition ∈ {legacy, model}` exists, defaults to **`legacy`** (every run so
   far), and per-module gradient *telemetry* uses the model-declared groups unconditionally.
2. **The frozen commands run under a launcher, in their own directories.** S09's nine X1 strings read
   `python train.py … runtime=kaggle_t4x2`, which refuses to start without `torchrun` (`multi_gpu: ddp`), and name no
   output directory, so X1 and X2/X4 cells would share `outputs/seednet_full256_f*_s*` and the pipeline's auto-resume
   would re-score one arm's checkpoint as another's result. `scripts/run_frozen.py` reads the overrides **out of the
   hashed files** (refusing to run if a hash moved), adds `torchrun` and one directory per cell, and records both
   additions in every cell's `frozen_cell.json`. For X1 each cell is checked to compose to exactly the config its
   literal frozen command composes.
3. **X5 and X6 are not implemented.** D19 places them after X1 reports, under the regime X1 selects; S10 §9 step 10.
Also, not a deviation but a change readers will meet: the single-stage scalar `sched/aux_weight` (which logged the
configured 0.2 as if applied) is gone; `sched/aux_weight_applied` and `sched/aux_weight_configured` replace it.
**Alternatives rejected.** `clip_partition=model` as the default (changes X2's regime relative to its reference and
fails G-neutral); editing the frozen JSON files to add a launcher and output paths (would move their hashes for an
execution detail that no hypothesis depends on).
**Reverse if** an X1 or X4 run logs `grad_norm/clip_fraction > 0.01` — the partition then mattered at clip 50 and
the run is a `legacy`-partition result, to be read as such (S10 P0.5's rollback condition, unchanged).

### D23 · Route A: representation and data, not capacity or regime (S12)
**Context.** H12a and H12b rejected (F59): fitting 0.98 instead of 0.90 of the training kernels moved no held-out number;
H16 supported (F61): capacity is ample; H14b rejected (F64). S09's frozen rule: *H12a rejected → read X2; H14b rejected →
the representation binds (route A)*. S12 adds why: the extra fit is memorisation (F60), errors are systematic (F62),
the network is at the level of a linear model on kernel statistics across bundles (F67), and the component supplying its
within-acquisition advantage is the session channel (F64, F69).
**Decision.** No further capacity or optimisation-regime arms. The next GPU round (S13, `preregistration_s12.json`) tests
representation changes that address the session channel (Y1 decoupled pathways, Y2 masked MixStyle) and removes what
does not work (Y3), plus the within-acquisition tier (Y4). Data/protocol items — RGB morphology (FW-18), the band budget
(FW-03), transfer standards (FW-29), more acquisitions (FW-12) — are where S12 places the larger headroom.
**Alternatives rejected.** Wider/deeper networks or longer schedules (H16, F60); session-adversarial heads, GroupDRO or
last-layer re-training (they need session to vary within class in training; under `grouped` it never does); redesigning
the network around the 3-D pathway (it is the session channel).
**Reverse if** an S13 arm raises grouped F1 by ≥ 2σ_ref (0.020) through an optimisation change alone, or a capacity
change does — the representation would then not be what binds.

### D24 · R1 is the reference regime for new arms (S12)
**Context.** X1 vs the shipped regime: grouped +0.001 (CI −0.010…+0.011), stratified +0.015 (CI −0.012…+0.041); stratified
seed sd 0.006 vs 0.033 (F60); no margin, so the training loss means what it says (F47); clean fit ≈ 1. Cost: up to 200
epochs instead of 150 (X1 cells stopped at 131–200).
**Decision.** New arms (S13) run under R1 and compare with the X1 cells. The ArcFace margin is dropped from the default
path (the cosine head stays, m = 0); mixup 30 epochs; clip 50 as an fp16 spike guard. The config default changes when
the S13 code lands, behind G-neutral for every other default.
**Deviation.** S10's `regime_rule` says X5/X6 "run under the shipped regime" when H12a is rejected; that rule is kept for
X5/X6 if they are ever run as frozen (D25), but no new arm uses the shipped regime.
**Alternatives rejected.** Keeping the shipped regime (equal held-out, 5.7× the stratified variance, a margin with no
measured benefit); a 150-epoch R1 (untested).
**Reverse if** an S13 arm run under both regimes ranks differently under them.
**S13 note (D29).** The S13 cells carry R1 explicitly, read from the frozen file; the config default itself is not yet
switched (D29 deviation 1).
**S14 note (D31).** The trigger fired on the cross-session axis (fusion vs joint reverses between regimes; F77, F78), not
on F1. Status: active — under review.

### D25 · X5 and X6 are not run as standalone arms (S12) — deviation recorded
**Context.** S10 froze X5 (tail stride 1, H17) and X6 (log-reflectance level block, H18a/b) to run after X1. X4 showed
capacity is ample (F61), so X5's capacity motive is gone; its correctness motive (327,680 untrainable parameters, F48)
remains. For X6, the linear proxy on the same rows shows exactly the pattern X6's guard was designed to catch: +0.02–0.04
F1, −0.02…−0.08 cross-session recall, +0.05–0.13 attraction (F70).
**Decision.** X5's change is part of Y3 (lean architecture, non-inferiority, H21). X6 is deferred until a design needs a
level channel; H17 and H18 stay `not run` (their frozen text is unchanged).
**Alternatives rejected.** Running X6 as frozen (2.7 GPU-pair-hours for a predicted session gain); running X5 alone
(a correctness fix with no expected effect).
**Reverse if** S13 or a later CPU control shows level raising cross-session recall, not lowering it.

### D26 · A training-rows session κ is reported for every run (S12)
**Context.** F69: class-disjoint session decodability on training rows ranks held-out attraction across linear
representations (ρ 0.93) and separates spectral-only from spatial networks (r 0.80), but does not rank among networks
with the spatial pathway (ρ 0.23).
**Decision.** The final report adds the κ of the embedding and of each pathway output (training rows, eval mode). It may
guard a design choice made on calib (e.g. "the candidate must not raise κ by > 0.05"); it never replaces a held-out
confirmation and never selects among close networks.
**Reverse if** a run's κ and its held-out attraction disagree in direction against the reference by more than their noise.
**S14 note (D32).** Not fired: Y3 f0 moved κ +0.036 and attraction −0.025, each within its noise (κ seed noise 0.049). But κ
has no ranking power among spatial-pathway networks (F81) — it is reported, not used to select.

### D27 · Infrastructure before the next Kaggle session (S12)
**Context.** F71: a final-epoch best checkpoint races the reload (one X2 cell unscored); `dirty` is true on every Kaggle
run because of the dataset symlink.
**Decision.** (1) A barrier after the stage and an atomic `best_stage1.pth` write (tmp + `os.replace`), with a 2-rank
regression test where the best epoch is the last; (2) `.gitignore` → `dataset_*`; (3) re-score
`X2/spatial_only__f1_s0` through the pipeline's re-score path (training is finished; only the final evaluation runs);
(4) the late-fusion scorer and the κ report (D26). These are P0 of `preregistration_s12.json`.
**Reverse if** — not applicable (housekeeping, no scientific choice).

### D28 · S13 runs as a single-seed screen (S13) — deviation recorded
**Context.** `preregistration_s12.json` (`88b377c5…`) froze S13 as 30 runs (Y1 12, Y2 6, Y3 9, Y4 3; seeds 0, 1, 2)
≈ 12.5 GPU-pair-hours. Before any S13 code had run on GPU, the PI asked for "a fast screening study using one seed
only … keep all scientifically necessary arms, controls, protocol contrasts and grouped folds, but remove repeated
random seeds; use one fixed seed consistently across S13 … successful arms can be replicated across multiple seeds
later". Run-to-run variance under R1 is already measured on this data and code path (X1: grouped sd 0.0099, stratified
0.0058; F60) — the frozen margins were set from it.
**Decision.** Every S13 cell runs at **seed 0**: 10 GPU runs (Y1 4 + 2 CPU fused cells, Y2 2, Y3 3, Y4 1). Everything
else is as frozen — R1, each arm's change, grouped folds 0/1 and the stratified contrast, the reference (6-run X1 mean),
every margin and threshold, the decision-rule logic, the guards, the held-out rule. The change is recorded as a
**hashed amendment**, `evidence/S13_representation_screening/preregistration_s13.json` (`ef598213…`), that names the
parent's hash;
the parent is not edited, and the runner verifies both hashes. Seed 0 is the first seed of every frozen design, fixed
before any run, and gives each cell an already-scored seed-matched R1 cell (X1 seed 0) for deltas reported beside.
**Consequence for reading.** A 2-run grouped mean has SE ≈ 0.007 (≈ 0.004 at 6 runs): the 0.020 margins sit ≈ 2.9 SE
from an estimate instead of ≈ 5, and gate G3 (Δ > 2σ over folds × seeds) cannot be met. Every outcome is therefore a
**screening verdict**: the frozen rule's *adopt* reads *passes the screen* — the arm is replicated at seeds 1 and 2 under
a new frozen file (FW-35) and may be the provisional base of later screening arms, but is not the reference form,
SeedNet v5 or a paper claim until replicated.
**Alternatives rejected.** Editing the parent file (moves its hash and erases what was frozen); keeping 3 seeds for
some arms (the PI asked for one seed consistently); dropping a fold or the stratified contrast instead (folds and
protocols are design factors, seeds are replication).
**Reverse if** the screen proves uninformative — e.g. an arm's two fold estimates straddle its threshold by more than
the margin — then that arm is replicated before its verdict is read, not after.

### D29 · P0 and the S13 arms as implemented (S13) — two deviations recorded
**Context.** `preregistration_s12.json → p0_before_any_arm` and its arms Y2/Y3 leave some execution details open;
D24 says the config default switches to R1 "when the S13 code lands, behind G-neutral".
**Decision.** As specified: atomic `best_stage{s}.pth` + sidecar writes (tmp + `os.replace`) and a barrier at the end
of the single stage, with a 2-rank regression test (F72); `.gitignore` → `dataset_*`; the late-fusion scorer
(`experiments/fusion.py`, calib-chosen weight, model-shaped `run.json`); the session κ in every single-stage report
(`evaluation.session_probe`, default on — reads only training rows); Y2/Y3 behind default-off keys, gated by
G-neutral against `413a11e` with both arms as detected sensitivity controls. Choices the frozen text leaves open:
MixStyle runs between GELU and the mask multiplication with the area-pooled mask as pixel weights (the same affine map
as after ×α, background held at zero), with detached statistics, a pass-through for a kernel without foreground, and
outside `torch.compile`; CBAM placement is computed from the cube's real patch side (passed by the pipeline) and a
mismatched input is refused; the κ probe uses the selected weights.
**Deviation 1.** D24's switch of the config default to R1 is **deferred**. Every S13 cell carries R1 explicitly from
the frozen file, so no S13 number depends on the default; switching now would re-compose S11's X2 cells (which inherit
the shipped regime from the defaults, including the P0 re-score) and void G-neutral's shipped regime.
**Deviation 2.** The barrier is also placed after each stage of the three-stage pipeline (A8's path), which reloads
checkpoints the same way.
**Reverse if** a later study needs `python train.py` to run R1 by default — switch it then, with S11's X2 cells given
explicit shipped overrides and G-neutral re-run on the R1 regime.

### D30 · The S13 screen, read as frozen (S14)
**Context.** S13 ran all 10 GPU cells and 2 fused cells at seed 0 (F73). Verdicts: H19a supported, H19b rejected,
H20 rejected, H21a and H21b supported, H15 supported — every verdict *clear* against its screen interval except H20's
same-session half, which does not decide it; no arm's folds straddle a threshold (D28's reversal check).
**Decision.** As the frozen rules and D28 say: **Y3 passes the screen** — it is replicated (D33) and may be the base of
later *screening* arms, but is not SeedNet v5, the config default or a paper claim until H21a/H21b hold on 3 seeds.
**Y1** (H19a ∧ ¬H19b) → no adoption, the joint network stays, "decoupling does not buy session robustness under R1".
**Y2** (H20 rejected, not a frontier move) → "instance statistics are not separable into session and variety here".
Neither rejection is replicated. **Y4** (H15 supported) → 0.728 is the provisional within-acquisition tier for the X1
architecture; the paper's tier-1 row waits for the reference architecture (FW-37).
**Alternatives rejected.** Adopting Y3 now on its seed-0 strength (D28 forbids it; one seed); replicating Y1 because
H19a passed (the rule needs H19a ∧ H19b; F78 shows calib cannot find a passing weight).
**Reverse if** S15's H21a or H21b fails → Y3's pass was false; X1's architecture stays.

### D31 · D24 under review: the regime matters for the claim, not for the score (S14)
**Context.** D24's trigger: "an S13 arm run under both regimes ranks differently under them". Y1's pathways were run
under both (X2 shipped, Y1 R1). Fusion vs the joint network: F1 +0.006 (shipped) and +0.010 (R1) — same ranking;
cross-session recall +0.018 (shipped) and −0.015 (R1) — reversed (F77, F78). Y3, the one arm that improved
cross-session recall, did so under R1.
**Decision.** R1 stays the reference regime for new arms (the primary metric ranks the same; changing regime mid-series
would void every matched comparison). D24 becomes *active — under review*. A claim about session robustness names
the regime it holds under; S12's F65 is challenged as regime-specific.
**Alternatives rejected.** Reverting to the shipped regime (higher variance, no F1 benefit, and Y3's gain is an R1
result); running every robustness arm under both regimes (doubles cost for a secondary axis).
**Reverse if** an arm's *primary-metric* ranking against its reference differs between regimes.

### D32 · The κ probe is a report, not a selector (S14)
**Context.** F81: the in-pipeline κ is reproducible, but seed noise reaches 0.049 and among 16 spatial-pathway networks it
does not rank held-out attraction (ρ 0.05); Y3 raised κ on fold 0 while lowering attraction.
**Decision.** D26 stands as reporting. κ is never used to choose among networks that share the spatial pathway, and a
guard built on it uses a threshold ≥ 0.1 (twice the seed noise) — enough to catch a spectral → spatial-scale move
(Δκ ≈ 0.15), not the differences between S13 arms. FW-27's guard is restated accordingly.
**Reverse if** a validation on ≥ 3 seeds per arm shows within-group ranking (ρ ≥ 0.6).

### D33 · S15: replicate Y3 and dissect it, frozen (S14)
**Context.** D28's replication clause for passing arms; F74's gain is larger than the frozen non-inferiority question asked,
and F75 cannot attribute it.
**Decision.** `evidence/S14_screen_reading/preregistration_s14.json` (SHA-256 `9e182670…`), frozen before any S15 cell
or runner exists: Y3 at seeds 1–2 (grouped f0/f1, stratified; 6 runs) and Y5 = Y3's changes split in two
(`desc_only`, `spatial_repair`; grouped f0/f1 at seed 0; 4 runs) — 10 runs ≈ 4.4 h. H21a/H21b are read on seeds 0–2 as the
parent froze them; the new superiority/robustness hypotheses H21c–H21e are read on the **fresh seeds only** (the seed-0
cells motivated them); H22a/H22b attribute the gain only if H21a ∧ H21b. Code: a runner with per-cell seeds; model,
engine, data and config code must be identical to `aed5257` (or G-neutral re-run).
**Alternatives rejected.** Replication alone (the paper would claim a lean network without knowing what makes it better);
dissection at 3 seeds (the attribution question is coarse — ≈ 0.026 to split — and a screen answers "roughly which");
adding the lean 80/20 tier now (it is wasted if Y3 fails).
**Reverse if** — not applicable (a design; its outcomes are read in S16).
