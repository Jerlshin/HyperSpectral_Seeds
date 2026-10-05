# Next-generation rice recognition: master research plan

Created: 2026-10-03. Owner: research synthesis initiated in Codex.
Status: S19–S29 complete; development screening closed (D52); S30 matched confirmation proposed, not frozen (2026-10-05). Resume from [RESEARCH_PROGRESS.md](RESEARCH_PROGRESS.md).

## Objective and scope

Determine the strongest defensible route to a novel, high-performing rice-variety
recognition system and a publication-quality scientific contribution. Reassess
architecture, modalities, spectral resolution, acquisition robustness and evaluation
together. Do not presume that SeedNet or the current experimental queue is optimal.

The original S19 phase delivered an evidence-backed strategy and testable architecture proposal,
not a claim of new model performance. CPU audits of existing artifacts are allowed;
new GPU training and new held-out model selection were outside that synthesis.
Historical studies remain intact. S17 execution and S18 reading are already reserved by S16; this synthesis is
recorded as S19 and does not silently amend S17's frozen experimental design.

**Execution extension, 2026-10-04:** the user authorized a substantial next phase
using the now-local raw data. S20/S21 complete P7–P10 below, including a separately frozen
CPU predictive screen. This explicitly supersedes the synthesis-only scope for new
work; S19's historical proposal and S17/S18 reservations remain intact.

## Scientific questions

1. What was acquired, calibrated, segmented and retained; which transformations and
   metadata determine what the model can learn?
2. Which historical results are reproducible, which are invalid, and which remain
   specific to an architecture, spectral axis, partition or acquisition regime?
3. What do 32, 64 and all 215 valid bands preserve or discard? Can resolution be
   separated from capacity, regularisation, noise and computation?
4. What information remains unused: reflectance level, spectral shape, spatial
   heterogeneity, fine morphology, RGB texture, repeated views or acquisition standards?
5. What can this acquisition design identify about variety independently of session,
   lot, environment and instrument? Which claims require new acquisitions?
6. What do verified rice/seed and broader learning studies establish under comparable
   evaluation? Which modern mechanisms address observed failure modes?
7. What should replace or extend SeedNet, and what is the smallest decisive experiment
   for each proposed component?
8. What is a credible performance ceiling, compute allocation and paper contribution
   without promising unsupported superiority or journal acceptance?

## Phases, deliverables and gates

| Phase | Work | Persisted deliverable | Decision gate |
|---|---|---|---|
| P0 | Inventory repository, working state, workflow and history | plan, progress log, provenance manifest | preserve existing uncommitted work; locate all studies and artifacts |
| P1 | Read S00–S16, registers, engineering docs, relevant code/configs and outputs | S19 repository synthesis; study/evidence matrix; code-to-claim checks | distinguish established, contradicted, conditional and untested claims |
| P2 | Audit acquisition support, band axes, preprocessing, RGB feasibility and evaluation | reproducible CPU audit, machine-readable tables, figures, limitations | separate estimable performance from unidentifiable generalisation |
| P3 | Search current primary literature across task, representation, robustness and multimodality | search log, verified bibliography/BibTeX, evidence and applicability matrix | source verification, protocol comparability and mechanism-to-failure mapping |
| P4 | Compare competing research routes from first principles | decision matrix, rejected ideas, hypotheses, architecture specification | every component has local evidence, external rationale and a falsification test |
| P5 | Design decisive experiments and publication strategy | staged compute plan, evaluation/statistics plan, paper claim ledger | no unearned test-set claims; replication and acquisition requirements explicit |
| P6 | Integrate, cross-check references/artifacts and provide handoff | updated research index/register additions, final study report, completed progress state | another session can resume without reconstructing reasoning |
| P7 | Verify archive, reconstruct kernel identity and foreground RGB, recover compact full-band HSI | checksums, pairing/exclusion audit, native masks, exact legacy k32 parity, manifests | all 180 scans; ambiguous identity fails closed; visual mask review before scoring |
| P8 | Extract pretrained RGB controls and freeze a bounded CPU comparison | checkpoint/source provenance, exact splits/axes/arms/decision rules and input hashes | train/calib-only fitting; no held-out tuning or invented GPU results |
| P9 | Run RGB, HSI, fusion and historical-v5 fusion controls once | predictions, per-variety/subgroup metrics, paired intervals, transfer/complementarity analysis | choose direction from evidence, including unimodal alternatives |
| P10 | Integrate findings and prepare the next falsifiable modelling step | S20 report, reproducible figures/code/tests/configs, revised queue and handoff | historical files/guards intact; executable resume instructions |

## Evidence discipline

- Follow [WORKFLOW.md](WORKFLOW.md); retain distinctions between SNV-256 and
  reflectance-215, grouped/stratified/within-acquisition, same/cross-session, screen
  and replication. Interpret repeated use of historical held-out data explicitly.
- Trace quantitative claims to saved evidence, not only prose. Recompute inexpensive
  summaries without training or using held-out outcomes to choose new models.
- Record source URL, title, authors, year, venue/identifier, verification date,
  relevant protocol, finding and transfer limitation for literature. Review broad
  architecture families, but spend implementation effort only after a mechanism gate.
- Distinguish measured observations, mathematical consequences under stated
  assumptions, literature-supported hypotheses and speculative possibilities.
- Assess paper novelty against prior art; a bundle of known modules alone is not
  an established novel contribution. Journal quartiles and fit need date/category
  verification when choosing a submission venue.
- Preserve raw historical files. Any revised conclusion is a new dated entry with
  a link to its predecessor and an explicit scope/reversal trigger.

## Final outputs

The S19 folder will contain the integrated research report, repository evidence
review, prior-art review, proposed system, experiment/publication plan and resumable
handoff. Its evidence folder will contain the inventory, reproducible audit code and
tables, literature search/verification record and bibliography. Figures will be
generated from saved evidence under `figures/S19_next_generation_strategy/`.

## Progress

- [x] P0: master plan created; initial inventory found S00–S16 and reserved S17.
- [x] P1: repository and research-history synthesis.
- [x] P2: data/protocol/information audit.
- [x] P3: verified current prior art.
- [x] P4: architecture and research-route decision.
- [x] P5: experiments and paper strategy.
- [x] P6: integration, validation and handoff.

- [x] P7: validated all 8,624 paired identities, native RGB masks/crops and compact full spectra.
- [x] P8: four frozen DINOv2 CPU views and separately frozen bounded screens.
- [x] P9: S20 30 arms × two legacy folds; S21 24 arms × two complementary folds.
- [x] P10: evidence, figures, tests, reports/registers and executable S22 handoff.

## Evidence-led revision after S20/S21 (2026-10-04)

S19's multimodal proposal remains a hypothesis, not an architecture commitment.
RGB appearance is useful, but current coupling controls do not justify learned joint
interactions. Advance equal probability fusion with explicit unimodal controls.
Keep k32 provisionally: extra bands help aggregate fit but fail the CPU transfer gate.
Retain all spectral assets; defer fixed-width nonlinear/geometry studies until their
mechanism and compute gate is separately specified. DINOv2 replaced DINOv3/ConvNeXt
for the tractable initial CPU screen; dense full215 storage was replaced by complete
compact measurements plus an optional dense builder. [D41–D44](DECISIONS.md).

The legacy grouped splitter does not produce exhaustive two-fold coverage. S20's
matched historical comparisons are retained with this correction; S21 freezes an
opt-in repair and holds every row out once. Do not rewrite old studies/guards or
reuse networks trained on different scans. This discovery changes the next experiment
from architectural expansion to an identifiable neural baseline comparison.

| Phase | Next action | Gate/status |
|---|---|---|
| P11 | S22: seed 0 exact-v5 R1 on both S21 partitions, then fixed RGB fusion | complete under amendment02/05; H40-screen pass; transfer unsupported |
| P12 | S23: additive learned residual head over frozen HSI/RGB | complete; H41/practical point gates pass; small uncertain margin over TTA |
| P13 | Crossed session/lot acquisition and locked external evaluation | required for broad transfer claim; current17 bridges all touch session8 |
| P14 | S24: remove each learned correction branch, retaining fixed multimodal anchor | complete; no qualifying simplification; H42 necessity gate fails |
| P15 | S25: selectively test full-head seeds1/2 on both existing encoders | complete; H43 passes; 9.76s for four heads, encoder uncertainty unmeasured |
| P16 | S26: fixed TTA-anchor substitution, calibration gate before test | complete; rejected both calib folds, zero new test scores/fits |
| P17 | S27: train the same small head against equal TTA from the outset | complete; H45 pass (+.0131 over equal TTA) |
| P18 | S28: head seeds 1/2 for the TTA-trained head (pre-declared pass path) | complete; H46 pass, seed SD .0009 |
| P19 | S29: frozen RGB backbone capacity (DINOv2 B/L) at system level | complete; H47 pass, ViT-L adopted (D51); head over fixed ViT-L only +.0067 |
| P20 | S30: matched final confirmation, encoder seeds 0/1/2 × both folds (4 new GPU fits) | proposed, not frozen; needs Kaggle authorization; C2 decides head vs fixed fusion |
| P21 | Crossed-session/lot acquisition pilot and locked external test (= P13) | top scientific priority; only route to the away-from-session-8 bottleneck (F113) |

Use [S20](studies/S20_rgb_pathway/README.md), [S21](studies/S21_complementary_rgb/README.md)
and [S22 runbook](studies/S22_complementary_v5/README.md). S17/S18 stay reserved and
unchanged. CPU class intervals measure variety heterogeneity on reused scans, not
independent-session replication. Corrected-fold results are now recorded in S22/S23;
the next bounded architecture experiment is S27, using the same learned head and its intended TTA anchor.

## Architecture-development compute amendment (2026-10-05)

The user explicitly reaffirmed a screening phase: preserve acquisition directions,
use seed0 on both corrected folds first, and spend extra seeds only on meaningful
finalists. [S22 amendment02](studies/S22_complementary_v5/amendment02.md) preserves
the six-cell parent and supersedes P11's allocation with two initial fits. Local
MPS/fp32 replaces unavailable CUDA/T4 hardware as a declared runtime change; v5
architecture, batch, R1 schedule, calib selection, splits and TTA remain fixed.

At amendment sealing P11 was running, not complete. P12 now means a minimal additive residual head over
frozen HSI/RGB features if fixed fusion passes the recorded consistency gate. No
automatic intermediate-baseline seed expansion. P13 remains required for general
session/lot claims. Final learned architecture and matched HSI/RGB/fixed-fusion
controls eventually receive seeds0/1/2 on both corrected folds; selective mechanism
ablations are confirmed when needed for a claimed contribution. This development
policy does not waive paper validation or rewrite reserved S17/S18.

The local runtime fallback was explicitly superseded by
[S22 amendment05](studies/S22_complementary_v5/amendment05.md) once the configured
private Kaggle CUDA context was located. Both complete screening fits use the
original two-T4/fp16 runtime; the six-epoch MPS attempt is retained as interrupted,
without held-out scoring. Seed allocation and scientific gates are unchanged.

The [confirmation queue](studies/S22_complementary_v5/confirmation_queue.md) specifies
which final systems and mechanism controls should receive additional seeds, and
how shared encoder checkpoints avoid redundant control fits. Deterministic RGB
refits are not seed replication; new acquisitions remain a separate requirement.

## Observed development outcomes (2026-10-05)

P11 completed two full v5 CUDA fits, 189/177 epochs, 47.33 min total bootstrap wall
time. Equal fusion gains .045339/.040033 F1, mean +.042686 [.027554,.057266],
but cross gain +.000912 [−.047802,.049617] is unsupported. Do not expand this
intermediate baseline merely to repeat an established classification effect.

P12 completed one 23,514-parameter head per fold on existing encoders, 6.85 min CPU
wall time. Mean F1 .604312; gain over matched equal-single +.010780
[.002308,.019329], over equal TTA only +.002717 [−.005946,.011384]. Retain it
provisionally. P14 completed four cheap branch heads in5.65min without new encoder
fits. Neither smaller correction passes adoption; strict H42 fails because the full
head's .004767 advantage over HSI-only is below .005. Do not round this into a pass.

P15 selectively repeated only the full head at head seeds1/2 on the same two seed0
encoders (four fits,9.76s). Mean F1 .604054, matched gain +.010522
[.001851,.018941], descriptive seed-delta SD .000279. H43 passes; TTA advantage
+.002459 [−.006634,.011215] remains small and uncertain. This is head sensitivity,
not encoder or fresh-session replication.

P16 substituted TTA probabilities into the fixed trained head. Both calibration
scores decreased; the predeclared gate stopped it without any new test scoring.
H44 is unevaluated. P17 therefore proposes training the same23,514-parameter head
against the TTA anchor from the outset. Profile/cache outer-training-only frozen
TTA inference, seal runtime/input hashes and gates, then seed0 both folds first.
[S27 brief](studies/S27_tta_trained_head/README.md). No new encoder fit or expanded
architecture is planned at this step; additional seeds remain conditional.

Final selected systems and matched controls eventually need independent encoder/head
seeds0/1/2 on both corrected folds and independent crossed acquisitions. Existing
three-head results do not meet that obligation. Ten cheap head fits and two full
HSI fits were completed across S22–S26; S26 used zero fits. No job remains running.


## Direction revision after S27–S29 (2026-10-05)

**What changed and why.** S27 was executed as the brief proposed. Against the prediction
recorded before its outcome, the head trained against the TTA anchor passed its practical gate
(+.0131 over fixed fusion, F109), and S28 showed it is head-seed stable (F110). The pre-declared
pass path was to retain the head and define the smallest confirmation.

A descriptive diagnostic of saved predictions showed that representations, not the fusion form,
limit the system:
- 69% of cross-session kernels are wrong in both modalities;
- the either-modality oracle is .697 against .619 fused;
- RGB was a ViT-S/14 probe.

Before spending GPU seeds, S29 therefore ran the cheapest representation test S19 had planned
and deferred: frozen DINOv2 ViT-B/L. ViT-L improves the system by +.0124 (H47 pass, D51), and
improves RGB transfer for the first time. It also shrinks the head's own margin to +.0067 (F112).
The learned head is now a borderline component whose retention S30 must decide with matched
encoder seeds.

**Why development screening stops here (D52).** Every S22–S29 gain is within-acquisition or
toward session-8 destinations. Away-from-session-8 recall is ≈ .18 for every system and
session attraction ≈ 57% (F113). The remaining learned margins (~.01) are comparable to the
historical encoder σ (.009). More modelling on reused scans would select on noise and cannot
reach the transfer bottleneck. The final selected system needs exactly one matched confirmation
(S30). The paper's transfer claim needs crossed acquisitions (P21).

**Compute this phase:** ≈ 21 min CPU train-row TTA, ≈ 19 min MPS frozen ViT-B/L extraction, and
ten head fits (seconds). Zero encoder fits. No broad sweep: two backbones and one fixed recipe.
