# Next-generation rice recognition: master research plan

Created: 2026-10-03. Owner: research synthesis initiated in Codex.
Status: S19–S21 complete; S22 GPU rebaseline prepared and unrun (2026-10-04). Resume from [RESEARCH_PROGRESS.md](RESEARCH_PROGRESS.md).

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
| P11 | S22: six exact-v5 R1 fits on S21 partitions, then fixed RGB fusion | prepared, source/inputs frozen; GPU unrun; H40 |
| P12 | Conditional bounded adaptation or geometry/measurement study | only after P11; no automatic fusion/extra-band adoption |
| P13 | Crossed session/lot acquisition and locked external evaluation | required for broad transfer claim; current17 bridges all touch session8 |

Use [S20](studies/S20_rgb_pathway/README.md), [S21](studies/S21_complementary_rgb/README.md)
and [S22 runbook](studies/S22_complementary_v5/README.md). S17/S18 stay reserved and
unchanged. CPU class intervals measure variety heterogeneity on reused scans, not
independent-session replication. There are no new GPU results in this phase.
