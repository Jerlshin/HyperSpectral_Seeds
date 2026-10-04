# Decisive experiments, compute allocation and publication route

2026-10-03. Planning protocol; no S19 models were trained or scored. Proposed new
hypotheses below are **not preregistered** until exact data/code/arms/thresholds and
hashes are frozen. Existing S17's parent JSON is unchanged.

## 1. What success means

The deployment question is recognition of a new physical rice kernel from an
acquisition not used to learn its variety. Separate narrower and broader estimands:

| Tier | Test unit and interpretation | Status |
|---|---|---|
| A within acquisition | Unseen kernels, familiar scans/sessions; 80/20 with exact seed IDs and calibration carve | Existing stratified machinery; useful comparison tier, not session robustness |
| B new scan | Reverse the two physical bundles, all 90 classes; report both folds | Existing grouped protocol; 73 classes still share sessions |
| C observed session bridges | 17 varieties, 34 directional class-session cells, with destination and class breakdown | Existing diagnostic confirmation subgroup; highly concentrated on session 8 |
| D new balanced session | All target varieties represented across separate training/development/test sessions | Needs new acquisition; strongest missing test |
| E new biological lot/site/year/instrument | Relevant unit held out while class support is maintained | Requires those units and metadata; separate claim from D |

Tier C is not an all-90-class leave-one-session-out benchmark. Dropping a session
can remove entire classes from training. Do not relabel this as closed-set domain
generalisation. For partial-support experiments, declare shared-class subset and
open-set/unknown-class handling; keep them out of the all-90 headline.

## 2. Immediate engineering gates (no GPU training)

1. Verify archive checksum against the dataset record; validate all paired file
   payloads before rebuilding. Stream extraction to avoid duplicate 17-GB archives.
   Build a versioned kernel manifest containing scan, session, variety, grid row/col,
   source bounding boxes/centroids, masks, RGB partner and exclusion reason.
2. Audit one full acquisition grid in every session, all 16 missing HSI components,
   all low-confidence pairs and a random validation sample. Preserve human-reviewed
   overlays, transform residuals and checksums. No zero-confidence automatic pairing.
3. Rebuild full 215 reflectance exactly, compare k32 re-slice against current cube
   within its documented floating-point tolerance and kernel order. Persist standards
   provenance; compare exclusion/train-only handling of 605.583333 nm's three pooled
   white references. Never silently replace historical k32 extraction.
4. Freeze source wavelengths, valid mask, historical 32/64 and new nested32/64/195/215
   axes. Check axis hashes and mask equality; keep training-only feature fitting.
5. Pin v5 with explicit overrides or complete D36 neutral default migration. Retain
   the old defaults/configs needed to replay historical runs. Fix stale narrative
   documentation in a separately traceable engineering change if desired.

These gates resolve data identity, not a preference question. They should be completed
before spending GPU time on multimodal claims.

## 3. Staged, bounded experiment portfolio

| Stage | Minimum comparisons | Question / gate | Compute policy |
|---|---|---|---|
| E0 descriptors | Quantile+morph LDA; regularized linear or RBF-SVM on frozen spectra at 32/64/195/215; optional TabPFN only if feasible | Is there stable full-band information beyond the existing summaries? | CPU where feasible; standardize/select hyperparameters on train/calib only; no broad classifier sweep |
| E1 frozen RGB | One DINOv3 candidate and one modest ConvNeXt; RGB versus silhouette/grayscale/downsample controls | Does high-resolution appearance add signal independent of crude morphology/background? | Cache one set of embeddings per frozen crop/checkpoint; linear heads are cheap |
| E2 cheap fusion | v5 or descriptor HSI + best frozen RGB; equal logit fusion versus calibrated logit fusion versus one shallow feature head | Is there useful complementary error structure? | Freeze list before confirmation; prioritize this over training large multimodal nets |
| E3 physical spectral representation | Shared compact wavelength-aware encoder versus index-only matched encoder; pixel bag versus small spatial grid; 32/nested64/195/215 budget controls | Are geometry and physical sampling useful, and which measurements earn their cost? | Profile first; screen at most a small fixed factorial subset, not every cross-product |
| E4 nuisance mechanism | Winning representation with/without standards-grounded perturbation/correction; generic matched jitter control | Does measured nuisance handling improve transfer without destroying class detail? | Run only if E0–E3/data diagnostics identify a plausible family |
| E5 confirmation | v5, strongest conventional reimplementation, HSI-only, RGB-only, simple fusion and final candidate | Does the proposed method beat the actual alternatives, not just its weakest ablation? | Replicate only finalists across both folds and ≥3 seeds; new session test once after lock |

Prioritize **E1→E2 and full-band E0**, then choose E3 based on evidence. A failure of
frozen features is not a proof that RGB has no information; inspect resolution,
foreground quality and domain mismatch before one bounded fine-tuning trial. If RGB
alone dominates, investigate whether HSI improves difficult classes/session transfer
before paying for fusion. If fuller spectra fail under two sound low-capacity
representations, retain32 for deployment and move compute to complementary geometry.

Before each stage, select at most two finalists on allowed development data and
write a complete arm list, including negative controls. Limit the initial new
trainable screen to roughly **6–10 configurations**, two folds and seed 0; distinguish
configurations from runs. Follow the existing screen/replication semantics. Confirmation
on seeds 1–2 tests stochastic reproducibility; it does not create new held-out data.
Record rejected arms and resource cost, not only winners.

**Useful planning effect sizes**, to be frozen or revised before new confirmation:
Δ macro-F1 ≥ .02 with a paired interval excluding zero; cross-session macro-recall
Δ ≥ .05 with no material same-session drop; for cheaper deployment, macro-F1
noninferiority margin .01 plus latency/memory benefit. These reflect current ~.01
run variation and decision relevance; they are not universal significance thresholds
or already-registered promises. Require session/class breakdown because a mean
improvement concentrated only in session5–8 is a narrower result. Use both G3 and
paired uncertainty; a margin pass alone is not a discovery.

## 4. Relationship to frozen S17/S18

S17 still contains Z1 (v5 80/20, three seeds), Z3 (4×4 tail, two folds seed0), Z2
(native v5 uniform430 k64, two folds seed0), about 4.3 hours on a T4 pair. S18 is
reserved to read it. Complete this bounded round for the existing paper baseline
and open mechanism question; do not expand it into another long SeedNet tuning loop.
E0/E1 asset work can proceed independently.

D37's full 215 continuation conditional on a positive v5 k64 screen is appropriate
for that **native v5 stem route only**. S19 proposes a separate information test of
full spectra regardless of that result, because the stem and selected wavelengths
change with count. Record a separately hashed study or pre-run amendment if execution
priorities change; never edit the S16 parent JSON or treat an unrun screen as negative.
S19's synthesis number intentionally skips reserved S17/S18 and does not report
hypothetical execution outcomes.

S15 cost 265 minutes for ten cells on two T4s; this is **pair wall time**, about
8.8 GPU-hours total, not 4.4 GPU-hours. New RGB/foundation-model throughput is not
measured. Profile peak memory and minutes per epoch/inference cache on the actual
machine; estimate total cost as runs×epochs×measured step time. Do not multiply old
v5 timings blindly by band count or claim a budget for unprofiled foundation models.
Set a stage cap after profiling and stop a route when its decision gate fails.

## 5. New acquisitions: the high-ceiling research investment

A third bundle in another session improves evidence but does not supply two distinct
training environments plus independent development and test sessions. A stronger
minimum target is **four sessions** with all evaluated varieties represented:
two training sessions, one development session and one locked test session. Session
means a genuinely repeated acquisition process (setup/day/recalibration), not a
random split of an image. More training sessions are desirable if affordable; four
is a design minimum, not a guarantee of broad population generalisation.

Cross biological lot and session as far as possible: at least two independently
sourced lots per variety, measured under each session protocol with **different
physical kernels** assigned to prediction partitions. Lot-held-out and session-held-out
are distinct experiments; two lots permit a limited holdout, not a reliable estimate
of all future lots. For stronger lot-generalisation development, obtain ≥3 lots.
Repeated standards or dedicated calibration kernels may be re-imaged across sessions,
but keep them out of classifier train/test identities and explicitly declare any
calibration access. Record genotype/accession provenance, supplier/lot, harvest,
storage, moisture, orientation, sensor settings, white/dark standards and lighting.

If resources require a pilot, first cross ~20 varieties across four sessions,
including a prospectively declared mix of hard/easy varieties and historical transfer
failures. Use it to test a mechanism, report the selection rule, and avoid extrapolating
its accuracy to all 90. Then extend the locked confirmation to all 90 or state the
paper's narrower population. Randomize grid position and acquisition order; avoid
class-specific backgrounds, tray identity or illumination. Acquire RGB and HSI of the
same physical kernels with geometric and radiometric references. Audit batch labels
and an image-only position/background classifier as negative controls.

No fixed kernel count is justified by current between-session variance. Use pilot
variance to simulate confidence/power for a practical effect and allocate additional
**independent scans/sessions/lots** before adding near-duplicate kernels. External
cross-dataset evaluation with differing varieties tests representation transfer or
few-shot adaptation, not zero-shot accuracy on the original90 labels. A second public
seed dataset is valuable if its license, labels and acquisition units support the
claimed protocol.

## 6. Statistics, leakage and reporting

Primary new-study metric: single-view macro-F1, with matched TTA reported separately.
For comparisons to frozen S16/S17 use their exact TTA definitions and also show
single-view values. Include accuracy, macro-recall, same/cross-session recall,
training-session attraction, class confusion, calibration (ECE plus Brier/NLL),
latency/memory and params. For an RGB-only/fusion claim report unimodal baselines
under identical kernels and pretraining access. A model with target-domain unlabeled
adaptation belongs in a separate row from inductive models.

Save predictions, probabilities, kernel IDs and every exclusion. Pair deltas on the
same physical kernels. Bootstrap at variety/scan levels with seeds kept paired;
report session-pair sensitivity and leave-one-bridge-family-out descriptive summaries.
Do not pretend kernels sharing one acquisition are independent biological replicates.
Current sparse/confounded sessions do not support a precise population CI for
unseen-session performance. For new acquisitions, uncertainty should resample the
unit corresponding to the claimed generalisation (session or lot), with class
stratification when supported. State limitations when there are too few clusters.

Learn PCA, scaling, band selection, nuisance statistics and fusion weights within
training/development only. Group augmentations, views and crops of one physical
kernel together. Unsupervised pretraining on confirmation images is transductive
access even without labels. Pin and disclose external-pretraining datasets and
possible overlap. A generic frozen encoder's licensing/availability is an execution
gate, not evidence that it will win.

The old benchmark has supported multiple adaptive research rounds. Label new
old-split results as benchmark development with frozen within-round confirmation.
Use new acquisitions as the final untouched test. Cawley–Talbot [L29] provides the
selection-bias rationale; strict bookkeeping does not erase previous knowledge.
Avoid a universal p-value from ranking correlated folds/seeds. Report effect size,
interval, acquisition support and the full experiment count.

## 7. Paper thesis and claims ledger

Working thesis: **Acquisition-aware evaluation reveals where rice recognition
fails; complementary high-resolution geometry and physically indexed spectra improve
transfer when nuisance variation is measured rather than assumed.** The second half
is an untested claim. If it fails, write the benchmark/diagnostic contribution honestly;
do not decorate v5 with arbitrary modules to retain the thesis.

| Claim | Evidence now | Needed before asserting in a paper |
|---|---|---|
| Patch and scan/session protocols answer different questions | Strong S01–S16, structural S19 audit | Reproducible protocol release; careful qualification of comparisons |
| Lean spatial repair improves the current k32 reference | Replicated grouped/fresh-seed result | Cite S16; do not claim isolated pooling or broad session invariance |
| More bands are useless | **Not supported generally** | Fixed-representation nested-band tests including195/215 and noise controls |
| RGB improves robust variety recognition | Plausible, untested here | Verified pairing, modality ablations, same/cross breakdown, fresh acquisitions |
| Physical wavelength/gap handling is useful | Correct geometry, untested predictive effect | Equal-budget index-versus-physical control and alternate band sets |
| Standards-based correction learns nuisance | Hypothesis only | Independent repeated-standard evidence, class-signal retention, new-session test |
| State of the art on RVHID90 | **Not established** | Reimplemented comparators on identical partitions/metrics/preprocessing; protocol reconstruction for published scores |
| First RGB+HSI rice system / first gated fusion | **False novelty route** | Existing Fabiyi 2020 and Huang 2026 already cover these ideas |
| Genotype recognition independent of environment/lot | **Unsupported** | Verified biological identity and crossed external lots/environments |
| Broad HSI method | Not yet | Additional datasets/sensors with comparable rigorous protocols |

Potential contributions, ordered by defensibility: a versioned acquisition-aware
benchmark and data lineage; a measured diagnosis separating resolution, spectral
coverage and acquisition cues; a compact method with one validated mechanism; an
external acquisition test and resource/negative-result evidence. Novelty is in the
specific scientific result and validated mechanism, not the count of modules.

Reimplement a compact strong conventional baseline, v5, a representative hybrid
3D/2D network, a wavelength-aware transformer/spectral encoder, and the closest
available RGB–HSI fusion method. Do not spend equal compute reproducing every weak
published classifier. Distinguish faithful replication from an approximation when
original code/splits are unavailable. Published .96 precision, .978 accuracy and
our .571 macro-F1 on acquisition-disjoint data are not an ordered leaderboard.

## 8. Journal route and deliverable package

Computers and Electronics in Agriculture is the first **scope-fit candidate** for a
rigorous agricultural sensing study; Information Fusion becomes plausible only if
the fusion mechanism itself is broadly novel and validated beyond one dataset.
TGRS/ISPRS venues need a substantially general HSI/representation contribution and
appropriate broader evidence. These are conditional editorial-fit judgments, not
verified current Q1 classifications or acceptance predictions. Confirm the exact
journal's current JCR/SJR year and subject category through the institution's ranking
source before submission; Q1 depends on ranking system/category and changes over time.

Proposed manuscript order: acquisition and estimands → reproducible data lineage →
mechanism hypothesis and method → matched baselines/band/modality ablations → locked
session/lot evidence → limitations and efficient deployment. Main figures: acquisition
support graph; modality/calibration pipeline; per-session and per-variety effects;
accuracy–compute–band Pareto curve. Supplement: full split IDs, wavelengths, pairing
QC, preregistration/amendments, all negative arms, uncertainty method and licenses.

A high-scoring within-acquisition result alone is a weak basis for the desired
journal ambition. The strongest route is an informative method contribution with
external acquisition evidence. If new acquisition cannot be obtained, constrain the
claims to this benchmark, add a second suitable dataset if possible, and expect a
more limited robustness claim—do not promise a 95% cross-session result.
