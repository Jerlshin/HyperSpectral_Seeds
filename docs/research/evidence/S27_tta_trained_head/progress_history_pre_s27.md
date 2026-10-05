# Research progress and exact resume state

Updated 2026-10-05. **S20–S26 complete; no experiment is running. S27 is proposed, not frozen or executed.**
Read the [master plan](MASTER_RESEARCH_PLAN.md), [S22 results](studies/S22_complementary_v5/results.md),
and [next S27 design brief](studies/S27_tta_trained_head/README.md).

## Actual allocation and explicit amendments

S22 ran exactly **seed 0 on both corrected acquisition folds**, two complete v5
fits rather than six. The immutable six-cell parent remains unchanged.
[Amendment02](studies/S22_complementary_v5/amendment02.md) records the screening
allocation before execution;03 declares the workerless local attempt;05 restores
the original two-T4 DDP/fp16 training runtime. R1, k32, 2,725,700-parameter v5,
batch128, maximum200 epochs, patience40, calib-only live/EMA selection and TTA
remain fixed. Final inference runs fp32; saved logits are serialized float16.
Both corrected folds hold all8,624 kernels out exactly once. Each mixes acquisition
directions: toward session8 counts6/11, away counts11/6; together they cover both
directions of all17 bridges. Do not label a whole fold a pure direction.

Final S22 plan `configs/research/s22_screening_amendment05.json`, SHA256
`1dd50dae26f356ac963b4713ae5820a5476ede2129f8c554178a09abc7ea7f95`.
The separate analysis-code05 receipt seals the scoring source without changing gates.
The new private [CUDA screen](https://www.kaggle.com/code/jgfreak/s22-corrected-fold-single-seed-development-screen),
version1, completed and was collected. Bootstrap wall time2839.98s (47.33min),
not a billing or GPU-hour estimate. Fold0/1 completed189/177 epochs and selected
LIVE epochs149/137. All ten attached dataset files and uploaded source hashes were
checked; credentials stayed local and are absent from the bundle.

The earlier loader failure at epoch0 and interrupted six-epoch MPS run contribute
no held-out scores. The stop request observed epoch5; asynchronous work finished
6 before termination, documented in the interruption receipt. All attempts remain
preserved. No additional S22 encoder seeds were run.

## Results and what they mean

| Corrected-fold predictor | Fold0 F1 | Fold1 F1 | Mean F1 | Mean cross recall |
|---|---:|---:|---:|---:|
| S21 frozen RGB | .444608 | .456433 | .450520 | .142046 |
| S21 linear HSI32 | .525388 | .544311 | .534850 | .077941 |
| S21 equal linear HSI/RGB | .588051 | .595589 | .591820 | .127858 |
| S22 neural v5 TTA | .557860 | .559958 | .558909 | .208578 |
| S22 equal v5-TTA/RGB | .603199 | .599991 | .601595 | .209490 |
| S23 additive head, encoder/head seed0 | .607767 | .600857 | .604312 | .233252 |
| S25 same head, average head seeds0/1/2 on encoder0 | .606458 | .601649 | .604054 | .235081 |

S22 fusion gains +.045339/+.040033 F1, mean +.042686 with paired-variety95%CI
[.027554,.057266]. H40-screen passes. Cross delta is only +.000912
[−.047802,.049617], with positive/negative fold deltas. Neither acquisition
direction supports a neural-fusion transfer improvement. RGB-only correct cases
(11–12% of all kernels) show useful error complementarity. Fixed fusion both rescues
and harms errors. The stronger neural HSI already recovers much of the transfer
information absent from linear HSI. S20's positive three-seed legacy gains motivated
the lean screen, but do not estimate corrected-fold initialization uncertainty.

S23 implemented and executed a **23,514-parameter additive residual head** over
frozen256-D HSI and384-D RGB features: independent32-D GELU branches, shared
readout initialized to zero, log equal-single-view probability anchor. Train-only
scales/gradients; calib-only selection with epoch0 eligible; fixed recipe, no sweep.
Two CPU head cells, including feature extraction, took410.86s. Matched F1 gain
+.010780 [.002308,.019329]; stronger TTA gain +.002717 [−.005946,.011384].
H41/practical point gates pass, but stronger-reference superiority and transfer
improvement are unconfirmed. CPU predictions exactly match original CUDAfp32
single-view predictions; stored float16 argmax ties are separately audited.

S24 completed four branch-removal heads in338.94s using the same frozen encoders.
The HSI-only/RGB-only learned corrections retain both modalities in the fixed
anchor. Mean F1 .599544/.595648. Neither qualifies for simplification. Full-head
advantages +.004767/+.008664 have positive class intervals, but **H42 fails**:
the HSI-only comparison misses the predeclared .005 threshold. Do not claim strict
necessity of both feature branches or an interaction; capacity also changes.

S25 selectively replicated only the retained full head: four additional fits,
head seeds1/2 on both fixed seed0 encoders, **9.76s** using S24 caches. Exact seed0
cache replay produced zero prediction differences. H43 passes; mean matched gain
+.010522 [.001851,.018941], descriptive seed-delta SD .000279. Against TTA the
mean gain is only +.002459 [−.006634,.011215]. Three heads on one encoder are
head sensitivity, not three full-system training seeds.

S26 fixed the trained head and substituted a TTA anchor, with **zero fits**.
Calibration F1 decreases .002395/.005613. Its prospective both-fold gate rejects
the intervention, with **zero new held-out scores** and H44 not evaluated. Do not
interpret it as measured held-out harm, change its gate, or rescue it by test tuning.
[Study reports](README.md) and the compact hash archives retain all outcomes.

Across S22–S26 there were two complete HSI fits and ten cheap head fits. No rejected
candidate received automatic three-seed expansion. All development outcomes reuse
acquisitions and tests; class intervals omit initialization and new-session uncertainty.

## Next exact action: S27, train against the intended anchor

1. Read the [S27 brief](studies/S27_tta_trained_head/README.md). Preserve S22–S26
   plans and completed outputs; do not rerun a coordinator or overwrite result folders.
2. Profile frozen TTA inference on outer-training rows only, CPU versus available GPU,
   before selecting/freezing runtime. Twelve views are material inference compute.
3. Implement a bounded runner using the same23,514-parameter architecture and S23
   recipe, trained from the outset against **equal TTA fusion**. Reuse S24 single-view
   features/scales, S21 RGB probes, and saved S22 calib/test TTA logits. Generate
   and hash train-only TTA logits once; no new encoder fit is needed.
4. Freeze exact sources/inputs, seed0 on both corrected folds, calib-only selection,
   epoch0 and a practical matched-TTA gate before any new outcomes. The proposed
   H45 gate is mean F1 gain≥.01, paired interval>0, positive gains both folds,
   nonnegative cross delta. Positive cross interval is required for transfer support.
5. Advance only a meaningful practical gain. A clear finalist can earn cheap head
   sensitivity and matched encoder-seed confirmation; do not open a broad sweep.

S27 is a proposed training change, not a completed experiment. No cross-attention,
RGB fine-tuning, band expansion or encoder adaptation is justified by current evidence.
Appearance exceeds silhouette; correct pairing loses to scan shuffle; full spectra
and concatenation can improve F1 while harming transfer. Retain these constraints.

## Eventual confirmation allocation

Freeze the final learned system and matched HSI/fixed-fusion controls at independent
encoder/head seeds0/1/2 on both corrected folds. For a frozen-encoder finalist,
controls share encoder fits: **four additional HSI fits** for encoder seeds1/2,
then matched heads on each encoder (six final cells). Deterministic frozen RGB
probe refits do not create seed evidence. Confirm relevant mechanism controls if
needed for a paper contribution or a meaningful borderline decision; do not
replicate every rejected intermediate. The [confirmation queue](studies/S22_complementary_v5/confirmation_queue.md)
is conditional, not an active six-cell job. All17 bridges touch session8; independent
crossed sessions/lots and locked external evaluation remain necessary.

## Saved assets and integrity

| Repository-relative path | State |
|---|---|
| `dataset_u430k32/`, `dataset_rgb_hsi_v3/` | unchanged validated reference/paired assets |
| `outputs/s20_rgb_features_v3/`, `outputs/s21_rgb_study/` | frozen DINO features and RGB probe exports |
| `outputs/s22_aborted_amendment02/`, `outputs/s22_partial_amendment03/` | preserved unsuccessful local attempts; no held-out score |
| `outputs/s22_kaggle_return/`, `outputs/s22_complementary_v5/f*_s0/` | complete returned CUDA archive, weights, logs, predictions |
| `outputs/s22_fusion_analysis/`, `outputs/s23_frozen_multimodal/` | complete fixed fusion and selected learned head |
| `outputs/s24_branch_multimodal/` | complete four branch heads and reusable train/calib/test feature caches |
| `outputs/s25_head_seed_screen/` | complete selective head sensitivity |
| `outputs/s26_tta_anchor/` | complete calibration rejection; no held-out predictor |
| `docs/research/evidence/S22…S26*/screen_results/` | compact saved evidence and immutable hash archives |

All frozen plan hashes are recorded in each study and `.sha256` sidecar. Large weights,
data and feature caches remain ignored local assets; evidence/source/configs are retained.
FiguresS22–S24 read saved outcomes and were visually checked. S22 selected drivers,
models and tests pass Ruff; nine selected current drivers/models pass strict mypy;
**24 targeted tests pass**. The earlier whole-suite eight historical failures remain
recorded; no repository-wide all-pass claim is made. Final validation receipts are
under `evidence/S22_complementary_v5/validation/final/`. Integrity replay passed
all10 sealed plans/receipts,1,000 input references,62 compact archived artifacts
and52 full completion artifacts; original six-cell S22 hash is unchanged.
The final link check found no missing local links.

S21 replay checks all48 arm/fold arithmetic, all8,624 identities/assets, and40
unchanged historical files. No historical guard was weakened. S17/S18 stay reserved.
Older handoffs are preserved in [CUDA-phase history](evidence/S22_complementary_v5/progress_history_cuda_screen.md),
[amendment history](evidence/S22_complementary_v5/progress_history_amendments_20261005.md)
and [S20 history](evidence/S20_rgb_pathway/progress_history.md).
