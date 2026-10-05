# Future work — the prioritised backlog

What to run next and why, ordered by **how much the answer would change the project per unit of
cost**. Each item names the findings it builds on, the decisions it could reverse, and what a
positive and a negative result would each mean — so whoever picks it up knows what they are
deciding before they spend the compute.

When an item is started it becomes a study (`S##`, see WORKFLOW §5); mark it here with the study
ID and leave the row in place.

---

## Current priority (2026-10-05 evening, after S31–S39: confirm SeedNet-MX, then acquisitions)

**Proposed architecture: [SeedNet-MX](studies/S36_next_generation_architecture/README.md)**
(trained foreground-token RGB + v5, independent evidence, fixed fusion; D58). Its matched
confirmation [S39](studies/S39_final_confirmation/README.md) is frozen. S30 is superseded.

- **FW-47 · S39 HSI seeds (needs the owner's Kaggle authorization).** Four v5 fits (seeds 1/2 ×
  folds) from the built bundle `outputs/s39_kaggle_push/`, ≈ 95 min on 2×T4. The RGB seeds run
  locally (queued). Then score M1–M4.
- **FW-42 (top scientific priority, now also an architecture prerequisite).** Crossed
  sessions/lots. The residual is scan-systematic (F122) and has no learning signal on one scan
  per variety (D56).
- **FW-48 (new, after FW-42) · cross-acquisition, cross-modal consistency learning.** The
  mechanism S36 §6 proposes. It needs ≥ 2 training acquisitions per variety.
- **FW-49 (new, conditional) · the training lever for HSI.** Training was the only transfer lever
  (F127), and v5 is trained from scratch. Fine-tune a pretrained spectral encoder (S19 §4
  HyperSL-type) as the HSI branch, matched to v5. Run it only if S39 confirms that the RGB-side
  training gain is beyond seed noise.
- **FW-50 (new, conditional, GPU) · full ViT-L fine-tuning.** Partial ViT-L ≈ ViT-B locally
  (F120). Worth one GPU screen only if S39's seed SD shows headroom.
- **Closed this phase:**
  - RGB fine-tuning (FW-40 remainder → S32);
  - readout/backbone screens (S31);
  - optics nuisance, CCAR and modality roles (S33/S35/S38, all falsified);
  - learned/complementary/kernel-level fusion (D56).

### Superseded priority note (2026-10-05, after S27–S29)

### (superseded) Priority (2026-10-05, after S27–S29: confirmation, then acquisitions)

**Development screening is closed (D52). Next compute: [S30 matched final confirmation](studies/S30_final_confirmation/README.md)
(proposed, not frozen). Next science: the crossed-session/lot acquisition pilot (FW-42).**
S27 (head trained against TTA, +.0131), S28 (head-seed stable) and S29 (ViT-L RGB, head .6272)
all passed. The selected system's learned head adds only +.0067 over fixed ViT-L fusion (F112).
That margin is for S30 C2 to decide with encoder seeds 0/1/2, which needs four GPU fits and
authorization to use the Kaggle quota. No model change has moved away-from-session-8 recall
(≈ .18; F113). That bottleneck is acquisition-limited.

- FW-45 → S27 complete (H45 pass), S28 complete (H46 pass).
- FW-40 partially completed by S29: frozen DINOv2 B/L capacity screen done; ViT-L adopted (D51).
  Higher-resolution crops, DINOv3/ConvNeXt and RGB fine-tuning remain deferred, now behind S30
  and acquisitions rather than next in line.
- **FW-46 (new) · S30 matched final confirmation.** 4 HSI fits (seeds 1/2 × 2 folds) exporting
  embeddings and train-row TTA, plus 12 cheap heads. Contrasts C1–C4 frozen before scoring. Decides
  head-vs-fixed fusion for the paper system. ≈ 95 min wall on 2×T4 (estimate from S22).
- **FW-42 (raised to top scientific priority).** Crossed sessions and lots, ≥ 4 sessions with all
  target varieties, two lots; pilot ≈ 20 varieties allowed (S19 §5). Only this can test
  away-from-session-8-type transfer; lock the S30 system before applying it.

### Superseded priority note (2026-10-05, before S27)


**Next proposed: [S27, the same small learned head trained against equal TTA](studies/S27_tta_trained_head/README.md).**
S22–S26 are complete and no job is running. S22's corrected fusion effect justifies
learning; the S23/S25 head adds stable single-view gain but little certain advantage
over TTA. S24 fails simplification/strict necessity; S26 fails calib and never scores
a new held-out predictor. Profile outer-training-only TTA inference, then freeze a
bounded seed0/both-fold candidate. Extra seeds go to a meaningful final candidate
and matched controls; no automatic intermediate three-seed validation. Legacy
coverage was defective (F102); never reuse old networks on corrected partitions
or silently amend reserved S17/S18.

- FW-39 completed: all8,624 retained pairs, native masks/crops, compact full215 and
  strict214 measurements, exact historical k32 parity, audited16 exclusions.
- FW-40 initial screen completed in S20/S21 with locally cached DINOv2 and simple
  fusion. Appearance helps; neural transfer improvement is unresolved. Broad RGB
  tuning, DINOv3/ConvNeXt/HyperSL comparisons remain deferred, not missing results.
- FW-41 band screen completed: larger axes improve F1 but lose transfer in these
  probes. Keep k32 provisionally; fixed-width nonlinear/occupied-region mechanisms
  remain conditional separate studies, not automatic next runs.
- FW-42 remains essential: all17 bridge pairs touch session8, whose RGB capture
  settings also change. Independent crossed acquisitions outrank claims of universal
  invariance from existing kernels.
- FW-43 remains open: closest-prior-art matching and a locked independent test are
  still required for a broad final-paper claim.
- FW-44 / H40 → S22 completed: immutable six-fit parent; explicit amendments02/05
  ran seed0 on both corrected folds. H40-screen passes, neural transfer gain unsupported.
  Subsequent learned-head and cheap selective replication are recorded in S23–S26.
  Finalist confirmation follows the [conditional queue](studies/S22_complementary_v5/confirmation_queue.md).
- FW-45 → S27 proposed: learn against the intended TTA anchor with the existing
  23,514-parameter additive architecture. Initially two head fits, no new encoder
  training. Cache train-only frozen TTA inference after runtime assessment; seal
  H45 before execution. Weak practical gain stops expansion; a clear gain earns
  selective sensitivity and matched finalist confirmation. F104–F108; D47–D49.

The dated S19 proposal and reserved HSI queue below are retained as history; this
section explicitly updates their priority and completion status.

## S19 next-generation research priorities (2026-10-03)

Read [S19](studies/S19_next_generation_strategy/README.md) and the
[master plan](MASTER_RESEARCH_PLAN.md) before expanding the compute queue. The existing
bounded S17 remains frozen below; the new route is complementary RGB/HSI information
and independent acquisition evidence, not an open-ended continuation of SeedNet tuning.
S19 updates the availability assumption behind FW-18: raw RGB/HSI archive is now local,
but the full215 cube and validated kernel-level pairing still need building.

### FW-39 · Verify multimodal data identity and full-band reconstruction — next engineering gate
- Builds on F90/F93 and D38/D39. Validate raw payload/checksum, grid/centroid identity,
  all16 missing HSI kernels, RGB masks and correspondence. Rebuild full215 and reproduce
  the existing k32 slice with recorded tolerance; declare white-reference fit/apply scope.
- Deliver versioned paired manifest, QC overlays/exclusions, wavelength and split hashes.
  Failure means repair pairing/calibration before training, not silently drop hard cases.

### FW-40 · Frozen RGB and spectral transfer probes plus simple fusion
- H26/H31; D38. Cache foreground DINOv3/ConvNeXt and optional HyperSL features after
  license/input/overlap checks. Compare modality-only and simple fusion with resolution/
  grayscale/morphology controls. Optional current TabPFN baseline only if90-class and
  resource support verified. A positive result earns bounded tuning; negative triggers
  data/representation diagnostics and a declared stop rule.

### FW-41 · Controlled band and distribution/geometry study
- H27–H29; D39. Separate native-v5 Z2 from fixed-width spectral comparisons of
  historical/nested32/64/195/215. Compare quantiles, pixel bags and occupied spatial
  regions; physical wavelength versus index encoding. Do not require a positive Z2
  for this different full-information test. Freeze a new study before confirmation.

### FW-42 · Standards and crossed acquisitions
- H30; F92/F96; D40. Measure residual nuisance independently of class, then test an
  explicitly bounded correction/consistency family. Plan at least two training, one
  development and one test session with common class support; cross/record seed lots.
  This expands FW-12's third-bundle idea for a stronger external-validation design.

### FW-43 · Matched baselines and final paper evidence
- Reconstruct exact closest-prior-art protocols; common-data implementations of a
  strong conventional, v5, spectral-spatial and RGB–HSI comparator. Freeze the final
  model before a new-acquisition test. Prepare code/splits/negative results/claims
  package; verify journal scope and current category/year quartile at submission.
- See [experiment and paper plan](studies/S19_next_generation_strategy/experiments_and_paper.md).
  No unrun hypothesis is a paper result, and no published score is comparable merely
  because it uses the same dataset name.

---

## Priority 0 — S17: v5's tier-1 row and two screens → frozen in [S16](studies/S16_replication_reading/README.md) (D37)

S15 (read in S16): the lean network replicated — grouped 0.571, stratified 0.745, cross-session 0.20, attraction 0.41 at
3 seeds, every run above every X1 run — and is **SeedNet v5** (F84–F86, D35). Its robustness comes from the spatial
end-map repair (F87). Frozen in `evidence/S16_replication_reading/preregistration_s16.json` (`3b623c45…`). **7 GPU runs
≈ 4.3 h on Kaggle T4 × 2 (one session).** S17 part 1 first: the v5 default switch behind G-neutral (D36), the runner, and
the k64 cube upload (PI).

### FW-37 · The paper's within-acquisition tier on v5 (H23) — **→ S17 Z1**
- **Builds on** F80 (80/20 = 70/30 for X1, one seed), F86 (v5's within-acquisition gain not confirmed). v5 at 80/20,
  stratified, seeds 0–2 (3 runs). H23: 80/20 − 70/30 ≤ +0.03. Either way, the mean ± CI is the D16 tier-1 row.

### FW-03 · Neural confirmation of the band budget — reduced to k64 on v5 (H24a–c) — **→ S17 Z2**
- See the full item below (Priority 1). S17 screens k64 (2 runs) against v5's k32: non-inferior → k32 stays and FW-03
  closes for k ≤ 64; superior without session cost → replicate and screen the 215-band cube; superior with session cost
  → F70's pattern, k32 stays. Needs `./dataset_u430k64` (`scripts/build_presliced_dataset.py --set uniform430_k64`,
  ≈ 4.6 GB) uploaded as a Kaggle dataset.

### FW-38 · Does the end map's extent carry robustness? 4 × 4 end map (H25a, H25b) — **→ S17 Z3**
- **Builds on** F87: the 1 × 1 → 2 × 2 repair alone gave +0.063 cross-session recall and −0.067 attraction. Two readings:
  trainability of the last block (no further change at 4 × 4) or pooling spatial statistics over a map (a dose–response).
  v5 with tail `[2,2,1,1]` (v5's rules), grouped f0/f1, seed 0 (2 runs, screen). A pass is replicated before it can
  replace v5 (D28).

### Read in S16
- **FW-35 · Y3 at seeds 1–2** — H21a–H21d supported, H21e rejected (F84–F86) → SeedNet v5 (D35).
- **FW-36 · Y3 dissection** — H22a (marginal) and H22b supported → redundant by the frozen rule; the robustness is the
  spatial repair's (F87). Not replicated (H22's rule).
- **FW-34 · the X2 re-score** — never run; decides nothing; dropped.

### Read in S14 (screening rejections — not replicated, D30)
- **FW-31 · Y1 decoupled pathways + calib-weighted fusion** — H19a supported, H19b rejected (F76). The robustness S12 saw
  came from under-fitted single pathways (F77), and calib-chosen weights favour the session-carrying pathway (F78).
- **FW-32 · Y2 masked MixStyle** — H20 rejected; the spatial pathway gets weaker without losing its session (F79).
- **FW-33 · Y3 lean architecture** — passed (F74, F75) → replicated in S15 (F84) → v5.
- **FW-17 · X3/Y4 80/20 tier** — H15 supported at one seed (F80) → FW-37 on v5.

## Priority 0c — CPU-first representation track (no held-out until each has its own frozen file)

### FW-27 · Pixel-set (multiple-instance) spectral encoder
- **Builds on** F67: per-band pixel quantiles + morph are linearly worth +0.08 grouped / +0.18 calib over the mean
  spectrum — information only the 3-D CNN reaches today. Pixel-wise seed models with voting are standard in the HSI seed
  literature; attention-MIL (Ilse et al. 2018) learns the pooling.
- **Design.** A shared per-pixel MLP/1-D conv over foreground pixel spectra (≈ 50 k parameters, 256 pixels sampled per
  kernel), gated-attention + mean pooling, morphometrics after pooling; trainable on CPU. Train → calib only.
- **Gate to a GPU arm:** calib macro-F1 ≥ quantile-LDA + 0.03 (≥ 0.73) **and** embedding κ no higher than the
  spectral-only network's + 0.05. If it passes only the first, within-kernel spread is session-laden for any learner.
  **S14 (D32):** κ's seed noise is ≈ 0.05 and it does not rank among spatial-style networks — keep the κ clause only as a
  coarse check (≥ 0.1); the decisive comparison is now against Y3 (calib F1 0.71–0.74), not X1. **S16:** the bar is v5 at
  3 seeds — calib F1 0.733, held-out 0.571 (0.562 no-TTA), cross-session 0.20.

### FW-28 · TabPFN-3 on kernel summaries — the strongest tabular baseline
- **Builds on** F32, F67. TabPFN-3 (2026) reports first place on many-class tabular data and handles ≤ 200 features;
  inputs: mean + morph (40), quantiles + morph (104). Calib first; held-out once, in the paper's baseline table. If it
  matches the network on grouped, the network's claim must be stated against it, not against LDA. **S14:** the network
  bar is now the lean network (grouped 0.562 at seed 0, F74), +0.033 over quantile-LDA no-TTA. **S16:** v5 at 3 seeds:
  0.571 grouped (0.562 no-TTA, +0.044 over quantile-LDA; F89), 0.745 stratified.

### FW-29 · A transfer-standard protocol tier (calibration transfer)
- **Builds on** F66, F68–F70; EPO (Roger et al. 2003), di-PLS. Estimate the session nuisance subspace from the
  between-session differences of *paired* cross-session varieties, cross-fitted by variety (leave-varieties-out), project
  it out, score the held-out varieties. Uses held-out bundles of *other* varieties as transfer standards — a new tier, so
  it needs a D16 extension before it is run. LDA first (CPU minutes).

### FW-30 · Hyperspectral foundation-model encoders
- **Builds on** Theisen & Neubert 2026 (remote-sensing HSI foundation models transfer to proximal sensing, especially
  with little data). After FW-27/FW-31: a frozen pretrained spectral encoder as the per-pixel encoder of FW-27.

## Priority 0 (done) — the S09 sequence (frozen in `preregistration_next.json`)

S09 found the network fit-limited within the acquisition (F34, F35), only ≈ 0.05 above a linear model on
its own scalars (F32), and its cross-session recall matched by shape alone (F33). D17 routes the next
compute through two diagnostics before any new architecture or tuning.

### FW-19 · Instrumentation (D18) — prerequisite, code only
> **Status: done (S11, D22).** All of P0.1–P0.7 plus the X2 pathway switch and the frozen-arm runner; gate
> G-neutral passed (F57). One deviation: the model-declared clip partition is opt-in (`clip_partition=legacy`
> default). Code must be committed and pushed before a Kaggle session clones it.
De-duplicate DDP eval rows before scoring; write the git commit into `run.json`; save calib and held-out
logits (float16); add `SpectralSeedNet.pathway_labels()`; log clean training accuracy (eval mode, no
augmentation, no margin) on a fixed 1,000-kernel training subset. None changes a metric; H13 needs the last.
> **Extended by S10 (D20)** — the full list is S10 §6 P0.1–P0.7: also log the *applied* aux weight and add
> `single.aux_weight_schedule ∈ {legacy, fixed}` with **default `legacy`** (F54 — do not change the applied
> schedule before X1/X2); `train/loss_main`, `train/acc_dominant` (mixup), `train/acc_plain` (margin); model-declared
> gradient/clip groups (`fuse` joins `fusion`); finite-step epoch means; structural tests (zero-gradient parameters,
> tail resolution, gain response — xfail until X5/X6); dry-run composition of every frozen command; docs fixes.
> Gate: a 2-epoch CPU run is bit-identical before and after. Ordered checklist: S10 §9.

### FW-15 · X1 — fit-first regime (H12a, H12b, H13) — **decides route A vs B**
> **Status: done (S11 run → S12 read).** H12a, H12b rejected; H13 supported: fit 0.98–0.99, held-out unchanged (F59, F60).
> Route A (D23). R1 becomes the reference regime (D24).
> **Status: ready to run (S11).** `python scripts/run_frozen.py --arms X1 X4 --nproc-per-node 2 --stream`
> (with X4); every cell composes to its frozen command; where each hypothesis is read from is fixed in S11 §6.
- **Builds on** F34, F35, F36, F37. Could reverse D06, D07.
- **Design.** Overrides `single.mixup_epochs=30 single.arcface_m=0.0 single.margin_warmup_start=31
  single.margin_warmup_end=31 grad_clip=50.0 single.epochs=200 single.patience=40`; grouped folds 0, 1 ×
  seeds 0–2, stratified seeds 0–2 (stratified needs ≥ 3: σ = 0.033). Exact commands are in the frozen file.
  A single bundled arm on purpose: it tests the *regime*; dissection follows only if it moves.
- **Cost.** 9 runs ≈ 4 h on T4 × 2.
- **If H12a and H12b hold:** the score bottleneck is model/training — invest in architecture/training,
  measured on all three D16 tiers. **If H12a holds, H12b fails:** gains do not transfer — route A
  (FW-03, FW-18, FW-12). **If H12a fails:** the regime is not the limit; X2 decides.
- **S10 reading guards (D21).** H13's reference under its own definition is 0.90 / 0.91 (F45), not 0.76–0.87; an
  X1 effect is credited to mixup, margin and epochs jointly (clip 50 ≈ neutral under AdamW, F53); the applied aux
  schedule (0.65 → 0.25, F54) must stay `legacy`. Run X4 (FW-20) in the same session.

### FW-16 · X2 — what the network uses (H14a, H14b)
> **Status: done (S11 run → S12 read).** H14a, H14b rejected: morph scalars do not carry cross-session recall (F63); the
> spatial pathway adds +0.10 and is the session channel (F64); late fusion ≈ joint (F65). One cell unscored (F71a).
> **Status: ready to run (S11).** `model.pathways` implemented (freeze + skip, aux term off with the spatial
> path); `python scripts/run_frozen.py --arms X2 --nproc-per-node 2 --stream`.
- **Builds on** F32, F33; tests D05's two-pathway rationale and D12's effect on the network.
- **Design.** Grouped folds 0, 1 × seeds 0, 1 per arm: `no_morph` (`data.morphology_path=''`, config-only),
  `spectral_only` and `spatial_only` (need a `model.pathways` switch masking a pathway in train *and* eval).
  Shipped regime, in parallel with X1; repeat the deciding arm under X1's regime only if X1 routes to B.
- **Cost.** 12 runs ≈ 4 h.
- **If H14a holds:** cross-session recall is shape — the paper says so; spectral session invariance is
  unproven. **If H14b holds:** 79.6 % of parameters are dead weight; the architecture work targets the
  spatial-spectral pathway, not the head.
- **S10 notes.** Of the spatial path's 2.27 M parameters, 1.94 M can train (F48). The `model.pathways` switch must
  mask the pathway *before* the aux head and set the aux term to zero in `spectral_only` (the aux head sits on the
  spatial output). Eval-time knock-outs already show strong in-sample reliance on morph (−0.22 calib F1, F51); X2's
  retrained arms are the test.

### FW-17 · X3 — the within-acquisition tier (H15, D16)
> **Status: run as S13 Y4 (one seed): 0.728 = the 70/30 level, H15 supported (F80). The 3-seed tier-1 row moves to FW-37.**
- `data=ablation/u430k32_stratified data.split_eval_frac=0.2`, seeds 0–2, under X1's chosen regime. The
  literature-comparable number, labelled as tier 1. H15 checks the S09 prediction that 80/20 adds ≤ 0.03.

### FW-18 · RGB-resolution morphology, CPU first
> **Status (S12): top data item.** Morphometrics are the least session-decodable input measured (κ 0.045, F69) and the
> only one with cross-session recall that does not trade against attraction; the raw archive with the RGB images is not
> on this machine — downloading it (17.3 GB) is the user's call.
- **Builds on** F33 (shape is the only acquisition-invariant cue measured), F43 (prior work's 78–96 % uses
  high-resolution RGB shape). The Zenodo record ships RGB images of the same trays.
- **Design.** Segment kernels in RGB, register them to HSI kernels (Fabiyi et al. did this), compute
  shape/texture descriptors, and run S09's LDA controls: grouped, stratified, cross-session.
- **If cross-session recall ≥ 0.25 and grouped ≥ 0.55 (LDA):** an RGB-shape × HSI-spectrum fusion is the
  architecture direction with acquisition-robust signal. This widens the research question (HSI → RGB +
  HSI) and needs a decision of its own.

## Priority 0b — the S10 arms (frozen in `evidence/S10_training_architecture_review/preregistration_s10.json`)

S10 found the regime, not capacity, to be the demonstrated limiter (F44, F47), the documented aux weight never
applied (F54), and three component defects: an untrainable spatial-tail block (F48), an inert chemometric
descriptor (F49) and a level-blind input path (F50). D19 orders the work: X4 with X1; X5 and X6 after X1.

### FW-20 · X4 — fit ceiling (H16)
> **Status: done (S12).** H16 supported: 1.000 clean fit; softeners worth +0.06–0.17 (F61).
> **Status: ready to run (S11)**, in the same session as X1. H16 is read from `clean_fit.json → final.live.acc`
> of the grouped cell.
- **Builds on** F45, F47; D19's reversal trigger.
- **Design.** X1's schedule with every softener off: `single.mixup_epochs=0 single.label_smooth_hi=0.0
  single.label_smooth_lo=0.0 stage1.aux_loss_weight_init=0.0 stage1.aux_loss_weight_final=0.0 single.dropout=0.0
  single.aug_profile=none single.patience=200` (+ X1's margin/clip/epochs). Grouped f0 s0 and stratified s0.
  Diagnostic only — never a candidate.
- **Cost.** 2 runs ≈ 1 h. **If H16 fails:** capacity/optimisation-limited — X5 and tail layout (S10 P3.6) first.

### FW-21 · X5 — spatial tail without untrainable parameters (H17)
> **Status: folded into S13 Y3** (D25) — capacity is ample (H16), so this is a correctness change tested for non-inferiority.
- **Builds on** F48. Last `ResBlock2D` stride 2 → 1 behind `model.spatial_tail_strides` (default = today's model,
  bit-for-bit). Grouped 2 × 3 under X1's regime if H12a holds. Non-inferiority; adopt as default if it holds.
- **Cost.** 6 runs ≈ 2.7 h.

### FW-22 · X6 — reflectance level in the spectral path (H18a, H18b)
> **Status: deferred (D25).** The linear proxy shows the H18a ∧ ¬H18b pattern: level buys same-session accuracy and
> costs cross-session robustness (F70).
- **Builds on** F50, S09 C3 (+0.085 grouped for LDA from level). Append standardised log mean reflectance (train-split
  statistics) to the descriptor behind `model.spectral_level_block` (default none). Grouped 2 × 3 with the session
  guard (H18b). **Changes what the model may know** — albedo — so it is reported with cross-session recall and
  attraction.
- **Cost.** 6 runs ≈ 2.7 h.

### FW-23 · Training-objective follow-ups (S10 P1.3, P1.4, P3.3–P3.5)
> **Status (S12):** the margin question is closed for now — m = 0 is non-inferior and R1 drops it (D24). The aux-weight
> and label-smoothing arms stay open but are low priority under D23 (they are regime arms).
Each one arm, after X1, under the chosen regime: aux weight as documented (0.2, then 0); a margin with ≥ 25 % of the
LR budget and ε = 0 (A7); label smoothing 0 / s ∈ {16, 24}; an effective weight decay only if X1 over-fits.

### FW-24 · Spectral descriptor repair or removal (S10 P3.1)
> **Status: removal scheduled as part of S13 Y3** — H14b was rejected, but the spectral pathway behaves as shrinkage LDA on
> SNV + morph (F66), so its inert blocks are removed rather than repaired.
Only if X2 shows the spectral path matters (H14b rejected): per-block standardisation instead of one LayerNorm,
an index bank initialised from sharp band pairs, D₁/D₂ dropped (linear in SNV). Otherwise remove the inert blocks.

### FW-25 · Masked normalisation in the stem (S10 P3.2)
GroupNorm statistics over the foreground only, so activation scale stops depending on kernel area.

## Priority 1 — decides what the project is about

### FW-01 · Does reflectance calibration restore cross-session recall?
> **Status (S12):** for the network, the spectral pathway alone reaches 0.214 cross-session recall (F64) and a shrinkage LDA
> on SNV + morph 0.202 (F66) — reflectance + SNV + shape transfers *some* variety signal; level and within-kernel spread
> carry the session (F67, F70).
> **Status (S09):** partly answered. Spectrum-only LDA on reflectance gives 0.039 (k32) / 0.053 (215) against
> 0.000 on SNV-256 — above chance, small. The network's 0.152 awaits X2's no-morph arm (FW-16). The SNV
> arm on identical rows still needs the SNV cube rebuilt (`--radiometry snv`).
- **Builds on** F23 (cross-session recall ≈ 0), F26 (fingerprint at the lamp peak), D12 (the reflectance cube was built to fix this).
- **Design.** Re-run the S05 linear and CNN proxies, unchanged, on the 215-band reflectance cube:
  the same folds, the same pre-registered arm families re-cut on the 215 axis (D13), and the same
  same/cross-session breakdown. Pre-register first (H8: *reflectance raises held-out cross-session
  recall above 0.05 for at least one proxy*). Also run the SNV arm on identical rows — the two cubes
  are row-aligned, so the comparison is kernel for kernel. The S05 scripts in
  `evidence/S05_band_research/code/` show exactly what to repeat; `extract.py` must be pointed at
  the new cube.
- **Why it is a sharp test.** All 17 cross-session varieties pair session 8 (no tile saturation,
  i.e. a different illumination level) with one other session (F22). If the confound is
  illumination, reflectance should remove it precisely for that contrast.
- **Cost.** CPU minutes for LDA; ~1 h for the CNN proxy. No network training.
- **If positive:** the session confound was largely illumination shape; reflectance is the
  correct input; cross-session recall becomes the honest variety metric (FW-05).
- **If negative:** the fingerprint is not (only) illumination — detector state, focus, handling or
  seed lot. Variety recognition across sessions is then not measurable from this dataset, and
  that *is* the paper's result. Revisit D12's cost (41 dropped bands).

### FW-02 · Run-to-run σ of SpectralSeedNet (ablation A12)
> **Status: done (S08 sweep → S09, F30).** σ = 0.009 grouped, 0.033 stratified, on k32.
- **Builds on** D03. Nothing about the network can be compared until σ is known.
- **Design.** Identical config, grouped fold 0, 5 seeds, on Kaggle T4 × 2 (D15). Report σ of
  held-out macro-F1 and of cross-session recall.
- **Cost.** 5 runs. Do it as the first use of the Kaggle setup.

### FW-03 · Neural confirmation of the band budget (S08)
> **Status (S09):** re-ordered after FW-15 — run under the regime X1 selects. LDA now prefers 215 bands over k32
> by +0.04 grouped (C3), against the S05 CNN proxy's preference for k32–64. Arms: k32, k64, full 215.
> **Status (S16):** k64 on v5 frozen as S17 Z2 (a 2-run screen, H24a–c); 215 bands only if k64 is superior (D37).
- **Builds on** F17 (fewer bands better for the CNN proxy), challenges D04.
- **Design.** SpectralSeedNet on `uniform430` k ∈ {16, 24, 32, 48, 64} vs the full 215-band cube,
  grouped, 2 folds × 3 seeds, with session reporting (D14). Pre-register non-inferiority at −0.01
  beyond 2σ from FW-02.
- **If k ≤ 64 is non-inferior:** reverse D04 — the default becomes a 430-nm-floored evenly spaced
  set, a 7–13× cheaper input and a deployable multispectral design.
- **If the full cube wins:** the CNN proxy's preference was a property of the proxy; D04 stands.

### FW-04 · The leakage gap for the network (ablation A1)
> **Status: done (S09, F30, F38).** +0.182, size-confounded; the three-level reporting it asked for is D16.
- **Builds on** D01, F02. `F1_stratified − F1_grouped`, same model, 3 seeds each. Now also report
  the *session* decomposition, which makes it a three-level gap: patch-level → bundle-held-out →
  cross-session.

### FW-05 · Make cross-session recall a first-class metric
> **Status: decided (D16 tier 3, D14 extension):** reported with a shape-only control.
- **Builds on** F22, F23, D14.
- **Design question.** Under grouped, 73 classes are evaluated within-session and 17 across. Options
  to evaluate: (a) report cross-session macro-recall as a co-headline beside macro-F1; (b) a
  leave-one-session-out protocol for the 17 cross-session varieties only; (c) restrict claims of
  "variety recognition" to the cross-session subset. Decide this *before* FW-03/FW-04 report, so
  the decision is not shaped by the numbers.

## Priority 2 — tests decisions taken on thin evidence

### FW-06 · What are the hard classes? (ablation A9)
> **Status (S09, F42):** 70 is a session failure; {0, 30, 41, 49, 51, 52} confuse mutually in-distribution.
> Open: the segmentation audit and mean-spectrum overlays.
- **Builds on** F09. No training: mean-spectrum overlays, segmentation-quality audit (area,
  eccentricity, solidity distributions; kernels lost to the shape gate), embeddings, 90 × 90
  confusion. New angle from S06: check each hard class's session pattern (class 70 is
  cross-session).

### FW-07 · Falsify D05 and D06 (ablations A3, A8)
- Both were adopted on one leaky run. Run A3 (branches, *symmetric* dropout) and A8 (stages) under
  grouped once σ is known. Either can reverse a decision.

### FW-08 · The paper's baseline table on the reflectance cube
> **Status (S09):** done for k32 and 215 (LDA, LinearSVC, four non-linear tabular models, with and without
> morphometrics, both protocols, session breakdown) — `evidence/S09_post_sweep_forensics/c3_*.csv, c5_*.csv`.
- LDA and LinearSVC on mean spectra under grouped, full cube and uniform430 k32, with session
  breakdown. Seconds of CPU; it is the floor every network number is read against (CHANGES §19.4).
  Already wired: `python scripts/run_protocol.py --baseline`.

### FW-09 · Record why k = 32 (D15)
- The frozen rule gave k\* = 24. Either write the reason for 32 into D15, or include k = 24 in FW-03.

## Priority 3 — housekeeping and longer-term

### FW-10 · Session invariance, only if FW-01 is negative
> **Status (S12):** narrowed. Session-adversarial heads cannot work under `grouped` (session never varies within a class in
> training); illumination-shape augmentation is open; the in-kernel noise floor is a fingerprint but removing it does not
> remove the session (F68). The live candidates are S13 Y1/Y2 and FW-29.
Speculative: per-session normalisation against the tile spectrum; augmenting illumination shape
during training; a session-adversarial head. Each needs a cross-session test set to be judged,
which only 17 classes provide — so expect wide intervals.

### FW-11 · Re-capture the Stage-1 golden digests
`test_stage1_epoch_loss_matches_golden` / `…_weights_…` fail from torch/BLAS drift (loss 23.0653
vs 23.0805). Re-capture on the target machine with `python scripts/capture_golden.py`.

### FW-12 · More acquisition units per class
The binding constraint (two bundles per class, mostly in one session) is in the data, not the
method. A third bundle per variety, imaged in a *different* session, would allow a real
three-way split and a cross-session test for every class. Worth raising with the dataset authors
or planning as new acquisition.

### FW-13 · A2 as a curve, with the network
Band selection on all data vs within-fold at several k, neural, grouped — the CHANGES §19.3 test,
using the per-fold band files the band study already writes.

### FW-26 · Restore the pinned pre-refactor reference for two gates
`scripts/capture_golden.py --verify` and `scripts/check_config_roundtrip.py` read commit `886560fe…` with
`git show`; that commit is not in this repository's history (S11), so both fail before doing anything — before and
after S11 alike. Either restore the ref (fetch the old history or vendor the two baseline files) or retire the two
scripts in favour of the committed-golden pytest gates, which run without it. Same housekeeping pass: two smoke
tests (`test_the_primary_pipeline_runs_on_all_256_acquired_bands`, `test_the_stem_reads_the_full_band_axis_in_the_run_that_ships`)
build a 256-band synthetic cube against the 215-band primary config and fail since S07; one expects an fp16 NaN that
torch 2.14 no longer produces (`test_amp_precision`); the W&B step-collision smoke test fails. All fail identically
before and after S11 (`evidence/S11_frozen_arms_execution/test_tiers.json`).

### FW-14 · Write-up
Lead with the protocol: stratified → grouped → cross-session, as a three-level gap with intervals;
then the band-budget result; publish the negative results (F05, F09, F18, F25).

---

## Done / moved to a study

| Item | Became |
|---|---|
| Band study inside the fold (CHANGES §19.3 / IC-4) | S03, S05 |
| White-tile reflectance (raised in S06) | S07 |
| Kaggle infrastructure for neural runs | S04 (Oct 2026 part), S08 |
| FW-02 σ, FW-04 leakage gap (first neural sweep, u430k32) | S08 (run) → S09 (analysis) |
| FW-19 instrumentation (D18, D20) + X2's pathway switch + the frozen-arm runner | S11 |
| FW-15 X1, FW-16 X2, FW-20 X4 (run S11, read) | S12 |
| FW-31 Y1, FW-32 Y2, FW-33 Y3, FW-17 Y4, FW-34 P0 (run S13, read) | S13 → S14 |
| FW-35 Y3 replication, FW-36 Y3 dissection (run S15, read) | S15 → S16 |
| FW-37 v5 tier-1 row, FW-03 (k64 screen), FW-38 end-map extent | S17 (frozen in S16) |

| FW-44 corrected-fold neural rebaseline and fixed fusion | S22 (two-fit amended screen complete) |
| FW-45 learned head against the TTA anchor | S27 + S28 (both gates pass) |
| FW-40 (part) frozen RGB backbone capacity | S29 (ViT-L adopted) |
| FW-40 (rest) RGB readouts and fine-tuning | S31 (readouts) + S32 (trained branch adopted, D55) |
| FW-46 S30 matched confirmation | superseded by S39 (D58) |
