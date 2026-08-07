# MIGRATION_PROGRESS

Execution tracker for `IMPROVEMENT_PLAN.md`. One section per phase; each records what
was run, the numbers it produced, and what those numbers change about the plan.

| Phase | Scope | Status | Date |
|---|---|---|---|
| **Phase 0** | `0-A` … `0-J` — measure before modifying | **✅ COMPLETE** | 2026-08-07 |
| **Tier 1** | `T1-1` … `T1-10` — correctness fixes | **✅ COMPLETE** | 2026-08-07 |
| **Tier 2** | `T2-1` … `T2-10` — optimisation & curriculum | **✅ COMPLETE** | 2026-08-07 |
| **Tier 4** | `T4-1` … `T4-6` — protocol | **✅ COMPLETE** | 2026-08-07 |
| **Tier 3** | `T3-1` … `T3-7` — architectural redesign | **✅ COMPLETE** | 2026-08-07 |
| | **`IMPROVEMENT_PLAN.md` §4.2 — every item implemented** | **✅ MIGRATION COMPLETE** | 2026-08-07 |

Tier 4 is done **before** Tier 3, per §4.4's first ordering constraint: redesigning
branches against a leaky split optimises for the leak.

**What "complete" means here, and what it does not.** Every item in §4.2's
implementation matrix is implemented, tested and gated. **None of the Δ macro-F1
figures in the plan has been verified**, because verifying any of them requires
a training run and this work produced none. Six items across Tiers 3 and 4 are
additionally *inert* until `scripts/prepare_dataset.py` is re-run: they read
arrays that run writes, and until then the code takes documented, exact
fallbacks. The claims that **have** been settled are the ones that are
properties of the code — every §4.2 validation criterion that does not require
a trained model, and the five Phase-0 predictions the measurements resolved.
Each is recorded in its tier's section below, alongside the three plan estimates
the measurements **refuted**.

No file under `src/spectralquadnet/`, `configs/` or `train.py` was modified during Phase 0
(`git status` confirms: the only new paths are this file and four scripts). Everything
below comes from four new read-only scripts, listed under [Artifacts](#artifacts). After
adding them: `ruff` clean, `black` clean, `mypy --strict` clean over all 72 source files,
and the existing gate suite still passes — **107 passed**.

> **Note on the numbers below.** Every TTA figure in Phase 0 was produced by the *pre-Tier-1*
> spectral transform. T1-1/T1-2 corrected it, which moves the selected checkpoint's 12-view
> result from 0.8934 to 0.8889. The no-TTA columns are unaffected. See
> [T1-1, T1-2](#t1-1-t1-2--tta--enginettapy).

---

# Phase 0 — measure before you modify ✅ COMPLETE

## How it was run

| | |
|---|---|
| Checkpoints | `outputs/output_v12_spa40/best_stage{1,2,3}.pth` |
| Splits | `build_splits` — train 6,036 / val 1,294 / test 1,294 (stratified, `random_state=42`) |
| Device | `mps` (Apple Metal), torch 2.13.0 |
| Original run | CUDA A100 80 GB, `TF32=True` (per `baseline_console_log.txt`) |
| Evaluated weights | the **EMA shadow**, as `final_eval.py` does |

**Reproduction check — the port from CUDA to Metal is faithful.**

| Quantity | Reference (A100) | This run (MPS) | Δ |
|---|---|---|---|
| Stage 1 test macro-F1, no TTA | 0.8770 | 0.8770 | 0.0000 |
| Stage 1 test macro-F1, 12-view TTA | 0.8933 | 0.8934 | +0.0001 |
| Stage 1 val macro-F1 (`stage1_meta.json`) | 0.88766 | 0.8877 | 0.0000 |
| Stage 3 val macro-F1 (`stage3_meta.json`) | 0.87451 | 0.8745 | 0.0000 |
| Stage 2 val macro-F1 (`stage2_meta.json`) | 0.88666 | 0.8851 | **−0.0016** |

Stage-1 predictions agree with the archived `test_preds_TTA.npy` on 1,293/1,294 patches
(99.92 %) — the reference arrays in `outputs/output_v12_spa40/` are Stage 1's, confirming
`_pick_best_checkpoint` selected Stage 1. The Stage-2 val shortfall is ≈2 patches out of
1,294 and is **5× smaller than the bootstrap SE of 0.0085**; the cause was not isolated
(Stage 3 also routes through the ArcFace head and reproduces exactly), and it does not
affect any Phase-0 conclusion.

---

## 0-A · `final_eval` on all three checkpoints → **F-7**

`final_evaluation` was called unmodified, once per checkpoint, with `output_dir`
redirected so the reference run's artifacts were not overwritten.

### Test split, macro-F1

| Checkpoint | val_f1 (recorded) | no TTA | 12-view TTA | 8-view spatial only | 12-view fp32 |
|---|---|---|---|---|---|
| `best_stage1.pth` (ep 488) | 0.8877 | **0.8770** | **0.8934** | 0.8886 | 0.8935 |
| `best_stage2.pth` (ep 50) | 0.8867 | 0.8737 | 0.8876 | 0.8846 | 0.8868 |
| `best_stage3.pth` (ep 120) | 0.8745 | **0.8771** | 0.8868 | **0.8900** | 0.8861 |

### Validation split, macro-F1

| Checkpoint | no TTA | 12-view TTA |
|---|---|---|
| `best_stage1.pth` | **0.8877** | 0.8869 |
| `best_stage2.pth` | 0.8851 | **0.8910** |
| `best_stage3.pth` | 0.8745 | 0.8834 |

### Verdict

**F-7 CONFIRMED — Stage 3 evaluated on *test* is within noise of Stage 1.**

Without TTA the two are indistinguishable to four decimal places: Stage 3 **0.8771** vs
Stage 1 **0.8770**, a paired difference of **−0.0001, 95 % CI [−0.0126, +0.0122],
p = 0.96**. With 12-view TTA Stage 1 leads by +0.0066, also inside noise (p = 0.27).

The 0.0132 val gap that made Stage 3 look like a regression, and that caused
`stage3_meta.json` to record *"val_f1 did not beat Stage 2; Stage 2 ckpt preferred for
eval"*, **does not survive onto the held-out split**. §1.1(ii) and §2.1.3 are supported:
the Stage-3 "regression" is a selection artifact of one 1,294-patch split, not a measured
loss of generalisation.

---

## 0-B · `tta_spatial=8 tta_spectral=0` → **F-6**

Run on all three checkpoints rather than only the selected one, so the direction of the
effect can be separated from checkpoint noise.

| Checkpoint | 12-view | 8-view spatial only | Δ (12 − 8) | 95 % CI | p |
|---|---|---|---|---|---|
| Stage 1 | 0.8934 | 0.8886 | +0.0048 | [−0.0064, +0.0165] | 0.41 |
| Stage 2 | 0.8876 | 0.8846 | +0.0030 | [−0.0063, +0.0124] | 0.53 |
| Stage 3 | 0.8868 | 0.8900 | −0.0031 | [−0.0121, +0.0054] | 0.46 |

### Verdict

**F-6 NOT SUPPORTED as stated, but the spectral views are not earning their cost.**

F-6 predicted spatial-only ≥ the 12-view result. On the selected checkpoint the point
estimate goes the other way (12-view is +0.0048 better), and on Stage 3 it goes F-6's way
(spatial-only +0.0031). **No comparison is significant** — every CI straddles zero. The
measured effect of the 4 spectral views is indistinguishable from zero in either
direction, while costing 50 % more inference compute.

This weakens the C-8 argument's *magnitude* without touching its *correctness*: the
transform still violates the zero-background invariant (§2.6.1), it just does not cost
measurable macro-F1 at the ±0.011 resolution these paired intervals afford. **T1-1 should
be re-scoped from a claimed +0.002…+0.006 gain to a correctness fix with an expected
effect near zero.**

### Bonus measurement — the §2.6.4 precision confound (T1-2)

`engine/tta.py` wraps every view in `autocast(...)`; `final_eval.py`'s no-TTA arm does
not, so the shipped TTA/no-TTA comparison confounds ensembling with a precision change.
Forcing the TTA path to fp32 moves macro-F1 by **≤0.0007 on every checkpoint** (Stage 1
+0.0001, Stage 2 −0.0008, Stage 3 −0.0007), all inside noise. **T1-2 is hygiene, not a
gain** — its "±0.001 (removes a confound)" estimate is correct.

TTA vs no-TTA itself *is* real on the two earlier checkpoints: Stage 1 +0.0164
(CI [+0.0056, +0.0281], p = 0.003), Stage 2 +0.0138 (p = 0.013), Stage 3 +0.0097
(p = 0.14, not significant).

---

## 0-C · Bootstrap confidence intervals

10,000 resamples, 95 % percentile intervals, paired on shared resamples
(`scripts/bootstrap_ci.py`).

### The noise floor

| Split | n | Bootstrap SE of macro-F1 | Typical 95 % CI width |
|---|---|---|---|
| val | 1,294 | 0.0083 – 0.0087 | 0.033 |
| test | 1,294 | 0.0081 – 0.0089 | 0.033 |

**A ±0.017 interval on every reported number.** For scale: **all ten Tier-1 items claim a
maximum effect of ≤0.008 — every one of them sits entirely below a single standard
error**, and so do 7 of the 10 Tier-2 items. Only T2-7, T2-8 and T2-10 claim an upper
bound above the noise floor.

### Paired stage comparisons

| Split | Comparison | Δ | 95 % CI | p | Verdict |
|---|---|---|---|---|---|
| val | S1 − S3 (no TTA) | +0.0132 | [+0.0018, +0.0249] | 0.022 | outside noise |
| val | S2 − S3 (no TTA) | +0.0106 | [+0.0009, +0.0207] | 0.032 | outside noise |
| val | S1 − S2 (no TTA) | +0.0026 | [−0.0087, +0.0139] | 0.64 | within noise |
| **test** | **S1 − S3 (no TTA)** | **−0.0001** | **[−0.0126, +0.0122]** | **0.96** | **within noise** |
| test | S1 − S2 (no TTA) | +0.0033 | [−0.0086, +0.0151] | 0.58 | within noise |
| test | S2 − S3 (no TTA) | −0.0034 | [−0.0159, +0.0089] | 0.57 | within noise |
| test | S1 − S3 (12-view TTA) | +0.0066 | [−0.0052, +0.0184] | 0.27 | within noise |
| test | S1 − S2 (12-view TTA) | +0.0058 | [−0.0053, +0.0171] | 0.30 | within noise |

The two "outside noise" rows are both on **val — the split selection was performed on**,
after six selection jobs. Per §2.1.3 those p-values are not honest; they are exactly the
winner's-curse signature the plan predicts. On the split nobody selected on, **all twelve
stage-vs-stage differences (3 checkpoint pairs × 4 TTA variants) are within noise**, the
largest being S1 − S3 under fp32 TTA at +0.0074, p = 0.21.

### val → test gap

| Array | val | test | gap |
|---|---|---|---|
| stage1 noTTA | 0.8877 | 0.8770 | +0.0106 |
| stage2 noTTA | 0.8851 | 0.8737 | +0.0114 |
| stage3 noTTA | 0.8745 | 0.8771 | −0.0026 |
| stage1 tta12 | 0.8869 | 0.8934 | −0.0065 |
| stage2 tta12 | 0.8910 | 0.8876 | +0.0035 |
| stage3 tta12 | 0.8834 | 0.8868 | −0.0034 |

The gap is positive for the two checkpoints that were *selected* on val (S1, S2) and
negative for the one that was not (S3) — consistent with T4-5's premise that 270 fitted
parameters (margins, CDWS, oversampling) touch the selection split. The magnitude
(≈0.011) is, however, itself ~1.3 SE.

---

## 0-D · SVD of the (9, 40) statistics tensor → **F-2**

`masked_spectral_stats` was run over all 6,036 training seeds, twice: on the raw patch,
and on `se(x)` — what `SpectralQuadNet.forward` actually hands to Branch B.

```
[raw] n=6,036  σ2/σ1 med=0.0167  σ3/σ1 med=7.16e-03  p90=1.28e-02  max=2.94e-02
      seeds with σ3/σ1 < 0.05 : 100.00%   energy in top-2 SVs: mean 0.999906  min 0.999085
      effective rank @1%      : mean 2.10  median 2  max 4
[ se] n=6,036  σ2/σ1 med=0.0167  σ3/σ1 med=7.16e-03  p90=1.28e-02  max=2.94e-02
      seeds with σ3/σ1 < 0.05 : 100.00%   energy in top-2 SVs: mean 0.999906  min 0.999085
      effective rank @1%      : mean 2.10  median 2  max 4
```

### Verdict

**F-2 CONFIRMED, and more strongly than predicted.** The prediction was σ₃/σ₁ < 0.05 for
>90 % of seeds; the measurement is **100.00 % of seeds**, with a *maximum* σ₃/σ₁ of
0.0294 across all 6,036 — i.e. no seed anywhere in the training set comes close to the
threshold. Two singular values carry **99.99 %** of the tensor's energy (worst-case seed:
99.91 %). Mean effective rank at a 1 % tolerance is **2.10**.

C-4 holds: **686,424 parameters (8.7 % of the model) operate on ≤ 2 effective degrees of
freedom**, one of which (the mean spectral shape **r**) Branch A already receives. T3-1
is justified on the measurement, not just the derivation.

*Incidental finding:* the raw and SE-gated tensors agree to 5 significant figures because
the trained `MaskedSpectralECA` gate is nearly an identity — measured per-band gain
**1.000 – 1.033** (std across bands 0.0016), against a designed range of "1.0× to 2.0×".
The shared spectral attention is doing essentially nothing. Not in the plan's defect
taxonomy; worth adding.

---

## 0-E · Fusion latent cosine similarity → **F-4**

Two measurements: the `cross_interaction.latents` **parameter** (what 0-E names), and the
**post-block latent state** on a real validation batch, captured with a forward hook that
reconstructs `input + output` across the last block's residual feed-forward — i.e. the
exact tensor `latents.mean(dim=1)` then pools.

```
stage 1: latents param   max cos=+0.7607  mean=+0.3371  ‖L_n‖=[0.67, 0.76, 0.75, 0.68]
         post-block Lₙ  max cos=+0.9998  mean=+0.9522  min cos(Lₙ, mean)=+0.9395
stage 2: latents param   max cos=+0.7499  mean=+0.3305  ‖L_n‖=[0.69, 0.74, 0.73, 0.66]
         post-block Lₙ  max cos=+0.9998  mean=+0.9458  min cos(Lₙ, mean)=+0.9311
stage 3: latents param   max cos=+0.7494  mean=+0.3355  ‖L_n‖=[0.67, 0.74, 0.74, 0.65]
         post-block Lₙ  max cos=+0.9999  mean=+0.9517  min cos(Lₙ, mean)=+0.9387
```

### Verdict

**F-4 REFUTED as stated; M-1's conclusion CONFIRMED by a different measurement.**

The metric F-4 names — max pairwise cosine of the latent *parameter* — is **0.75, not
>0.95**, in all three checkpoints. Training also grew the latent norms from the
σ=0.02 initialisation (‖L‖ ≈ 0.32) to ≈0.70, so the specific
initialisation-scale premise of §2.3.1 partly resolved itself during Stage 1.

But the thing M-1 actually claims — that `f = mean_n L_n ≈ L_1`, so the fusion's effective
latent capacity is 1 — **is true**. After the two cross/self-attention blocks the four
latents are near-identical on real data: max pairwise cosine **0.9998**, mean **0.95**,
and every latent sits at cosine **≥0.93** to their own mean. Diverse parameters, collapsed
states: the collapse happens *in the blocks*, not at initialisation.

**Consequence for the plan:** FU-1(b)'s fix (rescale the latent init) targets a cause that
the measurement rules out. The learned per-latent code `c_n` — fix (ii) — and the
`test_fusion_latents_are_diverse` regression test both need restating over the
**post-block** latents, or they will pass while the collapse persists.

---

## 0-F · Wavelength spacing → magnitude of **C-5**

```
bands=40  span=383.22–1006.47 nm
Δλ  min=2.444  median=12.222  mean=15.981  max=44.000 nm
Δλ  max/min ratio = 18.00×

Δλ histogram:
  [   2.44,    6.60) nm : █████████ (9)
  [   6.60,   10.76) nm : █████████ (9)
  [  10.76,   14.91) nm : ███ (3)
  [  14.91,   19.07) nm : · (0)
  [  19.07,   23.22) nm : ████ (4)
  [  23.22,   27.38) nm : ████████████ (12)
  [  27.38,   31.53) nm : · (0)
  [  31.53,   35.69) nm : · (0)
  [  35.69,   39.84) nm : · (0)
  [  39.84,   44.00) nm : ██ (2)

kernel=3 receptive span: min=4.89  max=70.89 nm  (14.50× spread)
```

The grid is strongly bimodal — a dense cluster below ~14 nm (21 of 39 gaps) and a sparse
cluster at 19–27 nm (16 gaps) — not merely noisy. Every `kernel=3` convolution in branches
A, B and D applies **one shared set of finite-difference weights across steps spanning
18×**, and its receptive field varies from 4.89 nm to 70.89 nm depending on where it lands.
C-5's magnitude is confirmed; **T3-5 / FE-1 is well-founded.**

---

## 0-G · Hard-class precision vs recall → **F-5**

From the 12-view TTA predictions of each checkpoint
(`outputs/phase0/preds/test_stage{n}_tta12.npy`) plus the archived
`outputs/output_v12_spa40/test_preds_TTA.npy`, against `test_targets.npy`. F-5 predicts
≥3 of `{49, 52, 41, 51, 37}` show `R_c ≪ P_c` (threshold: R − P < −0.05).

**Selected checkpoint (Stage 1) and the archived reference run — 5 hardest by F1 are
exactly `[49, 52, 41, 51, 37]`, the predicted set:**

| class | P | R | F1 | sup | R − P | |
|---|---|---|---|---|---|---|
| 37 | 0.727 | 0.571 | 0.640 | 14 | **−0.156** | ← wrong sign |
| 41 | 0.583 | 0.500 | 0.538 | 14 | **−0.083** | ← wrong sign |
| 49 | 0.538 | 0.467 | 0.500 | 15 | **−0.072** | ← wrong sign |
| 51 | 0.524 | 0.786 | 0.629 | 14 | +0.262 | rule directionally right |
| 52 | 0.500 | 0.571 | 0.533 | 14 | +0.071 | rule directionally right |

Counts of wrong-sign classes within the F-5 set: **Stage 1 → 3/5**, reference run → 3/5,
Stage 2 → 2/5 (but class 37 at P = 0.875, R = 0.500, R − P = **−0.375**), Stage 3 → 2/5.

**On the validation split — which is what the margin rule actually reads** — Stage 1 gives
2/5, driven by class 49 (P = 0.875, R = 0.500, **−0.375**) and class 37 (P = 0.900,
R = 0.600, **−0.300**).

### Verdict

**F-5 CONFIRMED on the selected checkpoint** (3/5, threshold ≥3), marginal elsewhere
(2/5). The substance holds regardless of the count: for classes 37, 41 and 49 the failure
mode is **too few positives predicted, not too many**, and
`m_c = m_base + m_delta·(1 − F1_c)` responds by *widening* their margin — suppressing the
logit further. **T2-8 (HD-3, signed `R_c − P_c` rule) is justified**, and the two classes
with the largest wrong-sign gaps also carry above-average CDWS weights (class 37: 1.19,
class 49: 1.33, against a median of ~0.93 in `stage1_meta.json`) — so the sampler
over-samples them while the margin rule suppresses them.

Note the hardest-5 set is not stable across splits: val ranks class **30** into the bottom
five (F1 0.615 in `stage1_meta.json`) where test does not, another ±1-class illustration
of the 0-C noise floor.

---

## 0-H · Scan / session group census

`scan_id` was reconstructed without re-segmenting the archive: `patch_extraction.py`
writes patches cube by cube in `zipfile.infolist()` order, so `labels.npy` is a
concatenation of constant-label blocks. The zip's central directory was replayed through
the same filter/`factorize` logic and aligned 1:1 against those blocks (107 groups vs 107
blocks, 8,624/8,624 patches assigned — verified, not assumed).

```
Patches on disk           : 8,624      Classes : 90
Distinct (session, variety) groups   : 107
Distinct physical cubes (zip index)  : 180
Distinct sessions                    : 9
Patches per group min/med/max        : 45/96/96
Groups per class  min/med/max        : 1/1/2      Cubes per class : 2/2/2
Varieties per group                  : 1–1

Groups-per-class histogram:  1 group(s): 73 classes   |   2 group(s): 17 classes
Cubes-per-class  histogram:  2 cube(s): 90 classes

Groups per session:
  Data-VIS-20170111-1: 5 groups,  432 patches      Data-VIS-20170113-1: 14 groups, 1293 patches
  Data-VIS-20170111-2: 15 groups, 1338 patches     Data-VIS-20170113-2: 15 groups, 1006 patches
  Data-VIS-20170112-1: 14 groups, 1296 patches     Data-VIS-20170116-1:  5 groups,  480 patches
  Data-VIS-20170112-2: 9 groups,   816 patches     Data-VIS-20170117-1:  5 groups,  382 patches
                                                   Data-VIS-20170203-1: 25 groups, 1581 patches
```

**Leakage in the current split — total.**

```
Groups in train / val / test        : 107 / 107 / 107
Groups shared train∩test            : 107  (100.0% of test groups)
Test patches whose group is in train: 1,294/1,294 (100.0%)
```

### Verdict — this is the most consequential Phase-0 result

C-1 is confirmed at maximum severity: **every one of the 107 capture groups appears in all
three splits.** There is no patch in val or test whose physical scan the model did not
train on.

**But F-1 / T4-1 as written is not constructible.** Every variety was captured exactly
twice, and for **73 of 90 classes both cubes are in the same session** — so a
session-disjoint `StratifiedGroupKFold(groups=scan_id)` would leave those 73 classes
entirely absent from one side of the split. Only 17 classes span two sessions.

Two options remain, and the plan must pick one before Tier 4:

1. **Cube-disjoint split** (feasible now — every class has exactly 2 cubes, so a
   leave-one-cube-out / 50-50 design works). Controls for seed identity and scan geometry,
   but *not* for session-level radiometric drift, since the two cubes of the 73
   same-session classes share illumination and are minutes apart. `scan_id.npy` as
   reconstructed here is at group granularity; separating the `-01`/`-02` cubes inside a
   group requires re-running segmentation, since `labels.npy` does not record the boundary.
2. **Session-disjoint split on the 17 two-session classes only**, reported as a separate
   diagnostic rather than the headline protocol.

F-1's predicted −0.05…−0.20 drop cannot be tested as specified. **T4-1 needs redesigning,
and it gates the Tier-3 re-baseline in §4.4.**

---

## 0-I · ArcFace sub-centre win rates → **F-8**

Win counts computed over all 6,036 training patches from `return_embed=True` embeddings
against `arcface_head.weight`, under two criteria:

* **global** — does `(c, k)` ever win `argmax_k` for class `c` on *any* sample? This is
  what F-8 is stated over, and it is what determines whether the sub-centre receives
  gradient at all (every class logit enters the softmax, so negatives feed it too).
* **own-class** — does `(c, k)` win for samples whose label *is* `c`? This is what decides
  whether the sub-centre represents a real sub-population, which is the feature's purpose.

```
stage 1  K=3  C=90  N_train=6,036 (head never bootstrapped — control only)
  global : dead (c,k) pairs 0/270 (0.0%)     classes with >=1 dead sub-centre 0/90
  own-cls: dead (c,k) pairs 146/270 (54.1%)  classes with >=1 dead sub-centre 86/90
  dominant sub-centre share (global): mean 0.372  min 0.340  max 0.465  (1/K = 0.333)

stage 2  K=3  C=90  N_train=6,036
  global : dead (c,k) pairs 0/270 (0.0%)     classes with >=1 dead sub-centre 0/90
  own-cls: dead (c,k) pairs 179/270 (66.3%)  classes with >=1 dead sub-centre 90/90
  dominant sub-centre share (global): mean 0.394  min 0.338  max 0.480

stage 3  K=3  C=90  N_train=6,036
  global : dead (c,k) pairs 0/270 (0.0%)     classes with >=1 dead sub-centre 0/90
  own-cls: dead (c,k) pairs 180/270 (66.7%)  classes with >=1 dead sub-centre 90/90
  dominant sub-centre share (global): mean 0.400  min 0.338  max 0.495
```

### Verdict

**F-8 REFUTED as stated; M-8's core prediction CONFIRMED exactly.**

No sub-centre has zero global win rate — the global shares are near-uniform (mean dominant
share 0.39 vs 0.333 for perfect balance). §2.4.3's mechanism claim that a losing
sub-centre's *"gradient is identically zero forever"* is therefore **wrong**: the head's
`max` is taken per class over all 90 logits, so every sub-centre keeps receiving gradient
through the negative-class terms, which is why they drift toward uniform usage.

The own-class criterion confirms the prediction that matters. In Stage 2, **179 of 270
(c, k) pairs never win for their own class**, leaving 91 live sub-centres across 90
classes — i.e. **~1.01 live sub-centres per class**. That is §2.4.3's stated outcome
verbatim: *"K = 3 therefore initialises to one live centroid and two randomly-oriented
decoys."* All 90/90 classes have at least one own-class-dead sub-centre, and the picture
worsens monotonically (Stage 1 146 → Stage 2 179 → Stage 3 180).

**T2-9 (HD-2) is justified**, but its validation criterion — *"dead-sub-centre count (0-I)
drops to 0"* — must be restated over the **own-class** criterion, since the global count
is already 0 and would pass trivially. The 512 parameters per class in the two decoy
sub-centres are, at inference, only able to raise off-target cosines.

---

## 0-J · Config-key wiring audit → **N-1a…f**

Values read off the constructed, checkpoint-loaded module and compared with
`configs/model/spectral_quadnet_v4.yaml`.

```
config key                       yaml           module  status
model.fusion_heads                  4                8  DEAD (overridden)
model.fusion_drop                 0.1              0.1  wired
model.specf_drop                 0.15              0.1  DEAD (overridden)
model.specf_dim                   256              256  wired
model.specf_heads                   8                8  wired
model.specf_layers                  4                4  wired
model.specf_patch                   8           unused  DEAD
model.subcenter_K                   3                3  wired
model.aux_head_hidden             128              128  wired
model.wl_embed_dim                 16           unused  DEAD
model.branch_drop_prob            0.2 stored, not used  DEAD
stage2.arcface_m0                0.18  see stage2 loop  n/a
```

### Verdict

**N-1a CONFIRMED** — `cross_interaction` runs **8** attention heads; the config says 4.
`SpectralQuadNet.__init__` constructs `CrossModalInteraction(num_modalities=4, d=256,
drop=...)` without `heads=`, so the signature default of 8 wins.

**N-1b CONFIRMED** — SpecFormer dropout is **0.10**; the config says 0.15. The branch is
constructed with a literal `dropout=0.10`.

**5 of 12 audited keys never reach the module they name.** `model.branch_drop_prob=0.2` is
stored as `self.branch_drop_prob` and then ignored: `forward` builds its own literal
`[0.0, 0.0, 0.30, 0.20]` drop vector (N-1d / T1-6). `wl_embed_dim` and `specf_patch` are
accepted and never read.

`scripts/check_config_roundtrip.py` passes on all of these, because it checks that every
key *has a home in the schema* — a strictly weaker claim than reaching the module.
**`test_config_keys_are_wired` (§4.3) is the highest-leverage test to add**, and this
probe is a working prototype of it.

---

## Appendix B — prediction register after Phase 0

| ID | Prediction | Status | Measured |
|---|---|---|---|
| **F-1** | Grouped split drops macro-F1 by 5–20 pts | **⛔ NOT TESTABLE as specified** | 73/90 classes have all cubes in one session; no session-disjoint split exists (0-H) |
| **F-2** | σ₃/σ₁ < 0.05 for >90 % of seeds | **✅ CONFIRMED** | 100.00 % of 6,036 seeds; max σ₃/σ₁ = 0.029; eff. rank 2.10 (0-D) |
| **F-3** | Band curve does not plateau at k = 40 | ⬜ not run (6 runs) | `band_selection_report.csv` rises monotonically to its k = 40 endpoint — the curve was never extended past the chosen k |
| **F-4** | max cos(Lₙ, Lₙ′) > 0.95 in every checkpoint | **❌ REFUTED (parameter)** / ✅ conclusion holds | parameter 0.75; post-block state 0.9998 (0-E) |
| **F-5** | ≥3 of {49,52,41,51,37} show Rᶜ ≪ Pᶜ | **✅ CONFIRMED** on the selected ckpt | 3/5 (S1, reference); 2/5 (S2, S3) (0-G) |
| **F-6** | 8-spatial ≥ 12-view | **❌ NOT SUPPORTED**, effect is zero | Δ = +0.0048 / +0.0030 / −0.0031; all CIs straddle 0 (0-B) |
| **F-7** | Stage 3 on test is within noise of Stage 1 | **✅ CONFIRMED** | Δ = −0.0001, CI [−0.0126, +0.0122], p = 0.96 (0-A) |
| **F-8** | ≥1 sub-centre per class has zero win rate | **❌ REFUTED (global)** / ✅ own-class | global 0/270 dead; own-class 179/270, 90/90 classes (0-I) |
| **F-9** | `cos(ĝ_A, ĝ_D)` in Stage 3 < 0.8 | **❌ REFUTED** (Tier 1) | 0.99996 at init, **0.9996** on `best_stage2.pth` with a real batch (T1-4) |
| **F-10** | BN buffers differ CPU vs accelerator | **✅ CONFIRMED, then fixed** (Tier 1) | the pre-fix pass ran different code per device by construction; post-fix CPU↔MPS agree to 1e-5 (T1-5) |

F-9 and F-10 were the two Appendix-B predictions Phase 0 did not cover: the §4.1 matrix
(0-A…0-J) contains no action for either. Both were folded into Tier 1 and are now answered —
see [T1-4](#t1-4-t1-10--stage-3s-sam-step--enginetrain_epochpy) and
[T1-5](#t1-5--batchnorm-re-estimation--enginecheckpointpy-dataloaderspy).

---

## What Phase 0 changes about the plan

1. **Stage 3 is not a regression.** F-7 confirmed. The Tier-1 note *"If 0-A shows Stage 3
   is not actually worse on test, T1-4/5/9 alone plausibly make Stage 3 the selected
   checkpoint"* is now live — **T1-4, T1-5, T1-9 are the highest-value Tier-1 items.**
2. **The noise floor is ±0.017 (95 % CI), SE 0.0085.** Every Tier-1 item and most of
   Tier 2 claim effects below one SE. Single-split A/B tests cannot resolve them; adopt
   repeated seeds or cross-validation before claiming any Tier-1/2 delta.
3. **T4-1 must be redesigned.** A session-disjoint grouped split does not exist for this
   dataset. Choose cube-disjoint, or restrict the session-disjoint analysis to the 17
   two-session classes. This is a blocker for §4.4's Tier-3 re-baseline.
4. **T1-1 and T1-2 are correctness-only.** Their measured effect is zero at this
   resolution. Do them for the invariant, not the number.
5. **FU-1's diagnosis needs updating.** The latent parameters did not stay at their
   σ = 0.02 scale, and they are not collapsed. The collapse is in the block outputs, so
   the fix and its regression test must target the post-block state.
6. **T2-9's validation criterion needs restating** over own-class win rate; the global
   count it names is already 0.
7. **Two new findings not in the taxonomy:** `MaskedSpectralECA`'s trained gate is a
   near-identity (1.000–1.033 against a designed 1.0–2.0×), and the archived
   `band_selection_report.csv` never extends past k = 40, so the elbow F-3 asks about is
   unverifiable from the recorded artifact alone.

---

## Artifacts

New scripts (read-only; no `src/` file was modified):

| Script | Actions |
|---|---|
| `scripts/phase0_eval_checkpoints.py` | 0-A, 0-B (+ fp32 TTA control, val predictions) |
| `scripts/bootstrap_ci.py` | 0-C |
| `scripts/phase0_probes.py` | 0-D, 0-E, 0-F, 0-G, 0-I, 0-J (`--only D,E`) |
| `scripts/phase0_group_audit.py` | 0-H |

Outputs under `outputs/phase0/` — **not tracked** (`.gitignore` excludes `outputs/`), so
these are local artifacts; the four scripts above are the reproducible record:

| File | Contents |
|---|---|
| `eval_results.json` | every 0-A / 0-B metric |
| `bootstrap_ci.json` | 0-C intervals and paired differences |
| `probe_results.json` | 0-D, 0-E, 0-F, 0-G, 0-I, 0-J |
| `group_audit.json`, `scan_table.csv`, `scan_id.npy` | 0-H; `scan_id.npy` is ready for P-1 / T4-1 |
| `preds/{split}_stage{n}_{variant}.npy` | 18 prediction arrays + 2 per-split target arrays |
| `final_eval_stage{1,2,3}/` | `final_evaluation`'s own artifacts, per checkpoint |
| `0A_0B_console.log` | full console output incl. per-class classification reports |
| `0C_console.log`, `0DEFGIJ_console.log`, `0H_console.log` | full console output for the rest |

Reproduce with:

```bash
python scripts/phase0_eval_checkpoints.py   # ~9 min on MPS
python scripts/bootstrap_ci.py              # seconds
python scripts/phase0_probes.py             # ~5 min on MPS
python scripts/phase0_group_audit.py        # seconds
```

---

# Tier 1 — correctness fixes ✅ COMPLETE

Gate satisfied: Phase 0 reports. All ten items are implemented, and every one is pinned by
a test that fails against the pre-Tier-1 behaviour rather than merely passing against the
new one.

| | |
|---|---|
| Suite | **233 passed**, 0 failed, 0 skipped (was 107) — `pytest tests/` |
| New test modules | 7, adding 127 tests |
| `ruff` / `black` / `mypy --strict` | clean over all 72 source files |
| `scripts/check_ast_no_op_move.py` | ✓ 129 identical, 45 declared, 16 new, **0 drift** |
| `scripts/check_config_roundtrip.py` | ✓ all 81 keys map 1:1 |
| Golden gates | forward logits, per-tensor SHA-256, Stage-1 epoch-1 loss/digests — all unchanged |

## Status by item

| ID | Change | Status | Measured effect | Claimed |
|---|---|---|---|---|
| **T1-1** | TTA foreground mean + re-mask | ✅ | −0.0046 test macro-F1 on the selected ckpt, CI [−0.0151, +0.0056] | +0.002 … +0.006 |
| **T1-2** | TTA forwards in fp32 | ✅ | folded into the T1-1 measurement; ≤0.0026 on the spatial-only arm | ±0.001 |
| **T1-3** | Focal modulates on unsmoothed `p_y` | ✅ | modulator range restored `[0.3955, 1]` → `[0, 1]` at ε=0.10; Stage 2/3 bit-identical | +0.001 … +0.003 |
| **T1-4** | SAM: identical objective both steps | ✅ | `cos(ĝ_A, ĝ_D) = 1.0` at ρ→0; **the mismatch it removes was 0.9996, not <0.8** | +0.003 … +0.008 |
| **T1-5** | BN pass: dropout off, natural prior | ✅ | σ² was inflated ~1.9× by dropout; buffers now device-independent to 1e-5 | +0.001 … +0.004 |
| **T1-6** | `model.branch_drop_prob` wired | ✅ | Stage 3 branch masking genuinely off; bit-identical at the shipped 0.20 | +0.001 … +0.003 |
| **T1-7** | ArcFace clamp `1e-6` → `1e-3` | ✅ | sine conditioning 6.6e-3 → 9.7e-6 rel. error; **grad norm falls 1.15×, not ≥10×** | ≈ 0 (stability) |
| **T1-8** | Record and honour `best_source` | ✅ | selection/evaluation mismatch now falsifiable; legacy bundles unchanged | ±0.003 |
| **T1-9** | Keep, score and compare the EMA shadow | ✅ | `val/f1_ema` in Stage-3 telemetry; sidecar gains `swa_val_f1`/`ema_val_f1` | +0.002 … +0.006 |
| **T1-10** | ProtoNCE applied | ✅ | `sched/proto_weight` now names a live term | +0.000 … +0.002 |

**Two of the ten predictions do not survive measurement** (details below): T1-4's premise
(F-9) and T1-7's validation criterion. Both fixes were kept — each is correct on its own
terms — but neither should be credited with the macro-F1 it was assigned.

---

## T1-1, T1-2 · TTA — `engine/tta.py`

The spectral view is now `spectral_view(x, s, mask=None)`: mean over the **foreground**,
result **re-masked**, and every forward inside `autocast(enabled=False)`.

### Measured on the three archived checkpoints (test split, MPS)

| ckpt | variant | before | after | Δ | 95 % CI | p |
|---|---|---|---|---|---|---|
| stage1 | no TTA | 0.8770 | 0.8770 | 0.0000 | — | — |
| stage1 | **12-view** | 0.8934 | **0.8889** | **−0.0046** | [−0.0151, +0.0056] | 0.40 |
| stage1 | 8-view spatial | 0.8886 | 0.8912 | +0.0026 | — | — |
| stage2 | 12-view | 0.8876 | 0.8827 | −0.0049 | [−0.0143, +0.0040] | 0.27 |
| stage2 | 8-view spatial | 0.8846 | 0.8862 | +0.0016 | — | — |
| stage3 | 12-view | 0.8868 | 0.8898 | +0.0030 | [−0.0064, +0.0126] | 0.50 |
| stage3 | 8-view spatial | 0.8900 | 0.8900 | −0.0000 | — | — |

10,000-resample paired bootstrap (2,000 for the CI columns), same protocol as 0-C. The
corrected transform changes **3.2–3.7 % of individual predictions** and no aggregate metric
outside noise.

### Verdict

**T1-1's +0.002…+0.006 is not observed; the point estimate on the selected checkpoint is
negative.** Phase 0 already re-scoped this item to "correctness fix with an expected effect
near zero" (0-B); the paired interval now confirms that at ±0.010. The invariant is the
reason to keep it: the background is exactly zero for all 12 views, the foreground fraction
is preserved, and the view is now the test-time analogue of the train-time `multiplicative`
augmentation instead of an off-manifold transform.

**The headline number in the paper changes: 12-view TTA on Stage 1 is 0.8889, not 0.8933.**
The `outputs/output_v12_spa40/test_preds_TTA.npy` artifact still evaluates to 0.8933 — it
was produced by the old transform — so the regression test that pins it still passes. Any
re-run will produce the lower number.

*Incidental:* with the transform corrected, 8-view spatial-only now **beats** 12-view on
both selected checkpoints (+0.0023 and +0.0035). That is the direction F-6 originally
predicted, which 0-B did not support under the old transform. Still inside noise, still not
a reason to claim it — but the 4 spectral views have now failed to earn their 50 % inference
cost under both transforms.

---

## T1-3 · Focal × label smoothing — `losses/focal.py`

One line: the modulator reads `softmax(z)_y`, not `exp(-ce_smoothed)`.

```
ε      H(q)     old modulator floor    new modulator at p_y = 1-1e-6
0.10   0.7738   0.3955                 < 1e-3
0.07   0.5678   0.2852                 < 1e-3
0.04   0.3475   0.1591                 < 1e-3
0.00   0.0      0.0  ✓                 < 1e-3
```

The floors are re-derived in `tests/unit/test_focal.py` and matched against §2.4.5's table
to 5e-4 — except the ε = 0.07 row, where the plan quotes `H(q) = 0.5686` against an exact
0.56784, a 5e-4 slip in the plan's own arithmetic, not in the loss.

**At ε = 0 the two forms are the same float**, so Stage 2 and Stage 3 are bit-identical;
`test_no_smoothing_path_is_bit_identical` asserts `==`, not `approx`. Only Stage 1 Phase 3
is affected — which is the phase that produced the shipped checkpoint, at ε(488) ≈ 0.051 and
a modulator floor of ≈0.20.

---

## T1-4, T1-10 · Stage 3's SAM step — `engine/train_epoch.py`

Both steps now evaluate one `_objective()`: focal + 0.02·SupCon + 0.01·ProtoNCE +
0.10·aux. `cos(ĝ_A, ĝ_D)` is logged as `sam/grad_cos` under the existing
`tracking.log_grad_norms` gate, and measures **1.0 within 1e-6** at ρ → 0.

Two fixes came with it:

* **ProtoNCE is applied (T1-10).** It was constructed, weighted at 0.01, logged as
  `sched/proto_weight` and never added to any loss.
* **A non-finite descent loss now calls `SAM.restore()` before skipping the batch.** The old
  guard skipped between `first_step` and `second_step`, so the weights stayed at
  `θ + ρĝ` and the *next* batch's `first_step` overwrote the cached originals — one bad
  batch baked the perturbation in permanently. Not in the plan's taxonomy; found while
  implementing T1-4.

### F-9 → **REFUTED**

The prediction was `cos(ĝ_A, ĝ_D) < 0.8`. Measured on the pre-fix pair of objectives:

| weights | batch | `cos(ĝ_A, ĝ_D)` | ‖ĝ_A‖ | ‖ĝ_D‖ |
|---|---|---|---|---|
| fresh init | synthetic | **0.99996** | — | — |
| `best_stage2.pth` | real, 16 patches | **0.9996** | 1010.2 | 975.7 |

§2.5.1 reasoned that the auxiliary heads' shallow, well-conditioned gradients would dominate
`ĝ_A` because the main path must traverse fusion + `EmbedNet` + the `s = 48` head. The
measurement inverts that: **`s = 48` amplifies the focal term too**, so the auxiliaries
contribute 3 % of the gradient's magnitude and almost none of its direction. The objective
mismatch was real and is worth removing — SAM is not defined for two objectives — but
**T1-4's +0.003…+0.008, the largest single Tier-1 estimate, has no measured basis**, and
neither does §2.5.1's claim that C-6 "alone can explain a large share of the Stage-3
regression".

---

## T1-5 · BatchNorm re-estimation — `engine/checkpoint.py`, `data/loaders.py`

Three defects, one fix each:

1. **Dropout off.** Every `nn.Dropout` *and* every `nn.MultiheadAttention` (whose dropout is
   a float attribute `set_dropout` cannot reach — N-2) is forced to `eval()` for the pass.
   Measured inflation on a structural stand-in at `p = 0.5`: **σ² was 1.9× too large**.
2. **Natural prior.** New `build_natural_prior_loader()` re-wraps the loader's dataset in a
   plain shuffled loader, dropping the CDWS-weighted `ClassBalancedBatchSampler`. Stage 3
   calls it. The old pass estimated statistics under a prior that over-represents hard
   classes up to 3× and applied them under the natural one.
3. **Device independence (N-12).** With nothing stochastic left, the pass is plain
   `no_grad()` on every device. `utils/device.py::no_grad_is_safe_for_dropout` — the
   Metal-only `enable_grad` workaround, and the *cause* of N-12 — is **deleted** rather than
   left inert; `tests/unit/test_device.py` still pins the upstream limitation it existed for,
   and now also pins that an `eval()`-mode attention forward under `no_grad` runs fine on MPS.

`test_bn_stats_device_independent` compares CPU against MPS buffer-for-buffer: **agrees to
1e-5**, §4.3's stated tolerance. `test_the_pass_is_reproducible` is the sharper statement —
two passes over the same data now give *bitwise* identical buffers.

---

## T1-6 · `model.branch_drop_prob` — `models/spectral_quadnet.py`

`forward` now computes `self.branch_drop_prob * BRANCH_DROP_PROFILE` where the profile is
the module constant `(0.0, 0.0, 1.5, 1.0)`, and skips the masking block entirely — drawing
no RNG — when the probability is 0.

**At the shipped `0.20` the scaled vector is bit-identical to the literal it replaces**
(`0.2 * float32(1.5) == float32(0.30)` exactly), and consumes the same `torch.rand` draws.
That is what keeps the Stage-1 golden loss and post-step SHA-256 digests valid, and the test
asserts `torch.equal`, not `allclose`.

The behavioural change is Stage 3, which sets `branch_drop_prob = 0.0` at entry and until
now still dropped branch C on 22.5 % of batches and branch D on 15 %.

The profile's *shape* is deliberately unchanged. M-5 (§2.2.7) argues it suppresses the wrong
branches; that is a Tier-2/3 question about which branches carry unique information, not a
wiring defect, and `test_the_profile_never_drops_the_spectral_branches` pins the current
shape so changing it has to be deliberate.

`tests/unit/test_branch_drop.py` also carries the Tier-1 slice of §4.3's
`test_config_keys_are_wired`: it pins the four keys 0-J found dead
(`fusion_heads`, `specf_drop`, `specf_patch`, `wl_embed_dim`) **as still dead**, asserting
the defect. Each assertion fails the day its Tier-2/3 item lands, which is the prompt to
move the key into the wired list.

---

## T1-7 · ArcFace cosine clamp — `models/heads.py`

`COS_CLAMP_EPS = 1e-3`, read as a module global so tests can vary it.

### The validation criterion → **REFUTED**

T1-7 asks for "max observed head grad-norm falls by ≥10×". Measured: **1.15×**. The reason
is algebraic, and it also refutes N-4's `3.4 × 10⁴` amplification claim. §2.4.4 reads the
derivative off the sine in isolation:

$$\left|\frac{\partial}{\partial c}\sqrt{1-c^2}\right| = \frac{|c|}{\sqrt{1-c^2}} \le 707 \text{ at } c = 1-10^{-6}$$

but the head normalises **both** its embedding and its weights, so what reaches them is the
*tangential* gradient, and the tangential factor vanishes at exactly the rate the sine
derivative diverges. Since $\phi = \cos(\theta + m)$,

$$\frac{\partial (s\phi)}{\partial \theta} = -s\sin(\theta+m), \qquad \left|\cdot\right| \le s$$

with no singularity anywhere. `test_head_gradient_is_bounded_by_s` measures that identity to
1e-4 at both clamps, on the real head, for every sample the clamp admits.

### The case that does survive: conditioning

| clamp | exact `sqrt(1-c²)` | fp32 | rel. error |
|---|---|---|---|
| `1 - 1e-6` | 1.41421e-3 | 1.42357e-3 | **6.6e-3** |
| `1 - 1e-3` | 4.47102e-2 | 4.47097e-2 | **9.7e-6** |

`1 - c²` at `c = 1 - 1e-6` is a difference of two numbers agreeing to six digits; fp32 keeps
about two of the result's. The margin is built from that sine, so the samples the old clamp
admitted contributed gradients whose *magnitude* was right and whose value was noise. The
new clamp retires exactly those samples — `clamp` has zero derivative outside its range —
and every sample it still admits matches the exact identity to 1e-3.

Angular cost: `arccos(1 - 1e-3) = 2.56°` against margins of 20–26°. No prediction changes;
`test_inference_logits_barely_move` asserts identical argmax.

**Keep the fix, drop the criterion.** The plan's own Δ column already says "≈ 0
(stability)", which is the accurate description.

---

## T1-8 · Selection–evaluation mismatch — `engine/stages/*`

Stages 1 and 2 now record `best_source ∈ {live, ema}` next to `val_f1` (ties keep `ema`, the
historical choice); Stage 3 records `best_source ∈ {swa, ema}` plus both scores.
`final_evaluation` evaluates the weights that name says, and prints which.

**Bundles without the key default to `ema`** — the pre-Tier-1 behaviour, which is what every
archived checkpoint was produced under, so `outputs/output_v12_spa40/` keeps reproducing its
recorded numbers. An unrecognised value falls back to `ema` with a warning rather than
crashing.

The mismatch this closes was not measurable from the artifacts, because they never recorded
which of the two won — that unfalsifiability *was* the defect (§2.1.4). It becomes
measurable on the next run.

---

## T1-9 · Stage 3 keeps its EMA shadow — `engine/stages/stage3_sam_swa.py`

The shadow is re-initialised at stage entry, updated every optimiser step (via
`train_one_epoch_sam`'s new `ema` argument), scored every epoch as `val/f1_ema` — the
plan's stated criterion — and compared against the SWA average at the end. The better of the
two goes into the bundle's `ema` slot; `best_source`, `swa_val_f1` and `ema_val_f1` all reach
the sidecar, and the stage returns `max(f1_swa, f1_ema)`.

This also repairs the bundle's semantics: `ema` meant "EMA of the trajectory" for Stages 1–2
and "SWA average" for Stage 3, unconditionally and unmarked. It is now whichever won, and
the sidecar says which.

Two `@pytest.mark.slow` tests run a real two-epoch Stage 3 on synthetic data end to end and
assert the telemetry keys, the sidecar fields and that the `ema` slot holds the recorded
winner.

---

## Test inventory

| Module | Tests | Pins |
|---|---:|---|
| `tests/unit/test_tta.py` | 22 | T1-1, T1-2 — zero background for all 4 scales and all 12 views, foreground fraction preserved, foreground-mean invariance, fp32 under an enclosing autocast |
| `tests/unit/test_focal.py` | 55 | T1-3 — modulator < 1e-3 at `p_y = 1-1e-6` for every ε × γ, the entropy floor it replaces, ls=0 bit-identity |
| `tests/unit/test_arcface_head.py` | 11 | T1-7 — the 707/22.4 bound, the `s·sin(θ+m)` identity, the dead zone, fp32 conditioning, unchanged argmax |
| `tests/unit/test_branch_drop.py` | 10 | T1-6 — bit-identical drop vector, deterministic and RNG-free at 0, config-wiring inventory |
| `tests/unit/test_bn_stats.py` | 11 | T1-5 — reproducibility, eval-mode agreement, dropout inflation, CPU↔MPS to 1e-5, natural-prior loader |
| `tests/unit/test_stage3_sam.py` | 9 | T1-4, T1-9, T1-10 — `cos = 1`, F-9's refutation, `SAM.restore`, ProtoNCE live, EMA scored and compared |
| `tests/unit/test_best_source.py` | 9 | T1-8 — sidecar field, live/ema/swa routing, legacy default, unknown-value fallback |

Each defect test has a companion asserting that the **pre-Tier-1 behaviour fails it**
(`test_the_old_spectral_view_broke_that_invariant`, `test_the_old_modulator_could_not_reach_zero`,
`test_a_train_mode_pass_would_have_inflated_the_variance`,
`test_the_old_clamp_admitted_numerically_meaningless_gradients`), so none of them can pass
vacuously.

---

## Gate maintenance

`scripts/check_ast_no_op_move.py` proves each relocated symbol is AST-identical to the
pinned pre-refactor reference. Tier 1 changes numerics **by design**, so four symbols
(`tta_predict`, `FocalLoss.forward`, `SpectralQuadNet.forward`,
`AdaptiveSubcenterArcFaceHead.forward`) newly drift and five more
(`update_bn_stats`, `train_one_epoch_sam`, `run_stage1/2/3`, `final_evaluation`) needed their
declarations rewritten. The gate's contract was widened accordingly: a `DECLARED` entry is
now either a **relocation** (numerics untouched, as every entry was through Phase 5) or a
**correctness fix** carrying its plan item id and the test that pins it. Result: 129
identical, 45 declared, 16 new, **0 drift**.

---

## What Tier 1 changes about the plan

1. **Two claimed effects are refuted by measurement, not by argument.** F-9 (T1-4's premise)
   measures 0.9996 where <0.8 was predicted; T1-7's ≥10× grad-norm criterion measures 1.15×
   and cannot be met, because the normalisation cancels the singularity N-4 is built on.
   Both fixes stand on correctness. **Tier 1's `+0.010…+0.032` subtotal should be read as
   `≈ 0` until a repeated-seed run says otherwise** — it was already entirely below one
   bootstrap SE (0-C), and its two largest line items are now unsupported.
2. **The reported TTA number moves down.** 0.8933 → 0.8889 on the selected checkpoint. The
   correct transform is worth having; the number it produces is the one to publish.
3. **§4.2's "If 0-A shows Stage 3 is not actually worse on test, T1-4/5/9 alone plausibly
   make Stage 3 the selected checkpoint" is weakened.** T1-4's mechanism is measured to be
   near-inert. T1-5 and T1-9 remain plausible — both change what Stage 3 *saves*, not how it
   trains — and are untested until a real Stage-3 run.
4. **A new defect was found and fixed:** a non-finite Stage-3 descent loss left the SAM
   ascent perturbation permanently in the weights. Not in the taxonomy; add it under §2.5.
5. **T1-4/T1-5/T1-9's real value is a run, not a diff.** Every one of them changes Stage 3's
   trajectory or its checkpoint selection, and none can be evaluated without retraining
   Stage 3 from `best_stage2.pth`. That run is the natural next step, and it is also what
   §4.2 needs before Tier 2's Stage-3 items (T2-1…T2-3) have a baseline to move.

## Reproducing

```bash
pytest tests/                             # 233 passed
python scripts/check_ast_no_op_move.py    # 0 drift
python scripts/check_config_roundtrip.py  # 81/81
ruff check src tests scripts && black --check src tests && mypy
```

---

# Tier 2 — optimisation and curriculum ✅ COMPLETE

All ten items are implemented. Every one is pinned by a test that fails against
the pre-Tier-2 behaviour rather than merely passing against the new one, and
seven carry a companion test that *measures the defect* — the greedy filter's
tautology, the F1 margin rule's inverted sign, the dead sub-centres the old
initialisation leaves, raw SAM's over-allocation to the head.

| | |
|---|---|
| Suite | **371 passed**, 0 failed, 0 skipped (was 233) — `pytest tests/`, 8 m 40 s |
| New test modules | 8, adding 99 tests; 4 existing modules extended |
| `ruff` / `black` / `mypy --strict` | clean over all 72 source files |
| `scripts/check_ast_no_op_move.py` | ✓ 118 identical, 56 declared, 33 new, **0 drift** |
| `scripts/check_config_roundtrip.py` | ✓ all 81 keys map 1:1; 15 declared additions, 3 declared value changes |
| `scripts/capture_golden.py --verify` | ✓ v1 still reproduces the baseline; v2 written and verified |
| Checkpoint schema | **v1 → v2** — `linear_head` removed, `arcface_head.confusion` added |

## Status by item

| ID | Change | Status | Measured effect | Claimed |
|---|---|---|---|---|
| **T2-1** | Per-class margins kept, annealed by κ, frozen per cycle | ✅ | margin vector survives Stage 3 (was replaced by one scalar at entry); κ steps only at cycle boundaries | +0.003 … +0.008 |
| **T2-2** | Greedy SWA evaluates the candidate average | ✅ | `swa/n_rejected > 0` on an 8-cycle run; **the old filter was provably a no-op** | +0.002 … +0.005 |
| **T2-3** | First 3 cycles discarded from SWA | ✅ | `swa_first_accepted_cycle = 4` at the shipped warm-up | +0.001 … +0.004 |
| **T2-4** | ASAM (`T_θ = diag\|θ\|`) | ✅ | head budget 2.29× → **0.038×** its parameter share; but §2.5.8's premise holds only at init | +0.002 … +0.006 |
| **T2-5** | Per-group gradient clipping | ✅ | a saturated head leaves the backbone bit-identical; the global clip rescaled it | +0.001 … +0.004 |
| **T2-6** | GradNorm ω from `grad_norm/branch_*` | ✅ | ω is a logged time series; branch norms converge to within 5 % of balanced | +0.002 … +0.005 |
| **T2-7** | Same-class spectral + spatial CutMix | ✅ | label-preserving; the ArcFace guard is untouched; zero extra RNG when off | +0.004 … +0.012 |
| **T2-8** | Signed `R−P` margin rule + pairwise confusion term | ✅ | `M(c)` falls with recall (the old rule raised it); `M(c)` non-monotone in F1 | +0.004 … +0.010 |
| **T2-9** | Soft-to-hard sub-centres + balance term + k-means init | ✅ | dead sub-centres 0 → 0; π entropy > 0.9 log K; **plain k-means was not enough** | +0.003 … +0.008 |
| **T2-10** | One cosine head for all three stages | ✅ | 89 of 90 logits identical across the Stage-1 → Stage-2 boundary | +0.005 … +0.015 |

**Nothing in this tier has been trained.** Every Δ column above is the plan's
estimate, not a measurement — Tier 2 changes how the model *optimises*, and
none of it is observable without a full three-stage run. What is measured is
the mechanism in each case, and two of those measurements do not support the
estimate attached to them (T2-4 and, indirectly, T2-2 — see below).

---

## T2-1, T2-2, T2-3 · the Stage-3 rewrite — `engine/stages/stage3_sam_swa.py`

### T2-1 · the margin is a vector again, and it is frozen inside a cycle

Stage 3 passed `arc_m = 0.25 + 0.05·cos(πe/E)`. `arc_m` is an **override**: a
float makes the head ignore its `margins` buffer for every sample in the batch,
so Stage 2's 90-value calibration was discarded on Stage 3's first epoch
(C-7a), and it was replaced by something that moved every epoch, so no two SWA
snapshots were iterates of one objective (C-7b).

Now: `base_margins` is captured at entry, `stage3_margin_kappa` returns κ for
the epoch, and `margins ← base_margins · κ` is written every epoch but only
*changes* at a cycle boundary. `arc_m=None` is passed, so the vector is what the
head reads. κ runs 1.0 → 0.85 linearly across cycles — **exactly 1.0 on the
first cycle**, which is what "preserves Stage 2's calibration" has to mean.

New telemetry: `sched/margin_kappa`, and `sched/arcface_margin` now reports the
vector's mean rather than a scalar schedule.

### T2-2 · the greedy filter was a tautology

The rejection test read, in this order:

```python
best_live_f1 = max(best_live_f1, f1_live)
...
if not cfg.stage3.greedy or f1_live >= best_live_f1 * 0.98:
```

After the `max`, `best_live_f1 >= f1_live` always, and the surviving reject case
needs `f1_live` to be **more than 2 % below the best ever seen** — which the
`max` has just guaranteed it is not, whenever `f1_live` is itself the best.
`test_the_old_acceptance_test_was_a_tautology` runs the arithmetic on a
deliberately non-monotone F1 sequence: 0 rejections.

Acceptance now forms the candidate average `((n−1)/n)·θ̄ + (1/n)·θ⁽ⁿ⁾` as a
**new dict**, evaluates it on a scratch `probe` model, and keeps it only if it
beats the average it would replace. `_blend` is separated out precisely so the
candidate can be scored before it becomes the average. On an 8-cycle synthetic
run, `swa/n_rejected > 0`.

**A consequence worth stating: T2-2's `+0.002…+0.005` assumed the greedy filter
was doing something and doing it wrong. It was doing nothing.** The change is
from "average everything" to "average what helps", which is a larger behavioural
change than the plan's framing suggests, and it is untested on real data.

### T2-3 · the first cycles are discarded

`swa_warmup_cycles = 3`, the plan's `⌈(1/(1−β₂))/N_steps⌉`. Cycles 1–3 are
skipped before any candidate is considered, and `swa_first_accepted_cycle`
reaches the sidecar so the criterion ("first accepted snapshot index ≥ 4") is
checkable after a run rather than argued. A stage shorter than the warm-up
accepts nothing and falls back to the final live model, as before.

The plan's alternative — warm-starting Adam's moments from Stage 2's optimiser
state — is not implementable here: Stage 2 does not persist optimiser state,
and adding it to the bundle is a second schema change for no measured gain.

---

## T2-4 · ASAM — `optim/sam.py`

`first_step` becomes `ρ·θ²g/‖θg‖` under `adaptive`, and `_grad_norm` matches its
denominator. A param group carrying `perturb: False` is skipped entirely —
OP-5's stated alternative, kept available and tested but not wired to a config
key, since ASAM is the implementation chosen.

`SAM.perturbation_mass()` is new: it returns ‖ε_group‖² per group **without
moving a weight**, which is what makes the criterion measurable at all.

### The measurement — §2.5.8's premise is initialisation-only

Head share of ‖ε‖², against its 0.88 % share of the parameters:

| weights | raw SAM | ASAM | ratio to parameter share |
|---|---|---|---|
| fresh init | 2.01 % | 0.034 % | 2.29× → **0.038×** |
| `best_stage2.pth` | 0.30 % | 0.0008 % | 0.34× → **0.0009×** |

T2-4's criterion ("perturbation mass on `arcface_head` falls below its parameter
share") is met comfortably, in both regimes. But **§2.5.8's claim that `s = 48`
makes the head consume a disproportionate budget is only true before the head is
fitted.** On `best_stage2.pth` — the weights Stage 3 actually starts from — raw
SAM already puts the head at *one third* of its parameter share. The defect
ASAM is aimed at is not present at the point Stage 3 runs, so
**T2-4's +0.002…+0.006 has no measured basis**, exactly as T1-4's did not.

ASAM is kept on its own terms: a ρ-ball in raw parameter space is undefined for
a module whose loss satisfies `L(cW) = L(W)`, which the normalised head does.
`test_asam_is_invariant_to_a_parameter_rescaling` measures that property
directly and shows raw SAM failing it.

---

## T2-5 · per-group gradient clipping — `optim/param_groups.py`, both loops

`clip_grad_norm_by_group` splits the model into `head` / `fusion` / `backbone`
by name prefix and clips each independently, returning the three **pre-clip**
norms as device tensors — so the loops log `grad_norm/preclip_{head,fusion,backbone}`
for free, at no extra synchronisation.

The criterion is measured as a decoupling rather than a ratio: with the head's
gradient set 10× over the threshold and the backbone's 10⁷× under it, the
backbone's gradient after `clip_grad_norm_by_group` is **`torch.equal`** to what
it was, and after the single global `clip_grad_norm_(model.parameters(), 1.0)`
it is not. The partition is checked too — every trainable parameter lands in
exactly one group, so three clips cover what one did.

---

## T2-6 · GradNorm auxiliary weights — `losses/auxiliary.py`

`ω_b ← ω_b·(ḡ/g_b)^α`, α = `cfg.aux_gradnorm_alpha` = 0.5, applied **once per
epoch** from the per-branch norms the loops already sample. `α = 0` is a true
no-op that reproduces `DEFAULT_BRANCH_WEIGHTS` bit-for-bit, so "GradNorm off"
remains an honest control.

Two implementation notes that are deviations worth recording:

1. **`g_b` is the branch's total gradient norm, not `‖∇_{θ_b} ω_b L_b‖`.** The
   plan writes the latter but then says the quantity "is **already computed and
   logged** by `branch_grad_norm_tensors`" — which computes the former. Isolating
   the auxiliary term's own gradient needs four extra backward passes per step.
   The plan's own identification is followed; the discrepancy is noted here
   because the two differ by the main path's contribution to each branch.
2. **The weights are bounded to [0.25, 4.0].** One epoch in which a branch's
   gradient is near zero would otherwise send its weight to infinity, and the
   loop does not recover. Not in the plan; a stability requirement of closing
   any multiplicative feedback loop.

`aux_weight/branch_{a,b,c,d}` is logged every epoch — T2-6's "ω becomes a logged
time series". The convergence half of the criterion is measured on a model where
`g_b ∝ ω_b·s_b`: 30 updates bring the spread of the branch norms to within 5 %
of balanced from a 12× starting ratio.

---

## T2-7 · same-class CutMix — `data/datasets.py`

Two operators, both label-preserving because the partner is drawn from the
anchor's own class **and own split**: `_spectral_cutmix` swaps a contiguous
`cutmix_bands`-wide wavelength window, `_spatial_cutmix` pastes a
`cutmix_spatial`-square region across every band at once, so the pasted region
keeps a physically coherent spectrum rather than a per-band mixture.

Because the label never changes, **no `losses/` change was needed**. T2-7 lists
`losses/` as a target file; the entry that would have required it is OP-6's
manifold-mixup alternative, which is not the one §4.2 names. T2-7's criterion is
that "the ArcFace `ValueError` guard is untouched", and it is — the guard moved
from "is this the ArcFace head" to "is the margin non-zero" for HD-1's reasons,
and `test_the_arcface_mixup_guard_is_untouched` pins that it still fires for
mixup while CutMix never reaches it.

**RNG discipline.** The two new guards test `p[...] > 0.0` *before* drawing, so
a profile with CutMix disabled consumes no randomness and reproduces the
pre-Tier-2 augmentation stream exactly — the same pattern T1-6 used for the
branch-drop block. The partner draw is uniform over the class minus the anchor
in a single `randint`, with no rejection loop.

All four active profiles enable it (`heavy` 0.10 → `light` 0.06). Those
probabilities are a choice, not a derivation: §3.6 OP-6 specifies the operators
and not their strengths.

---

## T2-8 · the signed margin rule — `models/heads.py`, `engine/stages/stage2_arcface.py`

`update_margins_from_f1` is **deleted**, not deprecated, and replaced by

$$M(c) = \operatorname{clip}\big(m_{\text{base}} + m_\Delta(R_c - P_c),\ 0.20,\ 0.50\big)$$

Three things follow, and each has a test:

* **The sign.** §4.3's `test_margin_rule_sign`: at fixed precision, `M(c)`
  *falls* as recall falls. The companion re-derives the old rule's value on the
  same three classes and shows it rising — the feedback loop of §2.4.1.
* **Non-monotonicity in F1** (T2-8's stated criterion). Two classes constructed
  to share an F1 of 0.45 and differ in which half is weak get margins on
  opposite sides of `m_base`. Any F1-driven rule gives them the same number.
* **The pairwise term.** `arcface_head.confusion` holds the row-normalised
  confusion matrix, diagonal zeroed; the non-target logits are reduced by
  `δ·Ω[y, c]`. A class confused with 1 and never with 3 has class 1's logit
  pushed down and class 3's left exactly where it was. Zero-filled — the state
  a fresh head and every migrated v1 checkpoint is in — the term is identically
  zero, and it never reaches inference.

The `θ + m < π/2` cap is applied under `no_grad` (it decides *which* margin
applies; differentiating through `acos` would re-introduce the `1/√(1−c²)`
factor HD-4 exists to bound). It subsumes the `phi = cos θ − mm` easy-margin
guard, which only opened past `π − m` and is now unreachable.

**Calibration cadence: once, at stage entry** — from a fresh
`evaluate_pr_and_confusion` pass over the validation split. Not per epoch. 270
fitted parameters read off the split that also selects the checkpoint is exactly
the contamination §2.1.4/C-9 describes, and P-5 (T4-5) is the item that gives
them their own split; recalibrating every epoch would multiply that leak rather
than reduce it. The cadence therefore matches the pre-Tier-2 one, and only the
*rule* changed.

`m_delta`'s YAML value moves 0.10 → 0.20, the value §3.5 specifies for the new
rule. The key is reused rather than duplicated so there stays one margin-scale
knob; the change is declared in `INTENDED_VALUE_CHANGES`.

---

## T2-9 · sub-centres — `models/heads.py`

**(i) Soft-to-hard.** `cos θ_{i,c} = τ·logsumexp(cos_k/τ)`, τ annealed 0.20 →
0.02 by `subcentre_tau` over each stage's epochs, held at the final value in
Stage 3. At `τ ≤ 0` the pooling is the hard `max_k` the head was defined on, so
the anneal has an exact endpoint. The log-sum-exp exceeds the max by up to
`τ log K` (0.22 at τ = 0.2, K = 3), which can leave the valid cosine range — the
result is re-clamped, or the margin's `√(1−c²)` is a NaN two lines later.

**(ii) The balance term.** `Σ_c KL(π_c ‖ 1/K)` at weight 0.01, produced by the
model's forward as `out["balance"]` (the head has already formed the assignment;
recomputing it in the loops would be a second forward) and added by both epoch
loops. Summed over the classes *present in the batch* — π is undefined for the
other 74 of 90 under a 16×8 balanced batch.

**(iii) k-means seeding.** `init_from_linear` is deleted. Spherical k-means on
the Stage-1 embeddings replaces it, and `train.py`'s "Bootstrapping ArcFace from
linear head" step becomes a re-seeding on an un-augmented, un-shuffled pass over
the train split.

### Plain k-means was not sufficient

The first implementation seeded centres from `randperm` rows and reliably
produced one dead sub-centre in six classes: two centres inside one mode, a
third mode uncovered, and Lloyd's algorithm cannot escape that. `_spherical_kmeans`
now seeds **k-means++**-style (D² sampling on cosine distance), after which
§4.3's `test_no_dead_subcentres` passes and the π entropy clears `0.9 log K`.
Not in the plan, and it is the difference between the item working and not.

The companion measures the defect: the `0.01·k` scheme — a 9° and an 18°
rotation at `d = 256`, §2.4.3's figures — leaves sub-centres that win nothing,
and at `τ = 0` their gradient is exactly zero, so they cannot recover.

**HD-2(iv) — per-class `K` by silhouette or eigengap — is deliberately not
implemented.** §4.2's T2-9 lists (i)–(iii) only, and a ragged `K` changes the
weight matrix's shape, i.e. a third checkpoint schema in one tier.

---

## T2-10 · one head for all three stages — schema v1 → v2

`linear_head`, `use_arcface`, `freeze_head`/`unfreeze_head` and
`init_from_linear` are removed. Stage 1 runs the sub-centre head at
`cfg.stage1.arcface_m = 0.0`, which the head short-cuts to the plain scaled
cosine — a NormFace classifier.

The Stage-1 → Stage-2 boundary is now continuous, which is the property T2-10's
criterion ("Stage 2's epoch-1 val F1 ≥ Stage 1's final") follows from and the
only half of it that is testable without a two-stage run:
**89 of the 90 logits are bit-identical across the transition**, and the 90th
differs by the margin warm-up's `arcface_m0 = 0.18 rad`.

The mixup exclusion moved with it. `train_one_epoch` used to raise on
`model._use_arcface and use_mixup`; it now raises on a **non-zero margin** plus
mixup, because under one head "is this ArcFace" has no answer, and a zero margin
takes interpolated targets perfectly well. Stage 1 Phases 1–2 keep mixup.

### The schema migration

`SCHEMA_VERSION = 2`, written into every bundle. `remap_state_dict` upgrades a
v1 state dict by dropping `linear_head.*` and zero-filling
`arcface_head.confusion`, and `load_ckpt` applies it automatically. The three
archived `output_v12_spa40` checkpoints still load `strict=True`, and
`test_migration_touches_only_the_declared_keys` asserts the stronger property:
every one of the 349 surviving tensors is `torch.equal` to what the bundle held.
`use_arcface` stays in the bundle as a constant `True` so existing readers
(`scripts/phase0_*.py`, the resume banner) keep working.

### The golden gates were re-scoped, and here is exactly how far

Removing `linear_head` removes its `nn.Linear`'s two construction draws from the
global RNG stream, and `_init_weights` runs *after* that point — so **146 of the
350 initialised tensors are drawn from a shifted stream**. No value comparison
against the v1 goldens is meaningful any more.

* `tests/regression/golden/` (v1) is **unchanged and still verified**:
  `capture_golden.py` re-runs the pinned reference implementation on every
  invocation and checks it still reproduces those files bit-for-bit.
* `tests/regression/golden/v2/` is new, captured from the current code, and is
  what the live drift gate now compares against. Its README states its
  provenance explicitly — it is *not* a baseline-equivalence artifact.
* Two new tests keep the change honest rather than assumed:
  `test_schema_v2_is_v1_minus_the_linear_head` asserts the key delta is exactly
  `−{linear_head.2.weight, linear_head.2.bias}, +{arcface_head.confusion}`, and
  `test_the_tensors_that_consume_no_randomness_did_not_move` asserts the 204
  tensors written by `ones_`/`zeros_` — which an RNG offset cannot reach — still
  hash to their v1 values. Had HD-1 changed anything structural below the head,
  those would have moved too.
* `capture_golden.py::check_schema_delta` enforces the same key delta at capture
  time, so a second structural change cannot ride along with HD-1.

---

## Test inventory

| Module | Tests | Pins |
|---|---:|---|
| `tests/unit/test_stage3_swa.py` | 12 | T2-1, T2-2, T2-3 — margin vector survives and is frozen per cycle, `_blend` is a running mean and returns a new dict, `n_rejected > 0`, first accepted cycle = 4, the old filter's tautology |
| `tests/unit/test_asam.py` | 8 | T2-4 — head budget below its parameter share, raw SAM above it at init and *below* it on trained weights, scale-invariance, `perturb: False`, the mass measurement moves nothing |
| `tests/unit/test_grad_clip_groups.py` | 7 | T2-5 — the partition is total, a saturated head leaves the backbone `torch.equal`, the global clip does not, each group clipped to the threshold independently |
| `tests/unit/test_gradnorm_aux.py` | 12 | T2-6 — the formula, the sign, α = 0 as an exact no-op, bounds, convergence of the norms, `aux_weight/*` telemetry, `_compute_aux_loss`'s default is unchanged |
| `tests/unit/test_cutmix.py` | 15 | T2-7 — window/region geometry, same-class and never-self partners, uniformity over the class, split scoping, label preservation, the ArcFace guard, zero RNG when off |
| `tests/unit/test_margin_rule.py` | 15 | T2-8 — §4.3's `test_margin_rule_sign`, the old rule's opposite sign, F1 non-monotonicity, clip range, Ω row-normalisation, the pairwise term's aim, the π/2 cap |
| `tests/unit/test_subcentres.py` | 19 | T2-9 — §4.3's `test_no_dead_subcentres`, the legacy scheme's dead ones, entropy > 0.9 log K, τ endpoints, gradient reaching every sub-centre, the balance term, k-means recovery and reproducibility |
| `tests/unit/test_unified_head.py` | 11 | T2-10 — one head, the removed API, zero margin ≡ cosine, the fast path ≡ the algebra, the continuous stage boundary, the head trains from Stage 1 |

Extended: `test_schedulers.py` (+11 — κ frozen within a cycle, κ endpoints, the
single-cycle degenerate case, τ annealing, and the baseline `_s3_margin`'s
per-epoch movement), `test_branch_drop.py` (+2 — the config-wiring inventory now
covers all 13 `model.*` keys and asserts none escapes classification),
`test_state_dict_compatibility.py` (+9 — the v1 → v2 migration),
`test_golden_forward_pass.py` (+2 — the schema delta and the unmoved tensors).

Every §4.3 test that belongs to Tier 2 is now present:
`test_no_dead_subcentres` (T2-9), `test_margin_rule_sign` (T2-8), and
`test_config_keys_are_wired` as an inventory over `cfg.model` (T2-9 wired three
keys; the four dead ones are all Tier-3 items, so the list was never going to
shorten here, and that is asserted rather than assumed).

---

## Gate maintenance

`check_ast_no_op_move.py` gained a third declaration kind. Through Tier 1 a
`DECLARED` entry was either a **relocation** or a **correctness fix**; Tier 2
adds **removal**, because T2-10 deletes a classification head and the API that
selected it, and a deletion is the one change a diff cannot show. Five symbols
are declared removed (`use_arcface`, `freeze_head`, `unfreeze_head`,
`init_from_linear`, `update_margins_from_f1`), each naming its plan item and
what replaced it. Result: 118 identical, 56 declared, 33 new, **0 drift**.

`check_config_roundtrip.py` gained 13 `INTENDED_ADDITIONS` and one
`INTENDED_VALUE_CHANGES` entry (`s2_arcface_m_delta`, 0.10 → 0.20). All 81
reference keys still map 1:1.

---

## What Tier 2 changes about the plan

1. **T2-2's premise understates the change.** The greedy filter was not
   mis-tuned, it was inert: `f1_live >= max(best, f1_live) * 0.98` cannot reject.
   Stage 3 has therefore always been plain SWA over every cycle-end snapshot,
   and T2-2 converts it to greedy SWA for the first time — a larger change than
   "evaluate the average instead of the candidate" suggests, and one whose sign
   on real data is unknown.
2. **T2-4's mechanism is absent at the point Stage 3 runs.** Raw SAM
   over-allocates to the head by 2.29× at initialisation and *under*-allocates by
   0.34× on `best_stage2.pth`. ASAM is still the right construction for a
   scale-invariant head, but **its +0.002…+0.006 should be read as ≈ 0** until a
   run says otherwise. This is the third plan estimate to fail measurement, after
   F-9 (T1-4) and T1-7's grad-norm criterion.
3. **Two items needed additions the plan does not specify, and both are
   load-bearing.** k-means++ seeding (without it T2-9's own criterion fails) and
   bounds on the GradNorm weights (without them one bad epoch is unrecoverable).
   Recorded so they are not read as gold-plating.
4. **The reported architecture changed.** The model has 13 top-level modules,
   not 14, and one classification head, not two. `README.md`'s overview is
   corrected; `docs/01`, `docs/03` and `docs/04` are **not**, and still describe
   the two-head design — `linear_head` in the block diagram (`docs/01` §93),
   `use_arcface` in the control-API table and `init_from_linear` in the
   sub-centre section (`docs/03` §30, §389–398), and the per-stage head row plus
   the Stage-2 bootstrap step in the curriculum table (`docs/04` §11, §94–97).
   `docs/03`'s parameter table is now 23,130 too high. **Updating those four
   documents is required follow-up work that this tier did not do**; it was left
   out deliberately rather than missed, because a rewrite of the architecture
   chapters belongs with the re-baseline below, not ahead of it.
5. **Tier 2's `+0.027…+0.077` is unverified in full.** Unlike Tier 1, nothing
   here is measurable from the archived artifacts — every item changes a training
   trajectory. The estimate should be treated as a hypothesis until a full
   Stage-1→3 run exists on schema v2, which is also what §4.4 sequences next.
6. **A re-run is now mandatory, not optional.** The archived checkpoints are
   schema v1 and load through a migration, but they were produced by a different
   architecture: Stage 1 trained a linear head that no longer exists. Any number
   reported from `outputs/output_v12_spa40/` describes the old model. The
   re-baseline §4.4 places before Tier 3 is the natural next step, and it now has
   to come *before* any further claim about Tier 2's effect.

## Reproducing

```bash
pytest tests/                             # 371 passed
python scripts/check_ast_no_op_move.py    # 0 drift
python scripts/check_config_roundtrip.py  # 81/81
python scripts/capture_golden.py --verify # v1 vs the baseline, v2 vs the tree
ruff check src tests scripts && black --check src tests && mypy
```

---

# Tier 4 — protocol ✅ COMPLETE

All six items are implemented. Tier 4 changes what the reported number *means*,
so unlike Tiers 1–2 the deliverable is not a gain: it is a protocol whose
claims are checkable, plus the machinery to run the P-fix re-baseline §4.4
sequences before Tier 3.

| | |
|---|---|
| Suite | **469 passed**, 0 failed, 0 skipped (was 371) — `pytest tests/`, 8 m 33 s |
| New test modules | 5, adding 98 tests |
| `ruff` / `black` / `mypy --strict` | clean over all 74 source files |
| `scripts/check_ast_no_op_move.py` | ✓ 118 identical, 56 declared, 33 new, **0 drift** |
| `scripts/check_config_roundtrip.py` | ✓ all 81 keys map 1:1; 20 declared additions |
| `scripts/capture_golden.py --verify` | ✓ v1 and v2 both reproduce, bit for bit |
| Checkpoint schema | **unchanged at v2** — Tier 4 touches no weight |

## Status by item

| ID | Change | Status | Measured effect | Claimed |
|---|---|---|---|---|
| **T4-1** | `groups.npy` persisted; scan-disjoint split with an explicit feasibility report | ✅ | the shipped split's leak is now **measured by the split itself**: 107 of 107 scans in train *and* val/test | −0.05 … −0.20 |
| **T4-2** | White-reference division if a panel exists, else per-pixel SNV | ✅ | no white panel in the archive (verified); SNV removes the per-pixel and per-session gains **exactly**, to float32 | +0.02 … +0.08 |
| **T4-3** | Resize the mask, re-mask, divide by α | ✅ | foreground count equals the resized mask exactly; a constant-spectrum seed is now flat to 1e-5 where the old order left an attenuated rim | +0.002 … +0.008 |
| **T4-4** | Persist 8 morphometrics, standardised on train only | ✅ | `morphology.npy` written; fitting on all rows instead moves the origin by > 1.0 σ-units | +0.002 … +0.006 |
| **T4-5** | `calib` split for margins / CDWS / oversampling | ✅ | all four fitted quantities route to `calib`; `val` carries none | +0.000 … +0.005 |
| **T4-6** | Publish the *k*-curve past *k*\* | ✅ | **the shipped elbow at k = 40 is NOT demonstrable** — the recorded curve stops there | unknown |

**Nothing in this tier has been trained, and one item cannot be until the
dataset is rebuilt.** T4-2, T4-3 and T4-4 change the arrays on disk, so they
take effect only after `python scripts/prepare_dataset.py` is re-run (~hours,
36 GB output). T4-1's grouped split additionally needs the `groups.npy` that
run writes. What *is* measured is each mechanism, on synthetic data whose true
answer is known, plus three measurements on the real artifacts — recorded
below.

---

## T4-1 · scan ids and the grouped split — `data/prep/patch_extraction.py`, `data/loaders.py`

`build_splits` becomes a four-line wrapper over a new `build_split_bundle`,
which returns `Splits(labels, train, calib, val, test, groups, report)`.
`cfg.data.split_scheme` selects between:

* **`stratified`** — the reference run's patch-level partition, **bit-identical**:
  the same two `train_test_split` calls, the same order, the same
  `random_state=42`, and the arrays left in the order sklearn returned them.
  That last detail is load-bearing — `DataLoader(shuffle=True)` permutes
  *positions*, so sorting the index array would have re-ordered the training
  stream at a fixed seed and moved the Stage-1 golden loss.
* **`grouped`** — P-1. Per class, order its scans, rotate by `split_fold`, hold
  out `max(1, round(m·eval_frac))` of them (never all), split those into
  val/test by scan when there are ≥ 2 and by patch when there is 1, then carve
  `calib` out of the train pool the same way.

### The §4.2 criterion is not constructible on this dataset, and here is why

§4.2 asks for "no `scan_id` in more than one split". Every variety was captured
exactly twice (0-H), so a class has **two** groups and can appear in at most
**two** of three splits. Meeting that criterion literally would leave every
class absent from one split — and a class absent from test cannot contribute to
a 90-class macro-F1, so the metric the whole project reports would become
undefined.

What is enforced instead is the contract §3.1 actually writes down:

$$\mathrm{scan}(i) \in \text{train} \;\Longrightarrow\; \mathrm{scan}(i) \notin \text{val} \cup \text{test}$$

which is achievable at two scans per class, keeps all 90 classes in all four
splits, and is exactly what `test_splits_are_group_disjoint` asserts. Where the
finer guarantee cannot be had, `SplitReport` names the classes rather than
implying it was: on two-scans-per-class data it reports
`val_test_group_disjoint: False` and lists all 90.

**Group granularity moved from `(session, variety)` to the cube.** 0-H could
only reconstruct the coarser key from `labels.npy`, which is why it found 73 of
90 classes with a single group and concluded F-1 was not testable. The writer
now emits the cube-level `scan_id` — `<session>/<variety>-<n>`, the key §3.1
specifies — under which every class has two groups and the split *is*
constructible. That is the redesign point 3 of "What Phase 0 changes about the
plan" asked for, resolved in favour of 0-H's option 1.

### Measured on the real artifacts

Run against the existing `labels.npy` and Phase 0's reconstructed
`outputs/phase0/scan_id.npy`:

```
Split: stratified (fold 0)  train: 6,036 (70%)  val: 1,294 (15%)  test: 1,294 (15%)
  ⚠ patch-level split — 107 of 107 scans are in train and in val/test (C-1).
```

**0-H's headline is now produced by production code at every startup**, not by
an audit script. And the strict grouped split, asked for the same coarse
groups, refuses with the count rather than a number:

```
ValueError: 73 of 90 classes were captured in a single scan (classes 0, 1, 2, 3, 5, 6, 8, 9,
10, 11 …), so no group-disjoint split exists for them. IMPROVEMENT_PLAN §3.1 P-1: report the
count and fall back to leave-one-scan-out CV rather than silently mixing.
```

That is §3.1's instruction implemented as behaviour. The documented
`single_group_policy="patch_split"` fallback exists, is opt-in, and records
every class whose eval patches share a scan with train.

---

## T4-2 · radiometry — `data/prep/radiometry.py` (new), `patch_extraction.py`

**The archive has no white panel.** Its 190 `.hdr` members are 180 seed cubes,
9 `black.hdr` dark references — which `preprocess_raw` already consumes — and
one `chessboard.hdr` geometric target. P-2(a) is therefore unavailable here and
the resolution is P-2(b), per-pixel SNV along λ after masking. The white-reference
path is implemented and tested anyway: the absence is a property of one archive,
not of the method, and `radiometry="white"` **raises** rather than silently
falling back, because a radiometric domain decided by a silent fallback is not
a domain anyone can report.

Two identities, measured rather than argued (`tests/unit/test_radiometry.py`):

| Claim | Measured |
|---|---|
| Under $x_{c,p} = a_p r_c$, SNV returns $(r-\bar r)/\mathrm{sd}(r)$ at every pixel | agrees to **2e-4** across a 10× spread of $a_p$ |
| A per-session gain $S$ leaves the SNV spectrum unchanged | agrees to **2e-5** at $S = 3.7$ (float32 accumulation over 40 bands) |

The companion measures the channel being closed: in the radiance domain the two
sessions of the *same* seed differ by 5× in mean, so "which session" is
recoverable by a threshold. After SNV their means agree to 1e-5.

**A finding worth recording: P-2 needs P-3.** SNV divides by a pixel's own
spread, so it is scale-blind — a background pixel left at a thousandth of a real
spectrum, which is exactly what the pre-T4-3 resize order produced, comes out of
SNV as a **unit-variance spectrum numerically indistinguishable from a seed**.
An exactly-zero background survives untouched (`0/eps = 0`); a nearly-zero one
does not. Applying P-2 without P-3 would have been worse than applying neither.
`test_snv_amplifies_a_background_residual_to_full_scale` pins that.

**Deviation from the plan, stated.** §3.1 says to store the gain "as two extra
channels". It is stored as a separate row-aligned `gain.npy` of shape
$(N, 2, 64, 64)$ instead. Concatenating it into `patches.npy` would change the
model's input arity from 40 to 42 channels — a Tier-3 architectural change
smuggled into a protocol tier, and one that would invalidate every golden gate.
The array is written, aligned and ready for FU-4/FE-2 to consume.

---

## T4-3 · the resize order — `data/prep/patch_extraction.py`

The per-region body is now `extract_patch`, which pads **both** the cube and the
mask, area-resizes **both**, and treats the resized mask as the fill map:
`keep = α > 0.5`, zero elsewhere, and `cube[keep] /= α[keep]`.

T4-3's criterion has two halves and both are asserted against a synthetic seed
whose true value is known everywhere (a constant spectrum on a **round** mask —
a grid-aligned rectangle has no partial-coverage pixels, so it would make the
test vacuous):

* *foreground pixel count matches the resized mask exactly* —
  `(patch != 0) == (α > 0)`, elementwise;
* *no partial-coverage pixels remain* — every foreground pixel carries the
  seed's spectrum to **1e-5**, where the old order leaves the boundary ring
  scaled by α and the sub-threshold ring at a small non-zero value.

The persisted `masks.npy` is α **zeroed outside `keep`**, a one-line deviation
from the plan's literal `mask_rs`: it makes `α > 0` and "this pixel is
foreground" the same predicate, so the mask cannot disagree with the patch it
describes on precisely the pixels M-11 is about.

A control test pins the scope of the change: on a mask that tiles the resize
grid exactly, the old and new orders are `array_equal`. P-3 is a boundary
correction and touches nothing else.

---

## T4-4 · morphometrics — `data/prep/segmentation.py`, `data/morphometrics.py` (new)

`morphometrics(region)` writes the eight descriptors §3.1 names, in that order,
straight off the `regionprops` object `segment` already built — three of which
(`area`, `eccentricity`, `solidity`) *already gate* the region filter, so the
pipeline was discarding numbers it had computed and acted on (M-13).

The criterion's second clause — "standardised on train only" — is a property of
the consumer, and the array is deliberately persisted **raw**: no split exists
at extraction time. `fit_morphometric_stats(morph, train_idx)` **raises** on an
empty `train_idx` rather than falling back to all rows, since that fallback is
the leak the function exists to prevent. Measured: fitting on everything instead
of on train moves the per-column origin by more than 1.0 in raw units on a
fixture whose held-out half is off-distribution.

---

## T4-5 · the calibration split — `data/loaders.py`, `engine/stages/stage{1,2}_*.py`

`cfg.data.calib_frac` carves `calib` out of the training pool, and `build_splits`'
`train_idx` excludes it, so P-5's "never used for gradients" is structural
rather than a convention.

Both stages now separate two jobs that shared one loader:

| Quantity | Fitted on | Was |
|---|---|---|
| 90 per-class margins (HD-3) | `calib` | `val` |
| $(90, 90)$ confusion matrix (HD-3) | `calib` | `val` |
| 90 CDWS sampling weights | `calib` | `val` |
| Phase-3 oversampling weights | `calib` | `val` |
| Early stopping / checkpoint selection | `val` | `val` |

`calib_ldr=None` restores the pre-Tier-4 routing exactly, which is what every
archived checkpoint was produced under and what `configs/data/spa40_90class.yaml`
still selects. `tests/unit/test_calib_split.py` traces each call site to the
split it received, and its companion asserts that without a calib split all four
land back on `val` — C-9, as behaviour.

**The realised fractions differ from the plan's 60/10/15/15**, and the reason is
T4-1's: at two scans per class the grouped split can only hold out *one* scan,
so it realises ≈ 50 % train+calib / 25 % val / 25 % test. `calib_frac` is
therefore defined as a fraction of the **training pool** (0.15 in the P-fix
config), not of the whole, so its meaning does not drift with the group
structure.

---

## T4-6 · the band-count curve — `data/prep/band_selection.py`

`n_select_max` 100 → 256 and `n_candidates` extended to
`[…, 100, 128, 160, 192, 224, 256]`, so the curve is *recorded* past the chosen
*k*. `find_elbow`'s arithmetic is untouched — T4-6 verifies the elbow, it does
not redefine it — and a new `verify_elbow` measures whether the claim is
supportable, writing `band_selection_elbow.json` next to the curve.

### Measured on the shipped artifact — the elbow at k = 40 is not demonstrable

```
Elbow at k = 40: NOT DEMONSTRABLE — the curve terminates at the chosen k —
peak is acc(40) by construction, so the 98 % criterion is satisfied vacuously (M-14)
  curve recorded to k = 40 (0 points past k)
  acc(k) = 0.4755   peak = 0.4755 at k = 40   headroom past k = +0.0000
```

`dataset/band_selection_report.csv` rises **monotonically** across all seven
recorded counts to its final point, which is the value it selected. M-14 is
confirmed on the artifact: the claim "$k^\star = 40$ by elbow" has no evidence
in the record either way, and §4.5's instruction — publish the curve to k = 256
or withdraw the claim — stands.

`deployed_curve_path` accepts the deployed estimator's own curve (F-3) and lets
it, rather than LDA/LinearSVC on 256-dimensional mean spectra, decide the winner
and the elbow. The six runs that produce that curve are not part of this tier;
the plumbing that consumes them is, and the report gains a `deployed` column so
the two estimators' curves live in one artifact.

---

## Test inventory

| Module | Tests | Pins |
|---|---:|---|
| `tests/unit/test_splits.py` | 28 | T4-1 — §4.3's `test_splits_are_group_disjoint` at 2/3/4/5 groups per class, every class present in every split, exact partition, val↔test disjointness where the counts allow it, leave-one-scan-out folds covering every scan, the single-scan refusal and its opt-in fallback, the stratified path's bit-identity, the Tier-4 config keys |
| `tests/unit/test_radiometry.py` | 21 | T4-2 — the two SNV identities, the radiance-domain leak, background exactness, the residual-amplification finding, white-reference algebra, the archive having no panel, mode resolution and its refusals |
| `tests/unit/test_patch_extraction.py` | 11 | T4-3 — both halves of the criterion, the old order's attenuated rim and non-zero background, α's contract, the grid-aligned control, SNV-after-masking |
| `tests/unit/test_morphometrics.py` | 13 | T4-4 — the eight columns and their order, shape discrimination, physical units surviving the resize, train-only standardisation and the leak it avoids, the empty-train refusal |
| `tests/unit/test_calib_split.py` | 10 | T4-5 — every fitted quantity traced to `calib`, selection traced to `val`, the fallback to `val` as the defect, calib carved from train and disjoint from everything, all 90 classes present |
| `tests/unit/test_band_curve.py` | 15 | T4-6 — the shipped curve's verdict, the demonstrable and still-rising cases, `find_elbow` unchanged, the deployed-curve reader |

Each defect test carries a companion asserting the **pre-Tier-4** behaviour fails
it (`test_the_stratified_split_shares_every_scan`,
`test_the_radiance_domain_leaks_the_session`,
`test_the_old_order_attenuated_the_boundary`,
`test_the_old_order_left_a_non_zero_background`,
`test_fitting_on_everything_would_move_the_origin`,
`test_without_a_calib_split_everything_falls_back_to_val`,
`test_the_shipped_curve_cannot_demonstrate_its_elbow`), so none of them can pass
vacuously.

---

## Gate maintenance

`check_ast_no_op_move.py`: four declarations extended — `build_splits`
(protocol change, with the bit-identity of the stratified path spelled out),
`build_patch_dataset` (the data-contract change), `select_bands` (T4-6) and
`run_stage1`/`run_stage2` (the `calib_ldr` routing). `find_elbow`,
`pad_to_square`, `resize_patch`, `segment` and `download` are **unchanged** and
still verdict IDENTICAL. Result: 118 identical, 56 declared, 33 new, **0 drift**.

`check_config_roundtrip.py` gained five `INTENDED_ADDITIONS` — `data.groups_path`,
`split_scheme`, `split_eval_frac`, `split_fold`, `calib_frac`. All five are read
by `build_split_bundle` on every path, and
`test_every_new_data_key_is_read_by_the_split_builder` proves it by changing each
one and watching the partition react; a key that was merely present would not.
No dead keys were added, which is the N-1 defect this tier was in a position to
repeat.

---

## What Tier 4 changes about the plan

1. **T4-1's §4.2 criterion is unachievable and the §3.1 contract is the right
   one.** Two scans per class caps a class at two of three splits, so "no
   `scan_id` in more than one split" and "90-class macro-F1" cannot both hold.
   The contract enforced is `train ∩ (val ∪ test) = ∅`; everything finer is
   reported per class rather than claimed. **§4.2's criterion column for T4-1
   should be restated.**
2. **0-H's blocker is resolved, not worked around.** A cube-level `scan_id`
   makes the grouped split constructible for all 90 classes. F-1's −0.05…−0.20
   is testable again — under a *cube*-disjoint split, which controls for seed
   identity and scan geometry but not for session-level radiometric drift, since
   73 of 90 classes have both cubes in one session. **F-1 should be re-stated at
   cube granularity, and the session-disjoint question restricted to the 17
   two-session classes as a separate diagnostic.**
3. **P-2 depends on P-3, which the plan lists as independent items.** SNV
   amplifies a near-zero background to full scale, so T4-2 applied without T4-3
   would have made the leak worse. §4.4's dependency graph shows the two as
   siblings; they are not.
4. **M-14 is confirmed on the artifact.** The recorded curve stops at the k it
   chose, so the elbow claim is unfalsifiable from the record. This is now
   asserted by a test against the committed CSV rather than by reading it.
5. **Three items cannot take effect without re-running extraction.** T4-2, T4-3
   and T4-4 rewrite `patches.npy` (36 GB, hours), and T4-1's grouped split needs
   the `groups.npy` that run produces. Until then the code is complete and inert,
   and `configs/data/spa40_90class.yaml` still selects the old protocol
   deliberately — so `outputs/output_v12_spa40/` keeps reproducing.
6. **The re-baseline is now the only thing standing between here and Tier 3.**
   §4.4 places it there, Tier 2 already made it mandatory (schema v2 changed the
   architecture), and Tier 4 has now changed what the number means. One run,
   under `data=spa40_90class_pfix`, produces the number every Tier-3 decision
   should be measured against.

## Reproducing

```bash
pytest tests/                             # 469 passed, at Tier-4 completion
python scripts/check_ast_no_op_move.py    # 0 drift
python scripts/check_config_roundtrip.py  # 81/81
python scripts/capture_golden.py --verify # v1 vs the baseline, v2 vs the tree
ruff check src tests scripts && black --check src tests && mypy
```

Until `scripts/prepare_dataset.py` has been re-run, `data=spa40_90class_pfix`
fails fast with a message naming the missing file, which is the intended
behaviour: a protocol that silently degrades to the old one when an array is
absent is the defect this tier removed.
---

# Tier 3 — architectural redesign ✅ COMPLETE

All seven items are implemented. Tier 3 is the tier §4.4 places last and prices
at 3–4 weeks, and it is the only one that changes what each branch *sees*: the
controlling constraint of §3.3 is that **each branch must see something the
others cannot reconstruct**, and before this tier two of the four received a
byte-identical tensor.

| | |
|---|---|
| Suite | **555 passed**, 0 failed, 0 skipped (was 469) — `pytest tests/`, ~40 min |
| New test modules | 7, adding 89 tests |
| `ruff` / `black` / `mypy --strict` | clean over all 76 source files |
| `scripts/check_ast_no_op_move.py` | ✓ 96 identical, 78 declared, 40 new, **0 drift** |
| `scripts/check_config_roundtrip.py` | ✓ all 81 keys map 1:1; 30 declared additions, **1 declared removal** |
| `scripts/capture_golden.py --verify` | ✓ v1 reproduces the baseline; v3 reproduces the tree |
| Checkpoint schema | **v2 → v3, and not migratable** — see [the refusal](#the-v1v2--v3-refusal) |
| Parameters | **7,856,203 → 5,194,578 (−2.66 M, −33.9 %)** |

## Status by item

| ID | Change | Status | Measured | Claimed |
|---|---|---|---|---|
| **T3-1** | BR-1: Branch B → learned NDI bank + continuum depths + morphology | ✅ | **−591,528 params**; σ₃/σ₁ on the new descriptor **0.91** against **0.0025** within-sample on the nine moments | +0.005 … +0.015, −591 k |
| **T3-2** | BR-3: 3-D spectral–spatial stem | ✅ | **+536,488 params**; the stem separates a band↔space swap the 1×1 `band_reduce` maps to identical pooled features | +0.010 … +0.030 |
| **T3-3** | BR-4: λ-derived token PE, relative-λ bias, λ-uniform tokenisation | ✅ | **−939,226 params**; a 40-band Branch D loads into a 20-band one `strict=True` and the two agree to 1e-4 — **F-3 settled** | +0.005 … +0.015, −931 k |
| **T3-4** | FU-1(b)+FU-2+FU-4+FU-5: gated low-rank bilinear fusion | ✅ | **−1,694,911 params** (+17,216 for the morphology token); Σ gₘ ≠ 1, and the fusion is no longer additive in each modality | +0.005 … +0.015, −1,641 k |
| **T3-5** | FE-1: continuous λ-kernels + Savitzky–Golay derivative channels | ✅ | **+10,336 params**; ∂/∂λ of a ramp is its slope at both 2.4 nm and 100 nm spacing, where an index kernel is off by 42× | +0.008 … +0.020 |
| **T3-6** | BR-2: derivative input, mass-weighted pooling, 8×8 grid | ✅ | Branch A's input is no longer byte-identical to D's — different **shape**, and gain-invariant where D's is not (relative drift 1.6e-4 vs 1.5) | +0.003 … +0.010 |
| **T3-7** | FE-2: the persisted mask passed explicitly | ✅ | under an explicit mask a global brightness offset leaves the foreground exactly unchanged; under the threshold it reclassifies **all 4,096 pixels** as seed | +0.001 … +0.003 |

**Nothing in this tier has been trained.** Like Tier 4, the deliverable is the
mechanism plus a test that pins it. Two items are additionally *inert* until the
arrays are rebuilt: T3-7's fill map and FU-4's morphometrics arrive from the
same `scripts/prepare_dataset.py` run T4-1…T4-4 need, and until then the model
takes its documented fallbacks — the `> 1e-5` band-sum threshold and a zero
morphometric vector. Both fallbacks are **exact**, and
`test_the_fallback_is_the_pre_tier3_behaviour_exactly` pins that.

---

## The parameter budget against §3.8

| Component | §3.8 says | Actual | Δ vs plan |
|---|---:|---:|---:|
| `se` (MaskedSpectralECA) | 6 | 6 | — |
| `branch_a` — profile → derivative profile | ~600,000 | 603,089 | +0.5 % |
| `branch_b` — moments → index bank + morphology | ~95,000 | **94,896** | −0.1 % |
| `branch_c` — 2-D CNN → 3-D spectral–spatial stem | ~2,300,000 | 2,230,646 | −3.0 % |
| `branch_d` — SpecFormer → λ-aware SpecFormer | ~1,250,000 | 1,241,640 | −0.7 % |
| `cross_interaction` — Perceiver → gated bilinear | ~550,000 | 496,005 | −9.8 % |
| `morphology_embed` (new) | ~17,000 | 17,216 | +1.3 % |
| `aux_head_{a..d}` | 178,024 | 178,024 | — |
| `embed_net` (merged with `output_proj`) | ~264,000 | 263,936 | −0.0 % |
| `linear_head` | 0 | 0 | — (gone at T2-10) |
| `arcface_head` | 69,120 | 69,120 | — |
| **Total** | **≈ 5,323,000** | **5,194,578** | **−2.4 %** |

Against the shipped model the delta is **−2,661,625 (−33.9 %)**, which is §3.8's
−2.56 M / −32 % with the extra 23,130 T2-10 had already removed. Every line
lands inside 10 % of its budget, and the one §3.8 spends the most words on —
Branch B's 686 k → 95 k — lands inside 0.1 %.

The reallocation §3.8 is actually about:

| | Before | After |
|---|---:|---:|
| Params per input scalar, Branch C | 10.34 | **13.62** |
| A ∪ B ∪ D share of the budget | 44.0 % | **37.3 %** |
| Params in modules reading ≤ 640 numbers | 3.46 M | 1.94 M |

**−2.66 M removed from modules operating on 640 numbers or on 4 tokens, +0.54 M
added to the only module that sees the full cube.**

---

## T3-5 · FE-1 — `models/front_end.py` (new)

C-5's magnitude was measured in Phase 0 (0-F): the selected bands are not
uniformly spaced, so a `Conv1d` over the band index is a finite difference
divided by the wrong step, and the step varies along the axis. Two mechanisms,
both in the new module:

**(b) Savitzky–Golay derivative operators on the irregular grid.** For each band
the `window` nearest bands **in λ** form a neighbourhood, a degree-2 polynomial
is fitted in the shifted coordinate λⱼ − λᵢ by ordinary least squares, and the
derivatives are read off the coefficients (a₁ and 2a₂). Two dense `(C, C)`
buffers, no parameters. Because the design matrix carries the true offsets the
estimator is **exact** on polynomials of degree ≤ 2 however uneven the grid is.

Measured on a grid holding both spacings §4.2 names — five bands 2.4 nm apart,
then five 100 nm apart:

| Quantity | Truth | SG on the λ metric | Central difference on the index |
|---|---|---|---|
| ∂/∂λ of `3λ + 7`, narrow end | 3.0 | 3.000 | **7.2** |
| ∂/∂λ of `3λ + 7`, wide end | 3.0 | 3.000 | **300.0** |
| ∂²/∂λ² of `2λ²`, everywhere | 4.0 | 4.002 | — |

The index-space operator's answer differs by **41.7×** between the two ends of
the same axis, for a function whose derivative is constant. That is C-5, and it
is now a test rather than an argument.

**(a) `LambdaConv1d`.** The kernel for the pair *(band i, neighbour j)* is
generated by a two-layer MLP from a Fourier featurisation of λⱼ − λᵢ, over the
`k = 5` nearest bands **in wavelength**. One kernel function is learned once and
shared by every band, so the Δλ-dependence is a property of the operator; the
parameter count (10,560) is **independent of the band count**, against the
`Conv1d(1, 96, 3)` stem's 288. It replaces Branch A's stem.

`model.wl_embed_dim` — dead since the reference implementation (N-1c) — is now
the Fourier width of both this and BR-4's attention bias, which is the home §3.2
nominates for it rather than deleting it.

> **A note on the normalisation.** The derivative channels are multiplied by the
> median band spacing (and its square) so they sit at the same order of
> magnitude as the reflectance channel instead of 39× and 1520× above it on the
> min-max-normalised axis. That is one scalar for the whole operator and
> therefore cannot restore the defect it is applied on top of —
> `test_normalisation_rescales_every_band_identically` asserts it, rather than
> leaving it as a claim.

---

## T3-7 · FE-2 — `models/stats_ops.py`, `blocks/attention.py`, and the loader path

Every masked operator re-derived the foreground from `sum_c |x_c| > 1e-5`. That
threshold is a proxy and it is wrong in three separate places: it breaks under
any brightness transform (C-8), it is binary so a resized boundary pixel that is
40 % seed counts as a whole one (M-12), and it cannot distinguish background
from a band the seed genuinely reflects nothing in.

`foreground_mask(x, mask)` is now the single place the fallback lives, and
`MaskedSpectralECA`, `extract_grid_spectra`, `masked_spectral_stats` and Branch
C's stem all take the mask as an argument. **Measured on a cube with a real zero
background, lifted by a constant 0.1:**

| | Foreground fraction seen |
|---|---:|
| Explicit fill map | 0.234 (exact) |
| `> 1e-5` threshold | **1.000** |

Under the threshold every masked statistic is computed over the frame. Under the
fill map the foreground mean of `x + c` is the foreground mean of `x` plus `c`,
exactly — which is FE-2's criterion verbatim.

The map is also used as a **weight** rather than a predicate, so a cell's mass
scales linearly with coverage. That is the half of M-12 the arithmetic can fix;
the other half is BR-2's mass-weighted pooling.

### The plumbing

`RiceSeedDataset.__getitem__` yields `(patch, label)` when no side array is
configured — the pre-Tier-3 contract, byte for byte — and
`(patch, label, mask, morph)` otherwise. Seven loops across `engine/` go through
one new helper, `engine/batch.py::unpack_batch`, which accepts both shapes and
the plain `TensorDataset(x, y)` the golden capture builds.

Three details are load-bearing:

* **The mask rides through the dihedral transform as a trailing channel.** It
  has to receive the *same* flip and rotation as the patch, and concatenating
  them is the only way to guarantee that without a second draw from the RNG
  stream `__getitem__`'s reproducibility depends on. The spectral augmentations
  run before the concatenation — a band dropout has no meaning for a fill map.
* **Spatial CutMix pastes the partner's mask too.** Pasting a seed's pixels
  while keeping the anchor's mask would claim the pasted region is background.
* **Mixup mixes both side inputs with the same permutation.** `_mixup` gained
  `return_perm`; re-drawing `torch.randperm` would pair each patch's spectrum
  with another patch's mask. Both quantities mix linearly for a reason: the fill
  map is a coverage *fraction* since P-3, and the morphometrics are standardised
  physical measurements.

TTA rotates and flips the mask with each spatial view and passes the input's
mask with each spectral one, since a contrast rescaling about the foreground
mean does not move the support.

---

## T3-1 · BR-1 — `models/branches/spectral_stats.py`

The nine masked moments are gone. §2.2.5 proves that tensor has rank ≤ 2 under
the per-pixel gain model x_{c,p} = a_p r_c, and 0-D confirmed it on the real
data: 686 k parameters were reading a two-dimensional signal.

Three groups replace them, on 94,896 parameters:

| | What | Params |
|---|---|---:|
| (i) | 64 learned normalised-difference indices — π± = softmax(θ±) ∈ Δ³⁹, z = (u−v)/(\|u\|+\|v\|+ε) | 5,120 |
| (ii) | The 16 deepest continuum-removed absorption depths, 1 − r/hull(r) | 0 |
| (iii) | The 8 persisted morphometrics (P-4) | 0 |
| | LayerNorm + 2-layer MLP → 256 | 89,776 |

**§4.2's criterion, measured.** Data generated from Appendix A.1's gain model
with a per-pixel gain field:

| Representation | σ₃/σ₁ | Criterion |
|---|---:|---|
| Nine moments, within a sample | **0.0025** | — (this is the defect) |
| Nine moments, worst sample | 0.014 | — |
| Index bank + depths + morphology | **0.91** | > 0.3 ✅ |

Two implementation notes worth recording.

**The index is invariant to the gain to 1.3e-7**, not approximately: NDI(a·r) =
NDI(r) exactly for a > 0. The denominator is written `|u| + |v| + ε` rather than
`u + v + ε`; on non-negative reflectance the two are identical — the domain the
plan assumes — and the absolute values keep the index finite and inside [−1, 1]
if a caller ever hands the branch a mean-centred spectrum.

**The continuum hull is exact, not iterative.** By Carathéodory's theorem in one
dimension the upper concave envelope is the pointwise maximum over *chords*: for
every pair of bands bracketing i, the line through them evaluated at λᵢ. The
interpolation weights depend only on the wavelength grid, so they are
precomputed once and the envelope is a masked maximum — differentiable through
the two active endpoints, and with no iteration count to tune. A rolling-ball
approximation would fail `test_the_continuum_hull_is_exact_on_a_concave_spectrum`
by however far it had not converged; this passes at 0.0.

The chord tensor is O(C³) — 64,000 entries at C = 40 — so the forward chunks
over the left endpoint, and the buffers are **non-persistent**: they are a
deterministic function of a wavelength vector `wl_pe_cnn.pe` already carries, and
half a megabyte of derivable constants has no business in every checkpoint.

---

## T3-2 · BR-3 — `models/branches/spatial_cnn.py`

C-3 in one line: **the network contained no joint spectral–spatial operator.**
`band_reduce` was two 1×1 convolutions, so the band axis was collapsed to 64
channels *before* any spatial kernel ran and the first 3×3 already saw a
spectrally-mixed map. "This absorption feature, in this part of the seed" was
not in any module's hypothesis class.

The replacement is §3.3's diagram exactly — `Conv3d(1→16→32→64)` over
(λ, h, w) with the spectral axis halving three times, then a 1×1 fold of the
surviving 5 spectral positions into 192 channels, then the existing
ResBlock2D/CBAM tail. Stem cost 178,256; branch 1,694,158 → 2,230,646.

**The falsifiable form of C-3, and how it is tested.** Two cubes with identical
per-band spatial marginals and a swapped band↔location pairing: a bright blob at
top-left in band 10 and bottom-right in band 30, against the same two blobs with
the bands exchanged. A 1×1 stem applies one linear map to every pixel's spectrum
independently, so the multiset of feature vectors over the grid is identical and
the pooled features agree to 1e-5 — which is exactly what Branch C's `mean`/`amax`
tail sees. The 3-D stem separates them.

Two further changes:

* **The mask re-zeros the padded region after every stage**, area-pooled to that
  stage's resolution. Without it the stem's own padding grows a non-zero
  response in the frame within two layers, and a CNN that can see the frame can
  learn it.
* **`BRANCH_DROP_PROFILE` is inverted.** The reference profile dropped Branch C
  at 0.30 and D at 0.20 and never touched A or B — it regularised hardest
  against the one branch no other can reconstruct, and not at all against the
  two that were near-duplicates (M-5, §2.2.7). §3.3 prescribes
  `(0.15, 0.15, 0, 0.15)`, which is what the profile now realises at the shipped
  `branch_drop_prob = 0.20`.

`SpectralQuadNet._init_weights` gained `nn.Conv3d` in its kaiming arm. Without
it the stem's three convolutions would have kept torch's default init while
every other convolution in the model got kaiming fan-out — a silent
inconsistency in the one branch the tier moves capacity *into*.

---

## T3-3 · BR-4 — `models/branches/specformer.py`

All three of §3.3's changes, and the one that settles F-3 is (iii).

**(iii) λ-uniform tokenisation.** `[0, 1]` is cut into `num_bands // (specf_patch // 2)`
= 10 equal-width windows and the bands in each are averaged. After this the axis
the tokenizer's 3/5/7 kernels run over is uniform in wavelength, so a kernel
width is a *bandwidth* — C-5 closed for Branch D the way FE-1 closes it for A.

**(i) λ-derived token embeddings.** `spec_pos_embed` is now a **buffer**
evaluated at each window's centre wavelength, using the same
`sinusoidal_wavelength_encoding` the CNN branches' band-axis PE uses.
`PhysicalWavelengthPE` was refactored to expose that encoding as a free
function; §3.3 asks for the module to be "four-way shared", and after BR-1
deleted Branch B's band axis, sharing the *function* is the form of that which
survives — the arithmetic is identical and `wl_pe_cnn.pe` is byte-for-byte
unchanged, which the golden init digests confirm.

**(ii) Relative-λ attention bias.** A per-head scalar b_ψ(λ̄ₜ − λ̄ᵤ) from a
1,320-parameter MLP over the Fourier features of the wavelength difference,
added to every spectral-stage logit. It is handed to `nn.MultiheadAttention` as
a float `attn_mask` — exactly the additive-logit contract that argument
implements — so no attention arithmetic is re-implemented. The CLS row and
column are left unbiased: the token has no wavelength, so any bias there would
be an arbitrary constant.

`specf_dim` drops 256 → 192 and the branch with it, 2,180,866 → 1,241,640.
`specf_drop` (N-1b, the branch was built with a literal 0.10) and `specf_patch`
are both wired.

### F-3, settled

§4.2's criterion is "Branch D transfers across band counts without retraining
the positional table". Measured:

```
forty  = SpecFormerBranch(wl[:40], patch_size=8)   # 10 tokens
twenty = SpecFormerBranch(wl[::2], patch_size=4)   # 10 tokens
twenty.load_state_dict(forty.state_dict(), strict=True)   # → all keys matched
```

and, fed a spectrum constant within each λ-window, the two produce embeddings
agreeing to **1e-4**. The shapes match *and* the meanings match, which a
shape-compatible load alone would not establish.

> One line makes this work and it is worth naming: the windows partition a
> **fixed** `(0, 1)` domain, not the observed `(min, max)` of the wavelengths
> handed in. A 20-band subset has a different observed maximum, so windows cut
> to the observed range would put token 3 at a different wavelength in the two —
> the same un-transferability the learned table had, arrived at by a different
> route. The first implementation had exactly that bug and the transfer test
> caught it.

---

## T3-4 · FU — `models/fusion.py`

§4.3 lists `test_fusion_latents_are_diverse`, asserting max cos(Lₙ, Lₙ′) < 0.9
after training. **That test cannot be written against this module, and the
reason is the finding itself.** FU-1(b)'s remedy for M-1 is not to fix the
latents' initialisation scale; it is to delete them. With five modality tokens
latent cross-attention compresses nothing, and 0-E's collapse metric becomes
*moot* rather than satisfied — which is §4.2's own wording for T3-4's criterion.

What replaces the Perceiver:

```
ν       = (log‖b_m‖₂)_m                     computed BEFORE normalisation  (FU-2)
γ       = σ(W_g[b̂_1‖…‖b̂_5‖ν]) ∈ (0,1)⁵      sigmoid, not softmax           (FU-2)
f₁      = Σ_m γ_m b̂_m                        first order
f₂      = Σ_{m<m′} (U_m b̂_m) ⊙ (U_m′ b̂_m′)   all 10 pairs, rank 128         (FU-1b)
f       = W_o[f₁ ‖ V f₂]
```

with `BatchNorm1d` per modality rather than `LayerNorm`, so the normaliser is a
*dataset* statistic and a low-SNR sample is not amplified to unit scale (M-2a).
2,190,916 → 496,005 parameters, and `output_proj` is deleted, leaving `EmbedNet`
as the single post-fusion residual block (N-10, FU-5).

Each of §2.3's findings has a test that would fail without its fix:

| Finding | Property asserted | Result |
|---|---|---|
| M-2b | Σ gₘ ≠ 1, and two gates can both exceed 0.9 | min \|Σg − 1\| = 1.52 over the batch |
| M-2a | scaling one branch by 1e-3 changes the gate | changes |
| M-3 | `f(2b₀, ·) − f(b₀, ·)` depends on the other modalities | it does |
| M-1 | no `latents`, no `blocks` attribute survives | ✅ |
| N-10 | no `output_proj` | ✅ |

**FU-4** adds the fifth token: `MorphologyEmbed`, 8 → 64 → 256, 17,216
parameters. §2.2.10's M-13 is that the segmentation stage computes eight
morphometrics, *gates the patch on them*, and throws them away; they are size
and shape, which no spectral operator in the network can derive from a resized
64×64 patch. They now reach both Branch B's descriptor and the fusion.

---

## T3-6 · BR-2 — `models/branches/spectral_profile.py`

§2.2.2's finding is that **Branch A was a strict functional subset of Branch D**:
both received `extract_grid_spectra(x, 4)` — the same tensor, byte for byte —
and both reduced it to a 256-vector. 2.77 M parameters across two branches
reading one input, under a claim of "four disjoint views of the same patch".

Three changes, none of which adds a parameter to the towers:

1. **A different input.** A consumes `[SNV(r), ∂/∂λ, ∂²/∂λ²]` of each cell's
   spectrum; D keeps the raw grid spectra.
2. **Mass-weighted cell pooling.** `extract_grid_spectra` now *returns* ωₙ, the
   cell foreground mass it always computed as its normaliser and always
   discarded, and Branch A's cells are pooled by it.
3. **An 8×8 grid** for A (64:1 compression) against D's 4×4 (256:1). The branch
   processes cells independently, so this costs nothing but a larger flattened
   batch.

**§4.3's `test_branch_inputs_are_distinct`, measured.** `branch_inputs` is the
method `forward` itself calls, so what is hashed is what the branches consume:

| Branch | Input shape | Response to a ×2.5 gain |
|---|---|---:|
| A | `(B·64, 3, 40)` | **1.6e-4** (relative) |
| B | `(B, 88)` | ~0 on the index/depth block |
| C | `(B, 40, 64, 64)` | full |
| D | `(B, 16, 40)` | **1.50** (relative) |

Four pairwise-distinct hashes, and — more usefully — A and D now respond
*differently to the same transform*, which is a claim two branches computing the
same function of the patch cannot satisfy. The finer grid is measured as the
pure-cell fraction: **0.75 of 8×8's cells are wholly seed or wholly background against 0.50 of
4×4's**, which is the M-12 mixing the coarser compression causes.

### The 8×8 grid is the tier's throughput cost, and it is Branch A's, not C's

§3.3 BR-2(3) prices the finer grid as "the parameter count is unchanged; only
the flattened batch grows from 16B to 64B", and adds "if throughput matters,
keep 4×4 for D" — which is what landed. Measured on CPU at batch 4:

| | ms / forward |
|---|---:|
| Branch A, 8×8 (256 cells) | **2,704** |
| Branch A, 4×4 (64 cells) | 849 |
| Branch C (the 3-D stem, full cube) | 67 |
| Branch D | 11 |
| Whole model | 2,876 |

**Branch A is 94 % of the forward pass, and the 8×8 grid costs 3.2× of it.**
The intuition the plan's framing invites — that a 3-D convolution over the full
`(40, 64, 64)` cube is the expensive addition — is wrong by a factor of 40 here.
The stem is three strided convolutions on a shrinking cube; Branch A is 256
independent runs of seven `LargeKernelBlock1D`s whose inverted bottleneck is
`96 → 384 → 96` over 40 bands, and the cell count multiplies all of it.

That is a CPU measurement with four threads and many small kernels, so it
overstates what an accelerator will see — the 256 cells are a batch dimension
and parallelise, where the 3-D stem's serial depth does not. It is recorded
because it is the one place Tier 3 makes the model materially slower, and
because the remedy if it bites is a knob: `model.grid_size_a` is a config key
and 4×4 restores the old cost while keeping the *other* two halves of BR-2 (the
derivative input and the mass weighting), which are what actually close C-2.

---

## The v1/v2 → v3 refusal

`SCHEMA_VERSION` goes to 3 and **`remap_state_dict` raises `SchemaTooOldError`
on anything below it.** This is the one place Tier 3 removes a capability rather
than adding one, so it is worth stating why.

Through Tier 2 the gap between an archived checkpoint and the live model was
mechanical: `linear_head.*` left, `arcface_head.confusion` arrived zero-filled,
and every other tensor passed through untouched — a migration that could be
asserted key by key, and was. Tier 3's gap is not mechanical. Branch B reads a
64-index bank where it read nine moments; Branch C's stem is 3-D where it was
two 1×1s; Branch D runs at `d_model = 192` where it ran at 256; the fusion has
no attention at all. A weight trained to read one of those inputs is not a
weight for the other, whatever its shape.

The failure mode the refusal rules out is the quiet one: a migration that drops
the keys it cannot place, loads the rest, and produces a model that is two
thirds trained and one third random. That model scores *something*, and the
something gets written down.

The tests that used to prove the migration was faithful now prove the refusal
is total — including one that loads the raw v1 state dict and requires
`strict=True` to fail on *branch* tensors rather than only on the head, so the
refusal is a statement of fact and not of conservatism. Two more that measured
Tier-1/Tier-2 findings on
`best_stage2.pth` are **inverted rather than deleted** — a test that silently
disappeared would leave the impression the measurement still stands behind the
current architecture. The numbers themselves (F-9's cos = 0.9996; ASAM's 375×)
stay recorded here.

The end-to-end coverage this gives up —
`test_stage3_checkpoint_reproduces_its_recorded_val_f1`, the strongest gate in
the suite — returns with the first Tier-3 run. Its loader half still executes.

---

## An unrelated finding, recorded because it moves a Tier-2 estimate

`test_raw_sam_overspends_on_the_head_at_initialisation` **inverted**. Under the
Tier-3 architecture raw SAM puts ~0.20 % of the perturbation on a head holding
1.33 % of θ, i.e. it *under*-spends where it used to over-spend by 2.3×.

Two Tier-3 changes drive it: the head is a larger share of a model that lost
2.66 M parameters, and FU-2 replaced the fusion's per-sample `LayerNorm` with
`BatchNorm1d`, so the embedding handed to the head is no longer rescaled to a
fixed norm per sample and the head's gradient at initialisation is
correspondingly smaller.

T2-4 already found §2.5.8's premise absent on `best_stage2.pth`; it is now
absent at initialisation too. **ASAM is still correct** — SAM's ρ-ball is not
scale-invariant and the ArcFace head is — and the criterion still holds. What no
longer holds is the mechanism T2-4's **+0.002 … +0.006 was reasoned from, at
either measurement point.** That estimate should be treated as unsupported.

---

## Test inventory

| Module | Tests | Pins |
|---|---:|---|
| `tests/unit/test_front_end.py` | 15 | T3-5 — §4.3's Δλ-scaled derivative test at 2.4 nm and 100 nm, the index-space companion that fails it, exactness on ramp/parabola, the uniform-grid degeneration, λ-order equivariance, the normalisation as a global constant, SNV invariance, the λ-kernel's neighbourhood and its band-count-independent cost |
| `tests/unit/test_branch_inputs.py` | 8 | T3-6 — §4.3's `test_branch_inputs_are_distinct`, A vs D by shape *and* by gain response, the finer grid's pure-cell fraction, C as the only branch keeping H and W, morphology reaching B and nothing else, and non-vacuity (the inputs are the forward's) |
| `tests/unit/test_masked_ops.py` | 12 | T3-7 + T3-1 — FE-2's brightness criterion and the threshold's failure on it, all four masked operators taking the mask, the exact fallback, soft coverage as a weight; 0-D reproduced within-sample and repeated on the new descriptor, the index bank's gain invariance, the hull's exactness and its absorption depth |
| `tests/unit/test_fusion.py` | 11 | T3-4 — Σ gₘ ≠ 1, a reachable two-modality conjunction, ν as the pre-norm log-norm, the multiplicative term M-3 found missing, all ten pairs, the fifth token, `output_proj`/`latents` gone, the §3.8 budget, `BatchNorm1d` over `LayerNorm` |
| `tests/unit/test_config_wiring.py` | 24 | §4.3's `test_config_keys_are_wired` in its **strong** form — 16 keys perturbed one at a time with the forward required to move, 2 dropout rates probed in train mode from a fixed RNG state, 3 excused with a named covering test, exhaustiveness over `cfg.model`, and a bit-identical unperturbed control |
| `tests/unit/test_branch_c_stem.py` | 8 | T3-2 — the band↔space swap the 3-D stem separates and the 1×1 `band_reduce` cannot, the spectral axis alive for three stages, the fold width derived not hardcoded, the mask zeroing the frame, the §3.8 budget and its direction |
| `tests/unit/test_specformer_lambda.py` | 11 | T3-3 — F-3's strict cross-band-count load *and* agreement on a shared spectrum, the PE as a buffer, λ-uniform windows against a non-uniform band grid, empty-window fallback, the bias as a function of Δλ, the unbiased CLS row, raw nanometres refused, and non-vacuity |

Extended: `test_branch_drop.py` (the dead-key list is **now empty**, and the
test asserting the defect is replaced by one asserting its closure),
`test_state_dict_compatibility.py` (the migration tests become refusal tests,
plus a forward-looking v3 round trip), `test_golden_forward_pass.py` (the v2 → v3
delta by declared prefix, and the modules Tier 3 did not touch), `test_asam.py`
and `test_stage3_sam.py` (the two inverted measurements above),
`test_unified_head.py` (`SCHEMA_VERSION >= 2` rather than `== 2`, so a later
architecture item does not fail a head test it has nothing to do with).

Every §4.3 test now has a home. The two that could not be written as specified
say so, in the module that would have carried them:
`test_fusion_latents_are_diverse` (there are no latents — FU-1(b) deletes them,
and §4.2 anticipates this: "0-E collapse metric becomes moot") and
`test_config_keys_are_wired` (present in both an inventory and a traced-forward
form, since the inventory alone cannot tell a key that is read from one that is
read and discarded — which is exactly how `branch_drop_prob` stayed dead for
five phases).

---

## Gate maintenance

`check_ast_no_op_move.py`: **22 new declarations**, all of a kind the file had
not needed before. Through Tier 1 an entry was a *relocation* or a *correctness
fix*; Tier 2 added *removal*; Tier 3's are **redesigns** — the module keeps the
name the checkpoint schema addresses it by and computes something else. Each
names its item, what it stopped doing, and the test that pins the replacement.
Result: 96 identical, 78 declared, 40 new, **0 drift**.

`check_config_roundtrip.py` gained ten `INTENDED_ADDITIONS`, one
`INTENDED_VALUE_CHANGES` entry (`specf_dim`, 256 → 192) and — new machinery —
an `INTENDED_REMOVALS` table. §2.7's remedy for a dead key is *wire it or delete
it*, and until now the script could only express the first. `fusion_heads` is
the first deletion: it named the head count of the fusion's attention, and the
gated bilinear fusion has no attention, so there is nothing to wire it to. An
entry there needs a reason for the same purpose an addition does — so that "the
key is gone" cannot quietly mean "the key was dropped".

`capture_golden.py`: **schema v3**. `golden/v2/` is frozen — the code that
produced it no longer exists — and is read only as the left-hand side of the
v2 → v3 delta. That delta is 115 keys out and 70 in, which is too many to
enumerate tensor by tensor without the declaration becoming unreadable, and an
unreadable declaration is not a check; `V3_DELTA` declares **25 prefixes**, each
with its plan item, and the gate fails on any key outside them. `golden/` (v1)
is still re-verified against the pinned baseline on every run and **still
reproduces bit for bit** — that gate has never moved.

---

## What Tier 3 changes about the plan

1. **§4.3's `test_fusion_latents_are_diverse` is not writable, and §4.2 already
   knew.** FU-1(b) deletes the latents rather than fixing their scale, and
   T3-4's criterion column says the collapse metric "becomes moot". The §4.3
   row should be restated as the property that made collapse matter: Σ gₘ ≠ 1
   and the fusion is not additive in each modality.
2. **§3.3 BR-4(i)'s "make it four-way shared" is not constructible after BR-1.**
   `PhysicalWavelengthPE` is a buffer indexed by band, Branch D's tokens each
   cover a λ-window, and BR-1 leaves Branch B with no band axis at all. What is
   shared is the *encoding function*, which is the content of the instruction;
   the instance cannot be.
3. **λ-uniform tokenisation needs a fixed domain, which §3.3 does not say.**
   "Partition [λ_min, λ_max]" reads as the observed range, and windows cut to
   the observed range are *not* transferable across band counts — the property
   BR-4(iii) exists to deliver. The domain must be the sensor's, fixed. The
   first implementation had it the plan's way and the F-3 test caught it.
4. **T3-4's parameter claim needs FU-4 counted separately.** §3.8 prices
   `cross_interaction` at ≈0.55 M and `morphology_embed` at ≈17 k on separate
   lines, but §3.4 FU-1(b)'s "≈0.29 M" is computed for four modalities. At five
   it is 0.50 M, which is what landed. The line item is right; the inline
   arithmetic is for M = 4.
5. **BR-2's cost estimate is missing.** §3.3 prices the 8×8 grid as free in
   parameters and says nothing about time, and the measurement above shows
   Branch A is 94 % of the forward pass with the grid multiplying all of it.
   The item is still worth doing — it is half of what closes C-2 — but the
   sequencing advice should say so, and `model.grid_size_a` should be the first
   thing tried if a training run turns out to be wall-clock-bound.
6. **Two Tier-3 items are inert until the extraction is re-run**, exactly as
   three Tier-4 items are, and for the same reason: T3-7's fill map and FU-4's
   morphometrics come from `scripts/prepare_dataset.py`. The fallbacks are exact
   and pinned, so the code is complete and the *effect* is pending.
7. **The estimate ledger has one more unsupported entry.** T2-4's
   +0.002 … +0.006 was reasoned from an over-allocation that is now absent at
   both measurement points (see above). Combined with T1-4's (F-9 refuted) and
   T1-7's (the validation criterion refuted), three Tier-1/2 estimates now rest
   on mechanisms the measurements did not find.
8. **`docs/` still describes the pre-Tier-3 network.** `README.md` is updated —
   the branch table, the parameter count, and a note that the archived numbers
   can no longer be regenerated from the checkpoints — but
   `docs/03_MODEL_ARCHITECTURE.md`, `docs/01_ABSTRACT_AND_OVERVIEW.md`'s block
   diagram and `docs/05`'s `Params : 7.88M` are not, which is the same state
   Tier 2 left them in after HD-1. They are wrong about Branch B's moments,
   Branch C's `band_reduce`, Branch D's positional table, the fusion's
   Perceiver and the parameter total. §4.5's claim table is the right place to
   fix all of it, and it should be written against the re-baseline rather than
   against this tree — a paper-facing rewrite that quotes untrained numbers
   would be the C-1 mistake in a different register.

---

## What is still true, and what is now the only thing left

Tier 3 completes the implementation matrix: **Phase 0, Tiers 1, 2, 4 and 3 are
all done.** What remains is not code.

> Until `python scripts/prepare_dataset.py` has been re-run, six items across
> two tiers are complete and inert (T4-2, T4-3, T4-4 rewrite the arrays; T4-1's
> grouped split needs the `groups.npy` that run writes; T3-7 and FU-4 need the
> `masks.npy` and `morphology.npy` it writes alongside), and **no number in this
> document is a trained result on the Tier-3 architecture.** Every Δ macro-F1 in
> the plan is still a claim.

The sequence §4.4 lays out has not changed, only shortened: rebuild the arrays,
re-baseline on P-fix, then train the Tier-3 architecture and report the three
numbers side by side — pre-Tier-3 on P-cur (the archived 0.8889), pre-Tier-3 on
P-fix, and Tier-3 on P-fix. The first two differences are the protocol
contribution; the third is the architecture's.


## Next steps, in order

```bash
# 1 · rebuild the arrays under the P-fix data contract (~hours, 36 GB)
#     writes groups.npy (T4-1), masks.npy (T4-3 / T3-7), morphology.npy (T4-4 / FU-4)
#     and applies the radiometry fix (T4-2) and the resize order (T4-3)
python scripts/prepare_dataset.py
python scripts/select_bands.py             # curve now runs to k = 256 (T4-6 / F-3)

# 2 · re-baseline the *architecture* on the P-fix protocol, both folds
python train.py data=spa40_90class_pfix
python train.py data=spa40_90class_pfix data.split_fold=1

# 3 · report three numbers side by side (§4.5):
#       pre-Tier-3 on P-cur   — the archived 0.8889, a claim about capture sessions
#       pre-Tier-3 on P-fix   — the leakage-corrected baseline
#       Tier-3    on P-fix    — the architecture's contribution
#     the first difference is the protocol's; the second is the redesign's
```

Step 2 is the first time any Tier-3 estimate becomes checkable. Until it runs,
the honest summary of this document is: **the plan's diagnosis is implemented
and gated; its prognosis is untested.**

## Reproducing

```bash
pytest tests/                             # 555 passed  (slow: ~40 min, see BR-2's cost)
python scripts/check_ast_no_op_move.py    # 96 identical, 78 declared, 40 new, 0 drift
python scripts/check_config_roundtrip.py  # 81/81, 30 additions, 1 removal
python scripts/capture_golden.py --verify # v1 vs the baseline, v3 vs the tree
ruff check src tests scripts train.py && black --check src tests scripts train.py && mypy
```
