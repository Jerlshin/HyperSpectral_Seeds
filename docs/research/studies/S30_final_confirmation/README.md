# S30 · Matched final confirmation of the selected multimodal system

> **Superseded (2026-10-05) by [S39](../S39_final_confirmation/README.md) under [D55/D58](../../DECISIONS.md).** The trained RGB branch retired the S29 system that this brief would confirm. The brief is kept unchanged below; it was never frozen or run.

2026-10-05 · **Proposed design brief; not frozen, not executed.** It needs four GPU encoder
fits on the private Kaggle research host. That spends the owner's GPU quota, so it runs
only on explicit authorization.

## Why this, and why now

Development screening is finished (D52). S22–S29 selected the following system:
- frozen v5 HSI encoder, k32 reflectance, R1 TTA;
- frozen DINOv2 **ViT-L/14** RGB with the S21 shrinkage-LDA probe;
- calibrated equal probability fusion;
- the 43,994-parameter TTA-trained additive head.

Every number so far uses **one HSI encoder seed per fold**. Two open questions can only be
answered with matched encoder seeds:

1. **Is the learned head needed?** On ViT-L it adds only +.0067 [.0001, .0133] over fixed equal
   fusion (F112). That is below the .01 practical threshold and comes from one head on one encoder.
2. **Are the S27–S29 gains larger than encoder-seed variation?** This is gate G3 of the
   [workflow](../../WORKFLOW.md). Historical v5 grouped σ ≈ .009 (F30) is comparable to every
   learned margin found since S23.

New backbones, resolutions, heads or fusion forms are **out of scope**. They would reopen screening.

## Smallest sufficient allocation

| Item | Count | Cost basis |
|---|---:|---|
| New v5 HSI fits: encoder seeds 1, 2 × corrected folds 0, 1 | **4** | S22 runtime: 2 fits = 47.3 min bootstrap wall on 2×T4 → ≈ 95 min wall for 4 (estimate) |
| Per new encoder, exported in the same GPU job | — | calib/held-out TTA logits (as S22); single-view logits + 256-D embeddings on train/calib/held-out; **train-row TTA logits** (avoids ≈ 20 min CPU per encoder-fold measured in S27) |
| Heads (CPU, seconds) | 6 ViT-L + 6 ViT-S | head seed = encoder seed, one head per encoder (confirmation queue rule) |
| RGB probes | 0 new | deterministic S29 probes are reused; refits are not seed evidence |

Seed 0 cells reuse S22/S27/S29 outputs, which gives six encoder × fold cells per arm.

## Arms per cell

HSI TTA · RGB ViT-L · equal ViT-S · **equal ViT-L** · head ViT-S · **head ViT-L**.

## Pre-declared contrasts (to be frozen with exact thresholds before any new fit is scored)

Means are over 3 encoder seeds × 2 folds. Paired variety intervals are computed after averaging
seeds, and the seed-to-seed SD of each paired delta is reported. A claim needs Δ ≥ threshold,
CI > 0, a positive mean on each fold, and Δ > 2 × seed SD (G3).

| ID | Contrast | Threshold | Consequence |
|---|---|---:|---|
| C1 | equal ViT-L − HSI TTA | .02 | multimodal value of the deployed system |
| C2 | head ViT-L − equal ViT-L | .01 | **pass: final system keeps the head; fail: final system = fixed equal ViT-L fusion (zero learned fusion parameters)** |
| C3 | equal ViT-L − equal ViT-S | .01 | RGB backbone contribution |
| C4 | head ViT-S − equal ViT-S | .01 | replicates S27/S28 across encoders (mechanism record) |

Cross recall is reported per contrast and per acquisition direction. A transfer claim needs a
cross interval > 0 **and** a gain in the away-from-session-8 direction, which no system has
shown so far (F113).

## Engineering preconditions

- Extend the S22 CUDA runner as a *new* sealed amendment, never editing the S22 parent. It adds
  single-view embedding and train-row TTA exports and keeps R1, batch 128, ≤200 epochs,
  patience 40 and calib live/EMA selection identical.
- Before freezing, verify on the seed-0 checkpoints that the GPU-exported train TTA matches the
  S27 CPU cache (tolerance as in S27: max |Δlogit| ≤ .05, ≤ 3 argmax flips).
- Credentials stay local. Training-only runtime probe before launch.

## After S30

Write the paper's method/results only from S30. The dominant open problem stays
acquisition-limited transfer. All 17 bridges touch session 8, and away-from-session-8 recall is
≈ .18 whatever the model. The next scientific investment is the crossed-session/lot acquisition
pilot of [S19 §5](../S19_next_generation_strategy/experiments_and_paper.md) and FW-42. It should
be locked *before* the final system is applied to it.
