# Research progress and exact resume state

Updated 2026-10-06. **S40 (v5 on all 215 bands, seed 0, both folds) is prepared and frozen for Kaggle
T4 x2; it has not run.** The state of 2026-10-05 (evening) follows unchanged.

Updated 2026-10-05 (evening). **S31–S38 complete. Proposed architecture: SeedNet-MX (S36, D58).
S39 matched confirmation is frozen: its RGB seed cells run locally (queued), and its HSI seed cells
are bundled but need the owner's authorization to spend Kaggle GPU quota. S30 is superseded.**

Before this phase:
- the pre-S31 handoff is preserved verbatim in
  [progress_history_pre_s31.md](evidence/S36_next_generation_architecture/progress_history_pre_s31.md);
- the pre-S27 handoff is in [progress_history_pre_s27.md](evidence/S27_tta_trained_head/progress_history_pre_s27.md).

## What this phase executed (owner directive, D53; all local: Apple M5 MPS + CPU)

| Study | Plan SHA-256 | Fits | Gate | Result |
|---|---|---:|---|---|
| S31 frozen ViT-L readouts + acquisition audit | `29ef87f3…` | 0 | H48 | **fail (cross clause)**: +.1258 RGB F1, cross −.007 |
| S32 trained foreground-token RGB | `251465cc…` | 4 | H50, H51 | **pass**: RGB .6642 (+.1692), fused .6977 (+.0773), cross CIs > 0 |
| S33 measured-optics blur + rendering | `2873190a…` | 2 | H52, H53 | **fail**: from-s8 −.031 |
| S34 strong-RGB reassessment | `cb8063a7…` | 0 | H54 | **pass**: +.0706 over the S29 learned system |
| S35 class-conditional rendering (CCAR) | `37da5f94…` | 0 | H55, H56 | **fail**: ±.003 |
| S37 multi-layer + morphometric trained readout | `96f31458…` | 2 | H57 | **fail**: +.0029 n.s. |
| S38 HSI role specialization | `e5ea3571…` | 0 | H59 | **fail**: cross −.037 |
| S36 architecture design | — | — | — | SeedNet-MX; kernel/scan error decomposition |
| S39 final confirmation | `d1db0694…` (+ HSI amendment06 `ce3ded4b…`) | 4 RGB done, 4 HSI pending | M1–M4 + G3 | **M3 pass** (RGB .6587 ± .005); M1/M2/M4 await the GPU seeds |

The phase used 12 RGB network fits (including 4 S39 seed cells) plus one 57-min frozen-feature extraction. No GPU quota was spent.

## Current numbers (S21 corrected folds, seed 0, held-out, mean of both folds)

| System | F1 | Same / cross | From / to session 8 |
|---|---:|---:|---:|
| **SeedNet-MX = trained ViT-B RGB + v5, equal fusion** | **.6977** | .8108 / .2953 | .254 / .337 |
| trained RGB alone (S32 ViT-B) | .6642 | .7672 / .2812 | .239 / .323 |
| trained RGB alone (S37 multi-layer + morph; = S32 within noise) | .6671 | .7733 / .2753 | .205 / .346 |
| frozen ViT-L, best readout (S31 `last4_tta`) | .6208 | .7417 / .1688 | .086 / .252 |
| S29 learned system (retired) | .6272 | .7408 / .2480 | .180 / .316 |
| HSI v5 TTA | .5591 | .6688 / .2086 | .178 / .240 |

## Next exact actions

1. **Done (21:42):** the S39 RGB seed cells and M3. Trained RGB averages .6587 over 3 seeds
   (SD .005), +.1637 [.1466, .1804] over frozen, cross +.094; G3 passes (F129). No job is running.
2. **S39 HSI seeds: owner authorization required.** `kaggle kernels push -p outputs/s39_kaggle_push`
   runs 4 v5 fits (≈ 95 min on 2×T4). Download `s39_cuda_outputs.tar.gz` and place the cells at
   `outputs/s22_complementary_v5/f{0,1}_s{1,2}/`. Then run
   `PYTHONPATH=src:scripts python scripts/run_final_confirmation.py run` for M1–M4.
3. **Crossed acquisition pilot (FW-42).** Lock SeedNet-MX first. The next novel mechanism,
   cross-acquisition cross-modal consistency (FW-48), needs ≥ 2 training acquisitions per variety.
4. **S40, 215-band v5 (D59).** Plan `configs/research/s40_full215_v5.json` (`744c65bc…`), cells
   f0_s0 and f1_s0 only. The cube `dataset_refl215_f16/` (15.26 GB) is built and verified locally but
   **not uploaded**. Upload it as a private Kaggle dataset, then follow README §10 "S40":
   `run_full215_v5.py link → check → run --nproc-per-node 2 --stream → archive`. Unpack
   `s40_full215_v5_outputs.tar.gz` at the repo root and compare with S22 f{0,1}_s0 as in the
   study page §6. The `run` command also resumes after an interruption.
5. Do **not** rerun the falsified mechanisms (S33, S35, S37, S38) or build learned/kernel-level
   fusion on this design (D56/D57).

## Saved assets

| Path | Content |
|---|---|
| `outputs/s31_rgb_readouts/` | frozen ViT-L 4-view, last-4-block readouts + 4.5 GB identity-view token cache |
| `outputs/s32_rgb_finetune/<arm>_f<fold>/`, `outputs/s33_rgb_acquisition/`, `outputs/s37_rgb_multilayer/` | checkpoints, traces, 4-view logits + embeddings for all 8,624 rows; S33 rendered logits |
| `outputs/s3{1,2,4,5,7,8}_*/` and `outputs/s3{2,3,7}_*/screen/` | sealed screens (COMPLETED.json hash manifests) |
| `outputs/s39_final_confirmation/` | S39 RGB seed cells + `partial_rgb/` (M3) |
| `outputs/s39_kaggle_push/` | built, **unpushed** S39 HSI Kaggle bundle |
| `docs/research/evidence/S31…S39*/` | archived evidence with replayed metric arithmetic; S36 search log, error decomposition, validation receipt |

**Code:**
- runners `scripts/run_{rgb_readout_audit,rgb_finetune,rgb_acquisition,multimodal_reassessment,regime_rendering,rgb_multilayer,modality_roles,final_confirmation}.py`;
- modules `src/spectralquadnet/experiments/{rgb_readout,screen_metrics,rgb_finetune,multilayer_finetune}.py`
  and `src/spectralquadnet/models/{rgb_branch,rgb_multilayer,acquisition_aware}.py`;
- 21 unit tests.

**Hash-pinned by frozen plans, do not edit:** `rgb_finetune.py`, `rgb_branch.py`,
`screen_metrics.py`, `rgb_readout.py`, `multilayer_finetune.py`, `rgb_multilayer.py` and their runners.
