# Research progress and exact resume state

Updated 2026-10-05. **S27–S29 complete; development screening closed (D52). No job is running.
S30 (matched final confirmation) is proposed, not frozen; it needs authorization to spend
Kaggle GPU quota.** Read the [master plan](MASTER_RESEARCH_PLAN.md) (direction revision at the
end), [S29](studies/S29_rgb_backbone_screen/README.md) and the [S30 brief](studies/S30_final_confirmation/README.md).
The pre-S27 handoff is preserved verbatim in
[progress_history_pre_s27.md](evidence/S27_tta_trained_head/progress_history_pre_s27.md).

## What this phase executed

| Study | Frozen plan SHA-256 | Fits | Gate | Result |
|---|---|---:|---|---|
| S27 TTA-trained head | `780d564e…917978bf` | 2 heads | H45 | **pass**: +.013124 [.004438, .021629] over equal TTA |
| S28 head seeds 1/2 | `b0396c94…2dc4a7ba6` | 4 heads | H46 | **pass**: +.013942 [.006019, .021463], seed SD .000871 |
| S29 DINOv2 ViT-B/L RGB | `ee51803a…cf431aa01f` | 4 heads, 6 probes | H47 | **pass**: ViT-L head − ViT-S head +.012431 [.005679, .019846] |

Supporting compute and checks:
- Train-row TTA cache: 21.3 min CPU fp32. MPS was only 1.22× faster in the profile.
- CPU vs saved CUDA calibration TTA: max |Δlogit| .0039, 0 argmax flips.
- Frozen ViT-B/L extraction: 19 min on MPS; ViT-S re-extraction matches S20 to 6.3e-5.
- Zero encoder fits. Every replay audit was exact:
  - S27 anchor = S22 equal-TTA predictions;
  - S28 helper refit = S27 selection, and saved heads replay exactly;
  - S29 ViT-S = S21, equal_s = S22, head_s = S27.

## Current numbers (S21 corrected folds, seed-0 v5 encoders, held-out, mean of both folds)

| System | Mean F1 | Same recall | Cross recall | Cross recall away / toward session 8 |
|---|---:|---:|---:|---:|
| HSI v5 TTA | .559098 | .668820 | .208578 | .178 / .240 |
| Equal fusion, DINOv2-S (S22) | .601595 | .715099 | .209490 | .178 / .241 |
| TTA-trained head, DINOv2-S (S27; 3 head seeds .615537) | .614719 | .729846 | .236887 | .184 / .290 |
| Equal fusion, DINOv2-L | .620455 | .731834 | .233346 | .173 / .294 |
| **TTA-trained head, DINOv2-L (selected, S29)** | **.627150** | .740818 | .247998 | .180 / .316 |

## What was learned

1. Training against the deployment anchor works where post-hoc substitution failed (S26). The
   recorded mechanistic prediction of a smaller gain was wrong (F109). The gain is stable over head
   seeds (F110).
2. The RGB representation is a bigger lever than the fusion form. ViT-L lifts RGB alone by .0445
   and RGB transfer by +.034 (CI > 0), and fixed fusion with ViT-L already beats the ViT-S learned
   system (F111).
3. The learned head's margin shrinks to +.0067 [.0001, .0133] over fixed ViT-L fusion, so it is
   now a borderline component (F112).
4. **Dominant bottleneck:** away-from-session-8 recall is ≈ .18 for every system, all cross gains
   go to session-8 destinations, and 57% of cross errors land on a class trained in the test
   kernel's session (14% chance) (F113). This is acquisition-limited.

## Next exact actions

1. **S30 (needs the owner's go-ahead for Kaggle GPU quota).** Seal a new S22-runner amendment
   that also exports single-view embeddings and train-row TTA. Freeze contrasts C1–C4 with exact
   thresholds and source/input hashes. Run encoder seeds 1/2 × folds 0/1 (4 fits, ≈ 95 min wall
   on 2×T4 by S22's rate), then 12 CPU heads. C2 (head ViT-L − equal ViT-L ≥ .01) decides whether
   the paper system keeps a learned fusion component or is fixed equal ViT-L fusion.
2. **Crossed acquisitions (FW-42).** Design and lock the ≥ 4-session, two-lot pilot of S19 §5
   before applying the S30 system to it. This is the only test of the F113 bottleneck.
3. Do **not** run more backbones, resolutions, fusion forms or band expansions on the existing
   test scans (D52).

## Saved assets

| Path | Content |
|---|---|
| `outputs/s27_tta_profile/`, `outputs/s27_tta_cache/` | runtime profile; train-row TTA logits for both folds + calibration audit |
| `outputs/s27_tta_trained_head/`, `outputs/s28_tta_head_seeds/` | heads, selections, predictions |
| `outputs/s28_rgb_features/` | frozen DINOv2 ViT-B/L features + provenance (extracted before renumbering to S29) |
| `outputs/s29_rgb_backbone_screen/` | probes, ViT-B/L heads, predictions |
| `docs/research/evidence/S27…S29*/screen_results/` | compact evidence with `ARCHIVED.json` hash manifests; metric arithmetic replayed |
| `~/.cache/torch/hub/checkpoints/dinov2_vit{b,l}14_pretrain.pth` | public checkpoints; SHA-256 in provenance |

Code: `scripts/run_tta_trained_head.py`, `scripts/run_tta_head_seeds.py`,
`scripts/run_rgb_backbone_screen.py`, the shared head recipe
`src/spectralquadnet/experiments/residual_head.py` and `tests/unit/test_residual_head.py`.
All S22–S29 plans and outputs are unchanged and their input hashes verify. S17/S18 stay reserved.
Nothing from S22–S29 is committed to git yet. The working tree carries all of it.
