# S28 · Head-seed sensitivity of the TTA-trained head

| | |
|---|---|
| **Status** | complete — H46-screen passes |
| **Dates** | 2026-10-05 |
| **Data** | S21 corrected folds; fixed S22 seed-0 v5 encoders; S27 train-TTA cache; S24 feature cache |
| **Code** | `scripts/run_tta_head_seeds.py`; shared recipe `src/spectralquadnet/experiments/residual_head.py` (tested in `tests/unit/test_residual_head.py`) |
| **Raw outputs** | `outputs/s28_tta_head_seeds/` |
| **Evidence** | [`evidence/S28_tta_head_seeds/`](../../evidence/S28_tta_head_seeds/) |
| **Registers** | F110 · D50 · H46-screen |

## Question and design

S27's pre-declared pass path: does the +.0131 practical gain survive head initialization?
Four cached head fits (head seeds 1/2 × both folds), S27 recipe unchanged, no encoder or TTA
inference. Plan SHA-256 `b0396c947dcc621606c6438ea76f9965e0b9fc648a2caa4e7c01bad2dc4a7ba6`.

Two guards came before any held-out row:
1. **Helper equivalence.** Seed 0 was refitted with the new shared helper on train/calib
   only. Its selected epoch and calibration F1 equal S27's exactly on both folds.
2. **Replay.** The saved S27 heads and anchors reproduce S27's held-out predictions with
   zero disagreements.

**H46-screen:** over head seeds 0/1/2 versus equal TTA, mean F1 gain ≥ .01 with paired
variety CI > 0, positive fold-mean gains, every seed's two-fold gain > 0, and mean cross
delta ≥ 0.

## Results (held-out, both corrected folds; 1.81 s wall)

| Head seed | Fold 0 F1 | Fold 1 F1 | Mean F1 | Cross recall | F1 gain vs equal TTA |
|---|---:|---:|---:|---:|---:|
| 0 (S27, reused) | .615717 | .613721 | .614719 | .236887 | +.013124 |
| 1 | .615950 | .616955 | .616453 | .244417 | +.014858 |
| 2 | .612023 | .618853 | .615438 | .241843 | +.013843 |

The three-head mean F1 is **.615537** (same recall .728742, cross .241049). The paired gain
over equal TTA is **+.013942 [.006019, .021463]**, with fold means +.011364 / +.016519 and a
descriptive seed SD of **.000871**. **H46 passes.**

Cross recall gains +.031559 [.003566, .064034]. This interval excludes zero, but it averages
three heads on *one* encoder per fold, ignores encoder and session variance, and is concentrated
in one direction. Toward session 8 recall is .290–.294 against .241 for equal TTA; away from
session 8 it is .184–.197 against .178. **This is not a transfer claim.**

## Decision

The TTA-trained head is a stable learned component on these encoders: head initialization
explains <.001 of the gain. Encoder-seed variance is still unmeasured; it belongs in the final
allocation, shared with every fusion control ([D50](../../DECISIONS.md), [S30 brief](../S30_final_confirmation/README.md)).
S29 then tested whether the RGB representation, not the fusion form, is the larger lever.

## Reproduce

```sh
PYTHONPATH=src:scripts OPENBLAS_NUM_THREADS=1 python scripts/run_tta_head_seeds.py freeze
PYTHONPATH=src:scripts OPENBLAS_NUM_THREADS=1 python scripts/run_tta_head_seeds.py run   # ~2 s
PYTHONPATH=src python docs/research/evidence/S27_tta_trained_head/code/archive_screen.py \
  outputs/s28_tta_head_seeds configs/research/s28_tta_head_seeds.json \
  docs/research/evidence/S28_tta_head_seeds/screen_results
```
