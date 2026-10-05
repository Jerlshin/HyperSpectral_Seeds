# S37 · Does the trained RGB branch gain from multi-layer readout and metric morphometrics?

| | |
|---|---|
| **Status** | complete — **H57 fails**: no gain over the S32 trained branch |
| **Dates** | 2026-10-05 |
| **Plan** | `configs/research/s37_rgb_multilayer.json`, SHA-256 `96f3145816b04c1598f384fdcea1751952c418002f0e64c14676b799a420e3fc`. Frozen before S33 was scored (a factorial companion to S33) |
| **Code** | `scripts/run_rgb_multilayer.py` (freeze · train · run); `src/spectralquadnet/models/rgb_multilayer.py`; `src/spectralquadnet/experiments/multilayer_finetune.py`; tests `tests/unit/test_rgb_multilayer.py` |
| **Registers** | F128 · H57–H58 |

## 1 · Question and design

S31 found two RGB readout properties that a frozen probe could only partly use:
- intermediate blocks carry most of the within-acquisition signal (F115);
- metric morphometrics are the one RGB cue that adds transfer (F117).

The S32 trained branch reads only the final block of a scale-normalised crop. The S37 arm,
`mlm_vitb`, is the S32 ViT-B recipe and selection unchanged, except:
- **multi-layer readout:** [class, fg-mean] after each of the last 4 blocks, each through LayerNorm
  and a linear map to 768, averaged;
- **metric morphometrics input:** the 8 values, log size terms, z-scored on training rows and not
  augmented, mapped to 768 and added;
- **learning rates:** the new maps and the classifier train at the head rate.

The control is S32's `vitb` cells (same seed, folds, recipe and selection).
**H57-screen** (RGB, 4-view): Δ F1 ≥ .01, CI > 0, both folds, cross ≥ 0. H58 (system) is descriptive.
The two changes are not attributed separately (minimum compute); attribution is deferred to S39
ablations, if needed.

## 2 · Results (held-out, both corrected folds, seed 0)

Cells: 35.9 / 23.5 min on MPS. Calib F1 (id / 4-view): .787 / .802 and .780 / .807, against
S32's .775 / .789 and .784 / .817.

| Arm | F1 | Same / cross | From / to session 8 |
|---|---:|---:|---:|
| S32 trained ViT-B, 4 views (control) | .6642 | .7672 / .2812 | **.239** / .323 |
| **S37 multi-layer + morphometrics, 4 views** | **.6671** | .7733 / .2753 | .205 / **.346** |
| Equal fusion, S32 branch | .6977 | .8108 / .2953 | .254 / .337 |
| Equal fusion, S37 branch | .6987 | .8138 / .2893 | .231 / .348 |

- **H57-screen: fail.** +.0029 F1 [−.0053, .0108] (folds −.0008 / +.0065), cross −.0059 [−.0227, .0121].
- **H58 (system):** +.0010 [−.0053, .0075], cross −.0060.
- Single view: +.0074 [−.0010, .0162].

## 3 · Finding and consequence

- **F128**: once the RGB branch is trained, neither multi-layer readout nor explicit metric
  morphometrics adds held-out F1 or transfer. F115/F117 describe what a *frozen* probe was missing,
  and fine-tuning recovers it (the final block re-learns texture; scale is visible through the
  crop and through v5's own morphometrics in fusion). [E3, 1 seed]
- By the rule S39 froze before this scoring, the confirmation uses S32's `vitb` branch.
  Descriptively, S37's .6671 is the highest RGB-only F1 measured, statistically equal to .6642.

## 4 · Reproduce

```sh
PYTHONPATH=src:scripts python scripts/run_rgb_multilayer.py freeze          # before S33 scoring
for f in 0 1; do PYTHONPATH=src:scripts python scripts/run_rgb_multilayer.py train --fold $f; done   # ~1 h MPS
PYTHONPATH=src:scripts python scripts/run_rgb_multilayer.py run
```
