# S37 · Does the trained RGB branch gain from multi-layer readout and metric morphometrics?

| | |
|---|---|
| **Status** | see §2 |
| **Dates** | 2026-10-05 |
| **Plan** | `configs/research/s37_rgb_multilayer.json`, SHA-256 `96f3145816b04c1598f384fdcea1751952c418002f0e64c14676b799a420e3fc`. Frozen before S33 was scored (a factorial companion to S33) |
| **Code** | `scripts/run_rgb_multilayer.py` (freeze · train · run); `src/spectralquadnet/models/rgb_multilayer.py`; `src/spectralquadnet/experiments/multilayer_finetune.py`; tests `tests/unit/test_rgb_multilayer.py` |
| **Registers** | H57–H58 |

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
