# S27 · Train the small correction against its intended TTA anchor

| | |
|---|---|
| **Status** | complete — see [results](results.md) |
| **Dates** | 2026-10-05 |
| **Data** | `dataset_u430k32` k32 reflectance, S21 corrected complementary folds 0/1; frozen S22 seed-0 v5 encoders; S20 DINOv2-S RGB features |
| **Code** | `scripts/run_tta_trained_head.py` (profile · cache · freeze · run); `evidence/S27_tta_trained_head/code/` |
| **Raw outputs** | `outputs/s27_tta_profile/`, `outputs/s27_tta_cache/`, `outputs/s27_tta_trained_head/` (git-ignored) |
| **Evidence** | [`evidence/S27_tta_trained_head/`](../../evidence/S27_tta_trained_head/) |
| **Registers** | F109–F110 · D50 · H45-screen |

The pre-freeze design brief is preserved unchanged as [design_brief.md](design_brief.md).

## 1 · Question

Does the S23 23,514-parameter additive residual head add a *practical* improvement
over the strong fixed equal v5-TTA/RGB fusion, once it is trained against that anchor
rather than against single-view fusion? This test can close or retain the learned
fusion-head line (D48/D49). It cannot test transfer to new sessions.

## 2 · Trigger

S23/S25: +.0105 F1 over the matched single-view anchor, but only +.0025 (CI spans 0)
over equal TTA. S26: substituting the TTA anchor into the already trained head fails
calibration on both folds (F108). The remaining fair test trains the head against
the deployment anchor from the start ([D49](../../DECISIONS.md)).

## 3 · Frozen design and gate

Plan `configs/research/s27_tta_trained_head.json` was frozen after the train-only cache
and before any head fit; its SHA-256 is recorded in the results and `.sha256` sidecar.

- **Model/recipe:** S23 unchanged: HSI256 and RGB384 → 32-D GELU, summed, dropout .2,
  zero-initialized 90-class readout added to log anchor probabilities; AdamW lr .001,
  wd .01, residual-square penalty .1, batch 256, ≤100 epochs, patience 15, head seed 0.
- **Anchor:** equal mean of softmax(HSI TTA / S22 calib temperature 1.0) and
  softmax(S21 DINO probe / S23 RGB temperature 2.0). Train rows use the new CPU TTA cache;
  calib/held-out rows use S22's saved CUDA TTA logits, so the held-out anchor *is* S22's
  equal-TTA system (replay audit: zero prediction disagreements required).
- **Selection:** train gradients; calib macro-F1 selects the checkpoint, epoch 0 eligible.
  Held-out cache is read only after both fold checkpoints are saved.
- **H45-screen:** learned − equal TTA: mean F1 ≥ .01, paired variety CI > 0, positive F1
  each fold, mean cross-recall delta ≥ 0. Transfer support also needs cross CI > 0.
- **Pre-declared decision:** pass → retain; cheap head seeds 1/2, then matched encoder
  confirmation. Fail → close the learned fusion-head line; fixed equal TTA fusion remains
  the multimodal system; no rescue by recipe change.

## 4 · Runtime assessment and train-only cache (before freezing)

Profile on 256 fold-0 training rows (12-view fp32 TTA): CPU 8.64 rows/s, Apple MPS
10.56 rows/s, MPS-vs-CPU max logit difference 1.2e-5, no argmax change. MPS is only
1.22× faster, so CPU fp32 was chosen to match the S23–S26 feature path. Both folds'
training rows (3,681/3,683) plus a calibration audit were cached once.

CPU TTA reproduces the saved CUDA calibration TTA logits to a maximum absolute difference
of .0039 with zero argmax disagreements, which validates the train-row anchor numerically.

**Observation recorded before any S27 outcome:** the frozen HSI TTA anchor classifies
**100% of outer-training rows correctly** (single-view: 100%/99.65%). The encoder has
memorized its own training rows, so the head never sees an anchor error to correct on
train. It can only learn confidence reshaping that may or may not transfer. S23 had the
same structure (anchor train accuracy 99.2–99.8%) and still gained on calibration. The
mechanism therefore predicts a smaller gain over TTA than over single-view fusion.
This prediction is recorded here; it does not change the frozen gate.

## 5 · Reproduce

```sh
PYTHONPATH=src:scripts OPENBLAS_NUM_THREADS=1 python scripts/run_tta_trained_head.py profile  # ~1 min
PYTHONPATH=src:scripts OPENBLAS_NUM_THREADS=1 python scripts/run_tta_trained_head.py cache    # ~17-20 min CPU
PYTHONPATH=src:scripts OPENBLAS_NUM_THREADS=1 python scripts/run_tta_trained_head.py freeze
PYTHONPATH=src:scripts OPENBLAS_NUM_THREADS=1 python scripts/run_tta_trained_head.py run      # seconds
PYTHONPATH=src python docs/research/evidence/S27_tta_trained_head/code/archive_screen.py \
  outputs/s27_tta_trained_head configs/research/s27_tta_trained_head.json \
  docs/research/evidence/S27_tta_trained_head/screen_results \
  outputs/s27_tta_profile/profile.json outputs/s27_tta_cache/calib_audit.json
```

Every stage refuses to overwrite its output folder; `freeze` refuses changed caches or a
calibration audit beyond 3 argmax disagreements / .05 logit difference.
