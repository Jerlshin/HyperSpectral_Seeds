# S23 · Minimal learned multimodal residual head

2026-10-05 · **Complete after S22 passed its recorded development gate.**
Read the [results](results.md): F1 .604312; matched single-view gain +.010780
[.002308,.019329]. H41/practical point gates pass, but the margin over TTA fusion
is only .002717 and uncertain. [S24](../S24_branch_multimodal/README.md) tests which
learned correction branch is needed before expensive seed expansion.
The architecture and recipe below were written before that outcome. The exact
input/source plan was then sealed, SHA256
`46cdecf0c7e114c5ce78fd3ed055571be2638c224bc9e99cd834496121f2cdb7`.
Both corrected folds use head seed 0 and the selected S22 seed-0 HSI encoders.

## Scientific question

Can a small learned additive correction improve equal calibrated fusion while
preserving acquisition-direction behavior, without fine-tuning either encoder?
This tests the value of learning beyond simple averaging. It does not test
cross-attention, individual-kernel interactions, or novel-session robustness.

## Architecture and controls

The selected corrected-fold v5 checkpoint supplies normalized 256-D HSI features;
frozen DINOv2 supplies 384-D RGB features. Independent linear maps to 32 dimensions
and GELU activations feed a shared 90-class linear readout. The readout is initialized
to zero; outputs are **log(equal calibrated probabilities) + residual logits**.
The initial predictor is exactly the equal-fusion anchor. There are 23,514 trainable
parameters; both encoders and the S21 RGB LDA probe are frozen. Feature scales fit
outer training rows only. Correction branches add independently, consistent with
S20/S21's absence of a pairing-specific benefit. No cross-modality interaction is
claimed for these additive terms.

Fixed recipe: AdamW lr.001, weight decay.01, dropout.2, residual-square penalty.1,
batch 256, maximum 100 epochs, patience 15, seed0 on both corrected folds. Train rows
supply gradients; calibration macro-F1 selects the checkpoint. Epoch0 is eligible,
so calibration can reject every learned correction. There is no hyperparameter sweep.
Both encoders already fitted/classified those training rows; frozen features are
not independent cross-fitted predictions. This and single-training-scan support
may favor memorization. Transfer gates remain necessary.

Controls: HSI single-view, frozen RGB, equal single-view fusion, learned residual;
S22 equal TTA fusion is an additional practical comparator. Learning uses single-view
features; compare its gain against the matched single-view anchor and report whether
it exceeds the stronger TTA reference. Do not attribute a TTA difference to learning.
CPU fp32 single-view inference is audited against the saved CUDA fp32-forward logits (serialized float16) on
identical calibration and held-out rows. The calibration check precedes fitting;
the held-out check follows the recorded checkpoint selection. Numerical differences
are diagnostics and cannot select or tune the head.

**H41-screen (prospective):** learned minus equal-single mean F1 ≥.01, paired variety
interval >0, each fold F1 gain positive and mean cross recall delta ≥0. Practical
adoption additionally requires mean F1 and cross recall at least S22's equal-TTA result. Cross recall
and destination-session directions are reported; class intervals do not include seed
or session uncertainty. No multi-seed confirmation is bought for a rejected head.

## Execution guard

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python scripts/run_frozen_multimodal.py freeze
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python scripts/run_frozen_multimodal.py run
```

Freeze is conditional on complete hash-verified S22 analysis. It pins both exact
HSI checkpoints, source, parent plan, RGB features/probe exports and calibration
receipt before this candidate touches held-out rows. Existing plans/outputs refuse
overwrite. A failed head run currently needs a new documented output amendment;
checkpoint resumption is not implemented for this cheap head-only experiment.
Compact result replay and direction analysis are implemented in
`evidence/S23_frozen_multimodal/code/read_screen.py`; figures read only that archive.

If it passes, continue architecture screening with this as a provisional candidate;
confirm the frozen final system and matched controls at additional seeds before paper
claims. Both initialization levels must be sampled: independently train HSI encoders
and learned heads. Shared deterministic DINO/probe artifacts do not replicate RGB
pretraining. New crossed sessions/lots are a separate scientific requirement.

Precision clarification: training was fp16/GradScaler; final CUDA single-view/TTA
forward passes explicitly use fp32. CPU predictions agree with the original CUDA
single-view predictions on both folds; float16 serialization changes a few argmax
ties. The frozen raw audit label is preserved, clarified in
[the engineering receipt](../../evidence/S23_frozen_multimodal/engineering/precision_clarification.json).
