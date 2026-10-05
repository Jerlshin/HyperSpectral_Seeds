# S26 · Fixed TTA anchor substitution

2026-10-05 · **Complete: rejected on calibration, zero new held-out scores.**
Read the [results](results.md). The specification below was a prospective inference intervention after S25, before any candidate
calibration or held-out score. No new encoder or head fit is planned.

S23's matched learned gain is about .0105 F1 across head seeds, but its practical
margin over equal TTA fusion is about .0025. The correction was trained against
single-view averaging. Test whether the same correction composes usefully with
the existing stronger TTA probability anchor. Keep learned weights, single-view
HSI/RGB feature inputs, feature scales and fusion weight fixed. Replace only
`log(equal-single probabilities)` by `log(equal-TTA probabilities)`.

This is a fixed deployment-recipe intervention. It does not retrain the head
against TTA, average learned features over TTA, or establish an interaction claim.
Cached features and saved S22 TTA logits suffice; no new network inference is
required for this screen. Actual deployment would still compute the prescribed
HSI TTA probabilities and single-view feature inputs.

First evaluate the unchanged seed-0 head on calibration rows. Require macro-F1
strictly above its saved S23 calibration score on **both folds**, before scoring
any new held-out predictor. If either fold fails, reject this substitution with
zero held-out scoring. This does not imply poor unseen performance; it is an
efficient calibration rejection under the fixed rule.

If calibration passes, **H44-screen** requires seed-0 mean F1 gain ≥.01 over S22
equal TTA, paired variety interval >0, positive F1 gain each fold and nonnegative
mean cross delta. A supported transfer gain requires a positive cross interval.
Only after a clear H44 pass, evaluate the same intervention with already trained
head seeds 1/2 on both folds. No head/encoder training or weight sweep is allowed.

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python scripts/run_tta_anchored_screen.py freeze
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python scripts/run_tta_anchored_screen.py run
```

The separate plan preserves S22–S25 and pins their source, outcomes, caches, exact
head states, probability temperatures and TTA logit files. Overwrite and changed-
input guards remain in force. Existing acquisitions and fixed seed-0 HSI encoders
remain development data; this is not multi-seed full-system or external validation.
