# S26 · TTA anchor substitution rejected on calibration

2026-10-05 · **Complete: calibration gate fails on both folds.**
Zero new encoder fits, zero new head fits, zero new held-out predictions.

Plan SHA256: `da646aaf9f64978fc6c94469ece43baab9c530fc35ad7564d20f0854ebe84af8`.
The fixed intervention keeps the S23 head weights, single-view feature inputs and
scales, but replaces its probability anchor with calibrated equal-TTA fusion.
Its separate frozen rule requires a calibration F1 increase in both folds before
any held-out scoring or conditional head-seed evaluation.

| Fold | Saved single-anchor head calib F1 | Fixed TTA-anchor head calib F1 | Delta |
|---|---:|---:|---:|
| 0 | .789418 | .787023 | −.002395 |
| 1 | .783242 | .777630 | −.005613 |

Both fail. **H44's held-out gate is not evaluated**; seeds 1/2 are not evaluated
for this intervention. This is a calibration rejection, not evidence about its
unobserved held-out performance. Do not change the gate or retrospectively inspect
test predictions to rescue the candidate. The archive preserves the exact plan,
calibration receipt, rejection and completed file hashes under
`evidence/S26_tta_anchor/screen_results/`.

The learned correction is tied to the anchor against which it was trained.
Retain S23/S25 as the provisional learned system. The next bounded architecture
experiment should train the same small correction **against the intended TTA
anchor from the outset**, rather than substitute it after fitting. Generate
training-only frozen TTA logits once; reuse existing calibrated/test TTA logits
and cached single-view features. No new HSI encoder fit is needed for that screen.
