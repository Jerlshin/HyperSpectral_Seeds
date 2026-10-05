# S27 · Results: the TTA-trained head passes its practical gate

2026-10-05 · **Complete. H45-screen passes.** Two head fits (head seed 0, both corrected
folds), zero encoder fits. Plan SHA-256
`780d564ec7d30487615f9658f4dd982fd0eb8752c7b0a25285dc69ac917978bf`.

## Execution

| Stage | Wall time | Notes |
|---|---:|---|
| Profile (fold-0 train rows only) | ~1 min | CPU 8.64 vs MPS 10.56 rows/s; CPU chosen |
| Train-row TTA cache + calib audit | 1,278 s (21.3 min) | fold 1 shared the machine with S29 feature extraction |
| Freeze → run (2 heads, held-out scoring) | 0.74 s | selected epochs 4 / 7; patience stops 19 / 22 |

Calibration (selection split, not a performance claim): anchor .753602 → head .795271 (fold 0);
.768540 → .792109 (fold 1). The equal-TTA anchor's calibration F1 is *below* S23's single-view
anchor on fold 0 (.7536 vs .7622) but above it on fold 1. The held-out anchor reproduces S22's
saved equal-TTA predictions with **zero disagreements** on both folds.

## Held-out results (S21 corrected folds, k32 reflectance, every kernel held out once)

| Predictor | Fold 0 F1 | Fold 1 F1 | Mean F1 | Accuracy | Same recall | Cross recall |
|---|---:|---:|---:|---:|---:|---:|
| S22 equal v5-TTA/RGB fusion (anchor) | .603199 | .599991 | .601595 | .619317 | .715099 | .209490 |
| S23 single-anchor head, seed 0 (reference) | .607767 | .600857 | .604312 | — | .716135 | .233252 |
| **S27 TTA-trained head, seed 0** | **.615717** | **.613721** | **.614719** | .636479 | .729846 | .236887 |

**H45-screen (pre-registered): pass.** Learned − equal TTA: **+.013124 F1 [.004438, .021629]**,
fold gains +.012518 / +.013730, cross recall **+.027397 [−.002791, .062940]** (+.008552 / +.046242).
All four criteria hold. The cross interval spans zero, so **transfer improvement is not supported**.
Descriptively, the TTA-trained head beats the S23 single-anchor head by +.010407 [.006183, .014614].

## Error accounting (descriptive, `evidence/S27_tta_trained_head/error_analysis.json`)

Relative to equal TTA, the head changes 447 / 630 predictions (fold 0 / 1). It rescues
139 / 202 kernels and harms 68 / 125, a net of +148 correct kernels out of 8,624. Session
attraction of cross-session errors (the share of errors predicting a class trained in the test
kernel's own session) moves .565→.587 on fold 0 and .596→.562 on fold 1, so it stays about 57%
against 14% chance. The head does not touch the session-fingerprint mechanism. Direction recall
(17 bridge varieties each): toward session 8 .241→.290, away from session 8 .178→.184. As in
S23, the cross gain sits almost entirely in the session-8-destination direction.

## Interpretation

The prediction recorded before outcomes (README §4) was **wrong**. A head trained against an
anchor that is ~100% correct on its training rows still learned a correction that transfers to
held-out scans: +.0131 over the stronger anchor, versus S23's +.0105 over its weaker anchor.
The S26 failure was therefore specific to *post-hoc* anchor substitution, not to learning
against TTA. The likely mechanism is calibration-level confidence reshaping of the anchor rather
than error correction, because the anchor never errs on training rows. This interpretation is
not separately tested.

## Decision

Per the frozen pass path: retain the TTA-trained head as the provisional learned multimodal
system, run the cheap head-seed sensitivity ([S28](../S28_tta_head_seeds/README.md)), and place
matched encoder-seed confirmation in the final allocation. See [D50](../../DECISIONS.md),
[F109](../../FINDINGS.md). Its RGB input was then revisited in
[S29](../S29_rgb_backbone_screen/README.md).

## Evidence

`evidence/S27_tta_trained_head/screen_results/`: metrics, per-class, predictions, learning
curves, selections, hypothesis, profile, calibration audit, acquisition directions, coverage and an
`ARCHIVED.json` hash manifest (metric arithmetic replayed from saved predictions). Also
`bottleneck_diagnostic.json` and `error_analysis.json` with their code. Train TTA caches and
head weights: `outputs/s27_tta_cache/`, `outputs/s27_tta_trained_head/`.
