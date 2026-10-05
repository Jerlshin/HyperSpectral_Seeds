# S24 · Which learned correction branch is needed?

2026-10-05 · **Complete.** [Results](results.md): neither simplification qualifies;
H42's strict necessity gate fails by the HSI-control margin. The full head remains
provisional. The specification below was a prospective bounded ablation after the completed S23 screen, before
any S24 fitting or held-out scoring. S23 gains .010780 F1 over equal single-view
fusion, but only .002717 over equal TTA fusion. The next efficient question is
whether its two learned feature branches are needed, before more encoder seeds.

Two arms: HSI-feature correction only and RGB-feature correction only. **Both
modalities remain in the fixed equal probability anchor.** These are correction-
branch removals, not unimodal predictors. Independent encoder/probe weights remain
frozen. The same 32-D projection/readout/dropout/loss, S23 feature scales and
temperatures, initialization draw order, train rows, calib selection with epoch 0,
100-epoch cap and patience 15 are retained. The removed branch contributes nothing
to the forward pass or gradients. Active trainable parameters: HSI correction
11,194; RGB correction 15,290; S23 both-feature correction 23,514.

Exactly four head fits: both arms × both corrected folds, seed 0. No new encoder
fit, GPU training, recipe sweep, or automatic seed expansion. Frozen HSI features
are extracted once per fold, with train/calib before fitting and held-out only
after both head checkpoints are selected. Features are cached so later head work
need not repeatedly extract the HSI baseline. The S23 both-feature arm and equal-
fusion controls are reused from their saved results; they are not retrained.

**H42-screen:** retain both learned feature branches as necessary only if S23 beats
each removed-branch control by mean F1 ≥.005, paired variety interval >0, positive
gain on each fold and nonnegative mean cross-recall delta. These are branch-removal
controls with fewer parameters, not a capacity-matched causal isolation.

**Provisional simplification rule:** a branch-only correction must pass S23's H41
criteria versus equal-single, match/exceed S22 TTA fusion's mean F1 and cross recall,
and remain within .005 F1 and .01 cross recall of S23. Among qualifying branches,
prefer fewer active trainable parameters. This is explicit architecture development
on reused acquisitions; eventual matched confirmation is required. Failure to prove
both branches necessary is not proof that they can never help.

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python scripts/run_branch_multimodal.py freeze
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python scripts/run_branch_multimodal.py run
```

Freeze pins the S23 plan, source, complete outcomes, feature scales and all inherited
checkpoint/data/probe inputs. Source/plan/output guards refuse mismatch and overwrite.
Branch-isolation and initial-anchor unit tests pass before execution. Calibration
selects checkpoints; test outcomes cannot alter the fixed recipe. Class intervals
exclude encoder/head seed and independent-session uncertainty. All 17 bridges still
touch session 8; new acquisitions remain required for broader transfer claims.
