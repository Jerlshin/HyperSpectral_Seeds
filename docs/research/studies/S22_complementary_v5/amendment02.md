# S22 amendment 02: efficient architecture screening

2026-10-05, before any S22 model fit or held-out neural score. Explicitly authorized
by the user: retain both corrected acquisition directions, screen at one seed,
replicate only meaningful candidates, and advance a minimal learned multimodal model
when fusion consistently helps. This supersedes the six-fit execution instructions
in the original S22 runbook; the original scientific contract remains readable there.

The immutable parent is `configs/research/s22_complementary_v5.json`, SHA256
`f44c0d2e22f2849816cf2a8f2cdad8ae90084c695b06e4c181e7fc9b000f6f73`.
The executable child is `configs/research/s22_screening_amendment02.json`; its
identical evidence copy is `evidence/S22_complementary_v5/amendment02.json`.
Both have their own SHA files. Original driver bytes are archived under
`amendment_02_parent/code`. The child hashes the parent, archived drivers and all
current training inputs; hash protection remains active.

## Compute allocation and rationale

- Initial screen: exactly **folds 0/1 × seed 0**, two v5 fits rather than six.
- Keep both corrected S21 partitions, all8,624 held-out rows exactly once and both
  directions of all17 bridge varieties. Seeds never substitute for acquisitions.
- S20 already found +.040896 mean F1 from fixed RGB fusion over three historical
  HSI seeds; S21 found +.056970 against linear HSI on complementary folds. These
  are development evidence, with known historical coverage and transfer limits.
  Repeatedly confirming an intermediate average is a poor use of the next four fits.
- Preserve v5 R1, k32, architecture, global batch128, maximum200 epochs, patience40,
  calibration-only live/EMA checkpoint selection and historical TTA. No band sweep.
- **Declared runtime change:** this host has no CUDA. MPS is available outside the
  filesystem sandbox. Training-only batch128 profiling (zero held-out scoring)
  passed finite-gradient checks: warmed CPU ≈3.9s, MPS ≈1.8s. Use single-device MPS
  in float32 instead of planned two-T4 DDP/fp16, with exact overrides in the child.
  This changes backend/numerics and is not a byte-identical historical CUDA replay.
  Both new fold controls share the runtime; later matched confirmation must also
  freeze its runtime. No accelerator inference is made from the old CPU smoke.

## Decision rules fixed before execution

H40-screen requires two-fold mean fusion-minus-v5 macro-F1 ≥.02, paired variety
interval above0, **positive F1 gain in each fold**, and mean cross recall delta ≥0.
A positive cross-recall interval is additionally required for a supported transfer
statement on these varieties. Direction-specific cross deltas must be reported.
Class intervals quantify variety heterogeneity, not seed or session uncertainty.
Original three-seed H40 is not evaluated or relabelled as confirmed.

A clear consistent pass advances the smallest learned multimodal candidate using
frozen representations and an additive low-capacity head. A small or inconsistent
result does not automatically buy four more control fits. Strong direction conflict
requires diagnosis, with a separately recorded bounded replication if needed.
Seed confirmation is reserved for the frozen final learned candidate, corrected-fold
HSI control, RGB control if trainable, and fixed-fusion comparator; shared HSI seeds
supply the fixed-fusion rows without another encoder fit. Mechanism ablations receive
multi-seed runs only if they support a paper claim or lie near a decision boundary.

## Execute

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python scripts/run_complementary_v5.py train --fold 0 --seed 0
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python scripts/run_complementary_v5.py train --fold 1 --seed 0
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python scripts/analyze_complementary_v5.py
```

The amended plan is the drivers' default. Local MPS execution needs host GPU access.
Completed outputs refuse overwrite; unfinished matching cells support `--resume`.
The old plan and historical studies are not rewritten. S17/S18 remain reserved.
