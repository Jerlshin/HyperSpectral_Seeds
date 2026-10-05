# S22 compatibility amendment03

2026-10-05, after amendment02's attempted fold0 failed before a completed epoch.
No checkpoint and no held-out predictions existed. The exact termination traceback,
epoch0 marker and provenance are archived under
`evidence/S22_complementary_v5/amendment02_failure/`; the complete failed directory
is `outputs/s22_aborted_amendment02/f0_s0/`. No cause for signal15 is asserted.

The child `configs/research/s22_screening_amendment03.json` changes train/eval
loader worker counts from2 to0. Source, architecture, splits, seed0 on both folds,
MPS/fp32, batch128, R1, calibration selection and TTA remain fixed. Worker scheduling
can change augmentation RNG consumption; this is a declared runtime compatibility
attempt, not an identical continuation of the aborted run. The child hashes parent02,
all current scientific inputs and the unchanged launcher/analyzer. Parent02 and the
original six-cell parent remain immutable. SHA256:
`1808076bcabd9189c55f78abc621e7fdd27b938834bea9f4c1f47263b82b8e76`.

Use these commands (the launcher default is amendment02; specify03 explicitly):

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python scripts/run_complementary_v5.py train --plan configs/research/s22_screening_amendment03.json --fold 0 --seed 0
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python scripts/run_complementary_v5.py train --plan configs/research/s22_screening_amendment03.json --fold 1 --seed 0
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python scripts/analyze_s22_screening.py
```

An unfinished matching cell can resume with `--resume`. The retry has completed
its first epoch and written a resumable checkpoint; scientific outcomes remain pending.

The preferred analysis entry point is `scripts/analyze_s22_screening.py`, separately
sealed by `configs/research/s22_analysis_code04.json` before any scoring. It sets
amendment03 as default and adds a local type annotation; scoring/statistics are
unchanged from the previously sealed analyzer. All old driver source remains intact.
The completion receipt records both training-plan and analysis-code hashes.
