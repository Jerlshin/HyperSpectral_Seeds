# S31–S39 validation receipt (2026-10-05)

- `verify_plan` passes for every frozen plan of the phase: S31, S32, S33, S34, S35, S37, S38, S39 and
  the S39 HSI amendment06. All pinned inputs match.
- Every scored screen was archived with `S27_tta_trained_head/code/archive_screen.py`. That tool
  replays the metric arithmetic from saved predictions (1e-12) and checks that every arm holds out
  all 8,624 kernels exactly once (`coverage.json` per study).
- Replay audits:
  - S31 `cls` and `equal_cls` reproduce S29 `dino_l` / `equal_l` with 0 disagreements, and the
    identity-view block-24 class token equals S29's features exactly (max |Δ| 0.0);
  - S32's frozen reference and S34's frozen component replay S31's sealed predictions;
  - the shared scoring helper reproduces S29's H47 contrast and direction recalls exactly.
- `ruff` and `mypy --strict`: clean on the 8 new runners, 4 experiment modules and 3 model modules.
- `pytest`: 21 new unit tests pass (`test_rgb_branch`, `test_rgb_multilayer`, `test_regime_rendering`,
  `test_acquisition_aware`, `test_final_confirmation`). The pre-existing four failures documented in
  `S29.../validation/preservation_commit_20261005.md` are unchanged and unrelated.
- Ordering of freezes:
  - S32 was frozen before S31's scoring; S34 before S32's; S35 and S37 before S33's;
    S39 before S37's.
  - Rules resolved later (reference readout, RGB branch) read sealed outputs whose
    `COMPLETED.json` hashes are verified at use.
- Disclosed deviations:
  - S31's extract stage ran from a runner revision differing only by a type-ignore comment;
  - ViT-L in S32 tunes its top 8 of 24 blocks (memory, profiled before freezing);
  - an interrupted waiting loop was killed without affecting any job.
- Credentials: none in any committed file or the built (unpushed) Kaggle bundle.
