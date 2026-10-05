# S22–S29 preservation receipt (2026-10-05, before the first S22–S29 commit)

Additive to `checks_20261005.md`; no frozen plan, evidence file or historical study was edited.

- Credential scan of every uncommitted file: no token/key material (only README's pointer to
  `~/.kaggle/kaggle.json`).
- `ruff check` on all S22–S29 runners, helpers and models: clean.
- `mypy --strict` on the S27–S29 code reproduces exactly the documented annotation-only
  exceptions in the two hash-pinned runners (`run_tta_head_seeds.py:138`,
  `run_rgb_backbone_screen.py:54,66`). Other strict findings are in pre-existing imported
  modules (`data/datasets.py`, `tracking/wandb_tracker.py`), unchanged since HEAD `17670c4`.
- `pytest tests/unit`: all pass except four tests that fail identically on a clean HEAD
  worktree (`17670c4`), so they are not S22–S29 regressions:
  - `test_amp_precision.py::test_branch_cs_power_normalisation_floor_survives_the_autocast_dtype`:
    the local torch build no longer produces the fp16 NaN gradient the test uses as its negative
    control (environment, not the guarded floor);
  - three `test_s15_plan.py` code-identity tests: the S15 digest guard pins the training code at
    `aed5257`; the later S20/S21 package edits (committed in `17670c4`) move the digest.
    The guard is deliberately not weakened.
