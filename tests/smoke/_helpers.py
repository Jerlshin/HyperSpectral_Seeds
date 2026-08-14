"""Shared helpers for the smoke tier — plain module, importable by both files.

Kept out of ``conftest.py`` because ``tests/`` is not a package: pytest loads a
conftest by path, but a *helper* has to be importable by name from both smoke
modules, and pytest's rootdir-prepend import mode makes this directory
importable as top-level ``_helpers``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN = REPO_ROOT / "train.py"

N_CLASSES = 8
BUNDLES_PER_CLASS = 2
PATCHES_PER_BUNDLE = 12
N_BANDS = 8
SPATIAL = 16

#: The **acquired** band count — what `configs/data/hsi256_grouped.yaml` declares
#: and what a default `python train.py` reads. The `full_spectrum_dataset`
#: fixture builds a miniature cube at exactly this width so the primary
#: composition's own `data.num_bands` can be left untouched: every other smoke
#: run overrides it down to `N_BANDS`, which exercises the machinery but not the
#: number the study is about.
FULL_BANDS = 256
#: Spatial side for the full-spectrum fixture. The stem's spatial strides are
#: fixed at (1, 2, 2), so 16x16 runs the same operators as 64x64 at 1/16 the
#: cost — the *spectral* axis is what this fixture exists to keep at full width.
FULL_SPATIAL = 16


def tiny_overrides(dataset: dict[str, str], output_dir: Path, **extra: Any) -> list[str]:
    """A structurally complete run at a size that finishes in seconds.

    Every stage, every loss term, every evaluation path is exercised; only the
    magnitudes shrink. Anything that skipped a component here would be testing
    a different pipeline from the one that ships.
    """
    base: dict[str, Any] = {
        "data.patches_data": dataset["patches_data"],
        "data.labels_path": dataset["labels_path"],
        "data.groups_path": dataset["groups_path"],
        "data.masks_path": dataset["masks_path"],
        "data.morphology_path": dataset["morphology_path"],
        "data.wavelength_path": dataset["wavelength_path"],
        "data.gain_path": "",
        "data.num_bands": N_BANDS,
        "data.num_classes": N_CLASSES,
        # Scaled to the same *fraction* of the spectral axis the shipped configs
        # use, so the augmentations mean the same thing at this band count.
        "data.cutmix_bands": 2,
        "data.max_cutout_bands": 1,
        "single.epochs": 2,
        "single.batch": 8,
        "single.warmup_ep": 1,
        "single.mixup_epochs": 1,
        "single.margin_warmup_start": 2,
        "single.margin_warmup_end": 2,
        "single.patience": 5,
        "single.bal_n_cls": 4,
        "single.bal_n_spc": 2,
        "model.stem_channels": 16,
        # `spectral_stride_schedule(8, folded_depth=1)` is `(2, 2, 2)`, so the
        # miniature run exercises a genuinely strided spectral reduction —
        # 8 → 4 → 2 → 1 — rather than the `(1, 1, 1)` an 8-band cube would get at
        # the shipped depth of 8. The primary path's `(8, 2, 2)` at 256 bands is
        # covered by `tests/unit/test_branch_c_stem.py`, which does not need a
        # 36 GB cube to check a shape.
        "model.stem_folded_depth": 1,
        "model.spatial_width_mult": 0.25,
        "model.spectral_hidden": 32,
        "model.index_bank_size": 8,
        "model.continuum_depths": 4,
        "model.aux_head_hidden": 16,
        "evaluation.bootstrap_samples": 32,
        "tta_spatial": 2,
        "tta_spectral": 1,
        "device": "cpu",
        "runtime.num_workers": 0,
        "runtime.compile": "off",
        "runtime.diagnostics_interval": 1,
        "tracking.backend": "console",
        "output_dir": str(output_dir),
        "hydra.run.dir": str(Path(output_dir) / "hydra"),
    }
    base.update(extra)
    return [f"{k}={v}" for k, v in base.items()]


def full_spectrum_overrides(dataset: dict[str, str], output_dir: Path, **extra: Any) -> list[str]:
    """The primary composition at its **shipped** band count, on a miniature cube.

    Deliberately does **not** override `data.num_bands`, `data.cutmix_bands`,
    `data.max_cutout_bands` or `model.stem_folded_depth`: those four are the
    256-band design, and a smoke test that overrode them would exercise a
    different pipeline from the one `python train.py` runs. Everything it *does*
    override is a magnitude — class count, epochs, batch, widths — so what is
    tested is the shipped spectral configuration end to end.
    """
    overrides = tiny_overrides(dataset, output_dir, **extra)
    dropped = (
        "data.num_bands=",
        "data.cutmix_bands=",
        "data.max_cutout_bands=",
        "model.stem_folded_depth=",
    )
    return [o for o in overrides if not o.startswith(dropped)]


#: The audited architecture's keys, shrunk to match. Kept beside
#: :func:`tiny_overrides` so the control arm and the default arm are configured
#: from one place and cannot drift into incomparable sizes.
AUDITED_TINY: dict[str, Any] = {
    "stage1.epochs": 2,
    "stage1.batch": 8,
    "stage1.patience": 5,
    "stage2.epochs": 1,
    "stage2.batch": 8,
    "stage2.bal_n_cls": 4,
    "stage2.bal_n_spc": 2,
    "stage2.patience": 3,
    "stage2.warmup_ep": 1,
    "stage3.epochs": 1,
    "stage3.cycle_len": 1,
    "stage3.swa_warmup_cycles": 0,
    "model.specf_dim": 16,
    "model.specf_heads": 2,
    "model.specf_layers": 1,
    "model.specf_tokens": 2,
    "model.grid_size_a": 2,
    "model.grid_size_d": 2,
    "model.fusion_rank": 8,
    "model.fusion_gate_hidden": 8,
}
