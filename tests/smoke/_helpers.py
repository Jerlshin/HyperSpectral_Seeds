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
    "model.specf_patch": 2,
    "model.grid_size_a": 2,
    "model.grid_size_d": 2,
    "model.fusion_rank": 8,
    "model.fusion_gate_hidden": 8,
}
