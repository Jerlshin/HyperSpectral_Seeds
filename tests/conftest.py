"""Shared fixtures — synthetic by default, real artifacts only where a gate demands them.

Most fixtures here need nothing but the committed golden files. The two that do
need real artifacts (:func:`checkpoint_paths` and the ``dataset/`` wavelength CSV)
skip themselves when those are absent, so ``pytest`` stays green on a fresh clone
where ``dataset/`` and ``outputs/`` are gitignored.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
GOLDEN = Path(__file__).resolve().parent / "regression" / "golden"
OUTPUTS = REPO_ROOT / "outputs" / "output_v12_spa40"

SEED = 42
BATCH = 4
SPATIAL = 64

# `scripts/` is not part of the installed package, but the regression gates
# below need its reference-implementation loader. Adding it to the path here
# keeps the `_baseline` import in the fixtures below plain and unconditional.
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


# ══════════════════════════════════════════════════════════════════════
#  The pinned reference implementation
# ══════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="session")
def baseline() -> types.ModuleType:
    """The pinned reference ``hsi_training.py``, executed side-effect-free from git.

    ``scripts/_baseline.py`` reads the file at a fixed git ref and runs only
    its declarations, so importing it neither seeds the RNG nor writes to
    disk. The regression tests call the functions on this module object
    rather than compare against transcribed constants — a transcription
    could itself carry the drift the test exists to catch.
    """
    from _baseline import load_baseline_module

    return load_baseline_module("hsi_training")


@pytest.fixture(scope="session")
def baseline_src() -> str:
    """Source text of the reference implementation, for reaching nested closures.

    ``phase_aware_lr`` and ``_s3_margin`` are defined *inside* ``run_stage1`` /
    ``run_stage3_swa``, so they cannot be read off the module object.
    """
    from _baseline import baseline_source

    return baseline_source("hsi_training")


# ══════════════════════════════════════════════════════════════════════
#  Config
# ══════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="session")
def cfg():
    """The composed ``output_v12_spa40`` experiment config."""
    from spectralquadnet.config.compose import load_experiment_config

    return load_experiment_config()


# ══════════════════════════════════════════════════════════════════════
#  Golden artifacts (committed — see tests/regression/golden/README.md)
# ══════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="session")
def physical_wl() -> torch.Tensor:
    """The exact normalised wavelength vector the golden values were captured with.

    Loaded from the committed ``.npy`` rather than ``dataset/wavelengths_spa_40b.csv``
    so the regression gates do not depend on the gitignored dataset directory.
    ``test_mmap_store.py`` separately checks that ``DataStore`` reproduces it from
    the real CSV when that CSV is present.
    """
    path = GOLDEN / "physical_wl_spa40.npy"
    if not path.exists():
        pytest.skip(f"{path} missing — run `python scripts/capture_golden.py`")
    return torch.from_numpy(np.load(path))


@pytest.fixture(scope="session")
def golden_logits() -> np.ndarray:
    path = GOLDEN / "forward_logits_seed42.npy"
    if not path.exists():
        pytest.skip(f"{path} missing — run `python scripts/capture_golden.py`")
    return np.load(path)


@pytest.fixture(scope="session")
def golden_init_digests() -> dict[str, str]:
    path = GOLDEN / "init_state_sha256.json"
    if not path.exists():
        pytest.skip(f"{path} missing — run `python scripts/capture_golden.py`")
    return json.loads(path.read_text())


@pytest.fixture(scope="session")
def golden_stage1_loss() -> dict[str, Any]:
    path = GOLDEN / "stage1_epoch1_loss_seed42.json"
    if not path.exists():
        pytest.skip(f"{path} missing — run `python scripts/capture_golden.py`")
    return json.loads(path.read_text())


@pytest.fixture(scope="session")
def stage1_train_step(cfg, physical_wl) -> dict[str, Any]:
    """One fixed-seed Stage-1 epoch through ``train_one_epoch``.

    The procedure is imported from ``scripts/capture_golden.py`` rather than
    re-implemented, so the test and the capture that produced the golden file
    cannot drift apart in setup. Session-scoped: it trains a real 15M-parameter
    model for four steps on CPU (~8 s).
    """
    from capture_golden import refactored_train_step

    return refactored_train_step(cfg, physical_wl)


# ══════════════════════════════════════════════════════════════════════
#  Model
# ══════════════════════════════════════════════════════════════════════


@pytest.fixture
def seeded_model(cfg, physical_wl):
    """A freshly seeded, eval-mode ``SpectralQuadNet`` on CPU.

    Mirrors ``scripts/capture_golden.py::forward_pass`` step for step. The
    ``set_seed`` call must stay immediately before construction — weight
    initialisation depends on that exact ordering.
    """
    from spectralquadnet.models.spectral_quadnet import SpectralQuadNet
    from spectralquadnet.utils.seed import set_seed

    set_seed(SEED)
    model = SpectralQuadNet.from_config(cfg, physical_wl)
    return model.to("cpu").eval()


@pytest.fixture(scope="session")
def synthetic_batch(cfg) -> torch.Tensor:
    """``(4, 40, 64, 64)`` input from a dedicated generator — identical to the capture."""
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    return torch.randn(BATCH, cfg.data.num_bands, SPATIAL, SPATIAL, generator=gen)


# ══════════════════════════════════════════════════════════════════════
#  Real trained artifacts
# ══════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="session")
def checkpoint_paths() -> dict[int, Path]:
    """The three real stage checkpoints, or skip if they are not on this machine."""
    paths = {s: OUTPUTS / f"best_stage{s}.pth" for s in (1, 2, 3)}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        pytest.skip(f"real checkpoints not present: {', '.join(missing)}")
    return paths
