"""Shared fixtures — synthetic by default, real artifacts only where a gate demands them.

Most fixtures here need nothing but the committed golden files. The two that do
need real artifacts (:func:`checkpoint_paths` and the ``dataset/`` wavelength CSV)
skip themselves when those are absent, so ``pytest`` stays green on a fresh clone
where ``dataset/`` and ``outputs/`` are gitignored.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN = Path(__file__).resolve().parent / "regression" / "golden"
OUTPUTS = REPO_ROOT / "outputs" / "output_v12_spa40"

SEED = 42
BATCH = 4
SPATIAL = 64


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


# ══════════════════════════════════════════════════════════════════════
#  Model
# ══════════════════════════════════════════════════════════════════════


@pytest.fixture
def seeded_model(cfg, physical_wl):
    """A freshly seeded, eval-mode ``SpectralQuadNet`` on CPU.

    Mirrors ``scripts/capture_golden.py::forward_pass`` step for step. The
    ``set_seed`` call must stay immediately before construction — REFACTOR_PLAN.md
    §3.6 makes the seed→construct ordering load-bearing.
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
