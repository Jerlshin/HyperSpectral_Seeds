"""Shared fixtures — synthetic by default, real artifacts only where a gate demands them.

Most fixtures here need nothing but the committed golden files. The two that do
need real artifacts (:func:`checkpoint_paths` and the ``dataset/`` wavelength CSV)
skip themselves when those are absent, so ``pytest`` stays green on a fresh clone
where ``dataset/`` and ``outputs/`` are gitignored.

``pytest_addoption``/``pytest_collection_modifyitems`` below make the default
invocation — bare ``pytest``, ``pytest tests/``, ``pytest tests/unit/`` — a fast
tier: ``regression``, ``slow`` and ``requires_dataset`` tests are collected but
reported as skipped, not deselected, so `-k`/`-m` filtering and test counts stay
honest. Pass the matching ``--run-*`` flag (or ``--run-all``) to opt them back
in; CI and pre-release runs should use ``pytest tests/ --run-all``.
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

_OPT_IN_MARKERS = ("regression", "slow", "requires_dataset")


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("spectralquadnet")
    for name in _OPT_IN_MARKERS:
        group.addoption(
            f"--run-{name.replace('_', '-')}",
            action="store_true",
            default=False,
            help=f"also run tests marked '{name}' (skipped by default for a fast dev loop)",
        )
    group.addoption(
        "--run-all",
        action="store_true",
        default=False,
        help="shorthand for every --run-* flag above",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    run_all = config.getoption("--run-all")
    enabled = {
        name
        for name in _OPT_IN_MARKERS
        if run_all or config.getoption(f"--run-{name.replace('_', '-')}")
    }
    for marker in _OPT_IN_MARKERS:
        if marker in enabled:
            continue
        skip = pytest.mark.skip(reason=f"needs --run-{marker.replace('_', '-')} (or --run-all)")
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)


#: Schema-v1 goldens — the pinned reference implementation's values, frozen.
GOLDEN_V1 = Path(__file__).resolve().parent / "regression" / "golden"
#: Schema-v2 goldens — the Tier-2 architecture. Frozen at Tier-3 completion and
#: no longer re-capturable; read only as the left-hand side of the v2 -> v3
#: structural delta.
GOLDEN_V2 = GOLDEN_V1 / "v2"
#: Schema-v3 goldens — the current architecture, after Tier 3 (T3-1 ... T3-7)
#: rebuilt three of the four branches and the fusion. This is what the live
#: drift gate compares against; see `golden/v3/README.md`.
GOLDEN = GOLDEN_V1 / "v3"
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


@pytest.fixture(autouse=True)
def _clear_accelerator_memory():
    """Frees CUDA/MPS cache after every test — a no-op on CPU-only runs.

    Model-instantiation tests build a fresh ``SpectralQuadNet`` per test; on a
    GPU-backed CI runner or a developer's Metal machine, letting each one's
    allocator cache linger is how a fast suite still creeps up on VRAM.
    """
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


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
    ``run_stage3_swa`` in the reference, so they cannot be read off the module
    object. The current code has neither: the first is a factory in
    ``optim/schedulers.py``, and the second was replaced by
    ``stage3_margin_kappa`` (T2-1).
    """
    from _baseline import baseline_source

    return baseline_source("hsi_training")


# ══════════════════════════════════════════════════════════════════════
#  Config
# ══════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="session")
def cfg():
    """The composed **audited** experiment config — SpectralQuadNet, three stages.

    The golden regression gates are digests of *that* architecture's weights and
    Stage-1 loss, so this fixture must keep composing it. `cfg_default` below is
    the new default composition (`SpectralSeedNet`, one stage, grouped protocol).
    """
    from spectralquadnet.config.compose import AUDITED_EXPERIMENT, load_experiment_config

    return load_experiment_config(AUDITED_EXPERIMENT)


@pytest.fixture(scope="session")
def cfg_default():
    """The composed default experiment — `SpectralSeedNet`, single stage, grouped."""
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
    path = GOLDEN_V1 / "physical_wl_spa40.npy"
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
def golden_v1_init_digests() -> dict[str, str]:
    """The schema-v1 digests, kept so the v1 → v2 delta stays an assertion.

    They are no longer a value gate on the current model — HD-1 shifted the
    RNG stream every later `_init_weights` draw reads from — but the *key set*
    still is, and that is what pins "one head was removed" against "something
    else also changed".
    """
    path = GOLDEN_V1 / "init_state_sha256.json"
    if not path.exists():
        pytest.skip(f"{path} missing — run `python scripts/capture_golden.py`")
    return json.loads(path.read_text())


@pytest.fixture(scope="session")
def golden_v2_init_digests() -> dict[str, str]:
    """The schema-v2 digests — the left-hand side of Tier 3's structural delta.

    Frozen: the code that produced them no longer exists, so they are read and
    never rewritten. What they still pin is that every key Tier 3 added or
    removed is one somebody declared.
    """
    path = GOLDEN_V2 / "init_state_sha256.json"
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


@pytest.fixture
def silence_dropout():
    """Zero **every** dropout rate in a model, so a ``train()``-mode forward is deterministic.

    ``SpectralQuadNet.set_dropout`` walks ``nn.Dropout`` instances only, and so
    cannot reach the ``dropout`` float inside ``nn.MultiheadAttention`` (N-2 —
    four sites in Branch D, two in the fusion). Any test that needs two
    train-mode forwards to agree has to close that gap itself; this is that
    one-liner, kept in one place so the tests below cannot disagree about what
    "dropout off" means.
    """

    def _silence(model: Any) -> None:
        model.set_dropout(0.0)
        for m in model.modules():
            if isinstance(m, torch.nn.MultiheadAttention):
                m.dropout = 0.0

    return _silence


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
