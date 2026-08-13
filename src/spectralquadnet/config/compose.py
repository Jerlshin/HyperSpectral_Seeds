"""Programmatic Hydra composition for callers that are not ``@hydra.main``.

``train.py`` is decorated with ``@hydra.main`` and gets its config injected.
Tests, notebooks and other scripts need the *same* composed config without
that decorator — this module is the single place that knows how to build it,
so the composition rules cannot drift between entrypoints.
"""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

from spectralquadnet.config.schema import register_configs

#: The composition ``python train.py`` runs with no arguments — CHANGES §16-§19:
#: ``SpectralSeedNet``, one stage, leave-one-bundle-out, selection on ``calib``.
DEFAULT_EXPERIMENT = "experiment/seednet_grouped"

#: The audited four-branch / three-stage / stratified configuration, kept as the
#: control arm for A1, A3 and A8. Named here rather than spelled at each call
#: site so the regression tests, the ablation registry and the docs cannot
#: disagree about which config *is* the baseline.
AUDITED_EXPERIMENT = "experiment/quadnet_audited"


def config_dir() -> Path:
    """Locate the repository's ``configs/`` directory.

    Walks up from this file, which works both for an editable install (the
    package lives in ``<repo>/src/``) and for a source checkout.
    """
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "configs"
        if (candidate / "experiment").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not locate the `configs/` directory relative to "
        f"{Path(__file__).resolve()} — pass config_path explicitly."
    )


def load_experiment_config(
    experiment: str = DEFAULT_EXPERIMENT,
    overrides: list[str] | None = None,
    config_path: Path | str | None = None,
) -> DictConfig:
    """Compose an experiment config exactly as ``python train.py`` would.

    Parameters
    ----------
    experiment
        Config name relative to ``configs/`` — e.g. :data:`DEFAULT_EXPERIMENT`
        or :data:`AUDITED_EXPERIMENT`.
    overrides
        Hydra override strings, e.g. ``["stage1.max_lr=1e-4"]``.
    config_path
        Override the auto-detected ``configs/`` directory.
    """
    register_configs()
    root = Path(config_path) if config_path is not None else config_dir()

    # initialize_config_dir refuses to run if a previous Hydra instance is live,
    # which happens when several tests compose configs in one process.
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(root), version_base=None):
        return compose(config_name=experiment, overrides=overrides or [])
