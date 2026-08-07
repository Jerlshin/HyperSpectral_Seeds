"""Experiment-tracking backends behind a single ExperimentTracker protocol.

:func:`build_tracker` is the only entry point callers need; it maps
``cfg.tracking.backend`` onto a concrete backend:

===============  ==================================================
``backend``      Result
===============  ==================================================
``none``         :class:`~.base.NullTracker` — silent
``console``      :class:`~.console_tracker.ConsoleTracker` (default)
``wandb``        :class:`~.wandb_tracker.WandbTracker`
``tensorboard``  :class:`~.tensorboard_tracker.TensorBoardTracker`
``multi``        :class:`~.multi_tracker.MultiTracker` over
                 ``cfg.tracking.backends``
===============  ==================================================

The optional backends import their SDK lazily, so ``import spectralquadnet``
never requires ``wandb`` or ``tensorboard``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from spectralquadnet.tracking.base import ExperimentTracker, NullTracker, flatten_hyperparams
from spectralquadnet.tracking.console_tracker import ConsoleTracker
from spectralquadnet.tracking.multi_tracker import MultiTracker

if TYPE_CHECKING:  # pragma: no cover - typing only
    from spectralquadnet.config.schema import ExperimentConfig

__all__ = [
    "ConsoleTracker",
    "ExperimentTracker",
    "MultiTracker",
    "NullTracker",
    "build_tracker",
    "flatten_hyperparams",
]


def build_tracker(cfg: ExperimentConfig | Any) -> ExperimentTracker:
    """Instantiate the tracking backend named by ``cfg.tracking.backend``.

    Takes the whole experiment config, not just its ``tracking`` group: W&B wants
    the run name and the full config for its run metadata, and TensorBoard's
    ``log_dir`` interpolates ``${output_dir}``.

    Raises:
        ValueError: The backend name is not one of the five known values, or
            ``backend=multi`` was selected with an empty ``backends`` list.
    """
    return _build_one(str(cfg.tracking.backend).lower(), cfg)


def _build_one(name: str, cfg: ExperimentConfig | Any) -> ExperimentTracker:
    tracking = cfg.tracking

    if name == "none":
        return NullTracker()

    if name == "console":
        return ConsoleTracker(show_diagnostics=bool(getattr(tracking, "show_diagnostics", False)))

    if name == "wandb":
        from spectralquadnet.tracking.wandb_tracker import WandbTracker

        return WandbTracker(
            project=tracking.project,
            entity=tracking.entity,
            run_name=cfg.run_name,
            watch_model=bool(tracking.watch_model),
            config=_as_plain_dict(cfg),
        )

    if name == "tensorboard":
        from spectralquadnet.tracking.tensorboard_tracker import TensorBoardTracker

        return TensorBoardTracker(log_dir=tracking.log_dir or f"{cfg.output_dir}/tensorboard")

    if name == "multi":
        names = [str(b).lower() for b in (tracking.backends or [])]
        if not names:
            raise ValueError(
                "tracking.backend=multi requires a non-empty tracking.backends list, "
                "e.g. tracking.backends=[console,wandb]"
            )
        if "multi" in names:
            raise ValueError("tracking.backends must not contain 'multi' (no nesting).")
        return MultiTracker([_build_one(n, cfg) for n in names])

    raise ValueError(
        f"Unknown tracking backend {name!r}. "
        "Expected one of: none, console, wandb, tensorboard, multi."
    )


def _as_plain_dict(cfg: ExperimentConfig | Any) -> dict[str, Any]:
    """Best-effort conversion of a composed config to a nested plain dict.

    ``OmegaConf.to_container`` handles the ``DictConfig`` that Hydra produces;
    plain dataclasses and dicts fall through unchanged so tests can pass either.
    """
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            container = OmegaConf.to_container(cfg, resolve=True)
            if isinstance(container, dict):
                return {str(k): v for k, v in container.items()}
    except ImportError:  # pragma: no cover - omegaconf is a hard dependency
        pass
    if isinstance(cfg, dict):
        return cfg
    return dict(getattr(cfg, "__dict__", {}))
