"""Fan-out composite: log to N backends at once.

The main use is keeping a readable terminal while streaming to a remote
backend, since :class:`~spectralquadnet.tracking.wandb_tracker.WandbTracker`
and :class:`~spectralquadnet.tracking.tensorboard_tracker.TensorBoardTracker`
leave the human channel inert by design::

    python train.py tracking.backend=multi tracking.backends=[console,wandb]

:meth:`MultiTracker.close` closes every child even if one raises, so a failing
backend cannot strand another's file handle or leave a W&B run unfinished.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from spectralquadnet.tracking.base import ExperimentTracker

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch.nn as nn


class MultiTracker:
    """Broadcast every call to each wrapped backend, in registration order."""

    def __init__(self, trackers: Sequence[ExperimentTracker]) -> None:
        self._trackers = list(trackers)

    def __len__(self) -> int:
        return len(self._trackers)

    # ── Machine channel ───────────────────────────────────────────────

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        for t in self._trackers:
            t.log_scalar(tag, value, step)

    def log_scalars(self, tags: dict[str, float], step: int) -> None:
        for t in self._trackers:
            t.log_scalars(tags, step)

    def log_table(self, tag: str, rows: list[dict[str, Any]], step: int) -> None:
        for t in self._trackers:
            t.log_table(tag, rows, step)

    def log_hyperparams(self, cfg: dict[str, Any]) -> None:
        for t in self._trackers:
            t.log_hyperparams(cfg)

    def watch(self, model: nn.Module) -> None:
        for t in self._trackers:
            t.watch(model)

    def close(self) -> None:
        errors: list[BaseException] = []
        for t in self._trackers:
            try:
                t.close()
            except Exception as exc:  # pragma: no cover - backend-specific
                errors.append(exc)
        if errors:
            raise errors[0]

    # ── Human channel ─────────────────────────────────────────────────

    def banner(self, title: str, lines: Sequence[str] = ()) -> None:
        for t in self._trackers:
            t.banner(title, lines)

    def log_message(self, text: str, level: str = "info") -> None:
        for t in self._trackers:
            t.log_message(text, level)

    def log_row(self, tag: str, cells: dict[str, str], step: int) -> None:
        for t in self._trackers:
            t.log_row(tag, cells, step)

    def progress_start(self, tag: str, total: int, description: str = "") -> None:
        for t in self._trackers:
            t.progress_start(tag, total, description)

    def progress_stop(self, tag: str) -> None:
        for t in self._trackers:
            t.progress_stop(tag)
