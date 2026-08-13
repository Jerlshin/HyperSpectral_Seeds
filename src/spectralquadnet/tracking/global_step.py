"""A monotone cross-stage step axis for the structured backends (IC-1).

The defect this closes
──────────────────────
Every stage numbered its own epochs from 1 and passed that number as
``step``. W&B rejects any step below the running maximum, so once Stage 1
ended at epoch 336 the *entire* Stage-2 and Stage-3 scalar stream was
discarded — ~200 ``Tried to log to step N that is less than the current step
336`` warnings, and seven uploaded panels that stop at 336 and contain zero
information about two thirds of the run (CHANGES.md §10.1). ``sam/grad_cos``,
the one measurement that would have justified Stage 3, was computed and lost.

The fix, and why it is a decorator
──────────────────────────────────
``global_step = stage_offset + stage_epoch``. Applying that inside every stage
would mean threading an offset through ``run_stage{1,2,3}``,
``train_one_epoch``, ``train_one_epoch_sam``, ``compute_class_difficulty`` and
``final_evaluation`` — six call paths, each with its own ``step=`` argument,
each an opportunity to miss one. :class:`StepOffsetTracker` instead applies the
offset at the *boundary*: it is a tracker, so it satisfies the same protocol,
and every ``step`` that crosses it is rebased exactly once.

**Only the machine channel is rebased.** ``log_row`` keeps the stage-local
epoch, because that is what the console renders as ``[Stage 2 | Ep 19/150]``
and a reader wants the epoch within the stage, not epoch 355 of an unlabelled
continuum. The scalar backends want the opposite, and now each gets what it
wants from one call site.

Two scalars are injected alongside every group so the split is recoverable from
the curves alone: ``progress/stage`` (1, 2, 3, …) and ``progress/stage_epoch``
(the un-rebased epoch). Plotting ``val/f1_best`` against ``progress/stage``
gives back the per-stage view the offset flattens.

Contiguity
──────────
:class:`GlobalStep` is advanced by the number of epochs a stage *actually ran*,
not by its budget, so an early-stopped Stage 1 does not leave a 64-epoch hole
before Stage 2. Monotonicity holds either way: ``advance`` never moves the
offset backwards.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from spectralquadnet.tracking.base import ExperimentTracker, MessageLevel

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch.nn as nn


class GlobalStep:
    """A monotone step offset shared by every stage of one run.

    Stages read :attr:`offset` when they open their tracker and call
    :meth:`advance` with the number of epochs they ran when they close it.

    Args:
        start: Initial offset. ``0`` for a fresh run; a resumed run can start
            past the epochs the skipped stages already logged.
    """

    def __init__(self, start: int = 0) -> None:
        self._offset = max(int(start), 0)

    @property
    def offset(self) -> int:
        """The step the next stage's epoch 1 maps to minus one."""
        return self._offset

    def resolve(self, stage_epoch: int) -> int:
        """``stage_offset + stage_epoch`` — the step a structured backend sees."""
        return self._offset + int(stage_epoch)

    def advance(self, epochs_run: int) -> int:
        """Move the offset forward by the epochs a stage actually ran.

        Negative inputs are clamped to zero: the offset is monotone by
        construction, which is the property W&B's step axis requires.
        """
        self._offset += max(int(epochs_run), 0)
        return self._offset

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"GlobalStep(offset={self._offset})"


class StepOffsetTracker:
    """Rebase the machine channel onto :class:`GlobalStep`; pass the rest through.

    Args:
        inner: The tracker every call is forwarded to.
        clock: The run's shared step counter.
        stage: Stage ordinal, emitted as ``progress/stage`` so a flattened
            curve can be split back apart.

    Note:
        This satisfies :class:`~spectralquadnet.tracking.base.ExperimentTracker`
        structurally, so it composes with ``MultiTracker`` and nests without a
        special case.
    """

    def __init__(self, inner: ExperimentTracker, clock: GlobalStep, stage: int = 0) -> None:
        self._inner = inner
        self._clock = clock
        self._stage = int(stage)

    # ── Machine channel — rebased ─────────────────────────────────────

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.log_scalars({tag: value}, step)

    def log_scalars(self, tags: dict[str, float], step: int) -> None:
        if not tags:
            return
        enriched = dict(tags)
        # Injected rather than left to each stage: three stages logging their
        # own stage index is three chances for one of them to disagree.
        enriched.setdefault("progress/stage", float(self._stage))
        enriched.setdefault("progress/stage_epoch", float(step))
        self._inner.log_scalars(enriched, self._clock.resolve(step))

    def log_table(self, tag: str, rows: list[dict[str, Any]], step: int) -> None:
        self._inner.log_table(tag, rows, self._clock.resolve(step))

    def log_hyperparams(self, cfg: dict[str, Any]) -> None:
        self._inner.log_hyperparams(cfg)

    def watch(self, model: nn.Module) -> None:
        self._inner.watch(model)

    def close(self) -> None:
        # Deliberately does *not* close `inner`: the wrapper's lifetime is one
        # stage and the run owns the backend across all of them.
        return None

    # ── Human channel — stage-local, see the module docstring ─────────

    def banner(self, title: str, lines: Sequence[str] = ()) -> None:
        self._inner.banner(title, lines)

    def log_message(self, text: str, level: MessageLevel = "info") -> None:
        self._inner.log_message(text, level)

    def log_row(self, tag: str, cells: dict[str, str], step: int) -> None:
        self._inner.log_row(tag, cells, step)

    def progress_start(self, tag: str, total: int, description: str = "") -> None:
        self._inner.progress_start(tag, total, description)

    def progress_stop(self, tag: str) -> None:
        self._inner.progress_stop(tag)


def stage_tracker(
    tracker: ExperimentTracker | None, clock: GlobalStep | None, stage: int
) -> ExperimentTracker | None:
    """Wrap ``tracker`` for one stage, or return it unchanged.

    ``None`` in either position means "no rebasing": a caller with no tracker
    logs nowhere, and a caller with no clock is running a single stage, where
    the stage-local epoch already *is* monotone.
    """
    if tracker is None or clock is None:
        return tracker
    return StepOffsetTracker(tracker, clock, stage)
