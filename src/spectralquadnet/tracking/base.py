"""The tracker protocol every backend implements, plus the no-op backend.

The protocol splits into two channels — machine (structured, numeric) and
human (formatted, readable) — because the two audiences want different
things from the same training run: a plotted loss curve versus a readable
terminal log with banners and running tables.

===================  =========================================================
Method               Channel
===================  =========================================================
``log_scalar``       machine — one metric, plottable over ``step``
``log_scalars``      machine — a group of metrics sharing a ``step``
``log_table``        machine + human — tabular records (per-class F1, …)
``log_hyperparams``  machine — the composed config, once per run
``watch``            machine — optional gradient/parameter histograms
``close``            lifecycle
``banner``           human — a stage header block
``log_message``      human — a one-line notice
``log_row``          human — one epoch's summary, pre-formatted
``progress_start``   human — open a stage's span (its label and epoch budget)
``progress_stop``    human — close it
===================  =========================================================

``progress_start``/``progress_stop`` bracket a stage's epoch loop and carry the
two facts a per-epoch line needs but an epoch does not know: what the stage is
called and how many epochs it may run. A backend with no notion of either
ignores them. ``log_row`` keeps its meaning regardless: it is one epoch's
summary, rendered as a single appended line by the console backend
(``[Stage 1 | Ep 181/600]  Time: …  Loss: …``) and as a set of series by the
structured ones. No backend redraws anything — see
:mod:`spectralquadnet.tracking.console_tracker` for why the live progress bar
was removed rather than made conditional.

The split matters for the console backend: the human channel produces
readable terminal output, while the machine channel carries diagnostics
(per-branch losses, per-branch gradient norms) that are only useful as
curves. :class:`~spectralquadnet.tracking.console_tracker.ConsoleTracker`
therefore stays quiet on ``log_scalar``/``log_scalars`` unless
``tracking.show_diagnostics`` is set — otherwise every epoch would print twice.

Backends are duck-typed, not inherited: :class:`ExperimentTracker` is a
``Protocol``, so :class:`NullTracker` and the four real backends satisfy it
structurally without a shared base class.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch.nn as nn

#: Levels accepted by :meth:`ExperimentTracker.log_message`.
#:
#: ``plain`` reproduces a bare ``print``; ``info``/``warn`` reproduce the
#: baseline's ``[INFO]``/``[WARN]`` prefixes, which the backend renders itself so
#: call sites pass the message text alone.
MessageLevel = str


@runtime_checkable
class ExperimentTracker(Protocol):
    """Structural type for every tracking backend."""

    # ── Machine channel ─────────────────────────────────────────────

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Record a single metric at ``step``."""
        ...

    def log_scalars(self, tags: dict[str, float], step: int) -> None:
        """Record a group of metrics sharing one ``step``."""
        ...

    def log_table(self, tag: str, rows: list[dict[str, Any]], step: int) -> None:
        """Record tabular records — e.g. the per-class F1 breakdown."""
        ...

    def log_hyperparams(self, cfg: dict[str, Any]) -> None:
        """Record the composed run config, once per run."""
        ...

    def watch(self, model: nn.Module) -> None:
        """Opt into backend-side gradient/parameter histograms, if supported."""
        ...

    def close(self) -> None:
        """Flush and release the backend. Must be idempotent."""
        ...

    # ── Human channel (additive; see the module docstring) ────────────

    def banner(self, title: str, lines: Sequence[str] = ()) -> None:
        """Render a stage header: a title plus optional detail lines."""
        ...

    def log_message(self, text: str, level: MessageLevel = "info") -> None:
        """Render a one-line notice at ``plain``/``info``/``warn``/``success``."""
        ...

    def log_row(self, tag: str, cells: dict[str, str], step: int) -> None:
        """Render epoch ``step`` of the stage identified by ``tag``.

        Cells are **pre-formatted strings**, not numbers: the call site owns
        the formatting (``.4f`` for a loss, ``.1%`` for an accuracy). A cell
        whose value is empty is *absent* rather than blank — that is how the
        checkpoint marker means "saved this epoch". The stage's name, the epoch
        budget, the elapsed clock and the ETA are **not** cells: they come from
        the span opened by :meth:`progress_start`, so every stage renders them
        the same way.
        """
        ...

    def progress_start(self, tag: str, total: int, description: str = "") -> None:
        """Open ``tag``'s span: ``description`` is the stage label, ``total`` its epoch budget."""
        ...

    def progress_stop(self, tag: str) -> None:
        """Close ``tag``'s span. Must be safe to call unmatched."""
        ...


class NullTracker:
    """The ``tracking.backend=none`` backend — every method is a no-op.

    Used by the CI smoke path and by any caller that wants the tracker
    parameter satisfied without producing output. Note that the engine's
    default is ``tracker=None``, not an implicit ``NullTracker()``:
    ``train_one_epoch(..., tracker=None)`` skips diagnostic accumulation
    entirely rather than accumulating it only to discard it here, which
    keeps the no-tracker path free of any tracking-related overhead.
    """

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        return None

    def log_scalars(self, tags: dict[str, float], step: int) -> None:
        return None

    def log_table(self, tag: str, rows: list[dict[str, Any]], step: int) -> None:
        return None

    def log_hyperparams(self, cfg: dict[str, Any]) -> None:
        return None

    def watch(self, model: nn.Module) -> None:
        return None

    def close(self) -> None:
        return None

    def banner(self, title: str, lines: Sequence[str] = ()) -> None:
        return None

    def log_message(self, text: str, level: MessageLevel = "info") -> None:
        return None

    def log_row(self, tag: str, cells: dict[str, str], step: int) -> None:
        return None

    def progress_start(self, tag: str, total: int, description: str = "") -> None:
        return None

    def progress_stop(self, tag: str) -> None:
        return None


def flatten_hyperparams(cfg: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested config dict to dotted keys.

    W&B and TensorBoard both want a flat ``{"stage1.max_lr": 0.0005}`` mapping;
    the composed Hydra config is nested. Lists are stringified because
    ``hparams`` accepts only scalars.
    """
    flat: dict[str, Any] = {}
    for key, value in cfg.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten_hyperparams(value, prefix=f"{dotted}."))
        elif isinstance(value, (list, tuple)):
            flat[dotted] = str(list(value))
        else:
            flat[dotted] = value
    return flat
