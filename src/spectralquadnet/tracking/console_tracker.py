"""Rich terminal rendering — the default tracking backend.

Has zero external-service dependencies, so the framework works offline out
of the box.

Two deliberate design points:

* **Text is never wrapped or cropped.** Every text-emitting method passes
  ``soft_wrap=True``, because ``rich`` otherwise wraps at the detected
  terminal width — and falls back to 80 columns whenever stdout is not a
  TTY, which is exactly the case when a run is piped to a log file.
* **Rows stream, they do not redraw.** A Stage 1 run is 600 epochs; a
  ``rich.live`` table that grows without bound would scroll the terminal's
  scrollback into uselessness and break piping to a log file. Instead the
  first row of each table emits a header, column widths are frozen from it,
  and every subsequent row is padded to match — so output stays aligned
  while remaining append-only.
* **Scalars are quiet by default.** The epoch summary arrives through
  :meth:`log_row`; per-branch diagnostics (losses, gradient norms) arrive
  through :meth:`log_scalars` and are meant to be read as curves in
  W&B/TensorBoard. Rendering both would print every epoch twice. Set
  ``tracking.show_diagnostics=true`` to echo them here as well.

Terminal I/O is on the critical path
────────────────────────────────────
``rich`` renders by building a layout and measuring every cell, and a write to
a terminal is a blocking syscall the training loop is holding the GIL across.
One padded row per epoch is cheap; the per-improvement hardest-class *table* is
not, and early in a run "on improvement" means "every epoch".

So the default rendering is now a **single redrawing progress bar** whose
description carries the same numbers the row did (:meth:`progress_start` opens
it, :meth:`log_row` updates it), and the row stream is what a non-TTY gets. The
choice is automatic because it has to be: a bar redrawn with ANSI cursor moves
into a piped ``training.log`` is unreadable, and that file is the record of the
run. ``runtime.progress`` forces either.

The throttling of the *expensive* diagnostics — the hardest-class table and the
branch-influence ablation behind it — is not here. It belongs to the caller
that knows the epoch number; see ``engine/diagnostics.py::should_render_details``.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch.nn as nn

#: Minimum width of a streamed table column, so short headers stay readable.
_MIN_COL_WIDTH = 5

#: Row cells promoted into the progress bar's suffix, in this order.
#:
#: The bar has one line; the row had ten columns. These are the ones that answer
#: "is it still learning" — the rest stay available in ``training.log`` (which
#: always gets the full row stream, since a log file is never a TTY) and as
#: curves in the structured backends.
_BAR_FIELDS: tuple[str, ...] = ("Loss", "F1 live/ema", "LR", "Ph", "κ", "swa", "ckpt")

#: ``level`` → (prefix, rich style) for :meth:`ConsoleTracker.log_message`.
_LEVEL_STYLES: dict[str, tuple[str, str]] = {
    "plain": ("", ""),
    "info": ("[INFO] ", "cyan"),
    "warn": ("[WARN] ", "yellow"),
    "success": ("", "green"),
}


class ConsoleTracker:
    """Render the run to a terminal with ``rich``.

    Args:
        show_diagnostics: Echo :meth:`log_scalars` groups to the terminal. Off
            by default so console output stays limited to one line per epoch.
        console: Injected :class:`rich.console.Console`, for tests.
        progress: ``auto`` renders a progress bar on a TTY and the append-only
            row stream otherwise; ``bar``/``rows``/``off`` force one. ``off``
            suppresses the per-epoch line entirely — banners, messages and
            tables still render.
    """

    def __init__(
        self,
        show_diagnostics: bool = False,
        console: Console | None = None,
        progress: str = "auto",
    ) -> None:
        self._console = console or Console()
        self._show_diagnostics = show_diagnostics
        #: tag → frozen column widths, set when a table emits its header.
        self._widths: dict[str, dict[str, int]] = {}
        self._progress_mode = str(progress).lower()
        self._progress: Progress | None = None
        #: tag → (task handle, the total it was opened with). The total is kept
        #: here rather than read back off ``Progress.tasks``, which is a *list*
        #: rebuilt from an internal dict — indexing it by ``TaskID`` happens to
        #: work while no task is ever removed, and stops working the moment one
        #: is.
        self._tasks: dict[str, tuple[TaskID, int]] = {}

    # ── Progress display ──────────────────────────────────────────────

    def _wants_bar(self) -> bool:
        """Whether a redrawing bar is appropriate for this stdout.

        ``is_terminal`` is the load-bearing test: ``rich`` reports False for a
        pipe or a file, which is exactly where a bar's cursor movements would
        turn ``training.log`` into control characters.
        """
        if self._progress_mode in ("rows", "off"):
            return False
        if self._progress_mode == "bar":
            return True
        return bool(self._console.is_terminal and sys.stdout.isatty())

    def progress_start(self, tag: str, total: int, description: str = "") -> None:
        if not self._wants_bar():
            return
        if self._progress is None:
            self._progress = Progress(
                TextColumn("[bold cyan]{task.description}"),
                BarColumn(bar_width=None),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TextColumn("eta"),
                TimeRemainingColumn(),
                TextColumn("{task.fields[detail]}"),
                console=self._console,
                # The bar shares this Console, so `log_message` and `banner`
                # scroll above it instead of fighting it for the cursor.
                transient=False,
            )
            self._progress.start()
        capped = max(1, int(total))
        self._tasks[tag] = (
            self._progress.add_task(description or tag, total=capped, detail=""),
            capped,
        )

    def progress_stop(self, tag: str) -> None:
        entry = self._tasks.pop(tag, None)
        if self._progress is None:
            return
        if entry is not None:
            # A stage that early-stopped leaves the bar short of its total;
            # completing it here means the display does not sit at 340/600
            # forever while the next stage's banner scrolls past.
            task, total = entry
            self._progress.update(task, completed=total)
        if not self._tasks:
            self._progress.stop()
            self._progress = None

    # ── Machine channel ───────────────────────────────────────────────

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.log_scalars({tag: value}, step)

    def log_scalars(self, tags: dict[str, float], step: int) -> None:
        if not self._show_diagnostics or not tags:
            return
        body = "  ".join(f"{k}={v:.4g}" for k, v in tags.items())
        self._console.print(f"[dim]  · step {step}  {body}[/dim]", soft_wrap=True)

    def log_table(self, tag: str, rows: list[dict[str, Any]], step: int) -> None:
        if not rows:
            return
        table = Table(title=f"{tag}  (step {step})", title_style="bold", header_style="bold")
        for column in rows[0]:
            table.add_column(str(column))
        for row in rows:
            table.add_row(*(str(row.get(c, "")) for c in rows[0]))
        self._console.print(table)

    def log_hyperparams(self, cfg: dict[str, Any]) -> None:
        # Rendered as a two-column table of the top-level config groups.
        rows = [{"group": k, "value": _summarise(v)} for k, v in cfg.items()]
        self.log_table("config", rows, step=0)

    def watch(self, model: nn.Module) -> None:
        # No console equivalent of gradient histograms; trainable parameter
        # count is the useful terminal-sized summary instead.
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.log_message(f"Params : {n_params / 1e6:.2f}M", level="plain")

    def close(self) -> None:
        # Leaving a `rich.Live` running would strand the terminal with a hidden
        # cursor and no echo if the process exits through this path.
        if self._progress is not None:
            self._progress.stop()
            self._progress = None
        self._tasks.clear()

    # ── Human channel ─────────────────────────────────────────────────

    def banner(self, title: str, lines: Sequence[str] = ()) -> None:
        body = "\n".join(lines)
        self._console.print()
        self._console.print(
            Panel(body, title=f"[bold]{title}[/bold]", border_style="cyan", expand=False)
            if body
            else Panel.fit(f"[bold]{title}[/bold]", border_style="cyan")
        )

    def log_message(self, text: str, level: str = "info") -> None:
        prefix, style = _LEVEL_STYLES.get(level, _LEVEL_STYLES["info"])
        line = f"{prefix}{text}"
        self._console.print(
            f"[{style}]{line}[/{style}]" if style else line, highlight=False, soft_wrap=True
        )

    def log_row(self, tag: str, cells: dict[str, str], step: int) -> None:
        entry = self._tasks.get(tag)
        if entry is not None and self._progress is not None:
            task, _total = entry
            detail = "  ".join(f"{k} {cells[k]}" for k in _BAR_FIELDS if cells.get(k, "").strip())
            self._progress.update(task, completed=step, detail=detail)
            return
        if self._progress_mode == "off":
            return

        widths = self._widths.get(tag)
        if widths is None:
            widths = {k: max(len(k), len(v), _MIN_COL_WIDTH) for k, v in cells.items()}
            self._widths[tag] = widths
            header = " │ ".join(k.ljust(widths[k]) for k in cells)
            self._console.print(f"[bold]{header}[/bold]", highlight=False, soft_wrap=True)
            self._console.print("[dim]" + "─" * len(header) + "[/dim]", soft_wrap=True)
        line = " │ ".join(str(v).ljust(widths.get(k, len(str(v)))) for k, v in cells.items())
        self._console.print(line.rstrip(), highlight=False, soft_wrap=True)


def _summarise(value: Any) -> str:
    """One-line rendering of a config group for :meth:`log_hyperparams`."""
    if isinstance(value, dict):
        return ", ".join(f"{k}={v}" for k, v in value.items())
    return str(value)
