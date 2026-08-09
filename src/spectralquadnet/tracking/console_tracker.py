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

Every line is written the same way, and why
───────────────────────────────────────────
:meth:`ConsoleTracker._write_line` is the single exit for all human-channel
output, and it enforces three properties that a training log needs and that the
obvious ``console.print(f"[{style}]{text}[/{style}]")`` does not have:

* **The payload is never parsed as markup.** ``rich`` reads ``[...]`` in a
  printed string as a style tag, and the messages here are full of brackets —
  ``[DATA]``, ``[DDP]``, ``[GPU0]``, a class name, an ``[INFO]`` prefix. Most
  are not valid styles and survive as literal text, but the ones that *are*
  (``[dim]``, a bare ``[/]``) either split the colour run mid-sentence or raise
  ``MarkupError`` and kill the run from a log line. ``markup=False`` plus an
  explicit ``style=`` puts one escape sequence around the whole line, so the
  styling cannot be truncated by the text it is styling.
* **The payload carries no escape sequences of its own.** :func:`_sanitize`
  strips ANSI/OSC sequences and bare carriage returns before anything is
  printed. A ``\r`` from an upstream progress line rewinds the cursor and
  overwrites whatever was already on the row, which is the ``12/90F1 live/ema``
  class of corruption.
* **Each line is terminated and pushed out before the next writer runs.**
  ``rich`` and the ``logging`` StreamHandler hold separate buffers over the same
  descriptor, and a line still sitting in one when the other writes is what
  produces ``P1[INFO] EMA re-init …``. Flushing per line bounds that window to a
  single line; ``train.py``'s ``_LateBoundStdout`` closes the rest of it by
  putting ``logging`` under the same redirect the progress bar installs.

A redrawing bar additionally requires ``Console.is_interactive`` — not just
``is_terminal``. ``rich`` repositions the cursor around a live region only on an
interactive console (see ``Live.process_renderables``); on a dumb terminal or a
Kaggle web console it appends every frame instead, interleaving bar renders with
log lines. ``progress=bar`` forced there therefore degrades to the row stream
rather than to garbage.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Sequence
from contextlib import suppress
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
from rich.table import Column, Table

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch.nn as nn

#: Minimum width of a streamed table column, so short headers stay readable.
_MIN_COL_WIDTH = 5

#: CSI (``ESC [ … final``), OSC (``ESC ] … BEL/ST``) and two-character escapes.
#:
#: Matches what an upstream library, a coloured exception or a terminal-aware
#: dependency may have baked into a string before it reached a ``log_message``
#: call. Re-emitting those verbatim inside a styled line is what leaves the left
#: half green and the rest plain: the embedded reset closes the run early.
_ANSI_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)?|[@-Z\\-_])")

#: C0 controls with no meaning in a log line. ``\t`` and ``\n`` are deliberately
#: absent — ``classification_report`` is multi-line and tab-aligned — and ``\r``
#: is handled separately, since dropping it silently would join two halves of a
#: rewound progress line into one unreadable one.
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

#: Row cells promoted into the progress bar's suffix, in this order.
#:
#: The bar has one line; the row had ten columns. These are the ones that answer
#: "is it still learning" — the rest stay available in ``training.log`` (which
#: always gets the full row stream, since a log file is never a TTY) and as
#: curves in the structured backends.
_BAR_FIELDS: tuple[str, ...] = ("Loss", "F1 live/ema", "LR", "Ph", "κ", "swa", "ckpt")

#: ``level`` → (prefix, rich style) for :meth:`ConsoleTracker.log_message`.
#:
#: The style is passed to ``Console.print``'s ``style=`` argument rather than
#: wrapped around the text as markup, so it applies to the **whole** line
#: including the prefix — see :meth:`ConsoleTracker._write_line`.
_LEVEL_STYLES: dict[str, tuple[str, str]] = {
    "plain": ("", ""),
    "info": ("[INFO] ", "cyan"),
    "warn": ("[WARN] ", "yellow"),
    "success": ("", "green"),
}


def _sanitize(text: object) -> str:
    """Strip escape sequences and cursor controls from text bound for the console.

    Three transformations, in order:

    1. ANSI/OSC escape sequences are removed. One embedded ``\\x1b[0m`` inside a
       line the tracker is colouring ends the colour run at that point, which is
       exactly the "green up to here, plain after" corruption.
    2. ``\\r\\n`` collapses to ``\\n`` and a bare ``\\r`` becomes ``\\n``. A lone
       carriage return means "rewind to column 0", so anything after it lands on
       top of what came before; promoting it to a line break keeps both halves
       readable instead of overwriting one with the other.
    3. Remaining C0 controls are dropped. ``\\t`` and ``\\n`` survive, because
       ``classification_report`` is a multi-line, column-aligned block that this
       tracker prints verbatim.
    """
    out = _ANSI_RE.sub("", str(text))
    out = out.replace("\r\n", "\n").replace("\r", "\n")
    return _CTRL_RE.sub("", out)


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

    # ── Write discipline ──────────────────────────────────────────────

    def _write_line(self, text: object, style: str = "") -> None:
        """The single exit for every human-channel line. See the module docstring.

        ``markup=False`` is the part that must not be relaxed: with it on, any
        ``[…]`` in a *message* is a style tag, and the two failure modes are a
        colour run that stops mid-sentence and a ``MarkupError`` raised from a
        log statement 180 epochs into a run.
        """
        self._console.print(
            _sanitize(text),
            style=style or None,
            markup=False,
            highlight=False,
            soft_wrap=True,
        )
        self._flush()

    def _flush(self) -> None:
        """Push the console's line out of every buffer between here and the fd.

        ``rich`` and the ``logging`` StreamHandler ``train.py`` installs write to
        the same descriptor through separate buffers. Whichever holds a partial
        line when the other writes is what splices ``[INFO] …`` onto the end of
        a progress row. Flushing after each line bounds that window to a single
        line rather than to a buffer's worth.
        """
        with suppress(ValueError, OSError):  # a closed or detached stream
            self._console.file.flush()

    # ── Progress display ──────────────────────────────────────────────

    def _wants_bar(self) -> bool:
        """Whether a redrawing bar is appropriate for this stdout.

        ``is_interactive`` is the load-bearing test, and it is stricter than the
        ``is_terminal`` this used to ask for. ``rich`` only repositions the
        cursor around a live region on an interactive console; on a dumb
        terminal, a pipe, or a web console that reports a TTY without honouring
        cursor control, it *appends* each frame instead — so the bar's redraws
        land between log lines rather than under them.

        ``progress=bar`` is therefore a request, not an override: it forces a bar
        wherever one can be drawn, and falls back to the row stream where one
        cannot.
        """
        if self._progress_mode in ("rows", "off"):
            return False
        if not self._console.is_interactive:
            return False
        if self._progress_mode == "bar":
            return True
        return bool(self._console.is_terminal and sys.stdout.isatty())

    def progress_start(self, tag: str, total: int, description: str = "") -> None:
        if not self._wants_bar():
            return
        if self._progress is None:
            self._progress = Progress(
                # `markup=False` on both text columns for the same reason
                # `_write_line` sets it: the description and the detail suffix
                # are payload, and a `[…]` in either is text, not a style.
                TextColumn("{task.description}", style="bold cyan", markup=False),
                # A bounded bar. `bar_width=None` expands to fill, which starves
                # the columns after it: `rich` then squeezes them until the
                # padding between cells is gone and `12/90` runs straight into
                # the detail suffix.
                BarColumn(bar_width=24),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TextColumn("eta", style="dim"),
                TimeRemainingColumn(),
                # `no_wrap` keeps a long suffix on the bar's own line instead of
                # letting it wrap into the row below, where the next refresh
                # would only repaint the first line and orphan the second.
                TextColumn(
                    "{task.fields[detail]}",
                    style="cyan",
                    markup=False,
                    table_column=Column(no_wrap=True),
                ),
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
            # `Live.stop` leaves the cursor at the end of the bar's own line.
            # The next thing written is a stage banner, and without this it
            # would begin on that line rather than under it.
            self._console.line()
            self._flush()

    # ── Machine channel ───────────────────────────────────────────────

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.log_scalars({tag: value}, step)

    def log_scalars(self, tags: dict[str, float], step: int) -> None:
        if not self._show_diagnostics or not tags:
            return
        body = "  ".join(f"{k}={v:.4g}" for k, v in tags.items())
        self._write_line(f"  · step {step}  {body}", style="dim")

    def log_table(self, tag: str, rows: list[dict[str, Any]], step: int) -> None:
        if not rows:
            return
        table = Table(title=f"{tag}  (step {step})", title_style="bold", header_style="bold")
        for column in rows[0]:
            table.add_column(_sanitize(column))
        for row in rows:
            table.add_row(*(_sanitize(row.get(c, "")) for c in rows[0]))
        self._console.print(table)
        self._flush()

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
        self._flush()

    # ── Human channel ─────────────────────────────────────────────────

    def banner(self, title: str, lines: Sequence[str] = ()) -> None:
        # The title and every body line are sanitized, but the Panel itself is
        # rendered from markup — the styling here is the tracker's own and does
        # not come from the payload.
        body = "\n".join(_sanitize(line) for line in lines)
        heading = _sanitize(title)
        self._console.print()
        self._console.print(
            Panel(body, title=f"[bold]{heading}[/bold]", border_style="cyan", expand=False)
            if body
            else Panel.fit(f"[bold]{heading}[/bold]", border_style="cyan")
        )
        self._flush()

    def log_message(self, text: str, level: str = "info") -> None:
        prefix, style = _LEVEL_STYLES.get(level, _LEVEL_STYLES["info"])
        self._write_line(f"{prefix}{_sanitize(text)}", style=style)

    def log_row(self, tag: str, cells: dict[str, str], step: int) -> None:
        cells = {k: _sanitize(v) for k, v in cells.items()}
        entry = self._tasks.get(tag)
        if entry is not None and self._progress is not None:
            task, _total = entry
            # Two spaces between fields and one inside each, so a squeeze at a
            # narrow width degrades to single spaces rather than to none — the
            # `12/90F1 live/ema` collision was fields with no separator of their
            # own relying on the grid's padding to keep them apart.
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
            self._write_line(header, style="bold")
            self._write_line("─" * len(header), style="dim")
        line = " │ ".join(str(v).ljust(widths.get(k, len(str(v)))) for k, v in cells.items())
        # One style for the whole row, matching the bar's suffix in the
        # interactive mode, so the two renderings of the same epoch line agree.
        self._write_line(line.rstrip(), style="cyan")


def _summarise(value: Any) -> str:
    """One-line rendering of a config group for :meth:`log_hyperparams`."""
    if isinstance(value, dict):
        return ", ".join(f"{k}={v}" for k, v in value.items())
    return str(value)
