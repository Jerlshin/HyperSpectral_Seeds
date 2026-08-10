"""Append-only terminal rendering — the default tracking backend.

Has zero external-service dependencies, so the framework works offline out
of the box.

One line per epoch, appended, forever
─────────────────────────────────────
There is no live region in this backend: no ``rich.progress`` bar, no cursor
motion, no ``\\r``, no ``\\033[2K``. Every epoch emits exactly **one** line, and
that line is written once and never rewritten::

    [Stage 1 | Ep 181/600]  Time: 00:12:45  ETA: 00:41:12  dt: 42.1s  Loss: 15.2508
      Tr: 61.2%  F1 live/ema: 0.771/0.685  Acc live/ema: 78.1%/72.0%  Best: 0.780
      LR: 3.00e-04  LS: 0.084  auxW: 0.42  tau: 0.35  Ph: P2  m: 0.00  ckpt ✓

(shown wrapped here; it is emitted as a single line).

That is the only rendering, on every stdout, and the reason is that the three
places this code runs cannot agree on anything richer:

* **A macOS terminal** honours cursor control, so a redrawing bar looks right —
  and is the one environment where it does.
* **An SSH session piped to a file, or `nohup`** gets no cursor at all: a bar
  redrawn into ``training.log`` is thousands of overwritten half-lines.
* **A Kaggle/Colab notebook** reports a terminal it cannot drive. ``rich``
  repositions the cursor only when ``Console.is_interactive``; where that is
  false it *appends* every frame instead, so a 600-epoch bar becomes 600
  interleaved bar renders between the log lines.

Choosing per-environment meant three renderings of the same run and three sets
of bugs. One append-only line is legible in all three, greppable, diffable
between runs, and identical in the terminal and in the log file.

The prefix is assembled here, not by the caller
───────────────────────────────────────────────
:meth:`~ConsoleTracker.progress_start` no longer opens a display; it opens a
**span** — a stage's label and epoch budget — and :meth:`~ConsoleTracker.log_row`
reads that span to build ``[Stage 1 | Ep 181/600]``, the elapsed clock and the
ETA that the removed progress bar used to own. The stage orchestrators pass
only what they measure.

Two clocks, deliberately: ``Time`` is elapsed since the tracker was built (the
run's own wall clock, continuous across stage boundaries), while ``ETA`` is
extrapolated from the *current span* alone, because Stage 2's epochs cost
nothing like Stage 1's.

Every line is written the same way, and why
───────────────────────────────────────────
:meth:`ConsoleTracker._write_line` is the single exit for all human-channel
output, and it enforces four properties that a training log needs and that the
obvious ``console.print(f"[{style}]{text}[/{style}]")`` does not have:

* **The payload is never parsed as markup.** ``rich`` reads ``[...]`` in a
  printed string as a style tag, and the messages here are full of brackets —
  ``[DATA]``, ``[DDP]``, ``[GPU0]``, ``[Stage 1 | Ep 181/600]`` itself. Most are
  not valid styles and survive as literal text, but the ones that *are*
  (``[dim]``, a bare ``[/]``) either split the colour run mid-sentence or raise
  ``MarkupError`` and kill the run from a log line. ``markup=False`` plus an
  explicit ``style=`` puts one escape sequence around the whole line, so the
  styling cannot be truncated by the text it is styling.
* **The payload carries no escape sequences of its own.** :func:`_sanitize`
  strips ANSI/OSC sequences and bare carriage returns before anything is
  printed. A ``\\r`` from an upstream progress line rewinds the cursor and
  overwrites whatever was already on the row, which is the ``12/90F1 live/ema``
  class of corruption.
* **Each line is terminated and pushed out before the next writer runs.**
  ``rich`` and the ``logging`` StreamHandler hold separate buffers over the same
  descriptor, and a line still sitting in one when the other writes is what
  produces ``P1[INFO] EMA re-init …``. Flushing per line bounds that window to a
  single line.
* **Each line is mirrored to ``logging``.** The terminal is scrollback; the
  record of the run is ``training.log``, and before this the file held only
  crashes, because the tracker wrote to stdout and the file handler only ever
  saw ``logging`` records. Lines go to the ``spectralquadnet.console`` logger,
  which ``train.py`` points at the file handler alone (``propagate=False``), so
  the terminal keeps exactly one copy.

Nothing is drawn that the stream cannot encode: on a ``LANG=C`` SSH session
:data:`_ASCII_FALLBACK` replaces ``✓``, ``κ``, ``★`` and the box rules with
ASCII rather than letting the write raise or emit ``?``.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from rich.console import Console

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch.nn as nn

#: Widest rule a banner will draw, so a 200-column terminal does not get a
#: 200-character line of ``═`` around a two-word title.
_MAX_RULE = 96

#: Cells rendered as ``key value`` rather than ``key: value``.
#:
#: These carry a marker, not a measurement — ``ckpt ✓``, ``sgdr ↻R1``,
#: ``swa ★ snap 3 (F1 0.812)`` — and a colon in front of a glyph reads as
#: punctuation damage. Every other cell gets ``key: value``.
_FLAG_KEYS: frozenset[str] = frozenset({"ckpt", "swa", "sgdr"})

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

#: Non-ASCII glyphs this backend emits, and what they degrade to on a stream
#: that cannot encode them (``LANG=C`` over SSH, a Windows console at cp1252).
#: Applied only when the encoding test in :meth:`ConsoleTracker._probe_ascii`
#: fails, so a normal UTF-8 terminal keeps the glyphs.
_ASCII_FALLBACK: dict[str, str] = {
    "✓": "ok",
    "✗": "x",
    "★": "*",
    "↻": "R",
    "κ": "kappa",
    "τ": "tau",
    "α": "alpha",
    "γ": "gamma",
    "ρ": "rho",
    "Δ": "d",
    "─": "-",
    "═": "=",
    "│": "|",
    "·": "-",
    "—": "-",
    "–": "-",
    "→": "->",
    "≤": "<=",
    "≥": ">=",
    "×": "x",
}

_ASCII_TABLE = str.maketrans({k: v for k, v in _ASCII_FALLBACK.items()})

#: Where mirrored human-channel lines go. ``train.py`` attaches the run's file
#: handler to this logger and clears ``propagate``, so the lines reach
#: ``training.log`` without being echoed to stdout a second time.
_MIRROR_LOGGER = "spectralquadnet.console"


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


def _hms(seconds: float) -> str:
    """``12345.6`` → ``"03:25:45"``. Hours are not wrapped at 24."""
    total = max(0, int(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


@dataclass
class _Span:
    """One stage's epoch loop: what it is called, how long it is, when it began.

    Opened by :meth:`ConsoleTracker.progress_start`, read by
    :meth:`ConsoleTracker.log_row` to build the ``[Stage 1 | Ep 181/600]`` prefix
    and the ETA, closed by :meth:`ConsoleTracker.progress_stop`.
    """

    label: str
    total: int
    started: float
    last_step: int = 0
    #: cell key → widest rendering seen, so columns line up down the log. Grows
    #: monotonically: a wider value (``Loss: 9.9`` → ``Loss: 15.2508``) widens
    #: the column from then on rather than being truncated into it.
    widths: dict[str, int] = field(default_factory=dict)


class ConsoleTracker:
    """Render the run to a terminal as append-only lines.

    Args:
        show_diagnostics: Echo :meth:`log_scalars` groups to the terminal. Off
            by default so console output stays limited to one line per epoch.
        console: Injected :class:`rich.console.Console`, for tests.
        progress: ``off`` suppresses the per-epoch line entirely — banners,
            messages and diagnostic blocks still render. Every other value
            (``auto``, and the legacy ``bar``/``rows``, kept so an existing
            command line still composes) renders the epoch line; there is no
            longer a redrawing mode to select. See the module docstring.
        mirror_to_log: Also emit each line to the ``spectralquadnet.console``
            logger, which is what puts the run into ``training.log``.
    """

    def __init__(
        self,
        show_diagnostics: bool = False,
        console: Console | None = None,
        progress: str = "auto",
        mirror_to_log: bool = True,
    ) -> None:
        # `force_jupyter=False` is what makes a Kaggle/Colab cell behave like
        # every other stream: left to itself `rich` detects the notebook and
        # renders through IPython's display machinery, which buffers separately
        # from the `logging` handler writing to the same cell and reorders the
        # two against each other.
        self._console = console or Console(force_jupyter=False, soft_wrap=True)
        self._show_diagnostics = show_diagnostics
        self._enabled = str(progress).lower() != "off"
        self._spans: dict[str, _Span] = {}
        self._t0 = time.monotonic()
        self._ascii_only = self._probe_ascii()
        self._log = logging.getLogger(_MIRROR_LOGGER) if mirror_to_log else None

    # ── Write discipline ──────────────────────────────────────────────

    def _probe_ascii(self) -> bool:
        """Whether this stream needs the ASCII degradation of :data:`_ASCII_FALLBACK`.

        A stream that cannot encode ``✓`` either raises ``UnicodeEncodeError``
        mid-run or silently emits ``?``; both are worse than ``ckpt ok``.
        Unknown encodings are treated as capable, since every default in the
        three target environments is UTF-8.
        """
        encoding = getattr(self._console.file, "encoding", None) or "utf-8"
        try:
            "".join(_ASCII_FALLBACK).encode(encoding)
        except (UnicodeEncodeError, LookupError):
            return True
        return False

    def _render(self, text: object) -> str:
        """Sanitize, then degrade to ASCII if the stream demands it."""
        out = _sanitize(text)
        return out.translate(_ASCII_TABLE) if self._ascii_only else out

    def _write_line(self, text: object, style: str = "") -> None:
        """The single exit for every human-channel line. See the module docstring.

        ``markup=False`` is the part that must not be relaxed: with it on, any
        ``[…]`` in a *message* is a style tag, and the two failure modes are a
        colour run that stops mid-sentence and a ``MarkupError`` raised from a
        log statement 180 epochs into a run.
        """
        line = self._render(text)
        self._console.print(
            line,
            style=style or None,
            markup=False,
            highlight=False,
            soft_wrap=True,
        )
        self._flush()
        if self._log is not None and line.strip():
            self._log.info(line)

    def _flush(self) -> None:
        """Push the console's line out of every buffer between here and the fd.

        ``rich`` and the ``logging`` StreamHandler ``train.py`` installs write to
        the same descriptor through separate buffers. Whichever holds a partial
        line when the other writes is what splices ``[INFO] …`` onto the end of
        an epoch line. Flushing after each line bounds that window to a single
        line rather than to a buffer's worth.
        """
        with suppress(ValueError, OSError):  # a closed or detached stream
            self._console.file.flush()

    # ── Stage spans (formerly the progress display) ───────────────────

    def progress_start(self, tag: str, total: int, description: str = "") -> None:
        """Open a stage span. Renders nothing — see the module docstring."""
        self._spans[tag] = _Span(
            label=description or tag, total=max(1, int(total)), started=time.monotonic()
        )

    def progress_stop(self, tag: str) -> None:
        """Close ``tag``'s span and report what it cost.

        The closing line is what an early-stopped stage leaves behind: the bar
        used to be completed to its total so it would not sit at ``340/600``
        forever, which threw away the one number worth keeping — that the stage
        stopped at 340.
        """
        span = self._spans.pop(tag, None)
        if span is None or not self._enabled:
            return
        self._write_line(
            f"[{span.label}] finished — {span.last_step}/{span.total} epochs "
            f"in {_hms(time.monotonic() - span.started)}",
            style="bold",
        )

    # ── Machine channel ───────────────────────────────────────────────

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.log_scalars({tag: value}, step)

    def log_scalars(self, tags: dict[str, float], step: int) -> None:
        if not self._show_diagnostics or not tags:
            return
        body = "  ".join(f"{k}={v:.4g}" for k, v in tags.items())
        self._write_line(f"  · step {step}  {body}", style="dim")

    def log_table(self, tag: str, rows: list[dict[str, Any]], step: int) -> None:
        """Render a plain-text, column-aligned block.

        Not ``rich.Table``: a rendered table measures every cell against a
        terminal width that is 80 whenever stdout is not a TTY, so the same
        diagnostic came out box-drawn in a terminal, squeezed in a pipe and
        HTML-rendered in a notebook. The block below is the same characters
        everywhere, and its width is set by the data rather than by the
        environment.
        """
        if not rows:
            return
        columns = [str(c) for c in rows[0]]
        # A column of numbers reads right-aligned, to a *shared* number of
        # decimals; a column of names reads left-aligned. Deciding per column
        # keeps `rank/class/f1` and `group/value` legible through one code path,
        # and the shared precision is what puts the decimal points of `0.1200`
        # and `0.2011` under each other — the whole reason to render the
        # bottom-k classes as a block rather than as a list.
        by_column = [[row.get(c, "") for row in rows] for c in rows[0]]
        numeric = [all(_is_number(v) for v in column) for column in by_column]
        formatted = [
            _format_column(column, is_num)
            for column, is_num in zip(by_column, numeric, strict=True)
        ]
        cells = [[self._render(column[r]) for column in formatted] for r in range(len(rows))]
        widths = [max(len(col), *(len(row[i]) for row in cells)) for i, col in enumerate(columns)]

        def _row(values: Sequence[str]) -> str:
            return (
                "  "
                + "  ".join(
                    v.rjust(widths[i]) if numeric[i] else v.ljust(widths[i])
                    for i, v in enumerate(values)
                ).rstrip()
            )

        title = f"{_sanitize(tag)}  (ep {step})" if step else _sanitize(tag)
        self._write_line(title, style="bold")
        self._write_line(_row(columns), style="bold")
        self._write_line("  " + "  ".join("─" * w for w in widths), style="dim")
        for row in cells:
            self._write_line(_row(row))

    def log_hyperparams(self, cfg: dict[str, Any]) -> None:
        # Rendered as a two-column block of the top-level config groups.
        rows = [{"group": k, "value": _summarise(v)} for k, v in cfg.items()]
        self.log_table("config", rows, step=0)

    def watch(self, model: nn.Module) -> None:
        # No console equivalent of gradient histograms; trainable parameter
        # count is the useful terminal-sized summary instead.
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.log_message(f"Params : {n_params / 1e6:.2f}M", level="plain")

    def close(self) -> None:
        self._spans.clear()
        self._flush()

    # ── Human channel ─────────────────────────────────────────────────

    def banner(self, title: str, lines: Sequence[str] = ()) -> None:
        """A stage header: a rule, the title, the detail lines, a rule.

        Plain text for the same reason :meth:`log_table` is — a ``rich.Panel``
        is measured against a terminal width that three environments report
        three different values for.
        """
        heading = self._render(title)
        body = [self._render(line) for line in lines]
        width = min(max(len(heading), *(len(b) for b in body), 40) + 2, _MAX_RULE)
        self._write_line("")
        self._write_line("═" * width, style="cyan")
        self._write_line(f" {heading}", style="bold cyan")
        if body:
            self._write_line("─" * width, style="dim")
            for line in body:
                self._write_line(f" {line}")
        self._write_line("═" * width, style="cyan")

    def log_message(self, text: str, level: str = "info") -> None:
        prefix, style = _LEVEL_STYLES.get(level, _LEVEL_STYLES["info"])
        self._write_line(f"{prefix}{_sanitize(text)}", style=style)

    def log_row(self, tag: str, cells: dict[str, str], step: int) -> None:
        """Emit one epoch as a single appended line. See the module docstring.

        Empty cells are dropped rather than rendered as an empty column: the
        checkpoint marker is present on the epochs that saved one and absent on
        the rest, which is what makes ``grep 'ckpt'`` a list of the improvements.
        """
        span = self._spans.get(tag)
        if span is not None:
            span.last_step = step
        if not self._enabled:
            return

        # Rendered (sanitized, and ASCII-degraded where the stream needs it)
        # before the widths are taken, so a column's padding is measured in the
        # characters that will actually be written — ``✓`` and its ``ok``
        # fallback are not the same width.
        fields: list[tuple[str, str]] = [("", self._prefix(span, tag, step))]
        fields.append(("Time", f"Time: {_hms(time.monotonic() - self._t0)}"))
        fields.append(("ETA", f"ETA: {self._eta(span, step)}"))
        for key, value in cells.items():
            text = self._render(value).strip()
            if not text:
                continue
            fields.append((key, f"{key} {text}" if key in _FLAG_KEYS else f"{key}: {text}"))

        widths = span.widths if span is not None else {}
        rendered = []
        for key, text in fields:
            width = widths[key] = max(widths.get(key, 0), len(text))
            rendered.append(text.ljust(width))
        # One style for the whole line — see `_write_line`; a per-cell style
        # would put a reset between every pair of columns.
        self._write_line("  ".join(rendered).rstrip(), style="cyan")

    def _prefix(self, span: _Span | None, tag: str, step: int) -> str:
        """``[Stage 1 | Ep 181/600]``, or ``[stage1 | Ep 181]`` with no open span."""
        if span is None:
            return f"[{tag} | Ep {step}]"
        return f"[{span.label} | Ep {step}/{span.total}]"

    def _eta(self, span: _Span | None, step: int) -> str:
        """Time to the span's last epoch at the rate this span has averaged.

        Deliberately not a run-level estimate: it extrapolates from the current
        span's own epochs, because Stage 2's epoch and Stage 1's differ by more
        than the ETA's error bar.

        ``--:--:--`` rather than an omitted field where there is nothing to
        extrapolate — the last epoch of a span, or a row logged outside one. A
        field that disappears takes the alignment of every column after it with
        it, and a line that shifts on one epoch out of 600 is exactly the kind
        of thing that reads as corruption.
        """
        if span is None or step <= 0 or step >= span.total:
            return "--:--:--"
        elapsed = time.monotonic() - span.started
        return _hms(elapsed / step * (span.total - step))


def _is_number(value: Any) -> bool:
    """Whether a cell is a bare number, for column alignment."""
    try:
        float(str(value))
    except ValueError:
        return False
    return True


def _format_column(values: list[Any], numeric: bool) -> list[str]:
    """One column's cells as text, numeric ones sharing a decimal count.

    ``0.12`` next to ``0.2011`` is what :func:`~spectralquadnet.engine.diagnostics.hardest_classes_report`
    produces — ``round`` drops trailing zeros — and right-aligning those two puts
    the ``2`` of one under the ``0`` of the other. Padding the column to its
    widest precision is the fix, and it is a rendering decision, so it lives here
    rather than in the report that structured backends also consume as numbers.

    A column holding exponent notation (``1e-05``) is left exactly as written:
    re-formatting it to a fixed decimal count would print ``0.00001`` or, worse,
    ``0.00``.
    """
    texts = [str(v) for v in values]
    if not numeric or any("e" in text.lower() for text in texts):
        return texts
    decimals = max(len(text.partition(".")[2]) for text in texts)
    return [f"{float(text):.{decimals}f}" for text in texts]


def _summarise(value: Any) -> str:
    """One-line rendering of a config group for :meth:`log_hyperparams`."""
    if isinstance(value, dict):
        return ", ".join(f"{k}={v}" for k, v in value.items())
    return str(value)
