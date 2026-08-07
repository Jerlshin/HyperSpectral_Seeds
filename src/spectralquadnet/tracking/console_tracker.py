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
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch.nn as nn

#: Minimum width of a streamed table column, so short headers stay readable.
_MIN_COL_WIDTH = 5

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
    """

    def __init__(self, show_diagnostics: bool = False, console: Console | None = None) -> None:
        self._console = console or Console()
        self._show_diagnostics = show_diagnostics
        #: tag → frozen column widths, set when a table emits its header.
        self._widths: dict[str, dict[str, int]] = {}

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
        return None

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
