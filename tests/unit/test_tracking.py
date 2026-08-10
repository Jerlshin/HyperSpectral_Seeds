"""Tracking-backend behaviour.

Unlike everything under ``tests/regression/`` there is no reference
implementation to compare against, since the tracking package has no
numerical behaviour of its own. What these tests pin instead is the contract
the stage orchestrators rely on:

* every backend structurally satisfies :class:`ExperimentTracker`, so a stage can
  be handed any of them;
* the factory maps ``cfg.tracking.backend`` onto the right class and rejects
  nonsense loudly rather than silently falling back to a default;
* :class:`MultiTracker` fans out and closes every child;
* the console backend renders each channel, and stays quiet about scalars unless
  ``show_diagnostics`` is set — the property that keeps its output to one
  line per epoch;
* that line is **append-only**: one line per epoch, carrying the stage, the
  epoch, the clock and the metrics, with no carriage return, no line-clear and
  no cursor motion on any stdout — the property the three target environments
  (macOS terminal, piped SSH session, Kaggle/Colab cell) have in common.

``wandb``/``tensorboard`` are exercised through the factory's error paths only;
instantiating them would open a real run directory or network session.
"""

from __future__ import annotations

import io
from typing import Any

import pytest
from omegaconf import OmegaConf
from rich.console import Console

from spectralquadnet.tracking import build_tracker
from spectralquadnet.tracking.base import ExperimentTracker, NullTracker, flatten_hyperparams
from spectralquadnet.tracking.console_tracker import ConsoleTracker
from spectralquadnet.tracking.multi_tracker import MultiTracker


def _cfg(**tracking: Any) -> Any:
    """A config shaped like the composed experiment, with only what a tracker reads."""
    base = {
        "backend": "console",
        "project": None,
        "entity": None,
        "log_dir": None,
        "watch_model": False,
        "backends": [],
        "log_grad_norms": True,
        "show_diagnostics": False,
    }
    base.update(tracking)
    return OmegaConf.create(
        {"tracking": base, "run_name": "unit_test", "output_dir": "outputs/unit_test"}
    )


class _Recorder:
    """A tracker that records calls instead of rendering them."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.closed = False

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.calls.append(("log_scalar", (tag, value, step)))

    def log_scalars(self, tags: dict[str, float], step: int) -> None:
        self.calls.append(("log_scalars", (dict(tags), step)))

    def log_table(self, tag: str, rows: list[dict[str, Any]], step: int) -> None:
        self.calls.append(("log_table", (tag, rows, step)))

    def log_hyperparams(self, cfg: dict[str, Any]) -> None:
        self.calls.append(("log_hyperparams", cfg))

    def watch(self, model: Any) -> None:
        self.calls.append(("watch", model))

    def close(self) -> None:
        self.closed = True

    def banner(self, title: str, lines: Any = ()) -> None:
        self.calls.append(("banner", (title, list(lines))))

    def log_message(self, text: str, level: str = "info") -> None:
        self.calls.append(("log_message", (text, level)))

    def log_row(self, tag: str, cells: dict[str, str], step: int) -> None:
        self.calls.append(("log_row", (tag, dict(cells), step)))

    def progress_start(self, tag: str, total: int, description: str = "") -> None:
        self.calls.append(("progress_start", (tag, total, description)))

    def progress_stop(self, tag: str) -> None:
        self.calls.append(("progress_stop", tag))


# ══════════════════════════════════════════════════════════════════════
#  Protocol conformance
# ══════════════════════════════════════════════════════════════════════


def test_backends_satisfy_the_protocol() -> None:
    """Every backend a stage may be handed is structurally an ExperimentTracker."""
    for tracker in (NullTracker(), ConsoleTracker(), MultiTracker([]), _Recorder()):
        assert isinstance(tracker, ExperimentTracker)


def test_null_tracker_swallows_every_method() -> None:
    tracker = NullTracker()
    tracker.banner("t", ["a"])
    tracker.log_message("m")
    tracker.log_row("tag", {"a": "1"}, 1)
    tracker.log_scalar("s", 1.0, 1)
    tracker.log_scalars({"s": 1.0}, 1)
    tracker.log_table("t", [{"a": 1}], 1)
    tracker.log_hyperparams({"a": 1})
    tracker.close()  # idempotent, and must not raise


# ══════════════════════════════════════════════════════════════════════
#  Factory
# ══════════════════════════════════════════════════════════════════════


def test_factory_selects_none_and_console() -> None:
    assert isinstance(build_tracker(_cfg(backend="none")), NullTracker)
    assert isinstance(build_tracker(_cfg(backend="console")), ConsoleTracker)


def test_factory_builds_a_multi_tracker() -> None:
    tracker = build_tracker(_cfg(backend="multi", backends=["console", "none"]))
    assert isinstance(tracker, MultiTracker)
    assert len(tracker) == 2


@pytest.mark.parametrize(
    ("tracking", "message"),
    [
        ({"backend": "multi", "backends": []}, "non-empty"),
        ({"backend": "multi", "backends": ["console", "multi"]}, "nesting"),
        ({"backend": "nope"}, "Unknown tracking backend"),
    ],
)
def test_factory_rejects_bad_backends(tracking: dict[str, Any], message: str) -> None:
    """A misconfigured backend fails at startup, not hours into Stage 1."""
    with pytest.raises(ValueError, match=message):
        build_tracker(_cfg(**tracking))


# ══════════════════════════════════════════════════════════════════════
#  MultiTracker fan-out
# ══════════════════════════════════════════════════════════════════════


def test_multi_tracker_fans_out_and_closes_all() -> None:
    a, b = _Recorder(), _Recorder()
    multi = MultiTracker([a, b])
    multi.banner("Stage 1", ["detail"])
    multi.log_row("stage1", {"Ep": "001"}, 1)
    multi.log_scalars({"train/loss": 0.5}, 1)
    multi.close()

    assert [name for name, _ in a.calls] == ["banner", "log_row", "log_scalars"]
    assert a.calls == b.calls
    assert a.closed and b.closed


def test_multi_tracker_closes_every_child_even_when_one_raises() -> None:
    class _Exploding(_Recorder):
        def close(self) -> None:
            raise RuntimeError("backend went away")

    survivor = _Recorder()
    multi = MultiTracker([_Exploding(), survivor])
    with pytest.raises(RuntimeError, match="backend went away"):
        multi.close()
    assert survivor.closed, "a failing backend must not strand another's handle"


# ══════════════════════════════════════════════════════════════════════
#  ConsoleTracker rendering
# ══════════════════════════════════════════════════════════════════════


def _console_output(tracker_kwargs: dict[str, Any], emit: Any) -> str:
    buffer = io.StringIO()
    console = Console(file=buffer, width=200, no_color=True, highlight=False)
    tracker = ConsoleTracker(console=console, **tracker_kwargs)
    emit(tracker)
    return buffer.getvalue()


def test_console_renders_each_human_channel() -> None:
    def emit(t: ConsoleTracker) -> None:
        t.banner("Stage 1 — Progressive", ["Phase 1: heavy aug"])
        t.log_message("EMA re-init at Phase 2 (ep 5)")
        t.log_message("No snapshots accepted", level="warn")
        t.log_message("SWA val: F1=0.870", level="plain")
        t.log_table("hardest", [{"rank": 1, "class": 7, "f1": 0.12}], step=3)

    out = _console_output({}, emit)
    assert "Stage 1 — Progressive" in out
    assert "Phase 1: heavy aug" in out
    assert "[INFO] EMA re-init at Phase 2 (ep 5)" in out
    assert "[WARN] No snapshots accepted" in out
    # `plain` carries no prefix — it reproduces a bare `print`.
    assert "SWA val: F1=0.870" in out
    assert "[INFO] SWA val" not in out
    assert "hardest" in out and "0.12" in out


def test_a_table_is_a_plain_aligned_block() -> None:
    """The throttled diagnostics render as text whose width is set by the data.

    Not ``rich.Table``: a rendered table is measured against a terminal width
    that is 80 whenever stdout is not a TTY, so the same block came out
    box-drawn in a terminal, squeezed in a pipe and HTML-rendered in a notebook.
    """

    def emit(t: ConsoleTracker) -> None:
        t.log_table(
            "hardest_classes/S1",
            [{"rank": 1, "class": 7, "f1": 0.1234}, {"rank": 2, "class": 41, "f1": 0.2}],
            step=181,
        )

    lines = [line for line in _console_output({}, emit).splitlines() if line.strip()]
    assert lines[0] == "hardest_classes/S1  (ep 181)"
    assert lines[1].split() == ["rank", "class", "f1"]
    assert set(lines[2].strip()) == {"─", " "}, "a rule under the header, not a box"
    # Numeric columns are right-aligned and padded to a shared precision, so the
    # rows are the same length and the decimal points sit under each other —
    # `round(0.2, 4)` renders `0.2`, which right-aligned alone would put its `2`
    # under the neighbouring row's third decimal.
    assert len(lines[3]) == len(lines[4])
    assert lines[3].endswith("0.1234") and lines[4].endswith("0.2000")


def test_epoch_line_carries_the_whole_contract_on_one_line() -> None:
    """One epoch is one line, and it says which stage, which epoch, and how long."""

    def emit(t: ConsoleTracker) -> None:
        t.progress_start("stage1", total=600, description="Stage 1")
        t.log_row(
            "stage1",
            {
                "Loss": "15.2508",
                "F1 live/ema": "0.771/0.685",
                "LR": "3.00e-04",
                "Ph": "P2",
                "m": "0.35",
                "ckpt": "✓",
            },
            step=181,
        )

    lines = [line for line in _console_output({}, emit).splitlines() if line.strip()]
    assert len(lines) == 1, "an epoch renders exactly one line"
    (line,) = lines
    assert line.startswith("[Stage 1 | Ep 181/600]")
    for field in ("Loss: 15.2508", "F1 live/ema: 0.771/0.685", "LR: 3.00e-04", "Ph: P2", "m: 0.35"):
        assert field in line
    # A marker cell is a flag, not a measurement: no colon in front of the glyph.
    assert "ckpt ✓" in line
    # The clock and the extrapolation the removed progress bar used to own.
    assert "Time: 00:00:0" in line
    assert "ETA: " in line


def test_epoch_lines_stream_without_a_header_or_a_rewind() -> None:
    """Successive epochs append; nothing is redrawn and no header is repeated."""

    def emit(t: ConsoleTracker) -> None:
        t.progress_start("stage1", total=600, description="Stage 1")
        for ep in (1, 2, 3):
            t.log_row("stage1", {"Loss": f"{ep / 10:.4f}"}, step=ep)

    out = _console_output({}, emit)
    assert len([line for line in out.splitlines() if line.strip()]) == 3
    for ep in (1, 2, 3):
        assert f"Ep {ep}/600" in out and f"Loss: {ep / 10:.4f}" in out
    assert "\r" not in out


def test_an_empty_cell_is_absent_rather_than_blank() -> None:
    """``ckpt`` present means the epoch saved one — that is what makes it greppable."""

    def emit(t: ConsoleTracker) -> None:
        t.progress_start("stage1", total=600, description="Stage 1")
        t.log_row("stage1", {"Loss": "1.0", "ckpt": "", "stale": "3/40"}, step=7)

    out = _console_output({}, emit)
    assert "ckpt" not in out
    assert "stale: 3/40" in out


def test_progress_off_suppresses_the_epoch_line_and_nothing_else() -> None:
    def emit(t: ConsoleTracker) -> None:
        t.progress_start("stage1", total=600, description="Stage 1")
        t.log_row("stage1", {"Loss": "1.0"}, step=7)
        t.log_message("still spoken for")

    out = _console_output({"progress": "off"}, emit)
    assert "Loss" not in out
    assert "still spoken for" in out


def test_span_close_reports_where_the_stage_actually_stopped() -> None:
    """An early-stopped stage's last epoch is the number worth keeping."""

    def emit(t: ConsoleTracker) -> None:
        t.progress_start("stage1", total=600, description="Stage 1")
        t.log_row("stage1", {"Loss": "1.0"}, step=340)
        t.progress_stop("stage1")

    out = _console_output({}, emit)
    assert "[Stage 1] finished — 340/600 epochs in 00:00:0" in out


def test_console_scalars_are_quiet_unless_diagnostics_are_shown() -> None:
    def emit(t: ConsoleTracker) -> None:
        t.log_scalars({"grad_norm/branch_a": 0.25}, step=7)

    assert _console_output({}, emit) == ""
    shown = _console_output({"show_diagnostics": True}, emit)
    assert "grad_norm/branch_a" in shown and "step 7" in shown


# ══════════════════════════════════════════════════════════════════════
#  Terminal integrity: sanitization, markup and uniform styling
# ══════════════════════════════════════════════════════════════════════


def _coloured_output(emit: Any, **tracker_kwargs: Any) -> str:
    """Render to a buffer that keeps ANSI, so the escape runs can be asserted on."""
    buffer = io.StringIO()
    console = Console(
        file=buffer, width=200, force_terminal=True, color_system="truecolor", highlight=False
    )
    emit(ConsoleTracker(console=console, **tracker_kwargs))
    return buffer.getvalue()


def test_a_message_is_one_uninterrupted_colour_run() -> None:
    """The whole line is styled, or none of it — never the left half.

    ``rich`` reads ``[…]`` in a printed string as a style tag. A message that
    happens to contain one used to *close* the tracker's colour mid-sentence and
    reopen a different one, which is the "green up to here, plain after"
    corruption. The style now rides on ``Console.print``'s ``style=``, outside
    anything the payload can reach.
    """
    out = _coloured_output(
        lambda t: t.log_message("margins c7: R=0.20 [dim]P=0.10[/dim] and the rest", "success")
    )
    assert out == "\x1b[32mmargins c7: R=0.20 [dim]P=0.10[/dim] and the rest\x1b[0m\n"
    # One opening sequence, one reset: the run is not broken up.
    assert out.count("\x1b[32m") == 1
    assert out.count("\x1b[0m") == 1


def test_a_bracket_in_a_message_cannot_raise() -> None:
    """A bare ``[/]`` is a `MarkupError` under markup parsing — i.e. a crashed run.

    ``hardest`` class names, ``[DATA]``/``[DDP]`` prefixes and a stray closing
    tag all reach `log_message` as ordinary text. Killing a 600-epoch run from a
    log statement is the failure this rules out.
    """
    out = _coloured_output(lambda t: t.log_message("Seeded 90 classes [/] worst cosine 0.31"))
    assert "[/]" in out, "the bracket survives as literal text"


def test_embedded_escapes_and_carriage_returns_are_stripped() -> None:
    """Text arriving with its own ANSI or a cursor rewind cannot corrupt the line.

    An embedded reset ends the tracker's colour run early; a bare ``\\r`` rewinds
    to column 0 so whatever follows overwrites what came before — the
    ``12/90F1 live/ema`` class of damage. Both are neutralised before printing.
    """
    out = _coloured_output(
        lambda t: t.log_message("\x1b[32mEMA re-init\x1b[0m at P2\r12/90 after", level="plain")
    )
    assert "\x1b[32m" not in out, "the payload's own colour is gone"
    assert "\r" not in out
    # The rewind became a line break, so both halves survive and are readable.
    assert "EMA re-init at P2\n" in out
    assert "12/90 after" in out


def test_an_epoch_line_is_styled_as_a_whole_line() -> None:
    """The epoch line carries one colour run: opened once, reset once, at the end."""
    out = _coloured_output(lambda t: t.log_row("stage1", {"Ph": "P2"}, step=181))
    row = [line for line in out.splitlines() if "Ph: P2" in line][0]
    assert row.startswith("\x1b[36m") and row.endswith("\x1b[0m")
    assert row.count("\x1b[0m") == 1, "one reset, at the end — not mid-line"


@pytest.mark.parametrize("interactive", [True, False])
@pytest.mark.parametrize("progress", ["auto", "bar", "rows"])
def test_no_cursor_control_reaches_any_console(interactive: bool, progress: str) -> None:
    """No mode, on any terminal, emits a rewind or a line-clear.

    ``force_terminal`` with ``force_interactive=False`` is the shape of the
    environments this was reported from — a Kaggle cell and a dumb terminal both
    claim a TTY they cannot drive, and a redrawing bar there appends every frame
    instead of repainting one. ``force_interactive=True`` is the macOS terminal,
    where a bar *would* have worked; it renders the same way, because one
    rendering that is legible everywhere beats two that each fail somewhere.
    ``bar``/``rows`` are legacy spellings and no longer select anything.
    """
    buffer = io.StringIO()
    console = Console(file=buffer, width=200, force_terminal=True, force_interactive=interactive)
    tracker = ConsoleTracker(console=console, progress=progress)
    tracker.progress_start("stage1", total=600, description="Stage 1")
    for ep in (180, 181):
        tracker.log_row("stage1", {"Loss": "1.2345"}, step=ep)
    tracker.progress_stop("stage1")

    out = buffer.getvalue()
    assert "Ep 181/600" in out, "the epoch line is rendered"
    assert "\r" not in out, "no cursor rewind"
    assert "\x1b[2K" not in out, "no line clear"
    assert "\x1b[?25l" not in out, "no hidden cursor — nothing owns a live region"


def test_lines_are_mirrored_to_the_log(caplog: pytest.LogCaptureFixture) -> None:
    """``training.log`` is the record of the run, so every human line reaches it."""
    buffer = io.StringIO()
    tracker = ConsoleTracker(console=Console(file=buffer, width=200, no_color=True))
    with caplog.at_level("INFO", logger="spectralquadnet.console"):
        tracker.progress_start("stage1", total=600, description="Stage 1")
        tracker.log_message("EMA re-init at Phase 2")
        tracker.log_row("stage1", {"Loss": "1.2345"}, step=181)

    mirrored = [record.getMessage() for record in caplog.records]
    assert any("EMA re-init at Phase 2" in m for m in mirrored)
    assert any(m.startswith("[Stage 1 | Ep 181/600]") for m in mirrored)
    assert all(m.strip() for m in mirrored), "blank spacing lines are not logged"


def test_glyphs_degrade_on_a_stream_that_cannot_encode_them() -> None:
    """A ``LANG=C`` SSH session gets ASCII, not ``UnicodeEncodeError`` or ``?``."""
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="ascii", newline="")
    tracker = ConsoleTracker(console=Console(file=stream, width=200, no_color=True))
    tracker.progress_start("stage3", total=60, description="Stage 3")
    tracker.log_row("stage3", {"κ": "0.850", "ckpt": "✓"}, step=30)
    stream.flush()

    out = raw.getvalue().decode("ascii")
    assert "kappa: 0.850" in out
    assert "ckpt ok" in out


# ══════════════════════════════════════════════════════════════════════
#  Hyperparameter flattening
# ══════════════════════════════════════════════════════════════════════


def test_flatten_hyperparams_dots_nested_keys_and_stringifies_lists() -> None:
    flat = flatten_hyperparams({"stage1": {"max_lr": 5e-4}, "tracking": {"backends": ["console"]}})
    assert flat == {"stage1.max_lr": 5e-4, "tracking.backends": "['console']"}
