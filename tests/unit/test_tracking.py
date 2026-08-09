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
  line per epoch.

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


def test_console_row_emits_a_header_once_per_tag() -> None:
    def emit(t: ConsoleTracker) -> None:
        for ep in (1, 2, 3):
            t.log_row("stage1", {"Ep": f"{ep:03d}/600", "Loss": f"{ep / 10:.4f}"}, step=ep)

    out = _console_output({}, emit)
    assert out.count("Ep") == 1, "the header is emitted once, then rows stream"
    for ep in (1, 2, 3):
        assert f"{ep:03d}/600" in out


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


def test_a_row_is_styled_as_a_whole_line() -> None:
    """The plain (row-stream) progress mode styles the entire epoch line at once."""
    out = _coloured_output(lambda t: t.log_row("stage1", {"Ep": "181/600", "Ph": "P2"}, step=181))
    row = [line for line in out.splitlines() if "181/600" in line][0]
    assert row.startswith("\x1b[36m") and row.endswith("\x1b[0m")
    assert row.count("\x1b[0m") == 1, "one reset, at the end — not mid-row"


def test_a_bar_is_refused_on_a_non_interactive_console() -> None:
    """``progress=bar`` forced onto a dumb terminal degrades to rows, not to garbage.

    ``rich`` only repositions the cursor around a live region when the console is
    interactive; where it is not — a pipe, a dumb terminal, a web console that
    reports a TTY it cannot drive — it appends every frame instead, interleaving
    bar redraws with log lines. The row stream is the readable answer there.
    """
    buffer = io.StringIO()
    # `force_terminal` with `force_interactive=False` is exactly the shape of the
    # environments in the bug report: it claims a terminal, it cannot drive one.
    console = Console(file=buffer, width=200, force_terminal=True, force_interactive=False)
    tracker = ConsoleTracker(console=console, progress="bar")
    tracker.progress_start("stage1", total=600, description="Stage 1")
    tracker.log_row("stage1", {"Ep": "181/600", "Loss": "1.2345"}, step=181)
    tracker.progress_stop("stage1")

    out = buffer.getvalue()
    assert "181/600" in out, "the epoch line is rendered as a row"
    assert "\r" not in out, "no cursor rewind reaches a console that cannot honour it"


# ══════════════════════════════════════════════════════════════════════
#  Hyperparameter flattening
# ══════════════════════════════════════════════════════════════════════


def test_flatten_hyperparams_dots_nested_keys_and_stringifies_lists() -> None:
    flat = flatten_hyperparams({"stage1": {"max_lr": 5e-4}, "tracking": {"backends": ["console"]}})
    assert flat == {"stage1.max_lr": 5e-4, "tracking.backends": "['console']"}
