"""IC-1 — the cross-stage step axis, and the collision it removes.

The audited run logged ~200 ``Tried to log to step N that is less than the
current step 336`` warnings and **every Stage-2 and Stage-3 scalar was silently
discarded**: seven uploaded panels that stop at 336, containing zero information
about two thirds of the run. ``sam/grad_cos`` — the one measurement that would
have said whether Stage 3's SAM was doing anything — was computed and lost
(CHANGES §10.1).

The property that fixes it is monotonicity of the ``step`` a structured backend
sees, and these tests assert exactly that: whatever the stages number their own
epochs, the sequence crossing the tracker boundary never decreases.
"""

from __future__ import annotations

from typing import Any

import pytest

from spectralquadnet.tracking.base import ExperimentTracker
from spectralquadnet.tracking.global_step import GlobalStep, StepOffsetTracker, stage_tracker


class RecordingTracker:
    """Captures every call, so a test can assert on the step sequence."""

    def __init__(self) -> None:
        self.scalars: list[tuple[dict[str, float], int]] = []
        self.tables: list[tuple[str, int]] = []
        self.rows: list[tuple[str, int]] = []
        self.closed = 0

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.scalars.append(({tag: value}, step))

    def log_scalars(self, tags: dict[str, float], step: int) -> None:
        self.scalars.append((dict(tags), step))

    def log_table(self, tag: str, rows: list[dict[str, Any]], step: int) -> None:
        self.tables.append((tag, step))

    def log_hyperparams(self, cfg: dict[str, Any]) -> None:
        return None

    def watch(self, model: Any) -> None:
        return None

    def close(self) -> None:
        self.closed += 1

    def banner(self, title: str, lines: Any = ()) -> None:
        return None

    def log_message(self, text: str, level: str = "info") -> None:
        return None

    def log_row(self, tag: str, cells: dict[str, str], step: int) -> None:
        self.rows.append((tag, step))

    def progress_start(self, tag: str, total: int, description: str = "") -> None:
        return None

    def progress_stop(self, tag: str) -> None:
        return None


# ══════════════════════════════════════════════════════════════════════
#  GlobalStep
# ══════════════════════════════════════════════════════════════════════


def test_the_offset_starts_at_zero_so_a_single_stage_run_is_unchanged() -> None:
    clock = GlobalStep()
    assert clock.offset == 0
    assert clock.resolve(1) == 1
    assert clock.resolve(150) == 150


def test_advancing_moves_the_offset_by_the_epochs_actually_run() -> None:
    """Contiguity: an early-stopped stage must not leave a hole before the next."""
    clock = GlobalStep()
    clock.advance(336)  # Stage 1 early-stopped at 336 of a 400 budget
    assert clock.resolve(1) == 337, "Stage 2 epoch 1 follows Stage 1 epoch 336"
    clock.advance(49)
    assert clock.resolve(1) == 386


def test_the_offset_never_moves_backwards() -> None:
    """Monotonicity is the whole property W&B's step axis requires."""
    clock = GlobalStep(start=100)
    clock.advance(-50)
    assert clock.offset == 100


# ══════════════════════════════════════════════════════════════════════
#  The collision, reproduced and removed
# ══════════════════════════════════════════════════════════════════════


def test_without_rebasing_three_stages_replay_the_same_steps() -> None:
    """The defect. Kept as a test so the fix has something to be a fix *of*."""
    inner = RecordingTracker()
    for _stage, epochs in ((1, 4), (2, 3), (3, 3)):
        for ep in range(1, epochs + 1):
            inner.log_scalars({"val/f1_best": 0.5}, step=ep)

    steps = [step for _, step in inner.scalars]
    assert steps == [1, 2, 3, 4, 1, 2, 3, 1, 2, 3]
    assert any(
        b < a for a, b in zip(steps, steps[1:], strict=False)
    ), "the raw sequence decreases at every stage boundary — which is what W&B rejects"


def test_rebasing_makes_the_step_sequence_strictly_increasing() -> None:
    inner = RecordingTracker()
    clock = GlobalStep()

    for stage, epochs in ((1, 4), (2, 3), (3, 3)):
        trk = StepOffsetTracker(inner, clock, stage=stage)
        for ep in range(1, epochs + 1):
            trk.log_scalars({"val/f1_best": 0.5}, step=ep)
        clock.advance(epochs)

    steps = [step for _, step in inner.scalars]
    assert steps == list(range(1, 11))
    assert all(b > a for a, b in zip(steps, steps[1:], strict=False))


def test_the_stage_and_its_local_epoch_are_recoverable_from_the_scalars() -> None:
    """Flattening the axis must not destroy the per-stage view."""
    inner = RecordingTracker()
    clock = GlobalStep()
    clock.advance(336)

    StepOffsetTracker(inner, clock, stage=2).log_scalars({"val/f1_best": 0.844}, step=19)

    payload, step = inner.scalars[0]
    assert step == 355
    assert payload["progress/stage"] == 2.0
    assert payload["progress/stage_epoch"] == 19.0
    assert payload["val/f1_best"] == 0.844


def test_the_human_channel_keeps_the_stage_local_epoch() -> None:
    """`[Stage 2 | Ep 19/150]` must not become `Ep 355`."""
    inner = RecordingTracker()
    clock = GlobalStep()
    clock.advance(336)

    StepOffsetTracker(inner, clock, stage=2).log_row("stage2", {"Loss": "1.0"}, step=19)

    assert inner.rows == [("stage2", 19)]


def test_tables_are_rebased_alongside_the_scalars() -> None:
    inner = RecordingTracker()
    clock = GlobalStep()
    clock.advance(10)
    StepOffsetTracker(inner, clock, stage=2).log_table("hardest/x", [{"class": 1}], step=5)
    assert inner.tables == [("hardest/x", 15)]


def test_closing_a_stage_wrapper_does_not_close_the_run_s_backend() -> None:
    """The wrapper's lifetime is one stage; the run owns the backend across all of them."""
    inner = RecordingTracker()
    StepOffsetTracker(inner, GlobalStep(), stage=1).close()
    assert inner.closed == 0


def test_an_empty_scalar_group_is_not_forwarded() -> None:
    """Otherwise every no-diagnostic epoch would emit a bare `progress/*` pair."""
    inner = RecordingTracker()
    StepOffsetTracker(inner, GlobalStep(), stage=1).log_scalars({}, step=3)
    assert inner.scalars == []


def test_it_satisfies_the_tracker_protocol() -> None:
    """So it composes with MultiTracker and nests without a special case."""
    assert isinstance(StepOffsetTracker(RecordingTracker(), GlobalStep(), 1), ExperimentTracker)


@pytest.mark.parametrize(
    ("tracker", "clock"),
    [(None, GlobalStep()), (RecordingTracker(), None)],
)
def test_stage_tracker_passes_through_when_there_is_nothing_to_rebase(tracker, clock) -> None:
    """No tracker means log nowhere; no clock means a single stage, already monotone."""
    assert stage_tracker(tracker, clock, stage=1) is tracker
