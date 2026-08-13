"""Expanding an experiment into runs, and executing them (IC-12).

A run is a **subprocess** invoking ``train.py``, not an in-process call, and
that is a deliberate choice with three reasons:

* **Independence.** Hydra's ``GlobalHydra``, the ``DataStore`` singleton,
  torch's global RNG, cuDNN's autotune cache and the compiled-graph cache are
  all process-global. Two arms in one process are not two independent runs, and
  the whole point of an ablation is that its arms differ in exactly the declared
  override.
* **Resumability.** A grid of 60 runs will be interrupted. Each cell writes its
  own ``output_dir``, so :func:`plan` can skip the finished ones and the sweep
  picks up where it stopped.
* **Independent executability.** Every cell prints the exact command that
  produced it. A reviewer can re-run one number without running the grid, which
  is the property that makes the results checkable.

Naming
──────
A run directory encodes its cell: ``<experiment>/<arm>__f<fold>_s<seed>``. That
is what lets :func:`plan` detect completion, and what lets the aggregator
recover ``(arm, fold, seed)`` from a path when a manifest is missing.
"""

from __future__ import annotations

import json
import logging
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from spectralquadnet.experiments.registry import Ablation
from spectralquadnet.reporting.artifacts import RESULTS_DIR, RUN_MANIFEST

_log = logging.getLogger(__name__)

#: Where every experiment's runs live, relative to the repo root.
DEFAULT_OUTPUT_ROOT = "outputs/experiments"


@dataclass(frozen=True)
class RunSpec:
    """One training run: its identity, its overrides and where it writes."""

    experiment: str
    arm: str
    fold: int
    seed: int
    config: str
    overrides: tuple[str, ...]
    output_dir: str

    @property
    def run_name(self) -> str:
        return f"{self.arm}__f{self.fold}_s{self.seed}"

    def command(self, python: str | None = None, train_script: str = "train.py") -> list[str]:
        """The exact argv this cell runs. Printed so it can be re-run by hand."""
        return [
            python or sys.executable,
            train_script,
            f"--config-name={self.config}",
            *self.overrides,
            f"run_name={self.experiment}/{self.run_name}",
            f"output_dir={self.output_dir}",
            f"data.split_fold={self.fold}",
            f"seed={self.seed}",
        ]

    def shell(self) -> str:
        """The command as a copy-pasteable shell string."""
        return " ".join(shlex.quote(part) for part in self.command())

    @property
    def is_complete(self) -> bool:
        """Whether this cell already produced a results manifest.

        Completion is defined by the *manifest*, not by the checkpoint: a run
        that trained and then crashed in the evaluation has a checkpoint and no
        number, and re-running it is cheap because ``train.py`` auto-resumes
        from that checkpoint.
        """
        return (Path(self.output_dir) / RESULTS_DIR / RUN_MANIFEST).exists()


@dataclass
class RunOutcome:
    """What happened when a cell ran."""

    spec: RunSpec
    status: str  # "completed" | "skipped" | "failed"
    returncode: int = 0
    seconds: float = 0.0
    log_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm": self.spec.arm,
            "fold": self.spec.fold,
            "seed": self.spec.seed,
            "status": self.status,
            "returncode": self.returncode,
            "seconds": round(self.seconds, 1),
            "output_dir": self.spec.output_dir,
            "command": self.spec.shell(),
            "log": self.log_path,
        }


@dataclass
class SweepReport:
    """The outcome of a whole experiment, written next to its runs."""

    experiment: str
    outcomes: list[RunOutcome] = field(default_factory=list)

    @property
    def failed(self) -> list[RunOutcome]:
        return [o for o in self.outcomes if o.status == "failed"]

    @property
    def completed(self) -> list[RunOutcome]:
        return [o for o in self.outcomes if o.status in ("completed", "skipped")]

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "n_total": len(self.outcomes),
            "n_completed": len(self.completed),
            "n_failed": len(self.failed),
            "runs": [o.as_dict() for o in self.outcomes],
        }

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2))
        return path


def expand(
    ablation: Ablation,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    seeds: tuple[int, ...] | None = None,
    folds: tuple[int, ...] | None = None,
    arms: tuple[str, ...] | None = None,
) -> list[RunSpec]:
    """Expand an ablation into its cells: arms × folds × seeds.

    Args:
        arms: Restrict to these arm names, so one cell of a grid can be re-run
            on its own.

    De-duplication: an arm that pins ``data.split_fold`` itself (A1's
    ``stratified``, which has no folds to rotate) would otherwise produce
    identical cells at every fold. Those collapse to one.
    """
    root = Path(output_root) / ablation.key
    wanted = set(arms) if arms else None
    seen: set[str] = set()
    specs: list[RunSpec] = []

    for arm in ablation.arms:
        if wanted is not None and arm.name not in wanted:
            continue
        pins_fold = any(o.startswith("data.split_fold=") for o in arm.overrides)
        for fold in (folds or ablation.folds):
            for seed in (seeds or ablation.seeds):
                # `{fold}` in an override is substituted here so A2's per-fold
                # band arrays resolve without the config needing to know about
                # the sweep.
                resolved = tuple(o.replace("{fold}", str(fold)) for o in arm.overrides)
                spec = RunSpec(
                    experiment=ablation.key,
                    arm=arm.name,
                    fold=fold,
                    seed=seed,
                    config=arm.resolved_config(ablation.config),
                    overrides=resolved,
                    output_dir=str(root / f"{arm.name}__f{fold}_s{seed}"),
                )
                key = spec.output_dir if not pins_fold else f"{arm.name}_s{seed}"
                if key in seen:
                    continue
                seen.add(key)
                specs.append(spec)
    return specs


def plan(specs: list[RunSpec], force: bool = False) -> tuple[list[RunSpec], list[RunSpec]]:
    """Split cells into ``(to_run, already_done)``.

    ``force`` re-runs everything. Without it a completed cell is skipped, which
    is what makes a 60-run grid survivable across interruptions.
    """
    if force:
        return list(specs), []
    todo = [s for s in specs if not s.is_complete]
    done = [s for s in specs if s.is_complete]
    return todo, done


def execute(
    spec: RunSpec,
    dry_run: bool = False,
    python: str | None = None,
    train_script: str = "train.py",
    extra_overrides: tuple[str, ...] = (),
) -> RunOutcome:
    """Run one cell, capturing its output to ``<output_dir>/sweep.log``.

    ``train.py`` exits non-zero on any exception, so the return code is a
    reliable success signal. The log is kept whether or not the run succeeded —
    a failed cell's traceback is the only thing that explains a hole in the
    table.
    """
    if spec.is_complete:
        return RunOutcome(spec=spec, status="skipped")

    command = spec.command(python=python, train_script=train_script)
    command.extend(extra_overrides)
    if dry_run:
        _log.info("[dry-run] %s", " ".join(shlex.quote(c) for c in command))
        return RunOutcome(spec=spec, status="skipped")

    out_dir = Path(spec.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "sweep.log"

    started = time.perf_counter()
    with log_path.open("w") as log_file:
        log_file.write(" ".join(shlex.quote(c) for c in command) + "\n\n")
        log_file.flush()
        completed = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.perf_counter() - started

    return RunOutcome(
        spec=spec,
        status="completed" if completed.returncode == 0 else "failed",
        returncode=completed.returncode,
        seconds=elapsed,
        log_path=str(log_path),
    )


def run_all(
    specs: list[RunSpec],
    experiment: str,
    dry_run: bool = False,
    force: bool = False,
    python: str | None = None,
    train_script: str = "train.py",
    extra_overrides: tuple[str, ...] = (),
    stop_on_failure: bool = False,
) -> SweepReport:
    """Execute a list of cells sequentially and report what happened.

    Sequential on purpose: these are GPU runs and two at once on one device
    means both are slower and neither is the timing the paper reports. Parallel
    execution across *machines* is what the per-cell command exists for.

    ``stop_on_failure`` is off by default — one bad cell should not cost the
    other 59 — but the report names every failure and the CLI exits non-zero
    when any cell failed, so a broken grid cannot be mistaken for a finished
    one.
    """
    todo, done = plan(specs, force=force)
    report = SweepReport(experiment=experiment)
    report.outcomes.extend(RunOutcome(spec=s, status="skipped") for s in done)

    for i, spec in enumerate(todo, start=1):
        _log.info("[%s %d/%d] %s", experiment, i, len(todo), spec.run_name)
        outcome = execute(
            spec,
            dry_run=dry_run,
            python=python,
            train_script=train_script,
            extra_overrides=extra_overrides,
        )
        report.outcomes.append(outcome)
        if outcome.status == "failed":
            _log.error(
                "[%s] %s FAILED (rc=%d) — see %s",
                experiment,
                spec.run_name,
                outcome.returncode,
                outcome.log_path,
            )
            if stop_on_failure:
                break
    return report
