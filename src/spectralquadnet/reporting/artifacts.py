"""Where a run's results go, and in what shape.

Every run writes the same tree, so the protocol driver and the ablation
aggregator can read any run without knowing which pipeline produced it::

    <output_dir>/
      results/
        run.json                 identity + every scored split's metrics
        metrics_<split>.json     one scored split, standalone
        confusion_<split>.npy    (C, C) counts, rows = true class
        confusion_<split>.csv    the same, with headers, for a spreadsheet
        per_class_<split>.csv    class, f1, precision, recall, support
        preds_<split>.npy        argmax predictions
        targets_<split>.npy      ground truth
      figures/
        <figure>.png             written by spectralquadnet.reporting.figures

``run.json`` is the contract. ``scripts/run_protocol.py`` and the ablation
aggregator read it and nothing else, which is what lets a protocol cell be
re-run, moved or archived without breaking the table it feeds.

Artifacts are also handed to the tracker, so a W&B run carries its own
confusion matrix and per-class table rather than a number whose breakdown lives
on somebody's laptop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:  # pragma: no cover - typing only
    from spectralquadnet.reporting.metrics import ClassificationResult
    from spectralquadnet.tracking.base import ExperimentTracker

#: Subdirectory of ``output_dir`` holding machine-readable results.
RESULTS_DIR = "results"

#: Subdirectory of ``output_dir`` holding rendered figures.
FIGURES_DIR = "figures"

#: The one file an aggregator has to be able to find.
RUN_MANIFEST = "run.json"


@dataclass
class RunArtifacts:
    """Writer for one run's results tree."""

    output_dir: Path

    @classmethod
    def for_run(cls, output_dir: str | Path) -> RunArtifacts:
        root = Path(output_dir)
        (root / RESULTS_DIR).mkdir(parents=True, exist_ok=True)
        (root / FIGURES_DIR).mkdir(parents=True, exist_ok=True)
        return cls(output_dir=root)

    @property
    def results(self) -> Path:
        return self.output_dir / RESULTS_DIR

    @property
    def figures(self) -> Path:
        return self.output_dir / FIGURES_DIR

    # ── Writing ───────────────────────────────────────────────────────

    def write_predictions(
        self, split: str, preds: npt.NDArray[Any], targets: npt.NDArray[Any]
    ) -> None:
        """Persist the raw arrays, so any later metric can be recomputed offline.

        This is what makes the paired bootstrap between two arms possible after
        the fact: it needs both arms' predictions on the *same* patches, and a
        stored macro-F1 cannot supply that.
        """
        np.save(self.results / f"preds_{split}.npy", np.asarray(preds))
        np.save(self.results / f"targets_{split}.npy", np.asarray(targets))

    def write_result(self, result: ClassificationResult) -> Path:
        """Write one scored split's metrics, confusion matrix and per-class table."""
        split = result.split
        path = self.results / f"metrics_{split}.json"
        path.write_text(json.dumps(result.as_dict(), indent=2))

        np.save(self.results / f"confusion_{split}.npy", result.confusion)
        self._write_confusion_csv(split, result.confusion)
        self._write_per_class_csv(split, result)
        return path

    def write_manifest(self, payload: dict[str, Any]) -> Path:
        """Write (or merge into) ``run.json`` — the file aggregators read.

        Merged rather than overwritten because a run scores several splits and
        several TTA variants, each finishing at a different point, and a
        crash between two of them should leave the earlier ones readable.
        """
        path = self.results / RUN_MANIFEST
        existing: dict[str, Any] = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text())
            except json.JSONDecodeError:
                # A half-written manifest from a killed run is not worth
                # preserving, and refusing to start over would strand the run.
                existing = {}
        merged = _deep_merge(existing, payload)
        path.write_text(json.dumps(merged, indent=2, default=str))
        return path

    # ── CSV helpers ───────────────────────────────────────────────────

    def _write_confusion_csv(self, split: str, confusion: npt.NDArray[Any]) -> None:
        n = confusion.shape[0]
        header = "true\\pred," + ",".join(str(c) for c in range(n))
        rows = [header]
        rows.extend(f"{i}," + ",".join(str(int(v)) for v in confusion[i]) for i in range(n))
        (self.results / f"confusion_{split}.csv").write_text("\n".join(rows) + "\n")

    def _write_per_class_csv(self, split: str, result: ClassificationResult) -> None:
        rows = ["class,f1,precision,recall,support"]
        rows.extend(
            f"{r['class']},{r['f1']:.6f},{r['precision']:.6f},{r['recall']:.6f},{r['support']}"
            for r in result.per_class_rows()
        )
        (self.results / f"per_class_{split}.csv").write_text("\n".join(rows) + "\n")


def publish(
    artifacts: RunArtifacts,
    result: ClassificationResult,
    tracker: ExperimentTracker,
    step: int,
    prefix: str | None = None,
) -> None:
    """Write ``result`` to disk **and** to the tracker, in one call.

    The two must not drift: a W&B panel showing one macro-F1 and a
    ``metrics_*.json`` showing another is worse than either alone, and the only
    reliable way to prevent it is for one function to do both.
    """
    tag = prefix or result.split
    artifacts.write_result(result)
    tracker.log_scalars(result.scalars(tag), step=step)
    tracker.log_table(f"per_class/{tag}", result.per_class_rows(), step=step)


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge; ``update`` wins on scalar collisions."""
    out = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_manifest(run_dir: str | Path) -> dict[str, Any]:
    """Read one run's ``run.json``, or ``{}`` when it has none.

    Returns ``{}`` rather than raising so an aggregator can sweep a directory
    of runs where one crashed before it scored anything, report the gap, and
    still produce the table for the cells that finished.
    """
    path = Path(run_dir) / RESULTS_DIR / RUN_MANIFEST
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}
