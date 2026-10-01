"""The ``jsonl`` backend: one JSON object per line in ``output_dir/metrics.jsonl``.

Both channels, machine-readable
───────────────────────────────
The console backend writes a run for people and W&B/TensorBoard write one for
their own viewers. A headless box — a Kaggle commit, an SSH sweep — needs a
record that is neither: something ``pandas.read_json(path, lines=True)`` or
``jq`` reads back after the session is gone, with no service and no SDK. This
backend writes every tracker call as one self-describing record::

    {"event": "scalars", "step": 12, "metrics": {"train/loss": 1.93, ...}, "t": ..., "elapsed_s": ...}
    {"event": "epoch", "tag": "single", "step": 12, "cells": {"Loss": "1.9312", ...}, ...}
    {"event": "table", "tag": "session/val_test", "step": 57, "rows": [...], ...}

``event`` is one of ``run_start``, ``hyperparams``, ``scalars``, ``table``,
``epoch``, ``message``, ``banner``, ``stage_start``, ``stage_stop`` and
``run_end``. Every record carries the wall-clock ``t`` (Unix seconds) and
``elapsed_s`` since this tracker opened.

I/O
───
The file is opened once, in **append** mode, so a resumed run continues the same
record — its ``run_start`` line marks the seam. Writes go through the file
object's buffer and are flushed on every ``epoch`` record, on warnings and at
close: a crash loses at most the lines of the epoch in flight, and training
never waits on a per-scalar ``fsync``. Non-finite floats are written as
``null`` so every line is strict JSON.

Only rank 0 builds a tracker (``train.py``), so there is exactly one writer.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

from spectralquadnet.tracking.base import MessageLevel

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch.nn as nn

#: The file name under ``output_dir``.
METRICS_FILE = "metrics.jsonl"


class JsonlTracker:
    """Append every tracker call to a JSON-lines file.

    Args:
        path: The file to append to. Parent directories are created.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: IO[str] | None = self.path.open("a", encoding="utf-8")
        self._t0 = time.time()
        self._write({"event": "run_start", "pid": os.getpid()}, flush=True)

    # ── Machine channel ──────────────────────────────────────────────

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self._write({"event": "scalars", "step": int(step), "metrics": {tag: value}})

    def log_scalars(self, tags: dict[str, float], step: int) -> None:
        self._write({"event": "scalars", "step": int(step), "metrics": dict(tags)})

    def log_table(self, tag: str, rows: list[dict[str, Any]], step: int) -> None:
        self._write({"event": "table", "tag": tag, "step": int(step), "rows": list(rows)})

    def log_hyperparams(self, cfg: dict[str, Any]) -> None:
        self._write({"event": "hyperparams", "config": cfg}, flush=True)

    def watch(self, model: nn.Module) -> None:
        return None

    def close(self) -> None:
        if self._fh is None:
            return
        self._write({"event": "run_end"}, flush=True)
        self._fh.close()
        self._fh = None

    # ── Human channel, recorded as events ────────────────────────────

    def banner(self, title: str, lines: Sequence[str] = ()) -> None:
        self._write({"event": "banner", "title": title, "lines": list(lines)})

    def log_message(self, text: str, level: MessageLevel = "info") -> None:
        self._write({"event": "message", "level": level, "text": text}, flush=level == "warn")

    def log_row(self, tag: str, cells: dict[str, str], step: int) -> None:
        self._write({"event": "epoch", "tag": tag, "step": int(step), "cells": dict(cells)}, True)

    def progress_start(self, tag: str, total: int, description: str = "") -> None:
        self._write(
            {"event": "stage_start", "tag": tag, "total": int(total), "description": description},
            flush=True,
        )

    def progress_stop(self, tag: str) -> None:
        self._write({"event": "stage_stop", "tag": tag}, flush=True)

    # ── Internals ────────────────────────────────────────────────────

    def _write(self, record: dict[str, Any], flush: bool = False) -> None:
        if self._fh is None:
            return
        now = time.time()
        record["t"] = round(now, 3)
        record["elapsed_s"] = round(now - self._t0, 3)
        self._fh.write(json.dumps(_jsonable(record), allow_nan=False, default=str) + "\n")
        if flush:
            self._fh.flush()


def _jsonable(value: Any) -> Any:
    """Plain JSON types: tensors and numpy scalars unwrapped, non-finite floats as ``None``."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if hasattr(value, "item") and callable(value.item) and getattr(value, "ndim", 0) == 0:
        value = value.item()  # a 0-d tensor or numpy scalar
        if isinstance(value, (bool, int)):
            return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    return value
