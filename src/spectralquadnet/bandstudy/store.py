"""Persistence, resumability and the terminal's view of a long sweep.

Three things a multi-hour sweep needs and gets wrong if they are added later.

**Resumability that is a property of the artifacts, not of a checkpoint file.**
Every cell writes a record keyed by its own identity, and a cell whose key is
already present is skipped. So the sweep resumes after a laptop lid, a kill, a
crashed method or a machine change, and the resume is correct even if the run
that wrote the earlier records is long gone. There is no separate progress file
to fall out of sync with the results.

**A fingerprint check.** Resuming into a directory whose records came from a
different configuration would silently build one table out of two experiments.
The manifest records :meth:`~spectralquadnet.bandstudy.config.BandStudyConfig.fingerprint`
and a mismatch refuses rather than appends.

**Append-only JSONL rather than a single JSON document.** A 10,000-row result
file rewritten after every cell is 10,000 rewrites and one truncated file away
from losing the lot. JSONL appends, survives a kill mid-write with the loss of
at most the last line, and streams into pandas without a parser of its own.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spectralquadnet.bandstudy.config import BandStudyConfig, StageResult

_log = logging.getLogger("spectralquadnet.bandstudy")

#: Name of the append-only record file inside each stage's directory.
RECORDS = "records.jsonl"

#: The manifest that identifies the study a directory holds.
MANIFEST = "study.json"


# ══════════════════════════════════════════════════════════════════════
#  Logging
# ══════════════════════════════════════════════════════════════════════


def setup_logging(cfg: BandStudyConfig, stage: str) -> Path:
    """Attach a file handler for this stage and return its path.

    Both a human-readable log and machine-readable records are written, and
    they carry different things on purpose: the JSONL holds what a table needs,
    the log holds what a post-mortem needs — the failures, the timings, the
    warnings, and the one line per revealed held-out split.
    """
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.logs_dir / f"{stage}_{time.strftime('%Y%m%d-%H%M%S')}.log"

    root = logging.getLogger("spectralquadnet")
    root.setLevel(logging.DEBUG if cfg.verbose else logging.INFO)

    file_handler = logging.FileHandler(path)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    )
    file_handler.setLevel(logging.DEBUG)
    root.addHandler(file_handler)

    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(logging.Formatter("%(levelname)-7s %(message)s"))
        stream.setLevel(logging.DEBUG if cfg.verbose else logging.INFO)
        root.addHandler(stream)

    root.info("=== stage %s | fingerprint %s | pid %d ===", stage, cfg.fingerprint(), os.getpid())
    return path


# ══════════════════════════════════════════════════════════════════════
#  The record store
# ══════════════════════════════════════════════════════════════════════


@dataclass
class RecordStore:
    """Append-only JSONL keyed for resumption.

    Args:
        directory: Where ``records.jsonl`` lives.
        key_fields: The record fields whose tuple identifies a cell. Two
            records with the same key are the same experiment run twice, and
            the second is skipped unless ``force``.
    """

    directory: Path
    key_fields: Sequence[str]
    force: bool = False

    def __post_init__(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._done: set[tuple[Any, ...]] = set()
        self._handle: Any = None
        if not self.force:
            self._done = {self.key(r) for r in self.read()}

    @property
    def path(self) -> Path:
        return self.directory / RECORDS

    def key(self, record: dict[str, Any]) -> tuple[Any, ...]:
        return tuple(record.get(f) for f in self.key_fields)

    def read(self) -> list[dict[str, Any]]:
        """Every record on disk. A truncated final line is dropped, not fatal."""
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        with self.path.open() as handle:
            for line_no, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    _log.warning(
                        "%s line %d is truncated (a killed run) — dropping it and continuing",
                        self.path,
                        line_no,
                    )
        return out

    def has(self, **identity: Any) -> bool:
        """Whether a cell with this identity already has a record."""
        return tuple(identity.get(f) for f in self.key_fields) in self._done

    def append(self, record: dict[str, Any]) -> None:
        """Write one record and mark its cell done.

        Flushed per record rather than per batch: the whole point is that a kill
        between two cells loses nothing, and a buffered writer would lose
        whatever the OS had not yet paged out.
        """
        if self._handle is None:
            mode = "w" if (self.force and not self._done) else "a"
            self._handle = self.path.open(mode)
        self._handle.write(json.dumps(record, default=_json_default) + "\n")
        self._handle.flush()
        self._done.add(self.key(record))

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> RecordStore:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _json_default(value: Any) -> Any:
    """Serialise numpy scalars and arrays, which json refuses on its own."""
    import numpy as np

    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


# ══════════════════════════════════════════════════════════════════════
#  The study manifest
# ══════════════════════════════════════════════════════════════════════


def check_or_write_manifest(cfg: BandStudyConfig) -> dict[str, Any]:
    """Write the study manifest, or verify a resume is into the same study.

    Raises:
        ValueError: The directory holds results from a configuration with a
            different fingerprint. The message names both and the two ways out —
            a fresh ``--output-root`` or an explicit ``--force``.
    """
    cfg.root.mkdir(parents=True, exist_ok=True)
    path = cfg.root / MANIFEST
    payload = {
        "fingerprint": cfg.fingerprint(),
        "config": cfg.as_dict(),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stages": {},
    }

    if not path.exists():
        path.write_text(json.dumps(payload, indent=2, default=_json_default))
        return payload

    existing = json.loads(path.read_text())
    if existing.get("fingerprint") != cfg.fingerprint() and not cfg.force:
        raise ValueError(
            f"{cfg.root} holds results from a different study configuration "
            f"(fingerprint {existing.get('fingerprint')}, this run is {cfg.fingerprint()}). "
            "Resuming would build one table out of two experiments. Use a fresh "
            "--output-root, or --force to overwrite this one. The differing fields are:\n  "
            + "\n  ".join(_diff_configs(existing.get("config", {}), cfg.as_dict()))
        )
    if cfg.force:
        existing["fingerprint"] = cfg.fingerprint()
        existing["config"] = cfg.as_dict()
        existing["forced_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        path.write_text(json.dumps(existing, indent=2, default=_json_default))
    return dict(existing)


def _diff_configs(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    keys = sorted(set(old) | set(new))
    return [
        f"{k}: {old.get(k)!r} -> {new.get(k)!r}"
        for k in keys
        if k not in ("fingerprint", "note", "output_root", "jobs", "verbose", "force", "dry_run")
        and old.get(k) != new.get(k)
    ]


def record_stage(cfg: BandStudyConfig, result: StageResult) -> None:
    """Merge one stage's outcome into the manifest."""
    path = cfg.root / MANIFEST
    payload = json.loads(path.read_text()) if path.exists() else {"stages": {}}
    payload.setdefault("stages", {})[result.stage] = {
        **result.as_dict(),
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path.write_text(json.dumps(payload, indent=2, default=_json_default))


# ══════════════════════════════════════════════════════════════════════
#  Terminal
# ══════════════════════════════════════════════════════════════════════


def console() -> Any:
    """A ``rich`` console, or a stdout shim when rich is unavailable.

    ``rich`` is a core dependency, so the shim is defensive rather than
    expected — but a study driver that cannot start because a *display* library
    is missing would be an absurd failure mode.
    """
    try:
        from rich.console import Console

        return Console()
    except ImportError:  # pragma: no cover - rich is a core dependency

        class _Shim:
            def print(self, *args: Any, **_: Any) -> None:
                print(*[_plain(a) for a in args])

            def rule(self, title: str = "", **_: Any) -> None:
                print(f"\n── {_plain(title)} " + "─" * max(0, 60 - len(str(title))))

        return _Shim()


def _plain(value: Any) -> str:
    """Strip rich markup so the shim's output is readable."""
    import re

    return re.sub(r"\[/?[a-z0-9 ._#]+\]", "", str(value))


@contextmanager
def progress(cfg: BandStudyConfig, description: str, total: int) -> Iterator[Any]:
    """A progress bar that degrades to a no-op callable.

    Yields a ``advance(n=1)`` callable rather than a rich task handle, so the
    call sites do not branch on whether rich is present.
    """
    if not cfg.progress or total <= 0:
        yield lambda n=1: None
        return
    try:
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )
    except ImportError:  # pragma: no cover
        yield lambda n=1: None
        return

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("eta"),
        TimeRemainingColumn(),
        transient=False,
    ) as bar:
        task = bar.add_task(description, total=total)
        yield lambda n=1: bar.advance(task, n)


def summary_table(title: str, headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> Any:
    """A ``rich`` table, or a plain-text one when rich is unavailable."""
    body = [list(r) for r in rows]
    try:
        from rich.table import Table

        table = Table(title=title, header_style="bold")
        for header in headers:
            table.add_column(str(header))
        for row in body:
            table.add_row(*[str(c) for c in row])
        return table
    except ImportError:  # pragma: no cover
        widths = [
            max(len(str(h)), *(len(str(r[i])) for r in body)) if body else len(str(h))
            for i, h in enumerate(headers)
        ]
        lines = [title, "  ".join(str(h).ljust(w) for h, w in zip(headers, widths, strict=True))]
        lines += ["  ".join(str(c).ljust(w) for c, w in zip(r, widths, strict=True)) for r in body]
        return "\n".join(lines)
