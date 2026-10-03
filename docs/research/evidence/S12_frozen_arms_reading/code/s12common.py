"""S12 shared paths, loaders and the cell index. Run every script from the repository root:

    python docs/research/evidence/S12_frozen_arms_reading/code/<script>.py

Reads the S11 frozen-arm cells (``outputs/experiments_u430k32/s11/{X1,X2,X4}``) and the S08 reference
sweep (``outputs/experiments_u430k32/protocol``). Reuses S09's row-aligned metadata (labels, sessions,
the 17 cross-session varieties, the pipeline's own split builders) rather than re-deriving it.
Small tables a claim rests on go to ``docs/research/evidence/S12_frozen_arms_reading/``.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "docs/research/evidence/S09_post_sweep_forensics/code"))
from common import (  # noqa: E402  (S09's loaders; importing them also puts src/ on the path)
    CROSS,
    GROUPS,
    LABELS,
    N_CLASSES,
    SAME,
    SCANS,
    SESSION,
    grouped_rows,
    stratified_rows,
)

SWEEP = REPO / "outputs" / "experiments_u430k32"
S11 = SWEEP / "s11"
S08 = SWEEP / "protocol"
EVID = REPO / "docs" / "research" / "evidence" / "S12_frozen_arms_reading"
CACHE = REPO / "outputs" / "s12_reading"
CACHE.mkdir(parents=True, exist_ok=True)

S11_COMMIT = "413a11e55febbaa5d44019de800c4ee569a2aa8c"
PREREG = {
    "S09": REPO / "docs/research/evidence/S09_post_sweep_forensics/preregistration_next.json",
    "S10": REPO / "docs/research/evidence/S10_training_architecture_review/preregistration_s10.json",
}
PREREG_SHA = {
    "S09": "f896d0e569e0c071f85fb1c7e2fbce23cd3266da1ef317204fa329af9bb542e7",
    "S10": "1c8ae6937796cd7867f58ae901c1630671869f949ef8067cfefd72b2d4922739",
}

_CELL_RE = re.compile(r"([a-z_]+)__f(\d)_s(\d)")

__all__ = [
    "CROSS", "GROUPS", "LABELS", "N_CLASSES", "SAME", "SCANS", "SESSION", "grouped_rows",
    "stratified_rows", "Cell", "cells", "reference_cells", "jsonl", "scalars_by_epoch", "save",
    "verify_preregistrations", "load_logits", "load_preds", "macro_f1", "recalls", "softmax",
    "train_session_by_class", "EVID", "CACHE", "S11", "S08", "REPO",
]


@dataclass(frozen=True)
class Cell:
    """One trained run: an S11 frozen cell or an S08 reference run."""

    arm: str          # X1 · X2 · X4 · S08
    variant: str      # grouped · stratified · no_morph · spectral_only · spatial_only
    fold: int
    seed: int
    path: Path

    @property
    def name(self) -> str:
        return f"{self.arm}/{self.variant}__f{self.fold}_s{self.seed}"

    @property
    def protocol(self) -> str:
        return "stratified" if self.variant == "stratified" else "grouped"

    @property
    def scored(self) -> bool:
        return (self.path / "results" / "run.json").exists()


def cells(arms: tuple[str, ...] = ("X1", "X2", "X4")) -> list[Cell]:
    out = []
    for arm in arms:
        for d in sorted((S11 / arm).iterdir()):
            m = _CELL_RE.fullmatch(d.name)
            if m:
                out.append(Cell(arm, m.group(1), int(m.group(2)), int(m.group(3)), d))
    return out


def reference_cells() -> list[Cell]:
    """The 12 S08 runs (shipped regime) — the frozen reference of H12/H14/H16."""
    out = []
    for d in sorted(S08.iterdir()):
        m = _CELL_RE.fullmatch(d.name)
        if m:
            out.append(Cell("S08", m.group(1), int(m.group(2)), int(m.group(3)), d))
    return out


def jsonl(cell: Cell) -> list[dict]:
    return [json.loads(line) for line in open(cell.path / "metrics.jsonl")]


def scalars_by_epoch(cell: Cell) -> pd.DataFrame:
    """Every ``scalars`` event merged by epoch (later events win on a duplicate key)."""
    rows: dict[int, dict] = {}
    for e in jsonl(cell):
        if e.get("event") == "scalars" and e.get("step") is not None:
            rows.setdefault(int(e["step"]), {}).update(e["metrics"])
    df = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    df.index.name = "epoch"
    return df.reset_index()


def verify_preregistrations() -> None:
    """Refuse to read a held-out row if either frozen file moved (WORKFLOW G1)."""
    import hashlib

    for k, p in PREREG.items():
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        assert h == PREREG_SHA[k], f"{k} pre-registration hash moved: {h}"


def load_logits(cell: Cell, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(rows, logits float32, targets) sorted by row; ``split`` e.g. ``val_test_tta``, ``calib_no_tta``."""
    z = np.load(cell.path / "results" / f"logits_{split}.npz")
    r, lg, t = z["rows"], z["logits"].astype(np.float32), z["targets"]
    order = np.argsort(r)
    r, lg, t = r[order], lg[order], t[order]
    assert len(np.unique(r)) == len(r), f"{cell.name}: duplicate rows in {split}"
    assert (LABELS[r] == t).all()
    return r, lg, t


def load_preds(cell: Cell, variant: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(rows, preds, targets) with any DDP-padded duplicate dropped (S08 runs carry one), sorted."""
    d = cell.path / "results"
    r = np.load(d / f"rows_val_test_{variant}.npy")
    p = np.load(d / f"preds_val_test_{variant}.npy")
    t = np.load(d / f"targets_val_test_{variant}.npy")
    _, first = np.unique(r, return_index=True)
    r, p, t = r[first], p[first], t[first]
    assert (LABELS[r] == t).all()
    return r, p, t


def macro_f1(t: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(t, p, labels=np.arange(N_CLASSES), average="macro", zero_division=0))


def recalls(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    from sklearn.metrics import recall_score

    return recall_score(t, p, labels=np.arange(N_CLASSES), average=None, zero_division=0)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train_session_by_class(train_rows: np.ndarray) -> np.ndarray:
    """Class → the session of its (single) training bundle under a grouped fold."""
    out = np.full(N_CLASSES, -1)
    for c in range(N_CLASSES):
        s = np.unique(SESSION[train_rows][LABELS[train_rows] == c])
        if len(s) == 1:
            out[c] = s[0]
    return out


def save(obj: pd.DataFrame | dict | list, name: str) -> None:
    EVID.mkdir(parents=True, exist_ok=True)
    p = EVID / name
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(p, index=False, float_format="%.6g")
    else:
        p.write_text(json.dumps(obj, indent=1, default=float))
    print(f"  wrote {p.relative_to(REPO)}")
