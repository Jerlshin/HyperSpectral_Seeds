"""S14 shared paths, loaders and the cell index. Run every script from the repository root:

    python docs/research/evidence/S14_screen_reading/code/<script>.py

Reads the S13 screening cells (``outputs/experiments_u430k32/s13/{Y1,Y2,Y3,Y4}``) and, as references, the
S11 cells S12 already read (``outputs/experiments_u430k32/s11/{X1,X2}``). Reuses S12's loaders (which reuse
S09's row-aligned metadata: labels, sessions, the 17 cross-session varieties, the pipeline's own split
builders) rather than re-deriving anything. Small tables a claim rests on go to
``docs/research/evidence/S14_screen_reading/``.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "docs/research/evidence/S12_frozen_arms_reading/code"))
from s12common import (  # noqa: E402  (S12's loaders; importing them also loads S09's metadata and src/)
    CROSS,
    LABELS,
    N_CLASSES,
    SAME,
    SESSION,
    Cell,
    grouped_rows,
    load_logits,
    load_preds,
    macro_f1,
    recalls,
    softmax,
    stratified_rows,
    train_session_by_class,
)
from s12common import cells as s11_cells  # noqa: E402

SWEEP = REPO / "outputs" / "experiments_u430k32"
S13 = SWEEP / "s13"
EVID = REPO / "docs" / "research" / "evidence" / "S14_screen_reading"
S12_EVID = REPO / "docs" / "research" / "evidence" / "S12_frozen_arms_reading"
CACHE = REPO / "outputs" / "s14_reading"
CACHE.mkdir(parents=True, exist_ok=True)

S13_COMMIT = "aed5257d71603edf4021e7763e325dd146c84170"
PREREG = {
    "S12": REPO / "docs/research/evidence/S12_frozen_arms_reading/preregistration_s12.json",
    "S13": REPO / "docs/research/evidence/S13_representation_screening/preregistration_s13.json",
}
PREREG_SHA = {
    "S12": "88b377c5bc32f31a9ccb95356a88eb0be35e8cb51216b035c88f1761919ae7f4",
    "S13": "ef5982131df48c2ba8105c751fcf21f527a01ea2c0addf4ad59380f1a0989460",
}

# The frozen reference (preregistration_s12.json → reference) and the seed-matched X1 cells
# (preregistration_s13.json → reading.reported_beside_deciding_nothing). Read, never re-derived.
REF = dict(f1=0.530786, f1_sd=0.009933, same=0.648493, same_sd=0.013873, cross=0.146637, cross_sd=0.009823,
           attraction=0.463958, attraction_sd=0.020362, strat_f1=0.726984, strat_f1_sd=0.005806)
THRESH = dict(H19a=0.5108, H19b_cross=0.1666, H19b_attraction=0.4240, H20_cross=0.1666, H20_same=0.6285,
              H21a=0.5108, H21b=0.7090, H15_delta=0.03)

_CELL_RE = re.compile(r"([a-z_0-9]+)__f(\d)_s(\d)")

__all__ = [
    "CROSS", "LABELS", "N_CLASSES", "SAME", "SESSION", "Cell", "grouped_rows", "load_logits", "load_preds",
    "macro_f1", "recalls", "softmax", "stratified_rows", "train_session_by_class", "s11_cells", "s13_cells",
    "x1_matched", "jsonl", "scalars_by_epoch", "save", "verify_preregistrations", "run_json", "REF", "THRESH",
    "EVID", "S12_EVID", "CACHE", "S13", "REPO", "S13_COMMIT", "protocol", "conf", "f1_from_conf",
    "recall_from_conf", "attraction_cross", "SWEEP",
]


def s13_cells() -> list[Cell]:
    """Every S13 cell, GPU and fused (a fused cell has only ``results/``)."""
    out = []
    for arm in ("Y1", "Y2", "Y3", "Y4"):
        for d in sorted((S13 / arm).iterdir()):
            m = _CELL_RE.fullmatch(d.name)
            if m:
                out.append(Cell(arm, m.group(1), int(m.group(2)), int(m.group(3)), d))
    return out


def protocol(c: Cell) -> str:
    return "stratified" if c.variant in ("stratified", "lean_stratified", "within_8020") else "grouped"


def x1_matched(c: Cell) -> Cell:
    """The fold- and seed-matched X1 (R1) cell of an S13 cell."""
    v = "stratified" if protocol(c) == "stratified" else "grouped"
    return next(x for x in s11_cells(("X1",)) if x.variant == v and x.fold == c.fold and x.seed == c.seed)


def run_json(c: Cell) -> dict:
    return json.load(open(c.path / "results" / "run.json"))


def jsonl(c: Cell) -> list[dict]:
    return [json.loads(line) for line in open(c.path / "metrics.jsonl")]


def scalars_by_epoch(c: Cell) -> pd.DataFrame:
    """Every ``scalars`` event with an epoch step, merged by epoch (later events win on a duplicate key)."""
    rows: dict[int, dict] = {}
    for e in jsonl(c):
        if e.get("event") == "scalars" and e.get("step") is not None:
            rows.setdefault(int(e["step"]), {}).update(e["metrics"])
    df = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    df.index.name = "epoch"
    return df.reset_index()


def verify_preregistrations() -> None:
    """Refuse to aggregate a held-out row if either frozen file moved (WORKFLOW G1)."""
    for k, p in PREREG.items():
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        assert h == PREREG_SHA[k], f"{k} pre-registration hash moved: {h}"


def save(obj: pd.DataFrame | dict | list, name: str) -> None:
    EVID.mkdir(parents=True, exist_ok=True)
    p = EVID / name
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(p, index=False, float_format="%.6g")
    else:
        p.write_text(json.dumps(obj, indent=1, default=float))
    print(f"  wrote {p.relative_to(REPO)}")


def conf(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.bincount(t * N_CLASSES + p, minlength=N_CLASSES * N_CLASSES).reshape(N_CLASSES, N_CLASSES)


def f1_from_conf(c: np.ndarray) -> float:
    tp = np.diag(c).astype(float)
    den = c.sum(0) + c.sum(1)
    return float(np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0).mean())


def recall_from_conf(c: np.ndarray, classes: np.ndarray) -> float:
    sup = c.sum(1)[classes]
    return float((np.diag(c)[classes] / np.maximum(sup, 1)).mean())


def attraction_cross(rows: np.ndarray, preds: np.ndarray, fold: int) -> float:
    """Share of cross-session held-out kernels predicted as a class whose training bundle is in the kernel's own
    session — the same definition as ``reporting/session.py`` (checked against run.json in extract_cells.py)."""
    tr, _, _ = grouped_rows(fold)
    sess_of_class = train_session_by_class(tr)
    m = np.isin(LABELS[rows], CROSS)
    return float((sess_of_class[preds[m]] == SESSION[rows[m]]).mean())
