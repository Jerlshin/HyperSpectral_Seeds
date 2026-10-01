"""S10 shared paths and loaders. Run every script from the repository root:

    python docs/research/evidence/S10_training_architecture_review/code/<script>.py

Reuses S09's split reconstruction (``grouped_rows`` / ``stratified_rows``) so both
studies read exactly the rows the sweep trained and selected on. Raw intermediates go
to ``outputs/s10_review/`` (git-ignored); the tables a claim rests on go to
``docs/research/evidence/S10_training_architecture_review/`` (tracked).

Partition discipline: every probe here reads ``train`` and ``calib`` rows only. No
held-out row is loaded by any S10 script.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "docs" / "research" / "evidence" / "S09_post_sweep_forensics" / "code"))

import common as s09  # noqa: E402  (S09's helpers: LABELS, GROUPS, grouped_rows, stratified_rows)

SWEEP = REPO / "outputs" / "experiments_u430k32" / "protocol"
CACHE = REPO / "outputs" / "s10_review"
EVID = REPO / "docs" / "research" / "evidence" / "S10_training_architecture_review"
DS32 = REPO / "dataset_u430k32"
CACHE.mkdir(parents=True, exist_ok=True)


def run_dirs() -> list[tuple[str, int, int, Path]]:
    """``(arm, fold, seed, path)`` for the 12 sweep runs, in a fixed order."""
    out = []
    for d in sorted(SWEEP.glob("*__f*_s*")):
        arm, rest = d.name.split("__")
        fold, seed = (int(t[1:]) for t in rest.split("_"))
        out.append((arm, fold, seed, d))
    return out


def run_config(d: Path):
    """The resolved config the run logged at start (``hyperparams`` event)."""
    from omegaconf import OmegaConf

    for line in open(d / "metrics.jsonl"):
        rec = json.loads(line)
        if rec["event"] == "hyperparams":
            return OmegaConf.create(rec["config"])
    raise RuntimeError(f"no hyperparams event in {d}")


def epoch_table(d: Path) -> pd.DataFrame:
    """Every per-epoch scalar the run logged, one row per epoch."""
    ep: dict[int, dict] = {}
    for line in open(d / "metrics.jsonl"):
        rec = json.loads(line)
        if rec["event"] == "scalars" and isinstance(rec.get("step"), int):
            ep.setdefault(rec["step"], {}).update(rec["metrics"])
    df = pd.DataFrame.from_dict(ep, orient="index").sort_index()
    df.index.name = "epoch"
    return df


def split_rows(arm: str, fold: int):
    """``(train, calib)`` rows only — the held-out rows are deliberately not returned."""
    tr, ca, _held = s09.grouped_rows(fold) if arm == "grouped" else s09.stratified_rows()
    return tr, ca


def save(df: pd.DataFrame | dict, name: str) -> None:
    EVID.mkdir(parents=True, exist_ok=True)
    p = EVID / name
    if isinstance(df, pd.DataFrame):
        df.to_csv(p, index=False, float_format="%.6g")
    else:
        p.write_text(json.dumps(df, indent=1, default=float))
    print(f"  wrote {p.relative_to(REPO)}")
