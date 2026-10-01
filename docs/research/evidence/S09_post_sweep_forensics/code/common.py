"""S09 shared paths and loaders. Run every script from the repository root:

    python docs/research/evidence/S09_post_sweep_forensics/code/<script>.py

Raw intermediates go to ``outputs/s09_forensics/`` (git-ignored); the small tables a
claim rests on go to ``docs/research/evidence/S09_post_sweep_forensics/`` (tracked).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "src"))

SWEEP = REPO / "outputs" / "experiments_u430k32"
CACHE = REPO / "outputs" / "s09_forensics"
EVID = REPO / "docs" / "research" / "evidence" / "S09_post_sweep_forensics"
DS32 = REPO / "dataset_u430k32"
DS215 = REPO / "dataset"
CACHE.mkdir(parents=True, exist_ok=True)

LABELS = np.load(DS32 / "labels.npy").astype(np.int64)
GROUPS = np.load(DS32 / "groups.npy").astype(np.int64)
SCANS = pd.read_csv(DS32 / "scan_table.csv").set_index("scan_id")
SESSION = SCANS.loc[GROUPS, "session_id"].to_numpy()          # session of every patch
N_CLASSES = 90

_n_sess = SCANS.groupby("label").session_id.nunique()
CROSS = np.array(sorted(_n_sess[_n_sess == 2].index))           # 17 cross-session varieties
SAME = np.array(sorted(_n_sess[_n_sess == 1].index))            # 73 same-session varieties
assert len(CROSS) == 17 and len(SAME) == 73


def grouped_rows(fold: int):
    """train / calib / held-out rows from the pipeline's own split builder, shipped params."""
    from spectralquadnet.data.loaders import SPLIT_SEED, grouped_split

    b = grouped_split(LABELS, GROUPS, eval_frac=0.3, calib_frac=0.15, fold=fold,
                      seed=SPLIT_SEED, single_group_policy="error")
    return np.asarray(b.train), np.asarray(b.calib), np.sort(np.concatenate([b.val, b.test]))


def stratified_rows():
    from spectralquadnet.data.loaders import _stratified_split

    b = _stratified_split(LABELS, 0.3, 0.15, GROUPS)
    return np.asarray(b.train), np.asarray(b.calib), np.sort(np.concatenate([b.val, b.test]))


def save(df: pd.DataFrame | dict, name: str) -> None:
    import json

    EVID.mkdir(parents=True, exist_ok=True)
    p = EVID / name
    if isinstance(df, pd.DataFrame):
        df.to_csv(p, index=False, float_format="%.6g")
    else:
        p.write_text(json.dumps(df, indent=1, default=float))
    print(f"  wrote {p.relative_to(REPO)}")
