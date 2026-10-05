"""Held-out scoring shared by the S31+ screens (same arithmetic as the S27-S29 runners).

Every arm is scored on the S21 corrected folds: macro-F1, accuracy, same/cross-session
recall, cross-error session attraction and acquisition direction (test bundle inside or
outside session 8). Paired contrasts resample varieties with both folds retained.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from spectralquadnet.experiments.rgb_probe import class_metrics, cluster_interval

Array = npt.NDArray[Any]
S21 = Path("docs/research/evidence/S21_complementary_rgb/preregistration.json")
CLASSES = 90


@dataclass(frozen=True)
class Cohort:
    """Labels, session structure and S21 splits (dataset metadata never enters a model)."""

    y: Array
    session: Array
    cross: Array
    splits: dict[int, dict[str, Array]]

    @classmethod
    def load(cls, root: Path = Path("dataset_u430k32")) -> Cohort:
        y = np.load(root / "labels.npy")
        groups = np.load(root / "groups.npy")
        scans = pd.read_csv(root / "scan_table.csv").set_index("scan_id")
        session = scans.session_id.loc[groups].to_numpy()
        cross = scans.groupby("label").session_id.nunique().sort_index().to_numpy() > 1
        raw = json.loads(S21.read_text())["splits"]
        splits = {f: {k: np.asarray(v, dtype=int) for k, v in raw[str(f)].items()} for f in (0, 1)}
        return cls(y, session, cross, splits)

    def heldout(self, fold: int) -> Array:
        s = self.splits[fold]
        return np.sort(np.r_[s["val"], s["test"]])

    def train_rows(self, fold: int) -> Array:
        s = self.splits[fold]
        return np.concatenate([s["train"], s["calib"]])


def score(cohort: Cohort, fold: int, arm: str, te: Array, prob: Array) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Metrics row, per-class rows and prediction rows for one arm on one fold."""
    y, session = cohort.y, cohort.session
    if len(te) != len(prob) or not np.isfinite(prob).all():
        raise ValueError("Misaligned or nonfinite probabilities")
    pred = prob.argmax(1)
    stat = class_metrics(y[te], pred, CLASSES)
    tr = cohort.train_rows(fold)
    trained_in = {c: set(session[tr][y[tr] == c]) for c in range(CLASSES)}
    wrong = np.flatnonzero((pred != y[te]) & cohort.cross[y[te]])
    metrics = {"fold": fold, "arm": arm, "f1": float(stat["f1"].mean()), "accuracy": float(np.mean(pred == y[te])),
               "same_recall": float(stat["recall"][~cohort.cross].mean()),
               "cross_recall": float(stat["recall"][cohort.cross].mean()),
               "cross_error_session_attraction": float(np.mean([session[te][i] in trained_in[pred[i]] for i in wrong])) if len(wrong) else float("nan")}
    classes = []
    for c in range(CLASSES):
        dest = np.unique(session[te][y[te] == c])
        if len(dest) != 1:
            raise ValueError("Held-out class spans destination sessions")
        classes.append({"fold": fold, "arm": arm, "label": c, "cross": bool(cohort.cross[c]), "destination": int(dest[0]),
                        "f1": float(stat["f1"][c]), "recall": float(stat["recall"][c])})
    preds = [{"fold": fold, "arm": arm, "index": int(i), "target": int(t), "prediction": int(p)}
             for i, t, p in zip(te, y[te], pred, strict=True)]
    return metrics, classes, preds


def contrast(per_class: pd.DataFrame, a: str, b: str) -> dict[str, Any]:
    """Paired variety contrast a - b (both folds); F1 over all classes, recall over bridges."""
    cls = per_class.set_index(["arm", "label", "fold"]).sort_index()
    df = (cls.loc[a].f1 - cls.loc[b].f1).unstack("fold").to_numpy()
    dr = (cls.loc[a].recall - cls.loc[b].recall)[cls.loc[a].cross].unstack("fold").to_numpy()
    return {"delta_f1": float(df.mean()), "f1_ci": cluster_interval(df), "fold_f1_deltas": df.mean(0).tolist(),
            "delta_cross": float(dr.mean()), "cross_ci": cluster_interval(dr), "fold_cross_deltas": dr.mean(0).tolist()}


def gate(c: dict[str, Any], threshold: float) -> bool:
    """The project's screening gate: mean >= threshold, CI > 0, both folds > 0, cross >= 0."""
    return bool(c["delta_f1"] >= threshold and c["f1_ci"][0] > 0 and min(c["fold_f1_deltas"]) > 0 and c["delta_cross"] >= 0)


def summarise(metrics: pd.DataFrame, per_class: pd.DataFrame) -> pd.DataFrame:
    """Two-fold means per arm (never a maximum over folds) plus bridge recall by direction.

    Direction follows S22-S29: each bridge class x fold cell is ``to_session8`` when its
    held-out bundle lies in session 8, else ``from_session8``; recall is the mean over cells.
    """
    cols = ["f1", "accuracy", "same_recall", "cross_recall", "cross_error_session_attraction"]
    out = metrics.groupby("arm", sort=False)[cols].mean()
    bridge = per_class[per_class.cross].assign(direction=lambda d: np.where(d.destination == 8, "to_session8", "from_session8"))
    out = out.join(bridge.pivot_table(index="arm", columns="direction", values="recall", aggfunc="mean"))
    return out
