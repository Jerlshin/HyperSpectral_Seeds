"""Same-session vs cross-session held-out performance, and collapse onto session identity.

Why a headline macro-F1 is not enough on this dataset
─────────────────────────────────────────────────────
The grouped protocol holds out whole acquisition *bundles*, but a bundle is not
the only acquisition unit. ``scan_table.csv`` records nine imaging **sessions**,
and 73 of the 90 varieties had both of their bundles imaged in the *same*
session; only 17 span two. So for 73 classes the held-out bundle shares its
session — lamp spectrum, detector state, room conditions — with the training
bundle of the same class, and a model that recognises the session has already
narrowed 90 candidates to the handful imaged that day.

The September 2026 band study measured what that does (``outputs/band_research``):
linear models on SNV mean spectra scored **0.000** held-out recall on the 17
cross-session varieties at every band budget, including the full cube, and
74-78 % of those kernels were predicted as a variety whose *training* bundle
came from the test kernel's own session, against ~15 % by chance. Every point of
their held-out score came from the same-session classes. The macro-F1 reported
by :func:`~spectralquadnet.reporting.metrics.score` cannot show any of this,
because it averages the two populations together.

What this module reports
────────────────────────
All of it is computed from the reported split's predictions plus two facts that
exist before any model is trained: the session of every patch
(``groups.npy`` → ``scan_table.csv``) and the sessions each class was *trained*
in (the fold's train ∪ calib rows).

* **Same- vs cross-session scores.** A held-out kernel is *same-session* when
  its true class has training data from the kernel's own session, and
  *cross-session* otherwise. A class is cross-session when every one of its
  scored kernels is. Macro-recall, macro-F1 (per-class F1 from the full
  confusion matrix, averaged over the subset's classes) and kernel accuracy are
  reported for each subset. Under the shipped grouped protocol the cross-session
  classes are exactly the 17 varieties whose bundles span two sessions.
* **Session attraction.** For every kernel, whether the *predicted* class was
  trained in the kernel's own session. Over cross-session kernels a correct
  prediction never is, so the rate is pure attraction. It is reported beside its
  chance level: the share of classes trained in that session, i.e. the rate a
  uniformly random prediction would show.
* **Session prediction entropy.** Each prediction is mapped to the training
  session(s) of the predicted class, and ``H(Ŝ | S)`` — the entropy, in bits, of
  that session given the kernel's own acquisition session — is averaged over
  sessions. It is reported beside two references computed from the same rows:
  the *oracle* ``H(S_y | S)`` a perfect classifier would produce, and the
  *chance* entropy of a uniformly random prediction. Below the oracle, the
  predicted session is more determined by the acquisition session than the
  truth is — a model fully collapsed onto session identity reaches zero. Above
  it, predictions are more dispersed than the truth, which is what scattered
  errors do. Entropy measures how *concentrated* the predicted sessions are, not
  *where*, so a model can sit above the oracle while its cross-session kernels
  still collapse: the CNN proxy of the band study scored 1.24 bits against an
  oracle of 0.66 with 71 % attraction. **The attraction rate is the collapse
  signal; entropy is the dispersion context around it.** Entropy is computed
  over all held-out kernels only — restricted to the cross-session subset it can
  be zero for the oracle and a collapsed model alike (one cross-session variety
  per session), which would say nothing.

None of this changes a prediction or a selection decision. It is a measurement,
written next to the headline numbers so that a reader can see which population
they came from.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import f1_score, recall_score

#: Columns ``scan_table.csv`` must provide. ``session`` (the name) is optional
#: and used only to label the per-session table.
REQUIRED_SCAN_COLUMNS: tuple[str, ...] = ("scan_id", "session_id")

#: Attraction above chance, over cross-session kernels, at which the report
#: states that predictions are collapsing onto session identity. Chosen well
#: clear of sampling noise on a ~1,600-kernel subset; the band study measured
#: +0.59 to +0.63 for mean-spectrum models.
ATTRACTION_FLAG_EXCESS: float = 0.25


# ══════════════════════════════════════════════════════════════════════
#  What each patch's session is, and where each class was trained
# ══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class SessionMap:
    """Per-row session ids and per-class training sessions for one fold.

    Attributes:
        session_of_row: ``(N,)`` acquisition session id of every patch row.
        labels: ``(N,)`` class of every patch row. Kept so a report can refuse a
            row order that does not reproduce the targets it was handed.
        train_sessions: ``{class: sessions}`` the class has training rows in.
        session_names: ``{session_id: name}`` for the per-session table.
        num_classes: Size of the label space.
    """

    session_of_row: npt.NDArray[np.int64]
    labels: npt.NDArray[np.int64]
    train_sessions: dict[int, frozenset[int]]
    session_names: dict[int, str]
    num_classes: int

    @classmethod
    def build(
        cls,
        scan_table: pd.DataFrame,
        groups: npt.NDArray[Any],
        labels: npt.NDArray[Any],
        fit_rows: npt.NDArray[Any],
        num_classes: int,
    ) -> SessionMap:
        """Join ``groups.npy`` to ``scan_table.csv`` and read off the training sessions.

        Args:
            scan_table: One row per scan, with at least :data:`REQUIRED_SCAN_COLUMNS`.
            groups: ``(N,)`` scan id of every patch.
            labels: ``(N,)`` class of every patch.
            fit_rows: Every row the model was fitted on — train ∪ calib.
            num_classes: Size of the label space.

        Raises:
            ValueError: A required column is missing, a patch's scan id is not in
                the table, or ``groups`` and ``labels`` disagree in length.
        """
        missing = [c for c in REQUIRED_SCAN_COLUMNS if c not in scan_table.columns]
        if missing:
            raise ValueError(f"scan table is missing column(s) {missing}")
        groups = np.asarray(groups).astype(np.int64)
        labels = np.asarray(labels).astype(np.int64)
        if groups.shape != labels.shape:
            raise ValueError(f"groups {groups.shape} and labels {labels.shape} must be row-aligned")

        scan_to_session = dict(
            zip(
                scan_table["scan_id"].astype(int).tolist(),
                scan_table["session_id"].astype(int).tolist(),
                strict=True,
            )
        )
        unknown = sorted(set(np.unique(groups).tolist()) - set(scan_to_session))
        if unknown:
            shown = ", ".join(str(g) for g in unknown[:10])
            raise ValueError(
                f"{len(unknown)} scan id(s) in groups.npy are not in the scan table "
                f"({shown}{' …' if len(unknown) > 10 else ''}) — the two files are from "
                "different extraction runs."
            )
        session_of_row = np.array([scan_to_session[int(g)] for g in groups], dtype=np.int64)

        fit = np.asarray(fit_rows).astype(np.int64)
        train_sessions = {
            c: frozenset(np.unique(session_of_row[fit][labels[fit] == c]).tolist())
            for c in range(int(num_classes))
        }
        names: dict[int, str] = {}
        if "session" in scan_table.columns:
            for sid, name in zip(scan_table["session_id"], scan_table["session"], strict=True):
                names.setdefault(int(sid), str(name))
        return cls(
            session_of_row=session_of_row,
            labels=labels,
            train_sessions=train_sessions,
            session_names=names,
            num_classes=int(num_classes),
        )

    @classmethod
    def from_config(
        cls, data_cfg: Any, splits: Any, num_classes: int
    ) -> tuple[SessionMap | None, str]:
        """``(map, reason)`` for a run, or ``(None, why not)``.

        Returns ``None`` rather than raising whenever the inputs do not exist,
        because the diagnostics are a measurement *about* a run and must never
        be the reason one fails: a stratified run without ``groups.npy``, or a
        config that leaves ``data.scan_table_path`` empty, simply reports no
        session breakdown and says so.

        Args:
            data_cfg: The ``data`` config group.
            splits: A :class:`~spectralquadnet.data.loaders.Splits` bundle.
            num_classes: Size of the label space.
        """
        table_path = str(getattr(data_cfg, "scan_table_path", "") or "")
        if not table_path:
            return None, "data.scan_table_path is empty — no session breakdown"
        if not Path(table_path).exists():
            return None, f"{table_path} does not exist — no session breakdown"
        groups = getattr(splits, "groups", None)
        if groups is None:
            return None, "the split carries no groups.npy — no session breakdown"
        fit_rows = np.concatenate([np.asarray(splits.train), np.asarray(splits.calib)])
        smap = cls.build(
            pd.read_csv(table_path), groups, np.asarray(splits.labels), fit_rows, num_classes
        )
        return smap, f"sessions from {table_path}"

    @property
    def n_sessions(self) -> int:
        return int(np.unique(self.session_of_row).size)


# ══════════════════════════════════════════════════════════════════════
#  The report
# ══════════════════════════════════════════════════════════════════════


@dataclass
class SubsetScore:
    """Scores over one population of held-out kernels (same- or cross-session)."""

    n_classes: int
    n_kernels: int
    macro_recall: float | None
    macro_f1: float | None
    accuracy: float | None
    classes: list[int] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_classes": self.n_classes,
            "n_kernels": self.n_kernels,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "accuracy": self.accuracy,
            "classes": list(self.classes),
        }


@dataclass
class EntropyScore:
    """``H(Ŝ | S)`` with its oracle and chance references, in bits."""

    predicted: float
    oracle: float
    chance: float
    maximum: float

    def as_dict(self) -> dict[str, float]:
        return {
            "predicted_bits": self.predicted,
            "oracle_bits": self.oracle,
            "chance_bits": self.chance,
            "max_bits": self.maximum,
        }


@dataclass
class SessionReport:
    """Everything the module docstring describes, for one scored split."""

    split: str
    n_kernels: int
    n_sessions: int
    n_duplicates_dropped: int
    macro_f1: float
    accuracy: float
    same: SubsetScore
    cross: SubsetScore
    n_mixed_classes: int
    attraction_all: float
    attraction_all_oracle: float
    attraction_all_chance: float
    attraction_cross: float | None
    attraction_cross_chance: float | None
    entropy_all: EntropyScore
    per_session: list[dict[str, Any]] = field(default_factory=list)

    @property
    def recall_gap(self) -> float | None:
        """Same-session minus cross-session macro-recall."""
        if self.same.macro_recall is None or self.cross.macro_recall is None:
            return None
        return self.same.macro_recall - self.cross.macro_recall

    def flags(self) -> list[str]:
        """Plain-language warnings a reader must see next to the headline number."""
        out: list[str] = []
        chance_recall = 1.0 / max(self.same.n_classes + self.cross.n_classes, 1)
        if self.cross.macro_recall is not None and self.cross.macro_recall <= chance_recall:
            out.append(
                f"cross-session macro-recall {self.cross.macro_recall:.3f} is at or below "
                f"chance ({chance_recall:.3f}): every point of the score comes from classes "
                "whose test bundle shares a session with their training bundle"
            )
        if (
            self.attraction_cross is not None
            and self.attraction_cross_chance is not None
            and self.attraction_cross - self.attraction_cross_chance > ATTRACTION_FLAG_EXCESS
        ):
            out.append(
                f"predictions collapse onto session identity: {self.attraction_cross:.0%} of "
                f"cross-session kernels are assigned a class trained in their own session "
                f"(chance {self.attraction_cross_chance:.0%})"
            )
        return out

    def scalars(self, prefix: str) -> dict[str, float]:
        """Tracker tags, e.g. ``val_test_tta/session/cross_macro_recall``."""
        base = f"{prefix}/session"
        tags: dict[str, float] = {
            f"{base}/attraction_all": self.attraction_all,
            f"{base}/attraction_all_chance": self.attraction_all_chance,
            f"{base}/entropy_bits": self.entropy_all.predicted,
            f"{base}/entropy_oracle_bits": self.entropy_all.oracle,
            f"{base}/entropy_chance_bits": self.entropy_all.chance,
        }
        optional = {
            f"{base}/same_macro_recall": self.same.macro_recall,
            f"{base}/same_macro_f1": self.same.macro_f1,
            f"{base}/cross_macro_recall": self.cross.macro_recall,
            f"{base}/cross_macro_f1": self.cross.macro_f1,
            f"{base}/recall_gap": self.recall_gap,
            f"{base}/attraction_cross": self.attraction_cross,
            f"{base}/attraction_cross_chance": self.attraction_cross_chance,
        }
        tags.update({k: float(v) for k, v in optional.items() if v is not None})
        return tags

    def as_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "n_kernels": self.n_kernels,
            "n_sessions": self.n_sessions,
            "n_duplicates_dropped": self.n_duplicates_dropped,
            "macro_f1": self.macro_f1,
            "accuracy": self.accuracy,
            "same_session": self.same.as_dict(),
            "cross_session": self.cross.as_dict(),
            "n_mixed_classes": self.n_mixed_classes,
            "recall_gap": self.recall_gap,
            "attraction": {
                "all": self.attraction_all,
                "all_oracle": self.attraction_all_oracle,
                "all_chance": self.attraction_all_chance,
                "cross": self.attraction_cross,
                "cross_chance": self.attraction_cross_chance,
            },
            "entropy": self.entropy_all.as_dict(),
            "flags": self.flags(),
            "per_session": list(self.per_session),
        }

    def lines(self) -> list[str]:
        """Console summary, three lines plus any flags."""

        def fmt(v: float | None) -> str:
            return "—" if v is None else f"{v:.4f}"

        out = [
            f"[Session] macroF1={self.macro_f1:.4f}  Acc={self.accuracy:.1%}  |  "
            f"same-session macro-recall {fmt(self.same.macro_recall)} "
            f"({self.same.n_classes} classes, {self.same.n_kernels} kernels)  |  "
            f"cross-session {fmt(self.cross.macro_recall)} "
            f"({self.cross.n_classes} classes, {self.cross.n_kernels} kernels)",
            f"[Session] attraction to own session: cross-session kernels "
            f"{fmt(self.attraction_cross)} (chance {fmt(self.attraction_cross_chance)})  |  "
            f"all kernels {self.attraction_all:.4f} (oracle {self.attraction_all_oracle:.4f}, "
            f"chance {self.attraction_all_chance:.4f})",
            f"[Session] H(Ŝ|S) = {self.entropy_all.predicted:.3f} bits  "
            f"(oracle {self.entropy_all.oracle:.3f}, chance {self.entropy_all.chance:.3f}, "
            f"max {self.entropy_all.maximum:.3f})",
        ]
        out.extend(f"[Session] ⚠ {flag}" for flag in self.flags())
        return out


def session_report(
    preds: npt.NDArray[Any],
    targets: npt.NDArray[Any],
    rows: npt.NDArray[Any],
    smap: SessionMap,
    split: str = "test",
) -> SessionReport:
    """Score one prediction array by acquisition session.

    Args:
        preds: ``(n,)`` predicted class per scored kernel.
        targets: ``(n,)`` true class per scored kernel.
        rows: ``(n,)`` patch row of every scored kernel, aligned with ``preds``.
            A row that appears twice is DDP padding (``DistributedSampler``
            repeats the first rows so every rank gets an equal shard); only its
            first occurrence is kept.
        smap: The fold's :class:`SessionMap`.
        split: Tag carried into the report.

    Raises:
        ValueError: The three arrays disagree in length, or ``rows`` does not
            reproduce ``targets`` — which means the row order was reconstructed
            wrongly and every per-session number would be about other kernels.
    """
    preds = np.asarray(preds).astype(np.int64)
    targets = np.asarray(targets).astype(np.int64)
    rows = np.asarray(rows).astype(np.int64)
    if not (len(preds) == len(targets) == len(rows)):
        raise ValueError(
            f"preds ({len(preds)}), targets ({len(targets)}) and rows ({len(rows)}) "
            "must be aligned"
        )
    _, first = np.unique(rows, return_index=True)
    first = np.sort(first)
    n_dup = int(len(rows) - len(first))
    preds, targets, rows = preds[first], targets[first], rows[first]
    if not np.array_equal(smap.labels[rows], targets):
        raise ValueError(
            "the row order does not reproduce the targets: labels[rows] != targets. "
            "The per-session numbers would describe other kernels, so none are reported."
        )

    num_classes = smap.num_classes
    labels = list(range(num_classes))
    sess = smap.session_of_row[rows]
    trained_in = smap.train_sessions

    true_in_session = np.array([s in trained_in[y] for s, y in zip(sess, targets, strict=True)])
    pred_in_session = np.array([s in trained_in[p] for s, p in zip(sess, preds, strict=True)])
    share_trained = _share_of_classes_trained_in(trained_in, num_classes)
    chance_in_session = np.array([share_trained.get(int(s), 0.0) for s in sess])

    per_recall = recall_score(targets, preds, labels=labels, average=None, zero_division=0)
    per_f1 = f1_score(targets, preds, labels=labels, average=None, zero_division=0)

    same_classes, cross_classes, mixed = _partition_classes(targets, true_in_session)
    same = _subset(same_classes, true_in_session, preds, targets, per_recall, per_f1)
    cross = _subset(cross_classes, ~true_in_session, preds, targets, per_recall, per_f1)

    cross_mask = ~true_in_session

    return SessionReport(
        split=split,
        n_kernels=int(len(rows)),
        n_sessions=int(np.unique(sess).size),
        n_duplicates_dropped=n_dup,
        macro_f1=float(np.mean(per_f1)),
        accuracy=float((preds == targets).mean()) if len(preds) else 0.0,
        same=same,
        cross=cross,
        n_mixed_classes=len(mixed),
        attraction_all=float(pred_in_session.mean()) if len(preds) else 0.0,
        attraction_all_oracle=float(true_in_session.mean()) if len(preds) else 0.0,
        attraction_all_chance=float(chance_in_session.mean()) if len(preds) else 0.0,
        attraction_cross=float(pred_in_session[cross_mask].mean()) if cross_mask.any() else None,
        attraction_cross_chance=(
            float(chance_in_session[cross_mask].mean()) if cross_mask.any() else None
        ),
        entropy_all=_entropy_score(sess, preds, targets, trained_in, num_classes),
        per_session=_per_session_rows(
            sess, preds, targets, true_in_session, pred_in_session, chance_in_session, smap
        ),
    )


# ══════════════════════════════════════════════════════════════════════
#  Pieces
# ══════════════════════════════════════════════════════════════════════


def _share_of_classes_trained_in(
    trained_in: dict[int, frozenset[int]], num_classes: int
) -> dict[int, float]:
    """``{session: fraction of classes with training rows in it}`` — the chance rate."""
    counts: dict[int, int] = {}
    for c in range(num_classes):
        for s in trained_in.get(c, frozenset()):
            counts[s] = counts.get(s, 0) + 1
    return {s: n / max(num_classes, 1) for s, n in counts.items()}


def _partition_classes(
    targets: npt.NDArray[np.int64], true_in_session: npt.NDArray[np.bool_]
) -> tuple[list[int], list[int], list[int]]:
    """Classes whose scored kernels are all same-session, all cross-session, or mixed."""
    same: list[int] = []
    cross: list[int] = []
    mixed: list[int] = []
    for c in np.unique(targets).tolist():
        flags = true_in_session[targets == c]
        if flags.all():
            same.append(int(c))
        elif not flags.any():
            cross.append(int(c))
        else:
            mixed.append(int(c))
    return same, cross, mixed


def _subset(
    classes: list[int],
    kernel_mask: npt.NDArray[np.bool_],
    preds: npt.NDArray[np.int64],
    targets: npt.NDArray[np.int64],
    per_recall: npt.NDArray[Any],
    per_f1: npt.NDArray[Any],
) -> SubsetScore:
    n_kernels = int(kernel_mask.sum())
    return SubsetScore(
        n_classes=len(classes),
        n_kernels=n_kernels,
        macro_recall=float(np.mean(per_recall[classes])) if classes else None,
        macro_f1=float(np.mean(per_f1[classes])) if classes else None,
        accuracy=float((preds[kernel_mask] == targets[kernel_mask]).mean()) if n_kernels else None,
        classes=sorted(classes),
    )


def _session_distribution(
    classes: npt.NDArray[np.int64], trained_in: dict[int, frozenset[int]], width: int
) -> npt.NDArray[np.float64]:
    """Mean over ``classes`` of the uniform distribution over each class's training sessions."""
    dist = np.zeros(width, dtype=np.float64)
    counted = 0
    for c in classes.tolist():
        sessions = trained_in.get(int(c), frozenset())
        if not sessions:
            continue
        for s in sessions:
            dist[s] += 1.0 / len(sessions)
        counted += 1
    return dist / counted if counted else dist


def _entropy_bits(p: npt.NDArray[np.float64]) -> float:
    q = p[p > 0]
    return float(-(q * np.log2(q)).sum()) if q.size else 0.0


def _entropy_score(
    sess: npt.NDArray[np.int64],
    preds: npt.NDArray[np.int64],
    targets: npt.NDArray[np.int64],
    trained_in: dict[int, frozenset[int]],
    num_classes: int,
) -> EntropyScore:
    """``H(Ŝ | S)`` for the predictions, the oracle (targets) and a uniform random guess."""
    all_sessions = {s for c in range(num_classes) for s in trained_in.get(c, frozenset())}
    width = int(max(all_sessions | set(sess.tolist()) | {0})) + 1
    n = max(len(sess), 1)
    h_pred = h_oracle = 0.0
    for s in np.unique(sess).tolist():
        m = sess == s
        w = float(m.sum()) / n
        h_pred += w * _entropy_bits(_session_distribution(preds[m], trained_in, width))
        h_oracle += w * _entropy_bits(_session_distribution(targets[m], trained_in, width))
    uniform = _session_distribution(np.arange(num_classes, dtype=np.int64), trained_in, width)
    return EntropyScore(
        predicted=h_pred,
        oracle=h_oracle,
        chance=_entropy_bits(uniform),
        maximum=float(np.log2(max(len(all_sessions), 1))),
    )


def _per_session_rows(
    sess: npt.NDArray[np.int64],
    preds: npt.NDArray[np.int64],
    targets: npt.NDArray[np.int64],
    true_in_session: npt.NDArray[np.bool_],
    pred_in_session: npt.NDArray[np.bool_],
    chance_in_session: npt.NDArray[np.float64],
    smap: SessionMap,
) -> list[dict[str, Any]]:
    """One row per held-out session — where the breakdown's numbers come from."""
    rows: list[dict[str, Any]] = []
    for s in np.unique(sess).tolist():
        m = sess == s
        cross = m & ~true_in_session
        rows.append(
            {
                "session_id": int(s),
                "session": smap.session_names.get(int(s), str(s)),
                "n_kernels": int(m.sum()),
                "n_cross_kernels": int(cross.sum()),
                "accuracy": float((preds[m] == targets[m]).mean()),
                "cross_accuracy": (
                    float((preds[cross] == targets[cross]).mean()) if cross.any() else None
                ),
                "attraction": float(pred_in_session[m].mean()),
                "attraction_chance": float(chance_in_session[m].mean()),
                "cross_attraction": float(pred_in_session[cross].mean()) if cross.any() else None,
            }
        )
    return rows


# ══════════════════════════════════════════════════════════════════════
#  Offline: re-score a finished run
# ══════════════════════════════════════════════════════════════════════


def rows_for_run(
    run_dir: Path,
    split_tag: str,
    labels: npt.NDArray[Any],
    groups: npt.NDArray[Any] | None,
    eval_frac: float,
    calib_frac: float,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """``(rows, fit_rows)`` for a finished run.

    ``rows_<split>.npy`` is read when the run wrote it. Runs from before it
    existed are reconstructed from ``run.json``: the split is rebuilt with the
    training pipeline's own builder and checked against the sizes the run
    recorded, and the report split's rows are taken in the order a
    single-process evaluation loader yields them. A DDP run scored in a
    different order; :func:`session_report`'s target check refuses it.

    Raises:
        FileNotFoundError: No manifest.
        ValueError: The rebuilt split does not match the one the run recorded.
    """
    from spectralquadnet.data.loaders import _stratified_split, grouped_split
    from spectralquadnet.reporting.artifacts import RESULTS_DIR, load_manifest

    manifest = load_manifest(run_dir)
    if not manifest:
        raise FileNotFoundError(f"{run_dir} has no {RESULTS_DIR}/run.json")
    run = manifest.get("run") or {}
    scheme = str(run.get("split_scheme", "grouped"))
    fold = int(run.get("split_fold", 0))
    if scheme == "grouped":
        if groups is None:
            raise ValueError("a grouped run needs groups.npy to rebuild its split")
        bundle = grouped_split(
            labels, groups, eval_frac=eval_frac, calib_frac=calib_frac, fold=fold
        )
    else:
        bundle = _stratified_split(labels, eval_frac, calib_frac, groups)

    recorded = ((run.get("split_report") or {}).get("sizes")) or {}
    rebuilt = {
        "train": len(bundle.train),
        "calib": len(bundle.calib),
        "val": len(bundle.val),
        "test": len(bundle.test),
    }
    if recorded and any(int(recorded.get(k, v)) != v for k, v in rebuilt.items()):
        raise ValueError(
            f"rebuilt split {rebuilt} does not match the run's recorded {recorded} — pass the "
            "--eval-frac / --calib-frac the run used"
        )
    fit_rows = np.concatenate([bundle.train, bundle.calib]).astype(np.int64)

    stored = run_dir / RESULTS_DIR / f"rows_{split_tag}.npy"
    if stored.exists():
        return np.load(stored).astype(np.int64), fit_rows
    report_split = str(run.get("report_split", "test"))
    if report_split == "val_test":
        rows = np.sort(np.concatenate([bundle.val, bundle.test]))
    else:
        rows = np.asarray(bundle.test)
    return rows.astype(np.int64), fit_rows


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m spectralquadnet.reporting.session RUN_DIR [RUN_DIR …]``.

    Writes ``results/session_<split>.json`` into each run and prints the summary.
    """
    from spectralquadnet.reporting.artifacts import RESULTS_DIR, RunArtifacts, load_manifest

    parser = argparse.ArgumentParser(description="Same- vs cross-session re-scoring of runs.")
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--variant", default="tta", choices=["tta", "no_tta"])
    parser.add_argument("--labels", default="./dataset/labels.npy")
    parser.add_argument("--groups", default="./dataset/groups.npy")
    parser.add_argument("--scan-table", default="./dataset/scan_table.csv")
    parser.add_argument("--eval-frac", type=float, default=0.30)
    parser.add_argument("--calib-frac", type=float, default=0.15)
    parser.add_argument("--num-classes", type=int, default=90)
    args = parser.parse_args(argv)

    labels = np.load(args.labels).astype(np.int64)
    groups = np.load(args.groups).astype(np.int64) if Path(args.groups).exists() else None
    table = pd.read_csv(args.scan_table)
    status = 0
    for run_dir in args.run_dirs:
        manifest = load_manifest(run_dir)
        result = (manifest.get("results") or {}).get(args.variant)
        if not result:
            print(f"{run_dir}: no '{args.variant}' result — skipped", file=sys.stderr)
            status = 1
            continue
        split_tag = str(result.get("split", f"test_{args.variant}"))
        try:
            rows, fit_rows = rows_for_run(
                run_dir, split_tag, labels, groups, args.eval_frac, args.calib_frac
            )
            if groups is None:
                raise ValueError("session diagnostics need groups.npy")
            smap = SessionMap.build(table, groups, labels, fit_rows, args.num_classes)
            preds = np.load(run_dir / RESULTS_DIR / f"preds_{split_tag}.npy")
            targets = np.load(run_dir / RESULTS_DIR / f"targets_{split_tag}.npy")
            report = session_report(preds, targets, rows, smap, split=split_tag)
        except (FileNotFoundError, ValueError) as exc:
            print(f"{run_dir}: {exc}", file=sys.stderr)
            status = 1
            continue
        RunArtifacts.for_run(run_dir).write_session(report)
        print(f"── {run_dir}")
        for line in report.lines():
            print("  " + line)
    return status


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
