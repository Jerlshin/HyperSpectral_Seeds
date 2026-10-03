"""The training-rows session κ — how much acquisition session a representation encodes (D26).

Under ``grouped`` every class is trained from one bundle, so inside the training
set the session is a function of the class: no supervised objective can tell a
session feature from a variety feature, and calib (same bundle) rewards both.
Cross-session behaviour shows only on held-out rows, which may select nothing.
S12 (F69) validated a proxy that never touches them:

**Class-disjoint session decodability.** On the fold's *training* rows, predict
each kernel's acquisition session from a representation with a classifier fitted
on some classes and scored on *other* classes — ``GroupKFold(5)`` with the class
as the group, ``StandardScaler`` → shrinkage LDA (``lsqr``, Ledoit–Wolf) — and
chance-correct the result as Cohen's κ. A representation that encodes session in
a way shared across varieties scores high; one whose session information is only
"which class is this" scores near zero. Across 13 linear representations it ranks
held-out cross-session attraction at Spearman 0.93, and it separates
spectral-only networks (κ 0.13–0.18) from every network with the spatial pathway
(κ 0.31–0.36); it does **not** rank within the latter (ρ 0.23).

What it is for (D26): reported beside the held-out session metrics, it may guard
a design choice made on calib; it never replaces a held-out confirmation and never
selects among close networks.

The classifier is a verbatim port of S12's
``evidence/S12_frozen_arms_reading/code/session_probe.py::probe``; the
representations are those of S12's ``embed_probe.py`` — the normalised embedding
the cosine head reads, and each live pathway's output — extracted from the
selected weights in eval mode, without augmentation, every training kernel once.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import torch
from torch.amp import autocast

from spectralquadnet.data.loaders import dedup_index
from spectralquadnet.engine.batch import side_inputs, unpack_batch
from spectralquadnet.utils.distributed import DistContext, gather_concat

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from spectralquadnet.reporting.session import SessionMap

#: Folds of the class-grouped cross-validation (S12).
N_SPLITS: int = 5

#: Written verbatim into every report, so the number carries its definition.
DEFINITION: str = (
    "class-disjoint session decodability on the fold's training rows: GroupKFold(5) by class, "
    "StandardScaler -> LDA(lsqr, shrinkage='auto'), Cohen's kappa of the predicted vs true "
    "acquisition session (S12 F69, D26); selected weights, eval mode, no augmentation"
)

#: The pathway modules whose outputs are probed besides the embedding, in order.
PATHWAY_MODULES: tuple[str, ...] = ("spatial", "spectral")


def session_kappa(
    features: npt.NDArray[Any],
    sessions: npt.NDArray[Any],
    classes: npt.NDArray[Any],
    n_splits: int = N_SPLITS,
) -> dict[str, Any]:
    """κ and balanced accuracy of predicting ``sessions`` from ``features``, class-disjointly.

    Args:
        features: ``(n, d)`` one row per training kernel.
        sessions: ``(n,)`` acquisition session of each row — the target.
        classes: ``(n,)`` variety of each row — the group: no class is ever in
            both the fitting and the scoring part of a fold.

    Returns:
        ``{"kappa", "balanced_acc", "n", "dim", "n_sessions", "n_classes"}``; the
        two scores are ``None`` (with a ``reason``) when the probe is undefined —
        fewer than two sessions, or fewer classes than folds.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(sessions).astype(np.int64)
    g = np.asarray(classes).astype(np.int64)
    out: dict[str, Any] = {
        "kappa": None,
        "balanced_acc": None,
        "n": int(len(y)),
        "dim": int(x.shape[1]) if x.ndim == 2 else 0,
        "n_sessions": int(np.unique(y).size),
        "n_classes": int(np.unique(g).size),
    }
    if out["n_sessions"] < 2:
        return {**out, "reason": "fewer than two sessions among the training rows"}
    if out["n_classes"] < n_splits:
        return {**out, "reason": f"fewer than {n_splits} classes for a class-grouped {n_splits}-fold"}
    pred = np.empty_like(y)
    with warnings.catch_warnings():
        # Collinear or constant columns (a zero-padded pathway, a saturated
        # unit) are the shrinkage estimator's business, not the report's.
        warnings.simplefilter("ignore")
        for fit, held in GroupKFold(n_splits=n_splits).split(x, y, g):
            model = make_pipeline(
                StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            ).fit(x[fit], y[fit])
            pred[held] = model.predict(x[held])
    return {
        **out,
        "kappa": float(cohen_kappa_score(y, pred)),
        "balanced_acc": float(balanced_accuracy_score(y, pred)),
    }


def live_pathways(model: nn.Module) -> tuple[str, ...]:
    """The pathway modules that run in ``model``'s forward (a disabled one is skipped)."""
    live = tuple(getattr(model, "pathways", PATHWAY_MODULES))
    return tuple(p for p in PATHWAY_MODULES if p in live and hasattr(model, p))


@torch.inference_mode()
def extract_representations(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    dist: DistContext | None = None,
) -> dict[str, npt.NDArray[np.float32]]:
    """``{"embedding", <pathway>…: (n, d)}`` over ``loader``, every kernel once, in dataset order.

    The embedding is the L2-normalised vector the cosine head reads; each live
    pathway's output is captured by a forward hook. fp32 (autocast off), eval
    mode, gathered across ranks and de-duplicated like every other evaluation
    pass. The model's train/eval flag is restored.
    """
    dist = dist or DistContext()
    captured: dict[str, list[torch.Tensor]] = {"embedding": []}
    hooks = []
    for name in live_pathways(model):
        captured[name] = []
        hooks.append(
            getattr(model, name).register_forward_hook(
                lambda _m, _i, out, key=name: captured[key].append(out.detach().float())
            )
        )
    was_training = model.training
    model.eval()
    try:
        with autocast(device_type=device.type, enabled=False):
            for batch in loader:
                x, _y, mask, morph = unpack_batch(batch, device)
                _logits, emb = model(x, return_embed=True, **side_inputs(mask, morph))
                captured["embedding"].append(emb.detach().float())
                del x, mask, morph, _logits, emb
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)

    keep = dedup_index(loader) if dist.enabled else None
    out: dict[str, npt.NDArray[np.float32]] = {}
    for name, parts in captured.items():
        if not parts:
            continue
        local = torch.cat(parts)
        width = int(local.shape[1])
        flat = gather_concat(dist, local.reshape(-1)).cpu().numpy()
        arr = flat.reshape(-1, width).astype(np.float32)
        out[name] = arr[keep] if keep is not None else arr
    return out


def session_probe_report(
    model: nn.Module,
    loader: DataLoader[Any],
    sessions: SessionMap,
    device: torch.device,
    dist: DistContext | None = None,
    weights: str | None = None,
) -> dict[str, Any] | None:
    """The run's session-κ block, or ``None`` on every rank but rank 0.

    ``loader`` must be an unshuffled evaluation loader over the **training rows**
    (its dataset's ``indices`` are the rows). The extraction is a collective, so
    every rank calls this; only rank 0 fits the probe.
    """
    dist = dist or DistContext()
    reps = extract_representations(model, loader, device, dist)
    if not dist.is_main:
        return None
    rows = np.asarray(loader.dataset.indices, dtype=np.int64)  # type: ignore[attr-defined]
    session = sessions.session_of_row[rows]
    classes = sessions.labels[rows]
    report: dict[str, Any] = {
        "definition": DEFINITION,
        "rows": "train",
        "weights": weights,
        "n_rows": int(rows.size),
        "representations": {},
    }
    for name, feats in reps.items():
        if feats.shape[0] != rows.size:
            report["representations"][name] = {
                "kappa": None,
                "reason": f"{feats.shape[0]} features for {rows.size} rows",
            }
            continue
        report["representations"][name] = session_kappa(feats, session, classes)
    return report


def scalars(report: dict[str, Any]) -> dict[str, float]:
    """``session_probe/kappa_<representation>`` for the tracker (defined values only)."""
    out: dict[str, float] = {}
    for name, entry in (report.get("representations") or {}).items():
        if entry.get("kappa") is not None:
            out[f"session_probe/kappa_{name}"] = float(entry["kappa"])
    return out


def lines(report: dict[str, Any]) -> list[str]:
    """One console line per representation."""
    out = [f"Session κ (training rows, class-disjoint, n={report.get('n_rows')}):"]
    for name, entry in (report.get("representations") or {}).items():
        if entry.get("kappa") is None:
            out.append(f"  {name:10s} undefined — {entry.get('reason', '?')}")
        else:
            out.append(
                f"  {name:10s} κ={entry['kappa']:.3f}  bal.acc={entry['balanced_acc']:.3f}  "
                f"(d={entry['dim']}, {entry['n_sessions']} sessions)"
            )
    return out
