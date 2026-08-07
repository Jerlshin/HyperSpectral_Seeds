"""Branch-influence ablation, per-class difficulty and the §4.2 diagnostics.

Relocated from ``HSI_modality_training/hsi_training.py`` @ ``886560f``:

=====================================  ==============
Symbol                                 Baseline lines
=====================================  ==============
:func:`compute_branch_influence`       1333-1364
:func:`compute_class_difficulty`       2157-2183
=====================================  ==============

:func:`compute_branch_influence` relocates verbatim — it reads no ``CONFIG``.
:func:`compute_class_difficulty` carries the mechanical
``CONFIG["num_classes"]``/``["cdws_max_weight"]``/``["cdws_eps"]`` →
``cfg.data.num_classes`` / ``cfg.stage2.cdws_*`` rewrite and so takes ``cfg``
first.

:func:`compute_branch_influence` measures how much each branch matters by
zeroing it (``branch_mask``) and taking the KL divergence of the ablated
prediction from the full one, then normalising the four numbers to percentages.
Note the cost: ``max_batches × 5`` forward passes, which is why every caller
passes a small ``max_batches`` (3 from :func:`compute_class_difficulty`) and only
calls it on a checkpoint improvement.

Phase 4 additions (REFACTOR_PLAN.md §4.2)
─────────────────────────────────────────
All three are **thin wrappers around values that already exist** — no new
statistics, exactly as the plan requires:

* :func:`branch_grad_norms` — the per-branch generalisation of the total
  pre-clip norm that ``clip_grad_norm_`` already returns, using the same
  ``n.startswith(...)`` prefix filtering ``_wd_groups`` uses.
* :func:`hardest_classes_report` — sorts the ``class_f1`` dict
  :func:`~spectralquadnet.engine.evaluate.evaluate_per_class` already returns
  into a bottom-K table for ``tracker.log_table``.
* :func:`compute_class_difficulty` now routes ``compute_branch_influence``'s
  return value to ``tracker.log_scalars`` in addition to the line it already
  emitted. The computation is untouched.

The baseline's ``print`` in :func:`compute_class_difficulty` becomes
``tracker.log_message``. ``tracker`` defaults to ``None`` and is coerced to
:class:`~spectralquadnet.tracking.base.NullTracker`, so a caller that passes no
tracker gets silence rather than a second, competing output path — an
observability sink with no backend attached should emit nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from spectralquadnet.engine.evaluate import evaluate_per_class
from spectralquadnet.losses.cdws import build_cdws_weights
from spectralquadnet.tracking.base import ExperimentTracker, NullTracker

if TYPE_CHECKING:  # pragma: no cover - typing only
    from spectralquadnet.config.schema import ExperimentConfig

#: Parameter-name prefixes the per-branch gradient norm is grouped by.
#:
#: These are ``SpectralQuadNet``'s own attribute names, i.e. the same strings
#: ``build_optimizer_s2`` matches on (``n.startswith("arcface_head")``) and the
#: same ones REFACTOR_PLAN.md §3.1 pins as checkpoint-critical.
BRANCH_PREFIXES: tuple[str, ...] = (
    "branch_a.",
    "branch_b.",
    "branch_c.",
    "branch_d.",
    "cross_interaction.",
    "arcface_head.",
)


@torch.no_grad()
def compute_branch_influence(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    max_batches: int = 5,
) -> dict[str, float]:
    model.eval()
    influences = torch.zeros(4, device=device)
    total = 0

    for i, (x, _) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        logits_full = model(x)
        p_full = torch.softmax(logits_full, dim=1)

        for b in range(4):
            mask = torch.ones(4, device=device)
            mask[b] = 0.0
            logits_ab = model(x, branch_mask=mask)
            p_ab = torch.softmax(logits_ab, dim=1).clamp(min=1e-10)
            influences[b] += F.kl_div(p_ab.log(), p_full, reduction="batchmean")
        total += 1

    if total == 0:
        return {"A": 0, "B": 0, "C": 0, "D": 0}

    influences /= total
    total_inf = influences.sum().clamp(min=1e-8)
    influences = influences / total_inf * 100.0
    return {k: float(influences[i]) for i, k in enumerate("ABCD")}


def compute_class_difficulty(
    cfg: ExperimentConfig | Any,
    ema_shadow: nn.Module,
    val_ldr: DataLoader[Any],
    device: torch.device,
    label: str = "Stage",
    tracker: ExperimentTracker | None = None,
    step: int = 0,
) -> tuple[dict[int, float], dict[int, float]]:
    trk = tracker if tracker is not None else NullTracker()
    class_f1 = evaluate_per_class(ema_shadow, val_ldr, device, cfg.data.num_classes)
    cdws_wts = build_cdws_weights(
        class_f1, cfg.data.num_classes, cfg.stage2.cdws_max_weight, cfg.stage2.cdws_eps
    )
    macro = float(np.mean(list(class_f1.values())))
    n_hard = sum(1 for f in class_f1.values() if f < 0.50)

    branch_inf = compute_branch_influence(ema_shadow, val_ldr, device, max_batches=3)

    trk.log_message(
        f"{label} class difficulty — macro F1={macro:.3f}  "
        f"hard classes (<0.50 F1): {n_hard}/{cfg.data.num_classes}  |  "
        f"Branch influence % → "
        f"A:{branch_inf['A']:.1f}  B:{branch_inf['B']:.1f}  "
        f"C:{branch_inf['C']:.1f}  D:{branch_inf['D']:.1f}"
    )
    trk.log_scalars(
        {
            "diag/macro_f1": macro,
            "diag/hard_classes": float(n_hard),
            **{f"influence/branch_{k.lower()}": v for k, v in branch_inf.items()},
        },
        step=step,
    )
    trk.log_table(f"hardest_classes/{label}", hardest_classes_report(class_f1), step=step)
    return class_f1, cdws_wts


# ══════════════════════════════════════════════════════════════════════
#  Phase 4 additions (REFACTOR_PLAN.md §4.2)
# ══════════════════════════════════════════════════════════════════════


@torch.no_grad()
def branch_grad_norm_tensors(
    model: nn.Module, prefixes: tuple[str, ...] = BRANCH_PREFIXES
) -> dict[str, torch.Tensor]:
    """L2 gradient norm per branch, as 0-dim tensors left on the device.

    ``train_one_epoch``/``train_one_epoch_sam`` already call
    ``clip_grad_norm_(model.parameters(), cfg.grad_clip)``, which returns the
    *total* pre-clip norm across every parameter. This splits that same quantity
    by owner, so a branch whose gradient has collapsed (a real failure mode for
    the spectral branches A/B — the reason ``_compute_aux_loss`` weights them 2×)
    shows up as a curve instead of being inferred after the fact.

    Call it **before** the clip, or the numbers describe the clipped gradient.

    Nothing is moved to the host: this runs on every optimiser step, and a
    ``.item()`` per parameter would force a device synchronisation each time.
    Callers accumulate the tensors and convert once per epoch;
    :func:`branch_grad_norms` is the one-shot variant that does convert.

    Args:
        model: The live model, after ``backward()`` and before the clip.
        prefixes: Parameter-name prefixes to group by; defaults to
            :data:`BRANCH_PREFIXES`.

    Returns:
        ``{"branch_a": tensor(0.31), …}`` — one entry per prefix that owns at
        least one parameter with a gradient. Prefixes with no gradient are
        omitted rather than reported as 0.0, so a frozen head (Stage 1 freezes
        ``arcface_head``) stays distinguishable from one that trains but is flat.
    """
    squares: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        for prefix in prefixes:
            if name.startswith(prefix):
                sq = param.grad.detach().float().pow(2).sum()
                squares[prefix] = sq if prefix not in squares else squares[prefix] + sq
                break
    return {prefix.rstrip("."): value.sqrt() for prefix, value in squares.items()}


@torch.no_grad()
def branch_grad_norms(
    model: nn.Module, prefixes: tuple[str, ...] = BRANCH_PREFIXES
) -> dict[str, float]:
    """:func:`branch_grad_norm_tensors`, resolved to Python floats.

    Uses a single stacked device→host transfer for every group.
    """
    norms = branch_grad_norm_tensors(model, prefixes)
    if not norms:
        return {}
    keys = list(norms)
    values = torch.stack([norms[k] for k in keys]).cpu().tolist()
    return {k: float(v) for k, v in zip(keys, values, strict=True)}


def hardest_classes_report(class_f1: dict[int, float], k: int = 10) -> list[dict[str, Any]]:
    """Bottom-``k`` classes by F1, formatted for ``tracker.log_table``.

    A thin sort over the dict :func:`evaluate_per_class` already returns — the
    same per-class F1 that drives ``build_cdws_weights`` and
    ``HardClassOversampledSampler``'s ``hard_f1_thresh``. Ties break on class id
    so the report is stable across runs.

    Args:
        class_f1: ``{class_id: f1}`` for every class.
        k: How many of the worst classes to report.
    """
    ordered = sorted(class_f1.items(), key=lambda kv: (kv[1], kv[0]))[:k]
    return [
        {"rank": rank, "class": cls, "f1": round(f1, 4)}
        for rank, (cls, f1) in enumerate(ordered, start=1)
    ]
