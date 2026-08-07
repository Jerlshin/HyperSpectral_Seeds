"""Branch-influence ablation, per-class difficulty and per-branch gradient diagnostics.

:func:`compute_branch_influence` measures how much each branch matters by
zeroing it (``branch_mask``) and taking the KL divergence of the ablated
prediction from the full one, then normalising the four numbers to
percentages. Note the cost: ``max_batches x 5`` forward passes, which is why
every caller passes a small ``max_batches`` (3 from
:func:`compute_class_difficulty`) and only calls it on a checkpoint
improvement.

:func:`branch_grad_norms` and :func:`hardest_classes_report` are thin
wrappers around values that already exist elsewhere (the total pre-clip norm
``clip_grad_norm_`` returns, and the ``class_f1`` dict
:func:`~spectralquadnet.engine.evaluate.evaluate_per_class` returns) — no new
statistics are computed here.

``tracker`` parameters default to ``None`` and are coerced to
:class:`~spectralquadnet.tracking.base.NullTracker`, so a caller that passes
no tracker gets silence rather than a second, competing output path — an
observability sink with no backend attached should emit nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from spectralquadnet.engine.batch import side_inputs, unpack_batch
from spectralquadnet.engine.evaluate import evaluate_per_class
from spectralquadnet.losses.cdws import build_cdws_weights
from spectralquadnet.tracking.base import ExperimentTracker, NullTracker

if TYPE_CHECKING:  # pragma: no cover - typing only
    from spectralquadnet.config.schema import ExperimentConfig

#: Parameter-name prefixes the per-branch gradient norm is grouped by.
#:
#: These are ``SpectralQuadNet``'s own attribute names, the same strings
#: ``build_optimizer_s2`` matches on (``n.startswith("arcface_head")``) and
#: the same ones that are part of the checkpoint's state_dict schema.
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
    """Ablate each branch in turn and measure its influence via KL divergence.

    For each of ``max_batches`` batches, computes the full four-branch
    prediction once, then re-predicts with each branch individually zeroed
    (``branch_mask``); the KL divergence of the ablated distribution from the
    full one is that branch's raw influence score.

    Args:
        model: Model to probe; run in eval mode.
        loader: Data loader to draw batches from.
        device: Device to run the forward passes on.
        max_batches: Number of batches to average over. Costs
            ``max_batches * 5`` forward passes (1 full + 4 ablated per batch).

    Returns:
        ``{"A": pct, "B": pct, "C": pct, "D": pct}`` — influence normalised
        to sum to 100, or all zeros if the loader yielded no batches.
    """
    model.eval()
    influences = torch.zeros(4, device=device)
    total = 0

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x, _y, mask, morph = unpack_batch(batch, device)
        side = side_inputs(mask, morph)
        logits_full = model(x, **side)
        p_full = torch.softmax(logits_full, dim=1)

        for b in range(4):
            ablation = torch.ones(4, device=device)
            ablation[b] = 0.0
            logits_ab = model(x, branch_mask=ablation, **side)
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
    """Evaluate per-class F1, derive CDWS weights, and log a difficulty summary.

    Runs :func:`~spectralquadnet.engine.evaluate.evaluate_per_class` and
    :func:`compute_branch_influence` against ``ema_shadow``, builds fresh
    class-difficulty-weighted-sampling weights from the resulting F1s, and
    logs a macro-F1/hard-class-count/branch-influence message plus the
    matching scalars and a hardest-classes table.

    Args:
        cfg: Composed experiment config.
        ema_shadow: EMA model to evaluate (the shadow, not the raw model).
        val_ldr: Validation data loader.
        device: Device to evaluate on.
        label: Prefix for the logged message and table name (e.g. ``"Stage 2"``).
        tracker: Experiment tracker to log to; ``None`` logs nowhere.
        step: Step index passed through to the tracker's scalar/table logs.

    Returns:
        ``(class_f1, cdws_weights)`` — per-class F1 and the sampling weights
        derived from it.
    """
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
#  Per-branch gradient diagnostics
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


@torch.no_grad()
def flat_grad(model: nn.Module) -> torch.Tensor:
    """Every trainable parameter's gradient, concatenated into one 1-D tensor.

    Parameters that ``requires_grad`` but have no ``.grad`` yet contribute
    zeros, so two calls around the same model always return **aligned** vectors
    — which is what makes the cosine in :func:`grad_cosine` meaningful.

    Costs one full copy of the gradient (≈32 MB at 7.9 M fp32 parameters), so
    callers gate it on a diagnostics flag rather than running it every step
    unconditionally.
    """
    parts = [
        p.grad.detach().reshape(-1) if p.grad is not None else p.new_zeros(p.numel())
        for p in model.parameters()
        if p.requires_grad
    ]
    return torch.cat(parts) if parts else torch.zeros(0)


@torch.no_grad()
def grad_cosine(g_a: torch.Tensor, g_d: torch.Tensor) -> torch.Tensor:
    """``cos(ĝ_A, ĝ_D)`` between two flattened gradients, as a 0-dim device tensor.

    Stage 3's SAM step ascends along ``ĝ_A`` and descends along ``ĝ_D``. SAM is
    defined for a *single* objective: when the two differ, only the component
    of ``ĝ_A`` along ``ĝ_D`` is a sharpness penalty and the rest is
    curvature-shaped noise with no descent guarantee (IMPROVEMENT_PLAN §2.5.1
    C-6, Appendix A.3). This is the number that says which regime the run is
    in — it is 1.0 for matched objectives in the ``rho -> 0`` limit, and below
    it only by the curvature the ρ-perturbation actually traverses.

    Written as three dot products over max-normalised inputs rather than as
    ``F.cosine_similarity``, which computes its denominator with a different
    reduction from its numerator: over 7.9 M fp32 elements spanning 36 decades
    that disagreement reaches **5e-3**, enough to report 1.005 for a vector
    against itself and to swamp the effect being measured. Sharing one
    reduction makes the identical-gradient case exact, and the scaling keeps
    ``dot(a, a)`` clear of fp32's overflow ceiling. float64 is not an option:
    Metal does not implement it.
    """
    a = g_a / g_a.abs().max().clamp_min(torch.finfo(g_a.dtype).tiny)
    d = g_d / g_d.abs().max().clamp_min(torch.finfo(g_d.dtype).tiny)
    return torch.dot(a, d) / torch.sqrt(torch.dot(a, a) * torch.dot(d, d)).clamp_min(1e-30)


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
