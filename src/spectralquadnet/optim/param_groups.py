"""Weight-decay parameter grouping, per-group clipping, and the optimiser builders.

The two grouping rules are load-bearing:

* **No weight decay on 1-D tensors or biases** — norms, ECA/CBAM gates and every
  ``.bias`` land in the ``no_wd`` group.
* **Stage 2 splits on the ``arcface_head`` name prefix**, giving the head
  ``cfg.stage2.head_lr`` and the backbone ``cfg.stage2.back_lr``. The resulting
  group order — head-wd, head-no-wd, backbone-wd, backbone-no-wd — is what
  ``stage2_arcface.py`` reads back as ``param_groups[0]`` and ``[2]`` when it
  logs the two learning rates, so the concatenation order must not change.

:func:`clip_grad_norm_by_group` is OP-3 / T2-5. One global
``clip_grad_norm_(model.parameters(), 1.0)`` rescales *every* parameter by
``1/||g||`` whenever the total norm exceeds the threshold, and the head's
gradients are amplified by ``s = 48`` relative to the backbone's. A single
saturated batch in the head therefore divided the whole model's effective
learning rate (§2.5.8 M-10). Clipping the three groups independently decouples
them.
"""

from __future__ import annotations

import inspect
import weakref
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.optim as optim

if TYPE_CHECKING:  # pragma: no cover - typing only
    from spectralquadnet.config.schema import ExperimentConfig

#: The three independently-clipped parameter groups, in match order.
#:
#: A parameter joins the first group whose prefix tuple it matches; anything
#: unmatched is ``backbone``, which is why that entry's tuple is empty. The
#: prefixes are ``SpectralQuadNet``'s own attribute names, the same strings
#: ``build_optimizer_s2`` and ``engine/diagnostics.py`` match on.
CLIP_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("head", ("arcface_head.",)),
    ("fusion", ("cross_interaction.", "embed_net.")),
    ("backbone", ()),
)


def _wd_groups(
    cfg: ExperimentConfig | Any,
    named_params: Iterable[tuple[str, torch.nn.Parameter]],
    lr: float,
) -> list[dict[str, Any]]:
    """Split named parameters into weight-decayed and non-decayed AdamW groups.

    Args:
        cfg: Composed experiment config, read for ``cfg.weight_decay``.
        named_params: Iterable of ``(name, parameter)`` pairs, typically from
            ``model.named_parameters()`` or a filtered subset of it.
        lr: Learning rate applied to both resulting groups.

    Returns:
        Two param-group dicts: weight-decayed params first, then the
        zero-weight-decay group (1-D tensors and biases).
    """
    wd, no_wd = [], []  # type: ignore[var-annotated]
    for n, p in named_params:
        if not p.requires_grad:
            continue
        (no_wd if (p.ndim == 1 or n.endswith(".bias")) else wd).append(p)
    return [
        {"params": wd, "lr": lr, "weight_decay": cfg.weight_decay},
        {"params": no_wd, "lr": lr, "weight_decay": 0.0},
    ]


#: ``model -> {group: parameters}``, so the name-prefix match runs once per
#: model rather than once per optimiser step. Same rationale — and the same
#: weak keys — as ``engine/diagnostics.py::_BRANCH_GROUP_CACHE``: the partition
#: is a function of the parameter names, which do not change, but
#: :func:`clip_grad_norm_by_group` is on the inner loop and was re-deriving it
#: every step.
_CLIP_GROUP_CACHE: weakref.WeakKeyDictionary[nn.Module, dict[str, list[torch.nn.Parameter]]] = (
    weakref.WeakKeyDictionary()
)


def split_by_clip_group(model: nn.Module) -> dict[str, list[torch.nn.Parameter]]:
    """Partition a model's trainable parameters into the :data:`CLIP_GROUPS`.

    Every trainable parameter lands in exactly one group, so the three
    per-group clips together cover the same set the one global clip did.
    Groups that own no parameter are omitted rather than returned empty.

    The result is memoised per model. ``requires_grad`` is read at partition
    time, as it always was — nothing in the three stages toggles it after
    construction since HD-1 removed the frozen-head phase, and a caller that
    starts doing so must invalidate this cache.
    """
    cached = _CLIP_GROUP_CACHE.get(model)
    if cached is not None:
        return cached
    groups: dict[str, list[torch.nn.Parameter]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        for label, prefixes in CLIP_GROUPS:
            if not prefixes or name.startswith(prefixes):
                groups.setdefault(label, []).append(param)
                break
    _CLIP_GROUP_CACHE[model] = groups
    return groups


def clip_grad_norm_by_group(model: nn.Module, max_norm: float) -> dict[str, torch.Tensor]:
    """Clip head, fusion and backbone gradients independently (OP-3 / T2-5).

    Call it exactly where the single global ``clip_grad_norm_`` used to be:
    after ``backward()`` (and after ``scaler.unscale_``), before the step.

    Args:
        model: The live model, with gradients populated.
        max_norm: Per-group maximum L2 norm — the same ``cfg.grad_clip`` the
            global clip used, now applied three times rather than once.

    Returns:
        ``{"head": tensor, "fusion": tensor, "backbone": tensor}`` — each
        group's **pre-clip** norm, left on the device as a 0-dim tensor so a
        caller running this every step is not forced into a host
        synchronisation. Groups owning no parameter are absent.
    """
    return {
        label: nn.utils.clip_grad_norm_(params, max_norm)
        for label, params in split_by_clip_group(model).items()
    }


def adamw_kwargs(fused: bool, params: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """``{"fused": True}`` when the fused AdamW kernel is usable here, else ``{}``.

    The fused kernel folds the whole parameter update — the two moment
    exponential averages, the bias correction, the decoupled decay and the step
    — into one launch across every parameter, instead of the foreach path's
    handful per group. At ~200 parameter tensors that is the difference between
    a launch-bound and a bandwidth-bound step on a T4.

    Three things disqualify it, and all three are checked rather than assumed:
    it exists only for CUDA (and, in recent torch, XPU) tensors; it requires
    every parameter to be a floating-point tensor on the same device type; and
    it is not present at all in older torch. On any of those the foreach path
    is used, which is what this pipeline has always run.

    **This is the one performance default that is not bit-exact.** The fused
    kernel computes the same update in the same precision but not in the same
    order. Set ``runtime.fused_optimizer=off`` to reproduce an eager run
    exactly.
    """
    if not fused:
        return {}
    if "fused" not in inspect.signature(optim.AdamW.__init__).parameters:
        return {}  # pragma: no cover - torch < 1.13
    if params is not None:
        for group in params:
            for p in group["params"]:
                if p.device.type != "cuda" or not p.is_floating_point():
                    return {}
    return {"fused": True}


def build_optimizer_s1(
    cfg: ExperimentConfig | Any, model: nn.Module, lr: float, fused: bool = False
) -> optim.AdamW:
    """AdamW over the whole model at a single learning rate, with weight-decay grouping."""
    groups = _wd_groups(cfg, model.named_parameters(), lr)
    return optim.AdamW(groups, **adamw_kwargs(fused, groups))


def build_optimizer_s2(
    cfg: ExperimentConfig | Any,
    model: nn.Module,
    head_lr: float,
    back_lr: float,
    fused: bool = False,
) -> optim.AdamW:
    """AdamW with the ArcFace head and the backbone on separate learning rates.

    Group order is head-wd, head-no-wd, backbone-wd, backbone-no-wd — callers
    that read ``param_groups[0]`` / ``[2]`` back for logging depend on it.
    """
    hp, bp = [], []  # type: ignore[var-annotated]
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (hp if n.startswith("arcface_head") else bp).append((n, p))
    groups = _wd_groups(cfg, hp, head_lr) + _wd_groups(cfg, bp, back_lr)
    return optim.AdamW(groups, **adamw_kwargs(fused, groups))


def build_optimizer_s3(
    cfg: ExperimentConfig | Any, model: nn.Module, lr: float, fused: bool = False
) -> optim.AdamW:
    """AdamW over the whole model at a single learning rate; wrapped in :class:`SAM` by the caller."""
    groups = _wd_groups(cfg, model.named_parameters(), lr)
    return optim.AdamW(groups, **adamw_kwargs(fused, groups))
