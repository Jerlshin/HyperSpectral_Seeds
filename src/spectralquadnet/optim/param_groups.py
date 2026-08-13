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

**IC-6 — the threshold, and measuring whether it binds.** At ``grad_clip = 1.0``
the backbone's pre-clip norm was 25–50 for the entire audited run, so the clip
fired on essentially every step and the backbone was doing normalised-gradient
descent at a fixed step size rather than AdamW-with-a-schedule (CHANGES.md
§8.1). The elaborate three-regime LR schedule was, in magnitude terms, largely
decorative. The default moves to 5.0 — clip *outliers*, which is a clip's job —
and :class:`ClipReport` adds the measurement that was missing: the post-clip
norm and the fraction of groups actually clipped, logged as
``grad_norm/postclip_*`` and ``grad_norm/clip_fraction``. The validation
criterion for the change is that ``clip_fraction`` falls from ≈1.0 to <0.2.
"""

from __future__ import annotations

import inspect
import weakref
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class ClipReport(Mapping[str, torch.Tensor]):
    """What one per-group clip did, as on-device 0-dim tensors (IC-6).

    Iterating and indexing address the **pre-clip** norms, so the callers and
    tests written against the plain ``dict[str, Tensor]`` this replaces are
    unaffected. :attr:`postclip` and :attr:`clipped` are the addition.

    Everything stays on the device deliberately: this runs on the inner loop and
    a ``.item()`` here would be a queue drain per step. The training loops
    accumulate these and resolve one epoch's worth in a single transfer.
    """

    #: Pre-clip L2 norm per group — what ``clip_grad_norm_`` returns.
    preclip: dict[str, torch.Tensor] = field(default_factory=dict)
    #: ``min(preclip, max_norm)`` — the norm the optimiser actually stepped on.
    postclip: dict[str, torch.Tensor] = field(default_factory=dict)
    #: 1.0 where the clip bound, 0.0 where it did not.
    clipped: dict[str, torch.Tensor] = field(default_factory=dict)

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.preclip[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.preclip)

    def __len__(self) -> int:
        return len(self.preclip)

    def scalars(self) -> dict[str, torch.Tensor]:
        """Tracker tags for all three series plus the aggregate clip fraction.

        ``grad_norm/clip_fraction`` is the mean of :attr:`clipped` over the
        groups — the single number CHANGES.md §8.1 asks to watch, since "the
        clip binds on every step" is a statement about the model, not about any
        one group.
        """
        tags: dict[str, torch.Tensor] = {}
        for label, value in self.preclip.items():
            tags[f"grad_norm/preclip_{label}"] = value
        for label, value in self.postclip.items():
            tags[f"grad_norm/postclip_{label}"] = value
        for label, value in self.clipped.items():
            tags[f"grad_norm/clipped_{label}"] = value
        if self.clipped:
            tags["grad_norm/clip_fraction"] = torch.stack(list(self.clipped.values())).mean()
        return tags


def clip_grad_norm_by_group(model: nn.Module, max_norm: float) -> ClipReport:
    """Clip head, fusion and backbone gradients independently (OP-3 / T2-5).

    Call it exactly where the single global ``clip_grad_norm_`` used to be:
    after ``backward()`` (and after ``scaler.unscale_``), before the step.

    Args:
        model: The live model, with gradients populated.
        max_norm: Per-group maximum L2 norm — the same ``cfg.grad_clip`` the
            global clip used, now applied three times rather than once.

    Returns:
        A :class:`ClipReport` over ``head``/``fusion``/``backbone``. Indexing it
        gives each group's **pre-clip** norm, as before; ``.postclip`` and
        ``.clipped`` carry IC-6's additions. Groups owning no parameter are
        absent from all three.
    """
    preclip: dict[str, torch.Tensor] = {}
    postclip: dict[str, torch.Tensor] = {}
    clipped: dict[str, torch.Tensor] = {}
    for label, params in split_by_clip_group(model).items():
        norm = nn.utils.clip_grad_norm_(params, max_norm)
        preclip[label] = norm
        # Derived from the returned pre-clip norm rather than re-measured: a
        # second pass over the gradients would double the cost of the clip to
        # compute a number `min` already determines exactly.
        postclip[label] = norm.clamp(max=max_norm)
        clipped[label] = (norm > max_norm).to(norm.dtype)
    return ClipReport(preclip=preclip, postclip=postclip, clipped=clipped)


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
