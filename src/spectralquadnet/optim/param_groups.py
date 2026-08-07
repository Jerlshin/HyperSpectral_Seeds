"""Weight-decay parameter grouping and the three per-stage optimiser builders.

The two grouping rules are load-bearing:

* **No weight decay on 1-D tensors or biases** — norms, ECA/CBAM gates and every
  ``.bias`` land in the ``no_wd`` group.
* **Stage 2 splits on the ``arcface_head`` name prefix**, giving the head
  ``cfg.stage2.head_lr`` and the backbone ``cfg.stage2.back_lr``. The resulting
  group order — head-wd, head-no-wd, backbone-wd, backbone-no-wd — is what
  ``stage2_arcface.py`` reads back as ``param_groups[0]`` and ``[2]`` when it
  logs the two learning rates, so the concatenation order must not change.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.optim as optim

if TYPE_CHECKING:  # pragma: no cover - typing only
    from spectralquadnet.config.schema import ExperimentConfig


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


def build_optimizer_s1(cfg: ExperimentConfig | Any, model: nn.Module, lr: float) -> optim.AdamW:
    """AdamW over the whole model at a single learning rate, with weight-decay grouping."""
    return optim.AdamW(_wd_groups(cfg, model.named_parameters(), lr))


def build_optimizer_s2(
    cfg: ExperimentConfig | Any, model: nn.Module, head_lr: float, back_lr: float
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
    return optim.AdamW(_wd_groups(cfg, hp, head_lr) + _wd_groups(cfg, bp, back_lr))


def build_optimizer_s3(cfg: ExperimentConfig | Any, model: nn.Module, lr: float) -> optim.AdamW:
    """AdamW over the whole model at a single learning rate; wrapped in :class:`SAM` by the caller."""
    return optim.AdamW(_wd_groups(cfg, model.named_parameters(), lr))
