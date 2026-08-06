"""Weight-decay parameter grouping and the three per-stage optimiser builders.

Relocated from ``HSI_modality_training/hsi_training.py`` @ ``886560f``:

=====================================  ==============
Symbol                                 Baseline lines
=====================================  ==============
:func:`_wd_groups`                     1750-1759
:func:`build_optimizer_s1`             1762-1763
:func:`build_optimizer_s2`             1766-1772
:func:`build_optimizer_s3`             1775-1776
=====================================  ==============

Declared deviation, mechanical: ``CONFIG["weight_decay"]`` → ``cfg.weight_decay``,
which makes ``cfg`` the leading parameter of all four functions (the convention
Phase 2 established in ``data/loaders.py``).

The two grouping rules are load-bearing and unchanged:

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
    return optim.AdamW(_wd_groups(cfg, model.named_parameters(), lr))


def build_optimizer_s2(
    cfg: ExperimentConfig | Any, model: nn.Module, head_lr: float, back_lr: float
) -> optim.AdamW:
    hp, bp = [], []  # type: ignore[var-annotated]
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (hp if n.startswith("arcface_head") else bp).append((n, p))
    return optim.AdamW(_wd_groups(cfg, hp, head_lr) + _wd_groups(cfg, bp, back_lr))


def build_optimizer_s3(cfg: ExperimentConfig | Any, model: nn.Module, lr: float) -> optim.AdamW:
    return optim.AdamW(_wd_groups(cfg, model.named_parameters(), lr))
