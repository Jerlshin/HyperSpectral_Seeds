"""Learning-rate and ArcFace-margin schedules.

:func:`phase_aware_lr` is built as a standalone factory (rather than a
closure defined inline in the Stage-1 training loop) so the schedule is
callable in isolation and can be unit-tested across its full epoch range —
see ``tests/unit/test_schedulers.py``.

All three functions return *multipliers* on the optimiser's base LR, not
absolute LRs.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch.optim as optim

if TYPE_CHECKING:  # pragma: no cover - typing only
    from spectralquadnet.config.schema import ExperimentConfig


def sgdr_scheduler(
    optimizer: optim.Optimizer,
    warmup_ep: int = 5,
    T_0: int = 10,
    T_mult: int = 2,
    eta_min_frac: float = 1e-3,
) -> optim.lr_scheduler.LambdaLR:
    """Linear warmup followed by cosine restarts with geometrically growing cycles.

    Stage 2's schedule. With the shipped config (``warmup_ep=3``, ``T_0=25``,
    ``T_mult=2``) the restarts land at epochs 28 and 78, which is what
    ``stage2_arcface.py`` prints as ``↻R1``/``↻R2``.
    """

    def _l(ep: int) -> float:
        if ep < warmup_ep:
            return max(ep / max(warmup_ep, 1), 1e-6)
        t = ep - warmup_ep
        clen = T_0
        elapsed = 0
        while t >= elapsed + clen:
            elapsed += clen
            clen = max(int(clen * T_mult), 1)
        ratio = (t - elapsed) / max(clen, 1)
        return eta_min_frac + 0.5 * (1 - eta_min_frac) * (1 + math.cos(math.pi * ratio))

    return optim.lr_scheduler.LambdaLR(optimizer, _l)


def arcface_margin(ep: int, m0: float, m_target: float, warmup_ep: int) -> float:
    """Cosine ramp of the global ArcFace margin from ``m0`` to ``m_target``.

    Stage 2 calls this per epoch until ``ep >= warmup_ep``, after which the
    margin is pinned at ``m_target`` and ``arc_m=None`` is passed to the head so
    it uses its own per-class adaptive margins instead.
    """
    if ep >= warmup_ep:
        return m_target
    return m0 + (m_target - m0) * 0.5 * (1 - math.cos(math.pi * ep / max(warmup_ep, 1)))


def phase_aware_lr(cfg: ExperimentConfig | Any, p1_end: int, p2_end: int) -> Callable[[int], float]:
    """Build Stage 1's three-phase LR multiplier schedule.

    Phase 1 warms up over 5 epochs then decays 1.0 → 0.6; Phase 2 continues
    0.6 → 0.2; Phase 3 runs 30-epoch cosine restarts between
    ``min_lr/max_lr`` and ``mid_lr/max_lr``.

    Returns the ``LambdaLR`` callable, which takes a **0-based** epoch index —
    the body's first statement converts it to the 1-based ``ep`` the phase
    boundaries are expressed in.
    """

    def _l(ep_idx: int) -> float:
        ep = ep_idx + 1

        # Phase 1: Warmup to Max LR, then slow cosine decay
        if ep <= p1_end:
            # 5-epoch linear warmup
            if ep <= 5:
                return ep / 5.0
            progress = (ep - 5) / max(1, p1_end - 5)
            # Decays from 1.0 (5e-4) down to 0.6 (3e-4)
            return 0.6 + 0.4 * 0.5 * (1.0 + math.cos(math.pi * progress))

        # Phase 2: Standard Cosine Decay
        elif ep <= p2_end:
            progress = (ep - p1_end) / max(1, p2_end - p1_end)
            # Decays from 0.6 (3e-4) down to 0.2 (1e-4)
            return 0.2 + 0.4 * 0.5 * (1.0 + math.cos(math.pi * progress))

        # Phase 3: fine-tuning with cosine restarts every CYCLE epochs. The
        # periodic peak back up to mid_lr lets the model escape the overfit
        # basin that forms when light augmentation alone can't fully prevent
        # memorisation late in training.
        else:
            CYCLE = 30
            cycle_ep = (ep - p2_end - 1) % CYCLE
            min_ratio = cfg.stage1.min_lr / cfg.stage1.max_lr
            mid_ratio = cfg.stage1.mid_lr / cfg.stage1.max_lr
            cosine_val = 0.5 * (1.0 + math.cos(math.pi * cycle_ep / CYCLE))
            return min_ratio + (mid_ratio - min_ratio) * cosine_val

    return _l
