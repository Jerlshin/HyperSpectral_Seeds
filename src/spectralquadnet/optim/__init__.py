"""SAM optimizer, weight-decay parameter groups and LR/margin schedulers."""

from spectralquadnet.optim.param_groups import (
    build_optimizer_s1,
    build_optimizer_s2,
    build_optimizer_s3,
    clip_grad_norm_by_group,
)
from spectralquadnet.optim.sam import SAM
from spectralquadnet.optim.schedulers import (
    arcface_margin,
    phase_aware_lr,
    sgdr_scheduler,
    stage3_margin_kappa,
    subcentre_tau,
)

__all__ = [
    "SAM",
    "arcface_margin",
    "build_optimizer_s1",
    "build_optimizer_s2",
    "build_optimizer_s3",
    "clip_grad_norm_by_group",
    "phase_aware_lr",
    "sgdr_scheduler",
    "stage3_margin_kappa",
    "subcentre_tau",
]
