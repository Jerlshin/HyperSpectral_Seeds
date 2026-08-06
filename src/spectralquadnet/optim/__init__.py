"""SAM optimizer, weight-decay parameter groups and LR/margin schedulers."""

from spectralquadnet.optim.param_groups import (
    build_optimizer_s1,
    build_optimizer_s2,
    build_optimizer_s3,
)
from spectralquadnet.optim.sam import SAM
from spectralquadnet.optim.schedulers import arcface_margin, phase_aware_lr, sgdr_scheduler

__all__ = [
    "SAM",
    "arcface_margin",
    "build_optimizer_s1",
    "build_optimizer_s2",
    "build_optimizer_s3",
    "phase_aware_lr",
    "sgdr_scheduler",
]
