"""Focal, contrastive (SupCon/ProtoNCE), mixup and class-difficulty-weighting losses."""

from spectralquadnet.losses.auxiliary import _aux_loss_weight, _compute_aux_loss
from spectralquadnet.losses.cdws import build_cdws_weights
from spectralquadnet.losses.contrastive import ProtoNCELoss, SupConLoss
from spectralquadnet.losses.focal import FocalLoss
from spectralquadnet.losses.mixup import mixed_aug, mixed_loss

__all__ = [
    "FocalLoss",
    "ProtoNCELoss",
    "SupConLoss",
    "_aux_loss_weight",
    "_compute_aux_loss",
    "build_cdws_weights",
    "mixed_aug",
    "mixed_loss",
]
