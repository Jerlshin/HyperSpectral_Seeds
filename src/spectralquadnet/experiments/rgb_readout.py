"""Frozen-backbone RGB readouts beyond the class token (S31).

The S20 crops are masked: every pixel outside the kernel is exactly zero. A patch's
foreground weight is the fraction of its pixels with any nonzero channel, so pooling
never averages plate background into a kernel descriptor. Views are the label-free
orientation group that keeps the crop square and centred (identity, horizontal flip,
vertical flip, 180-degree rotation). No row partition or label is read here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F

Array = npt.NDArray[Any]
VIEWS = ("id", "hflip", "vflip", "rot180")
PATCH = 14


def view(images: torch.Tensor, name: str) -> torch.Tensor:
    """Apply an orientation view to N x H x W x C (uint8) or N x C x H x W tensors."""
    dims = (1, 2) if images.shape[-1] == 3 else (2, 3)
    if name == "id":
        return images
    if name == "hflip":
        return images.flip(dims[1])
    if name == "vflip":
        return images.flip(dims[0])
    if name == "rot180":
        return images.flip(dims)
    raise ValueError(f"Unknown view: {name}")


def foreground_weights(images: torch.Tensor, patch: int = PATCH) -> torch.Tensor:
    """Per-patch foreground fraction (N x tokens) of masked uint8 N x H x W x C crops."""
    fg = (images.amax(-1) > 0).float()[:, None]
    return F.avg_pool2d(fg, patch).flatten(1)


def pooled_readout(
    patch_tokens: torch.Tensor, class_token: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Concatenate the class token with the foreground-weighted mean patch token."""
    w = weights / weights.sum(1, keepdim=True).clamp_min(1e-6)
    return torch.cat([class_token, torch.einsum("nt,ntd->nd", w, patch_tokens)], dim=1)


def layer_readouts(
    model: torch.nn.Module, x: torch.Tensor, weights: torch.Tensor, layers: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """(N x layers x 2D [class, fg-mean] of the last ``layers`` blocks, final patch tokens).

    DINOv2's final norm is applied to every layer, as in its own linear evaluation.
    """
    outputs = model.get_intermediate_layers(x, n=layers, return_class_token=True, norm=True)  # type: ignore[operator]
    readout = torch.stack([pooled_readout(p, c, weights) for p, c in outputs], dim=1)
    return readout, outputs[-1][0]


def standardise(train: Array, *others: Array) -> list[Array]:
    """Scale by outer-training statistics only."""
    mean, scale = train.mean(0), np.maximum(train.std(0), 1e-6)
    return [(a - mean) / scale for a in (train, *others)]
