"""Test-time augmentation: 8 spatial views + 4 spectral rescalings.

The view counts arrive as arguments from ``final_eval.py``, which passes
``cfg.tta_{spatial,spectral}``.

The view set is the 8 dihedral transforms of the patch (4 rotations x optional
horizontal flip) plus spectral gain factors linearly spaced over ``[0.95,
1.05]``, applied about each sample's own per-band spatial mean so the seed's
overall brightness moves while its spectral *shape* is preserved. The ``s ==
1.0`` view is skipped because it duplicates the identity spatial view — with
``n_spectral=4`` the scales are 0.95/0.983/1.017/1.05 and none is exactly 1.0,
so in the shipped config all 12 views contribute.

Logits are averaged directly (not softmax-averaged); this is the function
behind the ``test_preds_TTA.npy`` artifact.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.amp import autocast


@torch.no_grad()
def tta_predict(
    model: nn.Module,
    x: torch.Tensor,
    n_spatial: int = 8,
    n_spectral: int = 4,
) -> torch.Tensor:
    """Average logits over dihedral spatial views and spectral gain rescalings.

    Args:
        model: Model to predict with; called in eval mode by the caller.
        x: Input batch, shape ``(B, C, H, W)``.
        n_spatial: Number of spatial views to use, out of the 8 dihedral
            transforms, taken in a fixed rotation/flip order.
        n_spectral: Number of spectral gain views, linearly spaced over
            ``[0.95, 1.05]`` (the ``s == 1.0`` view, if it lands exactly, is
            skipped as a duplicate of the identity spatial view).

    Returns:
        Logits averaged over all ``n_spatial + n_spectral`` (minus any
        skipped duplicate) views, shape ``(B, num_classes)``.
    """
    device = x.device
    logits = []
    spatial_views = [(k, f) for k in range(4) for f in (False, True)][:n_spatial]

    for k, flip in spatial_views:
        aug = torch.rot90(x, k, [2, 3])
        if flip:
            aug = torch.flip(aug, [3])
        with autocast(device_type=device.type):
            out = model(aug)
            logits.append(out["main"] if isinstance(out, dict) else out)

    scales = torch.linspace(0.95, 1.05, n_spectral, device=device)
    for s in scales:
        if s == 1.0:
            continue
        mean = x.mean(dim=[2, 3], keepdim=True)
        aug_sp = mean + (x - mean) * s
        with autocast(device_type=device.type):
            out = model(aug_sp)
            logits.append(out["main"] if isinstance(out, dict) else out)

    return torch.stack(logits).mean(0)
