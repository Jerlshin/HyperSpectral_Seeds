"""Test-time augmentation: 8 spatial views + 4 spectral rescalings.

The view counts arrive as arguments from ``final_eval.py``, which passes
``cfg.tta_{spatial,spectral}``.

The view set is the 8 dihedral transforms of the patch (4 rotations x optional
horizontal flip) plus spectral gain factors linearly spaced over ``[0.95,
1.05]``, applied about each sample's own per-band **foreground** mean so the
seed's overall brightness moves while its spectral *shape* is preserved. The
``s == 1.0`` view is skipped because it duplicates the identity spatial view —
with ``n_spectral=4`` the scales are 0.95/0.983/1.017/1.05 and none is exactly
1.0, so in the shipped config all 12 views contribute.

Two properties of the spectral view are load-bearing (IMPROVEMENT_PLAN §2.6,
§3.7 — TT-1, TT-3), and both were violated before Tier 1:

* **The mean is over the foreground, and the view is re-masked.** Patches carry
  an exactly-zero background outside the segmented seed, and every masked
  statistic downstream (``masked_spectral_stats``, ``MaskedSpectralECA``,
  ``extract_grid_spectra``) infers that mask by testing for zero. Rescaling
  about the *whole-patch* mean :math:`f\\,m_c` moved the background off zero and
  shifted the foreground mean by a factor of the foreground fraction, so the
  network saw a patch from outside its training manifold. Rescaling about the
  foreground mean and multiplying by the mask again makes the view exactly the
  test-time analogue of the train-time ``multiplicative`` augmentation.
* **The forwards run in fp32.** ``engine/evaluate.py`` forces
  ``autocast(enabled=False)`` so a reported F1 never depends on the autocast
  state it was called from; this module wrapped its forwards in *enabled*
  autocast instead, which confounded the TTA-vs-no-TTA comparison with a
  precision change. Measured effect of the confound: ≤0.0007 macro-F1
  (MIGRATION_PROGRESS 0-B), i.e. hygiene rather than a gain.

Logits are averaged directly (not softmax-averaged); this is the function
behind the ``test_preds_TTA.npy`` artifact.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.amp import autocast

#: A pixel is foreground when its absolute response, summed over bands, exceeds
#: this. The patches are zero-filled outside the segmentation mask, so the
#: threshold only has to clear float noise — it is the same test the masked
#: statistics in ``models/stats_ops.py`` apply.
FOREGROUND_EPS: float = 1e-5


def foreground_mask(x: torch.Tensor) -> torch.Tensor:
    """The ``(B, 1, H, W)`` foreground indicator of a zero-background patch batch.

    Args:
        x: Input batch, shape ``(B, C, H, W)``.

    Returns:
        A float mask, 1.0 on pixels with any spectral response and 0.0 on the
        zero-filled background.
    """
    return (x.abs().sum(1, keepdim=True) > FOREGROUND_EPS).to(x.dtype)


def spectral_view(
    x: torch.Tensor, scale: float | torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Rescale each band's contrast about its foreground mean, then re-mask.

    Implements TT-1: :math:`x^{(s)} = \\big(\\mu + (x - \\mu)s\\big)\\odot m`
    with :math:`\\mu` the per-band mean *over the foreground only*. Both halves
    matter — see the module docstring.

    Args:
        x: Input batch, shape ``(B, C, H, W)``, zero outside the seed.
        scale: Contrast factor ``s``; ``s == 1`` is the identity.
        mask: Optional explicit ``(B, 1, H, W)`` foreground mask. Defaults to
            :func:`foreground_mask`, i.e. inferred from the zero background.

    Returns:
        The rescaled view, with the background still exactly zero and the
        foreground support unchanged.
    """
    m = foreground_mask(x) if mask is None else mask.to(x.dtype)
    n = m.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
    mu = (x * m).sum(dim=(-2, -1), keepdim=True) / n
    return (mu + (x - mu) * scale) * m


@torch.no_grad()
def tta_predict(
    model: nn.Module,
    x: torch.Tensor,
    n_spatial: int = 8,
    n_spectral: int = 4,
    mask: torch.Tensor | None = None,
    morph: torch.Tensor | None = None,
) -> torch.Tensor:
    """Average logits over dihedral spatial views and spectral gain rescalings.

    Every forward runs under ``autocast(enabled=False)``, matching
    ``engine/evaluate.py``, so the TTA and no-TTA numbers differ only by the
    ensembling (TT-3).

    Args:
        model: Model to predict with; called in eval mode by the caller.
        x: Input batch, shape ``(B, C, H, W)``.
        n_spatial: Number of spatial views to use, out of the 8 dihedral
            transforms, taken in a fixed rotation/flip order.
        n_spectral: Number of spectral gain views, linearly spaced over
            ``[0.95, 1.05]`` (the ``s == 1.0`` view, if it lands exactly, is
            skipped as a duplicate of the identity spatial view).
        mask: Optional explicit foreground mask, shape ``(B, 1, H, W)`` or
            ``(B, H, W)``, passed through to :func:`spectral_view` **and** to
            the model (FE-2 / T3-7). Each spatial view rotates and flips the
            mask with the patch: handing the model an untransformed mask would
            claim the seed is wherever it used to be, which is a worse defect
            than the threshold the mask replaces. Inferred from the zero
            background when omitted.
        morph: Optional ``(B, 8)`` morphometrics. Invariant to every view here —
            a dihedral transform does not change a seed's area — so it is passed
            through unchanged, which is itself the check that the transform set
            is label-preserving.

    Returns:
        Logits averaged over all ``n_spatial + n_spectral`` (minus any
        skipped duplicate) views, shape ``(B, num_classes)``.
    """
    device = x.device
    logits = []
    spatial_views = [(k, f) for k in range(4) for f in (False, True)][:n_spatial]
    m4 = mask if mask is None or mask.dim() == 4 else mask.unsqueeze(1)
    extra = {} if morph is None else {"morph": morph}

    def _predict(patch: torch.Tensor, view_mask: torch.Tensor | None) -> torch.Tensor:
        side = dict(extra)
        if view_mask is not None:
            side["mask"] = view_mask
        out = model(patch, **side)
        return out["main"] if isinstance(out, dict) else out  # type: ignore[no-any-return]

    with autocast(device_type=device.type, enabled=False):
        for k, flip in spatial_views:
            aug = torch.rot90(x, k, [2, 3])
            aug_m = None if m4 is None else torch.rot90(m4, k, [2, 3])
            if flip:
                aug = torch.flip(aug, [3])
                aug_m = None if aug_m is None else torch.flip(aug_m, [3])
            logits.append(_predict(aug, aug_m))

        scales = torch.linspace(0.95, 1.05, n_spectral, device=device)
        for s in scales:
            if s == 1.0:
                continue
            # The spectral view is a contrast rescaling about the foreground
            # mean; the support is untouched, so the mask is the input's.
            logits.append(_predict(spectral_view(x, s, m4), m4))

    return torch.stack(logits).mean(0)
