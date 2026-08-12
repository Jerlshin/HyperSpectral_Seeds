"""Branch C — a joint spectral–spatial stem over the full cube (BR-3 / T3-2).

C-3, in one line: the network contained **no joint spectral–spatial operator**.
``band_reduce`` was ``Conv2d(C, C, 1, groups=C)`` followed by ``Conv2d(C, 64, 1)``
— two 1×1 convolutions, so the band axis was collapsed to 64 channels *before*
any spatial kernel ran, and the very first 3×3 saw a spectrally-mixed feature
map. The function "this absorption feature, in this part of the seed" was not in
the hypothesis class of any module in the model.

BR-3 replaces the two 1×1s with a factorised 3-D stem that keeps the spectral
axis alive for three stages before folding it::

    x  (B, 1, 40, 64, 64)
     ├─ Conv3d(1→16,  k=(7,3,3), s=(2,1,1))  → (B,16,20,64,64)
     ├─ Conv3d(16→32, k=(5,3,3), s=(2,2,2))  → (B,32,10,32,32)
     ├─ Conv3d(32→64, k=(5,3,3), s=(2,2,2))  → (B,64, 5,16,16)
     └─ reshape (B, 320, 16, 16) → 1×1 → (B,192,16,16)
          └─ the existing ResBlock2D / CBAM tail

The spectral axis is *folded* into the channel axis at the end, not deleted at
the start: each of the 320 channels entering the 1×1 is a (spectral position ×
learned feature) pair, so the tail's 3×3 kernels operate on features that still
carry where in the spectrum they came from.

The stem costs ≈ 116 k parameters, comfortably inside the ≈ 600 k BR-1 freed
from Branch B, and the branch as a whole moves from 1.69 M to ≈ 2.23 M. That is
the reallocation §3.8 is about: Branch C is the only branch whose input is not
reconstructible from any other, and it was the one starved of capacity.

The persisted mask (FE-2 / T3-7) is applied after every stage, so a padded
region stays exactly zero however deep the stack goes and the CNN can never
learn the frame instead of the seed.

The stem is also the model's single most expensive module on the Metal backend,
for a reason that is a kernel-quality fact rather than an arithmetic one — see
:func:`conv3d_as_conv2d`, which is the same operator routed through
``Conv2d`` and is switched on per device by
:func:`~spectralquadnet.utils.device.apply_runtime_optimisations`.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectralquadnet.models.blocks.attention import CBAM
from spectralquadnet.models.blocks.conv_blocks import ResBlock2D
from spectralquadnet.models.stats_ops import foreground_mask

#: Floor under the signed square root in :meth:`SpectralSpatialCNN._pn`.
#:
#: ``d/dx sqrt(x)`` is ``1 / (2 sqrt(x))``, which is **unbounded at zero** — so
#: this constant is not cosmetic, it is the only thing standing between a dead
#: pooled channel (``amax`` of an all-negative feature map, exactly 0 after the
#: activation) and an infinite gradient. At 1e-8 the bound is 5,000.
#:
#: :func:`_pn_floor` is what makes it survive autocast. Applied at fp16 width
#: the literal rounds to **zero** — 1e-8 is below fp16's smallest subnormal,
#: 5.96e-8 — so the clamp becomes a no-op, ``sqrt(0)`` backpropagates ``inf``,
#: and the loss is ``NaN`` from a guard that looks present in the source. bf16
#: carries fp32's exponent range and represents it exactly.
PN_EPS: float = 1e-8


def _pn_floor(dtype: torch.dtype) -> float:
    """:data:`PN_EPS`, raised to the smallest value ``dtype`` can actually hold.

    fp32 and bf16 both represent 1e-8, so this returns it unchanged and the
    arithmetic is bit-identical to what it always was. fp16 cannot, so it gets
    its own smallest normal (6.1e-5) instead — a weaker floor, but a real one,
    where the literal would have silently been no floor at all.
    """
    return max(PN_EPS, torch.finfo(dtype).smallest_normal) if dtype.is_floating_point else PN_EPS


def conv3d_as_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """The same 3-D convolution, evaluated as ``kd`` 2-D convolutions.

    A ``Conv3d`` *is*, by definition, a sum over its depth taps::

        out[b, o, d, y, x] = Σ_i  Conv2d( x[:, :, d·s_d − p_d + i],  W[:, :, i] )

    so gathering the strided depth slice belonging to tap ``i``, folding it into
    the batch axis and running one ``Conv2d`` per tap reproduces the operator
    exactly: same weights, same multiply-accumulates, same result up to the
    order in which the ``kd`` partial sums are added (measured at 1.9e-07
    relative on the full model's logits, the same order as the ``fused``
    AdamW/eager difference this pipeline already accepts).

    Why bother reordering an operator into itself: it changes **which kernels
    run**, and on Metal that is worth a factor of three. ``Conv3d``'s backward
    is a poor kernel there — on this stem's shapes it takes 9.4× its own
    forward, where the branch's own 2-D tail runs a healthy 2.2×, and a
    ``Conv2d`` doing strictly *more* multiply-accumulates than the stage-2
    ``Conv3d`` finishes its backward in 152 ms against the ``Conv3d``'s 358.
    Over the three stages at batch 32 the decomposition is **3.12× faster**,
    which is **2.12×** on the whole training step (2103 → 994 ms) because the
    stem was that much of it.

    The trade is activation memory: each tap's gathered slice is a real tensor,
    so the stem holds ``kd`` depth-slices of its input where the fused kernel
    held one (+370 MB at batch 32, against the 2.2 GB the Branch-A
    checkpointing gives back). That, and the fact that cuDNN's 3-D kernels have
    no such defect, is why this is a per-device decision rather than the only
    path — see :func:`~spectralquadnet.utils.device.resolve_runtime`.
    """
    n_batch, in_ch, depth, height, width = x.shape
    k_depth = weight.shape[2]
    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding

    depth_out = (depth + 2 * pad_d - k_depth) // stride_d + 1
    # Only the depth axis is padded here; the 2-D calls do their own H/W padding.
    padded = F.pad(x, (0, 0, 0, 0, pad_d, pad_d))

    acc: torch.Tensor | None = None
    for tap in range(k_depth):
        # The output positions this tap contributes to, gathered in one strided
        # view, then folded into the batch so a single Conv2d covers all of them.
        last = tap + stride_d * (depth_out - 1) + 1
        sl = padded[:, :, tap:last:stride_d]
        sl = sl.permute(0, 2, 1, 3, 4).reshape(n_batch * depth_out, in_ch, height, width)
        out = F.conv2d(sl, weight[:, :, tap], stride=(stride_h, stride_w), padding=(pad_h, pad_w))
        acc = out if acc is None else acc + out

    assert acc is not None, "a convolution kernel cannot have zero depth taps"
    out_ch, h_out, w_out = acc.shape[1], acc.shape[2], acc.shape[3]
    folded = acc.reshape(n_batch, depth_out, out_ch, h_out, w_out).permute(0, 2, 1, 3, 4)
    # Added once, after the sum -- passing it to each Conv2d would apply it kd times.
    return folded if bias is None else folded + bias.view(1, -1, 1, 1, 1)


class SpectralSpatialStem3D(nn.Module):
    """Three strided 3-D convolutions, then a 1×1 fold of the spectral axis.

    Only the first stage leaves the spatial resolution alone: the spectral axis
    is halved first, so the expensive 3-D kernels run on a shrinking cube while
    the 64×64 spatial detail is still intact when the widest spectral kernel
    (``k = 7`` bands) passes over it.
    """

    #: Route the three ``Conv3d`` stages through :func:`conv3d_as_conv2d`.
    #: A pure execution choice — it holds no state, changes no parameter and is
    #: not in the state dict, so a checkpoint written with it on loads with it
    #: off. Left ``False`` here so constructing a stem directly (which every
    #: unit test and ``scripts/capture_golden.py`` does) is unconditionally the
    #: reference path; :func:`~spectralquadnet.utils.device.apply_runtime_optimisations`
    #: is the one place that turns it on, once, from the resolved runtime plan.
    decompose_conv3d: bool = False

    def __init__(self, num_bands: int = 40, out_channels: int = 192) -> None:
        super().__init__()
        self.num_bands = int(num_bands)

        self.stage1 = nn.Sequential(
            nn.Conv3d(1, 16, (7, 3, 3), stride=(2, 1, 1), padding=(3, 1, 1), bias=False),
            nn.GroupNorm(4, 16),
            nn.GELU(),
        )
        self.stage2 = nn.Sequential(
            nn.Conv3d(16, 32, (5, 3, 3), stride=(2, 2, 2), padding=(2, 1, 1), bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        self.stage3 = nn.Sequential(
            nn.Conv3d(32, 64, (5, 3, 3), stride=(2, 2, 2), padding=(2, 1, 1), bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )

        # Spectral depth after three stride-2 spectral steps, e.g. 40 → 20 → 10 → 5.
        depth = self.num_bands
        for _ in range(3):
            depth = (depth + 1) // 2
        self.folded_depth = depth

        self.fold = nn.Sequential(
            nn.Conv2d(64 * depth, out_channels, 1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )

    @staticmethod
    def _apply_mask(h: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Re-zero the padded region at this stage's spatial resolution.

        ``mask`` is ``(B, 1, H, W)`` at input resolution; it is area-pooled to
        whatever ``h``'s spatial size now is, which is the same operation the
        stride performed on the signal.
        """
        if mask is None:
            return h
        m = F.adaptive_avg_pool2d(mask, (int(h.shape[-2]), int(h.shape[-1])))
        return h * m.unsqueeze(2)

    def _stage(self, stage: nn.Sequential, h: torch.Tensor) -> torch.Tensor:
        """One ``Conv3d → GroupNorm → GELU`` stage, through whichever conv path is selected.

        The decomposition replaces the *convolution* only; the normalisation and
        activation are the same module objects either way, which is what keeps
        the two paths a kernel choice rather than two architectures.

        Applied in ``eval()`` as well as in training, deliberately: a stem that
        decomposed only under autograd would make a model's own validation
        logits disagree with its training ones at 1e-7, and chasing that later
        would cost more than the forward pass it saves.
        """
        if not self.decompose_conv3d:
            return stage(h)  # type: ignore[no-any-return]  # `nn.Module.__call__` -> Any
        conv = cast(nn.Conv3d, stage[0])
        # `padding` is `str | tuple` on `nn.Conv3d`; the stem only ever builds
        # the numeric form, and `"same"` has no meaning at stride 2 anyway.
        pad = cast("tuple[int, ...]", conv.padding)
        h = conv3d_as_conv2d(h, conv.weight, conv.stride, pad, conv.bias)
        return stage[2](stage[1](h))  # type: ignore[no-any-return]  # `nn.Module.__call__` -> Any

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """``(B, C, H, W) -> (B, out_channels, H // 4, W // 4)``."""
        h = self._apply_mask(self._stage(self.stage1, x.unsqueeze(1)), mask)
        h = self._apply_mask(self._stage(self.stage2, h), mask)
        h = self._apply_mask(self._stage(self.stage3, h), mask)
        n_batch = h.shape[0]
        folded = h.reshape(n_batch, -1, h.shape[-2], h.shape[-1])
        return self.fold(folded)  # type: ignore[no-any-return]  # `nn.Module.__call__` -> Any


class SpatialCNNBranch(nn.Module):
    """3-D spectral–spatial stem into a CBAM-gated ResNet-2D stack over spatial texture.

    Pools the fused feature map with concatenated signed-power-normalised mean
    and max statistics, which is more stable than raw mean/max pooling under
    the heavy-tailed activations a ResNet stack produces.
    """

    def __init__(self, num_bands: int = 256, out_dim: int = 256, stem_channels: int = 192) -> None:
        super().__init__()

        self.stem = SpectralSpatialStem3D(num_bands, stem_channels)

        self.stages = nn.Sequential(
            ResBlock2D(stem_channels, 128, 2),
            CBAM(128),
            ResBlock2D(128, 192, 2),
            CBAM(192),
            ResBlock2D(192, 256, 2),
            CBAM(256),
            ResBlock2D(256, out_dim, 2),
        )
        self.proj = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim), nn.BatchNorm1d(out_dim), nn.GELU()
        )

    @staticmethod
    def _pn(x: torch.Tensor) -> torch.Tensor:
        return x.sign() * x.abs().clamp(_pn_floor(x.dtype)).sqrt()

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.stages(self.stem(x, foreground_mask(x, mask)))
        return self.proj(  # type: ignore[no-any-return]  # `nn.Module.__call__` -> Any
            F.normalize(
                torch.cat([self._pn(h.mean([2, 3])), self._pn(h.amax([2, 3]))], 1), dim=1, eps=1e-4
            )
        )
