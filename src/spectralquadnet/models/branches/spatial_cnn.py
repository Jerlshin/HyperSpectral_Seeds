"""Branch C — a joint spectral–spatial stem over the full cube (BR-3 / T3-2).

C-3, in one line: the network contained **no joint spectral–spatial operator**.
``band_reduce`` was ``Conv2d(C, C, 1, groups=C)`` followed by ``Conv2d(C, 64, 1)``
— two 1×1 convolutions, so the band axis was collapsed to 64 channels *before*
any spatial kernel ran, and the very first 3×3 saw a spectrally-mixed feature
map. The function "this absorption feature, in this part of the seed" was not in
the hypothesis class of any module in the model.

BR-3 replaces the two 1×1s with a factorised 3-D stem that keeps the spectral
axis alive for three stages before folding it. On the primary 256-band input::

    x  (B, 1, 256, 64, 64)
     ├─ Conv3d(1→16,  k=(15,3,3), s=(8,1,1))  → (B,16,32,64,64)
     ├─ Conv3d(16→32, k=( 5,3,3), s=(2,2,2))  → (B,32,16,32,32)
     ├─ Conv3d(32→64, k=( 5,3,3), s=(2,2,2))  → (B,64, 8,16,16)
     └─ reshape (B, 512, 16, 16) → 1×1 → (B,192,16,16)
          └─ the existing ResBlock2D / CBAM tail

The spectral axis is *folded* into the channel axis at the end, not deleted at
the start: each of the 512 channels entering the 1×1 is a (spectral position ×
learned feature) pair, so the tail's 3×3 kernels operate on features that still
carry where in the spectrum they came from.

Why the spectral strides are derived rather than fixed
─────────────────────────────────────────────────────
The stem's cost is dominated by its **first** stage, which runs at the full
64×64 spatial resolution, and by the fold, whose input width is
``64 × folded_depth``. A stem with three hardcoded stride-2 spectral steps —
which is what this module used to be — folds a 256-band cube at depth 32, so
the fold alone becomes a ``Conv2d(2048 → 192, 1)`` and stage 2 runs on a cube
6.4× deeper than the design was measured on. That is a reduced-band stem
carrying a full cube, not a full-cube stem.

:func:`spectral_stride_schedule` instead derives the three spectral strides from
the band count so that the folded depth lands at or below
:data:`DEFAULT_FOLDED_DEPTH`, spending the extra reduction in stage 1 where one
input channel and 16 output channels make it cheapest. At 256 bands that is
``(8, 2, 2)``; at 40 it is ``(2, 2, 2)``, i.e. **exactly** the audited schedule,
which is what keeps the retained band-selection arms and the golden regression
gates comparing like with like.

The kernel depth widens with the stride (:func:`kernel_depth`) so that **every
band reaches at least one tap**. That is a stronger requirement than ``k >= s``
and it is the one that bites: at ``s = 8`` a 9-tap kernel with symmetric padding
covers input indices only up to ``C - s + 4``, so bands 253, 254 and 255 of the
acquired cube would never be multiplied by anything — silently, since every
shape still agrees. ``k = 2s - 1`` is the smallest width that closes it, and
``tests/unit/test_branch_c_stem.py`` asserts the coverage band by band rather
than trusting the arithmetic.

The stem costs ≈ 216 k parameters at 256 bands (≈ 178 k at 40), comfortably
inside the ≈ 600 k BR-1 freed from Branch B, and the branch as a whole is
≈ 2.27 M. That is the reallocation §3.8 is about: Branch C is the only branch
whose input is not reconstructible from any other, and it was the one starved of
capacity.

The persisted mask (FE-2 / T3-7) is applied after every stage, so a padded
region stays exactly zero however deep the stack goes and the CNN can never
learn the frame instead of the seed.

The stem is also the model's single most expensive module on the Metal backend,
for a reason that is a kernel-quality fact rather than an arithmetic one — see
:func:`conv3d_as_conv2d`, which is the same operator routed through
``Conv2d`` and is switched on per device by
:func:`~spectralquadnet.utils.device.apply_runtime_optimisations`. Branch C is
**59 % of the forward pass** at batch 128 on an M5 (1363 ms of 2323 ms), with
Branch A second at 29 %.

Recomputing the stem does not pay, and has been measured
──────────────────────────────────────────────────────
Branch A gives back 2.13× its activation memory for 4.8 % of step time
(``checkpoint_branch_a``), so the same treatment for the stem is the obvious
next move. It is a loss, at every batch size tried — full training step on an
M5, plain vs ``torch.utils.checkpoint`` around the stem:

======  =====================  =====================
batch   plain                  recomputed
======  =====================  =====================
32      32.3 ms/sample          39.7 ms/sample
64      34.3 ms/sample          47.5 ms/sample
128     90.4 ms/sample         173.8 ms/sample
======  =====================  =====================

and the allocator's high-water mark went **up** in each case rather than down
(4041 → 4268 MB at batch 32). The reason is :func:`conv3d_as_conv2d`: the
decomposition's ``kd`` gathered depth-slices are transient within the forward,
but a recompute rebuilds all of them *during* the backward, while the rest of
the backward's buffers are live — so the peak the checkpoint was meant to lower
is the one it raises. The gradients are bit-identical either way (the stem is
Conv3d/GroupNorm/GELU only), so this is purely a cost question, and the answer
is no. Do not re-add it without re-measuring against the fused path.
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


#: Spectral depth the stem folds into the channel axis, and therefore the
#: quantity :func:`spectral_stride_schedule` solves for. The fold is a
#: ``Conv2d(64 * folded_depth → out_channels, 1)``, so this — not the band count
#: — sets the stem's terminal width; 8 keeps it at 98 k parameters on a 256-band
#: cube, against 393 k for the un-derived three-halvings schedule.
DEFAULT_FOLDED_DEPTH: int = 8

#: Base spectral kernel depth of the three stages, before :func:`kernel_depth`
#: widens any of them to cover its stride. ``(7, 5, 5)`` is the audited stem.
BASE_KERNEL_DEPTHS: tuple[int, int, int] = (7, 5, 5)

#: Spatial strides of the three stages. Fixed, and deliberately not derived:
#: they take a 64×64 patch to 16×16, which is the resolution the 2-D tail and
#: every shape assertion downstream are written against. Only the *spectral*
#: stride is a function of the band count.
SPATIAL_STRIDES: tuple[int, int, int] = (1, 2, 2)


def spectral_stride_schedule(
    num_bands: int, folded_depth: int = DEFAULT_FOLDED_DEPTH
) -> tuple[int, int, int]:
    """Three spectral strides that bring ``num_bands`` to at most ``folded_depth``.

    The total reduction is the smallest power of two with
    ``ceil(num_bands / total) <= folded_depth``, distributed so that stages 2
    and 3 keep the audited halving and **stage 1 absorbs the remainder**. Stage 1
    is the cheapest place to spend it: it runs at full spatial resolution but
    with one input channel and sixteen output channels, so a band read there
    costs a fraction of what it costs after stage 2 has widened the cube.

    The schedule is therefore a smooth function of the band count rather than a
    special case for 256:

    ==========  =======  ==============  =============
    num_bands   total    strides         folded depth
    ==========  =======  ==============  =============
    8           1        ``(1, 1, 1)``   8
    16          2        ``(2, 1, 1)``   8
    40          8        ``(2, 2, 2)``   5
    100         16       ``(4, 2, 2)``   7
    256         32       ``(8, 2, 2)``   8
    ==========  =======  ==============  =============

    The 40-band row is the audited stem, tensor for tensor — which is what lets
    the retained band-selection arms and the golden gates stay comparable.

    Args:
        num_bands: Bands entering the stem.
        folded_depth: Upper bound on the spectral depth reaching the fold.

    Raises:
        ValueError: ``num_bands`` or ``folded_depth`` is not positive.
    """
    if num_bands < 1:
        raise ValueError(f"num_bands must be >= 1, got {num_bands}")
    if folded_depth < 1:
        raise ValueError(f"folded_depth must be >= 1, got {folded_depth}")

    total = 1
    while -(-num_bands // total) > folded_depth:
        total *= 2

    stride3 = 2 if total >= 8 else 1
    stride2 = 2 if total >= 4 else 1
    return total // (stride2 * stride3), stride2, stride3


def kernel_depth(base: int, stride: int) -> int:
    """The smallest odd kernel depth that is at least ``base`` and reads every band.

    A kernel narrower than its stride steps over bands outright, so a stride-8
    stage with the audited ``k = 7`` would never look at one band in eight — the
    cube would be *subsampled*, not reduced, which is precisely the band
    selection the primary pipeline exists to avoid.

    The bound is ``2 * stride - 1`` rather than the obvious ``stride``, and the
    difference is the tail of the axis. With an odd kernel and the symmetric
    ``k // 2`` padding the stem uses, the output depth is ``ceil(C / s)`` and the
    last output position reads input indices up to
    ``C - s + (k - 1) / 2``. Covering index ``C - 1`` therefore needs
    ``(k - 1) / 2 >= s - 1``. At ``s = 8, k = 9`` that fails by three, and bands
    253, 254 and 255 of the acquired cube would reach no parameter at all —
    silently, since every shape still agrees.

    ``2 * stride - 1`` is already odd, so the result is odd whenever ``base`` is.
    At the audited ``(2, 2, 2)`` schedule this returns ``(7, 5, 5)`` unchanged;
    on the full cube's ``(8, 2, 2)`` it returns ``(15, 5, 5)``.

    :func:`~spectralquadnet.models.branches.spatial_cnn.SpectralSpatialStem3D`'s
    coverage is asserted directly in ``tests/unit/test_branch_c_stem.py`` by
    pushing a one-hot spectrum through the real convolution, band by band.
    """
    k = max(int(base), 2 * int(stride) - 1)
    return k if k % 2 else k + 1


class SpectralSpatialStem3D(nn.Module):
    """Three strided 3-D convolutions, then a 1×1 fold of the spectral axis.

    Only the first stage leaves the spatial resolution alone: the spectral axis
    is reduced first, so the expensive 3-D kernels run on a shrinking cube while
    the 64×64 spatial detail is still intact when the widest spectral kernel
    passes over it.

    The three spectral strides come from :func:`spectral_stride_schedule` and the
    matching kernel depths from :func:`kernel_depth`, so the same class is native
    at 8, 40, 100 or 256 bands rather than tuned for one of them. Pass
    ``spectral_strides`` to override the derivation — the band-count ablations
    do not, and nothing on the primary path does either.

    Attributes:
        spectral_strides: The realised ``(s1, s2, s3)``.
        kernel_depths: The realised ``(k1, k2, k3)``.
        folded_depth: Spectral depth reaching the fold; the fold's input width
            is ``64 * folded_depth``.
    """

    #: Route the three ``Conv3d`` stages through :func:`conv3d_as_conv2d`.
    #: A pure execution choice — it holds no state, changes no parameter and is
    #: not in the state dict, so a checkpoint written with it on loads with it
    #: off. Left ``False`` here so constructing a stem directly (which every
    #: unit test and ``scripts/capture_golden.py`` does) is unconditionally the
    #: reference path; :func:`~spectralquadnet.utils.device.apply_runtime_optimisations`
    #: is the one place that turns it on, once, from the resolved runtime plan.
    decompose_conv3d: bool = False

    def __init__(
        self,
        num_bands: int = 256,
        out_channels: int = 192,
        folded_depth: int = DEFAULT_FOLDED_DEPTH,
        spectral_strides: tuple[int, int, int] | None = None,
    ) -> None:
        super().__init__()
        self.num_bands = int(num_bands)
        self.spectral_strides: tuple[int, int, int] = (
            spectral_stride_schedule(self.num_bands, folded_depth)
            if spectral_strides is None
            else (int(spectral_strides[0]), int(spectral_strides[1]), int(spectral_strides[2]))
        )
        self.kernel_depths: tuple[int, int, int] = (
            kernel_depth(BASE_KERNEL_DEPTHS[0], self.spectral_strides[0]),
            kernel_depth(BASE_KERNEL_DEPTHS[1], self.spectral_strides[1]),
            kernel_depth(BASE_KERNEL_DEPTHS[2], self.spectral_strides[2]),
        )

        channels = ((1, 16, 4), (16, 32, 8), (32, 64, 8))
        stages = [
            nn.Sequential(
                nn.Conv3d(
                    c_in,
                    c_out,
                    (k, 3, 3),
                    stride=(s, s_xy, s_xy),
                    padding=(k // 2, 1, 1),
                    bias=False,
                ),
                nn.GroupNorm(groups, c_out),
                nn.GELU(),
            )
            for (c_in, c_out, groups), k, s, s_xy in zip(
                channels, self.kernel_depths, self.spectral_strides, SPATIAL_STRIDES, strict=True
            )
        ]
        self.stage1, self.stage2, self.stage3 = stages

        # Odd kernels padded by `k // 2` give `ceil(depth / stride)` exactly, so
        # the realised depth is a closed form rather than a simulation.
        depth = self.num_bands
        for stride in self.spectral_strides:
            depth = -(-depth // stride)
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


#: Intermediate ResBlock widths of the 2-D tail, at ``width_mult = 1.0``.
#: The terminal width is ``out_dim`` and is **not** scaled: it is the fusion
#: contract every other modality is projected to.
TAIL_WIDTHS: tuple[int, int, int] = (128, 192, 256)


def _scaled_width(width: int, mult: float, multiple_of: int = 8) -> int:
    """``width * mult``, rounded to a multiple of ``multiple_of`` and at least it.

    Rounding keeps GroupNorm's group counts and the CBAM reduction divisible at
    every arm of A10's ``{0.5, 0.75, 1.0, 1.5}`` sweep. Exact at ``mult = 1.0``,
    which is what keeps the golden digests valid.
    """
    scaled = int(round(width * float(mult) / multiple_of)) * multiple_of
    return max(scaled, multiple_of)


class SpatialCNNBranch(nn.Module):
    """3-D spectral–spatial stem into a CBAM-gated ResNet-2D stack over spatial texture.

    Pools the fused feature map with concatenated signed-power-normalised mean
    and max statistics, which is more stable than raw mean/max pooling under
    the heavy-tailed activations a ResNet stack produces.

    This is the one component with convergent evidence of contribution in the
    audited run — rising fused influence, the largest parameter share, and the
    only operator in the network that can see texture × spectral position — so
    CHANGES §16.2 keeps it verbatim as ``SpectralSeedNet``'s spatial path.

    Args:
        width_mult: A10's capacity lever, scaling the three intermediate
            :data:`TAIL_WIDTHS`. ``1.0`` reproduces the audited branch exactly,
            tensor for tensor.
        stem_folded_depth: Passed to :class:`SpectralSpatialStem3D`; see
            :func:`spectral_stride_schedule`.
    """

    def __init__(
        self,
        num_bands: int = 256,
        out_dim: int = 256,
        stem_channels: int = 192,
        width_mult: float = 1.0,
        stem_folded_depth: int = DEFAULT_FOLDED_DEPTH,
    ) -> None:
        super().__init__()

        self.stem = SpectralSpatialStem3D(num_bands, stem_channels, folded_depth=stem_folded_depth)

        w1, w2, w3 = (_scaled_width(w, width_mult) for w in TAIL_WIDTHS)
        self.stages = nn.Sequential(
            ResBlock2D(stem_channels, w1, 2),
            CBAM(w1),
            ResBlock2D(w1, w2, 2),
            CBAM(w2),
            ResBlock2D(w2, w3, 2),
            CBAM(w3),
            ResBlock2D(w3, out_dim, 2),
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
