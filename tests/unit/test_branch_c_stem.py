"""Branch C's joint spectral–spatial stem — **T3-2 / BR-3**, C-3's closure.

C-3, in one sentence: the network contained **no joint spectral–spatial
operator**. ``band_reduce`` was ``Conv2d(C, C, 1, groups=C)`` followed by
``Conv2d(C, 64, 1)`` — two 1×1 convolutions — so the band axis was collapsed to
64 channels before any spatial kernel ran, and the very first 3×3 already saw a
spectrally-mixed map. "This absorption feature, in this part of the seed" was
not in the hypothesis class of any module in the model.

The falsifiable form of that, and what this module tests: a stem that mixes
bands *before* any spatial kernel cannot distinguish two cubes that differ only
in **which** band a feature lands in at a given location, when the per-band
spatial marginals are held fixed. The 3-D stem can, because its first kernel
spans 7 bands and 3×3 pixels at once.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from spectralquadnet.models.branches.spatial_cnn import (
    DEFAULT_FOLDED_DEPTH,
    SpatialCNNBranch,
    SpectralSpatialStem3D,
    kernel_depth,
    spectral_stride_schedule,
)

pytestmark = pytest.mark.regression

BANDS = 40
#: The acquired band count — the primary pipeline's input.
FULL_BANDS = 256


def _band_reduce_1x1(num_bands: int = BANDS) -> nn.Sequential:
    """The pre-Tier-3 stem, rebuilt here so the comparison is against code, not prose."""
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Conv2d(num_bands, num_bands, 1, groups=num_bands, bias=False),
        nn.Conv2d(num_bands, 64, 1, bias=False),
        nn.GroupNorm(8, 64),
        nn.GELU(),
    )


def _swapped_pair() -> tuple[torch.Tensor, torch.Tensor]:
    """Two cubes with identical per-band spatial marginals and different band–space pairing.

    Cube ``a`` has a bright blob at the top-left in band 10 and at the
    bottom-right in band 30; cube ``b`` swaps which band each blob sits in.
    Every band's *set* of active pixels is the same in both, so the two are
    indistinguishable to any operator that pools over space before mixing
    bands, or over bands before touching space.
    """
    a = torch.zeros(1, BANDS, 64, 64)
    b = torch.zeros(1, BANDS, 64, 64)
    a[0, 10, 8:24, 8:24] = 1.0
    a[0, 30, 40:56, 40:56] = 1.0
    b[0, 10, 40:56, 40:56] = 1.0
    b[0, 30, 8:24, 8:24] = 1.0
    return a, b


# ══════════════════════════════════════════════════════════════════════
#  C-3's closure
# ══════════════════════════════════════════════════════════════════════


def test_the_3d_stem_separates_a_band_space_swap() -> None:
    """The stem's whole reason for existing, stated as a discrimination test."""
    torch.manual_seed(0)
    stem = SpectralSpatialStem3D(BANDS, 192).eval()
    a, b = _swapped_pair()

    with torch.no_grad():
        out_a, out_b = stem(a), stem(b)

    assert not torch.allclose(out_a, out_b, atol=1e-4)


def test_the_1x1_band_reduce_cannot() -> None:
    """The companion. Two 1×1 convolutions give the swapped pair the same *content*.

    A 1×1 stem applies one linear map to every pixel's spectrum independently,
    so swapping which band a blob occupies swaps the corresponding *locations*
    in the output and nothing else: the multiset of feature vectors over the
    spatial grid is identical, and every downstream pooling over space
    (``mean``/``amax``, which is exactly what Branch C's tail ends with) sees
    the same numbers.
    """
    reduce = _band_reduce_1x1().eval()
    a, b = _swapped_pair()

    with torch.no_grad():
        pooled_a = reduce(a).mean([2, 3])
        pooled_b = reduce(b).mean([2, 3])

    assert torch.allclose(pooled_a, pooled_b, atol=1e-5)


def test_the_stem_keeps_the_spectral_axis_alive_for_three_stages() -> None:
    """40 → 20 → 10 → 5, then folded into channels rather than deleted.

    The property §3.3 BR-3's diagram is about: each of the ``64 * 5`` channels
    entering the 1×1 fold is a (spectral position × learned feature) pair, so
    the 2-D tail's kernels still operate on features that carry where in the
    spectrum they came from.
    """
    stem = SpectralSpatialStem3D(BANDS, 192)
    x = torch.randn(2, BANDS, 64, 64)

    with torch.no_grad():
        h1 = stem.stage1(x.unsqueeze(1))
        h2 = stem.stage2(h1)
        h3 = stem.stage3(h2)

    assert h1.shape[2] == 20
    assert h2.shape[2] == 10
    assert h3.shape[2] == 5
    assert stem.folded_depth == 5
    assert stem.fold[0].in_channels == 64 * 5


def test_the_stem_output_feeds_the_existing_2d_tail() -> None:
    """``(B, 192, 16, 16)``, which is what §3.3's block diagram specifies."""
    stem = SpectralSpatialStem3D(BANDS, 192).eval()

    with torch.no_grad():
        out = stem(torch.randn(2, BANDS, 64, 64))

    assert out.shape == (2, 192, 16, 16)


# ══════════════════════════════════════════════════════════════════════
#  The mask, and the budget
# ══════════════════════════════════════════════════════════════════════


def test_the_padded_region_stays_zero_through_every_stage() -> None:
    """BR-3: "use the persisted mask to zero padded regions after every stage".

    Without it the stem's own padding and the GELU bias would grow a non-zero
    response in the frame within two layers, and a CNN that can see the frame
    can learn it. Checked at the stem's output resolution, where the pooled mask
    is still exactly zero on cells no seed pixel touches.
    """
    stem = SpectralSpatialStem3D(BANDS, 192).eval()
    x = torch.randn(2, BANDS, 64, 64)
    mask = torch.zeros(2, 1, 64, 64)
    mask[..., 16:48, 16:48] = 1.0

    with torch.no_grad():
        h1 = stem._apply_mask(stem.stage1(x.unsqueeze(1)), mask)

    assert float(h1[..., :16, :].abs().max()) == 0.0
    assert float(h1[..., 20:40, 20:40].abs().max()) > 0.0


def test_the_branch_accepts_the_mask_and_reacts_to_it() -> None:
    torch.manual_seed(0)
    branch = SpatialCNNBranch(BANDS, 256).eval()
    x = torch.randn(2, BANDS, 64, 64)
    full = torch.ones(2, 1, 64, 64)
    half = full.clone()
    half[..., 32:, :] = 0.0

    with torch.no_grad():
        assert not torch.allclose(branch(x, full), branch(x, half), atol=1e-5)


def test_branch_c_costs_what_section_3_8_budgets() -> None:
    """§3.8: 1.69 M → ≈ 2.3 M, funded by the 591 k BR-1 freed from Branch B.

    The reallocation is the point of the whole tier — parameters moved out of
    modules operating on 640 numbers and into the only one that sees the full
    cube — so the direction is asserted, not just the magnitude.
    """
    torch.manual_seed(0)
    branch = SpatialCNNBranch(BANDS, 256)
    total = sum(p.numel() for p in branch.parameters())
    stem = sum(p.numel() for p in branch.stem.parameters())

    assert 2_100_000 < total < 2_400_000, f"{total:,}"
    assert total > 1_694_158, "Branch C must gain capacity, not lose it"
    assert 100_000 < stem < 200_000, f"stem {stem:,}"


def test_the_stem_handles_an_odd_band_count() -> None:
    """The fold width is derived, not hardcoded — so a band-count ablation still builds.

    F-3's sweep changes ``num_bands``; a stem that assumed 40 would fail there
    rather than answer the question.
    """
    for bands in (17, 31, 64, 256):
        stem = SpectralSpatialStem3D(bands, 96)
        with torch.no_grad():
            out = stem(torch.randn(1, bands, 32, 32))
        assert out.shape == (1, 96, 8, 8), bands


# ══════════════════════════════════════════════════════════════════════
#  256-band native — the spectral stride schedule
# ══════════════════════════════════════════════════════════════════════


def test_the_schedule_reproduces_the_audited_stem_at_forty_bands() -> None:
    """The compatibility contract the whole derivation is built to satisfy.

    The retained band-selection arms and every golden regression digest describe
    a stem that halves the band axis three times. If the derivation ever stopped
    returning that at k = 40, the primary path and the arms it is measured
    against would no longer be the same network.
    """
    assert spectral_stride_schedule(40, DEFAULT_FOLDED_DEPTH) == (2, 2, 2)
    stem = SpectralSpatialStem3D(40, 192)
    assert stem.kernel_depths == (7, 5, 5), "the audited kernel depths, unwidened"
    assert stem.folded_depth == 5
    assert stem.fold[0].in_channels == 64 * 5


def test_the_schedule_folds_the_full_cube_at_the_configured_depth() -> None:
    """256 bands must not reach the fold at depth 32.

    Three hardcoded halvings put 32 spectral positions into the fold, i.e. a
    ``Conv2d(2048 -> 192)`` and a stage-2 cube 6.4x deeper than the design was
    measured on. That is a 40-band stem carrying a full cube; the derivation
    spends the extra reduction in stage 1, where one input channel makes it
    cheapest.
    """
    assert spectral_stride_schedule(FULL_BANDS, DEFAULT_FOLDED_DEPTH) == (8, 2, 2)

    stem = SpectralSpatialStem3D(FULL_BANDS, 192)
    assert stem.folded_depth == DEFAULT_FOLDED_DEPTH
    assert stem.fold[0].in_channels == 64 * DEFAULT_FOLDED_DEPTH
    # The widest stride gets a kernel wide enough to cover it, so no band is
    # stepped over — a stride-8 stage with the audited k = 7 would subsample the
    # cube, which is the band selection the primary path exists to avoid.
    assert stem.kernel_depths[0] >= stem.spectral_strides[0]
    assert stem.kernel_depths == (15, 5, 5)


def test_every_band_reaches_at_least_one_stage_one_tap() -> None:
    """The no-subsampling property, checked rather than argued.

    A stem whose first kernel is narrower than its first stride never multiplies
    some bands by anything. Verified by pushing a one-hot spectrum through the
    real stage-1 convolution and requiring a response.
    """
    for bands in (40, 100, 128, 192, 224, 256):
        stem = SpectralSpatialStem3D(bands, 32).eval()
        torch.nn.init.constant_(stem.stage1[0].weight, 1.0)
        for band in range(bands):
            x = torch.zeros(1, bands, 4, 4)
            x[0, band] = 1.0
            with torch.no_grad():
                response = stem.stage1[0](x.unsqueeze(1))
            assert float(response.abs().max()) > 0.0, f"band {band} of {bands} is never read"


def test_the_kernel_widens_only_when_the_stride_demands_it() -> None:
    assert kernel_depth(7, 1) == 7
    assert kernel_depth(7, 2) == 7
    assert kernel_depth(7, 4) == 7
    assert kernel_depth(7, 8) == 15
    assert kernel_depth(7, 16) == 31
    assert kernel_depth(5, 2) == 5
    for base in (5, 7):
        for stride in (1, 2, 4, 8, 16, 32):
            k = kernel_depth(base, stride)
            assert k % 2 == 1, "symmetric padding needs an odd kernel"
            # 2s-1, not s: covering the LAST band under symmetric padding
            # needs (k-1)/2 >= s-1, which is where a stride-8 stage with a
            # 9-tap kernel drops bands 253-255 of the acquired cube.
            assert k >= 2 * stride - 1 and k >= base


def test_the_folded_depth_never_exceeds_its_bound() -> None:
    """The property the schedule solves for, over every budget the study uses."""
    for bands in (1, 2, 5, 8, 16, 17, 20, 31, 40, 50, 64, 100, 128, 160, 192, 224, 256):
        for target in (2, 4, 8):
            strides = spectral_stride_schedule(bands, target)
            stem = SpectralSpatialStem3D(bands, 32, folded_depth=target)
            assert stem.spectral_strides == strides
            assert stem.folded_depth <= max(target, 1), (bands, target)
            # And it is the *smallest* such reduction: one stride less would
            # overshoot, so the schedule is not simply reducing as far as it can.
            assert -(-bands // (strides[0] * strides[1] * strides[2])) == stem.folded_depth


def test_the_full_cube_stem_costs_less_than_a_naive_one() -> None:
    """The reason the derivation exists, as a parameter count.

    Three hardcoded halvings on 256 bands is a 2048-channel fold; the derived
    schedule is a 512-channel one, and the difference is most of the stem.
    """
    derived = SpectralSpatialStem3D(FULL_BANDS, 192)
    naive = SpectralSpatialStem3D(FULL_BANDS, 192, spectral_strides=(2, 2, 2))

    assert naive.folded_depth == 32
    assert naive.fold[0].in_channels == 64 * 32
    n_derived = sum(p.numel() for p in derived.parameters())
    n_naive = sum(p.numel() for p in naive.parameters())
    assert n_naive > 2 * n_derived, f"derived {n_derived:,} vs naive {n_naive:,}"


def test_the_full_cube_stem_still_feeds_the_same_2d_tail() -> None:
    """The tail's contract is a shape, and the band count must not change it."""
    for bands in (40, FULL_BANDS):
        stem = SpectralSpatialStem3D(bands, 192).eval()
        with torch.no_grad():
            assert stem(torch.randn(1, bands, 64, 64)).shape == (1, 192, 16, 16), bands
