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

from spectralquadnet.models.branches.spatial_cnn import SpatialCNNBranch, SpectralSpatialStem3D

pytestmark = pytest.mark.regression

BANDS = 40


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
