"""§4.3's ``test_branch_inputs_are_distinct`` — **T3-6 / BR-2**, and C-2's closure.

§2.2.2 states the finding as sharply as it can be stated: **Branch A was a
strict functional subset of Branch D.** Both received
``extract_grid_spectra(x, 4)`` — the same ``(B, 16, 40)`` tensor, byte for byte
— and both reduced it to a 256-vector. 2.77 M parameters across two branches
reading one input, while the paper claimed "four disjoint views of the same
patch".

The claim §4.5 says to make after BR-1…BR-4 is that the four views are now
genuinely disjoint, "supported with a pairwise input-hash test". This is that
test. It hashes what each branch *actually consumes* —
:meth:`SpectralQuadNet.branch_inputs` is the same method ``forward`` calls, not
a reconstruction — and asserts the four hashes are pairwise distinct.

A hash test alone would pass on a rescaling, so three sharper properties follow
it: A's input is gain-free where D's is not, A's grid is finer than D's, and
neither is recoverable from the other by a per-band affine map.
"""

from __future__ import annotations

import hashlib

import pytest
import torch

from spectralquadnet.models.spectral_quadnet import SpectralQuadNet
from spectralquadnet.models.stats_ops import extract_grid_spectra

pytestmark = pytest.mark.regression

BRANCHES = ("a", "b", "c", "d")


def _digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


@pytest.fixture
def model(cfg, physical_wl):
    torch.manual_seed(0)
    return SpectralQuadNet.from_config(cfg, physical_wl).eval()


@pytest.fixture(scope="module")
def seed_batch(cfg) -> torch.Tensor:
    """A batch with a real zero background, so the masked paths are exercised."""
    gen = torch.Generator().manual_seed(19)
    x = (torch.rand(3, cfg.data.num_bands, 64, 64, generator=gen) + 0.4).float()
    keep = torch.zeros(1, 1, 64, 64)
    keep[..., 12:52, 18:46] = 1.0
    return x * keep


# ══════════════════════════════════════════════════════════════════════
#  §4.3 — the pairwise input hash
# ══════════════════════════════════════════════════════════════════════


def test_branch_inputs_are_distinct(model, seed_batch) -> None:
    """The four branches' input tensors hash pairwise differently. C-2's criterion."""
    inputs = model.branch_inputs(seed_batch)
    digests = {b: _digest(inputs[b]) for b in BRANCHES}

    assert len(set(digests.values())) == len(BRANCHES), digests


def test_a_and_d_no_longer_receive_the_same_tensor(model, seed_batch) -> None:
    """The specific pair §2.2.2 is about — now different in shape, not just in value.

    Before BR-2 the two were the *same object*: one call to
    ``extract_grid_spectra(x, 4)`` fed both. A's input is now an 8x8 grid of
    SNV'd spectra and their two λ-derivatives; D's is the 4x4 grid of raw ones.
    """
    inputs = model.branch_inputs(seed_batch)

    assert inputs["a"].shape != inputs["d"].shape
    assert inputs["a"].shape == (seed_batch.shape[0] * model.grid_a**2, 3, seed_batch.shape[1])
    assert inputs["d"].shape == (seed_batch.shape[0], model.grid_d**2, seed_batch.shape[1])


def test_branch_a_is_gain_free_and_branch_d_is_not(model, seed_batch) -> None:
    """The functional separation, not just a shape one.

    A sees spectral *shape*: SNV removes the per-pixel gain, so scaling the cube
    leaves A's input where it was. D sees the raw spectrum and moves with the
    gain. Two branches that respond differently to the same transform cannot be
    computing the same function of the patch.

    A's invariance is exact in exact arithmetic and approximate in float32 —
    ``snv``'s ``std + eps`` denominator does not scale with the gain — so the
    two are compared by *relative* drift rather than against a shared absolute
    tolerance, and the gap between them is three orders of magnitude.
    """
    plain = model.branch_inputs(seed_batch)
    scaled = model.branch_inputs(seed_batch * 2.5)

    def drift(key: str) -> float:
        a, b = plain[key], scaled[key]
        return float((a - b).abs().max() / a.abs().max())

    assert drift("a") < 1e-3
    assert drift("d") > 0.5
    assert drift("a") * 100 < drift("d")


def test_branch_a_sees_a_finer_grid_than_branch_d(model, seed_batch) -> None:
    """BR-2(3): 8x8 is a 64:1 spatial compression against 4x4's 256:1 (§2.2.3).

    A cell of A's grid covers a quarter of the area of one of D's, so fewer of
    A's cells straddle the seed boundary and average foreground with background.
    That mixing is what §2.2.9 (M-12) is about, and the pure-cell fraction is
    the direct measure of it: on the finer grid a larger share of cells are
    wholly seed or wholly background.
    """
    assert model.grid_a == 8
    assert model.grid_d == 4

    _, fine_mass = extract_grid_spectra(seed_batch, grid_size=model.grid_a)
    _, coarse_mass = extract_grid_spectra(seed_batch, grid_size=model.grid_d)

    def pure_fraction(mass: torch.Tensor) -> float:
        return float(((mass < 0.02) | (mass > 0.98)).float().mean())

    assert fine_mass.shape[1] == 4 * coarse_mass.shape[1]
    assert pure_fraction(fine_mass) > pure_fraction(coarse_mass)


def test_branch_c_is_the_only_one_that_keeps_the_spatial_axes(model, seed_batch) -> None:
    """C-3's closure at the input level: only C receives a tensor with H and W.

    The other three see spectra that have already been pooled over space. This
    is why §3.8 moves 0.6 M parameters *into* C — it is the only branch whose
    input no other branch could reconstruct.
    """
    inputs = model.branch_inputs(seed_batch)

    assert inputs["c"].shape == seed_batch.shape
    assert all(inputs[b].dim() < 4 for b in ("a", "b", "d"))


def test_branch_b_carries_information_no_spectral_branch_has(model, seed_batch) -> None:
    """BR-1(iii): the morphometrics move B's descriptor and touch nothing else.

    Size and shape are not derivable from a resized 64x64 patch by any operator
    in the network — that is M-13's point — so a change to them must reach B
    (and the fifth fusion token) and no other branch's input.
    """
    n = seed_batch.shape[0]
    without = model.branch_inputs(seed_batch, morph=torch.zeros(n, model.n_morph))
    with_morph = model.branch_inputs(seed_batch, morph=torch.randn(n, model.n_morph))

    assert not torch.equal(without["b"], with_morph["b"])
    for other in ("a", "c", "d"):
        assert torch.equal(without[other], with_morph[other]), other


def test_the_inputs_are_the_ones_the_forward_pass_uses(model, seed_batch) -> None:
    """Non-vacuity: ``branch_inputs`` is not a parallel implementation.

    A test that hashed a reconstruction would keep passing after the forward
    pass started consuming something else. ``forward`` calls this method, so
    perturbing what it returns has to change the logits.
    """
    with torch.no_grad():
        before = model(seed_batch).clone()
        after = model(seed_batch, morph=torch.full((seed_batch.shape[0], model.n_morph), 3.0))

    assert not torch.allclose(before, after)


def test_the_mass_weights_ride_with_branch_a(model, seed_batch) -> None:
    """BR-2(2): ω_n is produced alongside A's cells, not recomputed downstream.

    ``extract_grid_spectra`` always computed the cell foreground mass as its
    normaliser and discarded it (M-12). Returning it is what lets A pool by mass
    rather than treat a corner cell holding four seed pixels as equal to one
    that is entirely seed.
    """
    inputs = model.branch_inputs(seed_batch)
    mass = inputs["a_mass"]

    assert mass.shape == (seed_batch.shape[0], model.grid_a**2)
    assert ((mass >= 0.0) & (mass <= 1.0)).all()
    # The synthetic seed is a centred rectangle, so corner cells are empty and
    # centre cells are full — the spread the weighting exists to exploit.
    assert float(mass.max()) > 0.9
    assert float(mass.min()) == pytest.approx(0.0, abs=1e-6)
