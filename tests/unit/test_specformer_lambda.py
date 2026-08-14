"""Branch D's wavelength axis — **T3-3 / BR-4**, and F-3's transfer question.

The pre-Tier-3 ``SpecFormerBranch`` accepted ``physical_wl`` and used it for
nothing. It tokenised by an index stride of 4 and added a **learned positional
embedding indexed by token position**, which makes the branch un-transferable in
the precise sense F-3 asks about: re-run band selection at a different ``k`` and
"token 3" refers to a different region of the spectrum, so the table has to be
relearned from scratch.

§4.2's validation criterion for T3-3 is exactly that — "Branch D transfers
across band counts without retraining the positional table (settles F-3)" — and
:func:`test_the_branch_transfers_across_band_counts` is it, in the strongest
form the property admits: a state dict trained at one band count loads into a
model built at another with ``strict=True``, and the two agree on a spectrum
they both sample.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from spectralquadnet.models.branches.specformer import (
    LambdaWindowPooling,
    RelativeLambdaBias,
    SpecFormerBranch,
)

pytestmark = pytest.mark.regression

D_MODEL = 192
HEADS = 8


@pytest.fixture(scope="module")
def wl() -> torch.Tensor:
    """The real, irregularly spaced 40-band selection the run uses."""
    path = "tests/regression/golden/physical_wl_spa40.npy"
    return torch.from_numpy(np.load(path))


#: The audited λ-window count. Asked for **directly**: before `specf_tokens`
#: existed, a caller wanting ten windows at two different band counts had to
#: back-compute two different `specf_patch` values (8 at k=40, 4 at k=20), which
#: is the same coupling BR-4(iii) removed from the branch itself.
N_TOKENS = 10


def _branch(wavelengths: torch.Tensor, n_tokens: int = N_TOKENS) -> SpecFormerBranch:
    torch.manual_seed(0)
    return SpecFormerBranch(
        physical_wl=wavelengths,
        num_bands=wavelengths.numel(),
        n_tokens=n_tokens,
        d_model=D_MODEL,
        n_heads=HEADS,
        n_layers=4,
        out_dim=256,
        dropout=0.0,
    ).eval()


# ══════════════════════════════════════════════════════════════════════
#  T3-3's validation criterion
# ══════════════════════════════════════════════════════════════════════


def test_the_branch_transfers_across_band_counts(wl) -> None:
    """A 40-band Branch D loads into a 20-band one, strict, with no table to relearn.

    Every learned tensor in the branch is now indexed by *token*, and a token is
    a fixed window of ``[λ_min, λ_max]``. The positional code is derived from
    that window's centre wavelength rather than stored, so the shapes match and
    the meanings match — which is what the pre-Tier-3 learned table could not
    do at either.

    Both branches are built from the **same** ``n_tokens`` at different band
    counts, which is the point: the window count is a spectral-resolution choice
    and no longer has to be back-computed from ``k``.
    """
    forty = _branch(wl)
    twenty = _branch(wl[::2])

    result = twenty.load_state_dict(forty.state_dict(), strict=True)

    assert not result.missing_keys and not result.unexpected_keys
    assert twenty.windows.n_tokens == forty.windows.n_tokens
    assert torch.allclose(twenty.windows.token_wl, forty.windows.token_wl, atol=1e-6)


def test_a_transferred_branch_agrees_on_a_spectrum_both_can_sample(wl) -> None:
    """Non-vacuity: the weights transfer *and* mean the same thing.

    A shape-compatible load proves nothing on its own. Fed a spectrum that is
    constant within every λ-window, the two branches pool identical token values
    and must produce identical embeddings — which they can only do if token
    ``t`` covers the same wavelengths in both.
    """
    forty = _branch(wl)
    twenty = _branch(wl[::2])
    twenty.load_state_dict(forty.state_dict(), strict=True)

    # A ramp in λ is not constant per window, so use the window index itself:
    # each band takes the value of the window it falls into.
    pool_40 = forty.windows.pool  # (L, 40)
    per_band_40 = (pool_40 > 0).float().argmax(dim=0).float()
    per_band_20 = per_band_40[::2]

    with torch.no_grad():
        out_40 = forty(per_band_40.expand(1, 16, -1).contiguous())
        out_20 = twenty(per_band_20.expand(1, 16, -1).contiguous())

    assert torch.allclose(out_40, out_20, atol=1e-4)


def test_the_positional_code_is_derived_from_wavelength_not_learned(wl) -> None:
    """BR-4(i): ``spec_pos_embed`` is a buffer, not a parameter.

    The distinction is the whole of F-3. A learned table is a parameter with one
    row per token index and no relationship to λ; a derived code is a function
    of the window centres and transfers by construction.
    """
    branch = _branch(wl)
    names = {n for n, _ in branch.named_parameters()}
    buffers = {n for n, _ in branch.named_buffers()}

    assert "spec_pos_embed" in buffers
    assert "spec_pos_embed" not in names
    assert "spec_cls" in names, "the CLS token has no wavelength, so it keeps a learned code"


# ══════════════════════════════════════════════════════════════════════
#  λ-uniform tokenisation (BR-4 iii)
# ══════════════════════════════════════════════════════════════════════


def test_the_windows_are_uniform_in_wavelength(wl) -> None:
    """Token centres are equally spaced in λ, whatever the band spacing is.

    After this pooling the axis the tokenizer's 3/5/7 kernels run over is
    uniform, so a kernel width is a *bandwidth* — which fixes C-5 for Branch D
    the way FE-1 fixes it for Branch A.
    """
    windows = LambdaWindowPooling(wl, n_tokens=10, lam_range=(0.0, 1.0))
    gaps = windows.token_wl[1:] - windows.token_wl[:-1]

    assert windows.n_tokens == 10
    assert torch.allclose(gaps, gaps[0].expand_as(gaps), atol=1e-6)
    # The band grid is not uniform, which is what makes the above non-trivial.
    band_gaps = wl[1:] - wl[:-1]
    assert float(band_gaps.max() / band_gaps.min()) > 3.0


def test_every_window_pools_at_least_one_band(wl) -> None:
    """A window no selected band falls into takes its nearest band, not zero.

    A zero token would be indistinguishable from a genuine null response, and
    would put a hole in the sequence the attention then reasons over.
    """
    sparse = torch.tensor([0.0, 0.02, 0.04, 0.96, 0.98, 1.0])
    windows = LambdaWindowPooling(sparse, n_tokens=6)

    assert torch.allclose(windows.pool.sum(dim=1), torch.ones(6), atol=1e-6)
    assert (windows.pool > 0).any(dim=1).all()


def test_raw_nanometres_are_refused(wl) -> None:
    """The `(0, 1)` domain makes the min-max normalisation a precondition, not a convention.

    Fed raw nanometres every band falls outside every window, each window takes
    its nearest band, and the branch silently sees ten copies of band 0 — a
    failure that produces numbers rather than an error. It raises instead.
    """
    with pytest.raises(ValueError, match="min-max normalised"):
        _branch(wl * 500.0 + 400.0)


def test_the_pooling_is_an_average_over_the_bands_in_each_window(wl) -> None:
    windows = LambdaWindowPooling(wl, n_tokens=10)
    ones = torch.ones(2, wl.numel())

    assert torch.allclose(windows(ones), torch.ones(2, 10), atol=1e-6)


# ══════════════════════════════════════════════════════════════════════
#  Relative-λ attention bias (BR-4 ii)
# ══════════════════════════════════════════════════════════════════════


def test_the_bias_is_a_function_of_the_wavelength_difference(wl) -> None:
    """One value per (head, Δλ) — so a head can specialise on "bands 60 nm apart".

    The bias for two token pairs at the same Δλ must agree; the bias at
    different Δλ must not. That is what separates a *relative* bias from 121
    unrelated learned logits.
    """
    torch.manual_seed(0)
    windows = LambdaWindowPooling(wl, n_tokens=10)
    bias = RelativeLambdaBias(windows.token_wl, n_heads=HEADS)

    with torch.no_grad():
        out = bias(n_prefix=0)

    assert out.shape == (HEADS, 10, 10)
    # (0, 2) and (5, 7) are the same Δλ on a uniform token grid.
    assert torch.allclose(out[:, 0, 2], out[:, 5, 7], atol=1e-3)
    assert not torch.allclose(out[:, 0, 2], out[:, 0, 5], atol=1e-4)


def test_the_cls_token_is_left_unbiased(wl) -> None:
    """It has no wavelength, so any bias on its row would be an arbitrary constant."""
    torch.manual_seed(0)
    windows = LambdaWindowPooling(wl, n_tokens=10)
    bias = RelativeLambdaBias(windows.token_wl, n_heads=HEADS)

    with torch.no_grad():
        padded = bias(n_prefix=1)

    assert padded.shape == (HEADS, 11, 11)
    assert float(padded[:, 0, :].abs().max()) == 0.0
    assert float(padded[:, :, 0].abs().max()) == 0.0


def test_the_bias_reaches_the_attention(wl) -> None:
    """Non-vacuity: zeroing ``b_psi``'s output must change the branch's embedding.

    A bias that is computed and then dropped would pass every structural test
    above.
    """
    branch = _branch(wl)
    x = torch.rand(2, 16, wl.numel()) + 0.3

    with torch.no_grad():
        before = branch(x).clone()
        branch.lambda_bias.mlp[2].weight.zero_()
        branch.lambda_bias.mlp[2].bias.zero_()
        after = branch(x)

    assert not torch.allclose(before, after, atol=1e-5)


def test_branch_d_costs_what_section_3_8_budgets(wl) -> None:
    """§3.8: 2.18 M → ≈ 1.25 M at ``d_model`` 256 → 192.

    "With λ-aware attention it needs less brute capacity" is the claim; the
    saving is what pays for Branch C.
    """
    total = sum(p.numel() for p in _branch(wl).parameters())

    assert 1_150_000 < total < 1_350_000, f"{total:,}"
    assert total < 2_180_866
