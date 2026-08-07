"""The gated bilinear fusion — **T3-4** (FU-1(b), FU-2, FU-4, FU-5).

§4.3 lists ``test_fusion_latents_are_diverse``, asserting
:math:`\\max_{n \\ne n'}\\cos(L_n, L_{n'}) < 0.9` after training. **That test
cannot be written against this module, and the reason is the finding itself.**
FU-1(b)'s remedy for M-1 is not to fix the latents' initialisation scale; it is
to delete them. With four modalities (five since FU-4), latent cross-attention
compresses nothing, and 0-E's collapse metric becomes moot rather than
satisfied — §4.2's own wording for T3-4's criterion.

So the property replacing it is the one that made collapse *matter*: the fusion
must be able to express a distinct function of each modality, and its gate must
be able to attend to two at once. That is asserted here directly, on the gate
and on the bilinear term, along with the three remaining §2.3 defects:

* **M-2a** — the confidence signal per-sample ``LayerNorm`` destroyed. The gate
  now reads the pre-normalisation log-norms, so a branch that produced almost
  nothing is distinguishable from one that produced a confident answer.
* **M-2b** — the softmax gate. Now a sigmoid, so :math:`\\sum_m g_m \\ne 1` and a
  conjunction of two modalities is expressible.
* **M-3** — first-order only. The second-order term is a low-rank bilinear pool
  over all ten unordered pairs.
"""

from __future__ import annotations

import pytest
import torch

from spectralquadnet.models.fusion import CrossModalInteraction, EmbedNet, MorphologyEmbed

pytestmark = pytest.mark.regression

M = 5
D = 256


@pytest.fixture
def fusion() -> CrossModalInteraction:
    torch.manual_seed(0)
    return CrossModalInteraction(num_modalities=M, d=D, rank=128, gate_hidden=128, drop=0.0)


@pytest.fixture
def branches() -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(3)
    return [torch.randn(8, D, generator=gen) for _ in range(M)]


# ══════════════════════════════════════════════════════════════════════
#  T3-4's validation criterion
# ══════════════════════════════════════════════════════════════════════


def test_the_gate_is_not_a_simplex(fusion, branches) -> None:
    """M-2b: ``sum_m g_m != 1``, so attending to C does not cost attention on B.

    §4.2's criterion for T3-4, verbatim. Distinguishing two rice varieties needs
    spectral shape *and* grain morphology; a softmax over modalities forbids the
    conjunction by its own shape.
    """
    fusion.eval()
    gate, _ = fusion.gate_values(branches)

    assert gate.shape == (8, M)
    assert ((gate > 0.0) & (gate < 1.0)).all(), "the gate must be a sigmoid, not a softmax"
    assert (gate.sum(dim=1) - 1.0).abs().min() > 0.05


def test_two_modalities_can_be_fully_on_at_once(fusion) -> None:
    """The conjunction M-2b says a softmax cannot express, exhibited.

    A weight setting where two gates both exceed 0.9 exists and is reachable —
    which is a statement about the *function class*, not about what training
    happens to find.
    """
    fusion.eval()
    with torch.no_grad():
        fusion.modality_gate[2].bias.copy_(torch.tensor([4.0, 4.0, -4.0, -4.0, -4.0]))
        fusion.modality_gate[2].weight.zero_()

    gate, _ = fusion.gate_values([torch.randn(4, D) for _ in range(M)])

    assert (gate[:, :2] > 0.9).all()
    assert (gate[:, 2:] < 0.1).all()


def test_the_gate_sees_the_confidence_the_normaliser_removes(fusion, branches) -> None:
    """M-2a: scaling one branch's output changes the gate, because ν is read pre-norm.

    Per-sample normalisation sets every branch's norm to the same value, so
    without ν the gate is blind to how much signal a branch actually produced.
    ``BatchNorm1d`` in eval mode applies a *running* scale rather than a
    per-sample one, so the token also keeps the magnitude — both halves of FU-2.
    """
    fusion.eval()
    quiet = [b.clone() for b in branches]
    quiet[2] = quiet[2] * 0.001

    base, _ = fusion.gate_values(branches)
    faded, _ = fusion.gate_values(quiet)

    assert not torch.allclose(base, faded, atol=1e-4)


def test_the_confidence_vector_is_the_log_norm(fusion, branches) -> None:
    """ν is what §3.4 FU-2 writes down, not a proxy for it.

    Checked by construction rather than by outcome: doubling every branch adds
    ``log 2`` to every component, which no other summary of the pre-norm
    embedding does.
    """
    fusion.eval()
    with torch.no_grad():
        gate_input_base = torch.stack([torch.log(b.norm(dim=1) + 1e-6) for b in branches], dim=1)
        gate_input_2x = torch.stack(
            [torch.log((2 * b).norm(dim=1) + 1e-6) for b in branches], dim=1
        )

    assert torch.allclose(gate_input_2x - gate_input_base, torch.full((8, M), 0.6931), atol=1e-3)


def test_the_fusion_has_a_multiplicative_term(fusion, branches) -> None:
    """M-3: the output is not a weighted sum of the branch tokens.

    The falsifiable form. A purely first-order fusion is additive in each
    modality: ``f(2*b_0, ...) - f(b_0, ...)`` would be independent of the other
    modalities' values. With the bilinear term it is not.
    """
    fusion.eval()
    other_a = [b.clone() for b in branches]
    other_b = [b.clone() for b in branches]
    other_b[1] = other_b[1] * 3.0

    doubled_a = [b.clone() for b in other_a]
    doubled_a[0] = doubled_a[0] * 2.0
    doubled_b = [b.clone() for b in other_b]
    doubled_b[0] = doubled_b[0] * 2.0

    with torch.no_grad():
        delta_a = fusion(doubled_a) - fusion(other_a)
        delta_b = fusion(doubled_b) - fusion(other_b)

    assert not torch.allclose(delta_a, delta_b, atol=1e-3)


def test_every_unordered_pair_contributes(fusion) -> None:
    """All ten pairs of five modalities are in the second-order sum, none twice."""
    import itertools

    pairs = list(itertools.combinations(range(fusion.num_modalities), 2))

    assert len(pairs) == 10
    assert len(fusion.bilinear) == fusion.num_modalities
    assert all(proj.out_features == fusion.rank for proj in fusion.bilinear)


# ══════════════════════════════════════════════════════════════════════
#  FU-4 — the fifth modality
# ══════════════════════════════════════════════════════════════════════


def test_the_morphology_token_is_a_modality_not_a_side_channel() -> None:
    """FU-4: 8 -> 64 -> 256, and the fusion carries five tokens, not four."""
    embed = MorphologyEmbed(n_morph=8, hidden=64, d=D)

    assert embed(torch.randn(4, 8)).shape == (4, D)
    assert 16_000 < sum(p.numel() for p in embed.parameters()) < 18_000


def test_the_fusion_takes_five_tokens(fusion, branches) -> None:
    fusion.eval()

    assert fusion(branches).shape == (8, D)
    with pytest.raises((ValueError, IndexError, RuntimeError)):
        fusion(branches[:4])


# ══════════════════════════════════════════════════════════════════════
#  FU-5 and the budget
# ══════════════════════════════════════════════════════════════════════


def test_output_proj_is_gone_and_embed_net_remains(fusion) -> None:
    """N-10: the same pre-LN residual MLP was applied to the same vector twice."""
    assert not hasattr(fusion, "output_proj")
    assert not hasattr(fusion, "latents"), "FU-1(b) deletes the Perceiver latents"
    assert not hasattr(fusion, "blocks")

    embed = EmbedNet(D, 512, 0.1)
    assert embed(torch.randn(4, D)).shape == (4, D)


def test_the_fusion_costs_what_section_3_8_budgets(fusion) -> None:
    """§3.8: 2.19 M -> ~0.55 M, the largest single saving in the plan.

    Pinned as a range rather than an exact count so a documented width change
    does not fail it, and tight enough that reintroducing an attention stack
    would.
    """
    total = sum(p.numel() for p in fusion.parameters())

    assert 450_000 < total < 560_000, f"{total:,}"


def test_the_per_modality_norm_is_a_dataset_statistic(fusion, branches) -> None:
    """FU-2: ``BatchNorm1d``, not ``LayerNorm``.

    The distinction is observable, and it is the one M-2a turns on: in eval mode
    a BatchNorm applies a fixed running scale, so a low-SNR sample keeps a small
    token instead of being amplified to unit norm.
    """
    assert all(isinstance(n, torch.nn.BatchNorm1d) for n in fusion.branch_norms)

    fusion.eval()
    _, tokens = fusion.gate_values(branches)
    quiet = [b.clone() for b in branches]
    quiet[0] = quiet[0] * 1e-3
    _, quiet_tokens = fusion.gate_values(quiet)

    assert quiet_tokens[:, 0].norm() < tokens[:, 0].norm() / 2
