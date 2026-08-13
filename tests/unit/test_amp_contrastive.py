"""IC-7 — the contrastive terms keep fp32; the epoch keeps autocast.

``use_amp = (supcon is None) and (scaler is not None)`` meant that merely
*passing* a SupCon module dropped the whole epoch — backbone forward included —
into fp32. On an 11 GB card already at 91% of its working set that doubles
activation memory and the allocator starts paging against a host page cache
holding the 5.65 GB mmapped cube. Phase 3 cost 190–405 s/epoch against Phases
1–2's 39 s, a 5–10× jump on identical data with a sampler drawing the same
number of indices (CHANGES §7.3, §9.2).

SupCon genuinely needs fp32, but only for the ``exp(·/τ)`` reduction at τ=0.10 —
not for the backbone. The fix confines the fp32 region to the similarity matrix.
IC-7's stated validation criterion is *"SupCon loss agrees with the fp32
reference to <1e-4"*, which is what :func:`test_supcon_matches_the_fp32_reference`
measures.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch.amp import autocast

from spectralquadnet.engine.train_epoch import _contrastive_terms
from spectralquadnet.losses.contrastive import ProtoNCELoss, SupConLoss

DEVICE = torch.device("cpu")
BATCH, DIM, CLASSES = 32, 64, 4


@pytest.fixture
def embedding_and_labels():
    gen = torch.Generator().manual_seed(0)
    emb = F.normalize(torch.randn(BATCH, DIM, generator=gen), dim=1)
    labels = torch.arange(BATCH) % CLASSES
    return emb, labels


def test_supcon_matches_the_fp32_reference_to_1e_4(embedding_and_labels) -> None:
    """IC-7's validation criterion, verbatim.

    Both sides are given the **same values** — the bf16-rounded embedding a bf16
    backbone would produce — so the comparison isolates what the change actually
    touches: the precision of the similarity matrix and its ``exp(·/τ)``
    reduction. The embedding's own rounding is inherent to training under
    autocast at all and is not what IC-7 is about.
    """
    emb, labels = embedding_and_labels
    supcon = SupConLoss(temperature=0.10)
    emb_bf16 = emb.to(torch.bfloat16)

    reference = float(supcon(emb_bf16.float(), labels))
    with autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
        through_helper, _ = _contrastive_terms(emb_bf16, labels, supcon, None)

    assert abs(float(through_helper) - reference) < 1e-4


def test_the_contrastive_term_is_computed_in_fp32_inside_an_autocast_region(
    embedding_and_labels,
) -> None:
    """The point of the whole change: fp32 where it matters, bf16 everywhere else."""
    emb, labels = embedding_and_labels
    supcon = SupConLoss(temperature=0.10)

    with autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
        loss, _ = _contrastive_terms(emb.to(torch.bfloat16), labels, supcon, None)

    assert loss.dtype == torch.float32


def test_a_bf16_similarity_matrix_would_have_been_materially_wrong(
    embedding_and_labels,
) -> None:
    """Why the fp32 region is not defensive tidiness.

    At τ=0.10 the similarity logits reach ±10 and the reduction runs over 32
    terms; bf16's 8 mantissa bits put a relative error of ~4e-3 on each, which
    is the same order as the logit gaps the loss ranks.
    """
    emb, labels = embedding_and_labels
    supcon = SupConLoss(temperature=0.10)
    emb_bf16 = emb.to(torch.bfloat16)

    # Same input values on both sides; only the matrix arithmetic differs.
    fp32_matrix = float(supcon(emb_bf16.float(), labels))
    bf16_matrix = float(supcon(emb_bf16, labels))

    assert abs(bf16_matrix - fp32_matrix) > 1e-4, (
        "if a bf16 similarity matrix were this accurate the fp32 region would be "
        "unnecessary; it is not, which is why the region is confined rather than removed"
    )


def test_both_terms_are_returned_and_absent_modules_contribute_zero(
    embedding_and_labels,
) -> None:
    emb, labels = embedding_and_labels
    sc, pt = _contrastive_terms(emb, labels, SupConLoss(0.1), ProtoNCELoss(0.1))
    assert float(sc) != 0.0 and float(pt) != 0.0

    sc_only, none_pt = _contrastive_terms(emb, labels, SupConLoss(0.1), None)
    assert float(sc_only) != 0.0
    assert none_pt == 0.0, "a plain 0.0 so the caller's arithmetic needs no branch"

    nothing = _contrastive_terms(emb, labels, None, None)
    assert nothing == (0.0, 0.0)


def test_the_helper_is_a_no_op_wrapper_outside_autocast(embedding_and_labels) -> None:
    """Entering `autocast(enabled=False)` outside an autocast region changes nothing,
    so the Stage-3 SAM loop (always fp32) is unaffected."""
    emb, labels = embedding_and_labels
    supcon = SupConLoss(temperature=0.10)
    direct = float(supcon(emb, labels))
    through, _ = _contrastive_terms(emb, labels, supcon, None)
    assert float(through) == pytest.approx(direct, abs=0.0, rel=0.0)


def test_the_train_loop_no_longer_disables_amp_for_a_supcon_epoch() -> None:
    """The condition itself, read off the source rather than inferred from timing."""
    import inspect

    from spectralquadnet.engine import train_epoch

    source = inspect.getsource(train_epoch.train_one_epoch)
    assert "use_amp = scaler is not None" in source
    assert "(supcon is None) and (scaler is not None)" not in source
