"""Sub-centre assignment, balancing and seeding (**T2-9** / HD-2, M-8).

Carries §4.3's ``test_no_dead_subcentres``.

The head gives each class ``K`` sub-centres and scores the class by
``max_k cos(e, W_ck)``. Under a hard max, a sub-centre that never wins never
appears in the loss, so it never receives gradient, so it never moves — and the
initialisation this replaces set every sub-centre to one vector plus ``0.01*k``
noise, i.e. two random decoys 9-18 degrees from a real prototype. Phase 0's
0-I counted the result. Three changes, each addressing one link in that chain:

* **(i) soft-to-hard assignment.** ``tau * logsumexp(cos_k / tau)``, annealed
  ``0.20 -> 0.02``. Every sub-centre receives gradient while ``tau`` is large;
  at ``tau -> 0`` the pooling is the ``max`` the head is defined on.
* **(ii) a load-balancing term.** ``sum_c KL(pi_c || uniform)`` — nothing else
  in the objective rewards using more than one sub-centre per class.
* **(iii) spherical k-means seeding.** Real cluster centres start inside their
  own class's data, so every sub-centre wins something on step one.

HD-2(iv) — making ``K`` per-class by silhouette or eigengap — is deliberately
out of scope: §4.2's T2-9 lists (i)-(iii) only, and a ragged ``K`` changes the
weight matrix's shape and so the checkpoint schema again.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from spectralquadnet.models.heads import AdaptiveSubcenterArcFaceHead, _spherical_kmeans

pytestmark = pytest.mark.regression

#: The head's real embedding width. It matters here rather than being incidental:
#: ``init_from_linear``'s ``0.01 * k`` noise is a **9 degree** rotation at
#: ``d = 256`` and a 2 degree one at ``d = 16``, and 9 degrees is the figure
#: §2.4.3's dead-sub-centre argument is built on.
DIM, CLASSES, K = 256, 6, 3
PER_CLASS = 60
#: Within-mode scatter, chosen so a class's three modes are separated by much
#: more than their own spread — i.e. the data really does have K modes.
SPREAD = 0.02


def make_head(tau: float = 0.0, **overrides) -> AdaptiveSubcenterArcFaceHead:
    torch.manual_seed(0)
    return AdaptiveSubcenterArcFaceHead(
        in_dim=DIM, num_classes=CLASSES, K=K, s=48.0, tau=tau, **overrides
    )


def clustered_embeddings(seed: int = 0, spread: float = SPREAD):
    """``K`` well-separated blobs per class — data a ``K``-sub-centre head should fit.

    The point of the construction: every class genuinely *has* ``K`` modes, so
    "some sub-centre never wins" is a property of the head, not of the data.
    """
    generator = torch.Generator().manual_seed(seed)
    modes = F.normalize(torch.randn(CLASSES, K, DIM, generator=generator), dim=-1)
    embeddings, labels = [], []
    for c in range(CLASSES):
        for k in range(K):
            noise = torch.randn(PER_CLASS // K, DIM, generator=generator) * spread
            embeddings.append(F.normalize(modes[c, k] + noise, dim=-1))
            labels.append(torch.full((PER_CLASS // K,), c))
    return torch.cat(embeddings), torch.cat(labels)


def legacy_init(head: AdaptiveSubcenterArcFaceHead, prototypes: torch.Tensor) -> None:
    """``init_from_linear``'s scheme, restated: one vector plus ``0.01 * k`` noise."""
    torch.manual_seed(1)
    with torch.no_grad():
        wn = F.normalize(prototypes, dim=1)
        for k in range(head.K):
            head.weight[k :: head.K].copy_(wn + torch.randn_like(wn) * 0.01 * k)


def win_counts(head: AdaptiveSubcenterArcFaceHead, x: torch.Tensor, y: torch.Tensor):
    """``(C, K)`` count of training samples for which each sub-centre wins its class's max."""
    with torch.no_grad():
        sub = head.subcentre_cosines(x)
        own = sub.gather(1, y.view(-1, 1, 1).expand(-1, 1, head.K)).squeeze(1)
        winners = own.argmax(dim=1)
    counts = torch.zeros(head.C, head.K)
    for label, k in zip(y.tolist(), winners.tolist(), strict=True):
        counts[label, k] += 1
    return counts


# ══════════════════════════════════════════════════════════════════════
#  §4.3 · test_no_dead_subcentres
# ══════════════════════════════════════════════════════════════════════


def test_no_dead_subcentres() -> None:
    """§4.3's gate: every ``(c, k)`` wins the max for at least one training sample."""
    x, y = clustered_embeddings()
    head = make_head()

    head.init_subcentres_from_embeddings(x, y)

    counts = win_counts(head, x, y)
    assert int((counts == 0).sum()) == 0, f"{int((counts == 0).sum())} of {CLASSES * K} never win"


def test_the_initialisation_it_replaces_leaves_them_dead() -> None:
    """The companion: jittering one prototype produces sub-centres that never win.

    ``0.01 * k`` noise on a 256-dimensional unit vector is a 9-18 degree
    rotation (§2.4.3), and a decoy that far behind a real prototype loses the
    max on every sample — for the whole run, because losing the max is also
    what keeps it from receiving gradient.
    """
    x, y = clustered_embeddings()
    head = make_head()
    prototypes = torch.stack([F.normalize(x[y == c].mean(0), dim=0) for c in range(CLASSES)])

    legacy_init(head, prototypes)

    counts = win_counts(head, x, y)
    assert int((counts == 0).sum()) > 0, "the legacy scheme should leave dead sub-centres"


def test_the_subcentre_entropy_clears_the_stated_threshold() -> None:
    """T2-9's other criterion: ``pi_{c,k}`` entropy ``> 0.9 log K``."""
    x, y = clustered_embeddings()
    head = make_head()
    head.init_subcentres_from_embeddings(x, y)

    counts = win_counts(head, x, y)
    pi = counts / counts.sum(dim=1, keepdim=True)
    entropy = -(pi.clamp_min(1e-12) * pi.clamp_min(1e-12).log()).sum(dim=1)

    assert float(entropy.min()) > 0.9 * math.log(
        K
    ), f"worst class entropy {float(entropy.min()):.3f} vs {0.9 * math.log(K):.3f}"


# ══════════════════════════════════════════════════════════════════════
#  (i) soft-to-hard assignment
# ══════════════════════════════════════════════════════════════════════


def test_zero_temperature_is_exactly_the_max() -> None:
    """The head must reduce to what it was, or the anneal has no endpoint."""
    head = make_head(tau=0.0)
    sub = torch.randn(5, CLASSES, K)

    assert torch.equal(head.pool_subcentres(sub), sub.max(dim=2).values.clamp(-0.999, 0.999))


def test_a_small_temperature_converges_to_the_max() -> None:
    head = make_head(tau=1e-3)
    sub = torch.rand(5, CLASSES, K) * 0.5

    assert torch.allclose(head.pool_subcentres(sub), sub.max(dim=2).values, atol=1e-2)


def test_every_subcentre_receives_gradient_at_a_soft_temperature() -> None:
    """The mechanism: at ``tau = 0.20`` a losing sub-centre still moves.

    Measured on the *legacy* initialisation, which is the state that has losing
    sub-centres to begin with. Under the hard max their gradient is exactly
    zero — no gradient, no movement, no chance of ever winning — which is what
    makes a dead sub-centre permanently dead. The soft assignment gives every
    one of them a share.
    """
    x, y = clustered_embeddings()
    prototypes = torch.stack([F.normalize(x[y == c].mean(0), dim=0) for c in range(CLASSES)])
    # Two samples per (class, mode) is enough to make the point and keeps the
    # backward cheap; the prototypes above are still fitted on the full set.
    keep = torch.cat([(y == c).nonzero().flatten()[::10] for c in range(CLASSES)])
    x, y = x[keep], y[keep]

    def grad_norms(tau: float) -> torch.Tensor:
        head = make_head(tau=tau).train()
        legacy_init(head, prototypes)
        head.zero_grad()
        # Push every sample away from its own class, so only the assignment
        # decides which sub-centre feels it.
        head(x, y).gather(1, y.view(-1, 1)).sum().backward()
        return head.weight.grad.norm(dim=1).view(CLASSES, K)

    soft, hard = grad_norms(0.20), grad_norms(0.0)

    assert (soft > 0).all(), "a soft assignment must reach every sub-centre"
    assert int((hard == 0).sum()) > 0, "the hard max leaves sub-centres with no gradient"


def test_the_pooled_cosine_stays_in_range() -> None:
    """``tau * logsumexp`` exceeds the max by up to ``tau log K``, which can leave [-1, 1].

    The margin algebra takes ``sqrt(1 - c^2)`` of this, so an out-of-range
    cosine is a NaN a few lines later.
    """
    head = make_head(tau=0.20)
    sub = torch.full((4, CLASSES, K), 0.999)

    pooled = head.pool_subcentres(sub)

    assert float(pooled.max()) <= 1.0
    assert torch.isfinite(pooled).all()


# ══════════════════════════════════════════════════════════════════════
#  (ii) the load-balancing term
# ══════════════════════════════════════════════════════════════════════


def test_a_uniform_assignment_costs_nothing() -> None:
    head = make_head(tau=0.20)
    assign = torch.full((12, K), 1.0 / K)
    y = torch.arange(12) % CLASSES

    assert float(head.balance_loss(assign, y)) == pytest.approx(0.0, abs=1e-6)


def test_a_collapsed_assignment_costs_log_k_per_class() -> None:
    """The worst case: every sample of every class routed to one sub-centre."""
    head = make_head(tau=0.20)
    assign = torch.zeros(12, K)
    assign[:, 0] = 1.0
    y = torch.arange(12) % CLASSES

    loss = float(head.balance_loss(assign, y))

    assert loss == pytest.approx(CLASSES * math.log(K), rel=1e-4)


def test_only_the_classes_present_in_the_batch_are_counted() -> None:
    """``pi`` is undefined for a class with no samples, and a 16x8 batch has 74 of them."""
    head = make_head(tau=0.20)
    assign = torch.zeros(4, K)
    assign[:, 0] = 1.0
    y = torch.zeros(4, dtype=torch.long)

    assert float(head.balance_loss(assign, y)) == pytest.approx(math.log(K), rel=1e-4)


def test_the_balance_term_carries_gradient_back_to_the_weights() -> None:
    """A penalty nothing can act on is telemetry, not a loss."""
    x, y = clustered_embeddings()
    head = make_head(tau=0.20).train()
    head.zero_grad()

    _, assign = head(x[:24], y[:24], return_assign=True)
    head.balance_loss(assign, y[:24]).backward()

    assert head.weight.grad is not None
    assert float(head.weight.grad.abs().sum()) > 0.0


def test_the_assignment_is_a_distribution_over_the_samples_own_class() -> None:
    x, y = clustered_embeddings()
    head = make_head(tau=0.20).train()

    _, assign = head(x[:24], y[:24], return_assign=True)

    assert assign.shape == (24, K)
    assert torch.allclose(assign.sum(dim=1), torch.ones(24), atol=1e-6)


def test_the_hard_assignment_is_one_hot() -> None:
    """At ``tau = 0`` there is nothing left to balance, and the term goes inert by construction."""
    x, y = clustered_embeddings()
    head = make_head(tau=0.0).train()

    _, assign = head(x[:24], y[:24], return_assign=True)

    assert torch.equal(assign.sum(dim=1), torch.ones(24))
    assert set(assign.unique().tolist()) <= {0.0, 1.0}


# ══════════════════════════════════════════════════════════════════════
#  (iii) spherical k-means seeding
# ══════════════════════════════════════════════════════════════════════


def test_kmeans_recovers_planted_clusters() -> None:
    generator = torch.Generator().manual_seed(4)
    modes = F.normalize(torch.randn(K, DIM, generator=generator), dim=-1)
    points = F.normalize(
        torch.cat([m + torch.randn(30, DIM, generator=generator) * SPREAD for m in modes]), dim=-1
    )

    centres = _spherical_kmeans(points, K, iters=25, generator=torch.Generator().manual_seed(0))

    best = (centres @ modes.t()).max(dim=1).values
    assert float(best.min()) > 0.95
    assert torch.allclose(centres.norm(dim=1), torch.ones(K), atol=1e-5)


def test_kmeans_survives_fewer_points_than_centres() -> None:
    """A class with two samples cannot support three prototypes; it must not crash trying."""
    points = F.normalize(torch.randn(2, DIM, generator=torch.Generator().manual_seed(5)), dim=-1)

    centres = _spherical_kmeans(points, K, iters=5, generator=torch.Generator().manual_seed(0))

    assert centres.shape == (K, DIM)
    assert torch.isfinite(centres).all()


def test_seeding_writes_the_right_rows_for_each_class() -> None:
    """Sub-centre ``k`` of class ``c`` is row ``c*K + k`` — the layout ``view(-1, C, K)`` implies."""
    x, y = clustered_embeddings()
    head = make_head()

    head.init_subcentres_from_embeddings(x, y)

    with torch.no_grad():
        rows = F.normalize(head.weight, dim=1).view(CLASSES, K, DIM)
        class_means = torch.stack([F.normalize(x[y == c].mean(0), dim=0) for c in range(CLASSES)])
        for c in range(CLASSES):
            own = (rows[c] @ class_means[c]).mean()
            other = (rows[c] @ class_means[(c + 1) % CLASSES]).mean()
            assert float(own) > float(other), c


def test_seeding_reports_what_it_did() -> None:
    x, y = clustered_embeddings()
    head = make_head()

    stats = head.init_subcentres_from_embeddings(x, y)

    assert stats["classes_seeded"] == CLASSES
    assert stats["min_separation"] < 0.95, "sub-centres of one class must not coincide"


def test_a_class_with_no_samples_keeps_its_initial_weights() -> None:
    x, y = clustered_embeddings()
    keep = y != CLASSES - 1
    head = make_head()
    before = head.weight[(CLASSES - 1) * K :].detach().clone()

    stats = head.init_subcentres_from_embeddings(x[keep], y[keep])

    assert stats["classes_seeded"] == CLASSES - 1
    assert torch.equal(head.weight[(CLASSES - 1) * K :].detach(), before)


def test_seeding_is_reproducible() -> None:
    x, y = clustered_embeddings()
    first, second = make_head(), make_head()

    first.init_subcentres_from_embeddings(x, y, seed=3)
    second.init_subcentres_from_embeddings(x, y, seed=3)

    assert torch.equal(first.weight.detach(), second.weight.detach())
