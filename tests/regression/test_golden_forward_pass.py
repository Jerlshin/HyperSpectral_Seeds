"""Golden forward-pass and training-step regression.

The **primary numerical gate**: covers both the eval-mode logits and the
Stage-1 epoch-1 loss, captured by ``scripts/capture_golden.py``.

The values compared here are **schema v3** (``golden/v3/``), captured from the
current architecture at Tier-3 completion. They stopped being the pinned
reference implementation's values at T2-10 and stopped being Tier 2's at T3-1:
each change shifts the global RNG stream ``_init_weights`` reads from, so no
value comparison across versions is meaningful. What survives — and is asserted
below — is the *structural* delta at each step: v2 is v1 minus ``linear_head.*``
plus ``arcface_head.confusion``; v3 changes 185 keys and every one of them falls
under a prefix somebody declared in ``capture_golden.py::V3_DELTA``. The v1
files stay committed and frozen, and ``capture_golden.py`` re-verifies them
against the baseline on every run — that gate has never moved.

If one of these tests fails, something changed the model's numerics. The fix is
to find the drift — never to regenerate the golden files to match. A
regeneration is legitimate only when a declared ``IMPROVEMENT_PLAN.md`` item
changes the architecture on purpose, which is what happened here.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.regression


def _digest(t: torch.Tensor) -> str:
    return hashlib.sha256(np.ascontiguousarray(t.detach().cpu().numpy()).tobytes()).hexdigest()


def test_forward_logits_match_golden(seeded_model, synthetic_batch, golden_logits):
    """Eval-mode logits reproduce the schema-v2 values within ``atol=1e-6``.

    In practice the difference is exactly zero when float32 matmul order is
    unchanged, but 1e-6 leaves room for a different BLAS build without
    letting a real regression through.
    """
    with torch.no_grad():
        out = seeded_model(synthetic_batch)

    assert isinstance(out, torch.Tensor), (
        "eval-mode forward must return a plain tensor; a dict means .eval() did not "
        "take effect and the auxiliary-head branch ran instead"
    )
    actual = out.numpy()

    assert actual.shape == golden_logits.shape
    assert actual.dtype == golden_logits.dtype

    max_abs = float(np.max(np.abs(actual - golden_logits)))
    within_tolerance = np.allclose(actual, golden_logits, atol=1e-6)
    assert within_tolerance, f"logits drifted from the baseline: max |Δ| = {max_abs:.3e}"


def test_weight_init_is_bit_identical(seeded_model, golden_init_digests):
    """Every initialised tensor hashes to its schema-v2 reference value.

    Stronger than the forward pass: it fails on a *single* reordered
    sub-module construction, which a 4-sample forward could average into
    invisibility. The golden digests cover every state-dict entry.
    """
    actual = {name: _digest(t) for name, t in seeded_model.state_dict().items()}
    expected = {k: v for k, v in golden_init_digests.items() if k != "__combined__"}

    assert actual.keys() == expected.keys(), (
        f"state-dict keys changed — missing: {sorted(expected.keys() - actual.keys())[:5]}, "
        f"unexpected: {sorted(actual.keys() - expected.keys())[:5]}"
    )

    drifted = sorted(k for k in expected if actual[k] != expected[k])
    assert not drifted, (
        f"{len(drifted)}/{len(expected)} initialised tensors differ from the baseline "
        f"(construction order or an init call changed). First few: {drifted[:8]}"
    )


def test_forward_is_deterministic_in_eval_mode(seeded_model, synthetic_batch):
    """Two eval-mode passes agree exactly — no dropout, no ``torch.rand`` draws.

    Guards the assumption that makes the golden comparison meaningful in the
    first place: in ``.eval()`` the branch-masking block takes its ``else`` path
    and consumes no randomness.
    """
    with torch.no_grad():
        a = seeded_model(synthetic_batch)
        b = seeded_model(synthetic_batch)
    assert torch.equal(a, b)


def test_stage1_epoch_loss_matches_golden(stage1_train_step, golden_stage1_loss):
    """The Stage-1 epoch-1 loss reproduces the reference value **exactly**.

    Exact equality, not a tolerance: nothing in the loop reorders a float
    operation, so any difference at all is a behavioural change. The
    accuracy is compared the same way.
    """
    assert stage1_train_step["loss"] == golden_stage1_loss["loss"], (
        f"Stage-1 epoch loss drifted: {stage1_train_step['loss']!r} != "
        f"{golden_stage1_loss['loss']!r}"
    )
    assert stage1_train_step["acc"] == golden_stage1_loss["acc"]


def test_stage1_epoch_weights_match_golden(stage1_train_step, golden_stage1_loss):
    """Post-step model and EMA weights hash to their reference values.

    This is the half of the gate that actually bites. A single loss scalar says
    nothing about what the *update* did, so it would survive a wrong gradient
    clip, a mis-split AdamW weight-decay group, a dropped auxiliary-head term or
    an EMA decay applied at the wrong moment. Hashing both weight sets after the
    epoch catches all four.
    """
    assert stage1_train_step["model_sha256"] == golden_stage1_loss["model_sha256"], (
        "model weights after one epoch differ from the baseline — the optimiser "
        "step, gradient clipping or the loss composition changed"
    )
    assert stage1_train_step["ema_sha256"] == golden_stage1_loss["ema_sha256"], (
        "EMA weights after one epoch differ — the EMA decay schedule or its "
        "update site in the accumulation block changed"
    )


def test_training_mode_returns_auxiliary_logits(seeded_model, synthetic_batch, cfg):
    """Training mode still yields the 4 auxiliary heads alongside the main logits.

    Not a numerical check — a structural one. Stage 1's deep supervision reads
    these five keys by name, so a silent contract change here would only
    surface much later, deep into a training run.
    """
    seeded_model.train()
    with torch.no_grad():
        out = seeded_model(synthetic_batch)

    assert set(out) == {"main", "aux_a", "aux_b", "aux_c", "aux_d"}
    for key, value in out.items():
        assert value.shape == (synthetic_batch.shape[0], cfg.data.num_classes), key


def test_training_mode_adds_the_balance_term_when_labels_are_given(
    seeded_model, synthetic_batch, cfg
):
    """HD-2(ii)'s load-balancing term rides on the same dict (T2-9).

    It appears only with labels — without them the head cannot say which
    class's sub-centres a sample is competing among — which is exactly why the
    mixup path, whose label is a pair, does not get one.
    """
    seeded_model.train()
    labels = torch.arange(synthetic_batch.shape[0])
    with torch.no_grad():
        out = seeded_model(synthetic_batch, labels=labels, arc_m=0.0)

    assert set(out) == {"main", "aux_a", "aux_b", "aux_c", "aux_d", "balance"}
    assert out["balance"].shape == ()
    assert float(out["balance"]) >= 0.0, "a KL divergence cannot be negative"


# ══════════════════════════════════════════════════════════════════════
#  The v1 → v2 schema delta (T2-10)
# ══════════════════════════════════════════════════════════════════════

#: What HD-1 removed and HD-3 added. The same declaration
#: `scripts/capture_golden.py` enforces at capture time, restated here so the
#: committed artifacts are checked even when nobody re-runs the capture.
V2_REMOVED = {"linear_head.2.weight", "linear_head.2.bias"}
V2_ADDED = {"arcface_head.confusion"}


def test_schema_v2_is_v1_minus_the_linear_head(golden_v2_init_digests, golden_v1_init_digests):
    """The v1 and v2 golden key sets differ by exactly the declared delta.

    Both files are frozen, so this is a check on the committed record rather
    than on live code — and that is the point: it is what keeps the historical
    claim "T2-10 removed one head and added one buffer" auditable after the
    code that produced either version is gone.
    """
    v1 = set(golden_v1_init_digests) - {"__combined__"}
    v2 = set(golden_v2_init_digests) - {"__combined__"}

    assert v1 - v2 == V2_REMOVED
    assert v2 - v1 == V2_ADDED


def test_every_v3_key_change_is_declared(golden_init_digests, golden_v2_init_digests):
    """§4.3's structural gate for Tier 3: no key appeared or vanished undeclared.

    The assertion that keeps "the architecture changed" from covering an
    unrelated change. Values are deliberately not compared — rebuilding three
    branches shifts every later ``_init_weights`` draw — so the key set carries
    the whole check, and it is checked against the *declarations* in
    ``capture_golden.py`` rather than against a transcribed list, since a
    transcription could itself carry the drift.
    """
    from capture_golden import V3_DELTA, undeclared

    v2 = set(golden_v2_init_digests) - {"__combined__"}
    v3 = set(golden_init_digests) - {"__combined__"}

    assert not undeclared(v2 - v3), f"undeclared removals: {undeclared(v2 - v3)[:8]}"
    assert not undeclared(v3 - v2), f"undeclared additions: {undeclared(v3 - v2)[:8]}"
    # Non-vacuity: a delta this size is the whole reason the check is by prefix.
    assert len(v2 - v3) > 100 and len(v3 - v2) > 50
    assert len(V3_DELTA) >= 20


def test_the_modules_tier3_did_not_touch_kept_their_key_sets(
    golden_init_digests, golden_v2_init_digests
):
    """The head, the auxiliary heads and ``embed_net`` are key-for-key identical to v2.

    The sharp version of "Tier 3 changed the branches and the fusion, and
    nothing else". §3.8 budgets ``aux_head_*``, ``embed_net`` and
    ``arcface_head`` at no change at all, and a redesign that quietly reshaped
    one of them would still pass the prefix check above.
    """
    untouched = ("arcface_head.", "aux_head_", "embed_net.", "se.", "wl_pe_cnn.")
    v2 = {k for k in golden_v2_init_digests if k.startswith(untouched)}
    v3 = {k for k in golden_init_digests if k.startswith(untouched)}

    assert v2 == v3
    assert len(v2) > 20, "the filter matched almost nothing — the prefixes are stale"
