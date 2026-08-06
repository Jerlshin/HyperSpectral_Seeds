"""Golden forward-pass and training-step regression — REFACTOR_PLAN.md §3.2.2.

The **primary numerical gate** for Phases 2-3, covering both of §3.2.2's
artifacts: the eval-mode logits (Phase 2) and the Stage-1 epoch-1 loss
(Phase 3). Both were captured from the pre-refactor ``hsi_training.py`` at SHA
``886560f`` by ``scripts/capture_golden.py``; see ``golden/README.md`` for the
exact procedures and provenance.

If one of these tests fails, the refactor changed the model's numerics. The fix
is to find the drift — never to regenerate the golden files.
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
    """Eval-mode logits reproduce the pre-refactor values within ``atol=1e-6``.

    The tolerance is §3.2.2's: float32 matmul order is unchanged because no math
    moved, so in practice the difference is exactly zero — but 1e-6 leaves room
    for a different BLAS build without letting a real regression through.
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
    """Every initialised tensor hashes to its pre-refactor value.

    Stronger than the forward pass for §3.6's purposes: it fails on a *single*
    reordered sub-module construction, which a 4-sample forward could average
    into invisibility. The golden digests cover all 352 state-dict entries.
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
    """The Stage-1 epoch-1 loss reproduces the pre-refactor value **exactly**.

    §3.2.2 asks for exact equality here, not a tolerance: nothing in the loop
    reorders a float operation, so any difference at all is a behavioural
    change. The accuracy is compared the same way.
    """
    assert stage1_train_step["loss"] == golden_stage1_loss["loss"], (
        f"Stage-1 epoch loss drifted: {stage1_train_step['loss']!r} != "
        f"{golden_stage1_loss['loss']!r}"
    )
    assert stage1_train_step["acc"] == golden_stage1_loss["acc"]


def test_stage1_epoch_weights_match_golden(stage1_train_step, golden_stage1_loss):
    """Post-step model and EMA weights hash to their pre-refactor values.

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
    these five keys by name, so a silent contract change here would only surface
    much later, in Phase 3.
    """
    seeded_model.train()
    with torch.no_grad():
        out = seeded_model(synthetic_batch)

    assert set(out) == {"main", "aux_a", "aux_b", "aux_c", "aux_d"}
    for key, value in out.items():
        assert value.shape == (synthetic_batch.shape[0], cfg.data.num_classes), key
