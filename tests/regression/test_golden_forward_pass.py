"""Golden forward-pass regression — REFACTOR_PLAN.md §3.2.2.

The **primary numerical gate** for Phases 2-3. The golden files were captured
from the pre-refactor ``hsi_training.py`` at SHA ``886560f`` by
``scripts/capture_golden.py``; see ``golden/README.md`` for the exact procedure
and provenance.

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
    assert np.allclose(actual, golden_logits, atol=1e-6), (
        f"logits drifted from the pre-refactor baseline: max |Δ| = {max_abs:.3e}"
    )


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
