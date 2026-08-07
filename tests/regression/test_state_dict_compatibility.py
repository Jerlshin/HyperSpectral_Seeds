"""Checkpoint compatibility with real trained weights.

``save_ckpt``/``load_ckpt`` persist plain tensors keyed by attribute path, so
the only way to break a checkpoint is to rename an ``nn.Module`` attribute
inside ``SpectralQuadNet.__init__`` — or to change what the module behind the
name *does*.

Tier 3 did the second thing, deliberately, and this module is where that is
recorded. The three archived bundles in ``outputs/output_v12_spa40/`` are
**schema v1**; the model is **v3**. Through Tier 2 the gap was a migration:
T2-10 (HD-1) removed ``linear_head`` and T2-8 (HD-3) added
``arcface_head.confusion``, both mechanical, and ``remap_state_dict`` closed it
tensor by tensor. Tier 3's gap is not mechanical. BR-1 replaced the nine
moments Branch B read with a 64-index bank, BR-3 replaced Branch C's two 1×1
convolutions with a 3-D stem, BR-4 dropped Branch D from ``d_model = 256`` to
192, and FU-1(b) removed the fusion's attention entirely. A weight trained to
read one of those inputs is not a weight for the other, whatever its shape.

So the property asserted here **inverted**: ``remap_state_dict`` must *refuse*
a pre-Tier-3 bundle, loudly, rather than load two thirds of it and let someone
report the number that comes out. The tests that used to prove the migration
was faithful now prove the refusal is total — and the ones that never depended
on loading into the live model (the bundle's own key schema, model/EMA key
parity, the sacred attribute set) are unchanged, because those are still what
protects the *next* set of checkpoints.

Skipped (not failed) when the checkpoints are absent: they are 63 MB each and
``outputs/`` is gitignored.
"""

from __future__ import annotations

import pytest
import torch

from spectralquadnet.engine.checkpoint import (
    SCHEMA_VERSION,
    SchemaTooOldError,
    remap_state_dict,
)

pytestmark = [pytest.mark.regression, pytest.mark.requires_dataset]

# The 14 top-level ``nn.Module`` attribute names that must never be renamed.
# ``morphology_embed`` is the 14th, added by FU-4 / T3-4.
SACRED_ATTRIBUTES = {
    "se",
    "wl_pe_cnn",
    "branch_a",
    "branch_b",
    "branch_c",
    "branch_d",
    "morphology_embed",
    "cross_interaction",
    "aux_head_a",
    "aux_head_b",
    "aux_head_c",
    "aux_head_d",
    "embed_net",
    "arcface_head",
}

#: Present in every schema-v1 bundle and in no model since HD-1.
RETIRED_BY_HD1 = "linear_head"

#: Added by FU-4 / T3-4; in no bundle written before Tier 3.
ADDED_BY_FU4 = "morphology_embed"

STAGES = (1, 2, 3)


def _load_bundle(path):
    return torch.load(path, map_location="cpu", weights_only=False)


# ══════════════════════════════════════════════════════════════════════
#  The refusal (T3-1 … T3-4)
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("key", ["model", "ema"])
def test_a_pre_tier3_bundle_is_refused(stage, key, checkpoint_paths, seeded_model):
    """Both halves of every archived bundle raise, and the message names the reason.

    The failure mode this rules out is the quiet one: a migration that dropped
    the keys it could not place, loaded the rest strict-free, and produced a
    model that is two thirds trained and one third random. That model would
    score *something*, and the something would be reported.
    """
    bundle = _load_bundle(checkpoint_paths[stage])
    assert key in bundle, f"stage {stage} bundle has no '{key}' entry"
    version = int(bundle.get("schema_version", 1))
    assert version < SCHEMA_VERSION

    with pytest.raises(SchemaTooOldError, match="predates Tier 3"):
        remap_state_dict(bundle[key], seeded_model.state_dict(), version)


@pytest.mark.parametrize("stage", STAGES)
def test_the_refusal_is_not_hiding_a_loadable_checkpoint(stage, checkpoint_paths, seeded_model):
    """The bundles genuinely cannot load — the refusal is a statement of fact.

    Without this, ``SchemaTooOldError`` would be indistinguishable from
    conservatism. ``strict=True`` on the raw v1 state dict must fail on its own,
    and it must fail on *branch* tensors rather than only on the head, which is
    what makes the gap architectural rather than the two-key one Tier 2 bridged.
    """
    raw = _load_bundle(checkpoint_paths[stage])["model"]

    with pytest.raises(RuntimeError) as excinfo:
        seeded_model.load_state_dict(raw, strict=True)

    message = str(excinfo.value)
    assert any(f"branch_{b}" in message for b in "bcd"), message[:400]


def test_the_v1_to_v2_migration_rules_are_still_declared():
    """The v1 → v2 constants stay in the source, and stay unreachable.

    They are the record of what a mechanical migration looked like, and
    ``_MIGRATABLE_THROUGH`` is the one line that decides whether they run. A
    change that quietly re-enabled them would put the loading of a v1 bundle
    back on the table without anyone choosing that.
    """
    from spectralquadnet.engine.checkpoint import (
        _MIGRATABLE_THROUGH,
        _V1_ONLY_PREFIXES,
        _V2_ADDED_BUFFERS,
    )

    assert list(_V1_ONLY_PREFIXES) == [f"{RETIRED_BY_HD1}."]
    assert _V2_ADDED_BUFFERS == ("arcface_head.confusion",)
    assert _MIGRATABLE_THROUGH < SCHEMA_VERSION


# ══════════════════════════════════════════════════════════════════════
#  What the bundles still pin
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("stage", STAGES)
def test_sacred_attribute_names(stage, checkpoint_paths, seeded_model):
    """The pinned attribute names, and exactly how the archived set differs.

    The names are the checkpoint's addressing scheme and are independent of
    what each module computes, so they survive Tier 3 untouched — the model
    gains ``morphology_embed`` and nothing else moves. Asserted against the
    real bundles so "nothing else moved" is checked rather than assumed.
    """
    ckpt_tops = {k.split(".")[0] for k in _load_bundle(checkpoint_paths[stage])["model"]}
    model_tops = {k.split(".")[0] for k in seeded_model.state_dict()}

    assert model_tops == SACRED_ATTRIBUTES
    assert ckpt_tops == (SACRED_ATTRIBUTES | {RETIRED_BY_HD1}) - {ADDED_BY_FU4}


@pytest.mark.parametrize("stage", STAGES)
def test_bundle_schema_unchanged(stage, checkpoint_paths):
    """``save_ckpt``'s bundle keys are intact.

    Pins the schema ``engine/checkpoint.py`` must keep producing. Unaffected by
    Tier 3: the envelope is what makes an archived run *readable* — its epoch,
    its stage, its recorded val F1 — and those stay meaningful even when the
    weights inside no longer fit any live model.
    """
    bundle = _load_bundle(checkpoint_paths[stage])
    required = {"epoch", "stage", "model", "ema", "val_f1", "val_acc", "use_arcface"}
    assert required <= set(bundle), f"missing: {sorted(required - set(bundle))}"


def test_ema_and_model_share_key_structure(checkpoint_paths):
    """``ModelEMA``'s shadow has the same keys as the live model, in every stage.

    This is why the attribute-name constraint applies to the EMA too: the
    shadow is a ``deepcopy`` of the model, so a rename breaks both halves of
    every bundle at once.
    """
    for stage in STAGES:
        bundle = _load_bundle(checkpoint_paths[stage])
        assert set(bundle["model"]) == set(bundle["ema"]), f"stage {stage}"


def test_a_current_bundle_round_trips(seeded_model):
    """A v3 state dict loads back into a v3 model, strict, with the buffers intact.

    The forward-looking half of this file. Everything above is about weights
    that can no longer be used; this is the property the *next* three
    checkpoints depend on, and it needs no dataset to check.
    """
    seeded_model.arcface_head.set_confusion(torch.full((seeded_model.arcface_head.C,) * 2, 0.5))
    fitted = {k: v.clone() for k, v in seeded_model.state_dict().items()}

    passed = remap_state_dict(fitted, fitted, SCHEMA_VERSION)
    result = seeded_model.load_state_dict(passed, strict=True)

    assert passed is fitted, "remap must be a no-op at the current version"
    assert not result.missing_keys and not result.unexpected_keys
    assert passed["arcface_head.confusion"].abs().sum() > 0
