"""Checkpoint compatibility with real trained weights.

The **highest-priority** guarantee for this codebase: real trained weights
exist, and ``save_ckpt``/``load_ckpt`` persist plain tensors keyed by
attribute path. The only way to break these checkpoints is to rename an
``nn.Module`` attribute inside ``SpectralQuadNet.__init__``.

This module loads all three real checkpoints from ``outputs/output_v12_spa40/``
into a freshly constructed model with ``strict=True`` and asserts zero
missing and zero unexpected keys — for the ``"model"`` bundle *and* the ``"ema"``
shadow, since ``ModelEMA.state_dict()`` is the shadow model's state dict and
inherits the same constraint.

Skipped (not failed) when the checkpoints are absent: they are 63 MB each and
``outputs/`` is gitignored.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.regression, pytest.mark.requires_dataset]

# The 14 top-level ``nn.Module`` attribute names that must never be renamed.
SACRED_ATTRIBUTES = {
    "se",
    "wl_pe_cnn",
    "branch_a",
    "branch_b",
    "branch_c",
    "branch_d",
    "cross_interaction",
    "aux_head_a",
    "aux_head_b",
    "aux_head_c",
    "aux_head_d",
    "embed_net",
    "linear_head",
    "arcface_head",
}

STAGES = (1, 2, 3)


def _load_bundle(path):
    return torch.load(path, map_location="cpu", weights_only=False)


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("key", ["model", "ema"])
def test_checkpoint_loads_strict(stage, key, checkpoint_paths, seeded_model):
    """``load_state_dict(strict=True)`` reports zero missing/unexpected keys."""
    bundle = _load_bundle(checkpoint_paths[stage])
    assert key in bundle, f"stage {stage} bundle has no '{key}' entry"

    result = seeded_model.load_state_dict(bundle[key], strict=True)

    assert not result.missing_keys, (
        f"stage {stage} '{key}': model expects {len(result.missing_keys)} keys the "
        f"checkpoint lacks — first few: {result.missing_keys[:8]}"
    )
    assert not result.unexpected_keys, (
        f"stage {stage} '{key}': checkpoint carries {len(result.unexpected_keys)} keys "
        f"the model has no home for — first few: {result.unexpected_keys[:8]}"
    )


@pytest.mark.parametrize("stage", STAGES)
def test_checkpoint_tensor_shapes_match(stage, checkpoint_paths, seeded_model):
    """Every tensor's shape and dtype survives the move.

    ``strict=True`` already raises on a shape mismatch, but it aborts at the first
    one. This reports the complete list, which is what you want when a config
    value (band count, class count, ``subcenter_K``) has drifted.
    """
    ckpt = _load_bundle(checkpoint_paths[stage])["model"]
    current = seeded_model.state_dict()

    mismatches = [
        f"{name}: checkpoint {tuple(t.shape)}/{t.dtype} vs model "
        f"{tuple(current[name].shape)}/{current[name].dtype}"
        for name, t in ckpt.items()
        if name in current and (t.shape != current[name].shape or t.dtype != current[name].dtype)
    ]
    assert not mismatches, "shape/dtype drift:\n  " + "\n  ".join(mismatches)


@pytest.mark.parametrize("stage", STAGES)
def test_sacred_attribute_names_present(stage, checkpoint_paths, seeded_model):
    """The 14 pinned attribute names are still exactly the model's top level."""
    ckpt_tops = {k.split(".")[0] for k in _load_bundle(checkpoint_paths[stage])["model"]}
    model_tops = {k.split(".")[0] for k in seeded_model.state_dict()}

    assert ckpt_tops == SACRED_ATTRIBUTES
    assert model_tops == SACRED_ATTRIBUTES


@pytest.mark.parametrize("stage", STAGES)
def test_bundle_schema_unchanged(stage, checkpoint_paths):
    """``save_ckpt``'s bundle keys are intact.

    Pins the schema ``engine/checkpoint.py`` must keep producing so the
    existing artifacts stay loadable.
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


def test_loaded_weights_change_the_output(checkpoint_paths, seeded_model, synthetic_batch):
    """Loading real weights actually replaces the initialised ones.

    Cheap guard against a vacuous pass: if ``load_state_dict`` silently no-opped,
    every assertion above would still succeed while the model kept its random
    init.
    """
    with torch.no_grad():
        before = seeded_model(synthetic_batch).clone()

    seeded_model.load_state_dict(_load_bundle(checkpoint_paths[3])["model"], strict=True)
    seeded_model.eval()

    with torch.no_grad():
        after = seeded_model(synthetic_batch)

    assert not torch.allclose(before, after), "trained weights produced identical logits"
