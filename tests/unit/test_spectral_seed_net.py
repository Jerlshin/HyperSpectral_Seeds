"""IC-10 — ``SpectralSeedNet``: what it is, what it costs, what it refuses to load.

CHANGES §16.2 budgets ≈2.80 M parameters and ≈1.40 GFLOP/sample — 54% of the
parameters and 34% of the compute of the audited model — by keeping exactly two
pathways and relocating Branch A's physics into fixed, zero-parameter features.

These tests pin the three claims that make that a *design* rather than a
guess: the parameter budget, that both pathways actually influence the output,
and that the spectral path really carries the SNV and λ-derivative operators
Branch A was built around.
"""

from __future__ import annotations

import pytest
import torch

from spectralquadnet.engine.checkpoint import (
    ArchitectureMismatchError,
    SchemaTooOldError,
    load_ckpt,
    save_ckpt,
)
from spectralquadnet.models.control import count_dropout_sites, set_dropout
from spectralquadnet.models.ema import ModelEMA
from spectralquadnet.models.registry import (
    ARCHITECTURES,
    build_model,
    count_parameters,
    describe,
    parameter_breakdown,
)
from spectralquadnet.models.spectral_quadnet import SpectralQuadNet
from spectralquadnet.models.spectral_seed_net import SpectralSeedNet
from spectralquadnet.utils.seed import set_seed

BATCH = 3

#: CHANGES §16.2's budget. The tolerance is generous because the exact count
#: depends on `aux_head_hidden` and the class count; what the test defends is
#: the *claim* — roughly half the audited model — not a specific integer.
EXPECTED_PARAMS = 2_800_000
PARAM_TOLERANCE = 0.06


@pytest.fixture
def seed_model(cfg_default, physical_wl):
    set_seed(42)
    return build_model(cfg_default, physical_wl)


@pytest.fixture
def batch(cfg_default):
    gen = torch.Generator().manual_seed(7)
    return torch.randn(BATCH, cfg_default.data.num_bands, 64, 64, generator=gen)


# ══════════════════════════════════════════════════════════════════════
#  Identity and budget
# ══════════════════════════════════════════════════════════════════════


def test_the_registry_offers_both_architectures() -> None:
    assert set(ARCHITECTURES) == {"spectral_quadnet", "spectral_seed_net"}


def test_the_default_config_builds_the_replacement(cfg_default, seed_model) -> None:
    assert cfg_default.model.arch == "spectral_seed_net"
    assert isinstance(seed_model, SpectralSeedNet)


def test_the_audited_config_still_builds_the_control_arm(cfg, physical_wl) -> None:
    """A3 and A8 compare *against* it; deleting it makes the removals unfalsifiable."""
    set_seed(42)
    assert isinstance(build_model(cfg, physical_wl), SpectralQuadNet)


def test_the_parameter_budget_matches_the_design(seed_model) -> None:
    total = count_parameters(seed_model, trainable_only=False)
    assert total == pytest.approx(
        EXPECTED_PARAMS, rel=PARAM_TOLERANCE
    ), f"CHANGES §16.2 budgets ≈{EXPECTED_PARAMS:,}; this model has {total:,}"


def test_it_is_about_half_the_audited_model(seed_model, cfg, physical_wl) -> None:
    set_seed(42)
    audited = count_parameters(build_model(cfg, physical_wl), trainable_only=False)
    ratio = count_parameters(seed_model, trainable_only=False) / audited
    assert 0.45 < ratio < 0.60, f"expected ~54% of the audited model, got {ratio:.1%}"


def test_the_spatial_path_dominates_the_budget(seed_model) -> None:
    """The joint spectral-spatial operator is the model; the rest is cheap."""
    breakdown = parameter_breakdown(seed_model)
    assert breakdown["spatial"] / breakdown["total"] > 0.7
    # 1.8% of parameters carrying the chemometric signal — CHANGES §5.1's
    # "by far the best cost/benefit ratio in the network".
    assert breakdown["spectral"] / breakdown["total"] < 0.10
    assert breakdown["se"] == 6, "MaskedSpectralECA is six parameters. Free. Kept."


def test_the_removed_branches_are_absent(seed_model) -> None:
    names = dict(seed_model.named_children())
    for gone in ("branch_a", "branch_d", "cross_interaction", "morphology_embed"):
        assert gone not in names


# ══════════════════════════════════════════════════════════════════════
#  Forward contract — the shape every caller in engine/ branches on
# ══════════════════════════════════════════════════════════════════════


def test_eval_mode_returns_bare_logits(seed_model, batch, cfg_default) -> None:
    seed_model.eval()
    with torch.no_grad():
        out = seed_model(batch)
    assert out.shape == (BATCH, cfg_default.data.num_classes)


def test_train_mode_returns_the_main_head_and_one_auxiliary(seed_model, batch) -> None:
    seed_model.train()
    out = seed_model(batch, labels=torch.zeros(BATCH, dtype=torch.long))
    assert set(out) == {"main", "aux_spatial"}, (
        "one auxiliary head, on the spatial path — four heads at 8x the main loss "
        "inverted the audited objective (CHANGES §7.1)"
    )


def test_return_embed_yields_a_normalised_embedding(seed_model, batch) -> None:
    seed_model.eval()
    with torch.no_grad():
        _, emb = seed_model(batch, return_embed=True)
    assert torch.allclose(emb.norm(dim=1), torch.ones(BATCH), atol=1e-5)


def test_both_pathways_influence_the_output(seed_model, batch) -> None:
    """Two pathways that both matter — otherwise one of them is dead weight."""
    seed_model.eval()
    with torch.no_grad():
        full = seed_model(batch)
        no_spatial = seed_model(batch, branch_mask=torch.tensor([0.0, 1.0]))
        no_spectral = seed_model(batch, branch_mask=torch.tensor([1.0, 0.0]))
    assert not torch.allclose(full, no_spatial, atol=1e-4)
    assert not torch.allclose(full, no_spectral, atol=1e-4)


def test_the_pathways_see_different_things(seed_model, batch) -> None:
    """The distinctness claim is about the *inputs*, not the embeddings."""
    inputs = seed_model.pathway_inputs(batch)
    assert inputs["spatial"].shape == batch.shape
    assert inputs["spectral"].shape == (BATCH, batch.shape[1])
    # A per-band global average cannot reconstruct spatial texture, which is
    # the entire reason both pathways exist.
    assert inputs["spatial"].ndim == 4 and inputs["spectral"].ndim == 2


def test_the_spectral_path_carries_branch_a_s_physics(seed_model, cfg_default) -> None:
    """SNV + D1 + D2 as fixed features — the relocation, not a loss.

    The descriptor width must account for `3 * num_bands` from the derivative
    stack, so removing Branch A kept its operators.
    """
    path = seed_model.spectral
    bands = cfg_default.data.num_bands
    expected = path.index_bank.n_indices + path.continuum.n_depths + 3 * bands + path.n_morph
    assert path.in_dim == expected
    # And they are genuinely parameter-free.
    assert sum(p.numel() for p in path.derivatives.parameters()) == 0


def test_morphometrics_enter_exactly_once(seed_model, batch) -> None:
    """They were concatenated into Branch B *and* embedded as a fifth token."""
    seed_model.eval()
    morph = torch.randn(BATCH, seed_model.n_morph)
    with torch.no_grad():
        without = seed_model(batch)
        with_morph = seed_model(batch, morph=morph)
    assert not torch.allclose(without, with_morph, atol=1e-5), "morph must reach the model"
    assert not hasattr(seed_model, "morphology_embed"), "and must not reach it twice"


# ══════════════════════════════════════════════════════════════════════
#  Head simplification (IC-9)
# ══════════════════════════════════════════════════════════════════════


def test_the_head_is_k1_with_no_balance_term(seed_model, cfg_default) -> None:
    assert seed_model.arcface_head.K == 1
    assert float(cfg_default.model.subcenter_balance_weight) == 0.0


def test_the_pairwise_penalty_is_off_by_default(seed_model) -> None:
    assert seed_model.arcface_head.pairwise_delta == 0.0


def test_setting_the_subcentre_temperature_is_a_no_op_at_k1(seed_model) -> None:
    """Kept callable so the stage loops need no hasattr guard."""
    seed_model.set_subcentre_tau(0.2)
    assert seed_model.arcface_head.tau == 0.0


# ══════════════════════════════════════════════════════════════════════
#  IC-14 — dropout reaches every stochastic site
# ══════════════════════════════════════════════════════════════════════


def test_set_dropout_reaches_multihead_attention(cfg, physical_wl) -> None:
    """Branch D's four attention sites ignored the stage schedule entirely (N-2)."""
    set_seed(42)
    quadnet = build_model(cfg, physical_wl)
    counts = count_dropout_sites(quadnet)
    assert counts["attention"] > 0, "Branch D has MultiheadAttention modules"

    quadnet.set_dropout(0.42)
    for module in quadnet.modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            assert module.dropout == 0.42
        if isinstance(module, torch.nn.Dropout):
            assert module.p == 0.42


def test_set_dropout_reports_how_many_sites_it_touched(seed_model) -> None:
    counts = count_dropout_sites(seed_model)
    touched = set_dropout(seed_model, 0.1)
    assert touched == counts["dropout"] + counts["attention"]
    assert touched > 0


# ══════════════════════════════════════════════════════════════════════
#  Checkpoint identity (IC-10)
# ══════════════════════════════════════════════════════════════════════


def test_the_two_architectures_declare_different_schemas() -> None:
    assert SpectralQuadNet.SCHEMA_VERSION == 3
    assert SpectralSeedNet.SCHEMA_VERSION == 4


def test_describe_reads_the_identity_off_the_model(seed_model) -> None:
    identity = describe(seed_model)
    assert identity.arch == "spectral_seed_net"
    assert identity.schema_version == 4


def test_a_bundle_records_the_architecture_that_wrote_it(cfg_default, seed_model, tmp_path) -> None:
    small = cfg_default.copy()
    small.output_dir = str(tmp_path)
    ema = ModelEMA(seed_model, decay=0.99)
    path = tmp_path / "best_stage1.pth"
    save_ckpt(small, str(path), 1, "Stage 1", seed_model, ema, val_f1=0.5, val_acc=0.5)

    bundle = torch.load(path, map_location="cpu", weights_only=False)
    assert bundle["arch"] == "spectral_seed_net"
    assert bundle["schema_version"] == 4


def test_a_quadnet_bundle_is_refused_by_the_new_class(
    cfg, cfg_default, physical_wl, tmp_path
) -> None:
    """IC-10's validation criterion: cross-architecture loads raise, not partially match."""
    small = cfg.copy()
    small.output_dir = str(tmp_path)
    set_seed(42)
    quadnet = build_model(cfg, physical_wl)
    path = tmp_path / "best_stage1.pth"
    save_ckpt(
        small, str(path), 1, "Stage 1", quadnet, ModelEMA(quadnet, 0.99), val_f1=0.5, val_acc=0.5
    )

    set_seed(42)
    seednet = build_model(cfg_default, physical_wl)
    with pytest.raises(ArchitectureMismatchError, match="spectral_quadnet"):
        load_ckpt(str(path), seednet, ModelEMA(seednet, 0.99), torch.device("cpu"))


def test_a_seednet_bundle_round_trips_into_its_own_class(
    cfg_default, physical_wl, tmp_path
) -> None:
    small = cfg_default.copy()
    small.output_dir = str(tmp_path)
    set_seed(42)
    source = build_model(cfg_default, physical_wl)
    path = tmp_path / "best_stage1.pth"
    save_ckpt(
        small, str(path), 1, "Stage 1", source, ModelEMA(source, 0.99), val_f1=0.5, val_acc=0.5
    )

    set_seed(1)  # deliberately different init
    target = build_model(cfg_default, physical_wl)
    load_ckpt(str(path), target, ModelEMA(target, 0.99), torch.device("cpu"))
    for (_, a), (_, b) in zip(
        source.state_dict().items(), target.state_dict().items(), strict=True
    ):
        assert torch.equal(a, b)


def test_a_pre_ic10_bundle_is_still_read_as_quadnet(cfg, physical_wl, tmp_path) -> None:
    """Bundles with no `arch` key predate IC-10 and can only be the audited model."""
    small = cfg.copy()
    small.output_dir = str(tmp_path)
    set_seed(42)
    model = build_model(cfg, physical_wl)
    path = tmp_path / "best_stage1.pth"
    save_ckpt(small, str(path), 1, "Stage 1", model, ModelEMA(model, 0.99), val_f1=0.5, val_acc=0.5)

    bundle = torch.load(path, map_location="cpu", weights_only=False)
    del bundle["arch"]
    torch.save(bundle, path)

    set_seed(1)
    target = build_model(cfg, physical_wl)
    load_ckpt(str(path), target, ModelEMA(target, 0.99), torch.device("cpu"))


def test_a_schema_v2_bundle_is_still_refused_as_too_old(cfg, physical_wl, tmp_path) -> None:
    from spectralquadnet.engine.checkpoint import remap_state_dict

    set_seed(42)
    model = build_model(cfg, physical_wl)
    with pytest.raises(SchemaTooOldError):
        remap_state_dict(model.state_dict(), model.state_dict(), version=2, target_version=3)
