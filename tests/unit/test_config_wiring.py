"""§4.3's ``test_config_keys_are_wired``, in its strong form — **T3-3, T3-4, T3-5**.

§4.3 calls this "the highest-leverage of these: it would have caught all five
dead paths of §2.7 automatically, and the config round-trip gate that *does*
exist explicitly does not check wiring — only that all 81 keys have a home."

``tests/unit/test_branch_drop.py`` carries the *inventory* form: every key in
``configs/model/*.yaml`` is classified, and the classification is asserted to be
exhaustive. An inventory cannot tell a key that is read from a key that is read
and discarded, which is precisely how ``branch_drop_prob`` stayed dead through
five phases — it was assigned to ``self.branch_drop_prob`` and never used.

This module closes that gap the way §4.3 asks: **perturb each key, rebuild, and
require the model to change.** A key whose value the forward pass does not
depend on cannot pass, whatever it is assigned to on the way through.

Two keys need a different treatment, and both say so where they are handled:
``branch_drop_prob`` is inert in eval mode by design (it is a training-only
device, which is T1-6's whole point), and ``subcenter_balance_weight`` is read
by the epoch loops rather than by the module, so "wired" means the model must
*produce* the term the loops weight.
"""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf, open_dict
from omegaconf.errors import ConfigAttributeError

from spectralquadnet.models.spectral_quadnet import SpectralQuadNet

pytestmark = pytest.mark.regression

BATCH = 2

#: ``key -> an alternative value``. Each must be a *legal* setting, not a
#: sentinel: the point is that the model behaves differently, not that it
#: breaks. Values are chosen to stay inside the shapes the rest of the network
#: expects (256-wide branch outputs, a head that still has sub-centres).
PERTURBATIONS: dict[str, object] = {
    "subcenter_K": 5,
    "subcenter_tau_init": 0.90,
    "aux_head_hidden": 96,
    "wl_embed_dim": 8,
    "specf_patch": 4,
    "specf_dim": 96,
    "specf_heads": 4,
    "specf_layers": 2,
    "grid_size_a": 4,
    "grid_size_d": 2,
    "index_bank_size": 32,
    "continuum_depths": 8,
    "n_morphometrics": 4,
    "stem_channels": 96,
    "fusion_rank": 64,
    "fusion_gate_hidden": 64,
    # ── Added by CHANGES ──────────────────────────────────────────────
    # A5 — the fusion arms. `gate` drops the second-order term; the logits must
    # move, or the 0.50 M parameters the bilinear pool costs are unobservable
    # and A5 could not measure them.
    "fusion_mode": "gate",
    # A3 — dropping a branch removes its parameters, its auxiliary head and its
    # fusion modality, so the forward is genuinely a different function.
    "enabled_branches": ["b", "c"],
    # A10 — the spatial path's capacity lever.
    "spatial_width_mult": 0.5,
}

#: Dropout rates. Inert in ``.eval()`` by definition, so they are probed in
#: train mode from a fixed RNG state instead — which is the only place a
#: dropout rate *can* be observed, and is exactly where N-1b hid: Branch D was
#: constructed with a literal ``dropout=0.10`` while the config said 0.15, and
#: no eval-mode test could ever have noticed.
DROPOUT_KEYS: dict[str, float] = {
    "specf_drop": 0.40,
    "fusion_drop": 0.40,
}

#: Keys whose effect is not observable in an eval-mode forward, with the reason
#: and the test that does cover them. Listed rather than skipped silently.
NOT_FORWARD_OBSERVABLE: dict[str, str] = {
    "branch_drop_prob": (
        "training-only by construction — T1-6's criterion is that eval mode never masks. "
        "Covered by tests/unit/test_branch_drop.py."
    ),
    "subcenter_tau_final": (
        "the endpoint of a schedule the stage loops drive, not a construction-time value. "
        "Covered by tests/unit/test_subcentres.py."
    ),
    "subcenter_balance_weight": (
        "read by the epoch loops, not by the module. Covered by "
        "tests/unit/test_branch_drop.py::test_the_balance_weight_reaches_the_loss."
    ),
    # ── Added by CHANGES ──────────────────────────────────────────────
    "arch": (
        "selects *which* module is built, so it cannot be a perturbation of one. "
        "Covered by tests/unit/test_spectral_seed_net.py."
    ),
    "branch_drop_profile": (
        "training-only, like branch_drop_prob — eval mode never masks. Covered by "
        "tests/unit/test_branch_drop.py."
    ),
    "per_class_margin": (
        "gates whether the stage loop *calls* update_margins_from_pr; the head is "
        "constructed identically either way. Covered by "
        "tests/unit/test_margin_rule.py and A7's arms."
    ),
    "pairwise_penalty": (
        "the penalty applies only in train mode with labels — an eval forward "
        "returns the plain scaled cosine either way, which is the head's design. "
        "Covered by tests/unit/test_config_wiring.py::"
        "test_the_pairwise_penalty_flag_reaches_the_head."
    ),
    "spectral_hidden": (
        "SpectralSeedNet's spectral MLP width — unread by SpectralQuadNet, which is "
        "what this module perturbs. Covered by "
        "tests/unit/test_spectral_seed_net.py::test_the_parameter_budget_matches_the_design."
    ),
    "aux_head_weight": (
        "a loss coefficient read by the stage loop, not a construction-time value. "
        "Covered by tests/unit/test_observability.py and the single-stage smoke run."
    ),
}


def _build(cfg, physical_wl, **overrides):
    """A freshly seeded model built from ``cfg`` with ``model.*`` keys overridden."""
    patched = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    with open_dict(patched):
        for key, value in overrides.items():
            patched.model[key] = value
    torch.manual_seed(0)
    return SpectralQuadNet.from_config(patched, physical_wl).eval()


@pytest.fixture(scope="module")
def probe(cfg) -> torch.Tensor:
    gen = torch.Generator().manual_seed(23)
    x = torch.rand(BATCH, cfg.data.num_bands, 64, 64, generator=gen) + 0.3
    keep = torch.zeros(1, 1, 64, 64)
    keep[..., 14:50, 20:44] = 1.0
    return x * keep


@pytest.fixture(scope="module")
def baseline_logits(cfg, physical_wl, probe) -> torch.Tensor:
    from spectralquadnet.utils.seed import set_seed

    set_seed(0)
    model = _build(cfg, physical_wl)
    with torch.no_grad():
        return model(probe).clone()


# ══════════════════════════════════════════════════════════════════════
#  The traced forward
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("key", sorted(PERTURBATIONS))
def test_changing_the_key_changes_the_forward_pass(key, cfg, physical_wl, probe) -> None:
    """Each key, perturbed one at a time: the logits must move.

    Construction is re-seeded identically on both sides, so a difference cannot
    come from the RNG — only from the key. This is the check that would have
    failed on ``fusion_heads``, ``specf_drop``, ``specf_patch`` and
    ``wl_embed_dim`` from the day each was added.
    """
    torch.manual_seed(0)
    base = _build(cfg, physical_wl)
    torch.manual_seed(0)
    changed = _build(cfg, physical_wl, **{key: PERTURBATIONS[key]})

    with torch.no_grad():
        before, after = base(probe), changed(probe)

    assert not torch.allclose(before, after, atol=1e-6), (
        f"model.{key} = {PERTURBATIONS[key]!r} produced identical logits — "
        "the key is accepted by the schema and reaches nothing (§2.7, N-1)"
    )


def test_every_model_key_is_either_perturbed_or_excused(cfg) -> None:
    """No key escapes: the two lists must together cover ``configs/model/*.yaml``.

    The failure mode is a *new* key added to the schema and to no module, which
    is how all five dead paths of §2.7 arrived. A key that is neither perturbed
    here nor excused with a reason fails this.
    """
    assert set(cfg.model) == set(PERTURBATIONS) | set(DROPOUT_KEYS) | set(NOT_FORWARD_OBSERVABLE)


def test_the_excuses_name_a_test_that_covers_them() -> None:
    """An exclusion has to point somewhere, or it is a hole with a comment on it."""
    for key, reason in NOT_FORWARD_OBSERVABLE.items():
        assert "tests/unit/" in reason, key


def test_the_pairwise_penalty_flag_reaches_the_head(cfg, physical_wl) -> None:
    """IC-9 — the confusion penalty is gated by a flag, not by a leftover delta.

    It applies only in train mode with labels, so it cannot be observed by an
    eval forward. What *is* observable at construction is that the flag decides
    whether the head carries a non-zero ``pairwise_delta`` at all — which is
    what makes A7's fourth arm a one-variable change.
    """
    on = _build(cfg, physical_wl, pairwise_penalty=True)
    off = _build(cfg, physical_wl, pairwise_penalty=False)

    assert on.arcface_head.pairwise_delta == pytest.approx(cfg.stage2.pairwise_margin_delta)
    assert off.arcface_head.pairwise_delta == 0.0

    # And the train-mode logits differ, which is where the term lives.
    x = torch.randn(BATCH, cfg.data.num_bands, 64, 64, generator=torch.Generator().manual_seed(3))
    labels = torch.tensor([0, 1])
    on.arcface_head.set_confusion(torch.rand(cfg.data.num_classes, cfg.data.num_classes))
    with torch.no_grad():
        emb = torch.nn.functional.normalize(torch.randn(BATCH, 256), dim=1)
        on.train(), off.train()
        assert not torch.allclose(
            on.arcface_head(emb, labels, global_m=0.0),
            off.arcface_head(emb, labels, global_m=0.0),
            atol=1e-6,
        )
    del x


# ══════════════════════════════════════════════════════════════════════
#  Non-vacuity
# ══════════════════════════════════════════════════════════════════════


def test_an_unperturbed_rebuild_is_bit_identical(cfg, physical_wl, probe) -> None:
    """The control. Two builds at the same seed with no override must agree exactly.

    Without this, every assertion above would pass on a model whose
    construction merely consumed RNG non-deterministically, and the suite would
    be measuring noise.
    """
    torch.manual_seed(0)
    first = _build(cfg, physical_wl)
    torch.manual_seed(0)
    second = _build(cfg, physical_wl)

    with torch.no_grad():
        assert torch.equal(first(probe), second(probe))


def test_the_composed_config_rejects_a_key_the_schema_does_not_declare(cfg) -> None:
    """The structured schema is what makes the inventory above meaningful.

    If the composed config accepted arbitrary keys, "every key is classified"
    would be a statement about the YAML rather than about the model's
    interface — and ``ModelConfig`` would stop being the list this file
    enumerates.
    """
    with pytest.raises(ConfigAttributeError):
        cfg.model.this_key_does_not_exist = 1


# ══════════════════════════════════════════════════════════════════════
#  Dropout rates — observable only in train mode
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("key", sorted(DROPOUT_KEYS))
def test_changing_a_dropout_rate_changes_a_train_mode_forward(key, cfg, physical_wl, probe) -> None:
    """N-1b's shape, generalised: a rate that reaches nothing is invisible in eval.

    Both forwards start from the same RNG state, so the same *draws* are made;
    only the rate they are compared against differs. A branch constructed with
    a hardcoded rate would produce identical logits here, which is what the
    reference implementation did for four phases.
    """
    torch.manual_seed(0)
    base = _build(cfg, physical_wl).train()
    torch.manual_seed(0)
    changed = _build(cfg, physical_wl, **{key: DROPOUT_KEYS[key]}).train()

    with torch.no_grad():
        torch.manual_seed(7)
        before = base(probe)["main"]
        torch.manual_seed(7)
        after = changed(probe)["main"]

    assert not torch.allclose(before, after, atol=1e-6), (
        f"model.{key} = {DROPOUT_KEYS[key]!r} produced identical train-mode logits — "
        "the rate is accepted by the schema and reaches no module (N-1b)"
    )


@pytest.mark.parametrize("key", sorted(DROPOUT_KEYS))
def test_the_dropout_rate_reaches_a_module(key, cfg, physical_wl) -> None:
    """The structural half: some ``nn.Dropout`` in the model carries the config's value.

    Weaker than the forward check on its own and stronger in one respect — it
    fails on a rate that is passed to a module that then never applies it,
    which a fixed-seed forward could in principle miss.
    """
    model = _build(cfg, physical_wl)
    rates = {m.p for m in model.modules() if isinstance(m, torch.nn.Dropout)}
    rates |= {m.dropout for m in model.modules() if isinstance(m, torch.nn.MultiheadAttention)}

    assert cfg.model[key] in rates
