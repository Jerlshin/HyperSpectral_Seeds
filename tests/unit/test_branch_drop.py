"""``model.branch_drop_prob`` reaches the forward pass (**T1-6** / N-1d).

The config key was assigned to ``self.branch_drop_prob`` and then ignored:
``forward`` built its own literal ``[0.0, 0.0, 0.30, 0.20]``. Stage 3 sets
``branch_drop_prob = 0.0`` at entry expecting branch masking to be off, and it
was not — every Stage-3 batch still dropped branch C 22.5% of the time and
branch D 15% (IMPROVEMENT_PLAN §2.2.7, §2.7, 0-J).

The fix scales a fixed *profile* by the config value, so the per-branch ratio
stays a property of the architecture and the strength becomes a property of the
config. ``test_shipped_config_reproduces_the_reference_vector`` pins that this
is numerically inert at the shipped ``0.20`` — Stage 1's golden training step
depends on it, down to the RNG draws.

Also carries §4.3's ``test_config_keys_are_wired``, as an *inventory*: every
key is classified, and the inventory is asserted to be exhaustive. **Tier 3
emptied the dead list.** ``specf_drop`` and ``specf_patch`` reached Branch D
(T3-3), ``wl_embed_dim`` became κ_φ's Fourier width (T3-5), and ``fusion_heads``
was deleted outright (T3-4) because the gated bilinear fusion that replaced the
Perceiver has no attention for a head count to describe. The companion
``tests/unit/test_config_wiring.py`` is the stronger form: it perturbs each key
and watches the forward pass react, which an inventory cannot do.
"""

from __future__ import annotations

import pytest
import torch

from spectralquadnet.models.spectral_quadnet import BRANCH_DROP_PROFILE, SpectralQuadNet

pytestmark = pytest.mark.regression

BATCH = 2

#: Config keys that reach a module today. ``branch_drop_prob`` joined this list
#: in Tier 1, the three ``subcenter_*`` keys in Tier 2 (HD-2 / T2-9), and the
#: last three of 0-J's dead keys plus eight new ones in Tier 3.
WIRED_KEYS = (
    "branch_drop_prob",
    "subcenter_K",
    "subcenter_tau_init",
    "subcenter_tau_final",
    "subcenter_balance_weight",
    "aux_head_hidden",
    "specf_dim",
    "specf_heads",
    "specf_layers",
    "fusion_drop",
    # ── retired from KNOWN_DEAD_KEYS by Tier 3 ────────────────────────
    "specf_drop",  # T3-3 / N-1b
    "specf_patch",  # T3-3 / BR-4(iii) — now the λ-uniform token count
    "wl_embed_dim",  # T3-5 / N-1c — now κ_φ's Fourier-feature width
    # ── new in Tier 3 ─────────────────────────────────────────────────
    "grid_size_a",  # T3-6 / BR-2
    "grid_size_d",  # T3-6 / BR-2
    "index_bank_size",  # T3-1 / BR-1(i)
    "continuum_depths",  # T3-1 / BR-1(ii)
    "n_morphometrics",  # T3-1, T3-4 / P-4
    "stem_channels",  # T3-2 / BR-3
    "fusion_rank",  # T3-4 / FU-1(b)
    "fusion_gate_hidden",  # T3-4 / FU-1(b), FU-2
)

#: Keys accepted by the schema that never reach the module they name. 0-J found
#: four; Tier 3 wired three and deleted the fourth, so this is now empty and the
#: test below asserts that it stays empty. An entry here is a defect with an
#: owner, never a permanent category.
KNOWN_DEAD_KEYS: tuple[str, ...] = ()


@pytest.fixture
def train_model(cfg, physical_wl, silence_dropout):
    """A train-mode model whose only remaining randomness is the branch mask."""
    torch.manual_seed(0)
    model = SpectralQuadNet.from_config(cfg, physical_wl).to("cpu")
    silence_dropout(model)
    return model.train()


@pytest.fixture(scope="module")
def batch(cfg) -> torch.Tensor:
    gen = torch.Generator().manual_seed(7)
    return torch.randn(BATCH, cfg.data.num_bands, 64, 64, generator=gen)


def main_logits(model: SpectralQuadNet, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    return out["main"] if isinstance(out, dict) else out


# ══════════════════════════════════════════════════════════════════════
#  The drop vector
# ══════════════════════════════════════════════════════════════════════


def test_shipped_config_reproduces_the_reference_vector(cfg) -> None:
    """``0.20 * profile`` is bit-identical to the literal it replaces.

    Not merely close: the vector is compared against ``torch.rand``, so a
    one-ULP difference would change which branches get dropped and invalidate
    the Stage-1 golden digests.
    """
    scaled = cfg.model.branch_drop_prob * torch.tensor(BRANCH_DROP_PROFILE)

    assert torch.equal(scaled, torch.tensor([0.15, 0.15, 0.0, 0.15]))


def test_the_profile_protects_the_only_irreplaceable_branch() -> None:
    """M-5 closed (BR-3 / T3-2): C is never dropped, A/B/D are, at equal rates.

    §2.2.7's finding was that the reference profile was backwards. It dropped
    branch C at 0.30 and D at 0.20 and never touched A or B — that is, it
    regularised hardest against the one branch no other branch can reconstruct,
    and not at all against the two that were near-duplicates of each other
    (§2.2.2). §3.3 BR-3 prescribes ``(0.15, 0.15, 0, 0.15)``.
    """
    assert BRANCH_DROP_PROFILE[2] == 0.0
    assert BRANCH_DROP_PROFILE[0] == BRANCH_DROP_PROFILE[1] == BRANCH_DROP_PROFILE[3] > 0.0


# ══════════════════════════════════════════════════════════════════════
#  T1-6's validation criterion
# ══════════════════════════════════════════════════════════════════════


def test_zero_branch_drop_gives_a_deterministic_train_mode_forward(train_model, batch) -> None:
    """The criterion: Stage 3's ``branch_drop_prob = 0`` must actually disable masking."""
    train_model.branch_drop_prob = 0.0

    first = main_logits(train_model, batch)
    for _ in range(3):
        assert torch.equal(main_logits(train_model, batch), first)


def test_zero_branch_drop_consumes_no_rng(train_model, batch) -> None:
    """With masking off, the forward draws nothing — so it cannot perturb the stream.

    A ``rand`` that is drawn and then compared against zero would still pass the
    determinism test above while shifting every downstream draw.
    """
    train_model.branch_drop_prob = 0.0

    before = torch.random.get_rng_state()
    main_logits(train_model, batch)
    assert torch.equal(torch.random.get_rng_state(), before)


def test_nonzero_branch_drop_still_masks(train_model, batch) -> None:
    """The other half: a non-zero probability must reach the mask.

    At ``p = 1.0`` the profile puts branches C and D past certain dropping, so
    exactly one of them survives — via the safe-index rescue — and the result
    cannot equal the unmasked forward.
    """
    train_model.branch_drop_prob = 0.0
    unmasked = main_logits(train_model, batch)

    train_model.branch_drop_prob = 1.0
    torch.manual_seed(0)
    assert not torch.equal(main_logits(train_model, batch), unmasked)


def test_eval_mode_never_masks(train_model, batch) -> None:
    """Inference is unaffected at any probability — masking is a training-only device."""
    train_model.eval()
    train_model.branch_drop_prob = 1.0
    masked_prob = main_logits(train_model, batch)

    train_model.branch_drop_prob = 0.0
    assert torch.equal(main_logits(train_model, batch), masked_prob)


def test_explicit_branch_mask_still_overrides(train_model, batch) -> None:
    """``compute_branch_influence`` passes ``branch_mask``, which must keep priority.

    With an explicit all-ones mask the forward is deterministic even at
    ``branch_drop_prob = 1.0``, and equal to the unmasked ``p = 0`` forward.
    """
    keep_all = torch.ones(4)
    train_model.branch_drop_prob = 0.0
    unmasked = main_logits(train_model, batch)

    train_model.branch_drop_prob = 1.0
    forced = train_model(batch, branch_mask=keep_all)["main"]

    assert torch.equal(forced, train_model(batch, branch_mask=keep_all)["main"])
    assert torch.equal(forced, unmasked)


# ══════════════════════════════════════════════════════════════════════
#  Config wiring inventory (§4.3 `test_config_keys_are_wired`, Tier-1 slice)
# ══════════════════════════════════════════════════════════════════════


def test_branch_drop_prob_is_read_from_the_config(cfg, physical_wl) -> None:
    torch.manual_seed(0)
    model = SpectralQuadNet.from_config(cfg, physical_wl)

    assert model.branch_drop_prob == cfg.model.branch_drop_prob


def test_the_wired_config_keys_reach_their_modules(cfg, physical_wl) -> None:
    """Each of these names a value observable on the constructed module."""
    torch.manual_seed(0)
    model = SpectralQuadNet.from_config(cfg, physical_wl)

    assert model.branch_drop_prob == cfg.model.branch_drop_prob
    assert cfg.model.subcenter_K == model.arcface_head.K
    assert model.aux_head_a.net[0].out_features == cfg.model.aux_head_hidden
    assert model.branch_d.tokenizer.norm.num_channels == cfg.model.specf_dim
    # HD-2 / T2-9: the head starts at the soft end of the tau anneal, and both
    # endpoints plus the balance weight are read from the config.
    # Branch D splits `specf_layers` across two factorised stacks — `n_layers //
    # 2` spectral blocks and `n_layers // 2` spatial ones — so the key is wired
    # to the total, not to either list's length.
    n_blocks = len(model.branch_d.spectral_blocks) + len(model.branch_d.spatial_blocks)
    assert n_blocks == cfg.model.specf_layers
    assert model.branch_d.spectral_blocks[0].attn.num_heads == cfg.model.specf_heads
    assert model.arcface_head.tau == cfg.model.subcenter_tau_init
    assert cfg.model.subcenter_tau_final < cfg.model.subcenter_tau_init
    assert cfg.model.subcenter_balance_weight > 0.0
    # ── Tier 3 ────────────────────────────────────────────────────────
    assert model.branch_b.index_bank.n_indices == cfg.model.index_bank_size
    assert model.branch_b.continuum.n_depths == cfg.model.continuum_depths
    assert model.branch_b.n_morph == cfg.model.n_morphometrics
    assert model.morphology_embed.n_morph == cfg.model.n_morphometrics
    assert model.branch_c.stem.fold[0].out_channels == cfg.model.stem_channels
    assert model.cross_interaction.rank == cfg.model.fusion_rank
    assert model.cross_interaction.modality_gate[0].out_features == cfg.model.fusion_gate_hidden
    assert model.grid_a == cfg.model.grid_size_a
    assert model.grid_d == cfg.model.grid_size_d
    assert set(WIRED_KEYS).isdisjoint(KNOWN_DEAD_KEYS)


def test_every_model_config_key_is_classified(cfg) -> None:
    """§4.3's ``test_config_keys_are_wired``: no key escapes the inventory.

    The failure mode this catches is a *new* key added to the schema and to no
    module — which is how all four of the current dead ones arrived.
    """
    assert set(cfg.model) == set(WIRED_KEYS) | set(KNOWN_DEAD_KEYS)


def test_the_balance_weight_reaches_the_loss(cfg, physical_wl) -> None:
    """``subcenter_balance_weight`` is read by the epoch loops, not by the module.

    So "wired" has to mean something different for it: the model must *produce*
    the term the loops weight, or the key would be dead in the way that
    matters.
    """
    torch.manual_seed(0)
    model = SpectralQuadNet.from_config(cfg, physical_wl).train()
    x = torch.randn(BATCH, cfg.data.num_bands, 64, 64)

    out = model(x, labels=torch.tensor([0, 1]), arc_m=0.0)

    assert "balance" in out


def test_no_model_config_key_is_dead(cfg, physical_wl) -> None:
    """0-J's four dead keys, closed. The inverse of the test this replaces.

    Until Tier 3 this file asserted the *defect* — that ``specf_drop``,
    ``specf_patch``, ``wl_embed_dim`` and ``fusion_heads`` were accepted by the
    schema and reached nothing — so that fixing one would fail the suite and
    force the key to move lists. It has. Three are wired and the fourth is
    deleted, and the assertion is now that the dead list is empty, which is the
    property §2.7 asked for in the first place.
    """
    torch.manual_seed(0)
    model = SpectralQuadNet.from_config(cfg, physical_wl)

    assert KNOWN_DEAD_KEYS == ()
    assert "fusion_heads" not in cfg.model, "T3-4 deleted it; nothing can consume a head count"

    # N-1b closed — Branch D's dropout is the config's, not a literal 0.10.
    assert model.branch_d.spectral_blocks[0].drop.p == pytest.approx(cfg.model.specf_drop)

    # BR-4(iii) — `specf_patch` sets the λ-uniform token count.
    assert model.branch_d.windows.n_tokens == cfg.data.num_bands // (cfg.model.specf_patch // 2)

    # FE-1 — `wl_embed_dim` is κ_φ's Fourier-feature width, in both users.
    assert model.branch_a.stem[0].features.n_freq == cfg.model.wl_embed_dim
    assert model.branch_d.lambda_bias.features.n_freq == cfg.model.wl_embed_dim
