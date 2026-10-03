"""S13 Y2 and Y3 — the two model changes, behind default-off keys (preregistration_s12.json).

* **Y3 lean architecture** — ``model.spectral_descriptor=snv_morph``,
  ``model.spatial_tail_strides=[2,2,2,1]``, ``model.cbam_min_hw=3``: the inert
  spectral blocks are not built (S10 F49), the tail stops at 2 × 2 so no 3 × 3
  tap multiplies only padding (F48), and no CBAM gate runs on a ≤ 2 × 2 map (B8).
* **Y2 masked MixStyle** — ``model.spatial_mixstyle=true``: per-(channel,
  spectral-slice) foreground statistics mixed across the batch after 3-D stem
  blocks 1 and 2, training only.

The defaults must be the shipped network bit for bit; S13's G-neutral gate checks
that end to end against ``413a11e``, these tests check the pieces.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from spectralquadnet.config.compose import load_experiment_config
from spectralquadnet.models.blocks.attention import CBAM
from spectralquadnet.models.branches.spatial_cnn import (
    MIXSTYLE_ALPHA,
    MIXSTYLE_P,
    MIXSTYLE_STAGES,
    MaskedMixStyle,
    resolve_tail_strides,
    tail_map_sides,
)
from spectralquadnet.models.front_end import snv
from spectralquadnet.models.registry import build_model
from spectralquadnet.models.spectral_seed_net import SpectralSeedNet

BANDS = 32
SIDE = 64
LEAN = [
    "model.spectral_descriptor=snv_morph",
    "model.spatial_tail_strides=[2,2,2,1]",
    "model.cbam_min_hw=3",
]


def _model(*overrides: str, input_side: int | None = None) -> SpectralSeedNet:
    cfg = load_experiment_config(overrides=["data=ablation/u430k32_grouped", *overrides])
    torch.manual_seed(0)
    model = build_model(cfg, torch.linspace(0.0, 1.0, BANDS), input_side=input_side)
    assert isinstance(model, SpectralSeedNet)
    return model


def _batch(n: int = 4, seed: int = 0, side: int = SIDE) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    x = torch.rand(n, BANDS, side, side, generator=gen) + 0.1
    x[:, :, : side // 8] = 0.0  # a background band, so masking is exercised
    return x


# ══════════════════════════════════════════════════════════════════════
#  Defaults = the shipped network
# ══════════════════════════════════════════════════════════════════════


def test_the_defaults_are_the_shipped_values() -> None:
    cfg = load_experiment_config(overrides=["data=ablation/u430k32_grouped"])
    assert cfg.model.spectral_descriptor == "full"
    assert list(cfg.model.spatial_tail_strides) == [2, 2, 2, 2]
    assert cfg.model.cbam_min_hw == 0 and cfg.model.spatial_mixstyle is False


def test_the_default_model_is_the_s10_network() -> None:
    model = _model()
    # S10 F48 counted 2,849,478 parameters at k = 32; the tail ends at 1 x 1.
    assert sum(p.numel() for p in model.parameters()) == 2_849_478
    assert model.spatial.tail_strides == (2, 2, 2, 2)
    assert model.spatial.tail_sides == (8, 4, 2, 1)
    assert [type(model.spatial.stages[i]) for i in (1, 3, 5)] == [CBAM, CBAM, CBAM]
    assert model.spatial.stem.mixstyle is None
    assert model.spectral.in_dim == 64 + 16 + 3 * BANDS + 8


def test_input_side_does_not_touch_the_default_construction() -> None:
    a, b = _model(), _model(input_side=16)
    for (ka, va), (kb, vb) in zip(a.state_dict().items(), b.state_dict().items(), strict=True):
        assert ka == kb and torch.equal(va, vb)


# ══════════════════════════════════════════════════════════════════════
#  Y3 — lean architecture
# ══════════════════════════════════════════════════════════════════════


def test_snv_morph_builds_only_snv_and_morph() -> None:
    model = _model("model.spectral_descriptor=snv_morph")
    spec = model.spectral
    assert spec.in_dim == BANDS + 8
    assert not any(hasattr(spec, m) for m in ("index_bank", "continuum", "derivatives"))
    r = torch.rand(3, BANDS) + 0.1
    morph = torch.randn(3, 8)
    feats = spec.features(r, morph)
    assert feats.shape == (3, BANDS + 8)
    assert torch.allclose(feats[:, BANDS:], morph)
    assert torch.equal(feats[:, :BANDS], snv(r))


def test_an_unknown_descriptor_is_refused() -> None:
    with pytest.raises(ValueError, match="spectral_descriptor"):
        _model("model.spectral_descriptor=pca")


@pytest.mark.parametrize("bad", [[2, 2, 2], [2, 2, 2, 3], [0, 2, 2, 2]])
def test_invalid_tail_strides_are_refused(bad: list[int]) -> None:
    with pytest.raises(ValueError, match="spatial_tail_strides"):
        resolve_tail_strides(bad)


def test_map_sides_follow_the_strides() -> None:
    assert tail_map_sides(64, (2, 2, 2, 2)) == (8, 4, 2, 1)
    assert tail_map_sides(64, (2, 2, 2, 1)) == (8, 4, 2, 2)
    assert tail_map_sides(16, (2, 2, 2, 1)) == (2, 1, 1, 1)


def test_the_lean_tail_ends_at_2x2_and_drops_only_the_gate_on_it() -> None:
    model = _model(*LEAN)
    assert model.spatial.tail_sides == (8, 4, 2, 2)
    assert isinstance(model.spatial.stages[1], CBAM) and isinstance(model.spatial.stages[3], CBAM)
    assert isinstance(model.spatial.stages[5], nn.Identity)
    seen: list[torch.Size] = []
    hook = model.spatial.stages[-1].register_forward_hook(lambda m, i, o: seen.append(o.shape))
    model.eval()
    with torch.no_grad():
        model(_batch())
    hook.remove()
    assert tuple(seen[0][-2:]) == (2, 2)


def test_the_lean_tail_has_no_structurally_dead_tap() -> None:
    """F48's defect is gone: every 3 x 3 tap of every tail conv receives a gradient."""
    model = _model(*LEAN).train()
    for seed in (0, 1):
        x = _batch(2, seed)
        x[:] = x.abs() + 0.1  # all foreground: a tap dead here is dead for every input
        out = model(x, labels=torch.tensor([0, 1]), arc_m=0.0)
        (out["main"].logsumexp(1).mean() + out["aux_spatial"].logsumexp(1).mean()).backward()
    dead = {
        name: int((m.weight.grad.abs().sum(dim=(0, 1)) == 0).sum())
        for name, m in model.named_modules()
        if name.startswith("spatial.stages.")
        and isinstance(m, nn.Conv2d)
        and m.weight[0, 0].numel() > 1
        and m.weight.grad is not None
    }
    assert not {k: v for k, v in dead.items() if v}, dead


def test_the_lean_model_is_smaller_by_the_removed_parts_only() -> None:
    full, lean = _model(), _model(*LEAN)
    n = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    assert n(lean) < n(full)
    assert n(lean.spatial.stem) == n(full.spatial.stem)  # the stem is untouched
    assert n(lean.embed_net) == n(full.embed_net) and n(lean.fuse) == n(full.fuse)


def test_a_cbam_placement_refuses_a_patch_of_another_side() -> None:
    model = _model(*LEAN, input_side=SIDE).eval()
    with pytest.raises(ValueError, match="cbam_min_hw"):
        model(_batch(side=32))


def test_a_small_patch_gets_its_own_placement() -> None:
    model = _model(*LEAN, input_side=16)
    assert all(isinstance(model.spatial.stages[i], nn.Identity) for i in (1, 3, 5))
    model.eval()
    with torch.no_grad():
        assert model(_batch(side=16)).shape == (4, 90)


# ══════════════════════════════════════════════════════════════════════
#  Y2 — masked MixStyle
# ══════════════════════════════════════════════════════════════════════


def test_mixstyle_is_the_frozen_configuration() -> None:
    assert (MIXSTYLE_P, MIXSTYLE_ALPHA, MIXSTYLE_STAGES) == (0.5, 0.1, (1, 2))
    model = _model("model.spatial_mixstyle=true")
    assert isinstance(model.spatial.stem.mixstyle, MaskedMixStyle)
    assert not list(model.spatial.stem.mixstyle.parameters())


def test_mixstyle_changes_no_initial_weight_and_no_eval_logit() -> None:
    off, on = _model(), _model("model.spatial_mixstyle=true")
    for (ka, va), (kb, vb) in zip(off.state_dict().items(), on.state_dict().items(), strict=True):
        assert ka == kb and torch.equal(va, vb)
    off.eval(), on.eval()
    with torch.no_grad():
        assert torch.equal(off(_batch()), on(_batch()))


def _stem_stage1(model: SpectralSeedNet, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    stem = model.spatial.stem
    mask = (x.abs().sum(1, keepdim=True) > 0).to(x.dtype)
    return stem._stage(stem.stage1, x.unsqueeze(1)), mask


def test_mixing_keeps_the_background_at_zero_and_reads_only_the_foreground() -> None:
    model = _model("model.spatial_mixstyle=true").train()
    stem = model.spatial.stem
    stem.mixstyle.p = 1.0
    x = _batch(6)
    with torch.no_grad():
        h, mask = _stem_stage1(model, x)
        torch.manual_seed(5)
        a = stem._apply_mask(stem._mix(1, h, mask), mask)
        moved = h.clone()
        moved[..., : SIDE // 8, :] += 100.0  # background only
        torch.manual_seed(5)
        b = stem._apply_mask(stem._mix(1, moved, mask), mask)
    assert float(a[..., : SIDE // 8, :].abs().max()) == 0.0
    assert torch.allclose(a, b, atol=1e-5)
    assert not torch.allclose(a, stem._apply_mask(h, mask))


def test_the_mixed_foreground_statistics_are_the_convex_mix() -> None:
    """Exactly MixStyle: foreground mean/sd become λ·own + (1−λ)·partner's."""
    n = 5
    mix = MaskedMixStyle(p=1.0).train()
    h = torch.randn(n, 3, 2, 8, 8) * torch.rand(n, 3, 2, 1, 1) * 3 + torch.randn(n, 3, 2, 1, 1)
    w = torch.zeros(n, 1, 8, 8)
    w[..., 2:7, 1:6] = 1.0  # a foreground window; background values are noise
    torch.manual_seed(11)
    out = mix(h, w)
    # Replay the module's draws, in its order: coin, λ, permutation.
    torch.manual_seed(11)
    torch.rand(())
    lam = torch.distributions.Beta(MIXSTYLE_ALPHA, MIXSTYLE_ALPHA).sample((n,)).view(n, 1, 1)
    perm = torch.randperm(n)

    def moments(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fg = t[..., 2:7, 1:6].flatten(-2)
        return fg.mean(-1), fg.var(-1, unbiased=False).add(1e-6).sqrt()

    mu, sd = moments(h)
    mu_out, sd_out = moments(out)
    assert torch.allclose(mu_out, lam * mu + (1 - lam) * mu[perm], atol=1e-4)
    assert torch.allclose(sd_out, lam * sd + (1 - lam) * sd[perm], atol=1e-3)


def test_a_kernel_without_foreground_keeps_its_activation() -> None:
    mix = MaskedMixStyle(p=1.0).train()
    h = torch.randn(3, 2, 2, 4, 4)
    w = torch.ones(3, 1, 4, 4)
    w[1] = 0.0
    torch.manual_seed(1)
    out = mix(h, w)
    assert torch.equal(out[1], h[1])


def test_mixstyle_is_a_training_only_op() -> None:
    mix = MaskedMixStyle(p=1.0).eval()
    h = torch.randn(4, 2, 2, 4, 4)
    assert mix(h, torch.ones(4, 1, 4, 4)) is h


def test_mixstyle_changes_a_training_forward() -> None:
    off, on = _model(), _model("model.spatial_mixstyle=true")
    on.spatial.stem.mixstyle.p = 1.0
    off.train(), on.train()
    x = _batch()
    torch.manual_seed(3)
    a = off(x, labels=torch.arange(4), arc_m=0.0)["main"]
    torch.manual_seed(3)
    b = on(x, labels=torch.arange(4), arc_m=0.0)["main"]
    assert not torch.allclose(a, b)


@pytest.mark.slow
def test_mixstyle_trains_under_torch_compile() -> None:
    """The Kaggle profile compiles the model (inductor). MixStyle's excluded forward makes a
    graph break; nothing around it may depend on the symbolic sizes that follow (the mask
    pooling once did, and inductor could not lower it)."""
    cfg = load_experiment_config(
        overrides=["data=ablation/u430k32_grouped", "model.spatial_mixstyle=true",
                   "model.stem_channels=16", "model.spatial_width_mult=0.25"]
    )
    torch.manual_seed(0)
    model = build_model(cfg, torch.linspace(0.0, 1.0, BANDS), input_side=32)
    compiled = torch.compile(model)
    model.train()
    x = _batch(4, side=32)
    for _ in range(3):  # p = 0.5: both branches of the coin, and a recompile if any
        out = compiled(x, labels=torch.arange(4), arc_m=0.0)
        (out["main"].logsumexp(1).mean() + out["aux_spatial"].logsumexp(1).mean()).backward()
        assert torch.isfinite(out["main"]).all()
