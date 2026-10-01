"""S10 P0.6 — the architecture's demonstrated structural defects, pinned as tests.

S10 found two defects in ``SpectralSeedNet`` that are properties of the code,
not of any run, and D19 decided **not** to fix either before X1/X2 report:

* **F48 (B8)** — the spatial tail takes the 64 × 64 patch to a 1 × 1 map. The
  last ``ResBlock2D``'s 3 × 3, stride-2 convolution sees a 2 × 2 input, so 5 of
  its 9 taps multiply padding only: 327,680 parameters never receive a
  gradient. X5 (``model.spatial_tail_strides=[2,2,2,1]``) is the pre-registered
  repair.
* **F50 (B9)** — reflectance *level* reaches the learned layers only through the
  6-parameter ECA gate: everything after it is invariant to a global gain. X6
  (``model.spectral_level_block=log``) is the pre-registered repair.

The F48 tests are ``xfail(strict=True)``: they assert the *correct* property and
fail today, so the day X5 lands they pass, ``strict`` turns that into a failure,
and whoever lands X5 must remove the marker — the repair cannot go unnoticed.
The F50 tests assert today's behaviour (invariance after the gate, response
before it) and must be revised, deliberately, by X6.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from spectralquadnet.config.compose import load_experiment_config
from spectralquadnet.models.registry import build_model
from spectralquadnet.models.spectral_seed_net import SpectralSeedNet
from spectralquadnet.models.stats_ops import foreground_mask, masked_mean_spectrum

SIDE = 64  # the pipeline's patch side: the defect is a function of it
BATCH = 2

F48 = (
    "S10 F48: the last spatial-tail ResBlock2D runs a 3x3 stride-2 conv on a 2x2 map, so 5 "
    "of 9 taps (327,680 parameters) never receive a gradient. Repaired by X5 "
    "(model.spatial_tail_strides=[2,2,2,1]); remove this marker when X5 lands."
)


def _model(bands: int, wl: torch.Tensor | None = None) -> SpectralSeedNet:
    overrides = ["data=ablation/u430k32_grouped"] if bands == 32 else []
    cfg = load_experiment_config(overrides=overrides)
    assert int(cfg.data.num_bands) == bands
    torch.manual_seed(0)
    model = build_model(cfg, wl if wl is not None else torch.linspace(0.0, 1.0, bands))
    assert isinstance(model, SpectralSeedNet)
    return model


def _batch(bands: int, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    # Every pixel foreground: a tap that is dead here is dead for every input.
    return torch.rand(BATCH, bands, SIDE, SIDE, generator=gen) + 0.1


def _dead_taps(model: nn.Module, bands: int) -> dict[str, int]:
    """``{conv name: taps whose gradient is exactly 0 for every channel pair}``."""
    model.train()
    for seed in (0, 1):
        out = model(_batch(bands, seed), labels=torch.tensor([0, 1]), arc_m=0.0)
        (out["main"].logsumexp(1).mean() + out["aux_spatial"].logsumexp(1).mean()).backward()
    dead: dict[str, int] = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Conv3d)) and module.weight.grad is not None:
            if module.weight[0, 0].numel() == 1:
                continue  # a 1x1 conv has a single tap
            per_tap = module.weight.grad.abs().sum(dim=(0, 1))
            n = int((per_tap == 0).sum())
            if n:
                dead[name] = n
    return dead


def _tail_shape(model: SpectralSeedNet, bands: int) -> torch.Size:
    seen: list[torch.Size] = []
    handle = model.spatial.stages[-1].register_forward_hook(lambda m, i, o: seen.append(o.shape))
    model.eval()
    with torch.no_grad():
        model(_batch(bands))
    handle.remove()
    return seen[0]


# ══════════════════════════════════════════════════════════════════════
#  F48 — the 1 x 1 spatial tail (xfail until X5)
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=F48)
def test_no_spatial_tail_conv_tap_is_structurally_dead_at_32_bands() -> None:
    dead = {k: v for k, v in _dead_taps(_model(32), 32).items() if k.startswith("spatial.stages.")}
    dead = {k: v for k, v in dead.items() if ".sp." not in k}  # CBAM gates: see below
    assert not dead, dead


@pytest.mark.slow
@pytest.mark.xfail(strict=True, raises=AssertionError, reason=F48)
def test_no_spatial_tail_conv_tap_is_structurally_dead_at_215_bands(physical_wl_full) -> None:
    dead = {
        k: v
        for k, v in _dead_taps(_model(215, physical_wl_full), 215).items()
        if k.startswith("spatial.stages.") and ".sp." not in k
    }
    assert not dead, dead


def test_the_dead_taps_are_exactly_the_ones_s10_counted() -> None:
    """The measured defect, so a change to it is visible: 5 taps x 256 x 256 = 327,680."""
    model = _model(32)
    dead = _dead_taps(model, 32)
    assert dead.get("spatial.stages.6.c2") == 5
    assert 5 * model.spatial.stages[6].c2.weight[:, :, 0, 0].numel() == 327_680
    # CBAM's 7x7 spatial gate on the 2x2 map: 40 of 49 taps only ever see padding
    # (S10 B8; a P3.6 question, untouched by X5's stride change).
    assert dead.get("spatial.stages.5.sp.0") == 40


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=F48)
def test_the_spatial_tail_keeps_at_least_a_2x2_map() -> None:
    shape = _tail_shape(_model(32), 32)
    assert shape[-2] >= 2 and shape[-1] >= 2, shape


def test_docs_and_code_agree_on_the_tail_shape() -> None:
    """``docs/03`` §3.0 said (B, 256, 4, 4); it is (B, 256, 1, 1) (S10 B8, fixed in S11)."""
    assert tuple(_tail_shape(_model(32), 32)[1:]) == (256, 1, 1)


# ══════════════════════════════════════════════════════════════════════
#  F50 — reflectance level enters only through the ECA gate (until X6)
# ══════════════════════════════════════════════════════════════════════


def _gated(model: SpectralSeedNet, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m = foreground_mask(x, None)
    return model.se(x, m), m


@pytest.mark.parametrize("gain", [0.8, 1.25])
def test_after_the_gate_the_spatial_path_is_gain_invariant(gain: float) -> None:
    model = _model(32).eval()
    with torch.no_grad():
        xg, m = _gated(model, _batch(32))
        a, b = model.spatial(xg, m), model.spatial(gain * xg, m)
    assert ((a - b).norm() / a.norm()).item() < 5e-3


@pytest.mark.parametrize("gain", [0.8, 1.25])
def test_after_the_gate_the_spectral_descriptor_is_gain_invariant(gain: float) -> None:
    """Index bank, continuum depths, SNV and its derivatives are all scale-free."""
    model = _model(32).eval()
    with torch.no_grad():
        xg, m = _gated(model, _batch(32))
        r = masked_mean_spectrum(xg, m)
        morph = torch.zeros(r.shape[0], 8)
        a = model.spectral.features(r, morph)
        b = model.spectral.features(gain * r, morph)
    assert torch.allclose(a, b, rtol=1e-3, atol=1e-4)


def test_before_the_gate_a_global_gain_still_moves_the_logits() -> None:
    """B9: the gate is the only door level has into the network — and it is open."""
    model = _model(32).eval()
    x = _batch(32)
    with torch.no_grad():
        a, b = model(x), model(1.25 * x)
    assert ((a - b).norm() / a.norm()).item() > 1e-4
