"""TTA view construction — the zero-background invariant and the fp32 contract.

Covers **T1-1** (TT-1: foreground mean + re-mask) and **T1-2** (TT-3: fp32
forwards). Both are properties of the transform rather than of a number, so
each test states the property and, where the pre-Tier-1 behaviour is what it
guards against, also asserts that the old formula *violates* it — a test that
only checks the new code passes would not have caught the defect.

The invariant at stake: patches are zero outside the segmented seed, and every
masked operator downstream (``masked_spectral_stats``, ``MaskedSpectralECA``,
``extract_grid_spectra``) recovers that mask by testing for zero. A view that
moves the background off zero silently redefines the seed as the whole patch
(IMPROVEMENT_PLAN §2.6.1-2).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from spectralquadnet.engine.tta import foreground_mask, spectral_view, tta_predict

BANDS, HEIGHT, WIDTH = 6, 16, 16
NUM_CLASSES = 5

#: The scales ``tta_predict`` uses at the shipped ``tta_spectral=4``.
SCALES = torch.linspace(0.95, 1.05, 4)


def masked_patch(batch: int = 3, seed: int = 0) -> torch.Tensor:
    """A ``(B, C, H, W)`` batch shaped like a real patch: a positive seed on exact zeros."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.rand(batch, BANDS, HEIGHT, WIDTH, generator=gen) + 0.5
    rows = torch.arange(HEIGHT).view(-1, 1)
    cols = torch.arange(WIDTH).view(1, -1)
    disc = ((rows - HEIGHT / 2 + 0.5) ** 2 + (cols - WIDTH / 2 + 0.5) ** 2) <= (HEIGHT * 0.35) ** 2
    return x * disc.float()


def legacy_spectral_view(x: torch.Tensor, scale: float | torch.Tensor) -> torch.Tensor:
    """The pre-Tier-1 transform: whole-patch mean, no re-mask. Verbatim from the baseline."""
    mean = x.mean(dim=[2, 3], keepdim=True)
    return mean + (x - mean) * scale


def masked_band_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-sample, per-band mean over the foreground only."""
    n = mask.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
    return (x * mask).sum(dim=(-2, -1), keepdim=True) / n


class CaptureModel(nn.Module):
    """Records every input it is shown, and whether autocast was live at the time."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.seen: list[torch.Tensor] = []
        self.autocast_enabled: list[bool] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.seen.append(x.detach().clone())
        self.autocast_enabled.append(torch.is_autocast_enabled(x.device.type))
        # A per-sample-varying return, so `tta_predict`'s mean is not trivially constant.
        return x.reshape(x.shape[0], -1).sum(1, keepdim=True).expand(-1, self.num_classes) * 1.0


# ══════════════════════════════════════════════════════════════════════
#  T1-1 · the background stays exactly zero
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("scale", SCALES.tolist())
def test_spectral_view_leaves_the_background_exactly_zero(scale) -> None:
    """``|x_s ⊙ (1 - m)|_inf == 0`` — exactly, not approximately, for every scale."""
    x = masked_patch()
    m = foreground_mask(x)
    view = spectral_view(x, scale)

    assert (view * (1 - m)).abs().max().item() == 0.0


@pytest.mark.parametrize("scale", SCALES.tolist())
def test_the_old_spectral_view_broke_that_invariant(scale) -> None:
    """The defect this fix exists for: the pre-Tier-1 view filled the background.

    Without this the test above would pass against any implementation that
    happened to leave zeros alone, including one that never touched the patch.
    """
    x = masked_patch()
    m = foreground_mask(x)

    assert (legacy_spectral_view(x, scale) * (1 - m)).abs().max().item() > 1e-3


@pytest.mark.parametrize("scale", SCALES.tolist())
def test_spectral_view_preserves_the_foreground_support(scale) -> None:
    """The seed keeps exactly the pixels it had — same mask, same foreground fraction."""
    x = masked_patch()
    m = foreground_mask(x)
    view_mask = foreground_mask(spectral_view(x, scale))

    assert torch.equal(view_mask, m)
    assert view_mask.mean().item() == m.mean().item()


@pytest.mark.parametrize("scale", SCALES.tolist())
def test_spectral_view_rescales_about_the_foreground_mean(scale) -> None:
    """Contrast moves; the per-band foreground mean does not.

    This is the half of TT-1 that is not about the background. Rescaling about
    the *whole-patch* mean ``f·m_c`` (``f`` = foreground fraction) drags the
    seed's mean towards zero by ``(1-s)(1-f)·m_c``, so a "contrast" view also
    changes brightness — by a factor that depends on how much of the patch the
    seed happens to occupy.
    """
    x = masked_patch()
    m = foreground_mask(x)
    mu = masked_band_mean(x, m)

    fixed = masked_band_mean(spectral_view(x, scale), m)
    assert torch.allclose(fixed, mu, atol=1e-6)

    # …and the old one did not, unless s == 1 or the seed fills the patch.
    drifted = masked_band_mean(legacy_spectral_view(x, scale) * m, m)
    assert not torch.allclose(drifted, mu, atol=1e-3)


def test_spectral_view_accepts_an_explicit_mask() -> None:
    """FE-2 / T3-7 will pass the persisted mask instead of inferring it from zeros."""
    x = masked_patch()
    m = foreground_mask(x)

    assert torch.equal(spectral_view(x, 1.05, mask=m), spectral_view(x, 1.05))


def test_identity_scale_is_the_identity_on_the_foreground() -> None:
    x = masked_patch()
    assert torch.allclose(spectral_view(x, 1.0), x, atol=1e-6)


# ══════════════════════════════════════════════════════════════════════
#  T1-1 · end to end, through every view `tta_predict` builds
# ══════════════════════════════════════════════════════════════════════


def test_tta_preserves_background_mask() -> None:
    """Every one of the 12 views the shipped config builds respects the mask.

    The §4.3 gate, stated over the whole view set rather than the spectral
    transform alone: zero background, unchanged foreground fraction.
    """
    x = masked_patch()
    expected_foreground = foreground_mask(x).sum().item()
    model = CaptureModel()

    tta_predict(model, x, n_spatial=8, n_spectral=4)

    assert len(model.seen) == 12, "8 dihedral + 4 spectral views, none skipped at these scales"
    for i, view in enumerate(model.seen):
        m = foreground_mask(view)
        assert (view * (1 - m)).abs().max().item() == 0.0, f"view {i} leaked into the background"
        assert m.sum().item() == expected_foreground, f"view {i} changed the foreground fraction"


def test_tta_view_count_follows_its_arguments() -> None:
    """0-B's ``tta_spatial=8 tta_spectral=0`` arm, and the identity-scale skip."""
    x = masked_patch()

    spatial_only = CaptureModel()
    tta_predict(spatial_only, x, n_spatial=8, n_spectral=0)
    assert len(spatial_only.seen) == 8

    # linspace(0.95, 1.05, 3) lands exactly on 1.0, which duplicates view 0.
    with_identity = CaptureModel()
    tta_predict(with_identity, x, n_spatial=4, n_spectral=3)
    assert len(with_identity.seen) == 4 + 2


# ══════════════════════════════════════════════════════════════════════
#  T1-2 · the forwards run in fp32
# ══════════════════════════════════════════════════════════════════════


def test_tta_forwards_run_with_autocast_disabled() -> None:
    """TT-3: an enclosing autocast must not reach the TTA forwards.

    ``engine/evaluate.py`` already forces ``enabled=False`` for exactly this
    reason. Until TTA did the same, the reported +0.0163 TTA gain was
    confounded with a precision change (§2.6.4); measured, the confound is
    ≤0.0007 macro-F1, so this is hygiene rather than a gain.
    """
    x = masked_patch()
    model = CaptureModel()

    with torch.amp.autocast(device_type="cpu", enabled=True):
        assert torch.is_autocast_enabled("cpu"), "the enclosing context is live"
        tta_predict(model, x)

    assert model.autocast_enabled, "the model was called at least once"
    assert not any(model.autocast_enabled), "every view must be evaluated in fp32"


def test_tta_output_is_fp32_under_an_enclosing_autocast() -> None:
    """The observable consequence: averaged logits come back fp32, not bf16/fp16."""
    x = masked_patch()

    with torch.amp.autocast(device_type="cpu", enabled=True):
        logits = tta_predict(CaptureModel(), x)

    assert logits.dtype == torch.float32
    assert logits.shape == (x.shape[0], NUM_CLASSES)
