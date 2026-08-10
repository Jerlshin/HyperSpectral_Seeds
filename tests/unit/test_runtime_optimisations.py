"""The two per-device execution paths, and the claim that they are *paths*.

``runtime.decompose_conv3d`` and ``runtime.checkpoint_branch_a`` exist because
two modules are badly served by their default execution on the Metal backend —
Branch C's 3-D stem by ``Conv3d``'s backward kernel, Branch A's towers by
holding 2.9 GB of a 4.0 GB step. Both are switched per device, which is only
defensible if switching them is *not* a change to the model. That is the claim
this file tests, and it has three parts:

1. **Same function.** The decomposed stem and the checkpointed towers produce
   the same activations as the operators they stand in for.
2. **Same gradients.** Which is the half a forward comparison misses, and the
   half training depends on.
3. **Same schema.** No parameter, no buffer, no state-dict key — so a
   checkpoint written on Metal loads on CUDA and a resumed run may switch
   device and switch paths with it.

The tolerances differ between the two on purpose. Checkpointing is asserted
**exactly**: recomputation is the identical function of identical inputs, and
the towers hold no dropout, no BatchNorm and nothing that reads the RNG, so
anything but bit-equality means something stateful crept into a
``LargeKernelBlock1D``. The decomposition is asserted at 1e-5: it reassociates
a sum of ``kd`` partial products, which fp32 does not do exactly, and pinning
it tighter would be pinning the machine's FMA behaviour rather than the code's.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from spectralquadnet.models.branches.spatial_cnn import SpectralSpatialStem3D, conv3d_as_conv2d
from spectralquadnet.models.branches.spectral_profile import SpectralProfileBranch
from spectralquadnet.utils.device import apply_runtime_optimisations, resolve_runtime

BANDS = 40

#: The three stage shapes the real stem runs, at a batch small enough for CI.
#: Kept as data rather than derived from the module so a silent change to the
#: stem's geometry shows up here as a shape mismatch.
STEM_STAGES = [
    ("stage1 1→16 k=(7,3,3) s=(2,1,1)", (2, 1, 40, 16, 16), 1, 16, (7, 3, 3), (2, 1, 1), (3, 1, 1)),
    (
        "stage2 16→32 k=(5,3,3) s=(2,2,2)",
        (2, 16, 20, 16, 16),
        16,
        32,
        (5, 3, 3),
        (2, 2, 2),
        (2, 1, 1),
    ),
    (
        "stage3 32→64 k=(5,3,3) s=(2,2,2)",
        (2, 32, 10, 8, 8),
        32,
        64,
        (5, 3, 3),
        (2, 2, 2),
        (2, 1, 1),
    ),
]


# ══════════════════════════════════════════════════════════════════════
#  The decomposition is the operator
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "name,shape,c_in,c_out,k,s,p", STEM_STAGES, ids=[c[0] for c in STEM_STAGES]
)
def test_the_decomposition_reproduces_conv3d(name, shape, c_in, c_out, k, s, p) -> None:
    """Forward, shape and input-gradient, on each of the stem's three geometries.

    The input gradient is the load-bearing one: it is what the whole change
    exists to make faster, and a decomposition can be right forwards and wrong
    backwards if a depth tap's stride is gathered off by one.
    """
    torch.manual_seed(0)
    conv = nn.Conv3d(c_in, c_out, k, s, p, bias=False)
    x = torch.randn(*shape)

    ref_in = x.clone().requires_grad_(True)
    alt_in = x.clone().requires_grad_(True)

    ref = conv(ref_in)
    alt = conv3d_as_conv2d(alt_in, conv.weight, conv.stride, conv.padding)

    assert alt.shape == ref.shape, f"{name}: {tuple(alt.shape)} != {tuple(ref.shape)}"
    assert torch.allclose(alt, ref, atol=1e-5), f"{name}: max |Δ| = {(alt - ref).abs().max():.2e}"

    ref.square().mean().backward()
    alt.square().mean().backward()
    assert ref_in.grad is not None and alt_in.grad is not None
    assert torch.allclose(alt_in.grad, ref_in.grad, atol=1e-5), name


def test_the_decomposition_applies_bias_once() -> None:
    """The stem's convolutions are all ``bias=False``; this is why that is not load-bearing.

    Passing the bias down to each of the ``kd`` ``Conv2d`` calls would add it
    ``kd`` times — a bug that is invisible on the stem as configured today and
    would appear the moment somebody flipped ``bias=True``.
    """
    torch.manual_seed(0)
    conv = nn.Conv3d(3, 8, (5, 3, 3), (2, 2, 2), (2, 1, 1), bias=True)
    nn.init.normal_(conv.bias, std=1.0)
    x = torch.randn(2, 3, 12, 16, 16)

    with torch.no_grad():
        ref = conv(x)
        alt = conv3d_as_conv2d(x, conv.weight, conv.stride, conv.padding, conv.bias)

    assert torch.allclose(alt, ref, atol=1e-5), f"max |Δ| = {(alt - ref).abs().max():.2e}"


def test_the_stem_computes_the_same_thing_either_way() -> None:
    """The whole stem, both paths, forward and every parameter gradient.

    Per-parameter rather than a single scalar: the fold's 1×1 sits downstream of
    all three stages, so a wrong gradient in ``stage1`` alone can still leave a
    summed loss looking correct.
    """
    torch.manual_seed(0)
    stem = SpectralSpatialStem3D(BANDS, 96)
    x = torch.randn(2, BANDS, 32, 32)

    stem.decompose_conv3d = False
    stem.zero_grad(set_to_none=True)
    native = stem(x)
    native.square().mean().backward()
    native_grads = {n: p.grad.clone() for n, p in stem.named_parameters() if p.grad is not None}

    stem.decompose_conv3d = True
    stem.zero_grad(set_to_none=True)
    decomposed = stem(x)
    decomposed.square().mean().backward()

    assert decomposed.shape == native.shape
    assert torch.allclose(decomposed, native, atol=1e-5), (
        f"stem output drifted: max |Δ| = {(decomposed - native).abs().max():.2e}"
    )
    assert len(native_grads) > 0
    for name, param in stem.named_parameters():
        assert param.grad is not None, name
        assert torch.allclose(param.grad, native_grads[name], atol=1e-5), name


def test_the_decomposed_stem_still_zeroes_the_padded_region() -> None:
    """FE-2's invariant is a property of the stem, not of the kernel it dispatches to."""
    torch.manual_seed(0)
    stem = SpectralSpatialStem3D(BANDS, 96).eval()
    stem.decompose_conv3d = True
    x = torch.randn(2, BANDS, 64, 64)
    mask = torch.zeros(2, 1, 64, 64)
    mask[..., 16:48, 16:48] = 1.0

    with torch.no_grad():
        h1 = stem._apply_mask(stem._stage(stem.stage1, x.unsqueeze(1)), mask)

    assert float(h1[..., :16, :].abs().max()) == 0.0
    assert float(h1[..., 20:40, 20:40].abs().max()) > 0.0


# ══════════════════════════════════════════════════════════════════════
#  Recomputation is not an approximation
# ══════════════════════════════════════════════════════════════════════


def _branch_a(seed: int = 0) -> SpectralProfileBranch:
    torch.manual_seed(seed)
    return SpectralProfileBranch(torch.linspace(400.0, 1000.0, BANDS), out_dim=64, tower_ch=32)


def test_checkpointing_branch_a_is_bit_exact() -> None:
    """Identical output *and* identical gradients — ``torch.equal``, not ``allclose``.

    Recomputation replays the same modules on the same inputs, so the only way
    this drifts is if a tower acquires state that a second forward would
    advance: a BatchNorm's running mean, a dropout draw, a counter. That is the
    regression worth catching, and a tolerance would hide it.
    """
    channels = torch.randn(8, 3, BANDS)

    plain = _branch_a()
    plain.train()
    plain.grad_checkpoint = False
    # Seeded immediately before the call, not merely at construction: `proj`
    # ends in a Dropout, which sits *outside* the checkpointed region but would
    # otherwise make this a test of whether two RNG streams happened to align.
    torch.manual_seed(99)
    out_plain = plain.forward_channels(channels)
    out_plain.square().mean().backward()
    grads = {n: p.grad.clone() for n, p in plain.named_parameters() if p.grad is not None}

    ckpt = _branch_a()
    ckpt.train()
    ckpt.grad_checkpoint = True
    torch.manual_seed(99)
    out_ckpt = ckpt.forward_channels(channels)
    out_ckpt.square().mean().backward()

    assert torch.equal(out_ckpt, out_plain), "recomputed forward differs from the stored one"
    assert len(grads) > 0
    for name, param in ckpt.named_parameters():
        assert param.grad is not None, name
        assert torch.equal(param.grad, grads[name]), f"gradient drifted under recompute: {name}"


def test_checkpointing_advances_the_rng_stream_by_the_same_amount() -> None:
    """A recompute that consumed extra randomness would desynchronise everything downstream.

    Stage 1's augmentation, the branch-drop mask and CDWS all draw from the
    global stream *after* the branch runs, so a checkpoint region that advanced
    it further than the plain path would change the run without changing any
    single tensor visibly — and the golden Stage-1 weight hashes would move for
    a reason no diff explains.

    The absolute amount consumed is not the invariant (``proj``'s dropout draws
    on both paths); the *difference* is.
    """
    channels = torch.randn(8, 3, BANDS)

    def stream_after(grad_checkpoint: bool) -> torch.Tensor:
        branch = _branch_a()
        branch.train()
        branch.grad_checkpoint = grad_checkpoint
        torch.manual_seed(1234)
        branch.forward_channels(channels).square().mean().backward()
        return torch.rand(4)

    assert torch.equal(stream_after(True), stream_after(False))


def test_an_inference_pass_skips_the_recompute(monkeypatch) -> None:
    """No grad, no backward, so nothing to recompute — the EMA shadow's every call.

    Asserted by watching the ``checkpoint`` symbol itself rather than by timing:
    the cost being avoided is a second forward, which is exactly what calling
    ``checkpoint`` under ``no_grad`` would still do.
    """
    import spectralquadnet.models.branches.spectral_profile as mod

    calls: list[object] = []
    real = mod.checkpoint

    def counting(fn, *args, **kwargs):
        calls.append(fn)
        return real(fn, *args, **kwargs)

    monkeypatch.setattr(mod, "checkpoint", counting)

    branch = _branch_a()
    branch.grad_checkpoint = True
    channels = torch.randn(4, 3, BANDS)

    branch.eval()
    with torch.no_grad():
        branch.forward_channels(channels)
    assert calls == [], "an inference pass paid for checkpointing"

    branch.train()
    branch.forward_channels(channels)
    assert len(calls) == 3, "training should checkpoint exactly the three towers"


# ══════════════════════════════════════════════════════════════════════
#  Neither one is a model change
# ══════════════════════════════════════════════════════════════════════


def test_neither_flag_touches_the_state_dict() -> None:
    """The property that lets a checkpoint cross devices and a resume switch paths."""
    stem = SpectralSpatialStem3D(BANDS, 96)
    branch = _branch_a()
    before = (set(stem.state_dict()), set(branch.state_dict()))

    stem.decompose_conv3d = True
    branch.grad_checkpoint = True

    assert (set(stem.state_dict()), set(branch.state_dict())) == before
    assert not any("decompose" in k or "checkpoint" in k for k in stem.state_dict())


# ══════════════════════════════════════════════════════════════════════
#  Resolution and application
# ══════════════════════════════════════════════════════════════════════


class _Runtime:
    """The two fields under test; everything else falls to ``resolve_runtime``'s defaults."""

    def __init__(self, decompose: str = "auto", checkpoint_a: str = "auto") -> None:
        self.decompose_conv3d = decompose
        self.checkpoint_branch_a = checkpoint_a
        self.num_workers = 0


@pytest.mark.parametrize(
    "device,expected",
    [("mps", True), ("cuda", False), ("cpu", False)],
    ids=["metal-on", "cuda-off", "cpu-off"],
)
def test_auto_means_metal_only(device, expected) -> None:
    """Both defaults are 'on for Metal', for the two different reasons in the config docs.

    CUDA is excluded from *both*: cuDNN's 3-D kernels have no backward defect to
    route around, and dedicated VRAM makes Branch A's recompute a cost rather
    than a rescue.
    """
    plan = resolve_runtime(_Runtime(), torch.device(device))
    assert plan.decompose_conv3d is expected
    assert plan.checkpoint_branch_a is expected


def test_an_explicit_setting_overrides_the_hardware() -> None:
    """``on``/``off`` win on every device — a small CUDA card can ask for the recompute."""
    forced_on = resolve_runtime(_Runtime("on", "on"), torch.device("cpu"))
    assert forced_on.decompose_conv3d is True
    assert forced_on.checkpoint_branch_a is True

    forced_off = resolve_runtime(_Runtime("off", "off"), torch.device("mps"))
    assert forced_off.decompose_conv3d is False
    assert forced_off.checkpoint_branch_a is False


def test_apply_reaches_every_target_and_reports_it() -> None:
    """One call from ``train.py`` has to find modules three levels down a tree."""
    root = nn.Module()
    root.stem = SpectralSpatialStem3D(BANDS, 96)  # type: ignore[assignment]
    root.nested = nn.Sequential(_branch_a())  # type: ignore[assignment]

    plan = resolve_runtime(_Runtime("on", "on"), torch.device("cpu"))
    notes = apply_runtime_optimisations(root, plan)

    assert root.stem.decompose_conv3d is True
    assert root.nested[0].grad_checkpoint is True
    assert len(notes) == 2, notes

    off = resolve_runtime(_Runtime("off", "off"), torch.device("cpu"))
    assert apply_runtime_optimisations(root, off) == []
    assert root.stem.decompose_conv3d is False
    assert root.nested[0].grad_checkpoint is False


def test_apply_sees_through_a_wrapper() -> None:
    """``train.py`` calls this before DDP and compile, but a caller that does not is not wrong.

    Both wrappers keep the eager module at ``.module``/``._orig_mod``, which
    :func:`~spectralquadnet.utils.device.unwrap_model` already knows how to
    follow — so the flags land on the real branches either way.
    """

    class _Wrapper(nn.Module):
        def __init__(self, inner: nn.Module) -> None:
            super().__init__()
            self.module = inner

    inner = nn.Module()
    inner.stem = SpectralSpatialStem3D(BANDS, 96)  # type: ignore[assignment]
    wrapped = _Wrapper(inner)

    plan = resolve_runtime(_Runtime("on", "off"), torch.device("cpu"))
    apply_runtime_optimisations(wrapped, plan)

    assert inner.stem.decompose_conv3d is True
