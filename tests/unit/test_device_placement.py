"""Building the λ-aware operators from a wavelength tensor that is not on the CPU.

``DataStore.load_wavelengths`` loads the wavelength CSV onto the **run's**
device, and ``build_run_context`` hands that tensor straight to ``build_model``.
So on every CUDA run — and on every rank of a ``torchrun`` job, where the device
is ``cuda:${LOCAL_RANK}`` — each λ-aware module is constructed from an
accelerator-resident tensor. Nothing about that path looks device-dependent,
which is exactly why ``ContinuumDepths.__init__`` was able to compare its own
permutation against a CPU ``torch.arange``::

    RuntimeError: Expected all tensors to be on the same device, but found at
    least two devices, cuda:0 and cpu!

raised while the model was being built, before a batch had been read.

What each test here can and cannot see
──────────────────────────────────────
The defect is a *cross-device* comparison, so a CPU-only runner cannot observe
it at all — it can only confirm that the CPU path still builds. The tests below
are therefore parametrised over the devices this machine actually has: CPU
always, CUDA and Metal when present. On the two-GPU box the failure was reported
from, ``device="cuda"`` is the failing case; on a Metal laptop ``device="mps"``
is, and it fails identically.

The last test builds the 256-band primary model on both ranks of a real
two-process group, which is the reported failure as a job rather than as a call.
Its rank body is shared with this file's ``__main__``, so the same coverage is
reachable through the launcher the report named::

    torchrun --standalone --nproc_per_node=2 tests/unit/test_device_placement.py

    # macOS: `--standalone` rendezvous resolves the hostname over IPv6 and hangs
    torchrun --nnodes=1 --node_rank=0 --nproc_per_node=2 \\
             --master_addr=127.0.0.1 --master_port=29513 \\
             tests/unit/test_device_placement.py
"""

from __future__ import annotations

import os
import socket
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp

from spectralquadnet.models.branches.spectral_stats import ContinuumDepths
from spectralquadnet.models.registry import build_model
from spectralquadnet.models.spectral_seed_net import SpectralSeedNet

#: The primary path's band count — the case in the report, and the one where the
#: hull's O(C²) form matters.
BANDS = 256
#: The smallest job that is a job: rank 1 is the one that has to agree with rank 0.
WORLD_SIZE = 2
#: The training step under DDP is a shape and device check, not a learning one,
#: so it runs at the smallest batch and a quarter of the trained patch side.
BATCH = 2
SPATIAL = 32


def _construction_devices() -> list[str]:
    """CPU, plus whichever accelerators this machine has."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


DEVICES = _construction_devices()


# ══════════════════════════════════════════════════════════════════════
#  The operator, built where its wavelengths already are
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("device", DEVICES)
def test_the_hull_builds_from_a_wavelength_tensor_already_on_the_device(device: str) -> None:
    """Construction must not care which device λ arrived on, and its buffers must follow it.

    Following λ is the contract ``SpectralDerivatives`` already holds to — both
    modules are handed the same tensor by the same caller, and a model whose two
    λ-derived operators disagreed about where they live would only fail later,
    inside a forward.
    """
    wavelengths = torch.linspace(383.2, 1006.5, BANDS, device=device)

    module = ContinuumDepths(wavelengths, n_depths=16)

    assert module.is_sorted, "an ascending λ axis is the identity permutation on every device"
    assert module.n_bands == BANDS
    assert module.order.device.type == torch.device(device).type
    assert module.lam.device.type == torch.device(device).type


@pytest.mark.parametrize("device", DEVICES)
def test_the_permutation_is_the_same_one_on_every_device(device: str) -> None:
    """The λ-ascending reordering is a property of the vector, not of the hardware.

    The cheap repair for the device error — declaring the axis sorted whenever
    the comparison is awkward — would pass the test above and silently return a
    wrong hull for a selection-ordered wavelength file, which is precisely the
    case ``ContinuumDepths`` permutes for. So the unsorted axis is checked
    against the CPU module element for element, hull included.
    """
    generator = torch.Generator().manual_seed(0)
    wavelengths = torch.rand(48, generator=generator) * 620.0 + 383.0
    spectra = torch.rand(3, 48, generator=generator) + 0.2

    on_cpu = ContinuumDepths(wavelengths, n_depths=8)
    on_device = ContinuumDepths(wavelengths.to(device), n_depths=8)

    assert not on_cpu.is_sorted, "a shuffled axis; sorted, the fixture would test nothing"
    assert on_device.is_sorted == on_cpu.is_sorted
    assert torch.equal(on_device.order.cpu(), on_cpu.order)
    assert torch.equal(on_device.lam.cpu(), on_cpu.lam)
    assert torch.allclose(
        on_device.envelope(spectra.to(device)).cpu(), on_cpu.envelope(spectra), atol=1e-5
    )


@pytest.mark.parametrize("device", DEVICES)
def test_the_primary_model_builds_from_an_accelerator_resident_wavelength_vector(
    cfg_default, physical_wl_full, device: str
) -> None:
    """``build_run_context``'s call, on every device this machine has.

    ``build_model(cfg, store.require_wavelengths())`` is the line that crashed;
    this is that line with the real 256-band λ axis and the real composed
    config, one step short of the ``.to(device)`` that follows it.
    """
    model = build_model(cfg_default, physical_wl_full.to(device))

    assert isinstance(model, SpectralSeedNet)
    assert model.spectral.continuum.n_bands == BANDS
    assert model.spectral.continuum.lam.device.type == torch.device(device).type
    assert model.spectral.derivatives.d1_op.device.type == torch.device(device).type


# ══════════════════════════════════════════════════════════════════════
#  The same construction, as a two-rank job
# ══════════════════════════════════════════════════════════════════════


def build_the_primary_model_on_this_rank() -> None:
    """What ``build_run_context`` does to the model, on whichever rank runs it.

    Shared by the spawned test below and by this file's ``__main__``, so the
    ``torchrun`` invocation in the module docstring exercises the code the test
    exercises rather than a second transcription of it.

    ``wrap_for_training`` is included because DDP broadcasts every buffer from
    rank 0 as it wraps — non-persistent ones included, which is what ``order``
    and ``lam`` are — so the wrap is the second place a misplaced λ buffer
    surfaces. One training step follows, on a two-sample batch: it is what puts
    the hull's buffers under a real forward and its gradients through a real
    all-reduce, and at this size it costs about a second.
    """
    from spectralquadnet.config.compose import load_experiment_config
    from spectralquadnet.utils.distributed import init_distributed, shutdown, wrap_for_training

    ctx = init_distributed(SimpleNamespace(multi_gpu="ddp", dist_timeout_s=300))
    try:
        cfg = load_experiment_config()
        bands = int(cfg.data.num_bands)
        assert bands == BANDS, f"the primary composition is the full cube; got {bands} bands"

        # `DataStore` puts λ on this rank's device before the model is built.
        # Synthetic here: construction is what is under test, and it reads the
        # vector's device and order, not its values.
        wavelengths = torch.linspace(0.0, 1.0, bands, device=ctx.device)

        model = build_model(cfg, wavelengths).to(ctx.device)

        assert isinstance(model, SpectralSeedNet)
        continuum = model.spectral.continuum
        assert continuum.is_sorted
        assert continuum.lam.device.type == ctx.device.type
        assert continuum.order.device.type == ctx.device.type

        ddp = wrap_for_training(model, ctx, sync_batchnorm=True)

        x = torch.randn(BATCH, bands, SPATIAL, SPATIAL, device=ctx.device)
        labels = torch.arange(BATCH, device=ctx.device) % int(cfg.data.num_classes)
        out = ddp(x, labels=labels)
        # Both heads, so every parameter receives gradient and the step is the
        # one `find_unused_parameters=False` is set for.
        (out["main"].sum() + out["aux_spatial"].sum()).backward()

        assert out["main"].shape == (BATCH, int(cfg.data.num_classes))
        assert torch.isfinite(out["main"]).all()
    finally:
        shutdown(ctx)


def _rank_entrypoint(rank: int, world_size: int, port: str) -> None:
    """``torchrun``'s environment, set by hand, so the test needs no launcher."""
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=port,
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    build_the_primary_model_on_this_rank()


def _free_port() -> str:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def test_the_primary_model_constructs_on_every_rank_of_a_two_rank_job(monkeypatch) -> None:
    """The reported failure, reproduced as a job.

    Two processes, the project's own ``init_distributed`` and
    ``wrap_for_training``, and the 256-band model built from a wavelength tensor
    on each rank's device. On a multi-GPU box that device is ``cuda:${LOCAL_RANK}``
    and this is the crash verbatim; over gloo it is the construction and
    buffer-broadcast plumbing without the cross-device comparison, which is
    still the half a CPU-only runner can check.

    ``mp.spawn`` re-raises a child's exception here with the child's traceback,
    so a rank that fails to build fails this test by name.
    """
    if torch.cuda.is_available() and torch.cuda.device_count() < WORLD_SIZE:
        # `init_distributed` refuses to place rank 1 on a one-GPU box, and it is
        # right to. Hiding the devices from the children runs the same job over
        # gloo rather than losing the test on a single-GPU machine.
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    mp.spawn(_rank_entrypoint, args=(WORLD_SIZE, _free_port()), nprocs=WORLD_SIZE, join=True)


if __name__ == "__main__":  # pragma: no cover - the `torchrun` entrypoint
    build_the_primary_model_on_this_rank()
    print(f"rank {os.environ.get('RANK', '0')}: SpectralSeedNet({BANDS} bands) constructed")
