"""Device resolution and the Metal compatibility rules.

``resolve_device("auto")`` prefers Metal → CUDA → CPU. These tests pin that
precedence without requiring any particular accelerator to be present on the
machine running them.

The last test pins the upstream Metal limitation that used to force
``update_bn_stats`` down a device-dependent code path. T1-5 removed the
dependency by switching every stochastic module off for that pass — see
``tests/unit/test_bn_stats.py`` — but the limitation is still real, and if a
future torch release lifts it this test is the notice.
"""

from __future__ import annotations

import torch

from spectralquadnet.utils.device import release_memory, resolve_device


def test_explicit_device_is_never_overridden() -> None:
    """An explicit choice always wins, so the original CUDA lineage stays reachable."""
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device("cuda:1") == torch.device("cuda:1")
    assert resolve_device("mps") == torch.device("mps")


def test_a_torch_device_passes_straight_through() -> None:
    dev = torch.device("cpu")
    assert resolve_device(dev) is dev


def test_auto_prefers_metal_then_cuda_then_cpu(monkeypatch) -> None:
    """The precedence, exercised on all four availability combinations."""

    def _availability(mps: bool, cuda: bool) -> None:
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)

    _availability(mps=True, cuda=True)
    assert resolve_device("auto") == torch.device("mps"), "Metal outranks CUDA"

    _availability(mps=True, cuda=False)
    assert resolve_device("auto") == torch.device("mps")

    _availability(mps=False, cuda=True)
    assert resolve_device("auto") == torch.device("cuda"), "the baseline's choice"

    _availability(mps=False, cuda=False)
    assert resolve_device("auto") == torch.device("cpu")


def test_release_memory_collects_before_it_sweeps(monkeypatch) -> None:
    """Order is the whole point, so it is pinned rather than assumed.

    A tensor held only by a reference cycle is live to the caching allocator
    until CPython's collector breaks the cycle. Sweeping first therefore frees a
    pool that is still pinned, and the memory comes back at the next automatic
    collection — generally after the allocation that needed it has already
    failed. That is the Phase 1 → 2 OOM.
    """
    calls: list[str] = []
    monkeypatch.setattr("spectralquadnet.utils.device.gc.collect", lambda: calls.append("gc") or 0)
    monkeypatch.setattr(
        "spectralquadnet.utils.device.empty_cache", lambda device=None: calls.append("empty")
    )

    release_memory(torch.device("cpu"))
    assert calls == ["gc", "empty"]

    calls.clear()
    release_memory(torch.device("cpu"), collect=False)
    assert calls == ["empty"], "the caller that has just collected can skip it"


def test_release_memory_is_a_no_op_without_an_accelerator(monkeypatch) -> None:
    """It is called per epoch in all three stages, so a CPU run must not pay for it."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    release_memory(torch.device("cpu"))  # must not raise


def test_metal_fused_attention_really_does_reject_dropout_under_no_grad() -> None:
    """The upstream limitation, and why ``update_bn_stats`` disables dropout.

    Skips unless Metal is actually present. Three facts, in order: a train-mode
    attention forward under ``no_grad`` raises on MPS; the same forward with the
    module in ``eval()`` — which is what ``update_bn_stats`` now does — does
    not; and neither does the grad-enabled math path the workaround used to
    take. The middle one is the fix; it is also device-independent, which the
    workaround was not.
    """
    import pytest

    if not torch.backends.mps.is_available():
        pytest.skip("no Metal backend on this machine")

    attn = torch.nn.MultiheadAttention(32, 4, dropout=0.15, batch_first=True).to("mps")
    x = torch.randn(2, 8, 32, device="mps")

    attn.train(True)
    with pytest.raises(NotImplementedError, match="does not support dropout"), torch.no_grad():
        attn(x, x, x, need_weights=False)

    attn.eval()
    with torch.no_grad():
        attn(x, x, x, need_weights=False)  # the path update_bn_stats now takes

    attn.train(True)
    with torch.enable_grad():
        attn(x, x, x, need_weights=False)  # the path it used to take on MPS only
