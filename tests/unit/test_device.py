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


# ══════════════════════════════════════════════════════════════════════
#  The memory budget, and why it has to say something
# ══════════════════════════════════════════════════════════════════════
#
# `configure_backend` sets PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 so a large
# `stage1.batch` cannot abort a multi-hour stage. What that trades the abort for
# is paging, and paging is silent: the loss falls, the epoch line prints, and the
# run is simply several times slower per sample than the same run one batch size
# down. Measured on an M5 (16 GB, 12.7 GB working set) — 32 -> 32.3 ms/sample,
# 64 -> 34.3, 128 -> 90.4 — the knee is entirely a memory effect. These tests pin
# that the crossing is reported, and reported once.


def test_peak_and_budget_are_reported_once(monkeypatch) -> None:
    """A silent slowdown is the failure mode; putting the number in the log is the fix."""
    from spectralquadnet.utils import device as dev

    dev.reset_memory_report()
    monkeypatch.setattr(dev, "memory_budget", lambda d: 12.7e9)
    monkeypatch.setattr(dev, "memory_allocated", lambda d: 9.6e9)

    first = dev.report_memory_use(torch.device("mps"))
    assert first is not None and "9.6 GB" in first and "12.7 GB" in first and "76%" in first
    assert dev.report_memory_use(torch.device("mps")) is None, "must not repeat per epoch"

    dev.reset_memory_report()
    assert dev.report_memory_use(torch.device("mps")) is not None, "reset must re-arm it"


def test_the_measured_paging_case_is_reported_though_it_is_below_any_threshold(monkeypatch) -> None:
    """The batch-128 knee sat at 76% of budget while paging badly.

    This is the reason the function reports instead of adjudicating: a threshold
    tuned to catch this case would have to fire far below the ceiling, and would
    then fire on a larger machine where the same allocation is comfortable. The
    line has to appear at 76% regardless, or it is absent exactly when wanted.
    """
    from spectralquadnet.utils import device as dev

    dev.reset_memory_report()
    monkeypatch.setattr(dev, "memory_budget", lambda d: 12.713e9)
    monkeypatch.setattr(dev, "memory_allocated", lambda d: 9.601e9)
    reported = dev.report_memory_use(torch.device("mps"))
    assert reported is not None
    assert dev.MEMORY_WARN_FRACTION > 9.601 / 12.713, "the measured case is below the warn line"


def test_a_healthy_run_still_gets_its_number(monkeypatch) -> None:
    """Batch 64 fits in 7.9 GB of 12.7 GB — reported, so the two can be compared."""
    from spectralquadnet.utils import device as dev

    dev.reset_memory_report()
    monkeypatch.setattr(dev, "memory_budget", lambda d: 12.7e9)
    monkeypatch.setattr(dev, "memory_allocated", lambda d: 7.9e9)
    assert "62%" in (dev.report_memory_use(torch.device("mps")) or "")


def test_a_device_that_reports_no_budget_is_not_guessed_at(monkeypatch) -> None:
    """CPU has no working-set ceiling, so there is nothing to put in the log."""
    from spectralquadnet.utils import device as dev

    dev.reset_memory_report()
    assert dev.memory_budget(torch.device("cpu")) is None
    assert dev.report_memory_use(torch.device("cpu")) is None


def test_release_memory_reads_the_peak_before_it_sweeps(monkeypatch) -> None:
    """Read after the sweep and the reading is the floor, which says nothing.

    `release_memory` runs at the epoch and phase boundaries, so the allocation
    standing when it is entered is the closest thing to the epoch's high-water
    mark available without a per-step probe.
    """
    from spectralquadnet.utils import device as dev

    dev.reset_memory_report()
    seen: list[str] = []
    monkeypatch.setattr(dev, "memory_budget", lambda d: 10.0e9)
    monkeypatch.setattr(dev, "memory_allocated", lambda d: 9.9e9)
    monkeypatch.setattr(dev, "empty_cache", lambda d=None: seen.append("swept"))

    dev.release_memory(torch.device("mps"))
    assert seen == ["swept"], "the sweep must still happen"
    assert dev.report_memory_use(torch.device("mps")) is None, "release_memory already reported"
