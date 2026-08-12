"""Stage 1's mixed-precision contract: bf16, no loss scaler, fp32 head.

Stage 1 trained under ``autocast`` at torch's per-device default — fp16 — and
died of it: the loss went non-finite, ``train_one_epoch`` skipped the batch,
and from one epoch onward it skipped *every* batch while still reporting an
epoch mean over ``len(loader)``. Several separate mechanisms had to line up for
a transient overflow to become a permanent one, and this module pins the fix to
each of them.

**1. The dtype.** fp16 carries a 5-bit exponent: it saturates at 65 504 and its
smallest normal is 6.1e-5. bf16 carries fp32's 8-bit exponent, so neither bound
exists, and it pays for that in mantissa bits it does not need here. The
default is now bf16 wherever the device can run it —
:func:`~spectralquadnet.utils.device.resolve_amp_dtype`.

**2. The scaler.** Loss scaling exists to drag fp16 gradients above that
6.1e-5 floor. Under bf16 there is no floor to clear, so
:func:`~spectralquadnet.utils.device.make_grad_scaler` builds the scaler
*disabled* — a documented pass-through — rather than leaving it hunting for an
overflow that cannot happen.

**3. The head.** bf16's 8 mantissa bits are *coarser than the cosine clamp
itself*, so the margin algebra cannot run in it, and neither could fp16: the
head divides a cosine by a temperature annealed to 0.02, and takes ``log`` of a
probability floored at 1e-8 — below fp16's smallest subnormal. Every path in
``models/heads.py`` now runs in fp32 regardless of the ambient autocast state,
and :func:`test_the_head_is_invariant_to_the_ambient_autocast_state` is the
gate that says so.

**3b/3c. The two epsilons that were not there.** Both failures have the same
shape and neither is visible in the source: a guard written as a float literal
is evaluated at the *tensor's* dtype, and under fp16 a small enough literal is
zero. The balance term's 1e-8 probability floor and Branch C's 1e-8 floor under
a signed square root — whose derivative is unbounded at zero — both flushed,
and both then produced a ``NaN`` from the line that existed to prevent one.

**And the one that made it permanent.** ``GradNormAuxWeights`` reads the
epoch's mean per-branch gradient norm, which is accumulated *before*
``GradScaler`` gets to veto the step — so a routine fp16 overflow put ``inf``
in the mean, ``(inf/inf) ** alpha`` put ``NaN`` in an auxiliary weight, and
``NaN`` is not caught by ``min``/``max`` bounds. Every later loss was ``NaN``.
:func:`test_a_non_finite_norm_cannot_poison_the_weights` is the guard.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from torch.amp import autocast

from spectralquadnet.engine.train_epoch import train_one_epoch
from spectralquadnet.losses.auxiliary import AUX_WEIGHT_BOUNDS, GradNormAuxWeights
from spectralquadnet.models.branches.spatial_cnn import PN_EPS, _pn_floor
from spectralquadnet.models.heads import (
    BALANCE_EPS,
    COS_CLAMP_EPS,
    TAU_FLOOR,
    AdaptiveSubcenterArcFaceHead,
)
from spectralquadnet.tracking.base import NullTracker
from spectralquadnet.utils.device import (
    RuntimePlan,
    bfloat16_is_native,
    describe_amp,
    make_grad_scaler,
    resolve_amp_dtype,
    resolve_runtime,
    supports_bfloat16,
)

CPU = torch.device("cpu")
DIM, CLASSES, K = 32, 6, 3


def make_head(tau: float = 0.20) -> AdaptiveSubcenterArcFaceHead:
    torch.manual_seed(0)
    return AdaptiveSubcenterArcFaceHead(in_dim=DIM, num_classes=CLASSES, K=K, s=48.0, tau=tau)


def embeddings(n: int = 16, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = F.normalize(torch.randn(n, DIM, generator=generator), dim=1)
    y = torch.arange(n) % CLASSES
    return x, y


# ══════════════════════════════════════════════════════════════════════
#  1 · The dtype policy
# ══════════════════════════════════════════════════════════════════════


def test_auto_prefers_bfloat16_wherever_the_device_can_run_it(monkeypatch) -> None:
    """The default flips from fp16 to bf16. This is the fix, stated as one line."""
    monkeypatch.setattr("spectralquadnet.utils.device.supports_bfloat16", lambda device: True)
    assert resolve_amp_dtype(CPU) is torch.bfloat16
    assert resolve_amp_dtype(CPU, "auto") is torch.bfloat16


def test_auto_falls_back_to_fp16_only_where_bfloat16_is_unavailable(monkeypatch) -> None:
    """fp16 remains reachable, because a device that cannot autocast to bf16 raises."""
    monkeypatch.setattr("spectralquadnet.utils.device.supports_bfloat16", lambda device: False)
    assert resolve_amp_dtype(CPU) is torch.float16


def test_an_explicit_dtype_is_honoured() -> None:
    """``runtime.amp_dtype=fp16`` has to keep meaning fp16, or it is not a control arm."""
    assert resolve_amp_dtype(CPU, "fp16") is torch.float16
    assert resolve_amp_dtype(CPU, "float16") is torch.float16
    assert resolve_amp_dtype(CPU, "bf16") is torch.bfloat16
    assert resolve_amp_dtype(CPU, "BF16") is torch.bfloat16, "case is not part of the value"


def test_an_impossible_bfloat16_request_degrades_rather_than_raising(monkeypatch) -> None:
    """A config asking for bf16 on hardware without it should still run, loudly."""
    monkeypatch.setattr("spectralquadnet.utils.device.supports_bfloat16", lambda device: False)
    assert resolve_amp_dtype(CPU, "bf16") is torch.float16


def test_amp_can_be_turned_off_entirely() -> None:
    """The fp32 arm — what every regression gate and every Stage-2 epoch runs."""
    for value in ("off", "none", "fp32", "float32"):
        assert resolve_amp_dtype(CPU, value) is None


def test_an_unrecognised_dtype_is_refused_at_resolution_time() -> None:
    """Not silently ignored: "tf32" reads as a precision request and is not one."""
    with pytest.raises(ValueError, match="amp_dtype"):
        resolve_amp_dtype(CPU, "tf32")


def test_cpu_and_metal_report_bfloat16_support_consistently() -> None:
    """``supports_bfloat16`` must agree with what ``autocast`` actually accepts.

    Asserted by *doing* it rather than by restating the version table: the
    predicate exists precisely because torch raises instead of degrading, so
    the only honest check is whether the context manager it gates runs.
    """
    assert supports_bfloat16(CPU), "CPU autocast is bf16-only upstream"
    with autocast(device_type="cpu", dtype=torch.bfloat16):
        assert (torch.randn(4, 4) @ torch.randn(4, 4)).dtype is torch.bfloat16


def test_the_resolved_plan_carries_the_dtype() -> None:
    """``resolve_runtime`` is the one place the decision is made; nothing re-derives it."""

    class _Cfg:
        amp_dtype = "bf16"
        num_workers = 0

    assert resolve_runtime(_Cfg(), CPU).amp_dtype is torch.bfloat16

    class _Off(_Cfg):
        amp_dtype = "off"

    assert resolve_runtime(_Off(), CPU).amp_dtype is None


def test_a_hand_built_plan_opts_out_of_amp() -> None:
    """The new field is optional and defaults to off.

    Both halves matter. Optional, so the plans built by hand in the benchmarks
    and the perf tests still construct; off, so one of them cannot inherit
    mixed precision it never asked for and report a throughput number under it.
    """
    plan = RuntimePlan(
        device=CPU,
        num_workers=0,
        eval_num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=4,
        compile_enabled=False,
        compile_backend="inductor",
        compile_mode="default",
        channels_last=False,
        fused_optimizer=False,
        decompose_conv3d=False,
        checkpoint_branch_a=False,
        empty_cache_interval=0,
        diagnostics_interval=50,
        progress="off",
    )

    assert plan.amp_dtype is None


# ══════════════════════════════════════════════════════════════════════
#  2 · The scaler pairing
# ══════════════════════════════════════════════════════════════════════


def test_bfloat16_gets_a_disabled_scaler() -> None:
    """The pairing that matters: no loss scaling under a dtype with fp32's range."""
    scaler = make_grad_scaler(torch.bfloat16, CPU)

    assert scaler is not None, "the loops branch on `scaler is not None` to select AMP"
    assert not scaler.is_enabled()


def test_fp16_keeps_its_loss_scaler() -> None:
    """The fp16 arm is still correct fp16, not fp16 with the scaling removed."""
    scaler = make_grad_scaler(torch.float16, CPU)

    assert scaler is not None
    assert scaler.is_enabled()


def test_no_amp_means_no_scaler() -> None:
    assert make_grad_scaler(None, CPU) is None


def test_a_disabled_scaler_is_a_true_pass_through() -> None:
    """Why bf16 can keep one code path rather than needing its own.

    ``scale`` must return the tensor itself and ``unscale_`` must be inert, or
    the per-group clip in ``train_one_epoch`` would be clipping scaled
    gradients under bf16.
    """
    scaler = make_grad_scaler(torch.bfloat16, CPU)
    assert scaler is not None
    weight = torch.nn.Parameter(torch.ones(3))
    optimizer = torch.optim.SGD([weight], lr=0.1)
    loss = (weight * 2.0).sum()

    assert scaler.scale(loss) is loss, "a disabled scaler must not rescale the loss"

    loss.backward()
    scaler.unscale_(optimizer)
    assert torch.equal(weight.grad, torch.full((3,), 2.0)), "gradients must arrive unscaled"

    scaler.step(optimizer)
    scaler.update()
    assert torch.allclose(weight.detach(), torch.full((3,), 0.8)), "the step must still happen"


def test_the_precision_line_names_what_was_chosen() -> None:
    """A run's log has to record its own precision, or a NaN report is unreproducible."""
    assert "bf16" in describe_amp(torch.bfloat16, CPU)
    assert "fp16" in describe_amp(torch.float16, CPU)
    assert "GradScaler" in describe_amp(torch.float16, CPU)
    assert "off" in describe_amp(None, CPU)
    assert bfloat16_is_native(CPU) is True


# ══════════════════════════════════════════════════════════════════════
#  3 · The head runs in fp32 under any autocast
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_the_head_returns_fp32_logits_under_autocast(dtype) -> None:
    """The head's output width is its own decision, not the ambient one.

    The input arrives in the autocast dtype — that is what a backbone running
    under AMP hands it — and the logits still come back fp32.
    """
    head = make_head().train()
    x, y = embeddings()

    with autocast(device_type="cpu", dtype=dtype):
        logits, assign = head(x.to(dtype), y, global_m=0.35, return_assign=True)

    assert logits.dtype is torch.float32
    assert assign.dtype is torch.float32
    assert torch.isfinite(logits).all()
    assert torch.isfinite(assign).all()


def test_the_head_is_invariant_to_the_ambient_autocast_state() -> None:
    """The no-drift gate: turning AMP on must not move a single float in the head.

    This is what lets the fp32 region be added without regenerating a golden
    file — entering ``autocast(enabled=False)`` outside an autocast region is a
    no-op, and ``.float()`` on an fp32 tensor returns it unchanged, so an fp32
    caller sees exactly the arithmetic it saw before.
    """
    head = make_head().train()
    x, y = embeddings()

    plain = head(x, y, global_m=0.35)
    with autocast(device_type="cpu", dtype=torch.bfloat16):
        under_amp = head(x, y, global_m=0.35)

    assert torch.equal(plain, under_amp)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_the_margin_algebra_survives_a_near_perfect_cosine(dtype) -> None:
    """``sqrt(1 - cos^2)`` at the clamp is cancellation, and bf16 has 8 mantissa bits.

    bf16 resolves ~3.9e-3 near 1.0 — coarser than :data:`COS_CLAMP_EPS` itself,
    so the clamp bound 0.999 *rounds to 1.0* and ``1 - cos^2`` rounds to zero.
    The sine feeding the margin then vanishes silently. The margin does not
    *disappear* when that happens, which is what makes it hard to notice:
    ``phi`` degrades from ``cos(theta + m)`` to ``cos(theta) cos(m)``, still
    smaller than ``cos theta``, still a plausible logit. So the assertion is
    against the exact identity rather than against an inequality.

    Measured on this construction, the target logit is **44.309** in fp32; the
    pre-fix head returns **45.073** under bf16 and **44.319** under fp16, and
    both fail the assertion below. That is the whole case for the fp32 region:
    switching the dtype to bf16 *without* it would have traded an ``inf`` for a
    margin schedule that quietly does not apply.
    """
    margin = 0.35
    head = make_head(tau=0.0).train()
    # Every sample sits exactly on its own class's first sub-centre, so the
    # pooled cosine is pinned at the clamp — the worst case for the sine.
    with torch.no_grad():
        head.weight.copy_(F.normalize(torch.randn(CLASSES * K, DIM), dim=1))
    x = F.normalize(head.weight[::K].clone(), dim=1)
    y = torch.arange(CLASSES)

    with autocast(device_type="cpu", dtype=dtype):
        logits = head(x.to(dtype), y, global_m=margin)

    target = logits.gather(1, y.view(-1, 1)).squeeze(1)
    cos_theta = 1.0 - COS_CLAMP_EPS
    sin_theta = math.sqrt(1.0 - cos_theta**2)
    exact = head.s * (cos_theta * math.cos(margin) - sin_theta * math.sin(margin))
    degraded = head.s * cos_theta * math.cos(margin)  # what a vanished sine gives

    assert torch.isfinite(logits).all()
    assert torch.allclose(target, torch.full_like(target, exact), atol=1e-3)
    assert abs(exact - degraded) > 0.5, "the two arms must be far enough apart to tell apart"


def test_a_temperature_below_the_floor_cannot_overflow_the_exponent() -> None:
    """The pooling divides a cosine by ``tau``; ``tau`` is floored so that ratio is bounded.

    The hazard is real at fp16 width and needs no exotic schedule to reach: a
    cosine of 0.5 over a temperature of 1e-6 is already past fp16's largest
    finite value, before ``logsumexp`` sees it.
    """
    assert not torch.isfinite(torch.tensor(0.5 / 1e-6, dtype=torch.float16)), (
        "the overflow this floor prevents"
    )

    head = make_head(tau=1e-9)
    sub = torch.rand(8, CLASSES, K)

    with autocast(device_type="cpu", dtype=torch.float16):
        pooled = head.pool_subcentres(sub.to(torch.float16))

    assert pooled.dtype is torch.float32
    assert torch.isfinite(pooled).all()
    assert float(pooled.abs().max()) <= 1.0 - COS_CLAMP_EPS
    # The floor is the hard max's neighbourhood, not a different pooling: it
    # exceeds `max_k` by at most `TAU_FLOOR * log K`.
    assert torch.allclose(pooled, sub.max(dim=2).values, atol=TAU_FLOOR * math.log(K) + 1e-3)


def test_the_temperature_floor_leaves_the_shipped_schedule_alone() -> None:
    """``subcenter_tau`` anneals 0.20 → 0.02, both far above the floor. Nothing moves."""
    head = make_head(tau=0.02)
    assert head._pooling_temperature() == 0.02
    head.set_tau(0.20)
    assert head._pooling_temperature() == 0.20
    head.set_tau(0.0)
    assert head._pooling_temperature() == 0.0, "the hard max must stay exactly the hard max"


def test_out_of_range_cosines_cannot_reach_acos() -> None:
    """``acos`` returns ``NaN`` — not a saturated value — a single ulp outside [-1, 1].

    ``pool_subcentres`` already holds the cosine inside the clamp, so this
    tests the guard rather than a reachable state: the head must not produce a
    ``NaN`` logit even when handed a pooled cosine that is out of range.
    """
    assert torch.isnan(torch.acos(torch.tensor(1.0 + 1e-6))), "the failure mode being guarded"

    head = make_head(tau=0.0).train()
    y = torch.arange(CLASSES)
    # Straight into the margin algebra, bypassing the pooling clamp.
    phi = head._margined_target(torch.full((CLASSES, CLASSES), 1.5), y, 0.35)

    assert torch.isfinite(phi).all()


# ══════════════════════════════════════════════════════════════════════
#  3b · The balance term's logarithm
# ══════════════════════════════════════════════════════════════════════


def test_the_old_balance_floor_was_below_fp16s_smallest_subnormal() -> None:
    """Why the term needed ``xlogy`` and not just a smaller epsilon.

    The floor was ``1e-8``. fp16's smallest *subnormal* is 5.96e-8, so under
    autocast the floor flushed to zero, ``log`` returned ``-inf``, and
    ``0 * -inf`` put a ``NaN`` straight into the loss — from a term weighted
    0.01 that nobody was watching.
    """
    assert float(torch.tensor(BALANCE_EPS, dtype=torch.float16)) == 0.0

    dead = torch.zeros(1)
    assert torch.isnan(dead * (dead * K).log()).all(), "the spelling that produced the NaN"
    assert torch.equal(torch.xlogy(dead, dead * K), torch.zeros(1)), "the limit, and the fix"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_a_dead_subcentre_costs_log_k_rather_than_nan(dtype) -> None:
    """A fully collapsed assignment is the worst case, and it is finite.

    ``pi = [1, 0, 0]`` is not hypothetical: it is exactly what the hard
    ``tau <= 0`` assignment produces, which is where the anneal ends.
    """
    head = make_head()
    assign = torch.zeros(12, K)
    assign[:, 0] = 1.0
    y = torch.arange(12) % CLASSES

    with autocast(device_type="cpu", dtype=dtype):
        loss = head.balance_loss(assign.to(dtype), y)

    assert loss.dtype is torch.float32
    assert torch.isfinite(loss)
    assert float(loss) == pytest.approx(CLASSES * math.log(K), rel=1e-4)


def test_the_balance_term_still_reaches_the_weights_under_autocast() -> None:
    """A penalty that carries no gradient in bf16 would be telemetry, not a loss."""
    head = make_head().train()
    x, y = embeddings()
    head.zero_grad()

    with autocast(device_type="cpu", dtype=torch.bfloat16):
        _, assign = head(x.to(torch.bfloat16), y, return_assign=True)
        head.balance_loss(assign, y).backward()

    assert head.weight.grad is not None
    assert torch.isfinite(head.weight.grad).all()
    assert float(head.weight.grad.abs().sum()) > 0.0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_the_backward_pass_stays_finite_end_to_end(dtype) -> None:
    """Forward finiteness is half the contract; the gradients are what the step reads."""
    head = make_head().train()
    x, y = embeddings()
    x = x.to(dtype).requires_grad_(True)
    head.zero_grad()

    with autocast(device_type="cpu", dtype=dtype):
        logits, assign = head(x, y, global_m=0.35, return_assign=True)
        loss = F.cross_entropy(logits, y) + 0.01 * head.balance_loss(assign, y)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(head.weight.grad).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()


# ══════════════════════════════════════════════════════════════════════
#  3c · The second epsilon that autocast flushed to zero
# ══════════════════════════════════════════════════════════════════════


def test_branch_cs_power_normalisation_floor_survives_the_autocast_dtype() -> None:
    """The same failure shape as the balance term, one branch upstream.

    Branch C pools its feature map and takes a *signed square root*, whose
    derivative ``1 / (2 sqrt(x))`` is unbounded at zero — so the ``1e-8`` clamp
    under it is load-bearing, not cosmetic. Evaluated at fp16 width the literal
    rounds to zero (fp16's smallest subnormal is 5.96e-8), the clamp becomes a
    no-op, and a pooled channel that is exactly zero — ``amax`` over an
    all-negative map, which is a state and not an accident — backpropagates
    ``inf``. ``sign() * inf`` at zero is then ``NaN``.

    The floor is raised per dtype instead: unchanged at fp32 and bf16, which
    both hold 1e-8 exactly, and fp16's smallest normal where it cannot.
    """
    assert _pn_floor(torch.float32) == PN_EPS, "the fp32 path must be bit-identical"
    assert _pn_floor(torch.bfloat16) == PN_EPS, "bf16 has fp32's exponent range"
    assert _pn_floor(torch.float16) > PN_EPS

    for dtype in (torch.float16, torch.bfloat16):
        dead = torch.zeros(4, dtype=dtype, requires_grad=True)
        (dead.sign() * dead.abs().clamp(_pn_floor(dtype)).sqrt()).sum().backward()
        assert torch.isfinite(dead.grad).all(), dtype

    naive = torch.zeros(4, dtype=torch.float16, requires_grad=True)
    (naive.sign() * naive.abs().clamp(PN_EPS).sqrt()).sum().backward()
    assert torch.isnan(naive.grad).all(), "the NaN the dtype-aware floor removes"


# ══════════════════════════════════════════════════════════════════════
#  4 · The GradNorm guard that made one bad epoch permanent
# ══════════════════════════════════════════════════════════════════════


def test_the_weight_bounds_do_not_catch_a_nan() -> None:
    """Why filtering the *input* was necessary: the existing clamp is not a filter.

    ``min``/``max`` against ``NaN`` in Python return the ``NaN``, because every
    comparison with it is false. :data:`AUX_WEIGHT_BOUNDS` therefore bounded
    every value except the one that mattered.
    """
    low, high = AUX_WEIGHT_BOUNDS
    assert math.isnan(min(max(float("nan"), low), high))


@pytest.mark.parametrize("bad", [float("inf"), float("nan"), -float("inf")])
def test_a_non_finite_norm_cannot_poison_the_weights(bad) -> None:
    """One overflowing fp16 gradient used to end the run. It now costs one epoch's update.

    The branch reporting the bad norm holds its weight; the others are left
    alone too, since a mean taken against ``inf`` is not a balance measurement.
    """
    gradnorm = GradNormAuxWeights(alpha=0.5)
    before = dict(gradnorm.weights)

    updated = gradnorm.update({"branch_a": bad, "branch_b": 2.0, "branch_c": 1.0, "branch_d": 1.0})

    low, high = AUX_WEIGHT_BOUNDS
    assert all(math.isfinite(v) for v in updated.values())
    assert updated["aux_a"] == before["aux_a"], "the branch that overflowed holds its weight"
    assert all(low <= v <= high for v in updated.values())


def test_the_run_recovers_on_the_next_clean_epoch() -> None:
    """Holding a weight is recoverable; writing a ``NaN`` into it is not."""
    gradnorm = GradNormAuxWeights(alpha=0.5)

    gradnorm.update({"branch_a": float("inf"), "branch_b": 1.0, "branch_c": 1.0, "branch_d": 1.0})
    gradnorm.update({"branch_a": 0.25, "branch_b": 4.0, "branch_c": 1.0, "branch_d": 1.0})

    assert all(math.isfinite(v) for v in gradnorm.weights.values())
    assert gradnorm.weights["aux_a"] > gradnorm.weights["aux_b"], "the rule still applies"


def test_a_finite_update_is_untouched_by_the_guard() -> None:
    """The filter must be inert on every epoch that did not overflow."""
    guarded = GradNormAuxWeights(alpha=0.5, init={"aux_a": 2.0, "aux_b": 1.0})

    updated = guarded.update({"branch_a": 1.0, "branch_b": 3.0})

    assert updated["aux_a"] == pytest.approx(2.0 * (2.0 / 1.0) ** 0.5)
    assert updated["aux_b"] == pytest.approx(1.0 * (2.0 / 3.0) ** 0.5)


# ══════════════════════════════════════════════════════════════════════
#  5 · One real Stage-1 epoch, end to end
# ══════════════════════════════════════════════════════════════════════


class _RecordingTracker(NullTracker):
    """A ``NullTracker`` that keeps the scalars, so ``train/skipped_batches`` is readable."""

    def __init__(self) -> None:
        self.scalars: dict[str, float] = {}

    def log_scalars(self, tags: dict[str, float], step: int) -> None:
        self.scalars.update(tags)


@pytest.mark.regression
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_a_real_stage1_epoch_under_amp_skips_nothing(cfg, physical_wl, dtype) -> None:
    """The whole point, on the real 15 M-parameter model: no batch is dropped.

    ``train_one_epoch`` skips any batch whose loss is non-finite, and reports
    the count — so "the loss was NaN" and "the epoch trained" are the same
    assertion, read from the same counter the run logs. Both dtypes are
    exercised: bf16 is the new default, and fp16 has to keep working, since the
    head's fp32 region and the ``GradScaler`` pairing are what make it safe
    rather than the dtype choice alone.
    """
    from spectralquadnet.models.ema import ModelEMA  # noqa: PLC0415
    from spectralquadnet.models.spectral_quadnet import SpectralQuadNet  # noqa: PLC0415

    torch.manual_seed(0)
    model = SpectralQuadNet.from_config(cfg, physical_wl)
    model.set_subcentre_tau(cfg.model.subcenter_tau_final)  # the sharpest temperature
    data = torch.utils.data.TensorDataset(
        torch.randn(4, cfg.data.num_bands, 64, 64).abs(),
        torch.randint(0, cfg.data.num_classes, (4,)),
    )
    tracker = _RecordingTracker()

    loss, _acc = train_one_epoch(
        cfg,
        model,
        torch.utils.data.DataLoader(data, batch_size=2),
        torch.optim.AdamW(model.parameters(), lr=1e-4),
        torch.nn.CrossEntropyLoss(label_smoothing=0.1),
        make_grad_scaler(dtype, CPU),
        ModelEMA(model, decay=0.999),
        CPU,
        use_mixup=False,
        arc_m=cfg.stage1.arcface_m,
        tracker=tracker,
        aux_weights=GradNormAuxWeights(alpha=cfg.aux_gradnorm_alpha),
        amp_dtype=dtype,
    )

    assert tracker.scalars["train/skipped_batches"] == 0.0
    assert math.isfinite(loss)


@pytest.mark.regression
def test_an_amp_epoch_leaves_the_checkpoint_schema_untouched(cfg, physical_wl) -> None:
    """Autocast must not reach the parameters, only the arithmetic they feed.

    ``autocast`` casts *operands*, never the ``nn.Parameter`` behind them, so a
    bf16 epoch has to leave an all-fp32 state dict with the same keys — and
    that state dict has to load back ``strict=True``, which is how
    ``save_ckpt``/``load_ckpt`` move it. Asserted rather than assumed, because
    a head that returned a bf16 buffer or a weight written in the autocast
    dtype would be invisible until the next resume refused to load.
    """
    from spectralquadnet.models.ema import ModelEMA  # noqa: PLC0415
    from spectralquadnet.models.spectral_quadnet import SpectralQuadNet  # noqa: PLC0415

    torch.manual_seed(0)
    model = SpectralQuadNet.from_config(cfg, physical_wl)
    before = {k: v.dtype for k, v in model.state_dict().items()}
    data = torch.utils.data.TensorDataset(
        torch.randn(2, cfg.data.num_bands, 64, 64).abs(),
        torch.randint(0, cfg.data.num_classes, (2,)),
    )

    train_one_epoch(
        cfg,
        model,
        torch.utils.data.DataLoader(data, batch_size=2),
        torch.optim.AdamW(model.parameters(), lr=1e-4),
        torch.nn.CrossEntropyLoss(),
        make_grad_scaler(torch.bfloat16, CPU),
        ModelEMA(model, decay=0.999),
        CPU,
        use_mixup=False,
        arc_m=cfg.stage1.arcface_m,
        amp_dtype=torch.bfloat16,
    )

    after = model.state_dict()
    assert {k: v.dtype for k, v in after.items()} == before
    # The eight integer entries are BatchNorm's `num_batches_tracked` and
    # Branch A's neighbour table; every *float* entry must still be fp32.
    reduced = {k: v.dtype for k, v in after.items() if v.dtype in (torch.bfloat16, torch.float16)}
    assert not reduced, f"the autocast dtype leaked into {sorted(reduced)}"
    assert all(torch.isfinite(v).all() for v in after.values())

    torch.manual_seed(0)
    SpectralQuadNet.from_config(cfg, physical_wl).load_state_dict(after, strict=True)
