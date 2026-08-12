"""Device resolution, accelerator-specific backend tuning and graph compilation.

``cfg.device`` is a *strategy string* (``"auto"``/``"cuda"``/``"cpu"``/``"mps"``)
rather than a live ``torch.device``, since YAML cannot carry the latter;
:func:`resolve_device` turns it into a concrete device.

``"auto"`` prefers **Metal (MPS) → CUDA → CPU**, so a default run on Apple
Silicon uses the GPU rather than falling through to CPU. An explicit
``device=cuda`` / ``device=cpu`` / ``device=mps`` is never overridden. Under
``torchrun`` the local rank pins the device (``cuda:${LOCAL_RANK}``) before this
precedence is consulted at all — see :mod:`spectralquadnet.utils.distributed`.

The autocast **dtype** is resolved here rather than left at torch's per-device
default, and the default is now **bfloat16** — see :func:`resolve_amp_dtype`.
fp16 carries a 5-bit exponent, so its largest finite value is 65 504 and its
smallest normal is 6.1e-5; bf16 carries fp32's 8-bit exponent and therefore
fp32's range, trading 13 mantissa bits for it. Stage 1 ran on the fp16 default
and diverged: an activation or a gradient leaves fp16's range, the loss comes
back ``inf``/``NaN``, ``train_one_epoch`` skips the batch, and once the epoch's
mean per-branch gradient norm is ``inf`` the GradNorm feedback loop in
``losses/auxiliary.py`` writes a non-finite auxiliary weight — after which
*every* subsequent batch is non-finite and the run silently trains on nothing.
bf16 removes the overflow, and :func:`make_grad_scaler` disables the loss
scaler with it, because loss scaling exists purely to drag fp16 gradients out
of that 6.1e-5 underflow floor and has nothing to do under bf16.

The precision bf16 gives up is bought back where it actually matters: the
ArcFace head's margin algebra runs in fp32 regardless of the autocast dtype
(see :mod:`spectralquadnet.models.heads`), since ``sqrt(1 - cos^2)`` at
``cos ~ 0.999`` is pure cancellation and bf16 has 8 mantissa bits to lose.

What is tuned here, and what is refused
───────────────────────────────────────
:func:`configure_backend` sets the accelerator knobs that are *free* — cuDNN
autotuning, the SDPA kernel preference, the Metal allocator's high-water mark —
and leaves alone the two that are not:

* **TF32** truncates a matmul's mantissa from 24 bits to 11. It is roughly 3×
  on Ampere and later, it is a precision change, and Turing (sm_75 — the T4)
  has no TF32 path to begin with. ``runtime.allow_tf32`` exists and defaults to
  off.
* **channels_last** re-selects convolution kernels whose reduction order
  differs from the contiguous ones'. ``runtime.channels_last`` likewise.

:func:`apply_runtime_optimisations` is the same idea one level down — the two
knobs that are not a *backend* setting but a choice of path *inside* a module,
both defaulting to "on for Metal, off elsewhere" and both measured rather than
assumed. On an M5, batch 32, against a **2103 ms / 4054 MB** step:

* **Branch C's 3-D stem** is 1318 ms of that, and 1240 ms of the 1318 is the
  backward of three convolutions whose forward costs 133. That is a bad kernel,
  not expensive arithmetic — the same branch's 2-D tail runs a normal 2.2×
  backward-to-forward ratio, and a ``Conv2d`` doing *more* work than the
  stage-2 ``Conv3d`` beats its backward 152 ms to 358. Writing the operator as
  the sum of ``kd`` ``Conv2d`` calls it is defined to be: **994 ms, 2.12×**.
* **Branch A holds 2935 MB of the 4054**, 72%, because ``grid_size_a=8``
  flattens a batch of 32 into 2048 independent sequences. Recomputing its three
  towers in the backward pass: **1901 MB, 2.13× less**, bit-exactly, for 4.8%
  of step time.

Together, **1077 ms and 2272 MB — 1.95× faster on 1.78× less memory**, which is
also what makes ``stage1.batch`` a choice again: batch 64 now runs at 31.2 ms
per sample against batch 32's 33.7.

:func:`maybe_compile` wraps ``torch.compile``. It is on by default for CUDA and
**off for Metal**, which is a measurement rather than a preference: on this
model at batch 32 the Metal inductor backend produced 983 ms per forward
against eager's 437 ms, because it cannot fuse the Branch-C 3-D stem and pays
the guard overhead for nothing.

This module used to carry a ``no_grad_is_safe_for_dropout`` predicate, because
Metal's fused attention kernel rejects dropout under ``no_grad`` and
``update_bn_stats`` was the one caller that ran the model in ``train()`` mode
there. Tier 1 (T1-5) switches every stochastic module off for that pass
outright, which is the correct behaviour for BatchNorm re-estimation anyway, so
nothing is left that depends on the accelerator — the predicate is gone rather
than left inert. ``tests/unit/test_device.py`` still pins the upstream Metal
limitation that motivated it.
"""

from __future__ import annotations

import contextlib
import gc
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn
from torch.amp import GradScaler

if TYPE_CHECKING:  # pragma: no cover - typing only
    from spectralquadnet.config.schema import RuntimeConfig

_log = logging.getLogger(__name__)

#: Upper bound on auto-selected DataLoader workers. Past this the mmap page-in
#: is bound by the page cache rather than by CPU, and each extra worker is
#: another process holding another mapping of the 5.6 GB cube.
MAX_AUTO_WORKERS: int = 8

#: Auto worker count off CUDA. Metal and CPU runs are accelerator-bound (the
#: forward is ~340 ms at batch 32 against ~16 ms of host-side augmentation), so
#: two workers already hide the feed, and macOS starts workers with ``spawn``,
#: which re-imports torch in every one of them.
NON_CUDA_AUTO_WORKERS: int = 2


def resolve_device(strategy: str | torch.device = "auto") -> torch.device:
    """Turn a config ``device`` string into a concrete :class:`torch.device`.

    ``"auto"`` picks the fastest locally available accelerator, preferring
    Apple's Metal backend, then CUDA, then CPU. Any other value (``"cuda"``,
    ``"cuda:1"``, ``"cpu"``, ``"mps"``) is passed straight to
    :class:`torch.device`, so an explicit choice is never silently overridden.

    Under ``torchrun`` this is not the function that chooses: every rank must
    own exactly one device, so :func:`~spectralquadnet.utils.distributed.init_distributed`
    resolves ``cuda:${LOCAL_RANK}`` itself and hands the result down.
    """
    if isinstance(strategy, torch.device):
        return strategy
    if strategy == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(strategy)


# ══════════════════════════════════════════════════════════════════════
#  Mixed precision
# ══════════════════════════════════════════════════════════════════════

#: What ``runtime.amp_dtype`` accepts, besides ``auto`` (resolve from the
#: hardware) and ``off``/``fp32`` (no autocast at all).
AMP_DTYPES: dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "half": torch.float16,
}

#: ``runtime.amp_dtype`` values that disable autocast outright.
AMP_OFF: frozenset[str] = frozenset({"off", "none", "no", "0", "false", "fp32", "float32"})


def supports_bfloat16(device: torch.device) -> bool:
    """Whether ``autocast(device_type=device.type, dtype=torch.bfloat16)`` runs here.

    Every accelerator this pipeline targets can *execute* bf16 — the question
    each backend answers differently is whether it does so natively or by
    conversion, which :func:`bfloat16_is_native` reports separately. CPU
    autocast is bf16-only upstream, so it is unconditionally true there.
    """
    if device.type == "cuda":
        if not torch.cuda.is_available():
            return False
        is_supported = getattr(torch.cuda, "is_bf16_supported", None)
        if is_supported is None:  # pragma: no cover - torch < 1.10
            return torch.cuda.get_device_capability(device)[0] >= 8
        return bool(is_supported())
    if device.type == "mps":
        # bf16 on Metal needs macOS 14; torch raises rather than degrading.
        newer = getattr(torch.backends.mps, "is_macos_or_newer", None)
        return bool(newer(14, 0)) if newer is not None else False
    return device.type == "cpu"


def bfloat16_is_native(device: torch.device) -> bool:
    """Whether bf16 arithmetic has hardware behind it, rather than being converted.

    Only used for the banner note. On a Turing card (sm_75 — the T4) bf16 has
    no Tensor Core path, so torch emulates it and the run is *correct but
    slower* than the fp16 it replaces. That is a trade this pipeline makes
    deliberately — a slow run beats a run whose loss is ``NaN`` from epoch 50 —
    but it is not a trade that should happen silently.
    """
    if device.type == "cuda":
        is_supported = getattr(torch.cuda, "is_bf16_supported", None)
        if is_supported is None:  # pragma: no cover - torch < 1.10
            return torch.cuda.get_device_capability(device)[0] >= 8
        try:
            return bool(is_supported(including_emulation=False))
        except TypeError:  # pragma: no cover - torch < 2.4 has no such keyword
            return torch.cuda.get_device_capability(device)[0] >= 8
    return supports_bfloat16(device)


def resolve_amp_dtype(device: torch.device, requested: str = "auto") -> torch.dtype | None:
    """The autocast dtype for this device, or ``None`` for "train in fp32".

    ``auto`` picks **bf16 wherever the device can run it** and falls back to
    fp16 only where it cannot. That is a stability choice, not a speed one: see
    the module docstring for the fp16 divergence it exists to prevent. An
    explicit ``fp16`` is honoured — the fp16 path is still supported, and
    :func:`make_grad_scaler` still gives it its loss scaler — but it is no
    longer what an unconfigured run gets.

    Args:
        device: The device this rank owns.
        requested: ``auto``, ``bf16``/``fp16`` (and their aliases), or one of
            :data:`AMP_OFF`.

    Raises:
        ValueError: On an unrecognised value, rather than silently training in
            a precision nobody asked for.
    """
    key = str(requested).strip().lower()
    if key in AMP_OFF:
        return None
    if key != "auto":
        if key not in AMP_DTYPES:
            raise ValueError(
                f"runtime.amp_dtype={requested!r} is not one of "
                f"{sorted(AMP_DTYPES) + sorted(AMP_OFF) + ['auto']}"
            )
        dtype = AMP_DTYPES[key]
        if dtype is torch.bfloat16 and not supports_bfloat16(device):
            _log.warning(
                "runtime.amp_dtype=bf16 requested but %s cannot autocast to it — "
                "falling back to fp16 with a loss scaler",
                device,
            )
            return torch.float16
        return dtype
    return torch.bfloat16 if supports_bfloat16(device) else torch.float16


def make_grad_scaler(amp_dtype: torch.dtype | None, device: torch.device) -> GradScaler | None:
    """The loss scaler that goes with ``amp_dtype``, or ``None`` when AMP is off.

    Three cases, and the middle one is the whole point:

    * **fp16** — an *enabled* scaler. fp16's smallest normal is 6.1e-5, and a
      backbone gradient below that flushes to zero; scaling the loss up before
      the backward and unscaling before the step is what keeps it representable.
    * **bf16** — a *disabled* scaler. bf16 has fp32's exponent range, so there
      is nothing to rescue and nothing to overflow; a scaler here would search
      for an inf that never arrives and multiply/divide every gradient twice
      per step for it. Disabled rather than ``None`` so the loops keep one code
      path: every ``GradScaler`` method is a documented pass-through when
      ``enabled=False`` (``scale`` returns its argument, ``unscale_`` returns
      immediately, ``step`` calls ``optimizer.step()``).
    * **AMP off** — ``None``, which is what ``train_one_epoch`` reads as "run
      the fp32 path".

    ``device=`` is what makes any of it real: a bare ``GradScaler()`` binds to
    CUDA, so on any other accelerator it prints "CUDA is not available.
    Disabling." and every call becomes a pass-through whether or not that was
    wanted.
    """
    if amp_dtype is None:
        return None
    return GradScaler(device=device.type, enabled=amp_dtype is torch.float16)


def describe_amp(amp_dtype: torch.dtype | None, device: torch.device) -> str:
    """The banner line for the resolved precision, so a run's log records it."""
    if amp_dtype is None:
        return "AMP=off (fp32)"
    if amp_dtype is torch.float16:
        return "AMP=fp16 + GradScaler"
    native = "" if bfloat16_is_native(device) else " (emulated — no bf16 Tensor Core path)"
    return f"AMP=bf16, no loss scaler{native}"


# ══════════════════════════════════════════════════════════════════════
#  The resolved runtime plan
# ══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class RuntimePlan:
    """Every ``-1``/``"auto"`` in :class:`~spectralquadnet.config.schema.RuntimeConfig`, decided.

    Resolved once, at startup, and threaded through the loader builders and the
    stage loops — so "how many workers does the val loader get" has one answer
    per run rather than one answer per call site.
    """

    device: torch.device
    num_workers: int
    eval_num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int | None
    compile_enabled: bool
    compile_backend: str
    compile_mode: str
    channels_last: bool
    fused_optimizer: bool
    decompose_conv3d: bool
    checkpoint_branch_a: bool
    empty_cache_interval: int
    diagnostics_interval: int
    progress: str
    #: Autocast dtype for the stages that train under AMP, or ``None`` for
    #: fp32. Defaulted — rather than required like every field above — so a
    #: plan built by hand (a benchmark, a test) opts *out* of mixed precision
    #: instead of silently into it. :func:`resolve_runtime` always sets it.
    amp_dtype: torch.dtype | None = None

    @property
    def loader_kwargs(self) -> dict[str, Any]:
        """The ``DataLoader`` performance keywords for a **training** loader.

        ``prefetch_factor`` and ``persistent_workers`` are illegal at
        ``num_workers=0`` — torch raises rather than ignoring them — so they are
        omitted entirely rather than passed as ``None``.
        """
        return _loader_kwargs(
            self.num_workers, self.pin_memory, self.persistent_workers, self.prefetch_factor
        )

    @property
    def eval_loader_kwargs(self) -> dict[str, Any]:
        """The same, for an evaluation loader.

        Evaluation loaders are built and dropped repeatedly (``build_loaders``
        is called once per stage, ``build_natural_prior_loader`` once more), so
        they never keep workers resident: a persistent worker pool that outlives
        its loader is a process leak, and under ``spawn`` re-creating one costs
        seconds.
        """
        return _loader_kwargs(self.eval_num_workers, self.pin_memory, False, self.prefetch_factor)


def _loader_kwargs(
    workers: int, pin: bool, persistent: bool, prefetch: int | None
) -> dict[str, Any]:
    kw: dict[str, Any] = {"num_workers": int(workers), "pin_memory": bool(pin)}
    if workers > 0:
        kw["persistent_workers"] = bool(persistent)
        if prefetch is not None:
            kw["prefetch_factor"] = int(prefetch)
    return kw


def _auto_workers(device: torch.device) -> int:
    """Half the available cores on CUDA, a fixed small number elsewhere.

    ``os.process_cpu_count`` (3.13) and ``len(os.sched_getaffinity(0))`` report
    the cores this process may actually use, which is what a container-limited
    T4 box needs; ``os.cpu_count()`` reports the machine's.
    """
    cores = getattr(os, "process_cpu_count", None)
    n = cores() if cores is not None else None
    if n is None:
        try:
            n = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
        except AttributeError:
            n = os.cpu_count() or 4
    if device.type != "cuda":
        return min(NON_CUDA_AUTO_WORKERS, max(0, n - 1))
    return max(1, min(MAX_AUTO_WORKERS, n // 2))


def _tri_state(value: int, default: bool) -> bool:
    """``-1`` means "use ``default``"; anything else is a plain truth value."""
    return default if int(value) < 0 else bool(value)


def resolve_runtime(
    cfg: RuntimeConfig | Any, device: torch.device, world_size: int = 1
) -> RuntimePlan:
    """Decide every auto-valued runtime knob for this device and topology.

    Args:
        cfg: The ``runtime`` config group.
        device: The device this rank owns.
        world_size: Number of DDP ranks. Workers are budgeted **per rank**, so
            a two-GPU box does not launch ``2 × num_workers`` processes against
            the same page cache.

    Returns:
        A frozen :class:`RuntimePlan`; nothing downstream re-derives any of it.
    """
    workers = int(getattr(cfg, "num_workers", -1))
    if workers < 0:
        workers = _auto_workers(device)
    if world_size > 1:
        workers = max(0, workers // world_size)

    eval_workers = int(getattr(cfg, "eval_num_workers", -1))
    if eval_workers < 0:
        eval_workers = min(workers, 4)

    pin = _tri_state(getattr(cfg, "pin_memory", -1), device.type == "cuda")
    if pin and device.type != "cuda":
        # `pin_memory=True` allocates through the CUDA host allocator. There is
        # no Metal equivalent, and DataLoader's pinning thread raises rather
        # than degrading, so an explicit request off CUDA is refused here where
        # the message can say why.
        _log.warning("runtime.pin_memory ignored: page-locked staging needs CUDA, got %s", device)
        pin = False

    persistent = _tri_state(getattr(cfg, "persistent_workers", -1), workers > 0) and workers > 0

    compile_mode_str = str(getattr(cfg, "compile", "auto")).lower()
    compile_enabled = (
        device.type == "cuda"
        if compile_mode_str == "auto"
        else compile_mode_str in ("on", "true", "1", "yes")
    )

    fused_str = str(getattr(cfg, "fused_optimizer", "auto")).lower()
    fused = (
        device.type == "cuda" if fused_str == "auto" else fused_str in ("on", "true", "1", "yes")
    )

    # Both default to "Metal only", for the two different reasons in
    # `RuntimeConfig`: one is a kernel-quality defect specific to this backend,
    # the other is unified memory making an activation peak the host's problem.
    decomp_str = str(getattr(cfg, "decompose_conv3d", "auto")).lower()
    decompose = (
        device.type == "mps" if decomp_str == "auto" else decomp_str in ("on", "true", "1", "yes")
    )

    ckpt_str = str(getattr(cfg, "checkpoint_branch_a", "auto")).lower()
    ckpt_a = device.type == "mps" if ckpt_str == "auto" else ckpt_str in ("on", "true", "1", "yes")

    return RuntimePlan(
        device=device,
        num_workers=workers,
        eval_num_workers=eval_workers,
        pin_memory=pin,
        persistent_workers=persistent,
        prefetch_factor=int(getattr(cfg, "prefetch_factor", 4)),
        compile_enabled=compile_enabled,
        compile_backend=str(getattr(cfg, "compile_backend", "inductor")),
        compile_mode=str(getattr(cfg, "compile_mode", "default")),
        channels_last=bool(getattr(cfg, "channels_last", False)),
        fused_optimizer=fused,
        decompose_conv3d=decompose,
        checkpoint_branch_a=ckpt_a,
        empty_cache_interval=int(getattr(cfg, "empty_cache_interval", 0)),
        diagnostics_interval=max(1, int(getattr(cfg, "diagnostics_interval", 50))),
        progress=str(getattr(cfg, "progress", "auto")),
        amp_dtype=resolve_amp_dtype(device, str(getattr(cfg, "amp_dtype", "auto"))),
    )


# ══════════════════════════════════════════════════════════════════════
#  Backend tuning
# ══════════════════════════════════════════════════════════════════════


def configure_backend(device: torch.device, cfg: RuntimeConfig | Any) -> list[str]:
    """Apply the accelerator's free performance settings; return what was set.

    "Free" is the operative word — see the module docstring for the two knobs
    that are not applied unless asked for.

    Returns:
        Human-readable lines for the startup banner, so a run's log records the
        kernel selection it actually ran under.
    """
    notes: list[str] = []

    if device.type == "cuda":
        benchmark = bool(getattr(cfg, "cudnn_benchmark", True))
        torch.backends.cudnn.benchmark = benchmark
        notes.append(f"cudnn.benchmark={benchmark}")

        tf32 = bool(getattr(cfg, "allow_tf32", False))
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        # torch >= 2.9 routes matmul precision through this setting; keeping the
        # two consistent stops a later torch from re-enabling TF32 behind the
        # `allow_tf32` flag's back.
        with contextlib.suppress(AttributeError, ValueError):  # pragma: no cover - old torch
            torch.set_float32_matmul_precision("high" if tf32 else "highest")
        notes.append(f"TF32={tf32}")

        major = torch.cuda.get_device_capability(device)[0]
        notes.append(f"sm_{''.join(str(v) for v in torch.cuda.get_device_capability(device))}")
        if major >= 8:
            # Flash/mem-efficient SDPA exist from Ampere on; Turing (the T4)
            # keeps the math path, which is the correct kernel there and not a
            # fallback.
            _enable_sdp_kernels(flash=True, mem_efficient=True, math=True)
        else:
            _enable_sdp_kernels(flash=False, mem_efficient=True, math=True)

    elif device.type == "mps":
        # The Metal allocator refuses an allocation that would push the process
        # past a fraction of recommended working-set size. Branch A runs
        # `batch * grid_size_a ** 2` spectra through three towers, so at
        # stage1.batch = 128 that is 8,192 sequences and the default 1.0 ratio
        # is reached mid-backward. Raising the *high* watermark while leaving
        # the low one alone lets the allocator keep going and still garbage
        # collect, rather than aborting the run.
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
        os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.8")
        notes.append("MPS watermark=unbounded/0.8")

    return notes


def apply_runtime_optimisations(model: nn.Module, plan: RuntimePlan) -> list[str]:
    """Select the per-device execution paths inside the model; return what was set.

    The companion to :func:`configure_backend`, one level down: that function
    tunes the *backend*, this one tunes the two modules whose default path is
    wrong on some accelerator. Both settings are execution choices and nothing
    else — no parameter, no buffer, no state-dict key, no draw from the RNG —
    so a checkpoint written under either loads under the other, and a resumed
    run may legitimately switch device and switch paths with it.

    Call this **before** ``wrap_for_training`` and ``maybe_compile``, and before
    the EMA shadow is deep-copied, so every later copy of the module inherits
    the same paths. Idempotent; safe on a model that has neither module.

    Args:
        model: The eager :class:`~spectralquadnet.models.spectral_quadnet.SpectralQuadNet`.
        plan: The resolved plan; only its two path flags are read.

    Returns:
        Banner lines, or an empty list when both flags are off — so a run's log
        records which kernels it actually ran, the same contract
        :func:`configure_backend` has.
    """
    # Imported here rather than at module scope: `utils.device` is imported by
    # config and data code that has no business dragging the model in.
    from spectralquadnet.models.branches.spatial_cnn import SpectralSpatialStem3D
    from spectralquadnet.models.branches.spectral_profile import SpectralProfileBranch

    notes: list[str] = []
    root = unwrap_model(model)

    stems = [m for m in root.modules() if isinstance(m, SpectralSpatialStem3D)]
    for stem in stems:
        stem.decompose_conv3d = plan.decompose_conv3d
    if plan.decompose_conv3d and stems:
        notes.append(f"Conv3d→Conv2d stem×{len(stems)}")

    branches = [m for m in root.modules() if isinstance(m, SpectralProfileBranch)]
    for branch in branches:
        branch.grad_checkpoint = plan.checkpoint_branch_a
    if plan.checkpoint_branch_a and branches:
        notes.append(f"branch-A recompute×{len(branches)}")

    if (plan.decompose_conv3d and not stems) or (plan.checkpoint_branch_a and not branches):
        _log.warning(
            "runtime path flags were set but the modules they target are absent — "
            "apply_runtime_optimisations was given %s, not a SpectralQuadNet",
            type(root).__name__,
        )
    return notes


def _enable_sdp_kernels(*, flash: bool, mem_efficient: bool, math: bool) -> None:
    """Select the scaled-dot-product-attention backends, across torch versions.

    Branch D's four ``nn.MultiheadAttention`` blocks dispatch through SDPA, and
    the enable/disable API moved namespaces in torch 2.2.
    """
    setters = (
        ("enable_flash_sdp", flash),
        ("enable_mem_efficient_sdp", mem_efficient),
        ("enable_math_sdp", math),
    )
    for name, value in setters:
        fn = getattr(torch.backends.cuda, name, None)
        if callable(fn):
            # The enable/disable API moved namespaces in torch 2.2 and the
            # backends available differ by architecture, so a refusal here is a
            # version/hardware fact rather than an error.
            with contextlib.suppress(RuntimeError, TypeError):  # pragma: no cover
                fn(value)


def describe_hardware(device: torch.device) -> list[str]:
    """Banner lines naming the accelerator, its memory and its arch."""
    if device.type == "cuda":
        lines = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            lines.append(
                f"[GPU{i}] {props.name}  |  {props.total_memory // 1024**3} GB  |  "
                f"sm_{props.major}{props.minor}  |  {props.multi_processor_count} SMs"
            )
        return lines
    if device.type == "mps":
        return ["[GPU] Apple Silicon (Metal / MPS)"]
    return ["[CPU] no accelerator"]


# ══════════════════════════════════════════════════════════════════════
#  Graph compilation
# ══════════════════════════════════════════════════════════════════════


def maybe_compile(model: nn.Module, plan: RuntimePlan) -> nn.Module:
    """``torch.compile`` the model when the plan asks for it, else return it unchanged.

    Failure to compile is **not** fatal: a graph break, a missing Triton, or an
    inductor version that dislikes the 3-D stem all fall back to eager with a
    warning, because a slower run is a better outcome than no run.

    Returns the compiled wrapper, which forwards attribute access to the
    original module — so ``model.arcface_head`` and ``model.state_dict()`` keep
    working and the checkpoint schema is untouched.
    """
    if not plan.compile_enabled:
        return model
    if not hasattr(torch, "compile"):  # pragma: no cover - torch < 2.0
        return model
    try:
        compiled = torch.compile(
            model, backend=plan.compile_backend, mode=plan.compile_mode, dynamic=False
        )
    except Exception as exc:  # pragma: no cover - environment-dependent
        _log.warning("torch.compile unavailable (%s) — continuing in eager mode", exc)
        return model
    # `torch.compile` is typed as returning a callable wrapper rather than a
    # Module; it forwards attribute access to the original, which is the whole
    # reason `state_dict()` and `arcface_head` keep working through it.
    return cast(nn.Module, compiled)


def reset_compilation() -> None:
    """Drop every compiled graph and its guards.

    Called at the Stage 2 → 3 boundary: Stage 3 changes the margin vector every
    cycle and runs SAM's double backward, which re-triggers recompilation for
    no benefit.
    """
    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None:
        dynamo.reset()


def unwrap_model(model: nn.Module) -> nn.Module:
    """The eager module underneath ``torch.compile`` and/or DDP wrappers.

    Checkpointing, EMA and the per-branch gradient split all address parameters
    by their **unprefixed** names, and both wrappers rename them
    (``_orig_mod.``, ``module.``). One accessor, used everywhere, so a
    state_dict never grows a wrapper prefix.
    """
    seen = 0
    while seen < 4:
        inner = getattr(model, "_orig_mod", None) or getattr(model, "module", None)
        if inner is None or not isinstance(inner, nn.Module):
            return model
        model = inner
        seen += 1
    return model


# ══════════════════════════════════════════════════════════════════════
#  Memory
# ══════════════════════════════════════════════════════════════════════


def empty_cache(device: torch.device | str | None = None) -> None:
    """Return the caching allocator's free blocks to the driver.

    Both allocators keep freed blocks in a per-size pool, which is the right
    default — but Stage 3 holds two whole extra models (the SWA probe and the
    averaged copy) and the stage boundaries free them all at once. Sweeping
    there stops the next stage from starting against a fragmented pool.

    A no-op on CPU, and on any accelerator that is not the live one.
    """
    dev = torch.device(device) if device is not None else None
    if (dev is None or dev.type == "cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    if (dev is None or dev.type == "mps") and torch.backends.mps.is_available():
        mps = getattr(torch, "mps", None)
        if mps is not None and hasattr(mps, "empty_cache"):
            mps.empty_cache()


def release_memory(device: torch.device | str | None = None, *, collect: bool = True) -> None:
    """Break reference cycles, then return the freed blocks to the driver.

    :func:`empty_cache` alone is not enough at a phase boundary, and the reason
    is the order of the two operations. The allocator only hands back blocks
    with no live tensor pointing at them; a tensor still reachable from a
    reference *cycle* — an autograd graph whose output the frame above still
    holds, an exception's ``__traceback__`` chaining back to the frame that
    raised, a module holding a hook that closes over the module — is live to the
    allocator until CPython's cycle collector runs. Calling ``empty_cache()``
    first therefore sweeps a pool that is still pinned, and the memory only
    comes back at the next automatic collection, which is generally after the
    allocation that needed it.

    Neither half is free: ``gc.collect()`` walks the process's whole object
    graph, and ``empty_cache()`` synchronises before it frees, so the blocks it
    hands back have to be re-requested from the driver by the next allocation.
    Both costs are per *call*, not per tensor, which is what makes the call site
    the whole design: this runs **per epoch and per phase boundary**, against an
    epoch of training and two full validation passes, and never per step. At
    per-step cadence the same pair would dominate the loop, because every block
    released would be re-requested by the very next batch.

    Args:
        device: Accelerator to sweep; ``None`` sweeps every available one.
        collect: Run the cycle collector first. Left on everywhere it matters;
            the flag exists for a caller that has just collected.
    """
    if collect:
        gc.collect()
    empty_cache(device)


def synchronize(device: torch.device) -> None:
    """Block until the device's queued work is done. Used only by benchmarks and teardown."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        mps = getattr(torch, "mps", None)
        if mps is not None and hasattr(mps, "synchronize"):
            mps.synchronize()
