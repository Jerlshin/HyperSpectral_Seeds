"""The two per-epoch training loops: AdamW (Stages 1-2) and SAM (Stage 3).

Both loops take an optional ``tracker``. When it is ``None``, nothing extra
happens: no diagnostic is accumulated and no device synchronisation is
added beyond the loop's normal work. When a tracker is supplied, three
families of scalars are recorded, and all three come from values the loop
already computes:

* ``loss/branch_{a,b,c,d}_{raw,weighted}`` — the per-branch terms
  ``_compute_aux_loss`` sums internally, returned alongside the total both
  before and after the per-branch weight (IC-2). Logging only the weighted
  term made the panel track the controller rather than the branch — see
  :class:`~spectralquadnet.losses.auxiliary.AuxComponents`.
* ``grad_norm/{branch_a,…,cross_interaction,arcface_head}`` — the pre-clip
  gradient norm split by owner, sampled at the same place the clip runs, and
  gated on ``cfg.tracking.log_grad_norms``.
* ``grad_norm/{preclip,postclip,clipped}_{head,fusion,backbone}`` and
  ``grad_norm/clip_fraction`` — what the per-group clip actually did (IC-6).
  ``clip_fraction`` is the number that says whether ``grad_clip`` is clipping
  outliers or renormalising every step.

All are accumulated as device tensors and resolved to floats **once per epoch**,
so the per-step cost is a handful of adds rather than a host synchronisation.

**S10 P0.3 / P0.5 (S11).** Four more families, all telemetry, none read by the
optimiser, all accumulated on the device and resolved with the rest:

* ``sched/aux_weight_applied`` — the auxiliary weight this epoch actually
  multiplied the aux term by. Until S11 the single-stage curriculum logged a
  configured 0.2 that was never applied (S10 F54).
* ``train/loss_main`` — the main-head term alone (mixup-interpolated, with label
  smoothing), without the auxiliary term; ``train/acc_dominant`` — accuracy
  against the label holding ``λ ≥ 0.5`` under mixup (``train/acc`` scores against
  the first label and is capped at ≈ 0.5); ``train/acc_plain`` — accuracy of the
  unpenalised cosine logits under a margin (``train/acc`` scores the margined
  ones). All three average over the steps that trained (finite loss).
* ``grad_norm/<module>`` — when the model declares ``grad_groups()``
  (``SpectralSeedNet``: gate, stem, tail, proj, spectral, fuse, embed, head, aux)
  the per-module pre-clip norm is logged under those labels instead of the
  four-branch prefixes, of which only ``arcface_head`` matched it.
* Every ``grad_norm/*`` epoch mean is over the steps whose norm was **finite**:
  a GradScaler overflow step (skipped by the scaler, fp16 only) used to make the
  whole epoch's mean ``inf``/``nan``. ``grad_norm/nonfinite_steps`` counts them.
  The GradNorm controller (three-stage only) still reads the unfiltered sums,
  exactly as before.

A fourth family is host-side bookkeeping rather than measurement —
``train/steps``, ``train/skipped_batches`` and ``train/epoch_s``. The skip count
is the one worth stating outright: a non-finite loss silently ``continue``\\ s
(see below), and an epoch that dropped a third of its batches to NaN otherwise
looks exactly like one that trained, because the mean is taken over
``len(loader)`` either way. Any non-zero count is additionally raised as a
``[WARN]`` line, so it appears in the log next to the epoch it happened in
rather than only as a curve nobody plotted.

Host synchronisation, and the one that is left
──────────────────────────────────────────────
A ``.item()``, a ``.cpu()`` or a Python ``if`` on a device tensor all drain the
accelerator's queue: the host stops until every kernel issued so far has
retired, and nothing the loader has prefetched can overlap across that point.
The AdamW loop used to do three per step — ``if not torch.isfinite(loss)``,
``loss.item()``, and ``(logits.argmax(1) == ya).float().mean().item()`` — and
the SAM loop four.

There is now **one** per step (two under SAM, one per objective evaluation),
and it is the finiteness test, which cannot be removed without changing
behaviour: the loop's contract is that a non-finite batch is *skipped*, and
that is a host-side control-flow decision. It is now the only one, because the
loss and the batch accuracy ride along in the same stacked transfer — the same
two floats, in the same order, resolved by the same rounding, so the epoch
means are bit-for-bit what they were.

Behaviour worth keeping in mind when reading these loops:

* **AMP stays on through the contrastive phases** (IC-7). It used to be
  ``use_amp = (supcon is None) and (scaler is not None)``, so merely *passing* a
  SupCon module dropped the whole epoch — backbone forward included — into
  fp32. On an 11 GB card already at 91% occupancy that doubles activation
  memory and the allocator starts paging: Phase 3 cost 190–405 s/epoch against
  Phases 1–2's 39 s, a 5–10× jump on identical data (CHANGES.md §7.3, §9.2).
  SupCon genuinely needs fp32 for the ``exp(·/0.1)`` reduction, but only for
  *that* — so the contrastive terms are now computed inside
  :func:`_contrastive_terms`, which re-enters fp32 for the similarity matrix
  alone after casting the ℓ2-normalised embedding up.
* **The autocast dtype is the caller's**, and defaults to ``torch.bfloat16``
  rather than to torch's per-device default of fp16. It is passed to
  ``autocast`` explicitly, so what the loop runs in is a decision
  :func:`~spectralquadnet.utils.device.resolve_amp_dtype` made against the
  hardware and wrote into the log, not a per-backend accident. fp16 is still
  reachable (``runtime.amp_dtype=fp16``) and still gets its loss scaler; it is
  no longer the default, because it is what produced the non-finite Stage-1
  losses this parameter exists to fix.
* **Non-finite losses skip the batch**, zeroing grads rather than raising, in
  both loops.
* **The optimiser steps on the accumulation boundary only**, and the EMA is
  updated in the same block, so ``ema.update`` is called once per *optimiser*
  step rather than once per batch.
* **Gradients are clipped per group, not globally** (OP-3): the head's
  ``s = 48``-amplified gradient no longer divides the backbone's effective
  learning rate.
* **Mixup and a non-zero angular margin are mutually exclusive** and the
  combination raises. Since HD-1 there is one head for every stage, so the
  test is on the *margin* rather than on which head is selected: Stage 1
  passes ``arc_m = 0``, a plain cosine classifier, which takes interpolated
  targets perfectly well. Same-class CutMix (T2-7) is label-preserving and so
  is unaffected by this guard.
* **The sub-centre balance term** ``out["balance"]`` (HD-2(ii) / T2-9) is added
  at ``cfg.model.subcenter_balance_weight`` whenever the model produced it.
* :func:`train_one_epoch_sam` returns the *first*-step loss and accuracy; the
  second (descent) step's loss is computed but only used for its gradients.
  Both steps evaluate the **same** compound objective, which is what makes the
  step a SAM step at all — see that function's docstring.
"""

from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from spectralquadnet.engine.batch import side_inputs, unpack_batch
from spectralquadnet.engine.diagnostics import (
    branch_grad_norm_tensors,
    declared_grad_groups,
    epoch_tag,
    flat_grad,
    grad_cosine,
    module_grad_norm_tensors,
)
from spectralquadnet.losses.auxiliary import (
    AuxComponents,
    GradNormAuxWeights,
    _aux_loss_weight,
    _compute_aux_loss,
)
from spectralquadnet.losses.mixup import mix_side_inputs, mixed_aug, mixed_loss
from spectralquadnet.models.ema import ModelEMA
from spectralquadnet.optim.param_groups import clip_grad_norm_by_group
from spectralquadnet.optim.sam import SAM
from spectralquadnet.tracking.base import ExperimentTracker
from spectralquadnet.utils.device import unwrap_model

if TYPE_CHECKING:  # pragma: no cover - typing only
    from spectralquadnet.config.schema import ExperimentConfig

_log = logging.getLogger(__name__)

#: How many steps per epoch contribute to ``sam/grad_cos``.
#:
#: :func:`~spectralquadnet.engine.diagnostics.flat_grad` copies every gradient
#: into one contiguous vector — ~32 MB at 7.9 M fp32 parameters — and the cosine
#: needs two of them, so measuring it on every SAM step adds ~64 MB of pure
#: copying per step to a stage that already runs two backward passes. It is a
#: diagnostic about the *stage*, not about any one batch, so it is sampled: the
#: logged value is the mean over ~32 evenly spaced steps rather than over all of
#: them. Nothing consumes it but the tracker.
GRAD_COS_SAMPLES: int = 32


def _contrastive_terms(
    emb: torch.Tensor,
    labels: torch.Tensor,
    supcon: nn.Module | None,
    proto: nn.Module | None,
) -> tuple[torch.Tensor | float, torch.Tensor | float]:
    """SupCon and ProtoNCE in fp32, inside an otherwise-autocast epoch (IC-7).

    The embedding arrives ℓ2-normalised in the autocast dtype. Both losses build
    a ``B × B`` (or ``B × |C|``) similarity matrix and exponentiate it at
    ``τ = 0.10``, so the argument reaches ±10 and the reduction is over 128
    terms — bf16's 8 mantissa bits put a relative error of ~4e-3 on each, which
    is the same order as the logit gaps the loss is trying to rank.

    Casting *here* rather than disabling autocast for the epoch is the whole
    point: the backbone forward that produced ``emb`` keeps its bf16 kernels,
    and only the few megaflops of the similarity matrix run wide. The returned
    values are fp32 and compose into the total loss unchanged.

    Returns:
        ``(supcon_loss, proto_loss)``, each ``0.0`` when its module is absent so
        the caller's arithmetic does not need a branch.
    """
    if supcon is None and proto is None:
        return 0.0, 0.0
    with autocast(device_type=emb.device.type, enabled=False):
        emb32 = emb.float()
        sc = supcon(emb32, labels) if supcon is not None else 0.0
        pt = proto(emb32, labels) if proto is not None else 0.0
    return sc, pt


def _resolve_step_scalars(loss: torch.Tensor, accuracy: torch.Tensor) -> tuple[float, float]:
    """The step's loss and accuracy as host floats, in **one** synchronisation.

    Stacking is not a numerical operation here: both are already 0-dim float32,
    ``stack`` only lays them out contiguously, and ``.tolist()`` reads the same
    two floats ``.item()`` would have read one at a time. What it buys is a
    single queue drain per step instead of two — and, with the finiteness test
    reading ``loss`` from this same transfer, one instead of three.
    """
    values = torch.stack([loss.detach().float(), accuracy.detach().float()]).cpu().tolist()
    return float(values[0]), float(values[1])


def _advance_sampler_epoch(loader: Any, epoch: int) -> None:
    """Re-seed a ``DistributedSampler``'s shuffle for this epoch.

    ``DistributedSampler`` permutes with ``seed + epoch`` and only learns the
    epoch through ``set_epoch``; without the call every epoch of a DDP run
    replays epoch 0's order and each rank sees the same shard of the same
    batches all run long. A single-process loader has no such sampler and the
    balanced batch samplers advance their own stream, so both are untouched.
    """
    sampler = getattr(loader, "sampler", None)
    if isinstance(sampler, torch.utils.data.DistributedSampler):
        sampler.set_epoch(int(epoch))


def _batch_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Top-1 accuracy over the batch, left on the device as a 0-dim tensor."""
    with torch.no_grad():
        return (logits.detach().argmax(1) == targets).float().mean()


def train_one_epoch(
    cfg: ExperimentConfig | Any,
    model: nn.Module,
    loader: DataLoader[Any],
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler | None,
    ema: ModelEMA,
    device: torch.device,
    scheduler: optim.lr_scheduler._LRScheduler | None = None,
    use_mixup: bool = True,
    mixup_alpha: float = 0.4,
    supcon: nn.Module | None = None,
    supcon_weight: float = 0.0,
    proto: nn.Module | None = None,
    proto_weight: float = 0.0,
    accum_steps: int = 1,
    arc_m: float | None = None,
    current_ep: int = 0,
    total_ep: int = 100,
    tracker: ExperimentTracker | None = None,
    aux_weights: GradNormAuxWeights | None = None,
    amp_dtype: torch.dtype = torch.bfloat16,
    aux_weight: float | None = None,
) -> tuple[float, float]:
    """Run one AdamW training epoch (Stage 1 or Stage 2).

    Combines the main classification loss with the per-branch auxiliary
    loss, the sub-centre balance term, and optionally SupCon/ProtoNCE
    contrastive terms; supports mixup and gradient accumulation. See the module
    docstring for cross-cutting behaviour (AMP-disable-under-SupCon,
    non-finite-loss skip, EMA update cadence, per-group clipping).

    Args:
        aux_weights: Optional GradNorm state (OP-2 / T2-6). When given, the
            per-branch auxiliary weights come from it instead of the fixed
            ``A/B = 2x`` vector, and it is updated once at the end of the epoch
            from that epoch's mean per-branch gradient norms.
        amp_dtype: Autocast dtype, used only when ``scaler`` is given and no
            SupCon loss is. Defaults to ``torch.bfloat16``; pair fp16 with an
            *enabled* scaler and bf16 with a disabled one —
            :func:`~spectralquadnet.utils.device.make_grad_scaler` builds the
            matching pair.
        aux_weight: The auxiliary-loss weight for this epoch. ``None`` (the
            three-stage callers) applies :func:`_aux_loss_weight`'s decay, as
            this loop always did; the single-stage curriculum passes
            :func:`~spectralquadnet.losses.auxiliary.single_stage_aux_weight`.
            Whatever is applied is logged as ``sched/aux_weight_applied``.

    Returns:
        ``(mean_loss, mean_accuracy)`` over the epoch.
    """
    started = time.perf_counter()
    _advance_sampler_epoch(loader, current_ep)
    model.train()
    # `model` may be a DDP and/or `torch.compile` wrapper; the gradient
    # grouping helpers below match on **unprefixed** parameter names
    # (`branch_a.`, `arcface_head.`, …), which a wrapper's `module.` /
    # `_orig_mod.` prefix would defeat silently — every group would come back
    # empty and the per-group clip would become no clip at all.
    core = unwrap_model(model)
    total_loss = total_acc = 0.0
    optimizer.zero_grad(set_to_none=True)

    # IC-7: the SupCon module no longer vetoes autocast for the epoch. The
    # contrastive terms take their own fp32 region in `_contrastive_terms`.
    use_amp = scaler is not None
    aux_w = _aux_loss_weight(cfg, current_ep, total_ep) if aux_weight is None else float(aux_weight)
    balance_w = float(getattr(getattr(cfg, "model", None), "subcenter_balance_weight", 0.0))
    branch_w = aux_weights.weights if aux_weights is not None else None
    clip_partition = str(getattr(cfg, "clip_partition", "legacy"))

    if use_mixup and (arc_m is None or arc_m > 0.0):
        raise ValueError("Mixup cannot be used with a non-zero ArcFace margin.")
    # Whether the head scores the target with a margin this epoch: labels reach
    # it (no mixup) and the margin is non-zero, or per-class (`None`).
    margin_active = (not use_mixup) and (arc_m is None or arc_m > 0.0)

    # Diagnostics are off entirely without a tracker; the branch norms are the
    # exception, because GradNorm consumes them whether or not anyone is
    # watching.
    want_diag = tracker is not None
    want_grad_norms = want_diag and bool(
        getattr(getattr(cfg, "tracking", None), "log_grad_norms", False)
    )
    want_gradnorm = aux_weights is not None and aux_weights.alpha != 0.0
    need_branch = want_grad_norms or want_gradnorm
    # Per-module groups when the model declares them (S10 P0.5); the four-branch
    # prefixes otherwise — which is also what GradNorm reads.
    module_groups = declared_grad_groups(core)
    diag_aux: dict[str, torch.Tensor] = {}
    diag_branch: dict[str, torch.Tensor] = {}
    diag_clip: dict[str, torch.Tensor] = {}
    # Finite-step sums and counts for the logged gradient norms, and the
    # unfiltered sums GradNorm has always read (S10 B14).
    norm_sums: dict[str, torch.Tensor] = {}
    norm_counts: dict[str, torch.Tensor] = {}
    diag_train: dict[str, torch.Tensor] = {}
    n_aux = n_branch = n_clip = n_trained = 0
    n_skipped = 0

    for step, batch in enumerate(loader):
        if step == 0:
            _log.info(
                "[TRAIN] First batch retrieved (ep=%d, batch_size=%s)",
                current_ep,
                next(iter(batch)).shape[0] if batch else "?",
            )
            _first_fwd_t = time.perf_counter()
        x, y, mask, morph = unpack_batch(batch, device)
        # Mixup interpolates two patches, so it interpolates their fill maps and
        # their morphometrics too — both are continuous quantities and the
        # composite's foreground genuinely is the mixture. The permutation is
        # `mixed_aug`'s, returned so the side inputs cannot be mixed with a
        # different one than the pixels were.
        perm: torch.Tensor | None = None
        if use_mixup:
            x_in, ya, yb, lam, perm = mixed_aug(x, y, mixup_alpha, return_perm=True)
        else:
            x_in, ya, yb, lam = x, y, y, 1.0
        side = side_inputs(*mix_side_inputs(mask, morph, perm, lam))
        aux_parts: AuxComponents | None = None

        with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            # ── SupCon / ProtoNCE path ─────────────────────────────────
            if supcon is not None:
                out = model(x_in, ya, return_embed=True, arc_m=arc_m, **side)
                logits = out["main"] if isinstance(out, dict) else out[0]
                emb = out.get("emb") if isinstance(out, dict) else out[1]
                if emb is None:
                    # `return_embed=True` was passed, so a model that returns a
                    # dict must put `emb` in it. Reaching here means a custom
                    # architecture broke the contract, and a silent `None` would
                    # surface as a contrastive term that is quietly zero.
                    raise RuntimeError(
                        f"{type(unwrap_model(model)).__name__} did not return an embedding "
                        "under return_embed=True, but a contrastive loss needs one."
                    )

                cls_l = criterion(logits, ya)
                l_main_term: torch.Tensor = cls_l
                sc_l, pt_l = _contrastive_terms(emb, ya, supcon, proto)

                if isinstance(out, dict):
                    aux_l, aux_parts = _compute_aux_loss(
                        criterion,
                        out,
                        ya,
                        yb,
                        lam,
                        use_mixup=False,
                        return_components=True,
                        weights=branch_w,
                    )
                else:
                    aux_l = torch.zeros((), device=device)

                loss = (
                    (1 - supcon_weight - proto_weight) * cls_l
                    + supcon_weight * sc_l
                    + proto_weight * pt_l
                    + aux_w * aux_l
                )

            # ── Standard CE / Focal path ───────────────────────────────
            else:
                # Under mixup the label is a pair, so no margin and no balance
                # term can be indexed by it; the head falls back to the plain
                # scaled cosine, which is what makes mixup admissible at all.
                arc_labels = None if use_mixup else ya
                out = model(x_in, labels=arc_labels, arc_m=arc_m, **side)

                if isinstance(out, dict):
                    l_main = mixed_loss(criterion, out["main"], ya, yb, lam)
                    l_main_term = l_main
                    aux_l, aux_parts = _compute_aux_loss(
                        criterion,
                        out,
                        ya,
                        yb,
                        lam,
                        use_mixup,
                        return_components=True,
                        weights=branch_w,
                    )
                    loss = l_main + aux_w * aux_l
                    logits = out["main"]
                else:
                    logits = out
                    loss = mixed_loss(criterion, logits, ya, yb, lam)
                    l_main_term = loss

            if isinstance(out, dict) and balance_w > 0.0 and "balance" in out:
                loss = loss + balance_w * out["balance"]

        # The step's two host-side numbers, fetched together. The accuracy is
        # read here rather than after the optimiser step because `logits` is not
        # modified by either — its value at this point is its value at the end
        # of the step — and folding it into the finiteness test's transfer is
        # what removes two of the three per-step synchronisations.
        loss_value, acc_value = _resolve_step_scalars(loss, _batch_accuracy(logits, ya))

        if not math.isfinite(loss_value):
            n_skipped += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        if want_diag and aux_parts:
            _accumulate(diag_aux, aux_parts.scalars())
            n_aux += 1
        if want_diag:
            _accumulate(
                diag_train,
                _train_telemetry(out, logits, l_main_term, ya, yb, lam, margin_active),
            )
            n_trained += 1

        # `use_amp` already implies `scaler is not None`, but mypy cannot see the
        # correlation between the two locals, hence the ignores below.
        if use_amp:
            scaler.scale(loss / accum_steps).backward()  # type: ignore[union-attr]
        else:
            (loss / accum_steps).backward()

        if (step + 1) % accum_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)  # type: ignore[union-attr]
            # Sampled after unscale_ and before the clip, so the norms are the
            # true pre-clip ones.
            if need_branch:
                norms = (
                    branch_grad_norm_tensors(core)
                    if module_groups is None
                    else module_grad_norm_tensors(core, module_groups)
                )
                tagged = {f"grad_norm/{k}": v for k, v in norms.items()}
                # Unfiltered, exactly as before: GradNorm's update reads these.
                _accumulate(diag_branch, tagged)
                if want_grad_norms:
                    _accumulate_finite(norm_sums, norm_counts, tagged)
                n_branch += 1
            clip_report = clip_grad_norm_by_group(core, cfg.grad_clip, clip_partition)
            if want_diag:
                _accumulate_finite(norm_sums, norm_counts, clip_report.scalars())
                _accumulate(diag_clip, {"grad_norm/nonfinite_steps": _nonfinite(clip_report)})
                n_clip += 1
            if use_amp:
                scaler.step(optimizer)  # type: ignore[union-attr]
                scaler.update()  # type: ignore[union-attr]
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if ema:
                # `core`, not `model`: the shadow's parameters are named
                # without a wrapper prefix, and `ModelEMA` pairs them by name.
                ema.update(core)

        total_loss += loss_value
        total_acc += acc_value

        if step == 0:
            _log.info(
                "[TRAIN] First step completed in %.1f s (ep=%d)",
                time.perf_counter() - _first_fwd_t,
                current_ep,
            )

    n = max(len(loader), 1)
    scalars = _diagnostic_scalars((diag_aux, n_aux), (diag_train, n_trained))
    scalars.update(_finite_means(norm_sums, norm_counts))
    # A count, not a mean: summed over the epoch's optimiser steps.
    scalars.update(_diagnostic_scalars((diag_clip, 1 if n_clip else 0)))
    scalars.update(_update_aux_weights(aux_weights, diag_branch, n_branch))
    if tracker is not None:
        scalars["sched/aux_weight_applied"] = float(aux_w)
        scalars.update(_epoch_bookkeeping(n, n_skipped, time.perf_counter() - started))
        tracker.log_scalars(scalars, step=current_ep)
        _warn_on_skips(tracker, n_skipped, n, current_ep, total_ep)

    return total_loss / n, total_acc / n


def _train_telemetry(
    out: Any,
    logits: torch.Tensor,
    l_main: torch.Tensor,
    ya: torch.Tensor,
    yb: torch.Tensor,
    lam: float,
    margin_active: bool = False,
) -> dict[str, torch.Tensor]:
    """``train/loss_main``, ``train/acc_dominant`` and ``train/acc_plain`` for one step.

    Device tensors, no host synchronisation, no gradient — telemetry only
    (S10 P0.3). ``acc_dominant`` scores against the label whose mixup weight is
    at least one half (``ya`` when mixup is off, since ``lam = 1``).
    ``acc_plain`` scores unpenalised logits: the model's ``main_plain`` under a
    margin, the logits themselves when no margin is applied (nothing to
    remove) — and is **omitted** when a margin is applied by a model that
    provides no ``main_plain`` (the four-branch control), rather than reporting
    the margined accuracy under a name that says otherwise.
    """
    with torch.no_grad():
        dominant = ya if lam >= 0.5 else yb
        plain = out.get("main_plain") if isinstance(out, dict) else None
        values = {
            "train/loss_main": l_main.detach().float(),
            "train/acc_dominant": _batch_accuracy(logits, dominant),
        }
        if plain is not None:
            values["train/acc_plain"] = _batch_accuracy(plain, ya)
        elif not margin_active:
            values["train/acc_plain"] = _batch_accuracy(logits, ya)
        return values


def train_one_epoch_sam(
    cfg: ExperimentConfig | Any,
    model: nn.Module,
    loader: DataLoader[Any],
    sam_opt: SAM,
    criterion: nn.Module,
    device: torch.device,
    supcon: nn.Module | None = None,
    supcon_weight: float = 0.0,
    proto: nn.Module | None = None,
    proto_weight: float = 0.0,
    arc_m: float | None = None,
    aux_weight: float = 0.0,  # fixed aux weight for Stage 3 (typically small)
    current_ep: int = 0,
    ema: ModelEMA | None = None,
    tracker: ExperimentTracker | None = None,
    aux_weights: GradNormAuxWeights | None = None,
) -> tuple[float, float]:
    """Run one Sharpness-Aware Minimisation training epoch (Stage 3).

    Each batch runs SAM's two-step contract: an ascent step to the local
    worst case (:meth:`SAM.first_step`), then a descent step from the
    original weights using the gradient computed there
    (:meth:`SAM.second_step`). SAM is not compatible with AMP, so this loop
    always runs in fp32.

    **Both steps evaluate the same objective** — focal + ``supcon_weight``·SupCon
    + ``proto_weight``·ProtoNCE + ``aux_weight``·aux + the sub-centre balance
    term. SAM minimises ``max_{||eps||<=rho} L(theta + eps)`` for one ``L``; the
    descent step used to drop everything but the focal term, which replaces the
    sharpness penalty ``rho·H·ĝ_D`` with ``rho·H·ĝ_A`` and leaves the component
    of ``ĝ_A`` orthogonal to ``ĝ_D`` acting as curvature-amplified noise with no
    descent guarantee (IMPROVEMENT_PLAN §2.5.1 C-6, §3.6 OP-4.1). The ProtoNCE
    term is applied rather than accepted and discarded (OP-7 / N-1e).

    ``cos(ĝ_A, ĝ_D)`` is logged as ``sam/grad_cos`` whenever
    ``cfg.tracking.log_grad_norms`` is set — the measurement F-9 asks for.

    Args:
        ema: Optional EMA shadow, updated once per optimiser step exactly as
            :func:`train_one_epoch` does. Stage 3 keeps its shadow alive so the
            SWA average can be *compared* against it rather than overwriting it
            unscored (§2.5.7 C-7f).
        aux_weights: Optional GradNorm state (OP-2 / T2-6), used and updated
            exactly as in :func:`train_one_epoch`.

    Returns:
        ``(mean_loss, mean_accuracy)`` over the epoch, computed from the
        first (ascent) step's forward pass.
    """
    started = time.perf_counter()
    _advance_sampler_epoch(loader, current_ep)
    torch.set_default_dtype(torch.float32)
    model.train()
    core = unwrap_model(model)  # see `train_one_epoch` — grouping needs raw names
    total_loss = total_acc = 0.0
    balance_w = float(getattr(getattr(cfg, "model", None), "subcenter_balance_weight", 0.0))
    branch_w = aux_weights.weights if aux_weights is not None else None

    # Diagnostics are off entirely without a tracker.
    want_diag = tracker is not None
    want_grad_norms = want_diag and bool(
        getattr(getattr(cfg, "tracking", None), "log_grad_norms", False)
    )
    need_branch = want_grad_norms or (aux_weights is not None and aux_weights.alpha != 0.0)
    diag_aux: dict[str, torch.Tensor] = {}
    diag_branch: dict[str, torch.Tensor] = {}
    diag_clip: dict[str, torch.Tensor] = {}
    diag_cos: dict[str, torch.Tensor] = {}
    n_aux = n_branch = n_clip = n_cos = 0
    # Both of SAM's steps can produce a non-finite loss, and they mean different
    # things: an ascent skip is a bad batch, a descent skip is a batch the
    # rho-perturbation itself blew up. Counted together for the log line, since
    # either way the batch contributed no update.
    n_skipped = 0

    # The embedding is only materialised when a term actually consumes it.
    want_embed = (supcon is not None) or (proto is not None)

    def _objective(
        x: torch.Tensor, y: torch.Tensor, side: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, AuxComponents | None]:
        """The one Stage-3 objective, evaluated at the model's current weights."""
        out = model(x, labels=y, arc_m=arc_m, return_embed=want_embed, **side)
        logits = out["main"] if isinstance(out, dict) else out
        emb = out.get("emb") if isinstance(out, dict) else None

        loss = criterion(logits, y)
        if emb is not None:
            sc_l, pt_l = _contrastive_terms(emb, y, supcon, proto)
            loss = loss + supcon_weight * sc_l + proto_weight * pt_l
        parts: AuxComponents | None = None
        if isinstance(out, dict) and aux_weight > 0.0:
            aux_l, parts = _compute_aux_loss(
                criterion, out, y, y, 1.0, use_mixup=False, return_components=True, weights=branch_w
            )
            loss = loss + aux_weight * aux_l
        if isinstance(out, dict) and balance_w > 0.0 and "balance" in out:
            loss = loss + balance_w * out["balance"]
        return loss, logits, parts

    clip_partition = str(getattr(cfg, "clip_partition", "legacy"))
    norm_sums: dict[str, torch.Tensor] = {}
    norm_counts: dict[str, torch.Tensor] = {}

    def _clip(sample: bool) -> None:
        """Per-group clip, recording the clip report on the ascent step only."""
        nonlocal n_clip
        clip_report = clip_grad_norm_by_group(core, cfg.grad_clip, clip_partition)
        if sample and want_diag:
            _accumulate_finite(norm_sums, norm_counts, clip_report.scalars())
            _accumulate(diag_clip, {"grad_norm/nonfinite_steps": _nonfinite(clip_report)})
            n_clip += 1

    # Even sampling of the gradient-cosine diagnostic — see GRAD_COS_SAMPLES.
    cos_stride = max(1, len(loader) // GRAD_COS_SAMPLES) if want_grad_norms else 0

    for step, batch in enumerate(loader):
        if step == 0:
            _log.info(
                "[TRAIN-SAM] First batch retrieved (ep=%d, batch_size=%s)",
                current_ep,
                next(iter(batch)).shape[0] if batch else "?",
            )
            _first_sam_fwd_t = time.perf_counter()
        x, y, mask, morph = unpack_batch(batch, device)
        side = side_inputs(mask, morph)

        # ── SAM first step (ascent) ───────────────────────────────────
        sam_opt.zero_grad()
        loss, logits, aux_parts = _objective(x, y, side)

        # One synchronisation for the ascent step, carrying both the finiteness
        # test and the epoch's loss/accuracy contribution — see the module
        # docstring. The descent step needs a second for its own test.
        loss_value, acc_value = _resolve_step_scalars(loss, _batch_accuracy(logits, y))

        if not math.isfinite(loss_value):
            n_skipped += 1
            sam_opt.zero_grad()
            continue

        if want_diag and aux_parts:
            _accumulate(diag_aux, aux_parts.scalars())
            n_aux += 1

        # `Tensor.backward` is untyped in this torch's stubs. The AdamW loop
        # above only escapes the same error because `nn.Module.__call__` widens
        # its loss to `Any`; `_objective`'s annotation keeps this one precise.
        loss.backward()  # type: ignore[no-untyped-call]
        # Pre-clip, and on the ascent step: these are the gradients SAM uses to
        # find the adversarial weight perturbation.
        if need_branch:
            tagged = {f"grad_norm/{k}": v for k, v in branch_grad_norm_tensors(core).items()}
            _accumulate(diag_branch, tagged)
            if want_grad_norms:
                _accumulate_finite(norm_sums, norm_counts, tagged)
            n_branch += 1
        _clip(sample=True)
        # Captured after the clip, which rescales each group by a scalar and so
        # cannot change the direction the cosine below measures within a group.
        sample_cos = cos_stride > 0 and step % cos_stride == 0
        g_ascent = flat_grad(core) if sample_cos else None
        sam_opt.first_step(zero_grad=True)

        # ── SAM second step (descent, same objective) ─────────────────
        loss2, _logits2, _ = _objective(x, y, side)
        if not torch.isfinite(loss2):
            n_skipped += 1
            # Put the weights back before skipping: `first_step` has already
            # moved them, and the next batch's `first_step` would overwrite the
            # cached originals.
            sam_opt.restore(zero_grad=True)
            continue

        loss2.backward()  # type: ignore[no-untyped-call]
        _clip(sample=False)
        if g_ascent is not None:
            _accumulate(diag_cos, {"sam/grad_cos": grad_cosine(g_ascent, flat_grad(core))})
            n_cos += 1
        sam_opt.second_step(zero_grad=True)
        if ema is not None:
            ema.update(core)  # unprefixed names — see `train_one_epoch`

        total_loss += loss_value
        total_acc += acc_value

        if step == 0:
            _log.info(
                "[TRAIN-SAM] First step completed in %.1f s (ep=%d)",
                time.perf_counter() - _first_sam_fwd_t,
                current_ep,
            )

    n = max(len(loader), 1)
    scalars = _diagnostic_scalars((diag_aux, n_aux), (diag_cos, n_cos))
    scalars.update(_finite_means(norm_sums, norm_counts))
    scalars.update(_diagnostic_scalars((diag_clip, 1 if n_clip else 0)))
    scalars.update(_update_aux_weights(aux_weights, diag_branch, n_branch))
    if tracker is not None:
        scalars.update(_epoch_bookkeeping(n, n_skipped, time.perf_counter() - started))
        tracker.log_scalars(scalars, step=current_ep)
        _warn_on_skips(tracker, n_skipped, n, current_ep, None)

    return total_loss / n, total_acc / n


# ══════════════════════════════════════════════════════════════════════
#  Diagnostic accumulation helpers
# ══════════════════════════════════════════════════════════════════════


def _epoch_bookkeeping(steps: int, skipped: int, seconds: float) -> dict[str, float]:
    """Host-side facts about the epoch: how many batches, how many dropped, how long.

    No device work and no synchronisation — three Python numbers the loop
    already had. They ride in the same ``log_scalars`` call as the measured
    diagnostics so that a run's curves and its throughput share a step index.
    """
    return {
        "train/steps": float(steps),
        "train/skipped_batches": float(skipped),
        "train/epoch_s": float(seconds),
    }


def _warn_on_skips(
    tracker: ExperimentTracker, skipped: int, steps: int, current_ep: int, total_ep: int | None
) -> None:
    """Say it out loud when an epoch dropped batches to a non-finite loss.

    Silent on the overwhelmingly common ``skipped == 0``. When it is not zero
    the count belongs on the console: the returned epoch mean divides by
    ``len(loader)`` regardless of how many batches actually contributed, so a
    diverging run reads as a *falling* loss right up until it is all NaN.
    """
    if skipped <= 0:
        return
    tracker.log_message(
        f"{epoch_tag(current_ep, total_ep)}{skipped}/{steps} batches skipped "
        "(non-finite loss); the epoch mean is over all batches regardless",
        level="warn",
    )


def _accumulate(sink: dict[str, torch.Tensor], values: dict[str, torch.Tensor]) -> None:
    """Add ``values`` into ``sink`` in place, detached and on-device.

    Nothing is transferred to the host here — this runs every step, and keys
    are already the final tracker tags.
    """
    for key, value in values.items():
        detached = value.detach()
        sink[key] = detached if key not in sink else sink[key] + detached


def _accumulate_finite(
    sums: dict[str, torch.Tensor],
    counts: dict[str, torch.Tensor],
    values: dict[str, torch.Tensor],
) -> None:
    """:func:`_accumulate`, skipping non-finite values and counting the rest.

    A GradScaler overflow step has an ``inf`` pre-clip norm; the scaler skips
    that step, so it contributes nothing to training, and it should contribute
    nothing to the epoch's mean either (S10 B14). Still on the device: the
    test is a ``where``, not a Python branch.
    """
    for key, value in values.items():
        v = value.detach().float()
        finite = torch.isfinite(v)
        kept = torch.where(finite, v, torch.zeros_like(v))
        n = finite.to(v.dtype)
        sums[key] = kept if key not in sums else sums[key] + kept
        counts[key] = n if key not in counts else counts[key] + n


def _finite_means(
    sums: dict[str, torch.Tensor], counts: dict[str, torch.Tensor]
) -> dict[str, float]:
    """Resolve :func:`_accumulate_finite`'s sums to means, in one transfer.

    A key whose every step was non-finite resolves to ``nan`` — the honest
    answer — rather than to 0.
    """
    if not sums:
        return {}
    keys = list(sums)
    stacked = torch.stack([sums[k] / counts[k] for k in keys]).cpu().tolist()
    return {k: float(v) for k, v in zip(keys, stacked, strict=True)}


def _nonfinite(report: Any) -> torch.Tensor:
    """1.0 when any group's pre-clip norm was non-finite on this step, else 0.0."""
    norms = list(report.preclip.values())
    if not norms:
        return torch.zeros(())
    return (~torch.isfinite(torch.stack([n.detach().float() for n in norms])).all()).float()


def _diagnostic_scalars(*groups: tuple[dict[str, torch.Tensor], int]) -> dict[str, float]:
    """Resolve one epoch's accumulated diagnostics to means, in a single sync.

    Each ``(sums, count)`` pair contributes ``sums[k] / count`` under ``k``
    verbatim. Empty when nothing was accumulated, so callers can skip the
    tracker call entirely.
    """
    keys: list[str] = []
    tensors: list[torch.Tensor] = []
    for sums, count in groups:
        if not count:
            continue
        for key, value in sums.items():
            keys.append(key)
            tensors.append(value / count)
    if not keys:
        return {}
    values = torch.stack(tensors).cpu().tolist()
    return {k: float(v) for k, v in zip(keys, values, strict=True)}


def _update_aux_weights(
    aux_weights: GradNormAuxWeights | None,
    branch_sums: dict[str, torch.Tensor],
    count: int,
) -> dict[str, float]:
    """Close OP-2's feedback loop once, at the end of the epoch.

    Returns the resulting ``aux_weight/branch_*`` scalars — so ``omega``
    becomes a logged time series rather than a constant nobody can audit —
    or ``{}`` when GradNorm is off or the epoch produced no gradient.
    """
    if aux_weights is None or not count:
        return {}
    wanted = {f"grad_norm/branch_{b}" for b in "abcd"}
    means = {
        key.removeprefix("grad_norm/"): value / count
        for key, value in branch_sums.items()
        if key in wanted
    }
    if not means:
        return {}
    resolved = torch.stack(list(means.values())).cpu().tolist()
    aux_weights.update(dict(zip(means.keys(), resolved, strict=True)))
    return aux_weights.scalars()
