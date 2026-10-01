"""Score the held-out split **once**, with and without TTA (CHANGES §19.4).

Two things this function is careful about, and both were defects in the audited
project:

**It evaluates the weights that won the selection score.** Every stage saves on
``max(F1_live, F1_ema)`` and writes that max to the sidecar as ``val_f1``, but
this function used to evaluate the EMA shadow unconditionally. When the live
model won the max, the reported number came from a model that had never scored
it, and the artifacts did not record which had won — making the mismatch
unfalsifiable after the fact. The stages record ``best_source`` and this honours
it. Bundles written before that key default to ``"ema"``, which is exactly the
behaviour they were produced under.

**It reports an interval, a confusion matrix and a per-class table, not a
number.** Sampling noise on a ~1,300-patch split is ±0.020 at 95%; the audited
run's entire Stage-2 + Stage-3 gain was +0.005. A bare macro-F1 on this dataset
is not interpretable and this function refuses to emit one.

Reporting discipline
────────────────────
The split scored here is ``cfg.evaluation.report_split``, and under the grouped
protocol that is ``val ∪ test`` **together**: they are two halves of the same
held-out acquisition bundle, so they are not independent of each other and
scoring one after selecting on the other would still be partially
self-fulfilling. Selection happened on ``calib``. This runs once, after every
design decision is frozen — that ordering is the claim, and nothing in the code
can enforce it, so the run records which split selected and which was reported
and the paper states it.

The session breakdown
─────────────────────
Given a :class:`~spectralquadnet.reporting.session.SessionMap`, every scored
variant is also broken down by acquisition session — same- vs cross-session
recall, attraction to the kernel's own session, and session-prediction entropy
(:mod:`spectralquadnet.reporting.session`). On this dataset 73 of 90 varieties
share a session across their two bundles, so the grouped macro-F1 alone cannot
say whether a model recognises varieties or sessions. The breakdown is a
measurement only: it reads the same predictions and changes nothing.

Exactly once per kernel, with its logits (D18, S10 P0.1)
────────────────────────────────────────────────────────
Under DDP the joined predictions are de-duplicated before anything is scored
(:func:`~spectralquadnet.data.loaders.dedup_index`): ``DistributedSampler`` pads
an odd-sized split with its first kernel, which used to be scored twice. The
selected weights' **logits** are written for the reported split and — given a
``calib_ldr`` — for ``calib``, each with and without TTA, as float16
``results/logits_<split>_<variant>.npz`` (``logits``, ``targets``, ``rows``).
No metric is computed on calib here; its logits are for calibration and margin
analyses that need the selection split's scores, not its argmax.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from spectralquadnet.data.loaders import dedup_index, eval_row_order
from spectralquadnet.engine.batch import side_inputs, unpack_batch
from spectralquadnet.engine.checkpoint import load_ckpt
from spectralquadnet.engine.diagnostics import hardest_classes_report
from spectralquadnet.engine.tta import tta_predict
from spectralquadnet.models.ema import ModelEMA
from spectralquadnet.reporting.artifacts import RunArtifacts, publish
from spectralquadnet.reporting.figures import render_run_figures
from spectralquadnet.reporting.metrics import ClassificationResult, score
from spectralquadnet.reporting.session import SessionMap, SessionReport, session_report
from spectralquadnet.tracking.base import ExperimentTracker, NullTracker
from spectralquadnet.utils.distributed import DistContext, gather_concat

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch.nn as nn

    from spectralquadnet.config.schema import ExperimentConfig

#: What a bundle without a recognised ``best_source`` is assumed to have been
#: selected on — the pre-Tier-1 behaviour, so archived runs keep reproducing.
DEFAULT_BEST_SOURCE = "ema"

#: Every ``best_source`` a stage writes. ``"live"`` names the bundle's ``model``
#: slot; ``"ema"`` and Stage 3's ``"swa"`` both name its ``ema`` slot, which is
#: where Stage 3 puts whichever of its two averages scored higher.
KNOWN_BEST_SOURCES = frozenset({"live", "ema", "swa"})


def final_evaluation(
    cfg: ExperimentConfig | Any,
    model: nn.Module,
    ema: ModelEMA,
    test_ldr: DataLoader[Any],
    device: torch.device,
    best_ckpt: str,
    tracker: ExperimentTracker | None = None,
    dist: DistContext | None = None,
    run_summary: dict[str, Any] | None = None,
    sessions: SessionMap | None = None,
    calib_ldr: DataLoader[Any] | None = None,
) -> dict[str, ClassificationResult]:
    """Load the selected weights and score the reporting split, ±TTA.

    Args:
        best_ckpt: Checkpoint to load. Its ``best_source`` decides which of the
            two weight sets is evaluated.
        run_summary: Run identity — architecture, protocol, fold, seed,
            parameter count — written into ``results/run.json`` so an
            aggregator can build a table from the results tree alone.
        sessions: The fold's session map. When given, each variant is also
            scored by acquisition session and the breakdown is written beside
            the headline metrics. ``None`` skips it.
        calib_ldr: The selection split's loader. When given (and
            ``evaluation.save_logits``), the selected weights' calib logits are
            written ±TTA. Nothing is scored on it.

    Returns:
        ``{"no_tta": …, "tta": …}`` (the second only when
        ``cfg.evaluation.tta``), so a caller in-process can compare arms
        without re-reading the files.
    """
    trk = tracker if tracker is not None else NullTracker()
    dist = dist or DistContext()
    ckpt = load_ckpt(best_ckpt, model, ema, device)

    recorded = ckpt.get("best_source")
    source = str(recorded) if recorded in KNOWN_BEST_SOURCES else DEFAULT_BEST_SOURCE
    if recorded is not None and recorded not in KNOWN_BEST_SOURCES:
        trk.log_message(
            f"Unrecognised best_source {recorded!r} — evaluating the {source} weights",
            level="warn",
        )
    eval_model = model if source == "live" else ema.shadow
    eval_model.eval()

    report_split = str(getattr(cfg.evaluation, "report_split", "test"))
    select_split = (run_summary or {}).get("select_split", "?")
    want_tta = bool(getattr(cfg.evaluation, "tta", True))
    want_logits = bool(getattr(cfg.evaluation, "save_logits", True))
    n_boot = int(getattr(cfg.evaluation, "bootstrap_samples", 0))
    epoch = int(ckpt.get("epoch", 0))

    trk.banner(
        "FINAL EVALUATION — scored once, after freezing",
        [
            f"Schema v{ckpt.get('schema_version', 1)} ({ckpt.get('arch', 'spectral_quadnet')})  |  "
            f"Checkpoint: ep {epoch} | {ckpt['stage']} | "
            f"selection F1={ckpt.get('val_f1', 0):.4f}",
            f"Selected weights: {source}"
            + (
                "" if recorded in KNOWN_BEST_SOURCES else "  (assumed — not recorded in the bundle)"
            ),
            f"Selected on: {select_split}  →  Reported on: {report_split} "
            f"({len(test_ldr.dataset)} patches)",  # type: ignore[arg-type]
            (
                f"TTA: {cfg.tta_spatial} spatial + {cfg.tta_spectral} spectral "
                f"= {cfg.tta_spatial + cfg.tta_spectral} views, reported separately"
                if want_tta
                else "TTA: off"
            ),
            (
                f"Bootstrap: {n_boot} resamples, 95% percentile CI"
                if n_boot
                else "Bootstrap: off — the reported number carries no interval"
            ),
        ],
    )

    artifacts = RunArtifacts.for_run(cfg.output_dir) if dist.is_main else None
    results: dict[str, ClassificationResult] = {}
    session_results: dict[str, SessionReport] = {}
    rows = _rows_after_dedup(test_ldr)
    if sessions is not None and rows is None:
        trk.log_message(
            "Session breakdown skipped: the report loader's row order cannot be established",
            level="warn",
        )

    variants: list[tuple[str, bool]] = [("no_tta", False)]
    if want_tta:
        variants.append(("tta", True))

    for key, use_tta in variants:
        logits, targets = _predict(cfg, eval_model, test_ldr, device, dist, use_tta)
        preds = logits.argmax(1)
        split_tag = f"{report_split}_{key}"
        result = score(
            targets,
            preds,
            num_classes=int(cfg.data.num_classes),
            split=split_tag,
            n_boot=n_boot,
            seed=int(cfg.seed),
            context={**(run_summary or {}), "tta": use_tta, "epoch": epoch},
        )
        results[key] = result

        ci = f"  CI95={result.macro_f1_ci}" if result.macro_f1_ci else ""
        trk.log_message(
            f"[{'TTA   ' if use_tta else 'No TTA'}]  "
            f"macroF1={result.macro_f1:.4f}{ci}  "
            f"balAcc={result.balanced_accuracy:.4f}  Acc={result.accuracy:.1%}",
            level="plain",
        )
        trk.log_table(
            f"hardest_classes/{split_tag}",
            hardest_classes_report(result.per_class_f1),
            step=epoch,
        )
        aligned = rows if rows is not None and len(rows) == len(preds) else None
        breakdown = _session_breakdown(sessions, preds, targets, aligned, split_tag, trk)
        if breakdown is not None:
            session_results[key] = breakdown
        if artifacts is not None:
            artifacts.write_predictions(split_tag, preds, targets, rows=aligned)
            if want_logits:
                artifacts.write_logits(split_tag, logits, targets, rows=aligned)
            publish(artifacts, result, trk, step=epoch, prefix=split_tag)
            if breakdown is not None:
                artifacts.write_session(breakdown)
                trk.log_scalars(breakdown.scalars(split_tag), step=epoch)
                trk.log_table(f"session/{split_tag}", breakdown.per_session, step=epoch)

    logit_files: list[str] = [f"{report_split}_{k}" for k, _ in variants] if want_logits else []
    if want_logits and calib_ldr is not None:
        calib_rows = _rows_after_dedup(calib_ldr)
        for key, use_tta in variants:
            logits, targets = _predict(cfg, eval_model, calib_ldr, device, dist, use_tta)
            if artifacts is not None:
                aligned = (
                    calib_rows
                    if calib_rows is not None and len(calib_rows) == len(logits)
                    else None
                )
                artifacts.write_logits(f"calib_{key}", logits, targets, rows=aligned)
            logit_files.append(f"calib_{key}")

    if artifacts is not None:
        artifacts.write_manifest(
            {
                "run": dict(run_summary or {}),
                "logits": {"dtype": "float16", "splits": logit_files},
                "checkpoint": {
                    "path": best_ckpt,
                    "epoch": epoch,
                    "stage": ckpt.get("stage"),
                    "best_source": source,
                    "selection_f1": float(ckpt.get("val_f1", 0.0)),
                    "arch": ckpt.get("arch", "spectral_quadnet"),
                    "schema_version": ckpt.get("schema_version"),
                },
                "results": {
                    k: _result_payload(v, session_results.get(k)) for k, v in results.items()
                },
            }
        )
        if bool(getattr(cfg.evaluation, "save_artifacts", True)):
            written = render_run_figures(artifacts, results)
            trk.log_message(f"Figures ({len(written)}) → {artifacts.figures}", level="plain")
        trk.log_message(f"Results → {artifacts.results}", level="plain")

    return results


def _result_payload(
    result: ClassificationResult, breakdown: SessionReport | None
) -> dict[str, Any]:
    """One variant's manifest entry: the headline metrics, plus the breakdown if any."""
    payload = result.as_dict()
    if breakdown is not None:
        payload["session"] = breakdown.as_dict()
    return payload


def _session_breakdown(
    sessions: SessionMap | None,
    preds: Any,
    targets: Any,
    rows: Any,
    split_tag: str,
    trk: ExperimentTracker,
) -> SessionReport | None:
    """Score one variant by acquisition session, or say why it was not.

    Never raises: the breakdown is a measurement about the run, and a failure to
    compute it — a row order that does not reproduce the targets, say — is
    reported and skipped rather than allowed to discard the headline numbers
    the pass has already produced.
    """
    if sessions is None or rows is None:
        return None
    try:
        report = session_report(preds, targets, rows, sessions, split=split_tag)
    except ValueError as exc:
        trk.log_message(f"Session breakdown skipped for {split_tag}: {exc}", level="warn")
        return None
    for line in report.lines():
        trk.log_message(line, level="plain")
    return report


def _rows_after_dedup(loader: DataLoader[Any]) -> Any:
    """The patch row of each prediction :func:`_predict` returns, or ``None``.

    :func:`_predict` de-duplicates a DDP pass into dataset order, so whenever the
    raw gathered order can be established at all, the de-duplicated order is
    simply ``dataset.indices``.
    """
    if eval_row_order(loader) is None:
        return None
    return np.asarray(loader.dataset.indices, dtype=np.int64)  # type: ignore[attr-defined]


@torch.inference_mode()
def _predict(
    cfg: ExperimentConfig | Any,
    eval_model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    dist: DistContext,
    use_tta: bool,
) -> tuple[Any, Any]:
    """One pass over ``loader``, returning ``(logits, targets)`` as numpy arrays.

    ``inference_mode`` and on-device accumulation for the same reason
    ``engine/evaluate.py::_run_eval`` uses them: a ``.cpu()`` per batch is a
    queue drain per batch, and the TTA pass runs twelve forwards for each.
    Logits are fp32 here (the forward runs with autocast off) and gathered
    row-wise; under DDP the result is de-duplicated into dataset order, so each
    kernel appears once.
    """
    logit_parts, targets = [], []
    for batch in loader:
        x, y, mask, morph = unpack_batch(batch, device)
        side = side_inputs(mask, morph)
        logits = (
            tta_predict(eval_model, x, cfg.tta_spatial, cfg.tta_spectral, **side)
            if use_tta
            else eval_model(x, **side)
        )
        logit_parts.append(logits.float())
        targets.append(y)
        del x, mask, morph, logits
    joined = torch.cat(logit_parts)
    n_cls = joined.shape[1]
    # `gather_concat` joins 1-D tensors; the (B, C) block travels flattened.
    flat = gather_concat(dist, joined.reshape(-1)).cpu().numpy()
    lg: Any = flat.reshape(-1, n_cls)
    t: Any = gather_concat(dist, torch.cat(targets)).cpu().numpy()
    keep = dedup_index(loader) if dist.enabled else None
    if keep is not None:
        lg, t = lg[keep], t[keep]
    return lg, t
