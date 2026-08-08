"""Final test-set evaluation, with and without 12-view TTA.

Writes ``test_preds_noTTA.npy``, ``test_preds_TTA.npy`` and
``test_targets.npy`` under ``cfg.output_dir``. Two details that matter for
reproducing a reported number: the checkpoint loaded is whichever stage
``_pick_best_checkpoint`` ranks highest by ``val_f1`` — not necessarily Stage 3
— and the weights evaluated are the ones that **won that ``val_f1``**.

Every stage saves on ``max(F1_live, F1_ema)`` and writes the max to the sidecar
as ``val_f1``, but this function used to evaluate the EMA shadow
unconditionally. When the live model won the max, the shipped test number came
from a model that had never scored it, and the artifacts did not record which
had won, making the mismatch unfalsifiable after the fact (IMPROVEMENT_PLAN
§2.1.4, T1-8). The stages now record ``best_source`` in the bundle and this
function honours it. Bundles written before Tier 1 carry no such key and fall
back to ``"ema"``, which is exactly the behaviour they were produced under —
so the archived ``output_v12_spa40`` numbers still reproduce.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader

from spectralquadnet.engine.batch import side_inputs, unpack_batch
from spectralquadnet.engine.checkpoint import load_ckpt
from spectralquadnet.engine.diagnostics import hardest_classes_report
from spectralquadnet.engine.tta import tta_predict
from spectralquadnet.models.ema import ModelEMA
from spectralquadnet.tracking.base import ExperimentTracker, NullTracker
from spectralquadnet.utils.distributed import DistContext, gather_concat

if TYPE_CHECKING:  # pragma: no cover - typing only
    from spectralquadnet.config.schema import ExperimentConfig
    from spectralquadnet.models.spectral_quadnet import SpectralQuadNet

#: What a bundle without a recognised ``best_source`` is assumed to have been
#: selected on — the pre-Tier-1 behaviour, so archived runs keep reproducing.
DEFAULT_BEST_SOURCE = "ema"

#: Every ``best_source`` a stage writes. ``"live"`` names the bundle's ``model``
#: slot; ``"ema"`` and Stage 3's ``"swa"`` both name its ``ema`` slot, which is
#: where Stage 3 puts whichever of its two averages scored higher.
KNOWN_BEST_SOURCES = frozenset({"live", "ema", "swa"})


def final_evaluation(
    cfg: ExperimentConfig | Any,
    model: SpectralQuadNet,
    ema: ModelEMA,
    test_ldr: DataLoader[Any],
    device: torch.device,
    best_ckpt: str,
    tracker: ExperimentTracker | None = None,
    dist: DistContext | None = None,
) -> None:
    """Load the best checkpoint's selected weights and report test-set metrics with/without TTA.

    Args:
        cfg: Composed experiment config.
        model: Model instance to load the checkpoint's live weights into.
            Evaluated when the bundle's ``best_source`` is ``"live"``.
        ema: EMA wrapper; its shadow is loaded from the checkpoint and
            evaluated when ``best_source`` is ``"ema"`` or ``"swa"`` (the
            default for bundles that predate the key).
        test_ldr: Test-set loader.
        device: Device to evaluate on.
        best_ckpt: Path to the checkpoint to load and evaluate.
        tracker: Experiment tracker for reporting metrics/tables; ``None``
            logs nowhere.
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

    trk.banner(
        "FINAL TEST EVALUATION",
        [
            f"Schema v{ckpt.get('schema_version', 1)}  |  "
            f"Checkpoint: ep {ckpt['epoch']} | {ckpt['stage']} | "
            f"F1={ckpt.get('val_f1', 0):.3f}  Acc={ckpt.get('val_acc', 0):.1%}",
            f"Selected weights: {source}"
            + (
                "" if recorded in KNOWN_BEST_SOURCES else "  (assumed — not recorded in the bundle)"
            ),
            f"TTA: {cfg.tta_spatial} spatial + {cfg.tta_spectral} spectral "
            f"= {cfg.tta_spatial + cfg.tta_spectral} total views",
        ],
    )

    results = {}
    for tag, use_tta in [("No TTA", False), ("TTA   ", True)]:
        preds, targets = [], []
        # `inference_mode` and on-device accumulation, for the same reason as
        # `engine/evaluate.py::_run_eval`: a `.cpu()` per batch is a queue drain
        # per batch, and the TTA pass runs twelve forwards for each of them.
        with torch.inference_mode():
            for batch in test_ldr:
                x, y, mask, morph = unpack_batch(batch, device)
                side = side_inputs(mask, morph)
                logits = (
                    tta_predict(eval_model, x, cfg.tta_spatial, cfg.tta_spectral, **side)
                    if use_tta
                    else eval_model(x, **side)
                )
                preds.append(logits.argmax(1))
                targets.append(y)
        p = gather_concat(dist, torch.cat(preds)).cpu().numpy()
        t = gather_concat(dist, torch.cat(targets)).cpu().numpy()
        results[tag] = (p, t)
        f1_macro = f1_score(t, p, average="macro", zero_division=0)
        f1_weighted = f1_score(t, p, average="weighted", zero_division=0)
        acc = accuracy_score(t, p)
        trk.log_message(
            f"[{tag}]  F1(macro)={f1_macro:.4f}  F1(wt)={f1_weighted:.4f}  Acc={acc:.1%}",
            level="plain",
        )
        prefix = "test_tta" if use_tta else "test"
        trk.log_scalars(
            {
                f"{prefix}/f1_macro": float(f1_macro),
                f"{prefix}/f1_weighted": float(f1_weighted),
                f"{prefix}/acc": float(acc),
            },
            step=int(ckpt["epoch"]),
        )

    p_tta, t_tta = results["TTA   "]
    trk.log_message("Classification Report (TTA):", level="plain")
    trk.log_message(classification_report(t_tta, p_tta, zero_division=0), level="plain")

    # Per-class failure analysis — the bottom-K of the same per-class F1
    # the report above already tabulates.
    per_class = f1_score(
        t_tta, p_tta, average=None, zero_division=0, labels=list(range(cfg.data.num_classes))
    )
    trk.log_table(
        "hardest_classes/test_tta",
        hardest_classes_report({i: float(v) for i, v in enumerate(per_class)}),
        step=int(ckpt["epoch"]),
    )

    if not dist.is_main:
        return

    out = cfg.output_dir
    np.save(f"{out}/test_preds_noTTA.npy", results["No TTA"][0])
    np.save(f"{out}/test_preds_TTA.npy", p_tta)
    np.save(f"{out}/test_targets.npy", t_tta)
    trk.log_message(f"Outputs saved → {out}", level="plain")
