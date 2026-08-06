"""Final test-set evaluation, with and without 12-view TTA.

Relocated from ``HSI_modality_training/hsi_training.py`` @ ``886560f``:

=====================================  ==============
Symbol                                 Baseline lines
=====================================  ==============
:func:`final_evaluation`               2624-2668
=====================================  ==============

Declared deviation, mechanical: ``CONFIG["tta_{spatial,spectral}"]`` →
``cfg.tta_*`` and ``CONFIG["output_dir"]`` → ``cfg.output_dir``.

This is the function that produced ``test_preds_noTTA.npy``,
``test_preds_TTA.npy`` and ``test_targets.npy`` in ``outputs/output_v12_spa40/``,
and the 0.8745 macro F1 recorded in ``stage3_meta.json``. Two details that make
that reproducible: the evaluated model is the **EMA shadow**, not the live model,
and the checkpoint it loads is whichever stage ``_pick_best_checkpoint`` ranks
highest by ``val_f1`` — not necessarily Stage 3.

Phase 4's gate (§3.5) is exactly this function, run against the existing
checkpoint directory, reproducing those artifacts.

Phase 4 also replaced every ``print`` here one-for-one with a ``tracker`` call
(§4.1) and added the §4.2 hardest-classes table, which is derived from the same
predictions the classification report is built from — no new statistics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader

from spectralquadnet.engine.checkpoint import load_ckpt
from spectralquadnet.engine.diagnostics import hardest_classes_report
from spectralquadnet.engine.tta import tta_predict
from spectralquadnet.models.ema import ModelEMA
from spectralquadnet.tracking.base import ExperimentTracker, NullTracker

if TYPE_CHECKING:  # pragma: no cover - typing only
    from spectralquadnet.config.schema import ExperimentConfig
    from spectralquadnet.models.spectral_quadnet import SpectralQuadNet


def final_evaluation(
    cfg: ExperimentConfig | Any,
    model: SpectralQuadNet,
    ema: ModelEMA,
    test_ldr: DataLoader[Any],
    device: torch.device,
    best_ckpt: str,
    tracker: ExperimentTracker | None = None,
) -> None:
    trk = tracker if tracker is not None else NullTracker()
    ckpt = load_ckpt(best_ckpt, model, ema, device)
    eval_model = ema.shadow
    eval_model.eval()

    trk.banner(
        "FINAL TEST EVALUATION",
        [
            f"ArcFace: {eval_model._use_arcface}  |  "
            f"Checkpoint: ep {ckpt['epoch']} | {ckpt['stage']} | "
            f"F1={ckpt.get('val_f1', 0):.3f}  Acc={ckpt.get('val_acc', 0):.1%}",
            f"TTA: {cfg.tta_spatial} spatial + {cfg.tta_spectral} spectral "
            f"= {cfg.tta_spatial + cfg.tta_spectral} total views",
        ],
    )

    results = {}
    for tag, use_tta in [("No TTA", False), ("TTA   ", True)]:
        preds, targets = [], []
        for x, y in test_ldr:
            x = x.to(device, non_blocking=True)
            logits = (
                tta_predict(eval_model, x, cfg.tta_spatial, cfg.tta_spectral)
                if use_tta
                else eval_model(x)
            )
            preds.append(logits.argmax(1).cpu())
            targets.append(y.cpu())
        p, t = torch.cat(preds).numpy(), torch.cat(targets).numpy()
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

    # §4.2 per-class failure analysis — the bottom-K of the same per-class F1
    # the report above already tabulates.
    per_class = f1_score(
        t_tta, p_tta, average=None, zero_division=0, labels=list(range(cfg.data.num_classes))
    )
    trk.log_table(
        "hardest_classes/test_tta",
        hardest_classes_report({i: float(v) for i, v in enumerate(per_class)}),
        step=int(ckpt["epoch"]),
    )

    out = cfg.output_dir
    np.save(f"{out}/test_preds_noTTA.npy", results["No TTA"][0])
    np.save(f"{out}/test_preds_TTA.npy", p_tta)
    np.save(f"{out}/test_targets.npy", t_tta)
    trk.log_message(f"Outputs saved → {out}", level="plain")
