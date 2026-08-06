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
checkpoint directory, reproducing that number.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader

from spectralquadnet.engine.checkpoint import load_ckpt
from spectralquadnet.engine.tta import tta_predict
from spectralquadnet.models.ema import ModelEMA

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
) -> None:
    w = 66
    print(f"\n{'═' * w}\n  FINAL TEST EVALUATION\n{'═' * w}")
    ckpt = load_ckpt(best_ckpt, model, ema, device)
    eval_model = ema.shadow
    eval_model.eval()

    print(
        f"  ArcFace: {eval_model._use_arcface}  |  "
        f"Checkpoint: ep {ckpt['epoch']} | {ckpt['stage']} | "
        f"F1={ckpt.get('val_f1', 0):.3f}  Acc={ckpt.get('val_acc', 0):.1%}"
    )
    print(
        f"  TTA: {cfg.tta_spatial} spatial + {cfg.tta_spectral} spectral "
        f"= {cfg.tta_spatial + cfg.tta_spectral} total views"
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
        print(
            f"\n  [{tag}]  F1(macro)={f1_score(t, p, average='macro', zero_division=0):.4f}  "
            f"F1(wt)={f1_score(t, p, average='weighted', zero_division=0):.4f}  "
            f"Acc={accuracy_score(t, p):.1%}"
        )

    p_tta, t_tta = results["TTA   "]
    print("\nClassification Report (TTA):\n")
    print(classification_report(t_tta, p_tta, zero_division=0))

    out = cfg.output_dir
    np.save(f"{out}/test_preds_noTTA.npy", results["No TTA"][0])
    np.save(f"{out}/test_preds_TTA.npy", p_tta)
    np.save(f"{out}/test_targets.npy", t_tta)
    print(f"\nOutputs saved → {out}")
