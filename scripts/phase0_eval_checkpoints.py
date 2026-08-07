#!/usr/bin/env python3
"""Phase 0 · actions 0-A and 0-B — evaluate all three stage checkpoints.

Reads ``IMPROVEMENT_PLAN.md`` §4.1:

* **0-A** — run ``final_evaluation`` over ``best_stage{1,2,3}.pth``, no-TTA and
  TTA, to answer **F-7** (*is Stage 3 actually worse on test?*).
* **0-B** — re-evaluate each checkpoint with ``tta_spatial=8 tta_spectral=0``
  to answer **F-6** (*do the 4 spectral views help or hurt?*).

Nothing here modifies model code. ``final_evaluation`` is called unmodified,
once per stage, with ``output_dir`` redirected to ``outputs/phase0/stage{n}/``
so the reference run's artifacts under ``outputs/output_v12_spa40/`` are never
overwritten.

Beyond the two required actions the script also records, for each checkpoint:

* validation-split predictions (no-TTA and 12-view TTA), which
  ``scripts/bootstrap_ci.py`` (0-C) needs to bootstrap the val→test gap;
* a **fp32** 12-view TTA pass. ``engine/tta.py`` wraps every view in
  ``autocast(device_type=...)`` while the no-TTA path in ``final_eval.py`` does
  not, so the shipped TTA/no-TTA comparison confounds ensembling with a
  precision change (§2.6.4 / T1-2). Measuring both isolates it.

Every prediction array is written to ``outputs/phase0/preds/`` as
``{split}_stage{n}_{variant}.npy`` and every metric to
``outputs/phase0/eval_results.json``.

Usage::

    python scripts/phase0_eval_checkpoints.py
    python scripts/phase0_eval_checkpoints.py --device cpu
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from omegaconf import OmegaConf, open_dict
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

import spectralquadnet.engine.tta as tta_mod
from spectralquadnet.config.compose import load_experiment_config
from spectralquadnet.data.loaders import build_loaders, build_splits
from spectralquadnet.data.mmap_store import DataStore
from spectralquadnet.engine.checkpoint import load_ckpt, stage_ckpt_path
from spectralquadnet.engine.stages.final_eval import final_evaluation
from spectralquadnet.engine.tta import tta_predict
from spectralquadnet.models.ema import ModelEMA
from spectralquadnet.models.spectral_quadnet import SpectralQuadNet
from spectralquadnet.tracking import build_tracker
from spectralquadnet.utils.device import resolve_device
from spectralquadnet.utils.seed import set_seed

#: TTA configurations evaluated per checkpoint, as
#: ``variant -> (n_spatial, n_spectral, force_fp32)``. ``None`` means no TTA at
#: all — a single plain forward, matching ``final_eval``'s "No TTA" arm.
_VARIANTS: dict[str, tuple[int, int, bool] | None] = {
    "noTTA": None,
    "tta12": (8, 4, False),  # the shipped configuration
    "tta8_spatial_only": (8, 0, False),  # 0-B / F-6
    "tta12_fp32": (8, 4, True),  # isolates the §2.6.4 precision confound
}


@contextlib.contextmanager
def _tta_in_fp32() -> Iterator[None]:
    """Neutralise the ``autocast`` context ``engine/tta.py`` opens per view.

    Wrapping ``tta_predict`` in ``autocast(enabled=False)`` from the outside
    does not work: the inner ``autocast(device_type=...)`` inside the view loop
    is a nested context with ``enabled=True`` and simply re-enables it. So the
    module-level symbol is swapped for a no-op factory for the duration of the
    call and restored afterwards. Nothing in ``src/`` is edited — this is a
    measurement-only patch that exists to separate the ensembling effect from
    the precision change (§2.6.4).

    ``tta_any`` is the module widened to ``Any``: rebinding a module attribute
    is by definition outside what the type checker models.
    """
    tta_any: Any = tta_mod
    original = tta_any.autocast
    tta_any.autocast = lambda *a, **k: contextlib.nullcontext()
    try:
        yield
    finally:
        tta_any.autocast = original


@torch.no_grad()
def _predict(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    spec: tuple[int, int, bool] | None,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Predict over ``loader``, optionally averaging TTA views.

    ``spec=None`` reproduces ``final_eval``'s no-TTA arm exactly: a bare
    ``model(x)`` with no autocast context, i.e. whatever the ambient precision
    is (fp32 here). Otherwise ``(n_spatial, n_spectral, force_fp32)`` selects a
    ``tta_predict`` configuration.
    """
    preds, targets = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        if spec is None:
            logits = model(x)
        else:
            n_spatial, n_spectral, fp32 = spec
            ctx: Any = _tta_in_fp32() if fp32 else contextlib.nullcontext()
            with ctx:
                logits = tta_predict(model, x, n_spatial, n_spectral)
        preds.append(logits.argmax(1).cpu())
        targets.append(y.cpu())
    return torch.cat(preds).numpy(), torch.cat(targets).numpy()


def _metrics(t: npt.NDArray[Any], p: npt.NDArray[Any]) -> dict[str, float]:
    """Macro-F1, weighted-F1 and accuracy for one prediction array."""
    return {
        "f1_macro": float(f1_score(t, p, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(t, p, average="weighted", zero_division=0)),
        "acc": float(accuracy_score(t, p)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=None, help="Override cfg.device (auto/mps/cuda/cpu).")
    ap.add_argument("--out", default="outputs/phase0", help="Directory for Phase-0 artifacts.")
    args = ap.parse_args()

    cfg = load_experiment_config()
    # RNG ordering per train.py: seed before the model is constructed. Every
    # weight is overwritten by load_ckpt below, but keeping the order identical
    # means this script and train.py build byte-identical modules.
    set_seed(cfg.seed)
    device = resolve_device(args.device or cfg.device)

    out_root = Path(args.out)
    preds_dir = out_root / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)

    tracker = build_tracker(cfg)
    tracker.banner(
        "PHASE 0 · 0-A / 0-B — three-checkpoint evaluation",
        [
            f"Device: {device}   Checkpoint dir: {cfg.output_dir}",
            f"Variants: {', '.join(_VARIANTS)}",
        ],
    )

    store = DataStore.from_config(cfg.data, device)
    all_labels, train_idx, val_idx, test_idx = build_splits(cfg)
    tracker.log_message(
        f"Split sizes — train {len(train_idx):,}  val {len(val_idx):,}  test {len(test_idx):,}",
        level="plain",
    )

    model = SpectralQuadNet.from_config(cfg, store.require_wavelengths()).to(device)
    ema = ModelEMA(model, decay=cfg.ema_decay)
    _, val_ldr, test_ldr = build_loaders(cfg, store, device, train_idx, val_idx, test_idx, 256)

    results: dict[str, Any] = {
        "device": str(device),
        "split_sizes": {
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx),
        },
        "tta_config": {"tta_spatial": int(cfg.tta_spatial), "tta_spectral": int(cfg.tta_spectral)},
        "stages": {},
    }

    for stage in (1, 2, 3):
        ckpt_path = stage_ckpt_path(cfg, stage)
        if not Path(ckpt_path).is_file():
            tracker.log_message(f"Stage {stage}: no checkpoint at {ckpt_path} — skipping", "warn")
            continue

        # ── 0-A: final_evaluation, unmodified, on this checkpoint ──────
        stage_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        with open_dict(stage_cfg):
            stage_cfg.output_dir = str(out_root / f"final_eval_stage{stage}")
        Path(stage_cfg.output_dir).mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        final_evaluation(stage_cfg, model, ema, test_ldr, device, ckpt_path, tracker)
        tracker.log_message(
            f"[0-A] stage {stage} final_evaluation took {time.time() - t0:.0f}s", level="plain"
        )

        # ── The extra variants, on the same loaded EMA shadow ──────────
        ckpt = load_ckpt(ckpt_path, model, ema, device)
        eval_model = ema.shadow
        eval_model.eval()

        stage_rec: dict[str, Any] = {
            "checkpoint": ckpt_path,
            "epoch": int(ckpt["epoch"]),
            "stage_label": str(ckpt["stage"]),
            "use_arcface": bool(ckpt.get("use_arcface", False)),
            "val_f1_recorded": float(ckpt.get("val_f1", 0.0)),
            "val_acc_recorded": float(ckpt.get("val_acc", 0.0)),
            "splits": {},
        }

        for split_name, loader in (("val", val_ldr), ("test", test_ldr)):
            split_rec: dict[str, Any] = {}
            for variant, spec in _VARIANTS.items():
                # The spatial-only and fp32 variants exist to answer questions
                # about the *test* number; running them on val doubles the cost
                # for nothing.
                if split_name == "val" and variant in ("tta8_spatial_only", "tta12_fp32"):
                    continue
                t0 = time.time()
                p, t = _predict(eval_model, loader, device, spec)
                m = _metrics(t, p)
                m["seconds"] = round(time.time() - t0, 1)
                split_rec[variant] = m
                np.save(preds_dir / f"{split_name}_stage{stage}_{variant}.npy", p)
                np.save(preds_dir / f"{split_name}_targets.npy", t)
                tracker.log_message(
                    f"  stage {stage} · {split_name:<4} · {variant:<18} "
                    f"F1(macro)={m['f1_macro']:.4f}  F1(wt)={m['f1_weighted']:.4f}  "
                    f"Acc={m['acc']:.1%}  ({m['seconds']}s)",
                    level="plain",
                )
            stage_rec["splits"][split_name] = split_rec

        results["stages"][str(stage)] = stage_rec

    (out_root / "eval_results.json").write_text(json.dumps(results, indent=2))
    tracker.log_message(f"\nWrote {out_root / 'eval_results.json'}", level="plain")
    tracker.close()


if __name__ == "__main__":
    main()
