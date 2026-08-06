"""Stage checkpoints, JSON meta sidecars and auto-resume detection.

Relocated from ``HSI_modality_training/hsi_training.py`` @ ``886560f``:

=====================================  ==============
Symbol                                 Baseline lines
=====================================  ==============
:func:`stage_ckpt_path`                2070-2071
:func:`stage_meta_path`                2074-2075
:func:`stage_exists`                   2078-2079
:func:`latest_completed_stage`         2082-2086
:func:`save_ckpt`                      2089-2106
:func:`_is_json_serialisable`          2109-2113
:func:`load_stage_meta`                2116-2130
:func:`load_ckpt`                      2133-2139
:func:`update_bn_stats`                2142-2150
:func:`_pick_best_checkpoint`          2675-2694
=====================================  ==============

Declared deviation, mechanical and confined to the five functions that touched
it: ``CONFIG["output_dir"]`` → ``cfg.output_dir``, which makes ``cfg`` their
leading parameter. Only the *value* of ``output_dir`` moved (from a hardcoded
absolute path to a config field, REFACTOR_PLAN.md §4.3) — every filename
template is untouched.

REFACTOR_PLAN.md §3.5 pins the following, all preserved exactly:

* **Filenames** — ``best_stage{1,2,3}.pth`` and ``stage{1,2,3}_meta.json``.
* **Bundle schema** — ``{"epoch", "stage", "model", "ema", "val_f1", "val_acc",
  "use_arcface", **metadata}``, with the sidecar dropping ``model``/``ema`` and
  keeping only what :func:`_is_json_serialisable` accepts.
* **Resume order** — :func:`latest_completed_stage` probes 3 → 2 → 1 and
  requires *both* the ``.pth`` and its ``.json`` to call a stage complete, so a
  crash between the two writes resumes rather than skipping.
* **Selection rule** — :func:`_pick_best_checkpoint` ranks by ``val_f1`` from the
  sidecar, falling back to the bundle, then to 0.0.

:func:`load_stage_meta` re-integerises dict keys because JSON stringifies them:
``class_f1``/``cdws_weights`` are ``{int: float}`` in memory and must come back
that way for :func:`~spectralquadnet.losses.cdws.build_cdws_weights` and the
samplers to index them.
"""

from __future__ import annotations

# The baseline imports `json as _json`; the alias is kept so every function body
# below is *provably* byte-identical to it under `check_ast_no_op_move.py`
# rather than differing by a cosmetic module name.
import json as _json
import os
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from spectralquadnet.models.ema import ModelEMA

if TYPE_CHECKING:  # pragma: no cover - typing only
    from spectralquadnet.config.schema import ExperimentConfig


def stage_ckpt_path(cfg: ExperimentConfig | Any, s: int) -> str:
    return os.path.join(cfg.output_dir, f"best_stage{s}.pth")


def stage_meta_path(cfg: ExperimentConfig | Any, s: int) -> str:
    return os.path.join(cfg.output_dir, f"stage{s}_meta.json")


def stage_exists(cfg: ExperimentConfig | Any, s: int) -> bool:
    return os.path.isfile(stage_ckpt_path(cfg, s)) and os.path.isfile(stage_meta_path(cfg, s))


def latest_completed_stage(cfg: ExperimentConfig | Any) -> int:
    for s in (3, 2, 1):
        if stage_exists(cfg, s):
            return s
    return 0


def save_ckpt(
    cfg: ExperimentConfig | Any,
    path: str,
    epoch: int,
    stage: str,
    model: nn.Module,
    ema: ModelEMA,
    val_f1: float,
    val_acc: float,
    **metadata: Any,
) -> None:
    bundle = {
        "epoch": epoch,
        "stage": stage,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "val_f1": val_f1,
        "val_acc": val_acc,
        "use_arcface": model._use_arcface,
        **metadata,
    }
    torch.save(bundle, path)
    sidecar = {
        k: v for k, v in bundle.items() if k not in ("model", "ema") and _is_json_serialisable(v)
    }
    sn = int(stage.split()[-1]) if stage.split()[-1].isdigit() else 0
    with open(stage_meta_path(cfg, sn), "w") as f:
        _json.dump(sidecar, f, indent=2)


def _is_json_serialisable(v: Any) -> bool:
    try:
        _json.dumps(v)
        return True
    except (TypeError, ValueError):
        return False


def load_stage_meta(cfg: ExperimentConfig | Any, s: int) -> dict[str, Any]:
    p = stage_meta_path(cfg, s)
    if not os.path.isfile(p):
        return {}
    with open(p) as f:
        raw = _json.load(f)
    out = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            try:
                out[k] = {int(kk): vv for kk, vv in v.items()}
                continue
            except (ValueError, TypeError):
                pass
        out[k] = v
    return out


def load_ckpt(path: str, model: nn.Module, ema: ModelEMA, device: torch.device) -> dict[str, Any]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    ema.load_state_dict(ckpt["ema"])
    flag = ckpt.get("use_arcface", False)
    # `use_arcface` is `SpectralQuadNet`'s, not `nn.Module`'s; the annotation is
    # kept broad so `load_ckpt` works with the EMA shadow and any wrapper too.
    model.use_arcface(flag)  # type: ignore[operator]
    ema.shadow.use_arcface(flag)
    return ckpt  # type: ignore[no-any-return]


def update_bn_stats(loader: DataLoader[Any], model: nn.Module, device: torch.device) -> None:
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.reset_running_stats()
            m.momentum = None
    with torch.no_grad():
        for x, _ in loader:
            model(x.to(device, non_blocking=True))
    model.eval()


def _pick_best_checkpoint(cfg: ExperimentConfig | Any, *ckpt_paths: str) -> str:
    """Select checkpoint with highest val_f1 across all stages."""
    best_val, best_path = -1.0, ckpt_paths[-1]
    for p in ckpt_paths:
        if not os.path.isfile(p):
            continue
        try:
            sn = int(os.path.basename(p).replace("best_stage", "").replace(".pth", ""))
            meta = load_stage_meta(cfg, sn)
            v = meta.get("val_f1", meta.get("val_acc", None))
        except (ValueError, KeyError):
            v = None
        if v is None:
            try:
                v = torch.load(p, map_location="cpu", weights_only=False).get("val_f1", 0.0)
            except Exception:
                v = 0.0
        if v > best_val:
            best_val, best_path = v, p
    return best_path
