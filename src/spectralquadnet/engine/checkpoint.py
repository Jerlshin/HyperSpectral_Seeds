"""Stage checkpoints, JSON meta sidecars and auto-resume detection.

The following conventions are load-bearing and must not drift:

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

# Aliased as `_json` (not `json`) so this module stays symbol-for-symbol
# comparable against the reference implementation `scripts/check_ast_no_op_move.py`
# checks it against.
import json as _json
import os
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from spectralquadnet.models.ema import ModelEMA
from spectralquadnet.utils.device import no_grad_is_safe_for_dropout

if TYPE_CHECKING:  # pragma: no cover - typing only
    from spectralquadnet.config.schema import ExperimentConfig


def stage_ckpt_path(cfg: ExperimentConfig | Any, s: int) -> str:
    """Path to stage ``s``'s ``.pth`` checkpoint under ``cfg.output_dir``."""
    return os.path.join(cfg.output_dir, f"best_stage{s}.pth")


def stage_meta_path(cfg: ExperimentConfig | Any, s: int) -> str:
    """Path to stage ``s``'s JSON meta sidecar under ``cfg.output_dir``."""
    return os.path.join(cfg.output_dir, f"stage{s}_meta.json")


def stage_exists(cfg: ExperimentConfig | Any, s: int) -> bool:
    """True only if both stage ``s``'s checkpoint and its meta sidecar are present."""
    return os.path.isfile(stage_ckpt_path(cfg, s)) and os.path.isfile(stage_meta_path(cfg, s))


def latest_completed_stage(cfg: ExperimentConfig | Any) -> int:
    """Highest stage (3, 2 or 1) that has a complete checkpoint, or 0 if none do.

    Used to decide where auto-resume should pick up training.
    """
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
    """Persist the full checkpoint bundle plus a JSON-only meta sidecar.

    The sidecar carries every JSON-serialisable bundle key except ``model``
    and ``ema`` (their tensors are only in the ``.pth``), so downstream code
    can inspect ``val_f1``/``val_acc``/metadata without deserialising torch
    state.

    Args:
        cfg: Composed experiment config, read for ``cfg.output_dir``.
        path: Destination path for the ``.pth`` bundle.
        epoch: Epoch number to record.
        stage: Stage label (e.g. ``"stage 2"``); its trailing token must be
            the stage's digit, since the sidecar path is derived from it.
        model: Model whose ``state_dict()`` and ``_use_arcface`` flag are saved.
        ema: EMA shadow whose ``state_dict()`` is saved under ``"ema"``.
        val_f1: Validation macro-F1 to record.
        val_acc: Validation accuracy to record.
        **metadata: Extra JSON-serialisable fields merged into the bundle
            (and, where serialisable, into the sidecar).
    """
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
    """Whether ``v`` survives a round trip through ``json.dumps``."""
    try:
        _json.dumps(v)
        return True
    except (TypeError, ValueError):
        return False


def load_stage_meta(cfg: ExperimentConfig | Any, s: int) -> dict[str, Any]:
    """Load stage ``s``'s JSON sidecar, re-integerising any string-keyed dict values.

    JSON forces object keys to strings, so any top-level dict value (e.g.
    ``class_f1``, ``cdws_weights``) has its keys coerced back to ``int``
    where possible, restoring the ``{int: float}`` shape callers expect.

    Returns:
        The sidecar contents, or ``{}`` if the file does not exist.
    """
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
    """Load a checkpoint bundle into ``model`` and its EMA shadow in place.

    Also restores the ``use_arcface`` head-selection flag on both, so a
    resumed model routes through the same classification head it was saved
    with.

    Returns:
        The full deserialised bundle (including ``val_f1``/``val_acc``/metadata).
    """
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
    """Recompute BatchNorm running statistics over ``loader`` (used after SWA averaging).

    Resets every BatchNorm layer's running mean/var and momentum to ``None``
    (cumulative-average mode), then runs one pass over ``loader`` in
    ``train()`` mode so the running stats reflect the SWA-averaged weights
    rather than any individual snapshot's stats.
    """
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.reset_running_stats()
            m.momentum = None
    # This is the only place the model runs in `train()` mode under `no_grad`,
    # and on Metal that combination routes attention through a fused inference
    # kernel with no dropout support (see utils/device.py). Keeping grad enabled
    # there selects the math path instead. It costs a transient autograd graph
    # per batch — discarded immediately, nothing calls backward() — and changes
    # no forward value, so the BatchNorm statistics estimated here are identical.
    grad_ctx = torch.no_grad() if no_grad_is_safe_for_dropout(device) else torch.enable_grad()
    with grad_ctx:
        for x, _ in loader:
            model(x.to(device, non_blocking=True))
    model.eval()


def _pick_best_checkpoint(cfg: ExperimentConfig | Any, *ckpt_paths: str) -> str:
    """Select the checkpoint with the highest recorded ``val_f1`` across stages.

    For each path, ``val_f1`` is read from the JSON sidecar first (cheap);
    if absent, falls back to ``val_acc``, then to loading the full ``.pth``
    bundle, then to 0.0 if even that fails.

    Args:
        cfg: Composed experiment config, used to locate each path's sidecar.
        *ckpt_paths: Candidate checkpoint paths to rank.

    Returns:
        The path with the highest score, or the last path given if none
        exist on disk.
    """
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
