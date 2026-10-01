"""What produced a number: the code revision, the environment and the regime (D18).

S09 §8 found that ``results/run.json`` recorded no git commit, so a result could
not be tied to the code that produced it; S10 F54 then found that the regime a
run *applied* (the auxiliary weight) differed from the one its config named. A
results file therefore records three things beside its metrics:

* :func:`code_revision` — ``git rev-parse HEAD`` and whether the working tree had
  uncommitted changes (``dirty``), read from the checkout this package is
  imported from. ``SPECTRALQUADNET_GIT_COMMIT`` overrides it for a deployment
  that ships the code without its ``.git`` directory.
* :func:`environment` — interpreter, torch, CUDA, the device names.
* :func:`training_regime` — the resolved values of every knob the S09/S10
  pre-registrations vary (mixup, margin, clip, epochs, patience, label smoothing,
  dropout, augmentation, the aux schedule *and the weights it applies*, the
  pathways, whether morphometrics reached the model), so an arm can be
  identified from its ``run.json`` alone.

Never raises: provenance is a record about the run, not a precondition for it.
"""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:  # pragma: no cover - typing only
    from spectralquadnet.config.schema import ExperimentConfig

#: Environment variable that supplies the commit when ``git`` cannot.
COMMIT_ENV = "SPECTRALQUADNET_GIT_COMMIT"


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def code_revision(start: Path | None = None) -> dict[str, Any]:
    """``{"commit", "dirty", "source"}`` for the checkout this package runs from.

    ``dirty`` lists nothing, only whether ``git status --porcelain`` (tracked
    *and* untracked files outside ``.gitignore``) was non-empty — a dirty run's
    number is not reproducible from its commit alone, and saying so is the point.
    """
    env = os.environ.get(COMMIT_ENV)
    if env:
        return {"commit": env, "dirty": None, "source": f"env:{COMMIT_ENV}"}
    here = (start or Path(__file__)).resolve()
    cwd = here if here.is_dir() else here.parent
    commit = _git(["rev-parse", "HEAD"], cwd)
    if commit is None:
        return {"commit": None, "dirty": None, "source": "unavailable"}
    status = _git(["status", "--porcelain"], cwd)
    return {
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
        "source": "git",
    }


def environment(device: torch.device | None = None, world_size: int = 1) -> dict[str, Any]:
    """Interpreter, framework and accelerator versions."""
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "world_size": int(world_size),
        "device": str(device) if device is not None else None,
    }
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(device)
    return info


def training_regime(cfg: ExperimentConfig | Any, morph_input: bool | None = None) -> dict[str, Any]:
    """The resolved training regime, as the run applied it.

    The aux weight is reported as what :func:`single_stage_aux_weight` returns
    at the first and last epoch and on average — the applied values, not the
    configured key (S10 F54).
    """
    from spectralquadnet.losses.auxiliary import single_stage_aux_weight

    single = cfg.single
    total = int(single.epochs) + max(int(getattr(single, "supcon_epochs", 0)), 0)
    applied = [single_stage_aux_weight(cfg, ep, total) for ep in range(1, total + 1)]
    model = cfg.model
    return {
        "pipeline": str(cfg.pipeline),
        "epochs": int(single.epochs),
        "patience": int(single.patience),
        "batch": int(single.batch),
        "max_lr": float(single.max_lr),
        "min_lr": float(single.min_lr),
        "warmup_ep": int(single.warmup_ep),
        "weight_decay": float(cfg.weight_decay),
        "ema_decay": float(cfg.ema_decay),
        "grad_clip": float(cfg.grad_clip),
        "clip_partition": str(getattr(cfg, "clip_partition", "legacy")),
        "mixup": float(single.mixup),
        "mixup_epochs": int(single.mixup_epochs),
        "arcface_m": float(single.arcface_m),
        "arcface_s": float(single.arcface_s),
        "margin_warmup": [int(single.margin_warmup_start), int(single.margin_warmup_end)],
        "label_smooth": [float(single.label_smooth_hi), float(single.label_smooth_lo)],
        "focal_gamma": float(single.focal_gamma),
        "dropout": float(single.dropout),
        "aug_profile": str(single.aug_profile),
        "aux_weight_schedule": str(getattr(single, "aux_weight_schedule", "legacy")),
        "aux_weight_applied": {
            "first": applied[0] if applied else None,
            "last": applied[-1] if applied else None,
            "mean": (sum(applied) / len(applied)) if applied else None,
        },
        "pathways": [str(p) for p in getattr(model, "pathways", ["spatial", "spectral"])],
        "morphometrics_input": morph_input,
        "amp_dtype": str(getattr(cfg.runtime, "amp_dtype", "")),
        "clean_fit_kernels": int(getattr(single, "clean_fit_kernels", 0)),
    }
