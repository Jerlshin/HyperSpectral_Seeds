"""Fine-tuning loop for the multi-layer RGB branch with metric morphometrics (S37).

Same recipe semantics as ``rgb_finetune.finetune`` (augmentation, schedule, label smoothing,
calib-only epoch choice), extended in two places only:
* the per-layer projections, the morphometric map and the classifier are *head* parameters
  (``head_lr``, layer decay does not apply to them);
* each row's standardized metric morphometrics are passed with its crop. They are camera-locked
  (S31 §3), so they are not augmented: orientation views and shifts do not change a kernel's size.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from torch import nn

from spectralquadnet.experiments.rgb_finetune import (
    Recipe,
    augment,
    batches,
    lr_factor,
    macro_f1,
    to_unit,
)
from spectralquadnet.experiments.rgb_readout import VIEWS, view
from spectralquadnet.models.rgb_multilayer import MultiLayerRGBBranch

Array = npt.NDArray[Any]
HEAD_PREFIXES = ("project.", "morph.", "classifier.")


def morph_features(raw: Array, train: Array) -> Array:
    """log of the positive size terms (area, axes, equivalent diameter), then train-only z-scores."""
    x = np.asarray(raw, dtype=np.float64).copy()
    x[:, [0, 1, 2, 6]] = np.log(x[:, [0, 1, 2, 6]])
    mean, scale = x[train].mean(0), np.maximum(x[train].std(0), 1e-6)
    return np.asarray((x - mean) / scale, dtype=np.float32)


def parameter_groups(model: MultiLayerRGBBranch, lr: float, head_lr: float, decay: float,
                     weight_decay: float) -> list[dict[str, object]]:
    depth = len(model.backbone.blocks)  # type: ignore[arg-type]
    groups: dict[tuple[float, float], list[nn.Parameter]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(HEAD_PREFIXES):
            rate = head_lr
        elif name.startswith("backbone.blocks."):
            rate = lr * decay ** (depth - int(name.split(".")[2]))
        elif name.startswith("backbone.norm"):
            rate = lr
        else:
            rate = lr * decay ** (depth + 1)
        wd = 0.0 if param.ndim == 1 or name.endswith(("pos_embed", "cls_token", "mask_token")) else weight_decay
        groups.setdefault((rate, wd), []).append(param)
    return [{"params": p, "lr": r, "weight_decay": wd, "base_lr": r} for (r, wd), p in groups.items()]


@torch.no_grad()
def predict(model: MultiLayerRGBBranch, images: Array, morph: Array, rows: Array, device: torch.device,
            recipe: Recipe, views: tuple[str, ...] = VIEWS, batch: int = 64) -> tuple[Array, Array]:
    """Per-view logits (V x N x C) and identity-view embeddings (N x D), row order preserved."""
    model.eval()
    logits, embeddings = [], []
    dtype = torch.bfloat16 if recipe.precision == "bf16" else torch.float32
    for idx in batches(rows, batch, None):
        x = to_unit(torch.from_numpy(np.array(images[idx], copy=True))).to(device)
        m = torch.from_numpy(morph[idx]).to(device)
        per_view = []
        for name in views:
            with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
                z, e = model(view(x, name), m)
            per_view.append(z.float().cpu())
            if name == "id":
                embeddings.append(e.float().cpu())
        logits.append(torch.stack(per_view))
    return torch.cat(logits, dim=1).numpy(), torch.cat(embeddings).numpy()


def finetune(model: MultiLayerRGBBranch, images: Array, morph: Array, y: Array, train: Array, calib: Array,
             recipe: Recipe, seed: int, device: torch.device,
             log: Any = print) -> tuple[MultiLayerRGBBranch, dict[str, Any], list[dict[str, Any]]]:
    """Train on ``train``; keep the epoch with the best identity-view calib macro-F1."""
    if set(train) & set(calib):
        raise ValueError("Train and calib overlap")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(parameter_groups(model, recipe.lr, recipe.head_lr, recipe.layer_decay,
                                                   recipe.weight_decay), betas=(0.9, 0.999))
    steps_per_epoch = math.ceil(len(train) / recipe.batch)
    total, warmup = recipe.epochs * steps_per_epoch, int(recipe.warmup_epochs * steps_per_epoch)
    dtype = torch.bfloat16 if recipe.precision == "bf16" else torch.float32
    best: dict[str, Any] = {"calib_f1": -1.0}
    best_state: dict[str, torch.Tensor] = {}
    trace, step = [], 0
    for epoch in range(recipe.epochs):
        model.train()
        start, losses = time.perf_counter(), []
        for idx in batches(train, recipe.batch, rng):
            idx = np.sort(idx)
            x = augment(to_unit(torch.from_numpy(np.array(images[idx], copy=True))), recipe, generator).to(device)
            m = torch.from_numpy(morph[idx]).to(device)
            target = torch.from_numpy(y[idx]).long().to(device)
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * lr_factor(step, total, warmup)
            with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
                logits, _ = model(x, m)
            loss = F.cross_entropy(logits.float(), target, label_smoothing=recipe.label_smoothing)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch} step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            step += 1
        cal_logits, _ = predict(model, images, morph, calib, device, recipe, views=("id",))
        f1 = macro_f1(y[calib], cal_logits[0])
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "calib_f1": f1, "seconds": time.perf_counter() - start}
        trace.append(row)
        log(row)
        if f1 > best["calib_f1"]:
            best = {"epoch": epoch, "calib_f1": f1}
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best, trace
