"""Fixed S23/S27 recipe for the anchored additive residual head.

Train-row gradients, calibration macro-F1 checkpoint selection with epoch 0 (the
anchor itself) eligible, patience on calibration. The loop is identical to the
frozen S27 runner, so S28+ studies share one implementation instead of copies.
"""
from __future__ import annotations

import copy
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from spectralquadnet.experiments.rgb_probe import class_metrics
from spectralquadnet.models.multimodal_residual import FrozenResidualFusion


def macro_f1(y: npt.NDArray[Any], scores: torch.Tensor) -> float:
    return float(class_metrics(y, scores.argmax(1).numpy(), scores.shape[1])["f1"].mean())


def fit_residual_head(
    train: list[torch.Tensor],
    calib: list[torch.Tensor],
    y_train: npt.NDArray[Any],
    y_calib: npt.NDArray[Any],
    model: dict[str, Any],
    recipe: dict[str, Any],
    seed: int,
) -> tuple[FrozenResidualFusion, dict[str, Any], list[dict[str, Any]]]:
    """Fit one head; inputs are ``[hsi, rgb, anchor_probabilities]`` tensors."""
    yt = torch.from_numpy(np.asarray(y_train, dtype=np.int64))
    torch.manual_seed(seed)
    head = FrozenResidualFusion(**model)
    optimizer = torch.optim.AdamW(head.parameters(), lr=recipe["lr"], weight_decay=recipe["weight_decay"])
    best = anchor = macro_f1(y_calib, calib[2])
    traces = [{"epoch": 0, "calib_f1": best, "train_accuracy": float((train[2].argmax(1) == yt).float().mean())}]
    state, chosen, stale, epoch = copy.deepcopy(head.state_dict()), 0, 0, 0
    for epoch in range(1, recipe["epochs"] + 1):
        head.train()
        order = torch.randperm(len(yt))
        for start in range(0, len(yt), recipe["batch"]):
            batch = order[start:start + recipe["batch"]]
            z, residual = head(*(v[batch] for v in train))
            loss = torch.nn.functional.cross_entropy(z, yt[batch]) + recipe["residual_penalty"] * residual.square().mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite residual-head loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            cz, _ = head(*calib)
            tz, _ = head(*train)
        score = macro_f1(y_calib, cz)
        traces.append({"epoch": epoch, "calib_f1": score, "train_accuracy": float((tz.argmax(1) == yt).float().mean())})
        if score > best:
            best, state, chosen, stale = score, copy.deepcopy(head.state_dict()), epoch, 0
        else:
            stale += 1
        if stale >= recipe["patience"]:
            break
    head.load_state_dict(state)
    head.eval()
    selection = {"head_seed": seed, "selected_epoch": chosen, "epochs_completed": epoch,
                 "calib_f1": best, "anchor_calib_f1": anchor,
                 "parameters": sum(p.numel() for p in head.parameters())}
    return head, selection, traces
