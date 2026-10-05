"""Fixed fine-tuning recipe for the trainable RGB branch (S32).

Training rows only update weights; calib rows (held-aside training scans) only choose the
epoch and the temperature. Augmentation is label-free and acquisition-motivated: the four
orientation views of S31, a small translation inside the crop margin, and a mild exposure
jitter applied to foreground pixels only (session 8 changed aperture/exposure, F-series S20).
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F

from spectralquadnet.experiments.rgb_probe import class_metrics
from spectralquadnet.experiments.rgb_readout import VIEWS, view
from spectralquadnet.models.rgb_branch import RGBBranch, layer_decay_groups

Array = npt.NDArray[Any]


@dataclass(frozen=True)
class Recipe:
    epochs: int = 12
    batch: int = 32
    lr: float = 3e-5
    head_lr: float = 1e-3
    layer_decay: float = 0.8
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    label_smoothing: float = 0.1
    drop_path: float = 0.1
    keep_tokens: int = 128
    train_blocks: int = 0  # 0 = full fine-tuning; k > 0 = only the top k blocks (+ final norm, head)
    shift: int = 8
    exposure: float = 0.1
    blur_sigma: float = 0.0  # S33 acquisition augmentation: Gaussian sigma ~ U(0, blur_sigma) px
    blur_p: float = 0.5
    precision: str = "bf16"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def to_unit(images: torch.Tensor) -> torch.Tensor:
    """uint8 N x H x W x 3 -> float N x 3 x H x W in [0, 1]."""
    return images.permute(0, 3, 1, 2).float() / 255


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of 1 x C x H x W (zero padding, i.e. the masked black background)."""
    radius = max(1, int(round(3 * sigma)))
    t = torch.arange(-radius, radius + 1, dtype=x.dtype)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = (k / k.sum()).to(x.device)
    c = x.shape[1]
    x = F.conv2d(x, k.view(1, 1, 1, -1).expand(c, 1, 1, -1), padding=(0, radius), groups=c)
    return F.conv2d(x, k.view(1, 1, -1, 1).expand(c, 1, -1, 1), padding=(radius, 0), groups=c)


def augment(images: torch.Tensor, recipe: Recipe, generator: torch.Generator) -> torch.Tensor:
    """Per-sample orientation view, integer shift, foreground exposure jitter and (optionally)
    acquisition blur; the foreground support never changes (labels untouched)."""
    n = len(images)
    choice = torch.randint(len(VIEWS), (n,), generator=generator)
    shifts = torch.randint(-recipe.shift, recipe.shift + 1, (n, 2), generator=generator)
    gain = 1 + (torch.rand(n, generator=generator) * 2 - 1) * recipe.exposure
    contrast = 1 + (torch.rand(n, generator=generator) * 2 - 1) * recipe.exposure
    blur = torch.rand(n, generator=generator) * recipe.blur_sigma
    blur_on = torch.rand(n, generator=generator) < recipe.blur_p
    out = []
    for i in range(n):
        x = view(images[i : i + 1], VIEWS[int(choice[i])])
        x = torch.roll(x, shifts=(int(shifts[i, 0]), int(shifts[i, 1])), dims=(2, 3))
        fg = x.amax(1, keepdim=True) > 0
        mean = x[fg.expand_as(x)].mean() if fg.any() else x.new_tensor(0.0)
        x = torch.where(fg, ((x - mean) * float(contrast[i]) + mean) * float(gain[i]), x)
        if recipe.blur_sigma > 0 and bool(blur_on[i]) and float(blur[i]) > 0.05:
            x = gaussian_blur(x, float(blur[i]))
        # A clamp at a tiny positive floor keeps every foreground pixel foreground.
        out.append(torch.where(fg, x.clamp(1 / 255, 1), torch.zeros_like(x)))
    return torch.cat(out)


def batches(rows: Array, size: int, rng: np.random.Generator | None) -> Iterator[Array]:
    order = rng.permutation(rows) if rng is not None else rows
    for start in range(0, len(order), size):
        yield order[start : start + size]


def lr_factor(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def render(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Test-time acquisition rendering: blur N x 3 x H x W crops, keep the foreground support."""
    if sigma <= 0:
        return x
    fg = x.amax(1, keepdim=True) > 0
    blurred = torch.cat([gaussian_blur(x[i : i + 1], sigma) for i in range(len(x))])
    return torch.where(fg, blurred.clamp(1 / 255, 1), torch.zeros_like(x))


@torch.no_grad()
def predict(model: RGBBranch, images: Array, rows: Array, device: torch.device, recipe: Recipe,
            views: tuple[str, ...] = VIEWS, batch: int = 64, sigma: float = 0.0) -> tuple[Array, Array]:
    """Per-view logits (V x N x C, float32) and identity-view embeddings (N x D, float32).

    ``sigma`` > 0 first renders every crop into a softer acquisition regime (S33).
    """
    model.eval()
    logits, embeddings = [], []
    dtype = torch.bfloat16 if recipe.precision == "bf16" else torch.float32
    for idx in batches(rows, batch, None):
        x = render(to_unit(torch.from_numpy(np.array(images[idx], copy=True))), sigma).to(device)
        per_view = []
        for name in views:
            with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
                z, e = model(view(x, name))
            per_view.append(z.float().cpu())
            if name == "id":
                embeddings.append(e.float().cpu())
        logits.append(torch.stack(per_view))
    return torch.cat(logits, dim=1).numpy(), torch.cat(embeddings).numpy()


def macro_f1(y: Array, logits: Array) -> float:
    return float(class_metrics(y, logits.argmax(1), logits.shape[1])["f1"].mean())


def finetune(model: RGBBranch, images: Array, y: Array, train: Array, calib: Array, recipe: Recipe,
             seed: int, device: torch.device, log: Any = print) -> tuple[RGBBranch, dict[str, Any], list[dict[str, Any]]]:
    """Train on ``train``; keep the epoch with the best calib macro-F1 (identity view)."""
    if set(train) & set(calib):
        raise ValueError("Train and calib overlap")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(layer_decay_groups(model, recipe.lr, recipe.head_lr, recipe.layer_decay,
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
            target = torch.from_numpy(y[idx]).long().to(device)
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * lr_factor(step, total, warmup)
            with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
                logits, _ = model(x)
            loss = F.cross_entropy(logits.float(), target, label_smoothing=recipe.label_smoothing)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch} step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            step += 1
        cal_logits, _ = predict(model, images, calib, device, recipe, views=("id",))
        f1 = macro_f1(y[calib], cal_logits[0])
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "calib_f1": f1,
               "seconds": time.perf_counter() - start}
        trace.append(row)
        log(row)
        if f1 > best["calib_f1"]:
            best = {"epoch": epoch, "calib_f1": f1}
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best, trace
