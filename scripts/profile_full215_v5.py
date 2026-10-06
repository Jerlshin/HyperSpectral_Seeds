#!/usr/bin/env python
"""S40 engineering profile: SeedNet v5 at a given band count. Not a learning curve.

The run is composed as ``run_complementary_v5.py`` composes an S40 cell: the
frozen v5 overrides from the plan, with only ``data=`` swapped, and the frozen
S21 rows. This script trains nothing. It scores no held-out row, and it touches the
held-out rows only to report how many there are. It measures:

* the band geometry and banner, the parameter count and per-module breakdown;
* the 3-D stem's realised schedule, and that every band reaches a stage-1 tap
  (a one-hot spectrum is pushed through the real convolution);
* the output shape of every top-level module and stem stage on real rows;
* the host cost of one augmented training sample and one evaluation sample
  (single process, pages pre-read, so this is CPU work and not disk);
* one training step at the per-rank batch (forward, backward, AdamW) on the
  chosen accelerator under fp16 autocast. The step time and the memory held
  between forward and backward are reported;
* the fp32 bytes autograd saves for one per-rank batch, counted on CPU.

    python scripts/profile_full215_v5.py --data refl215_f16_grouped --device mps \
        --out outputs/s40_full215_v5/profile/refl215_f16_mps.json
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from spectralquadnet.config.compose import load_experiment_config
from spectralquadnet.data.datasets import RiceSeedDataset
from spectralquadnet.engine.batch import side_inputs, unpack_batch
from spectralquadnet.engine.pipelines import build_run_context
from spectralquadnet.engine.pipelines.context import describe_band_geometry
from spectralquadnet.experiments.frozen_rows import load_frozen_rows
from spectralquadnet.models.branches.spatial_cnn import SpectralSpatialStem3D
from spectralquadnet.models.registry import parameter_breakdown
from spectralquadnet.tracking.base import NullTracker
from spectralquadnet.utils.distributed import init_distributed, shutdown

PLAN = Path("configs/research/s40_full215_v5.json")
CPU_RUNTIME = [
    "runtime=default",
    "tracking=none",
    "device=cpu",
    "runtime.multi_gpu=off",
    "runtime.num_workers=0",
    "runtime.eval_num_workers=0",
    "runtime.compile=off",
    "runtime.prewarm_cache=off",
    "runtime.progress=off",
]


def compose(plan: dict[str, Any], data: str) -> Any:
    overrides = [o for o in plan["v5_overrides"] if not o.startswith("data=")]
    overrides = [
        f"data={data}",
        *overrides,
        "data.split_fold=0",
        "seed=0",
        "output_dir=outputs/s40_full215_v5/profile/_unused",
        *CPU_RUNTIME,
    ]
    return load_experiment_config("experiment/seednet_full256", overrides=overrides)


def stem_report(model: torch.nn.Module) -> dict[str, Any]:
    stem = next(m for m in model.modules() if isinstance(m, SpectralSpatialStem3D))
    conv: torch.nn.Conv3d = copy.deepcopy(stem.stage1[0]).float().eval()  # type: ignore[assignment]
    torch.nn.init.constant_(conv.weight, 1.0)
    unread = []
    for band in range(stem.num_bands):
        x = torch.zeros(1, 1, stem.num_bands, 4, 4)
        x[0, 0, band] = 1.0
        with torch.no_grad():
            if float(conv(x).abs().max()) == 0.0:
                unread.append(band)
    return {
        "num_bands": stem.num_bands,
        "spectral_strides": list(stem.spectral_strides),
        "kernel_depths": list(stem.kernel_depths),
        "folded_depth": stem.folded_depth,
        "fold_in_channels": int(stem.fold[0].in_channels),  # type: ignore[arg-type]
        "bands_never_read": unread,
    }


def batch_of(dataset: RiceSeedDataset, positions: list[int]) -> Any:
    return torch.utils.data.default_collate([dataset[i] for i in positions])


def shapes(model: torch.nn.Module, batch: Any) -> dict[str, list[int]]:
    seen: dict[str, list[int]] = {}
    hooks = []
    named = [
        (n, m)
        for n, m in model.named_modules()
        if n and (n.count(".") == 0 or n.startswith("spatial.stem.") and n.count(".") == 2)
    ]
    for name, module in named:

        def hook(_m: Any, _i: Any, out: Any, name: str = name) -> None:
            t = out[0] if isinstance(out, (tuple, list)) else out
            if isinstance(t, dict):
                t = t.get("main")
            if isinstance(t, torch.Tensor):
                seen[name] = list(t.shape)

        hooks.append(module.register_forward_hook(hook))
    x, y, mask, morph = unpack_batch(batch, "cpu")
    model.eval()
    with torch.no_grad():
        out = model(x, **side_inputs(mask, morph))
    seen["<output eval>"] = list(out.shape)
    for h in hooks:
        h.remove()
    seen["<input x>"] = list(x.shape)
    return seen


def host_cost(dataset: RiceSeedDataset, n: int, rng: np.random.Generator) -> dict[str, float]:
    pos = rng.choice(len(dataset), size=n, replace=False).tolist()
    for i in pos:  # page the rows in first: this measures CPU work, not the disk
        dataset[i]
    t = []
    for i in pos:
        s = time.perf_counter()
        dataset[i]
        t.append(time.perf_counter() - s)
    return {
        "samples": n,
        "median_ms": 1e3 * statistics.median(t),
        "mean_ms": 1e3 * statistics.fmean(t),
    }


def saved_activation_bytes(model: torch.nn.Module, batch: Any) -> int:
    seen: dict[tuple[int, int], int] = {}

    def pack(t: torch.Tensor) -> torch.Tensor:
        try:
            key = (t.untyped_storage().data_ptr(), t.untyped_storage().nbytes())
            seen[key] = t.untyped_storage().nbytes()
        except (RuntimeError, NotImplementedError):
            pass
        return t

    x, y, mask, morph = unpack_batch(batch, "cpu")
    params = {p.untyped_storage().data_ptr() for p in model.parameters()}
    model.train()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        model(x, labels=y, **side_inputs(mask, morph))
    return sum(v for (ptr, _), v in seen.items() if ptr not in params)


def device_step(model: torch.nn.Module, batch: Any, device: str, steps: int) -> dict[str, Any]:
    dev = torch.device(device)
    model = copy.deepcopy(model).to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x, y, mask, morph = unpack_batch(batch, dev)
    sync = {"mps": torch.mps.synchronize, "cuda": torch.cuda.synchronize}.get(
        dev.type, lambda: None
    )
    mem = (
        torch.mps.current_allocated_memory
        if dev.type == "mps"
        else torch.cuda.memory_allocated
        if dev.type == "cuda"
        else lambda: 0
    )
    times, held = [], []
    for step in range(steps + 3):
        opt.zero_grad(set_to_none=True)
        sync()
        base = mem()
        start = time.perf_counter()
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type != "cpu"):
            out = model(x, labels=y, **side_inputs(mask, morph))
            loss = F.cross_entropy(out["main"].float(), y) + 0.42 * F.cross_entropy(
                out["aux_spatial"].float(), y
            )
        sync()
        held.append(mem() - base)
        loss.backward()  # type: ignore[no-untyped-call]
        opt.step()
        sync()
        if step >= 3:
            times.append(time.perf_counter() - start)
    return {
        "device": device,
        "autocast": "fp16" if dev.type != "cpu" else "off",
        "batch": int(x.shape[0]),
        "steps_timed": steps,
        "step_median_s": statistics.median(times),
        "step_min_s": min(times),
        "forward_held_bytes": int(statistics.median(held[3:])),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data", default="refl215_f16_grouped")
    p.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    p.add_argument("--batch", type=int, default=64, help="per-rank batch (global 128 / 2 ranks)")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--host-samples", type=int, default=128)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    plan = json.loads(PLAN.read_text())
    cfg = compose(plan, args.data)
    rows = load_frozen_rows(
        Path(plan["partition_plan"]), Path(cfg.data.labels_path), Path(cfg.data.groups_path), 0
    )
    torch.set_num_threads(4)
    dist = init_distributed(cfg.runtime, fallback=torch.device("cpu"))
    try:
        ctx = build_run_context(cfg, NullTracker(), dist, split_override=rows)
        rng = np.random.default_rng(0)
        train_ds = RiceSeedDataset(
            rows.train,
            aug_strength=str(cfg.single.aug_profile),
            store=ctx.store,
            data_cfg=cfg.data,
            device="cpu",
            morph=ctx.morph,
        )
        eval_ds = RiceSeedDataset(
            rows.train, store=ctx.store, data_cfg=cfg.data, device="cpu", morph=ctx.morph
        )
        positions = rng.choice(len(train_ds), size=args.batch, replace=False).tolist()
        batch = batch_of(train_ds, positions)
        report: dict[str, Any] = {
            "study": "S40",
            "kind": "engineering profile, no held-out row scored",
            "data": args.data,
            "band_geometry": ctx.band_geometry,
            "banner": describe_band_geometry(ctx.band_geometry),
            "patch_dtype_on_disk": str(ctx.store.require_patches().dtype),
            "rows": {k: int(len(getattr(rows, k))) for k in ("train", "calib", "val", "test")},
            "parameters": parameter_breakdown(ctx.model),
            "stem": stem_report(ctx.model),
            "shapes": shapes(ctx.model, batch_of(eval_ds, positions[:2])),
            "host_ms_per_sample": {
                "train_aug_" + str(cfg.single.aug_profile): host_cost(
                    train_ds, args.host_samples, rng
                ),
                "eval_no_aug": host_cost(eval_ds, args.host_samples, rng),
            },
            "cpu_fp32_saved_activation_bytes": saved_activation_bytes(ctx.model, batch),
            "device_step": device_step(ctx.model, batch, args.device, args.steps),
            "host": {
                "machine": platform.machine(),
                "system": platform.system(),
                "torch": torch.__version__,
                "python": platform.python_version(),
            },
        }
    finally:
        shutdown(dist)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "banner",
                    "parameters",
                    "stem",
                    "host_ms_per_sample",
                    "cpu_fp32_saved_activation_bytes",
                    "device_step",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
