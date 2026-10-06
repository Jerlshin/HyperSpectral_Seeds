#!/usr/bin/env python
"""S40: training FLOPs per sample (forward + backward) of SeedNet v5 at 215 vs 32 bands."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

from spectralquadnet.config.compose import load_experiment_config
from spectralquadnet.models.registry import build_model

PLAN = json.loads(Path("configs/research/s40_full215_v5.json").read_text())
V5 = [o for o in PLAN["v5_overrides"] if not o.startswith("data=")]
out: dict[str, dict[str, float]] = {}
for data, bands in (("refl215_f16_grouped", 215), ("ablation/u430k32_grouped", 32)):
    cfg = load_experiment_config("experiment/seednet_full256", overrides=[f"data={data}", *V5])
    torch.manual_seed(0)
    model = build_model(cfg, torch.linspace(0, 1, bands)).train()
    n = 4
    x, y = torch.rand(n, bands, 64, 64), torch.arange(n)
    counter = FlopCounterMode(display=False)
    with counter:
        o = model(x, labels=y, mask=torch.ones(n, 64, 64), morph=torch.randn(n, 8))
        (o["main"].sum() + o["aux_spatial"].sum()).backward()
    per = {k: sum(v.values()) / n / 1e9 for k, v in counter.get_flop_counts().items()}
    out[str(bands)] = {
        "train_gflop_per_sample": counter.get_total_flops() / n / 1e9,
        **{
            k.replace("SpectralSeedNet.", ""): round(v, 3)
            for k, v in per.items()
            if k.startswith("SpectralSeedNet.spatial.stem") and k.count(".") <= 3
        },
    }
out["ratio_215_over_32"] = {
    "train": out["215"]["train_gflop_per_sample"] / out["32"]["train_gflop_per_sample"]
}
json.dump(out, sys.stdout, indent=2)
