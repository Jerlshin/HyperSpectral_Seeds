"""S10 — SpectralSeedNet's realised structure at the sweep's input (32 bands, 64 × 64).

Instantiates the model from the sweep's own config (no checkpoint needed) and records, for every
module on the spatial path, its parameter count, the feature-map size it outputs, and how many of
its parameters can ever receive a gradient. A 3×3, stride-2, padding-1 convolution applied to a
2×2 map produces a 1×1 output whose receptive field covers only kernel rows/cols 1–2: the other
5 of 9 taps multiply zero padding, every step, for every sample.

Writes structure.csv and structure.json (stem strides, kernel depths, folded depth, descriptor width).
"""
from __future__ import annotations

import pandas as pd
import torch
from checkpoints import WL
from s10common import run_config, run_dirs, save

from spectralquadnet.models.spectral_seed_net import SpectralSeedNet


def main() -> None:
    _, _, _, d = run_dirs()[0]
    cfg = run_config(d)
    torch.manual_seed(0)
    m = SpectralSeedNet.from_config(cfg, WL).eval()
    sizes: dict[str, tuple[int, ...]] = {}
    hooks = []
    named = [("spatial.stem", m.spatial.stem)] + [(f"spatial.stages.{i}", s) for i, s in enumerate(m.spatial.stages)]
    for name, mod in named:
        hooks.append(mod.register_forward_hook(lambda _m, _i, o, n=name: sizes.__setitem__(n, tuple(o.shape))))
    with torch.no_grad():
        m(torch.rand(2, cfg.data.num_bands, 64, 64), mask=torch.ones(2, 64, 64))
    for h in hooks:
        h.remove()

    rows = []
    for name, mod in named:
        n = sum(p.numel() for p in mod.parameters())
        shape = sizes[name]
        dead = 0
        if type(mod).__name__ == "ResBlock2D" and shape[-1] == 1:
            w = mod.c2.weight
            dead = w.shape[0] * w.shape[1] * 5  # 5 of 9 taps see only padding
        rows.append(dict(module=name, kind=type(mod).__name__, params=n, out_channels=shape[1],
                         out_hw=f"{shape[-2]}x{shape[-1]}", dead_params=dead))
    rows.append(dict(module="spatial.proj", kind="Linear+BN+GELU", params=sum(p.numel() for p in m.spatial.proj.parameters()),
                     out_channels=256, out_hw="vector", dead_params=0))
    for name in ("spectral", "fuse", "embed_net", "arcface_head", "aux_head_spatial", "se"):
        rows.append(dict(module=name, kind=type(getattr(m, name)).__name__,
                         params=sum(p.numel() for p in getattr(m, name).parameters()), out_channels=None,
                         out_hw="vector", dead_params=0))
    df = pd.DataFrame(rows)
    save(df, "structure.csv")
    total = sum(p.numel() for p in m.parameters())
    save({"num_bands": int(cfg.data.num_bands), "total_params": total,
          "dead_params": int(df.dead_params.sum()), "dead_share": float(df.dead_params.sum() / total),
          "stem_spectral_strides": list(m.spatial.stem.spectral_strides),
          "stem_kernel_depths": list(m.spatial.stem.kernel_depths),
          "stem_folded_depth": int(m.spatial.stem.folded_depth),
          "spectral_descriptor_width": int(m.spectral.in_dim),
          "eca_kernel": int(m.se.conv.kernel_size[0])}, "structure.json")


if __name__ == "__main__":
    main()
