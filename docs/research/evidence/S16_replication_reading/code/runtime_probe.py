"""S16 · relative training cost of the S17 arms, measured on CPU before the round is frozen (no data, no held-out).

The S17 runtime estimate scales measured S13/S15 wall clocks (Kaggle T4 × 2) by each arm's per-step cost relative to
v5 at k = 32. GPU and CPU differ in absolute speed, not much in the ratio between two configurations of the same
network: this measures that ratio — one training step (forward + backward, batch 32, 64 × 64 patches, a random cube
and mask) of each built model, median of 4 after 2 warm-up steps, 8 threads.

Writes: runtime_probe.json
"""
from __future__ import annotations

import time

import torch
from s16common import EVID, save  # noqa: F401  (puts src/ on the path)

from spectralquadnet.config.compose import load_experiment_config
from spectralquadnet.models.branches.spatial_cnn import tail_map_sides
from spectralquadnet.models.registry import build_model

V5 = ["model.spectral_descriptor=snv_morph", "model.cbam_min_hw=3"]
ARMS = {"X1 (shipped architecture), k32": (32, None), "v5, k32": (32, "[2,2,2,1]"), "v5, k64 (Z2)": (64, "[2,2,2,1]"),
        "v5, 4x4 end map (Z3), k32": (32, "[2,2,1,1]")}


def main() -> None:
    torch.set_num_threads(8)
    out: dict = {"method": __doc__.split("\n\n")[1].replace("\n", " "), "arms": {}}
    for name, (k, tail) in ARMS.items():
        ov = ["data=ablation/u430k32_grouped", f"data.num_bands={k}"]
        ov += [*V5, f"model.spatial_tail_strides={tail}"] if tail else []
        cfg = load_experiment_config(overrides=ov)
        torch.manual_seed(0)
        m = build_model(cfg, torch.linspace(0, 1, k), input_side=64).train()
        x, mask = torch.rand(32, k, 64, 64), torch.ones(32, 64, 64)
        mask[:, :8] = 0
        morph, y = torch.randn(32, 8), torch.randint(0, 90, (32,))

        def step(m=m, x=x, y=y, mask=mask, morph=morph) -> None:  # bind this arm's tensors
            o = m(x, labels=y, mask=mask, morph=morph)
            lg = next(v for v in o.values() if torch.is_tensor(v) and v.dim() == 2 and v.shape[1] == 90)
            lg.float().logsumexp(1).mean().backward()

        step(), step()
        ts = []
        for _ in range(4):
            t = time.perf_counter()
            step()
            ts.append(time.perf_counter() - t)
        sides = tail_map_sides(64, tuple(int(s) for s in (tail or "[2,2,2,2]").strip("[]").split(",")))
        out["arms"][name] = dict(bands=k, tail=tail or "[2,2,2,2]", tail_sides=list(sides),
                                 parameters=int(sum(p.numel() for p in m.parameters())), step_s=sorted(ts)[1])
        print(name, out["arms"][name], flush=True)
    base = out["arms"]["v5, k32"]["step_s"]
    for v in out["arms"].values():
        v["relative_to_v5_k32"] = v["step_s"] / base
    save(out, "runtime_probe.json")


if __name__ == "__main__":
    main()
