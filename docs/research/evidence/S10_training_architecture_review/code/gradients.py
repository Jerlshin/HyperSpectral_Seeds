"""S10 — where the backbone gradient norm comes from (CPU, train rows only).

The sweep logged one number for the whole "backbone" clip group (S09 F36: median 8.9, ≈ 44
in the margin phase). This splits it by module and by loss term, on each run's selected
checkpoint, in train mode (dropout and BatchNorm batch statistics as in training), on
four unaugmented batches of 64 training kernels, under the two objectives the shipped
regime used after mixup:

  clean   CE(label smoothing 0.056) at margin 0      + 0.313 × aux CE   (epoch 111)
  margin  CE(label smoothing 0.040) at margin 0.30   + 0.250 × aux CE   (epochs 132–150)

The aux weights are the ones the loop *applied* (``losses/auxiliary.py::_aux_loss_weight`` reads
``stage1.aux_loss_weight_{init,final}`` = 0.65 → 0.25), not the configured and logged 0.2 (S10 B7).

It also records the input to the spatial pathway's signed square root (``SpatialCNNBranch._pn``):
its derivative 1/(2√|h|) is unbounded at 0, so the share of pooled activations near zero says how
much the pooling amplifies gradient.

Writes grad_modules.csv (per run × objective × module) and grad_pn.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from checkpoints import MORPH_RAW, batch, build
from s10common import run_config, run_dirs, save, split_rows

from spectralquadnet.data.morphometrics import standardise_morphometrics
from spectralquadnet.models.stats_ops import foreground_mask

N_BATCH, BS = 4, 64
OBJECTIVES = {"clean": (0.0, 0.056, 0.3133), "margin": (0.30, 0.040, 0.25)}


def module_of(name: str) -> str:
    p = name.split(".")
    if p[0] == "spatial" and p[1] == "stem":
        return f"spatial.stem.{p[2]}"
    if p[0] == "spatial" and p[1] == "stages":
        return f"spatial.tail.{p[2]}"
    if p[0] in ("spatial", "spectral"):
        return f"{p[0]}.{p[1]}"
    return p[0]


def main() -> None:
    rows, pn_rows = [], []
    for arm, fold, seed, d in run_dirs():
        cfg = run_config(d)
        tr, _ = split_rows(arm, fold)
        morph, _ = standardise_morphometrics(MORPH_RAW, tr)
        ck = torch.load(d / "best_stage1.pth", map_location="cpu", weights_only=False)
        model = build(cfg, seed)
        model.load_state_dict(ck["model" if ck["best_source"] == "live" else "ema"])
        model.train()
        rng = np.random.default_rng(seed)
        picks = [rng.choice(tr, BS, replace=False) for _ in range(N_BATCH)]
        for obj, (m, ls, aux_w) in OBJECTIVES.items():
            for term in ("total", "main only", "aux only"):
                acc: dict[str, float] = {}
                for rows_b in picks:
                    torch.manual_seed(0)
                    x, mk, mo, y = batch(rows_b, morph)
                    model.zero_grad(set_to_none=True)
                    with torch.enable_grad():
                        out = model(x, labels=y, arc_m=m, mask=mk, morph=mo)
                        main = F.cross_entropy(out["main"], y, label_smoothing=ls)
                        aux = F.cross_entropy(out["aux_spatial"], y, label_smoothing=ls)
                        loss = {"total": main + aux_w * aux, "main only": main, "aux only": aux_w * aux}[term]
                        loss.backward()
                    sq: dict[str, float] = {}
                    for name, p in model.named_parameters():
                        if p.grad is not None:
                            k = module_of(name)
                            sq[k] = sq.get(k, 0.0) + float(p.grad.pow(2).sum())
                    for k, v in sq.items():
                        acc[k] = acc.get(k, 0.0) + v ** 0.5 / N_BATCH
                    # the shipped clip groups: head = arcface_head; fusion = embed_net (`fuse` is NOT in it); backbone = rest
                    grp = {"head": 0.0, "fusion": 0.0, "backbone": 0.0}
                    for k, v in sq.items():
                        grp["head" if k == "arcface_head" else "fusion" if k == "embed_net" else "backbone"] += v
                    for g, v in grp.items():
                        acc[f"[clip group] {g}"] = acc.get(f"[clip group] {g}", 0.0) + v ** 0.5 / N_BATCH
                for k, v in acc.items():
                    rows.append(dict(arm=arm, fold=fold, seed=seed, objective=obj, term=term, module=k, grad_norm=v))
        # signed-sqrt input at the final pooled map (1×1 at 64×64 input → mean ≡ max)
        model.eval()
        with torch.no_grad():
            x, mk, mo, y = batch(picks[0], morph)
            msk = foreground_mask(x, mk)
            h = model.spatial.stages(model.spatial.stem(model.se(x, msk), msk))
            a = h.abs().flatten()
            pn_rows.append(dict(arm=arm, fold=fold, seed=seed, final_map_hw=f"{h.shape[-2]}x{h.shape[-1]}",
                                frac_abs_lt_1e_2=float((a < 1e-2).float().mean()),
                                frac_abs_lt_1e_3=float((a < 1e-3).float().mean()),
                                median_abs=float(a.median()),
                                pn_derivative_median=float((0.5 / a.clamp_min(1e-8).sqrt()).median()),
                                pn_derivative_p99=float((0.5 / a.clamp_min(1e-8).sqrt()).quantile(0.99))))
        print(f"{arm} f{fold} s{seed}", flush=True)
    save(pd.DataFrame(rows), "grad_modules.csv")
    save(pd.DataFrame(pn_rows), "grad_pn.csv")


if __name__ == "__main__":
    main()
