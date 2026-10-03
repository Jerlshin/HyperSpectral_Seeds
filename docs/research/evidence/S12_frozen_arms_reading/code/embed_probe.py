"""S12 · does the training-rows session probe rank the *networks* the way held-out attraction does?

For a set of grouped fold-0 checkpoints spanning the attraction range (X2 spectral_only / spatial_only /
no_morph, S08 full, X1 full, X4), this script rebuilds each model exactly as its pipeline did (the cell's
frozen overrides, its split, its train-standardised morphometrics), loads the selected weights (live or EMA,
as recorded), and extracts — on the fold's **training rows only**, eval mode, no augmentation — three
representations: the normalised embedding the cosine head reads, the spatial pathway's output and the
spectral pathway's output. Each is scored with ``session_probe.probe`` (class-disjoint session κ).

The held-out attraction each run produced is read from ``cells.csv`` / S09 ``runs.csv``; no held-out row is
loaded here. CPU, ≈ 1 min per checkpoint.

Writes: embed_probe.csv · embed_probe_validation.json
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from s12common import EVID, REPO, S08, S11, save
from session_probe import probe

from spectralquadnet.config.compose import load_experiment_config
from spectralquadnet.data.loaders import build_split_bundle, standardised_morphometrics
from spectralquadnet.data.mmap_store import DataStore
from spectralquadnet.engine.batch import side_inputs, unpack_batch
from spectralquadnet.engine.checkpoint import load_ckpt
from spectralquadnet.engine.clean_fit import CleanFitProbe
from spectralquadnet.models.ema import ModelEMA
from spectralquadnet.models.registry import build_model
from spectralquadnet.utils.seed import set_seed

CKPTS = [
    ("X2", "spectral_only", 0), ("X2", "spectral_only", 1), ("X2", "spatial_only", 0), ("X2", "spatial_only", 1),
    ("X2", "no_morph", 0), ("X2", "no_morph", 1), ("S08", "grouped", 0), ("S08", "grouped", 1),
    ("X1", "grouped", 0), ("X1", "grouped", 1), ("X4", "grouped", 0),
]


def run_dir(arm, variant, seed):
    return (S08 if arm == "S08" else S11 / arm) / f"{variant}__f0_s{seed}"


def extract(arm: str, variant: str, seed: int) -> dict[str, np.ndarray]:
    d = run_dir(arm, variant, seed)
    extra = json.load(open(d / "frozen_cell.json"))["arm_overrides"] if arm != "S08" else []
    cfg = load_experiment_config(overrides=["data=ablation/u430k32_grouped", "data.split_fold=0", f"seed={seed}", *extra])
    device = torch.device("cpu")
    set_seed(seed)
    store = DataStore.from_config(cfg.data, device)
    splits = build_split_bundle(cfg)
    model = build_model(cfg, store.require_wavelengths())
    ema = ModelEMA(model, decay=0.999)
    load_ckpt(str(d / "best_stage1.pth"), model, ema, device)
    meta = json.load(open(d / "stage1_meta.json"))
    net = ema.shadow if meta.get("best_source", "live") == "ema" else model
    net.eval()
    morph = standardised_morphometrics(store, splits.train)
    pr = CleanFitProbe.build(store=store, data_cfg=cfg.data, device=device, train_rows=splits.train,
                             labels=splits.labels, morph=morph, n=len(splits.train), seed=0)
    feats: dict[str, list] = {"embedding": [], "spatial": [], "spectral": []}
    hooks = [net.spatial.register_forward_hook(lambda m, i, o: feats["spatial"].append(o.detach())),
             net.spectral.register_forward_hook(lambda m, i, o: feats["spectral"].append(o.detach()))]
    ys = []
    with torch.no_grad():
        for batch in pr.loader:
            x, y, mask, mo = unpack_batch(batch, device)
            logits, emb = net(x, return_embed=True, **side_inputs(mask, mo))
            feats["embedding"].append(emb)
            ys.append((logits.argmax(1) == y).float())
    for h in hooks:
        h.remove()
    out = {k: torch.cat(v).numpy() for k, v in feats.items() if v}
    out["rows"] = np.asarray(pr.rows)
    out["clean_acc"] = np.array(torch.cat(ys).mean().item())
    out["source"] = np.array(meta.get("best_source", "live"))
    return out


def main() -> None:
    torch.set_num_threads(4)
    s09 = pd.read_csv(REPO / "docs/research/evidence/S09_post_sweep_forensics/runs.csv")
    cdf = pd.read_csv(EVID / "cells.csv")
    rows = []
    for arm, variant, seed in CKPTS:
        t0 = time.time()
        f = extract(arm, variant, seed)
        tr = f["rows"]
        if arm == "S08":
            r = s09[(s09.arm == "grouped") & (s09.fold == 0) & (s09.seed == seed)].iloc[0]
            att, cross, f1 = r.attraction_cross, r.cross_recall_tta, r.f1_tta
        else:
            r = cdf[cdf.cell == f"{arm}/{variant}__f0_s{seed}"].iloc[0]
            att, cross, f1 = r.attraction_cross_tta, r.cross_recall_tta, r.f1_tta
        for rep in ("embedding", "spatial", "spectral"):
            if rep not in f:
                continue
            X = np.zeros((int(tr.max()) + 1, f[rep].shape[1]))
            X[tr] = f[rep]
            if np.allclose(f[rep], 0):            # a disabled pathway's output is exactly zero
                continue
            rows.append(dict(model=f"{arm} {variant}", seed=seed, representation=rep, dim=f[rep].shape[1],
                             clean_acc_train_all=float(f["clean_acc"]), source=str(f["source"]),
                             heldout_attraction=att, heldout_cross_recall=cross, heldout_f1=f1, **probe(X, tr)))
        print(f"  {arm} {variant} s{seed}: {time.time() - t0:.0f}s", rows[-1], flush=True)
    df = pd.DataFrame(rows)
    save(df, "embed_probe.csv")
    e = df[df.representation == "embedding"]
    out = dict(n=int(len(e)),
               spearman_kappa_vs_attraction=spearmanr(e.kappa, e.heldout_attraction).correlation,
               pearson_kappa_vs_attraction=float(np.corrcoef(e.kappa, e.heldout_attraction)[0, 1]),
               spearman_kappa_vs_cross_recall=spearmanr(e.kappa, e.heldout_cross_recall).correlation)
    save(out, "embed_probe_validation.json")
    print(df.round(3).to_string())
    print(out)


if __name__ == "__main__":
    main()
