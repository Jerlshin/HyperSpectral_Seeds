"""S14 · the training-rows session κ (D26) across the S13 cells — validated, completed and checked against attraction.

1. Validation of the in-pipeline probe (P0.5): the S12 offline probe (``session_probe.probe`` on representations
   extracted from the checkpoint, training rows, eval mode) is re-run on one S13 checkpoint and compared with the
   κ that cell's own final report wrote (``run.json → session_probe``).
2. Fold-matched references S12 did not compute: X1 grouped f1 s0 (R1), X2 spectral_only f1 s0 and X2
   spatial_only f1 s0 (shipped). The last one was never scored on held-out (F71a) — κ needs training rows only.
3. D26's reversal trigger: for each S13 grouped single-network cell vs its fold/seed-matched X1 cell, do Δκ and
   Δ held-out attraction agree in direction, beyond their noise? Noise scales: κ — the largest |Δκ| between two
   seeds of the same arm and fold (S12 ``embed_probe.csv``); attraction — X1's run-level sd (0.020).
4. κ vs attraction across every grouped network with both (S12's 11 fold-0 checkpoints + S13 + the new refs).

No held-out row is loaded: held-out attraction is read from ``cells.csv`` tables. CPU, ≈ 1 min per checkpoint.
Writes: session_kappa.csv · session_kappa_validation.json
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch
from s14common import EVID, S12_EVID, S13, SWEEP, save
from scipy.stats import spearmanr
from session_probe import probe  # S12's probe, verbatim

from spectralquadnet.config.compose import load_experiment_config
from spectralquadnet.data.loaders import build_split_bundle, standardised_morphometrics
from spectralquadnet.data.mmap_store import DataStore
from spectralquadnet.engine.batch import side_inputs, unpack_batch
from spectralquadnet.engine.checkpoint import load_ckpt
from spectralquadnet.engine.clean_fit import CleanFitProbe
from spectralquadnet.models.ema import ModelEMA
from spectralquadnet.models.registry import build_model
from spectralquadnet.utils.seed import set_seed

OFFLINE = [  # (label, run dir, fold, seed) — validation first, then the missing fold-1 references
    ("Y3 lean_grouped (validation)", S13 / "Y3/lean_grouped__f0_s0", 0, 0),
    ("Y1 spectral_only (validation)", S13 / "Y1/spectral_only__f1_s0", 1, 0),
    ("X1 grouped", SWEEP / "s11/X1/grouped__f1_s0", 1, 0),
    ("X2 spectral_only", SWEEP / "s11/X2/spectral_only__f1_s0", 1, 0),
    ("X2 spatial_only", SWEEP / "s11/X2/spatial_only__f1_s0", 1, 0),
]


def extract(d, fold: int, seed: int) -> dict[str, np.ndarray]:
    """S12 ``embed_probe.extract``, generalised to any fold and to the S13 model keys."""
    extra = json.load(open(d / "frozen_cell.json"))["arm_overrides"]
    cfg = load_experiment_config(overrides=["data=ablation/u430k32_grouped", f"data.split_fold={fold}",
                                            f"seed={seed}", *extra])
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
    with torch.no_grad():
        for batch in pr.loader:
            x, y, mask, mo = unpack_batch(batch, device)
            _, emb = net(x, return_embed=True, **side_inputs(mask, mo))
            feats["embedding"].append(emb)
    for h in hooks:
        h.remove()
    out = {k: torch.cat(v).numpy() for k, v in feats.items() if v}
    out["rows"] = np.asarray(pr.rows)
    return out


def offline_kappas() -> pd.DataFrame:
    torch.set_num_threads(4)
    rows = []
    for label, d, fold, seed in OFFLINE:
        t0 = time.time()
        f = extract(d, fold, seed)
        tr = f["rows"]
        for rep in ("embedding", "spatial", "spectral"):
            if rep not in f or np.allclose(f[rep], 0):      # a disabled pathway's output is exactly zero
                continue
            X = np.zeros((int(tr.max()) + 1, f[rep].shape[1]))
            X[tr] = f[rep]
            rows.append(dict(model=label, cell=str(d.relative_to(SWEEP)), fold=fold, seed=seed, representation=rep,
                             **probe(X, tr)))
        print(f"  {label} f{fold}: {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    cache = EVID / "session_kappa_offline.csv"
    off = pd.read_csv(cache) if cache.exists() else offline_kappas()
    if not cache.exists():
        save(off, "session_kappa_offline.csv")

    # 1 · validation of the in-pipeline κ
    s14 = pd.read_csv(EVID / "cells.csv")
    val = {}
    for label, d, *_ in OFFLINE[:2]:
        name = str(d.relative_to(S13))
        rj = s14[s14.cell == name].iloc[0]
        for rep in ("embedding", "spatial", "spectral"):
            o = off[(off.model == label) & (off.representation == rep)]
            if len(o) and not pd.isna(rj[f"kappa_{rep}"]):
                val[f"{name}:{rep}"] = dict(offline=float(o.kappa.iloc[0]), in_pipeline=float(rj[f"kappa_{rep}"]),
                                            abs_diff=abs(float(o.kappa.iloc[0]) - float(rj[f"kappa_{rep}"])))

    # 2 · one table: every grouped network with a κ and a held-out attraction
    s12p = pd.read_csv(S12_EVID / "embed_probe.csv")
    s12c = pd.read_csv(S12_EVID / "cells.csv").set_index("cell")
    tab = [dict(model=r.model, regime="R1" if r.model.startswith("X1") or r.model.startswith("X4") else "shipped",
                fold=0, seed=r.seed, representation=r.representation, kappa=r.kappa, attraction=r.heldout_attraction,
                cross=r.heldout_cross_recall, f1=r.heldout_f1, source="S12 offline")
           for r in s12p.itertuples()]
    for r in off[~off.model.str.contains("validation")].itertuples():
        name = r.cell.split("/", 1)[1]
        c = s12c.loc[name] if name in s12c.index else None
        scored = c is not None and not pd.isna(c.f1_tta)
        tab.append(dict(model=r.model, regime="R1" if r.model.startswith("X1") else "shipped", fold=r.fold, seed=r.seed,
                        representation=r.representation, kappa=r.kappa,
                        attraction=c.attraction_cross_tta if scored else np.nan,
                        cross=c.cross_recall_tta if scored else np.nan, f1=c.f1_tta if scored else np.nan,
                        source="S14 offline"))
    for r in s14[(s14.protocol == "grouped") & (s14.variant != "fused")].itertuples():
        for rep in ("embedding", "spatial", "spectral"):
            k = getattr(r, f"kappa_{rep}")
            if not pd.isna(k):
                tab.append(dict(model=f"{r.arm} {r.variant}", regime="R1", fold=r.fold, seed=r.seed, representation=rep,
                                kappa=k, attraction=r.attraction_cross_tta, cross=r.cross_recall_tta, f1=r.f1_tta,
                                source="S13 in-pipeline"))
    tab = pd.DataFrame(tab)
    save(tab, "session_kappa.csv")

    # 3 · D26 trigger: matched S13 cell vs X1 s0 (same fold)
    emb = tab[tab.representation == "embedding"]
    x1 = {f: emb[(emb.model == "X1 grouped") & (emb.fold == f) & (emb.seed == 0)].iloc[0] for f in (0, 1)}
    pairs = s12p[s12p.representation == "embedding"].groupby("model").kappa.agg(lambda s: s.max() - s.min())
    kappa_noise = float(pairs.max())
    att_noise = 0.020362
    d26 = []
    for r in emb[emb.source == "S13 in-pipeline"].itertuples():
        ref = x1[r.fold]
        dk, da = r.kappa - ref.kappa, r.attraction - ref.attraction
        d26.append(dict(cell=f"{r.model} f{r.fold}", d_kappa=dk, d_attraction=da,
                        agree=bool(np.sign(dk) == np.sign(da)),
                        beyond_noise=bool(abs(dk) > kappa_noise and abs(da) > att_noise)))
    fires = [x for x in d26 if not x["agree"] and x["beyond_noise"]]

    have = emb.dropna(subset=["attraction"])
    corr = dict(n=int(len(have)), spearman=float(spearmanr(have.kappa, have.attraction).correlation),
                pearson=float(np.corrcoef(have.kappa, have.attraction)[0, 1]))
    sp = have[~have.model.str.contains("spectral_only")]
    corr_spatial = dict(n=int(len(sp)), spearman=float(spearmanr(sp.kappa, sp.attraction).correlation),
                        pearson=float(np.corrcoef(sp.kappa, sp.attraction)[0, 1]))
    out = dict(validation=val, kappa_noise_between_seeds=kappa_noise, attraction_noise_sd=att_noise,
               x1_reference={f"f{f}": dict(kappa=float(x1[f].kappa), attraction=float(x1[f].attraction)) for f in (0, 1)},
               d26_pairs=d26, d26_trigger_fires_for=[x["cell"] for x in fires],
               kappa_vs_attraction_all_networks=corr, kappa_vs_attraction_networks_with_spatial_pathway=corr_spatial)
    save(out, "session_kappa_validation.json")
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
