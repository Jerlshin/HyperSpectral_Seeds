#!/usr/bin/env python3
"""S11 validation of the P0.2 clean-fit probe against S10's offline measurement.

S10 P0.2's acceptance criterion: *"value on a shipped checkpoint reproduces
``ckpt_fit.csv`` within 0.01"*. This script runs the probe exactly as the
training pipeline builds it — the run's composed config, its own split, its
train-standardised morphometrics, ``CleanFitProbe`` + ``measure_fit`` — on the
selected checkpoint of each S08 sweep run, live and EMA weights, and compares:

* ``full``   — the probe over **all** training kernels: must equal S10's
               ``ckpt_fit.csv`` (split=train) to numerical precision, which shows
               the two implement the same definition;
* ``subset`` — the shipped 1,000-kernel fixed subset (``clean_fit_seed=0``): the
               number X1/X4 will log. Its gap to ``full`` is sampling error.

Train rows only — no held-out row is read. CPU.

Usage (repository root; needs ``outputs/experiments_u430k32/`` and
``dataset_u430k32/``)::

    python docs/research/evidence/S11_frozen_arms_execution/code/clean_fit_reproduction.py \\
        [--full-runs grouped__f0_s0 stratified__f0_s0]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch

from spectralquadnet.config.compose import load_experiment_config
from spectralquadnet.data.loaders import build_split_bundle, standardised_morphometrics
from spectralquadnet.data.mmap_store import DataStore
from spectralquadnet.engine.checkpoint import load_ckpt
from spectralquadnet.engine.clean_fit import CleanFitProbe, measure_fit
from spectralquadnet.models.ema import ModelEMA
from spectralquadnet.models.registry import build_model
from spectralquadnet.utils.seed import set_seed

SWEEP = Path("outputs/experiments_u430k32/protocol")
S10 = Path("docs/research/evidence/S10_training_architecture_review/ckpt_fit.csv")
OUT = Path("docs/research/evidence/S11_frozen_arms_execution/clean_fit_reproduction.csv")


def measure_run(run: str, full: bool) -> list[dict[str, object]]:
    arm, cell = run.split("__")
    fold, seed = int(cell[1]), int(cell.split("_s")[1])
    cfg = load_experiment_config(
        overrides=[f"data=ablation/u430k32_{arm}", f"data.split_fold={fold}", f"seed={seed}"]
    )
    device = torch.device("cpu")
    set_seed(seed)
    store = DataStore.from_config(cfg.data, device)
    splits = build_split_bundle(cfg)
    model = build_model(cfg, store.require_wavelengths())
    ema = ModelEMA(model, decay=0.999)
    load_ckpt(str(SWEEP / run / "best_stage1.pth"), model, ema, device)
    morph = standardised_morphometrics(store, splits.train)
    rows: list[dict[str, object]] = []
    sizes = [("subset", 1000)] + ([("full", len(splits.train))] if full else [])
    for kind, n in sizes:
        probe = CleanFitProbe.build(
            store=store,
            data_cfg=cfg.data,
            device=device,
            train_rows=splits.train,
            labels=splits.labels,
            morph=morph,
            n=n,
            seed=0,
        )
        assert probe is not None
        for weights, net in (("live", model), ("ema", ema.shadow)):
            t0 = time.time()
            fit = measure_fit(net, probe.loader, device)
            rows.append(
                dict(arm=arm, fold=fold, seed=seed, weights=weights, kind=kind, n=fit.n,
                     acc=fit.acc, ce=fit.ce, seconds=round(time.time() - t0, 1))
            )
            print(rows[-1], flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="+", default=sorted(p.name for p in SWEEP.glob("*__f*_s*")))
    ap.add_argument("--full-runs", nargs="+", default=["grouped__f0_s0", "stratified__f0_s0"])
    args = ap.parse_args()
    torch.set_num_threads(4)

    rows = [r for run in args.runs for r in measure_run(run, run in args.full_runs)]
    df = pd.DataFrame(rows)
    ref = pd.read_csv(S10)
    ref = ref[ref.split == "train"][["arm", "fold", "seed", "weights", "acc", "ce"]]
    ref = ref.rename(columns={"acc": "s10_acc", "ce": "s10_ce"})
    df = df.merge(ref, on=["arm", "fold", "seed", "weights"], how="left")
    df["d_acc"] = df["acc"] - df["s10_acc"]
    df["d_ce"] = df["ce"] - df["s10_ce"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, float_format="%.6g")
    full = df[df.kind == "full"]
    sub = df[df.kind == "subset"]
    print(f"\nfull-train |Δacc| max {full.d_acc.abs().max():.2e}, |Δce| max {full.d_ce.abs().max():.2e}")
    print(f"subset     |Δacc| max {sub.d_acc.abs().max():.4f}, mean {sub.d_acc.abs().mean():.4f}")
    ok = (full.d_acc.abs().max() < 1e-4) and (sub.d_acc.abs().max() <= 0.02)
    print("reproduces S10:", "YES" if ok else "NO")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
