#!/usr/bin/env python3
"""S13 runtime, estimated from what the S11 cells measured on the same Kaggle T4 × 2 runtime.

Every S13 cell has a measured S11 analogue on the same data, code path and
hardware: the R1 full network (X1), the two single-pathway networks (X2, under
the shipped 150-epoch regime), and the stratified protocol (X1 stratified). From
each S11 cell's ``metrics.jsonl`` this reads the wall clock from launch to the
final-evaluation banner (so start-up — compile, page-cache warm-up — is inside it),
the epochs run and the final evaluation's own duration, and builds a per-epoch
cost. S13's cost per cell is then

    minutes = epochs × s/epoch (analogue) × adjustment + final evaluation + κ probe

with **epochs** bracketed by what R1 cells actually did (X1 stopped at 131–200,
mean 186) and the 200-epoch cap; **adjustment** = 1.00 except Y4 (×1.14: an 80/20
split trains on 0.80 × 0.85 = 68 % of kernels vs 0.70 × 0.85 = 59.5 %) and Y2
(×1.03: the MixStyle section runs eagerly between compiled stem convolutions —
an allowance, not a measurement). The κ probe is one eval-mode pass over the
training rows (≈ 1/12 of a TTA pass over a comparable split) plus a CPU LDA: < 1 min.

Writes runtime_estimate.csv and runtime_estimate.json (evidence/S13_representation_screening/).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parents[5]
S11 = REPO / "outputs/experiments_u430k32/s11"
EVID = Path(__file__).resolve().parents[1]


def timing(cell: Path) -> dict[str, float]:
    """``{train_s, epochs, final_eval_s, total_s}`` of one S11 cell."""
    elapsed, epochs, final_start = [], 0, None
    for line in (cell / "metrics.jsonl").read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "elapsed_s" in event:
            elapsed.append(float(event["elapsed_s"]))
        if event.get("event") == "epoch":
            epochs += 1
        if event.get("event") == "banner" and "FINAL" in json.dumps(event) and final_start is None:
            final_start = float(event.get("elapsed_s", 0.0))
    total = max(elapsed)
    train = final_start if final_start is not None else total
    return {"train_s": train, "epochs": float(epochs), "final_eval_s": total - train, "total_s": total}


def per_epoch(pattern: str) -> dict[str, float]:
    cells = [timing(d) for d in sorted(S11.glob(pattern)) if (d / "metrics.jsonl").exists()]
    cells = [c for c in cells if c["epochs"] > 0]
    return {
        "n": len(cells),
        "s_per_epoch": mean(c["train_s"] / c["epochs"] for c in cells),
        "final_eval_s": mean(c["final_eval_s"] for c in cells),
        "epochs_mean": mean(c["epochs"] for c in cells),
        "epochs_min": min(c["epochs"] for c in cells),
        "epochs_max": max(c["epochs"] for c in cells),
        "total_min_mean": mean(c["total_s"] for c in cells) / 60,
    }


def main() -> None:
    analogues = {
        "X1 grouped (full, R1)": per_epoch("X1/grouped__*"),
        "X1 stratified (full, R1)": per_epoch("X1/stratified__*"),
        "X2 spectral_only (shipped, ≤150)": per_epoch("X2/spectral_only__*"),
        "X2 spatial_only (shipped, ≤150)": per_epoch("X2/spatial_only__*"),
    }
    x1 = analogues["X1 grouped (full, R1)"]
    epochs_typ, epochs_cap = x1["epochs_mean"], 200.0
    probe_s = 45.0
    plan = [
        ("Y1/spectral_only", 2, "X2 spectral_only (shipped, ≤150)", 1.00),
        ("Y1/spatial_only", 2, "X2 spatial_only (shipped, ≤150)", 1.00),
        ("Y2/mixstyle", 2, "X1 grouped (full, R1)", 1.03),
        ("Y3/lean_grouped", 2, "X1 grouped (full, R1)", 1.00),
        ("Y3/lean_stratified", 1, "X1 stratified (full, R1)", 1.00),
        ("Y4/within_8020", 1, "X1 stratified (full, R1)", 1.14),
    ]
    rows = []
    for name, n, source, adj in plan:
        a = analogues[source]
        per = a["s_per_epoch"] * adj
        typ = (epochs_typ * per + a["final_eval_s"] + probe_s) / 60
        cap = (epochs_cap * per + a["final_eval_s"] + probe_s) / 60
        rows.append({
            "cells": name, "n_runs": n, "analogue": source, "adjustment": adj,
            "s_per_epoch": round(per, 2), "minutes_per_run_typical": round(typ, 1),
            "minutes_per_run_cap": round(cap, 1),
            "minutes_total_typical": round(n * typ, 1), "minutes_total_cap": round(n * cap, 1),
        })
    total_typ = sum(r["minutes_total_typical"] for r in rows)
    total_cap = sum(r["minutes_total_cap"] for r in rows)
    with (EVID / "runtime_estimate.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    out = {
        "source": "outputs/experiments_u430k32/s11/*/metrics.jsonl (Kaggle T4 x2, commit 413a11e)",
        "analogues": {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in analogues.items()},
        "epochs_typical": round(epochs_typ, 1),
        "epochs_cap": epochs_cap,
        "gpu_runs": sum(r["n_runs"] for r in rows),
        "minutes_typical": round(total_typ, 0),
        "minutes_cap": round(total_cap, 0),
        "hours_typical": round(total_typ / 60, 2),
        "hours_cap": round(total_cap / 60, 2),
        "extras_minutes": {"Y1 fusion (CPU)": 0.5, "P0 re-score of X2/spatial_only__f1_s0": 3.0},
        "parent_design": {"gpu_runs": 30, "hours": round(3 * total_typ / 60, 1)},
    }
    (EVID / "runtime_estimate.json").write_text(json.dumps(out, indent=1) + "\n")
    for r in rows:
        print(r)
    print({k: v for k, v in out.items() if k != "analogues"})


if __name__ == "__main__":
    main()
