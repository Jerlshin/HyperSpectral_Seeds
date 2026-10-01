"""S10 — training dynamics of the 12 sweep runs, read from their metrics.jsonl.

Complements S09's curves.csv with the series S09 did not extract (auxiliary-head loss,
pathway influence, per-group clip fractions, steps/epoch, skipped batches) and with the
phase boundaries of the shipped regime made explicit. Train and calib only.

Writes:
  dynamics_epochs.csv   one row per run × epoch
  dynamics_phases.csv   per arm × phase medians (warm-up, mixup early/mid/late, margin ramp, margin held)
  transitions.csv       per run: what changed at the mixup → clean boundary (epoch 110 → 111) and
                        across the margin phase; where the selected checkpoint sits
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from s10common import run_dirs, save

KEEP = {
    "train/loss": "train_loss", "train/acc": "train_acc", "loss/branch_spatial_raw": "aux_loss_raw",
    "val/f1_live": "calib_f1_live", "val/f1_ema": "calib_f1_ema", "val/acc_live": "calib_acc_live",
    "sched/lr": "lr", "sched/arcface_m": "margin", "sched/mixup": "mixup", "sched/label_smooth": "label_smooth",
    "grad_norm/preclip_backbone": "preclip_backbone", "grad_norm/preclip_fusion": "preclip_fusion",
    "grad_norm/preclip_head": "preclip_head", "grad_norm/clipped_backbone": "clipped_backbone",
    "grad_norm/clipped_fusion": "clipped_fusion", "grad_norm/clipped_head": "clipped_head",
    "train/steps": "steps", "train/skipped_batches": "skipped",
    # SpectralSeedNet has no pathway_labels(): branch_mask index 0 = spatial (logged "a"), 1 = spectral ("b")
    "influence/branch_a": "influence_spatial", "influence/branch_b": "influence_spectral",
}
PHASES = [("1 warm-up 1-5", 1, 5), ("2 mixup 6-30", 6, 30), ("3 mixup 31-60", 31, 60), ("4 mixup 61-110", 61, 110),
          ("5 clean, margin ramp 111-130", 111, 130), ("6 clean, margin 0.30 131-150", 131, 150)]


def main() -> None:
    from s10common import epoch_table

    rows, trans = [], []
    for arm, fold, seed, d in run_dirs():
        t = epoch_table(d)
        t = t[[c for c in KEEP if c in t.columns]].rename(columns=KEEP)
        t = t[t.index <= 150]
        t.insert(0, "seed", seed)
        t.insert(0, "fold", fold)
        t.insert(0, "arm", arm)
        rows.append(t.reset_index())
        best = t[["calib_f1_live", "calib_f1_ema"]].max(axis=1)
        b_ep = int(best.idxmax())

        def at(col: str, ep: int) -> float:
            return float(t.loc[ep, col]) if ep in t.index else np.nan

        trans.append(dict(
            arm=arm, fold=fold, seed=seed, last_epoch=int(t.index.max()), best_epoch=b_ep,
            best_calib_f1=float(best.max()),
            lr_mult_at_111=at("lr", 111) / 5e-4,
            train_acc_110_mixup=at("train_acc", 110), train_acc_111_clean=at("train_acc", 111),
            train_loss_110=at("train_loss", 110), train_loss_111=at("train_loss", 111),
            calib_f1_live_110=at("calib_f1_live", 110), calib_f1_live_111=at("calib_f1_live", 111),
            calib_f1_best_by_110=float(best.loc[:110].max()),
            calib_f1_best_111_150=float(best.loc[111:].max()),
            calib_gain_after_mixup_off=float(best.loc[111:].max() - best.loc[:110].max()),
            calib_f1_live_mean_126_130=float(t.loc[126:130, "calib_f1_live"].mean()),
            calib_f1_live_mean_131_150=float(t.loc[131:150, "calib_f1_live"].mean()),
            train_loss_end=float(t.train_loss.iloc[-1]),
            margin_train_acc_end=float(t.train_acc.iloc[-1]),
            aux_loss_raw_111=at("aux_loss_raw", 111), aux_loss_raw_end=float(t.aux_loss_raw.iloc[-1]),
            preclip_backbone_median_61_110=float(t.loc[61:110, "preclip_backbone"].median()),
            preclip_backbone_median_131_150=float(t.loc[131:150, "preclip_backbone"].median()),
            clipped_backbone_min_after_30=float(t.loc[31:, "clipped_backbone"].min()),
            epochs_fully_clipped_after_30=float((t.loc[31:, "clipped_backbone"] >= 0.999).mean()),
            epochs_with_nonfinite_preclip_mean=int(t.preclip_backbone.isna().sum()),
            skipped_batches_total=int(t.skipped.sum()),
            steps_per_epoch=int(t.steps.median()),
            ema_minus_live_mean_11_110=float((t.calib_f1_ema - t.calib_f1_live).loc[11:110].mean()),
            ema_minus_live_mean_131_150=float((t.calib_f1_ema - t.calib_f1_live).loc[131:150].mean()),
        ))
    ep = pd.concat(rows, ignore_index=True)
    save(ep, "dynamics_epochs.csv")
    save(pd.DataFrame(trans), "transitions.csv")

    ph = []
    for label, a, b in PHASES:
        sub = ep[(ep.epoch >= a) & (ep.epoch <= b)]
        med = sub.groupby("arm").median(numeric_only=True)
        for arm, r in med.iterrows():
            ph.append({"arm": arm, "phase": label, **{k: r[k] for k in KEEP.values() if k in r.index}})
    save(pd.DataFrame(ph), "dynamics_phases.csv")


if __name__ == "__main__":
    main()
