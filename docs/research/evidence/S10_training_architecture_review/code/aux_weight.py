"""S10 — the auxiliary-loss weight the single-stage loop actually applied (finding F54).

`engine/train_epoch.py::train_one_epoch` sets `aux_w = _aux_loss_weight(cfg, current_ep, total_ep)` on every call.
That function reads the three-stage curriculum's `stage1.aux_loss_weight_{init,final}`; the single-stage keys
`model.aux_head_weight` and `single.aux_loss_weight` are never read. This script evaluates the *repository's own*
function on the sweep's logged config, for the shipped 150 epochs and for X1's 200, and writes aux_weight_applied.csv.
"""
from __future__ import annotations

import pandas as pd
from s10common import run_config, run_dirs, save

from spectralquadnet.losses.auxiliary import _aux_loss_weight


def main() -> None:
    cfg = run_config(run_dirs()[0][3])
    rows = []
    for total in (150, 200):
        for ep in range(1, total + 1):
            rows.append(dict(total_epochs=total, epoch=ep, applied=_aux_loss_weight(cfg, ep, total),
                             configured_model_aux_head_weight=float(cfg.model.aux_head_weight),
                             configured_single_aux_loss_weight=float(cfg.single.aux_loss_weight)))
    df = pd.DataFrame(rows)
    save(df, "aux_weight_applied.csv")
    print(df.groupby("total_epochs").applied.agg(["first", "mean", "last"]))


if __name__ == "__main__":
    main()
