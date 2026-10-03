"""S12 · what each pathway knows, and whether joint training adds anything over combining them afterwards.

X2 trained the spectral pathway alone (``spectral_only``: SNV-family descriptor + morphometrics → MLP) and
the spatial pathway alone (``spatial_only``: 3-D/2-D CNN, no morphometrics), under the shipped regime, on
the same folds and seeds as the S08 full network. With their saved logits (calib and held-out, ±TTA):

1. **Complementarity** — on the same held-out kernels, how often is each pathway right when the other is
   wrong; the oracle (either right) bounds what any combination could reach.
2. **Late fusion** — mean of the two pathways' log-probabilities, (a) equal weight, (b) the weight chosen
   on *calib* (grid 0…1). Compared with the jointly trained full network of the same fold and seed.
   This is a diagnostic: no fused model is a candidate and nothing is selected on held-out.
3. **Session behaviour** of each, with attraction computed here the way ``reporting/session.py`` does
   (share of cross-session kernels predicted as a class whose training bundle is in their own session).

Writes: pathway_fusion.csv · pathway_complementarity.csv · pathway_fusion_delta.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from s12common import (CROSS, SAME, SESSION, cells, grouped_rows, load_logits, load_preds,
                       macro_f1, recalls, reference_cells, save, train_session_by_class)


def session_metrics(rows: np.ndarray, t: np.ndarray, p: np.ndarray, tr_sess: np.ndarray) -> dict:
    rec = recalls(t, p)
    is_cross = np.isin(t, CROSS)
    attr = float((tr_sess[p[is_cross]] == SESSION[rows][is_cross]).mean())
    return dict(f1=macro_f1(t, p), acc=float((t == p).mean()), same_recall=float(rec[SAME].mean()),
                cross_recall=float(rec[CROSS].mean()), attraction_cross=attr)


def logp(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def main() -> None:
    cs = {c.name: c for c in cells(("X2",))}
    ref = {(c.fold, c.seed): c for c in reference_cells() if c.variant == "grouped"}
    tr_sess = {f: train_session_by_class(grouped_rows(f)[0]) for f in (0, 1)}

    # attraction as computed here must reproduce the pipeline's own number
    c0 = cs["X2/spectral_only__f0_s0"]
    r, p, t = load_preds(c0, "tta")
    mine = session_metrics(r, t, p, tr_sess[0])["attraction_cross"]
    theirs = json.load(open(c0.path / "results/run.json"))["results"]["tta"]["session"]["attraction"]["cross"]
    assert abs(mine - theirs) < 1e-9, (mine, theirs)

    fus, comp, paired = [], [], []
    for fold, seed in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        so, sp = cs[f"X2/spectral_only__f{fold}_s{seed}"], cs[f"X2/spatial_only__f{fold}_s{seed}"]
        nm = cs[f"X2/no_morph__f{fold}_s{seed}"]
        if not sp.scored:
            continue  # spatial_only f1 s0 was never scored (S12 §4.3)
        for v in ("tta", "no_tta"):
            rows, z_so, t = load_logits(so, f"val_test_{v}")
            rows2, z_sp, _ = load_logits(sp, f"val_test_{v}")
            _, z_nm, _ = load_logits(nm, f"val_test_{v}")
            assert np.array_equal(rows, rows2)
            _, c_so, ct = load_logits(so, f"calib_{v}")
            _, c_sp, _ = load_logits(sp, f"calib_{v}")
            # weight chosen on calib only
            grid = np.linspace(0, 1, 21)
            calib_f1 = [macro_f1(ct, (w * logp(c_so) + (1 - w) * logp(c_sp)).argmax(1)) for w in grid]
            w_star = float(grid[int(np.argmax(calib_f1))])
            preds = {
                "spectral_only": z_so.argmax(1), "spatial_only": z_sp.argmax(1), "no_morph": z_nm.argmax(1),
                "fusion equal": (0.5 * logp(z_so) + 0.5 * logp(z_sp)).argmax(1),
                "fusion calib-w": (w_star * logp(z_so) + (1 - w_star) * logp(z_sp)).argmax(1),
            }
            rf, pf, tf = load_preds(ref[(fold, seed)], v)
            assert np.array_equal(rf, rows)
            preds["full (S08, joint)"] = pf
            if v == "tta":
                paired.append((fold, t, rows, preds["fusion equal"], pf))
            for k, pr in preds.items():
                m = session_metrics(rows, t, pr, tr_sess[fold])
                fus.append(dict(fold=fold, seed=seed, view=v, model=k,
                                w_spectral=w_star if k == "fusion calib-w" else (0.5 if k == "fusion equal" else np.nan),
                                calib_f1_at_w=max(calib_f1) if k == "fusion calib-w" else np.nan, **m))
            if v == "tta":
                a, b = preds["spectral_only"] == t, preds["spatial_only"] == t
                fj = preds["full (S08, joint)"] == t
                for pop, mask in (("all", np.ones_like(t, bool)), ("same", np.isin(t, SAME)), ("cross", np.isin(t, CROSS))):
                    aa, bb, ff = a[mask], b[mask], fj[mask]
                    comp.append(dict(fold=fold, seed=seed, population=pop, n=int(mask.sum()),
                                     spectral_acc=aa.mean(), spatial_acc=bb.mean(), both=(aa & bb).mean(),
                                     only_spectral=(aa & ~bb).mean(), only_spatial=(~aa & bb).mean(),
                                     neither=(~aa & ~bb).mean(), oracle=(aa | bb).mean(), full_acc=ff.mean(),
                                     full_right_when_only_spectral=ff[aa & ~bb].mean() if (aa & ~bb).any() else np.nan,
                                     full_right_when_only_spatial=ff[~aa & bb].mean() if (~aa & bb).any() else np.nan,
                                     error_corr=float(np.corrcoef(~aa, ~bb)[0, 1])))
    # paired kernel bootstrap of (equal-weight late fusion − joint full network), pooled over the matched cells
    rng = np.random.default_rng(12)
    from hypotheses import conf, cross_recall_from_conf, f1_from_conf
    def stat(fn, idx):
        return np.mean([fn(conf(t[i], a[i])) - fn(conf(t[i], b[i])) for (f, t, r, a, b), i in zip(paired, idx)])
    full_idx = [np.arange(len(x[1])) for x in paired]
    out = {}
    for name, fn in (("macro_f1", f1_from_conf), ("cross_recall", cross_recall_from_conf)):
        draws = [stat(fn, [rng.integers(0, len(x[1]), len(x[1])) for x in paired]) for _ in range(2000)]
        out[name] = dict(delta=float(stat(fn, full_idx)), lo=float(np.percentile(draws, 2.5)),
                         hi=float(np.percentile(draws, 97.5)))
    out["cells"] = [f"f{x[0]}" for x in paired]
    out["note"] = "test-set sampling only (3 matched fold/seed cells); seed variance not included"
    save(out, "pathway_fusion_delta.json")
    print(out)
    fdf, cdf = pd.DataFrame(fus), pd.DataFrame(comp)
    save(fdf, "pathway_fusion.csv")
    save(cdf, "pathway_complementarity.csv")
    print(fdf[fdf.view == "tta"].groupby("model")[["f1", "same_recall", "cross_recall", "attraction_cross", "w_spectral"]]
          .mean().round(4).to_string())
    print(cdf.groupby("population").mean(numeric_only=True).round(3).drop(columns=["fold", "seed"]).to_string())


if __name__ == "__main__":
    main()
