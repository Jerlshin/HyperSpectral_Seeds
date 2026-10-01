"""S09 · extract everything the u430k32 protocol sweep recorded into small evidence tables.

Reads outputs/experiments_u430k32/ (12 SpectralSeedNet runs + LDA/LinearSVC baselines) and
the row-aligned dataset_u430k32 metadata. No model is re-run; nothing is selected on held-out.

Writes to docs/research/evidence/S09_post_sweep_forensics/:
  runs.csv               one row per run: split sizes, selection epoch, calib, held-out ±TTA, session
  curves.csv             one row per run × epoch: loss, train acc, calib F1/acc, LR, margin, mixup
  influence.csv          pathway influence (leave-one-pathway-out KL on calib) per diagnostic epoch
  per_class.csv          run × class (TTA): F1 / precision / recall, cross-session flag, LDA recall
  session_direction.csv  cross-session recall split by which session the held-out bundle came from
  kernel_consistency.csv per grouped fold: seed agreement, vote-ensemble, oracle, consistent errors
  tta_delta.csv          paired TTA − no-TTA per run with a 2,000-resample paired bootstrap CI
  error_structure.csv    where errors go: own-session attraction for same- and cross-session kernels
  baselines.csv          LDA / LinearSVC per protocol with same/cross-session recall
  confusion_pairs.csv    top-25 off-diagonal row-normalised confusions per protocol (mean over runs)
  summary.json           headline aggregates (mean ± sd ± bootstrap-over-runs CI)
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score

from common import CROSS, EVID, GROUPS, LABELS, N_CLASSES, SAME, SCANS, SESSION, SWEEP, save

RNG = np.random.default_rng(0)
PROTO = SWEEP / "protocol"
RUN_RE = re.compile(r"(grouped|stratified)__f(\d)_s(\d)")


def run_dirs():
    for d in sorted(PROTO.iterdir()):
        m = RUN_RE.fullmatch(d.name)
        if m:
            yield d, m.group(1), int(m.group(2)), int(m.group(3))


def load_preds(d, variant):
    """(rows, preds, targets) with DDP's padded duplicate dropped, sorted by row."""
    r = np.load(d / f"results/rows_val_test_{variant}.npy")
    p = np.load(d / f"results/preds_val_test_{variant}.npy")
    t = np.load(d / f"results/targets_val_test_{variant}.npy")
    _, first = np.unique(r, return_index=True)
    r, p, t = r[first], p[first], t[first]
    assert (LABELS[r] == t).all()
    return r, p, t


def macro_f1(t, p):
    return f1_score(t, p, labels=np.arange(N_CLASSES), average="macro", zero_division=0)


def recalls(t, p):
    return recall_score(t, p, labels=np.arange(N_CLASSES), average=None, zero_division=0)


def jsonl(d):
    return [json.loads(line) for line in open(d / "metrics.jsonl")]


# class → session of its training bundle, per grouped fold (for attraction / direction)
def train_session_by_class(train_rows):
    out = np.full(N_CLASSES, -1)
    for c in range(N_CLASSES):
        s = np.unique(SESSION[train_rows][LABELS[train_rows] == c])
        assert len(s) == 1
        out[c] = s[0]
    return out


def main() -> None:
    from common import grouped_rows, stratified_rows

    sweep = {(r["arm"], r["fold"], r["seed"]): r for r in json.load(open(PROTO / "sweep.json"))["runs"]}
    split_rows = {("grouped", f): grouped_rows(f) for f in (0, 1)}
    split_rows[("stratified", 0)] = stratified_rows()

    runs, curves, infl, per_class, tta_rows, err_rows, dir_rows = [], [], [], [], [], [], []
    preds_by_fold: dict[tuple, list] = {}

    for d, arm, fold, seed in run_dirs():
        tr, ca, held = split_rows[(arm, fold)]
        meta = json.load(open(d / "stage1_meta.json"))
        last = json.load(open(d / "last_stage1.json"))
        rows = jsonl(d)

        # ── curves: merge every scalar event of an epoch into one row ─────────
        per_epoch: dict[int, dict] = {}
        for r in rows:
            if r["event"] != "scalars":
                continue
            per_epoch.setdefault(r["step"], {}).update(r["metrics"])
        for ep, m in sorted(per_epoch.items()):
            if "train/loss" in m:
                curves.append(dict(arm=arm, fold=fold, seed=seed, epoch=ep,
                                   train_loss=m["train/loss"], train_acc=m["train/acc"],
                                   calib_f1_live=m["val/f1_live"], calib_f1_ema=m["val/f1_ema"],
                                   calib_acc_live=m["val/acc_live"], calib_acc_ema=m["val/acc_ema"],
                                   lr=m["sched/lr"], margin=m["sched/arcface_m"],
                                   mixup=m["sched/mixup"], label_smooth=m["sched/label_smooth"],
                                   clip_fraction=m.get("grad_norm/clip_fraction"),
                                   preclip_backbone=m.get("grad_norm/preclip_backbone"),
                                   preclip_fusion=m.get("grad_norm/preclip_fusion"),
                                   preclip_head=m.get("grad_norm/preclip_head")))
            if "influence/branch_a" in m:
                infl.append(dict(arm=arm, fold=fold, seed=seed, epoch=ep,
                                 spatial=m["influence/branch_a"], spectral=m["influence/branch_b"],
                                 calib_diag_macro_f1=m.get("diag/macro_f1")))

        # ── held-out, both variants ─────────────────────────────────────────
        res = {}
        for v in ("no_tta", "tta"):
            r_, p_, t_ = load_preds(d, v)
            assert np.array_equal(r_, held), "eval rows differ from the reconstructed split"
            rec = recalls(t_, p_)
            res[v] = dict(rows=r_, preds=p_, targets=t_, rec=rec,
                          f1=macro_f1(t_, p_), acc=float((p_ == t_).mean()),
                          f1w=f1_score(t_, p_, average="weighted", zero_division=0))
            sess = json.load(open(d / f"results/session_val_test_{v}.json"))
            res[v]["sess"] = sess

        tta, nt = res["tta"], res["no_tta"]
        # paired bootstrap of the TTA delta over kernels
        b = RNG.integers(0, len(held), size=(2000, len(held)))
        deltas = np.array([macro_f1(tta["targets"][i], tta["preds"][i]) - macro_f1(nt["targets"][i], nt["preds"][i])
                           for i in b[:400]])  # 400 resamples: macro-F1 over 4k kernels is slow; CI is stable
        tta_rows.append(dict(arm=arm, fold=fold, seed=seed, f1_no_tta=nt["f1"], f1_tta=tta["f1"],
                             delta=tta["f1"] - nt["f1"], ci_lo=np.percentile(deltas, 2.5),
                             ci_hi=np.percentile(deltas, 97.5),
                             changed_preds=float((tta["preds"] != nt["preds"]).mean())))

        s = tta["sess"]
        clean = [c_ for c_ in curves if (c_["arm"], c_["fold"], c_["seed"]) == (arm, fold, seed)
                 and c_["mixup"] == 0 and c_["margin"] == 0]
        runs.append(dict(
            clean_fit_epoch=clean[0]["epoch"] if clean else np.nan,
            clean_train_acc=float(np.mean([c_["train_acc"] for c_ in clean])) if clean else np.nan,
            clean_calib_acc=float(np.mean([c_["calib_acc_live"] for c_ in clean])) if clean else np.nan,
            final_train_loss=[c_ for c_ in curves if (c_["arm"], c_["fold"], c_["seed"]) == (arm, fold, seed)][-1]["train_loss"],
            arm=arm, fold=fold, seed=seed, seconds=sweep[(arm, fold, seed)]["seconds"],
            n_train=len(tr), n_calib=len(ca), n_eval=len(held),
            train_per_class=len(tr) / N_CLASSES,
            best_epoch=meta["epoch"], stop_epoch=last["epoch"], early_stopped=last["epoch"] < 150,
            calib_best_f1=meta["val_f1"], calib_best_acc=meta["val_acc"],
            f1_no_tta=nt["f1"], f1_tta=tta["f1"], acc_no_tta=nt["acc"], acc_tta=tta["acc"],
            f1w_tta=tta["f1w"],
            same_recall_tta=float(tta["rec"][SAME].mean()), cross_recall_tta=float(tta["rec"][CROSS].mean()),
            same_recall_no_tta=float(nt["rec"][SAME].mean()), cross_recall_no_tta=float(nt["rec"][CROSS].mean()),
            attraction_cross=s["attraction"]["cross"] if arm == "grouped" else np.nan,
            attraction_cross_chance=s["attraction"]["cross_chance"] if arm == "grouped" else np.nan,
            entropy_bits=s["entropy"]["predicted_bits"], entropy_oracle_bits=s["entropy"]["oracle_bits"],
        ))

        pc = pd.read_csv(d / "results/per_class_val_test_tta.csv")
        pc.insert(0, "seed", seed); pc.insert(0, "fold", fold); pc.insert(0, "arm", arm)
        pc["cross_session"] = pc["class"].isin(CROSS)
        per_class.append(pc)
        preds_by_fold.setdefault((arm, fold), []).append((seed, tta["preds"], tta["targets"], held))

        # ── grouped only: direction of the cross-session contrast; error structure ──
        if arm == "grouped":
            tsess = train_session_by_class(np.concatenate([tr, ca]))
            ks, kt, kp = SESSION[held], tta["targets"], tta["preds"]
            for c in CROSS:
                sel = kt == c
                dir_rows.append(dict(fold=fold, seed=seed, cls=int(c), train_session=int(tsess[c]),
                                     heldout_session=int(np.unique(ks[sel])[0]),
                                     recall=float((kp[sel] == c).mean()),
                                     own_session_pred=float((tsess[kp[sel]] == ks[sel]).mean())))
            wrong = kp != kt
            for pop, cls in (("same", SAME), ("cross", CROSS)):
                sel = wrong & np.isin(kt, cls)
                # chance: a uniformly random *wrong* class lands in the kernel's own session
                n_own = np.array([(tsess == s_).sum() for s_ in ks[sel]])
                own_true = (tsess[kt[sel]] == ks[sel]).astype(int)  # the true class itself (excluded)
                chance = ((n_own - own_true) / (N_CLASSES - 1)).mean()
                err_rows.append(dict(fold=fold, seed=seed, population=pop, n_errors=int(sel.sum()),
                                     error_rate=float(sel.sum() / np.isin(kt, cls).sum()),
                                     errors_to_own_session=float((tsess[kp[sel]] == ks[sel]).mean()),
                                     chance=float(chance)))

    runs = pd.DataFrame(runs)
    save(runs, "runs.csv")
    save(pd.DataFrame(curves), "curves.csv")
    save(pd.DataFrame(infl), "influence.csv")
    save(pd.DataFrame(tta_rows), "tta_delta.csv")
    save(pd.DataFrame(err_rows), "error_structure.csv")
    dirs = pd.DataFrame(dir_rows)
    dirs["direction"] = np.where(dirs.heldout_session == 8, "train other → test session 8",
                                 "train session 8 → test other")
    save(dirs, "session_direction.csv")

    # ── baselines (same rows as the network) ─────────────────────────────────
    base_rows, lda_rec = [], {}
    for bdir in sorted(p for p in (SWEEP / "baselines").iterdir() if p.is_dir()):
        arm, fold = bdir.name.split("_f")[0], int(bdir.name.split("_f")[1])
        for name in ("lda", "linsvc"):
            p = np.load(bdir / f"results/preds_val_test_{name}_mean_spectrum.npy")
            t = np.load(bdir / f"results/targets_val_test_{name}_mean_spectrum.npy")
            rec = recalls(t, p)
            lda_rec[(arm, fold, name)] = rec
            base_rows.append(dict(arm=arm, fold=fold, model=name, n_eval=len(t), macro_f1=macro_f1(t, p),
                                  acc=float((p == t).mean()), same_recall=float(rec[SAME].mean()),
                                  cross_recall=float(rec[CROSS].mean())))
    save(pd.DataFrame(base_rows), "baselines.csv")

    per_class = pd.concat(per_class, ignore_index=True)
    per_class["lda_recall"] = [lda_rec[(a, f, "lda")][c] for a, f, c in
                               zip(per_class.arm, per_class.fold, per_class["class"])]
    save(per_class, "per_class.csv")

    # ── seed agreement / ensemble / oracle per fold ─────────────────────────
    cons = []
    for (arm, fold), lst in preds_by_fold.items():
        P = np.stack([p for _, p, _, _ in lst]); t = lst[0][2]
        vote = np.array([np.bincount(col, minlength=N_CLASSES).argmax() for col in P.T])
        right = P == t
        pairs = [(i, j) for i in range(len(P)) for j in range(i + 1, len(P))]
        all_wrong = (~right).all(0)
        same_wrong = all_wrong & (P == P[0]).all(0)
        # restrict to 3 seeds for a like-for-like comparison of grouped vs stratified
        P3, r3 = P[:3], right[:3]
        vote3 = np.array([np.bincount(col, minlength=N_CLASSES).argmax() for col in P3.T])
        cons.append(dict(arm=arm, fold=fold, n_seeds=len(P),
                         mean_single_f1=np.mean([macro_f1(t, p) for p in P3]),
                         vote3_f1=macro_f1(t, vote3), oracle3_acc=float(r3.any(0).mean()),
                         mean_single_acc=float(r3.mean()),
                         pairwise_agreement=float(np.mean([(P3[i] == P3[j]).mean() for i, j in pairs if j < 3])),
                         all_seeds_wrong=float((~r3).all(0).mean()),
                         errors_identical_across_seeds=float(((~r3).all(0) & (P3 == P3[0]).all(0)).sum()
                                                             / max((~r3).any(0).sum(), 1)),
                         vote_all_f1=macro_f1(t, vote), all_wrong_all=float(all_wrong.mean()),
                         same_wrong_all=float(same_wrong.mean())))
    save(pd.DataFrame(cons), "kernel_consistency.csv")

    # ── confusion structure: mean row-normalised confusion over the runs of a protocol ──
    variety = SCANS.groupby("label").variety.first()
    sessions = SCANS.groupby("label").session_id.unique()
    pairs = []
    for arm in ("grouped", "stratified"):
        C = sum(np.load(d / "results/confusion_val_test_tta.npy") for d, a, _, _ in run_dirs() if a == arm)
        R = C / C.sum(1, keepdims=True)
        np.fill_diagonal(R, 0)
        for flat in np.argsort(-R.ravel())[:25]:
            t, p_ = divmod(int(flat), N_CLASSES)
            pairs.append(dict(arm=arm, true=t, true_variety=variety[t], pred=p_, pred_variety=variety[p_],
                              rate=R[t, p_], true_cross_session=t in CROSS,
                              true_sessions=" ".join(map(str, sessions[t])),
                              pred_sessions=" ".join(map(str, sessions[p_]))))
    save(pd.DataFrame(pairs), "confusion_pairs.csv")

    # ── headline summary with a bootstrap over runs ─────────────────────────
    def agg(x):
        x = np.asarray(x, float)
        bs = RNG.choice(x, size=(5000, len(x))).mean(1)
        return dict(mean=x.mean(), sd=x.std(ddof=1), min=x.min(), max=x.max(), n=len(x),
                    boot_lo=np.percentile(bs, 2.5), boot_hi=np.percentile(bs, 97.5))

    g, s_ = runs[runs.arm == "grouped"], runs[runs.arm == "stratified"]
    summary = {k: {"grouped": agg(g[k]), "stratified": agg(s_[k])} for k in
               ("f1_tta", "f1_no_tta", "acc_tta", "calib_best_f1", "same_recall_tta", "cross_recall_tta",
                "best_epoch", "seconds")}
    summary["gap_tta"] = float(s_.f1_tta.mean() - g.f1_tta.mean())
    summary["fold_means_grouped_tta"] = g.groupby("fold").f1_tta.mean().to_dict()
    summary["calib_minus_heldout_grouped"] = agg(g.calib_best_f1 - g.f1_tta)
    summary["corr_calib_vs_heldout"] = {a: float(np.corrcoef(x.calib_best_f1, x.f1_tta)[0, 1])
                                        for a, x in (("grouped", g), ("stratified", s_))}
    # macro-recall identity: overall = 73/90·same + 17/90·cross
    summary["recall_decomposition_grouped"] = dict(
        same_weight=len(SAME) / N_CLASSES, cross_weight=len(CROSS) / N_CLASSES,
        same=g.same_recall_tta.mean(), cross=g.cross_recall_tta.mean(),
        ceiling_if_same_perfect=len(SAME) / N_CLASSES + len(CROSS) / N_CLASSES * g.cross_recall_tta.mean())
    save(summary, "summary.json")


if __name__ == "__main__":
    main()
