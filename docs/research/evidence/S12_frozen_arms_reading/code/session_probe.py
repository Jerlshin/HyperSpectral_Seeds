"""S12 · a training-rows-only proxy for session reliance, validated against held-out attraction.

Problem it addresses. Under ``grouped`` every class is trained from one bundle, so session is a deterministic
function of class inside the training set: no supervised objective can tell a session feature from a variety
feature, and calib (same bundle) rewards both. Cross-session behaviour is visible only on held-out rows, which
may not select anything (WORKFLOW §3.1). A design study needs a signal it *can* select on.

Proxy. **Class-disjoint session decodability**: on a fold's *training* rows, predict the acquisition session
from a representation with a classifier fitted on some classes and scored on *other* classes (GroupKFold by
class, 5 splits; StandardScaler → shrinkage LDA; balanced accuracy, chance-corrected as Cohen's κ). A
representation that encodes session in a way shared across varieties scores high; one whose session
information is only "which class is this" scores near chance.

Validation. For every representation that S12 also scored on held-out (``linear_controls.csv``,
``pixel_controls.csv``: shrinkage LDA), correlate the proxy with held-out cross-session attraction and recall.
Held-out numbers are read from those tables, never recomputed here.

Writes: session_probe.csv · session_probe_validation.json
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from s12common import CACHE, EVID, LABELS, REPO, SESSION, grouped_rows, save

warnings.filterwarnings("ignore")


def snv(x):
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-5)


def families() -> dict[str, np.ndarray]:
    F = dict(np.load(CACHE / "pixel_features.npz"))
    MEAN = np.load(REPO / "outputs/s09_forensics/mean_k32.npy").astype(np.float64)
    MORPH = np.load(REPO / "dataset_u430k32/morphology.npy").astype(np.float64)
    level = np.log(MEAN.mean(1, keepdims=True))
    h = lambda *a: np.hstack([np.asarray(x, np.float64) for x in a])  # noqa: E731
    return {
        # names match linear_controls.csv / pixel_controls.csv (shrinkage rows) for validation
        "raw reflectance": MEAN,
        "SNV": snv(MEAN),
        "morphometrics": MORPH,
        "raw + morph (S09 best linear)": h(MEAN, MORPH),
        "SNV + morph (spectral-path analogue)": h(snv(MEAN), MORPH),
        "SNV + level + morph": h(snv(MEAN), level, MORPH),
        "level + morph": h(level, MORPH),
        "mean + sd + morph": h(F["mean"], F["sd"], MORPH),
        "quantiles 10/50/90 + morph": h(F["q10"], F["q50"], F["q90"], MORPH),
        "core + rim + morph": h(F["core"], F["rim"], MORPH),
        "pixel-SNV mean + sd + morph": h(F["psnv_mean"], F["psnv_sd"], MORPH),
        "mean + log-cov + morph": h(F["mean"], F["logcov"], MORPH),
        "all pixel statistics + morph": h(*[F[k] for k in ("mean", "sd", "q10", "q50", "q90", "core", "rim",
                                                          "psnv_mean", "psnv_sd", "logcov")], MORPH),
        # single families, to localise the channel
        "pixel sd only": F["sd"],
        "pixel-SNV sd only": F["psnv_sd"],
        "log-cov only": F["logcov"],
    }


def probe(X: np.ndarray, rows: np.ndarray) -> dict:
    y, g = SESSION[rows], LABELS[rows]
    pred = np.empty_like(y)
    for tr, te in GroupKFold(n_splits=5).split(X[rows], y, g):
        m = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto")).fit(X[rows][tr], y[tr])
        pred[te] = m.predict(X[rows][te])
    return dict(balanced_acc=balanced_accuracy_score(y, pred), kappa=cohen_kappa_score(y, pred))


def main() -> None:
    fam = families()
    rows = []
    for f in (0, 1):
        tr = grouped_rows(f)[0]
        for name, X in fam.items():
            rows.append(dict(representation=name, fold=f, n_features=X.shape[1], **probe(X, tr)))
    df = pd.DataFrame(rows)
    save(df, "session_probe.csv")

    # validation against held-out (shrinkage-LDA rows of the two control tables)
    lin = pd.read_csv(EVID / "linear_controls.csv")
    lin = lin[lin.lda == "shrinkage"]
    pix = pd.read_csv(EVID / "pixel_controls.csv")
    pix = pix[pix.protocol == "grouped"]
    held = pd.concat([lin[["representation", "fold", "f1", "cross_recall", "attraction_cross"]],
                      pix.rename(columns={"heldout_f1": "f1"})[["representation", "fold", "f1", "cross_recall", "attraction_cross"]]])
    held = held.drop_duplicates(["representation", "fold"])
    m = df.merge(held, on=["representation", "fold"])
    agg = m.groupby("representation")[["kappa", "attraction_cross", "cross_recall", "f1"]].mean()
    out = {
        "n_representations": int(len(agg)),
        "spearman_kappa_vs_attraction": spearmanr(agg.kappa, agg.attraction_cross).correlation,
        "spearman_kappa_vs_cross_recall": spearmanr(agg.kappa, agg.cross_recall).correlation,
        "pearson_kappa_vs_attraction": float(np.corrcoef(agg.kappa, agg.attraction_cross)[0, 1]),
        "table": agg.round(4).reset_index().to_dict(orient="records"),
        "note": "held-out values are read from linear_controls.csv / pixel_controls.csv (shrinkage LDA); the proxy "
                "itself uses training rows only",
    }
    save(out, "session_probe_validation.json")
    print(df.groupby("representation")[["n_features", "balanced_acc", "kappa"]].mean().sort_values("kappa").round(3).to_string())
    print({k: v for k, v in out.items() if k != "table"})
    print(agg.sort_values("kappa").round(3).to_string())


if __name__ == "__main__":
    main()
