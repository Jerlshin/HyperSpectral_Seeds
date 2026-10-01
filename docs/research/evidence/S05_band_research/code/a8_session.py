"""A8 — DIAGNOSTIC breakdown of pre-registered arms (no selection): does held-out performance
depend on whether a variety's two bundles share an acquisition session?"""
import json, numpy as np, pandas as pd, warnings
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import recall_score
from common import fold, load, LABELS, REPO
warnings.filterwarnings("ignore")
P = json.load(open("preregistration.frozen.json"))
ms = load("mean_snv").astype(np.float64); gs = np.load("cache/kernel_gainstats.npy"); y = LABELS
st = pd.read_csv(f"{REPO}/dataset/scan_table.csv")
same = st.groupby("label").session.nunique() == 1           # True: both bundles in one session
same_cls = set(same[same].index); cross_cls = set(same[~same].index)
print(f"same-session varieties: {len(same_cls)}   cross-session varieties: {len(cross_cls)}")
arms = ["uniform_k256", "uniform_k64", "uniform430_k64", "uniform430+gstat1_k64", "uniform430+gstat3_k64", "glw_k64", "glw430_k64", "gstat3_only"]
rows = []
for f in (0, 1):
    tr, ca, held = fold(f); fit = np.concatenate([tr, ca])
    for arm in arms:
        if arm == "gstat3_only":
            X = gs
        else:
            a = P["lda_arms"][arm]; X = ms[:, a["bands"][str(f)]]
            if a.get("extra"): X = np.c_[X, gs[:, a["extra"]]]
        m = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage=1e-3)).fit(X[fit], y[fit])
        pred = m.predict(X[held])
        rec = recall_score(y[held], pred, average=None, labels=np.arange(90))
        rows.append(dict(arm=arm, fold=f, same=np.mean([rec[c] for c in same_cls]), cross=np.mean([rec[c] for c in cross_cls]), all=rec.mean()))
d = pd.DataFrame(rows).groupby("arm")[["all", "same", "cross"]].mean().loc[arms]
d["same-cross"] = d["same"] - d["cross"]
print("held-out balanced recall (mean of 2 folds), by whether the variety's two bundles share a session")
print(d.round(4).to_string())
