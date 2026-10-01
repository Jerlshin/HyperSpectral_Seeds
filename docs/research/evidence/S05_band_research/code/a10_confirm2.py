import hashlib, json, sys, numpy as np, pandas as pd, warnings
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, recall_score
from common import fold, load, LABELS, REPO
warnings.filterwarnings("ignore")
pre = open("preregistration2.json", "rb").read()
assert hashlib.sha256(pre).hexdigest() == open("preregistration2.sha256").read().strip()
P2 = json.loads(pre); P1 = json.load(open("preregistration.frozen.json"))
ms = load("mean_snv").astype(np.float64); gs = np.load("cache/kernel_gainstats.npy"); y = LABELS
st = pd.read_csv(f"{REPO}/dataset/scan_table.csv"); ns = st.groupby("label").session.nunique()
same = np.array([ns[c] == 1 for c in range(90)])
arms = {**{a: dict(bands=v) for a, v in P2["lda_arms"].items()},
        **{a: P1["lda_arms"][a] for a in P1["lda_arms"] if P1["lda_arms"][a]["kind"] == "bands" and a.split("_k")[0] in ("uniform", "glw", "uniform430", "uniform430+gstat3")}}
rows = []
for f in (0, 1):
    tr, ca, held = fold(f); fit = np.concatenate([tr, ca])
    print(f"!! HELD-OUT REVEAL fold {f} (round 2 + diagnostic breakdown of round-1 arms)", flush=True)
    for arm, a in arms.items():
        X = ms[:, a["bands"][str(f)]]
        if a.get("extra"): X = np.c_[X, gs[:, a["extra"]]]
        m = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage=1e-3)).fit(X[fit], y[fit])
        pred = m.predict(X[held]); rec = recall_score(y[held], pred, average=None, labels=np.arange(90))
        rows.append(dict(arm=arm, fold=f, k=X.shape[1], round=2 if arm in P2["lda_arms"] else 1,
                         f1=f1_score(y[held], pred, average="macro"), rec_same=rec[same].mean(), rec_cross=rec[~same].mean()))
d = pd.DataFrame(rows); d.to_json("confirm2.json", orient="records", indent=1)
t = d.groupby(["round", "arm"]).agg(k=("k", "first"), f1=("f1", "mean"), rec_same=("rec_same", "mean"), rec_cross=("rec_cross", "mean")).reset_index()
t["fam"] = t.arm.str.replace(r"_k\d+$", "", regex=True)
pd.set_option("display.width", 200)
print(t.sort_values(["fam", "k"]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
