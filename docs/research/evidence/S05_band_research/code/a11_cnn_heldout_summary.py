import json, glob, numpy as np, pandas as pd
from sklearn.metrics import recall_score
from common import LABELS, REPO
st = pd.read_csv(f"{REPO}/dataset/scan_table.csv"); ns = st.groupby("label").session.nunique()
same = np.array([ns[c] == 1 for c in range(90)])
d = pd.DataFrame([json.loads(l) for l in open("cnn_heldout.jsonl")])
rs, rc = [], []
for _, r in d.iterrows():
    z = np.load(f"preds/heldout_{r.arm}_f{r.fold}_s{r.seed}.npz")
    rec = recall_score(LABELS[z["idx"]], z["pred"], average=None, labels=np.arange(90))
    rs.append(rec[same].mean()); rc.append(rec[~same].mean())
d["rec_same"], d["rec_cross"] = rs, rc
t = d.groupby("arm").agg(k=("k", "first"), n=("f1", "size"), f1=("f1", "mean"), f1_sd=("f1", "std"), acc=("acc", "mean"),
                         rec_same=("rec_same", "mean"), rec_cross=("rec_cross", "mean")).reset_index()
t["fam"] = t.arm.str.replace(r"_k\d+$", "", regex=True)
print(t.sort_values(["fam", "k"]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
