"""A9 — where does the session fingerprint live? Training rows only (one bundle per class).
Per band: F = n_bar * var_between_sessions(session means of class means) / mean within-session var of class means.
Null (no session effect, varieties randomly assigned to sessions) -> F ~ 1."""
import json, numpy as np, pandas as pd
from common import fold, load, LABELS, GROUPS, REPO, WL
ms = load("mean_snv").astype(np.float64); gs = np.load("cache/kernel_gainstats.npy")
st = pd.read_csv(f"{REPO}/dataset/scan_table.csv").set_index("scan_id")
sess = st.loc[GROUPS, "session_id"].values
res = {}
for f in (0, 1):
    tr, ca, _ = fold(f); rows = np.concatenate([tr, ca])
    X = np.c_[ms, gs][rows]; y = LABELS[rows]; s = sess[rows]
    cm = np.stack([X[y == c].mean(0) for c in range(90)]); cs = np.array([np.bincount(s[y == c]).argmax() for c in range(90)])
    S = np.unique(cs)
    smeans = np.stack([cm[cs == k].mean(0) for k in S]); nk = np.array([(cs == k).sum() for k in S])
    between = (nk[:, None] * (smeans - cm.mean(0)) ** 2).sum(0) / (len(S) - 1)
    within = sum(((cm[cs == k] - smeans[i]) ** 2).sum(0) for i, k in enumerate(S)) / (90 - len(S))
    res[f] = between / within
F = (res[0] + res[1]) / 2
json.dump(F.tolist(), open("a9_session_F.json", "w"))
names = [f"{w:.0f}" for w in WL] + ["log_mu", "log_sd", "-mu/sd"]
print("session F-ratio (null ~1; >~2.2 is p<0.05 for F(8,81))")
for lo, hi in [(383,430),(430,500),(500,600),(600,700),(700,800),(800,900),(900,1007)]:
    idx = np.flatnonzero((WL >= lo) & (WL < hi))
    print(f"  {lo}-{hi} nm: median F {np.median(F[idx]):.2f}   max {F[idx].max():.2f}")
for i in (256, 257, 258):
    print(f"  {names[i]}: F {F[i]:.2f}")
