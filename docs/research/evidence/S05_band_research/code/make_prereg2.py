"""Second pre-registration: session-clean band designs. Candidate set per fold = bands whose
session F-ratio computed on THAT fold's training rows is < 2.2 (p~0.05 for F(8,81))."""
import hashlib, json, time, numpy as np, pandas as pd
from common import fold, load, LABELS, GROUPS, REPO, WL
import bandsel as sel
ms = load("mean_snv").astype(np.float64)
st = pd.read_csv(f"{REPO}/dataset/scan_table.csv").set_index("scan_id"); sess = st.loc[GROUPS, "session_id"].values
arms = {}; clean = {}
for f in (0, 1):
    tr, ca, _ = fold(f); rows = np.concatenate([tr, ca])
    X = ms[rows]; y = LABELS[rows]; s = sess[rows]
    cm = np.stack([X[y == c].mean(0) for c in range(90)]); cs = np.array([np.bincount(s[y == c]).argmax() for c in range(90)])
    S = np.unique(cs); sm = np.stack([cm[cs == k].mean(0) for k in S]); nk = np.array([(cs == k).sum() for k in S])
    Fb = ((nk[:, None] * (sm - cm.mean(0)) ** 2).sum(0) / (len(S) - 1)) / (sum(((cm[cs == k] - sm[i]) ** 2).sum(0) for i, k in enumerate(S)) / (90 - len(S)))
    cand = np.flatnonzero(Fb < 2.2); clean[f] = cand
    order = sel.greedy_lda_wrapper(ms[tr], LABELS[tr], 96, candidates=cand, seed=0)   # selection: train rows only
    for k in [16, 32, 48, 64, 96]:
        arms.setdefault(f"clean_glw_k{k}", {})[str(f)] = sorted(int(i) for i in order[:k])
    for k in [16, 32, 48, 64, 96, 128]:
        pos = np.unique(np.round(np.linspace(0, len(cand) - 1, k)).astype(int))
        arms.setdefault(f"clean_uniform_k{k}", {})[str(f)] = sorted(int(i) for i in cand[pos])
    arms.setdefault("clean_all", {})[str(f)] = [int(i) for i in cand]
    print(f"fold {f}: {len(cand)} clean bands; glw first 12 nm: {np.round(WL[order[:12]]).astype(int).tolist()}")
print("clean-set Jaccard across folds:", round(len(set(clean[0]) & set(clean[1])) / len(set(clean[0]) | set(clean[1])), 3))
P = dict(frozen_at=time.strftime("%Y-%m-%d %H:%M:%S %Z"), follows="preregistration.frozen.json (sha 5ae90caf...) — second, exploratory-but-frozen round, motivated by the session diagnostic a8/a9",
         hypotheses={"H6": "session-clean arms raise held-out cross-session recall above 0.05 (all first-round spectral arms: 0.000)",
                     "H7": "session-clean arms lose overall macro-F1 vs unconstrained arms of equal k — the size of the loss estimates the session-recognition share of the grouped score"},
         lda_shrinkage=1e-3, lda_arms=arms)
s = json.dumps(P, indent=1).encode(); open("preregistration2.json", "wb").write(s); open("preregistration2.sha256", "w").write(hashlib.sha256(s).hexdigest())
print("frozen", P["frozen_at"], hashlib.sha256(s).hexdigest()[:16])
