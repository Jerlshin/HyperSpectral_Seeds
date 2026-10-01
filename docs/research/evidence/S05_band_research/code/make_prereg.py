"""Freeze the held-out confirmation list BEFORE any held-out row is scored.
Writes preregistration.json + its sha256. Arms are explicit band indices per fold."""
import hashlib, json, subprocess, time
import numpy as np
from common import REPO, WL
import bandsel as sel

g = json.load(open("glw_canonical.json"))
decision = json.load(open("decision_rule.json"))
def both(b):
    b = [int(i) for i in b]; return {"0": b, "1": b}
def per_fold(name, k):
    return {str(f): sorted(int(i) for i in g[f"{name}_f{f}"][:k]) for f in (0, 1)}
def repo(name, k):
    return {str(f): sorted(int(i) for i in np.load(f"{REPO}/outputs/band_study/bands/{name}_f{f}_k{k}.npy")) for f in (0, 1)}

lda = {}
for k in [8, 16, 24, 32, 48, 64, 96, 128, 192, 256]:
    lda[f"uniform_k{k}"] = dict(kind="bands", bands=both(sel.uniform(k)))
for k in [8, 16, 24, 32, 48, 64, 96, 128, 192]:
    lda[f"uniform430_k{k}"] = dict(kind="bands", bands=both(sel.uniform_nm(k, 430.0)))
for k in [16, 32, 48, 64, 128]:
    lda[f"uniform430+gstat1_k{k}"] = dict(kind="bands", bands=both(sel.uniform_nm(k, 430.0)), extra=[2])
    lda[f"uniform430+gstat3_k{k}"] = dict(kind="bands", bands=both(sel.uniform_nm(k, 430.0)), extra=[0, 1, 2])
for k in [8, 16, 24, 32, 48, 64, 96, 128]:
    lda[f"glw_k{k}"] = dict(kind="bands", bands=per_fold("glw", k))
for k in [16, 32, 48, 64, 96, 128]:
    lda[f"glw430_k{k}"] = dict(kind="bands", bands=per_fold("glw430", k))
for name, k in [("l1_path", 224), ("l1_path", 192), ("spa", 40), ("mrmr", 40)]:
    lda[f"repo_{name}_k{k}"] = dict(kind="bands", bands=repo(name, k))
for m in [32, 48, 64, 96]:
    lda[f"dct_m{m}"] = dict(kind="dct", m=m)

cnn = {}
for k in [16, 32, 64, 256]:
    cnn[f"uniform_k{k}"] = both(sel.uniform(k))
for k in [16, 32, 48, 64, 128]:
    cnn[f"uniform430_k{k}"] = both(sel.uniform_nm(k, 430.0))
for k in [16, 32, 64]:
    cnn[f"glw_k{k}"] = per_fold("glw", k)
for k in [32, 64]:
    cnn[f"glw430_k{k}"] = per_fold("glw430", k)
cnn["repo_l1_path_k224"] = repo("l1_path", 224)

commit = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
P = dict(
    frozen_at=time.strftime("%Y-%m-%d %H:%M:%S %Z"), repo_commit=commit,
    protocol="grouped, eval_frac=0.3, calib_frac=0.15, folds 0/1; fit on train∪calib (the fold's training bundles); score val∪test (the other bundle) once",
    lda_shrinkage=1e-3, cnn_seeds=[0, 1],
    hypotheses={
        "H1_artifact": "Arms containing 383-430 nm bands (uniform, glw) lose their within-bundle advantage over their >=430 nm counterparts on held-out bundles.",
        "H2_budget": "Held-out performance of uniform430 peaks at k<=64 and is non-inferior (>= -0.01 macro-F1) to the full 256-band cube somewhere in k in [24, 64].",
        "H3_method": "No supervised selector beats uniform430 by more than 0.01 macro-F1 (mean of 2 folds) at k>=32 on held-out.",
        "H4_brightness": "The within-bundle gain from log(mu), log(sd) (gstat3) over gstat1 shrinks on held-out (illumination-confounded).",
        "H5_extraction": "Low-pass DCT (m=48-64) >= every selected band set of equal size on held-out.",
    },
    decision_rule=decision,
    lda_arms=lda, cnn_arms=cnn,
)
s = json.dumps(P, indent=1).encode()
open("preregistration.json", "wb").write(s)
open("preregistration.sha256", "w").write(hashlib.sha256(s).hexdigest())
json.dump(cnn, open("arms_cnn_heldout.json", "w"))
print("frozen", P["frozen_at"], hashlib.sha256(s).hexdigest()[:16], len(lda), "LDA arms", len(cnn), "CNN arms")
