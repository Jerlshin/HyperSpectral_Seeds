"""A5 — spatial-spectral proxy: a small 2-D CNN on the 16x16 masked-average-pooled cube.
Closer to the deployed network than any mean-spectrum model (it sees within-kernel spatial
spectral structure), cheap enough for ~minute-scale arms on MPS.

usage: python a5_cnn.py <split: calib|heldout> <arms.json> <out.jsonl> [seeds]
arms.json: {"name": {"0": [bands fold0], "1": [bands fold1]}, ...}
"""
import json, sys, time, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
from common import fold, load, LABELS

split, arms_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
seeds = [int(s) for s in sys.argv[4].split(",")] if len(sys.argv) > 4 else [0]
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
pool = load("pool16", mmap=True)
fill = load("fill16")
EPOCHS, BS, LR = 60, 128, 2e-3

class Net(nn.Module):
    def __init__(s, k, w=96, ncls=90):
        super().__init__()
        s.spec = nn.Sequential(nn.Conv2d(k + 1, w, 1), nn.BatchNorm2d(w), nn.GELU())   # per-pixel spectral mixing (+fill)
        s.b1 = nn.Sequential(nn.Conv2d(w, w, 3, padding=1), nn.BatchNorm2d(w), nn.GELU())
        s.b2 = nn.Sequential(nn.Conv2d(w, 2 * w, 3, padding=1, stride=2), nn.BatchNorm2d(2 * w), nn.GELU())
        s.b3 = nn.Sequential(nn.Conv2d(2 * w, 2 * w, 3, padding=1), nn.BatchNorm2d(2 * w), nn.GELU())
        s.drop = nn.Dropout(0.3)
        s.fc = nn.Linear(2 * w, ncls)
    def forward(s, x, m):
        h = s.spec(torch.cat([x, m], 1))
        h = s.b1(h) + h
        h = s.b2(h)
        h = s.b3(h) + h
        mm = F.avg_pool2d(m, 2)
        h = (h * mm).sum((2, 3)) / mm.sum((2, 3)).clamp(min=1e-3)           # masked global mean
        return s.fc(s.drop(h))

def d4(x, m):
    k = int(torch.randint(4, (1,)))
    x, m = torch.rot90(x, k, (2, 3)), torch.rot90(m, k, (2, 3))
    if torch.rand(1) < 0.5:
        x, m = x.flip(3), m.flip(3)
    return x, m

def run(bands, tr, ev, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    bands = np.asarray(bands)
    X = torch.from_numpy(np.asarray(pool[:, bands], dtype=np.float32))
    M = torch.from_numpy(fill.astype(np.float32))[:, None]
    mu = X[tr].mean((0, 2, 3), keepdim=True); sd = X[tr].std((0, 2, 3), keepdim=True) + 1e-6
    X = ((X - mu) / sd) * (M > 0)
    y = torch.from_numpy(LABELS)
    Xtr, Mtr, ytr = X[tr].to(dev), M[tr].to(dev), y[tr].to(dev)
    Xev, Mev = X[ev].to(dev), M[ev].to(dev)
    net = Net(len(bands)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.05)
    steps = EPOCHS * int(np.ceil(len(tr) / BS))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.15)
    for ep in range(EPOCHS):
        net.train()
        perm = torch.randperm(len(tr), device=dev)
        for i in range(0, len(tr), BS):
            idx = perm[i:i + BS]
            xb, mb = d4(Xtr[idx], Mtr[idx])
            loss = F.cross_entropy(net(xb, mb), ytr[idx], label_smoothing=0.1)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
    net.eval()
    with torch.no_grad():
        logits = 0
        for k in range(4):                                  # D4 rotation TTA
            logits = logits + net(torch.rot90(Xev, k, (2, 3)), torch.rot90(Mev, k, (2, 3))).float()
        pred = logits.argmax(1).cpu().numpy()
    yt = LABELS[ev]
    return f1_score(yt, pred, average="macro"), accuracy_score(yt, pred), pred

arms = json.load(open(arms_path))
done = set()
if os.path.exists(out_path):
    for l in open(out_path):
        r = json.loads(l); done.add((r["arm"], r["fold"], r["seed"]))
for f in (0, 1):
    tr, ca, held = fold(f)
    ev = ca if split == "calib" else held
    if split == "heldout":          # fixed epochs, no early stopping -> train on all training-bundle rows
        tr = np.concatenate([tr, ca])
    if split == "heldout":
        print(f"!! HELD-OUT REVEAL fold {f}: scoring pre-registered arms from {arms_path}", flush=True)
    for name, per_fold in arms.items():
        for seed in seeds:
            if (name, f, seed) in done:
                continue
            t = time.time()
            f1, acc, pred = run(per_fold[str(f)], tr, ev, seed)
            os.makedirs("preds", exist_ok=True)
            np.savez_compressed(f"preds/{split}_{name}_f{f}_s{seed}.npz", idx=ev, pred=pred)
            del pred
            import gc; gc.collect()
            if dev.type == "mps": torch.mps.empty_cache()
            rec = dict(arm=name, fold=f, seed=seed, split=split, k=len(per_fold[str(f)]), f1=f1, acc=acc)
            open(out_path, "a").write(json.dumps(rec) + "\n")
            print(f"fold {f} {name:24s} k={rec['k']:3d} seed {seed}  F1 {f1:.4f} acc {acc:.4f}  ({time.time()-t:.0f}s)", flush=True)
