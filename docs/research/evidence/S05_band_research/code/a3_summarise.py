import json, sys
import numpy as np, pandas as pd
rows = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "a3_budget_all.json"))
df = pd.DataFrame(rows)
for proxy in df.proxy.unique():
    d = df[df.proxy == proxy]
    # mean over 2 folds x 5 outer; SE over the 10 cells (they are not independent across outer, so this is a lower bound)
    t = d.groupby(["method", "k"]).f1.agg(["mean", "std", "count"]).reset_index()
    piv = t.pivot(index="method", columns="k", values="mean").round(3)
    print(f"\n==== proxy {proxy}: within-training-bundle 5-fold CV macro-F1 (mean of 2 folds x 5 splits) ====")
    print(piv.to_string())
    # paired difference vs uniform at same (fold, outer, k)
    base = d[d.method == "uniform"].set_index(["fold", "outer", "k"]).f1
    print("\n  paired Δ vs uniform (mean ± sd over the 10 cells):")
    for m in sorted(d.method.unique()):
        if m == "uniform":
            continue
        x = d[d.method == m].set_index(["fold", "outer", "k"]).f1
        dd = (x - base.reindex(x.index)).dropna().reset_index()
        g = dd.groupby("k").f1.agg(["mean", "std"])
        print(f"  {m:13s} " + "  ".join(f"k{k}:{r['mean']:+.3f}±{r['std']:.3f}" for k, r in g.iterrows()))
