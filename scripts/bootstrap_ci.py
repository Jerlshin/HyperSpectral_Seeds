#!/usr/bin/env python3
"""Phase 0 · action 0-C — bootstrap confidence intervals on macro-F1.

``IMPROVEMENT_PLAN.md`` §2.1.3 argues that the Stage-1/Stage-2/Stage-3 ranking
is decided on a **single 1,294-patch validation split** by six selection jobs,
so the winner's margin is plausibly inside the noise floor. This script puts a
number on that floor: it resamples the evaluation split with replacement and
reports, for every saved prediction array,

* a percentile CI on macro-F1, and
* a **paired** CI on each stage-vs-stage difference — paired because the same
  resample is scored for every stage, which removes the split's own variance
  and is the only comparison that answers "is the gap outside noise?".

Predictions come from ``scripts/phase0_eval_checkpoints.py``'s output
directory. Nothing is recomputed on the GPU; this reads ``.npy`` files.

Macro-F1 is computed over a fixed ``labels=range(num_classes)`` (matching
``final_eval``'s ``f1_score(..., average="macro", zero_division=0)``), not over
the labels present in a given resample, so every replicate averages over the
same 90 denominators.

Usage::

    python scripts/bootstrap_ci.py
    python scripts/bootstrap_ci.py --n-boot 20000 --alpha 0.05
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.metrics import f1_score


def macro_f1_fast(t: npt.NDArray[Any], p: npt.NDArray[Any], n_classes: int) -> float:
    """Macro-F1 over a fixed label set, via bincount rather than sklearn.

    Equivalent to ``f1_score(t, p, average="macro", zero_division=0,
    labels=range(n_classes))`` but ~100x faster, which is what makes 10,000
    replicates cheap. Verified against sklearn on the full arrays before any
    resampling (see :func:`_verify`).
    """
    tp = np.bincount(t[t == p], minlength=n_classes)[:n_classes]
    n_pred = np.bincount(p, minlength=n_classes)[:n_classes]
    n_true = np.bincount(t, minlength=n_classes)[:n_classes]
    denom = n_pred + n_true
    f1 = np.divide(2.0 * tp, denom, out=np.zeros(n_classes), where=denom > 0)
    return float(f1.mean())


def _verify(t: npt.NDArray[Any], p: npt.NDArray[Any], n_classes: int) -> None:
    """Assert the fast macro-F1 matches sklearn on the un-resampled arrays."""
    ref = f1_score(t, p, average="macro", zero_division=0, labels=list(range(n_classes)))
    fast = macro_f1_fast(t, p, n_classes)
    if abs(ref - fast) > 1e-9:
        raise SystemExit(f"macro_f1_fast disagrees with sklearn: {fast} vs {ref}")


def bootstrap(
    t: npt.NDArray[Any],
    preds: dict[str, npt.NDArray[Any]],
    n_classes: int,
    n_boot: int,
    seed: int,
) -> dict[str, npt.NDArray[Any]]:
    """Score every prediction array on the same resamples.

    Returns:
        ``{name: (n_boot,) array of macro-F1}``. Because the resample indices
        are shared, differences between two rows are paired.
    """
    rng = np.random.default_rng(seed)
    n = len(t)
    draws = {k: np.empty(n_boot) for k in preds}
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        tb = t[idx]
        for k, p in preds.items():
            draws[k][b] = macro_f1_fast(tb, p[idx], n_classes)
    return draws


def ci(x: npt.NDArray[Any], alpha: float) -> tuple[float, float]:
    """Percentile interval at level ``1 - alpha``."""
    lo, hi = np.percentile(x, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", default="outputs/phase0/preds", help="Prediction array directory.")
    ap.add_argument("--out", default="outputs/phase0", help="Directory for Phase-0 artifacts.")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--alpha", type=float, default=0.05, help="1-alpha is the CI level.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-classes", type=int, default=90)
    args = ap.parse_args()

    preds_dir = Path(args.preds)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    C = args.num_classes

    print("═" * 78)
    print(
        f"PHASE 0 · 0-C — bootstrap CIs on macro-F1   ({args.n_boot:,} resamples, "
        f"{100 * (1 - args.alpha):.0f}% percentile intervals)"
    )
    print("═" * 78)

    results: dict[str, Any] = {
        "n_boot": args.n_boot,
        "alpha": args.alpha,
        "seed": args.seed,
        "splits": {},
    }
    point_by_split: dict[str, dict[str, float]] = {}

    for split in ("val", "test"):
        tpath = preds_dir / f"{split}_targets.npy"
        if not tpath.is_file():
            print(f"\n[{split}] no targets at {tpath} — skipping")
            continue
        t = np.load(tpath)
        arrays = {
            f.stem.replace(f"{split}_", ""): np.load(f)
            for f in sorted(preds_dir.glob(f"{split}_stage*.npy"))
        }
        if not arrays:
            print(f"\n[{split}] no prediction arrays — skipping")
            continue
        for name, p in arrays.items():
            if len(p) != len(t):
                raise SystemExit(f"{name}: {len(p)} predictions vs {len(t)} targets")
            _verify(t, p, C)

        print(f"\n── {split} split  (n = {len(t):,}) ───────────────────────────────")
        draws = bootstrap(t, arrays, C, args.n_boot, args.seed)
        point = {k: macro_f1_fast(t, v, C) for k, v in arrays.items()}
        point_by_split[split] = point

        print(f"{'array':<26} {'macro-F1':>9}  {'95% CI':>18}  {'CI width':>9}  {'SE':>7}")
        rows = {}
        for k in sorted(arrays):
            lo, hi = ci(draws[k], args.alpha)
            se = float(draws[k].std(ddof=1))
            print(f"{k:<26} {point[k]:>9.4f}  [{lo:.4f}, {hi:.4f}]  {hi - lo:>9.4f}  {se:>7.4f}")
            rows[k] = {
                "macro_f1": point[k],
                "ci_low": lo,
                "ci_high": hi,
                "ci_width": hi - lo,
                "bootstrap_se": se,
            }

        # ── Paired differences ─────────────────────────────────────────
        # Two families, both paired on the same resample:
        #   · same TTA variant, different stage  — answers F-7 (§2.1.3)
        #   · same stage, different TTA variant  — answers F-6 (§2.6.3)
        # Cross-stage *and* cross-variant pairs are skipped: they confound the
        # two effects and answer nothing.
        print("\n  Paired differences (same resample scored for both):")
        print(f"  {'comparison':<52} {'Δ':>8}  {'95% CI':>18}  {'p(2-sided)':>10}  verdict")
        pairs = {}
        for a, b in combinations(sorted(arrays), 2):
            sa, va = a.split("_", 1)[0], a.split("_", 1)[-1]
            sb, vb = b.split("_", 1)[0], b.split("_", 1)[-1]
            if (va != vb) and (sa != sb):
                continue
            d = draws[a] - draws[b]
            lo, hi = ci(d, args.alpha)
            delta = point[a] - point[b]
            # Bootstrap two-sided p: twice the mass on the wrong side of zero.
            p_val = 2 * min((d <= 0).mean(), (d >= 0).mean())
            p_val = float(min(1.0, p_val))
            sig = "OUTSIDE noise" if lo > 0 or hi < 0 else "within noise"
            print(
                f"  {a + ' − ' + b:<52} {delta:>+8.4f}  [{lo:+.4f}, {hi:+.4f}]  "
                f"{p_val:>10.4f}  {sig}"
            )
            pairs[f"{a}_minus_{b}"] = {
                "family": "across_stages" if va == vb else "across_tta_variants",
                "delta": delta,
                "ci_low": lo,
                "ci_high": hi,
                "p_two_sided": p_val,
                "significant_at_alpha": bool(lo > 0 or hi < 0),
            }

        results["splits"][split] = {"n": int(len(t)), "arrays": rows, "paired": pairs}

    # ── val → test generalisation gap, per stage ───────────────────────
    if "val" in point_by_split and "test" in point_by_split:
        print("\n── val → test gap (selection split vs held-out) ─────────────")
        print(f"  {'array':<26} {'val':>8} {'test':>8} {'gap':>8}")
        gaps = {}
        for k in sorted(set(point_by_split["val"]) & set(point_by_split["test"])):
            v, te = point_by_split["val"][k], point_by_split["test"][k]
            print(f"  {k:<26} {v:>8.4f} {te:>8.4f} {v - te:>+8.4f}")
            gaps[k] = {"val": v, "test": te, "gap": v - te}
        results["val_test_gap"] = gaps

    (out_root / "bootstrap_ci.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_root / 'bootstrap_ci.json'}")


if __name__ == "__main__":
    main()
