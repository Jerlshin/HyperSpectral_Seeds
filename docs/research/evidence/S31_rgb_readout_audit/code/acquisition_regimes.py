"""Label-free RGB acquisition regimes: 2-means on standardized per-scan plate-background colour
and kernel sharpness (acquisition_audit.csv). Sessions are only cross-tabulated afterwards; no
variety label is read. Output: evidence/S31_rgb_readout_audit/acquisition_regimes.csv.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans

ROOT = Path("docs/research/evidence/S31_rgb_readout_audit")
FEATURES = ["bg_r", "bg_g", "bg_b", "core_laplacian_var", "edge_gradient"]


def main() -> None:
    d = pd.read_csv(ROOT / "acquisition_audit.csv")
    x = d[FEATURES].to_numpy()
    labels = KMeans(2, n_init=20, random_state=0).fit_predict((x - x.mean(0)) / x.std(0))
    # Name the regimes by measurement, not by session: "soft" = lower kernel texture energy.
    soft = int(pd.Series(d.core_laplacian_var).groupby(labels).median().idxmin())
    d["regime"] = ["soft" if v == soft else "sharp" for v in labels]
    d[["scan_id", "session_id", "regime", *FEATURES]].to_csv(ROOT / "acquisition_regimes.csv", index=False)
    print(pd.crosstab(d.session_id, d.regime).to_string())


if __name__ == "__main__":
    main()
