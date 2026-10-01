"""Cache the foreground-masked mean spectrum of every patch, for both cubes.

Same definition as ``spectralquadnet.experiments.baselines.mean_spectra`` (the function the
sweep's LDA baseline used), so the k32 cache reproduces the sweep baseline exactly.
Writes outputs/s09_forensics/mean_{k32,r215}.npy. ~1 min for k32, a few for the 30 GB r215.
"""
from common import CACHE, DS32, DS215

from spectralquadnet.experiments.baselines import mean_spectra

import numpy as np

for name, ds in (("k32", DS32), ("r215", DS215)):
    out = CACHE / f"mean_{name}.npy"
    if out.exists():
        print("cached", out)
        continue
    x = mean_spectra(ds / "patches.npy")
    np.save(out, x.astype(np.float32))
    print("wrote", out, x.shape)
