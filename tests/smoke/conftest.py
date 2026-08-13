"""A miniature dataset with the real one's load-bearing structure.

The one property that must survive the shrink is **two class-pure acquisition
bundles per class**. It is simultaneously:

* why a patch-level split leaks — a model that learns a tray's residual
  radiometric signature scores correctly on that tray's held-out patches
  without knowing anything about the variety;
* why three-way group disjointness is *mathematically impossible* — with two
  bundles per class, a class can appear in at most two of three splits, so
  ``val`` and ``test`` must be halves of the same held-out bundle;
* why there are exactly two folds and no third.

A synthetic fixture that got this wrong would test the code and none of the
protocol. 90 classes become 8 and 8,624 patches become a few hundred; the
structure is preserved exactly.
"""

from __future__ import annotations

import numpy as np
import pytest
from _helpers import BUNDLES_PER_CLASS, N_BANDS, N_CLASSES, PATCHES_PER_BUNDLE, SPATIAL


@pytest.fixture(scope="session")
def synthetic_dataset(tmp_path_factory) -> dict[str, str]:
    """Write the miniature dataset once per session and return its paths.

    Each bundle carries a small additive offset, standing in for the residual
    tray signature a patch-level split leaks. The background is exact zero, so
    the masked pooling operators see the invariant they are built on.
    """
    root = tmp_path_factory.mktemp("synthetic_dataset")
    rng = np.random.default_rng(0)

    n = N_CLASSES * BUNDLES_PER_CLASS * PATCHES_PER_BUNDLE
    patches = np.zeros((n, N_BANDS, SPATIAL, SPATIAL), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)
    groups = np.zeros(n, dtype=np.int64)
    masks = np.ones((n, SPATIAL, SPATIAL), dtype=np.float16)
    morph = rng.normal(size=(n, 8)).astype(np.float32)

    row = 0
    for c in range(N_CLASSES):
        class_signature = rng.normal(size=(N_BANDS, 1, 1)) * 0.5
        for b in range(BUNDLES_PER_CLASS):
            bundle_offset = rng.normal(size=(N_BANDS, 1, 1)) * 0.1
            for _ in range(PATCHES_PER_BUNDLE):
                patch = (
                    class_signature
                    + bundle_offset
                    + rng.normal(size=(N_BANDS, SPATIAL, SPATIAL)) * 0.2
                )
                patch[:, :2, :] = 0.0
                patch[:, -2:, :] = 0.0
                masks[row, :2, :] = 0.0
                masks[row, -2:, :] = 0.0
                patches[row] = patch
                labels[row] = c
                groups[row] = c * BUNDLES_PER_CLASS + b
                row += 1

    np.save(root / "patches.npy", patches)
    np.save(root / "labels.npy", labels)
    np.save(root / "groups.npy", groups)
    np.save(root / "masks.npy", masks)
    np.save(root / "morphology.npy", morph)
    # Irregularly spaced, like the SPA-selected subset: the λ-aware operators
    # are only meaningful against a non-uniform grid.
    wl = np.sort(rng.uniform(400, 1000, size=N_BANDS))
    (root / "wavelengths.csv").write_text(
        "Band,Wavelength (nm)\n" + "\n".join(f"{i},{v:.2f}" for i, v in enumerate(wl)) + "\n"
    )
    return {
        "patches_data": str(root / "patches.npy"),
        "labels_path": str(root / "labels.npy"),
        "groups_path": str(root / "groups.npy"),
        "masks_path": str(root / "masks.npy"),
        "morphology_path": str(root / "morphology.npy"),
        "wavelength_path": str(root / "wavelengths.csv"),
    }


