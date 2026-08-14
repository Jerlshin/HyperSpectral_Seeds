# `notebooks/` — exploratory, and two of them are stale

These are exploratory artifacts, not shipped code: `pyproject.toml` excludes them from `ruff`,
`mypy` does not read them, and no test imports them. They produced several figures in
`figures/`.

**Status against the current codebase**, checked by import:

| Notebook | Imports | Runs today? |
|---|---|---|
| `VIS_NIR_hyperspectral_modality.ipynb` | numpy / matplotlib / `spectral` only | **Yes.** Reads `dataset/patches.npy` and `dataset/wavelengths.csv` — the primary path's arrays — and is unaffected by the re-architecture. |
| `embedding_overview.ipynb` | `HSI_modality_training.hsi_training` | **No.** That monolith was deleted when the package was extracted; the notebook also builds a model at `num_bands=40`. |
| `pipeline_overview.ipynb` | `data_setup_v3` | **No.** Same reason. |

The two stale notebooks are **left as they are, deliberately**. They are a record of the
pre-package exploration, they are not on any execution path, and rewriting them against the
current API would produce plausible-looking cells nobody has run. If you want their analyses on
the current codebase, the package already exposes them as tested entry points:

| Instead of | Use |
|---|---|
| `embedding_overview.ipynb`'s t-SNE of test embeddings | `python -m spectralquadnet.experiments.cli analyse --run outputs/<run>` — ablation A9, which writes the t-SNE, the mean-spectrum overlays, the 90×90 confusion matrix and a segmentation audit for the persistent hard cluster |
| `pipeline_overview.ipynb`'s preprocessing walkthrough | `python scripts/prepare_dataset.py` plus `docs/02_DATASET_AND_PREPROCESSING.md`, which documents the same pipeline against the code that actually runs |

A notebook whose imports resolve is worth keeping; one whose imports do not is a claim about
code that no longer exists, and this file is here so that claim is not made silently.
