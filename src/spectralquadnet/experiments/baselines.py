"""LDA and LinearSVC on mean spectra, under the *identical* protocol.

CHANGES §19.4: *"Report the LDA-on-mean-spectra baseline (0.5916 leaky)
recomputed under the grouped protocol. **This is the paper's most important
baseline** and it costs seconds."*

Why it is the most important one
────────────────────────────────
It is the honest floor. Two numbers bracket what this dataset contains:

* LDA on 40-band mean spectra, patch-level 5-fold: **0.5916** accuracy. About
  59 points are available from the global mean spectrum alone, with no spatial
  information whatsoever.
* The full 5.19 M model under the same leaky protocol: **~0.845**. So ~25 points
  are attributable to everything beyond the mean spectrum.

That pair is what justifies keeping a joint spectral–spatial operator at all,
and what makes "the model is too big" answerable rather than a matter of taste.
Recomputing the first number under the *grouped* protocol is what turns it from
an anecdote into the control every deep-learning number is measured against.

Cost: seconds. It reads the mean spectrum of each patch (a chunked pass over
the mmapped cube) and fits two linear models. No GPU, no training loop.

The same split builder
──────────────────────
The splits come from :func:`~spectralquadnet.data.loaders.build_split_bundle`,
the same function the neural runs use, driven by the same config. A baseline
computed on a differently-built split is not a baseline for anything.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from spectralquadnet.data.loaders import build_split_bundle
from spectralquadnet.reporting.artifacts import RunArtifacts
from spectralquadnet.reporting.metrics import ClassificationResult, score

if TYPE_CHECKING:  # pragma: no cover - typing only
    from spectralquadnet.config.schema import ExperimentConfig

_log = logging.getLogger(__name__)

#: Patches read per chunk when computing mean spectra. Bounded so the pass runs
#: in a fixed amount of RAM against a 5.6 GB mmapped cube.
CHUNK: int = 2048

#: Background threshold. Identical to the one the model's ``foreground_mask``
#: uses, so the baseline averages over the same pixels the network sees.
FOREGROUND_EPS: float = 1e-5


@dataclass(frozen=True)
class Baseline:
    """One classical model, named and constructed."""

    name: str
    #: Built per call rather than stored fitted — a shared estimator across
    #: folds would carry the previous fold's coefficients into the next.
    factory: Any
    note: str


def default_baselines(seed: int = 0) -> list[Baseline]:
    """The two CHANGES §19.4 asks for.

    Both are pipelined with ``StandardScaler``: the 40 bands differ in scale by
    more than an order of magnitude across the VIS–NIR range, and an unscaled
    ``LinearSVC`` would be fitting the loudest bands rather than the most
    discriminative.
    """
    return [
        Baseline(
            name="lda_mean_spectrum",
            factory=lambda: make_pipeline(
                StandardScaler(), LinearDiscriminantAnalysis(solver="svd", tol=1e-4)
            ),
            note="LDA on the foreground mean spectrum — the 0.5916 leaky reference.",
        ),
        Baseline(
            name="linsvc_mean_spectrum",
            factory=lambda: make_pipeline(
                StandardScaler(), LinearSVC(C=0.1, max_iter=5000, random_state=seed)
            ),
            note="LinearSVC, C=0.1 — more robust than LDA off-Gaussian.",
        ),
    ]


def mean_spectra(
    patches_path: str | Path, chunk: int = CHUNK
) -> npt.NDArray[np.float32]:
    """``(N, C)`` foreground-masked mean spectrum per patch.

    Masked, not a plain spatial mean: the patches are zero outside the kernel by
    construction (the preprocessing divides by the fill fraction after the
    resize), so including the padding would scale every spectrum by that
    patch's fill fraction and reintroduce a size-dependent gain the pipeline
    went to some trouble to remove.
    """
    patches = np.load(patches_path, mmap_mode="r")
    n, c = patches.shape[0], patches.shape[1]
    out = np.zeros((n, c), dtype=np.float32)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = np.asarray(patches[start:stop], dtype=np.float32)
        flat = block.reshape(len(block), c, -1)
        mask = (np.abs(flat).sum(axis=1, keepdims=True) > FOREGROUND_EPS).astype(np.float32)
        valid = mask.sum(axis=2).clip(min=FOREGROUND_EPS)
        out[start:stop] = (flat * mask).sum(axis=2) / valid
    return out


def run_baselines(
    cfg: ExperimentConfig | Any,
    output_dir: str | Path | None = None,
    baselines: list[Baseline] | None = None,
    n_boot: int = 2000,
) -> dict[str, ClassificationResult]:
    """Fit and score every baseline under ``cfg``'s split protocol.

    Fitted on ``train`` (calib excluded, exactly as the neural runs are) and
    scored on the same split ``cfg.evaluation.report_split`` names, so the
    number is directly comparable to the model's.

    Returns:
        ``{baseline_name: ClassificationResult}``.
    """
    splits = build_split_bundle(cfg)
    x = mean_spectra(cfg.data.patches_data)
    y = np.asarray(splits.labels)

    if str(cfg.evaluation.report_split) == "val_test":
        eval_idx = np.sort(np.concatenate([splits.val, splits.test]))
        split_name = "val_test"
    else:
        eval_idx = splits.test
        split_name = "test"

    x_train, y_train = x[splits.train], y[splits.train]
    x_eval, y_eval = x[eval_idx], y[eval_idx]
    _log.info(
        "baselines: fit on %d patches, score %d, %d classes, %d bands",
        len(x_train),
        len(x_eval),
        int(cfg.data.num_classes),
        x.shape[1],
    )

    artifacts = RunArtifacts.for_run(output_dir) if output_dir else None
    results: dict[str, ClassificationResult] = {}

    for baseline in baselines or default_baselines(seed=int(cfg.seed)):
        model = baseline.factory()
        model.fit(x_train, y_train)
        preds = np.asarray(model.predict(x_eval))
        result = score(
            y_eval,
            preds,
            num_classes=int(cfg.data.num_classes),
            split=f"{split_name}_{baseline.name}",
            n_boot=n_boot,
            seed=int(cfg.seed),
            context={
                "baseline": baseline.name,
                "note": baseline.note,
                "split_scheme": str(cfg.data.split_scheme),
                "split_fold": int(cfg.data.split_fold),
                "seed": int(cfg.seed),
                "n_features": int(x.shape[1]),
                "n_train": int(len(x_train)),
                # Recorded because it is the honest description of what a
                # classical baseline "parameter count" is, and a table that
                # compares it to 2.8 M without saying so is misleading.
                "parameters": int(x.shape[1] * int(cfg.data.num_classes)),
            },
        )
        results[baseline.name] = result
        ci = f"  CI95={result.macro_f1_ci}" if result.macro_f1_ci else ""
        _log.info(
            "  %-22s macroF1=%.4f%s  acc=%.4f",
            baseline.name,
            result.macro_f1,
            ci,
            result.accuracy,
        )
        if artifacts is not None:
            artifacts.write_predictions(result.split, preds, y_eval)
            artifacts.write_result(result)

    if artifacts is not None:
        artifacts.write_manifest(
            {
                "run": {
                    "arch": "classical_baseline",
                    "pipeline": "none",
                    "split_scheme": str(cfg.data.split_scheme),
                    "split_fold": int(cfg.data.split_fold),
                    "seed": int(cfg.seed),
                    "select_split": "none (no model selection)",
                    "report_split": split_name,
                    "parameters": int(x.shape[1] * int(cfg.data.num_classes)),
                },
                "results": {k: v.as_dict() for k, v in results.items()},
                # There is no checkpoint, no early stopping and no selection
                # event, which is itself worth recording: this number carries
                # none of the selection bias the neural ones have to argue about.
                "note": "Classical baseline: fitted once on train, scored once. No selection.",
            }
        )
    return results
