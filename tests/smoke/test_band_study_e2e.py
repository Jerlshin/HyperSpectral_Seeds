"""End-to-end gates for the band study and the data path it depends on.

Marked ``slow``: the second test runs a real ``train.py`` composition, which is
the only thing that can prove the band-slicing path composes. The first runs the
whole study — every stage, every artifact, every table and figure — on a
miniature cube in seconds.

Why both are here rather than in the unit tier
──────────────────────────────────────────────
The unit tests check that each piece is right. Neither of the two failures these
catch is a component failure:

* the study's stages can each be correct and still not compose — a selection
  written under one key and read under another leaves an empty grid that looks
  like "nothing to do";
* the band-index data path can slice correctly in isolation and still produce a
  model whose λ-aware operators are built for a different band count, because
  ``num_bands``, the wavelength CSV and the index array are three separate
  config keys that have to agree.

Both are exactly the shape of the integration defects the project's audit
found, which is why the repository keeps a smoke tier at all.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from _helpers import N_BANDS, REPO_ROOT, TRAIN, tiny_overrides

pytestmark = pytest.mark.slow

#: How many of the synthetic cube's bands the sliced run reads.
SUBSET = 5


def test_the_whole_study_runs_and_produces_its_report(synthetic_dataset, tmp_path) -> None:
    """`prepare → select → proxy → analyse → report` on a miniature cube.

    Asserts the artifacts a reader is promised exist and are non-trivial, and —
    the load-bearing one — that the recommendation was reached without any
    held-out record being written.
    """
    from spectralquadnet.bandstudy.analysis import run_analysis
    from spectralquadnet.bandstudy.config import BandStudyConfig
    from spectralquadnet.bandstudy.pipeline import stage_prepare, stage_proxy, stage_select
    from spectralquadnet.bandstudy.report import build_report

    cfg = BandStudyConfig(
        patches_path=synthetic_dataset["patches_data"],
        labels_path=synthetic_dataset["labels_path"],
        groups_path=synthetic_dataset["groups_path"],
        wavelength_path=synthetic_dataset["wavelength_path"],
        output_root=str(tmp_path / "study"),
        budgets=(2, 4, N_BANDS),
        methods=("uniform", "random", "mi", "mrmr", "spa"),
        proxies=("lda", "linsvc"),
        replicates=2,
        random_draws=3,
        progress=False,
    )

    assert stage_prepare(cfg).n_failed == 0
    assert stage_select(cfg).n_failed == 0
    assert stage_proxy(cfg).n_failed == 0
    payload = run_analysis(cfg)
    report = build_report(cfg)

    assert report.exists() and report.stat().st_size > 4000
    for name in (
        "curves",
        "trends",
        "null_margins",
        "stability",
        "cross_fold_agreement",
        "redundancy",
        "method_ranking",
    ):
        path = cfg.analysis_dir / f"{name}.csv"
        assert path.exists() and path.stat().st_size > 0, f"missing analysis table {name}"

    recommendation = payload["recommendation"]
    assert recommendation["recommended_budget"] in cfg.budgets
    assert recommendation["recommended_method"] in cfg.methods
    assert recommendation["decision_inputs"]["split_decisions_were_made_on"] == "calib"

    # The held-out split must not have been touched by anything above.
    assert not (
        cfg.confirm_dir / "records.jsonl"
    ).exists(), (
        "a confirm record exists but the confirm stage was never run — something read val ∪ test"
    )

    # Every canonical band set is on disk, named by (method, fold, k), and is
    # directly loadable by a training run.
    for fold in cfg.folds:
        for method in cfg.methods:
            for budget in cfg.budgets:
                path = cfg.bands_dir / f"{method}_f{fold}_k{budget}.npy"
                assert path.exists(), f"no band file {path.name}"
                bands = np.load(path)
                assert len(bands) == budget
                wavelengths = cfg.bands_dir / f"{method}_f{fold}_k{budget}_wavelengths.csv"
                assert len(wavelengths.read_text().strip().splitlines()) == budget + 1


def test_the_study_resumes_instead_of_recomputing(synthetic_dataset, tmp_path) -> None:
    """A second invocation must run nothing and still produce the same tables."""
    from spectralquadnet.bandstudy.config import BandStudyConfig
    from spectralquadnet.bandstudy.pipeline import stage_prepare, stage_proxy, stage_select

    cfg = BandStudyConfig(
        patches_path=synthetic_dataset["patches_data"],
        labels_path=synthetic_dataset["labels_path"],
        groups_path=synthetic_dataset["groups_path"],
        wavelength_path=synthetic_dataset["wavelength_path"],
        output_root=str(tmp_path / "resume"),
        budgets=(2, N_BANDS),
        methods=("uniform", "random", "mi"),
        proxies=("lda",),
        replicates=2,
        random_draws=2,
        progress=False,
    )
    stage_prepare(cfg)
    first_select = stage_select(cfg)
    first_proxy = stage_proxy(cfg)
    assert first_select.n_done > 0 and first_proxy.n_done > 0

    second_select = stage_select(cfg)
    second_proxy = stage_proxy(cfg)
    assert second_select.n_done == 0 and second_select.n_skipped == first_select.n_done
    assert second_proxy.n_done == 0 and second_proxy.n_skipped == first_proxy.n_done


def test_a_training_run_reads_a_band_subset_off_the_full_cube(synthetic_dataset, tmp_path) -> None:
    """The claim the neural stage rests on: k bands is a config change.

    A run pointed at the **full** cube plus a ``band_indices_path`` must train
    and report exactly as one pointed at a materialised k-band cube would, with
    no reduced array written anywhere. If this breaks, every neural arm the
    study plans is unrunnable — and it would break by producing a shape error
    deep inside a branch, naming neither the band file nor the config key.
    """
    bands = np.array([0, 2, 3, 5, 7][:SUBSET], dtype=np.int64)
    band_path = tmp_path / "bands.npy"
    np.save(band_path, bands)

    source = Path(synthetic_dataset["wavelength_path"]).read_text().strip().splitlines()
    wavelength_path = tmp_path / "wavelengths_subset.csv"
    wavelength_path.write_text("\n".join([source[0]] + [source[1 + int(b)] for b in bands]) + "\n")

    out = tmp_path / "sliced"
    overrides = tiny_overrides(
        synthetic_dataset,
        out,
        **{
            "data.band_indices_path": str(band_path),
            "data.wavelength_path": str(wavelength_path),
            "data.num_bands": SUBSET,
            "data.cutmix_bands": 2,
            "data.max_cutout_bands": 1,
        },
    )
    result = subprocess.run(
        [sys.executable, str(TRAIN), *overrides],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=2400,
        check=False,
    )
    assert result.returncode == 0, f"sliced train.py failed:\n{result.stdout[-6000:]}"

    manifest = json.loads((out / "results" / "run.json").read_text())
    assert manifest["run"]["split_scheme"] == "grouped"
    assert manifest["results"], "the sliced run produced no scored split"

    # Nothing was materialised: the whole point is that a budget sweep costs
    # index files rather than one reduced cube per cell.
    assert not list(tmp_path.glob("patches_*.npy"))
