"""End-to-end smoke tests on a synthetic dataset — the whole pipeline, small.

Marked ``slow``: these build a real dataset on disk, construct real models and
run real epochs. They take minutes, not seconds, and are opt-in via
``--run-slow`` (or ``--run-all``).

What they are for
─────────────────
Every unit test in this suite checks a component. None of them would have caught
the audited run's two most consequential defects, because both were *integration*
failures: Stage 2's scalars were discarded by a step counter the stages could not
see, and the grouped split existed, was correct, was tested, and was never
composed into the config anybody ran.

So these run the actual entrypoint composition against a synthetic cube shaped
like the real one — 90 classes is reduced to 8, and 8,624 patches to a few
hundred, but every structural property the protocol depends on is preserved:
**two class-pure acquisition bundles per class**, which is the constraint the
whole evaluation protocol is built around.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from _helpers import (
    AUDITED_TINY,
    FULL_BANDS,
    FULL_SPATIAL,
    N_CLASSES,
    REPO_ROOT,
    TRAIN,
    full_spectrum_overrides,
    tiny_overrides,
)

pytestmark = pytest.mark.slow

# The synthetic dataset and the tiny-run overrides live in `conftest.py` so the
# driver smoke tests share them: two fixtures shaped almost the same would let
# the two tiers diverge silently, which is the failure mode they exist to catch.
_overrides = tiny_overrides


def _run_train(overrides: list[str], config: str | None = None) -> subprocess.CompletedProcess:
    command = [sys.executable, str(TRAIN)]
    if config:
        command.append(f"--config-name={config}")
    command.extend(overrides)
    return subprocess.run(
        command, cwd=REPO_ROOT, capture_output=True, text=True, timeout=2400, check=False
    )


def _manifest(output_dir: Path) -> dict:
    path = output_dir / "results" / "run.json"
    assert path.exists(), f"no results manifest at {path}"
    return json.loads(path.read_text())


# ══════════════════════════════════════════════════════════════════════
#  The default pipeline, end to end
# ══════════════════════════════════════════════════════════════════════


def test_the_default_single_stage_run_completes_and_reports(
    synthetic_dataset, tmp_path
) -> None:
    """`python train.py` with no arguments: SeedNet, one stage, grouped, calib."""
    out = tmp_path / "single"
    result = _run_train(_overrides(synthetic_dataset, out))
    assert result.returncode == 0, f"train.py failed:\n{result.stdout[-6000:]}"

    manifest = _manifest(out)
    run = manifest["run"]
    assert run["arch"] == "spectral_seed_net"
    assert run["pipeline"] == "single"
    assert run["split_scheme"] == "grouped"
    assert run["select_split"] == "calib", "selection must never be on the reported split"
    assert run["report_split"] == "val_test"

    for variant in ("no_tta", "tta"):
        scored = manifest["results"][variant]
        assert 0.0 <= scored["macro_f1"] <= 1.0
        assert scored["macro_f1_ci"] is not None, "a bare number is not interpretable"
        assert len(scored["per_class"]) == N_CLASSES


def test_the_results_tree_and_figures_are_written(synthetic_dataset, tmp_path) -> None:
    out = tmp_path / "artifacts"
    assert _run_train(_overrides(synthetic_dataset, out)).returncode == 0

    results = out / "results"
    assert (results / "run.json").exists()
    assert list(results.glob("confusion_*.npy")), "the 90x90 confusion matrix (CHANGES §19.4)"
    assert list(results.glob("per_class_*.csv"))
    assert list(results.glob("preds_*.npy")), "needed for the paired bootstrap between arms"

    figures = out / "figures"
    pytest.importorskip("matplotlib")
    assert list(figures.glob("*.png")), "figures are generated automatically"


def test_the_run_auto_resumes_instead_of_retraining(synthetic_dataset, tmp_path) -> None:
    out = tmp_path / "resume"
    assert _run_train(_overrides(synthetic_dataset, out)).returncode == 0
    second = _run_train(_overrides(synthetic_dataset, out))
    assert second.returncode == 0
    assert "[SKIP]" in second.stdout, "a completed stage must be loaded, not retrained"


# ══════════════════════════════════════════════════════════════════════
#  The protocol guard actually fires
# ══════════════════════════════════════════════════════════════════════


def test_a_stratified_run_warns_that_its_number_is_a_mixture(
    synthetic_dataset, tmp_path
) -> None:
    out = tmp_path / "stratified"
    result = _run_train(
        _overrides(
            synthetic_dataset,
            out,
            **{"data.split_scheme": "stratified", "data.calib_frac": 0.15},
        )
    )
    assert result.returncode == 0, result.stdout[-4000:]
    assert "bundle recognition" in result.stdout


def test_the_grouped_run_reports_group_disjointness(synthetic_dataset, tmp_path) -> None:
    out = tmp_path / "disjoint"
    result = _run_train(_overrides(synthetic_dataset, out))
    assert result.returncode == 0
    assert "train↔val∪test group-disjoint: True" in result.stdout

    report = _manifest(out)["run"]["split_report"]
    assert report["train_eval_group_disjoint"] is True
    assert report["val_test_group_disjoint"] is False, (
        "two bundles per class makes three-way disjointness impossible — a stated "
        "data-collection ceiling, not a bug"
    )


# ══════════════════════════════════════════════════════════════════════
#  The audited control arm still runs
# ══════════════════════════════════════════════════════════════════════


def test_the_audited_three_stage_pipeline_still_runs(synthetic_dataset, tmp_path) -> None:
    """A3 and A8 compare against it, so it has to keep working."""
    out = tmp_path / "audited"
    result = _run_train(
        _overrides(
            synthetic_dataset,
            out,
            **AUDITED_TINY,
        ),
        config="experiment/quadnet_audited",
    )
    assert result.returncode == 0, f"the control arm failed:\n{result.stdout[-8000:]}"

    manifest = _manifest(out)
    assert manifest["run"]["arch"] == "spectral_quadnet"
    assert manifest["run"]["pipeline"] == "three_stage"


def test_no_wandb_step_collision_across_three_stages(synthetic_dataset, tmp_path) -> None:
    """IC-1's validation criterion, on the pipeline that used to collide.

    The console backend does not emit W&B's warning, so the property is checked
    where it originates: the step a structured backend would see must be
    monotone across the stage boundaries.
    """
    out = tmp_path / "steps"
    result = _run_train(
        _overrides(
            synthetic_dataset,
            out,
            **AUDITED_TINY,
            **{
                "tracking.backend": "tensorboard",
                "tracking.log_dir": str(out / "tb"),
            },
        ),
        config="experiment/quadnet_audited",
    )
    if result.returncode != 0 and "tensorboard" in result.stdout.lower():
        pytest.skip("tensorboard extra not installed")
    assert result.returncode == 0, result.stdout[-6000:]


# ══════════════════════════════════════════════════════════════════════
#  Baselines and analysis, on the same synthetic data
# ══════════════════════════════════════════════════════════════════════


def test_the_classical_baselines_run_under_the_same_protocol(
    synthetic_dataset, tmp_path
) -> None:
    """CHANGES §19.4 calls this the paper's most important baseline."""
    from spectralquadnet.config.compose import load_experiment_config
    from spectralquadnet.experiments.baselines import run_baselines

    cfg = load_experiment_config(
        overrides=_overrides(synthetic_dataset, tmp_path / "baseline")
        + ["hydra.run.dir=" + str(tmp_path / "h")]
    )
    results = run_baselines(cfg, output_dir=tmp_path / "baseline", n_boot=32)
    assert set(results) == {"lda_mean_spectrum", "linsvc_mean_spectrum"}
    for result in results.values():
        assert 0.0 <= result.macro_f1 <= 1.0
        assert result.macro_f1_ci is not None


def test_a9_characterises_the_hard_classes(synthetic_dataset, tmp_path) -> None:
    from spectralquadnet.config.compose import load_experiment_config
    from spectralquadnet.experiments.analysis import run_analysis

    cfg = load_experiment_config(overrides=_overrides(synthetic_dataset, tmp_path / "a9"))
    per_class = {c: 0.2 if c < 3 else 0.9 for c in range(N_CLASSES)}
    report = run_analysis(cfg, per_class_f1=per_class, output_dir=tmp_path / "a9", k=3)

    assert report.hard_classes == [0, 1, 2]
    assert len(report.easy_classes) == 3
    assert report.morphometrics["available"] is True
    assert report.verdict, "A9 must always state a reading, including 'inconclusive'"
    assert (tmp_path / "a9" / "results" / "a9_hard_classes.json").exists()


# ══════════════════════════════════════════════════════════════════════
#  The primary methodology, end to end at the shipped band count
# ══════════════════════════════════════════════════════════════════════


def test_the_primary_pipeline_runs_on_all_256_acquired_bands(
    full_spectrum_dataset, tmp_path
) -> None:
    """`python train.py` at its **own** `data.num_bands`, with nothing overridden.

    Every other smoke run shrinks the spectral axis to 8 bands along with
    everything else, which exercises the machinery but not the number this study
    is about. This one keeps `data.num_bands`, the two band-fraction augmentation
    widths and `model.stem_folded_depth` exactly as `configs/` ships them, so
    what it covers is the shipped spectral configuration: the stem's derived
    `(8, 2, 2)` schedule with its 15-tap first kernel, the 856-wide spectral
    descriptor, the `O(C^2)` continuum hull at C = 256, and the band-geometry
    check that gates all three.
    """
    out = tmp_path / "full_spectrum"
    result = _run_train(full_spectrum_overrides(full_spectrum_dataset, out))
    assert result.returncode == 0, f"the primary pipeline failed:\n{result.stdout[-8000:]}"

    # The run must *say* which spectral regime it is in, on its own first screen.
    assert "the full acquired cube, no band selection" in result.stdout
    assert "REDUCED arm" not in result.stdout

    manifest = _manifest(out)
    run = manifest["run"]
    assert run["arch"] == "spectral_seed_net"
    assert run["split_scheme"] == "grouped"
    assert run["band_selection"] is False, "the primary path selects no bands"
    geometry = run["band_geometry"]
    assert geometry["configured"] == FULL_BANDS
    assert geometry["selected"] == geometry["stored"] == FULL_BANDS
    assert geometry["wavelengths"] == FULL_BANDS

    for variant in ("no_tta", "tta"):
        scored = manifest["results"][variant]
        assert 0.0 <= scored["macro_f1"] <= 1.0
        assert scored["macro_f1_ci"] is not None
        assert len(scored["per_class"]) == N_CLASSES


def test_the_stem_reads_the_full_band_axis_in_the_run_that_ships(
    full_spectrum_dataset, tmp_path
) -> None:
    """The model a primary run builds has the derived schedule, not a fallback.

    Constructed from the same composition the run above executes, so this is a
    statement about the shipped config rather than about a hand-built module.
    """
    import torch

    from spectralquadnet.config.compose import load_experiment_config
    from spectralquadnet.data.mmap_store import DataStore
    from spectralquadnet.models.registry import build_model

    cfg = load_experiment_config(
        overrides=full_spectrum_overrides(full_spectrum_dataset, tmp_path / "probe")
    )
    assert cfg.data.num_bands == FULL_BANDS
    assert cfg.data.cutmix_bands == 51 and cfg.data.max_cutout_bands == 19

    DataStore.reset()
    try:
        store = DataStore.from_config(cfg.data, "cpu")
        model = build_model(cfg, store.require_wavelengths())
        stem = model.spatial.stem
        assert stem.spectral_strides == (8, 2, 2)
        assert stem.kernel_depths == (15, 5, 5)
        assert stem.folded_depth == cfg.model.stem_folded_depth == 8
        assert stem.fold[0].in_channels == 64 * 8
        # …and the spectral descriptor carries the full-resolution SNV/derivative
        # block: `3 * num_bands` of its width, whatever the shrunk bank sizes are.
        assert model.spectral.in_dim == (
            int(cfg.model.index_bank_size)
            + int(cfg.model.continuum_depths)
            + 3 * FULL_BANDS
            + int(cfg.model.n_morphometrics)
        )
        assert model.spectral.derivatives.d1_op.shape == (FULL_BANDS, FULL_BANDS)

        with torch.no_grad():
            logits = model.eval()(torch.rand(2, FULL_BANDS, FULL_SPATIAL, FULL_SPATIAL) + 0.2)
        assert logits.shape == (2, cfg.data.num_classes)
    finally:
        DataStore.reset()
