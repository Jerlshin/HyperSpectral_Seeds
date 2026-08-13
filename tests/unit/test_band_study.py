"""Unit gates for the spectral-dimensionality study.

The tests are grouped by the property they protect, and every one of them exists
because the corresponding failure would be *silent*: a band study that leaks, or
that recommends a plateau it cannot demonstrate, or that resumes into a
different experiment, produces a table that looks exactly like a correct one.

Nothing here needs the real dataset. The fixtures build a miniature cube with
the structure the protocol depends on — two class-pure acquisition bundles per
class — and a planted informative wavelength window, so a method that works can
be distinguished from one that does not.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from spectralquadnet.bandstudy import methods as bsmethods
from spectralquadnet.bandstudy import stability as bsstability
from spectralquadnet.bandstudy.analysis import classify_trend
from spectralquadnet.bandstudy.config import BandStudyConfig, cost_estimate
from spectralquadnet.bandstudy.store import RecordStore, check_or_write_manifest

N_CLASSES = 8
BUNDLES = 2
PER_BUNDLE = 8
N_BANDS = 24
SPATIAL = 8

#: The bands that actually separate the classes in the synthetic cube. A method
#: that works finds these; the nulls do not, reliably.
INFORMATIVE = slice(8, 14)


@pytest.fixture(scope="module")
def synthetic_cube(tmp_path_factory) -> dict[str, str]:
    """A miniature cube with two class-pure bundles per class and a planted signal."""
    root = tmp_path_factory.mktemp("band_study_cube")
    rng = np.random.default_rng(0)
    n = N_CLASSES * BUNDLES * PER_BUNDLE

    patches = np.zeros((n, N_BANDS, SPATIAL, SPATIAL), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)
    groups = np.zeros(n, dtype=np.int64)
    continuum = 1.0 + 0.4 * np.sin(np.linspace(0.0, 3.0, N_BANDS))

    row = 0
    for c in range(N_CLASSES):
        signature = np.zeros(N_BANDS)
        signature[INFORMATIVE] = rng.normal(size=INFORMATIVE.stop - INFORMATIVE.start) * 0.8
        for b in range(BUNDLES):
            offset = rng.normal() * 0.02
            for _ in range(PER_BUNDLE):
                patch = (continuum + signature + offset)[:, None, None] + rng.normal(
                    size=(N_BANDS, SPATIAL, SPATIAL)
                ) * 0.05
                patch[:, :1, :] = 0.0  # exact-zero background, as the real cube has
                patches[row] = patch
                labels[row] = c
                groups[row] = c * BUNDLES + b
                row += 1

    np.save(root / "patches.npy", patches)
    np.save(root / "labels.npy", labels)
    np.save(root / "groups.npy", groups)
    wl = np.linspace(385.0, 1006.0, N_BANDS)
    (root / "wavelengths.csv").write_text(
        "index,Wavelength (nm)\n" + "\n".join(f"{i},{v:.4f}" for i, v in enumerate(wl)) + "\n"
    )
    return {
        "patches_path": str(root / "patches.npy"),
        "labels_path": str(root / "labels.npy"),
        "groups_path": str(root / "groups.npy"),
        "wavelength_path": str(root / "wavelengths.csv"),
    }


@pytest.fixture
def cfg(synthetic_cube, tmp_path) -> BandStudyConfig:
    return BandStudyConfig(
        **synthetic_cube,
        output_root=str(tmp_path / "study"),
        budgets=(2, 4, 8, N_BANDS),
        methods=("uniform", "random", "mi", "mrmr"),
        proxies=("lda",),
        replicates=2,
        random_draws=3,
        progress=False,
    )


# ══════════════════════════════════════════════════════════════════════
#  Leakage discipline — the property the whole study is built around
# ══════════════════════════════════════════════════════════════════════


def test_selection_rows_never_include_calib_val_or_test(cfg) -> None:
    """The rows a selector may see are disjoint from every split that is scored.

    This is CHANGES §4.1 as an assertion. The shipped selector ran
    `mutual_info_classif(X, y)` over all 8,624 patches including the 1,294 that
    become test, which is label leakage independent of the split protocol.
    """
    from spectralquadnet.bandstudy.data import fold_splits

    for fold_id in cfg.folds:
        fold = fold_splits(cfg, fold_id)
        heldout = fold.reveal_heldout("test asserting disjointness")
        assert not set(fold.train) & set(fold.calib)
        assert not set(fold.train) & set(heldout)
        assert not set(fold.calib) & set(heldout)
        assert fold.train.size and fold.calib.size and heldout.size


def test_grouped_split_holds_out_whole_acquisition_bundles(cfg) -> None:
    """No scan id appears in both the selection rows and the held-out rows."""
    from spectralquadnet.bandstudy.data import fold_splits

    fold = fold_splits(cfg, 0)
    groups = np.load(cfg.groups_path)
    train_scans = set(groups[fold.train].tolist())
    eval_scans = set(groups[fold.reveal_heldout("test")].tolist())
    assert not train_scans & eval_scans


def test_the_analysis_drops_any_record_that_is_not_on_calib(cfg, tmp_path) -> None:
    """A decision may never be computed from a held-out score, even by accident."""
    from spectralquadnet.bandstudy.analysis import load_records

    cfg.proxy_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "fold": 0,
            "method": "mi",
            "budget": 4,
            "rep": "full",
            "draw": 0,
            "proxy": "lda",
            "split": "calib",
            "macro_f1": 0.4,
        },
        {
            "fold": 0,
            "method": "mi",
            "budget": 4,
            "rep": "full",
            "draw": 0,
            "proxy": "lda",
            "split": "val_test",
            "macro_f1": 0.9,
        },
    ]
    (cfg.proxy_dir / "records.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    frame = load_records(cfg)
    assert list(frame["split"].unique()) == ["calib"]
    assert len(frame) == 1


def test_confirm_refuses_to_run_before_a_recommendation_exists(cfg) -> None:
    """The held-out split is not reachable until the choice it confirms is fixed."""
    from spectralquadnet.bandstudy.pipeline import stage_confirm

    with pytest.raises(FileNotFoundError, match="nothing to confirm"):
        stage_confirm(cfg)


# ══════════════════════════════════════════════════════════════════════
#  The methods
# ══════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def context(synthetic_cube) -> bsmethods.SelectionContext:
    from spectralquadnet.bandstudy.data import extract_features

    tiny = BandStudyConfig(**synthetic_cube, output_root="/tmp/unused_band_ctx", progress=False)
    spectra, _ = extract_features(tiny)
    labels = np.load(tiny.labels_path)
    return bsmethods.SelectionContext(
        x=np.asarray(spectra, dtype=np.float64), y=labels.astype(np.int64), seed=0
    )


@pytest.mark.parametrize("name", sorted(bsmethods.METHODS))
def test_every_method_returns_exactly_k_distinct_in_range_bands(name, context) -> None:
    """The invariant every downstream table assumes: k distinct valid indices.

    A method returning 19 bands at k = 20 would put a differently-sized set into
    a budget curve, and the Kuncheva index would return `nan` for it rather than
    failing — so the curve would silently be of a different quantity.
    """
    budgets = [1, 2, 5, 12, N_BANDS]
    outcome = bsmethods.run_method(name, context, budgets, draws=3)
    assert outcome.failure is None, outcome.failure
    for k in budgets:
        sets = outcome.per_budget[k]
        assert sets, f"{name} produced no set at k={k}"
        for chosen in sets:
            assert len(chosen) == k, f"{name} returned {len(chosen)} bands at k={k}"
            assert len(set(chosen)) == k, f"{name} returned duplicates at k={k}"
            assert all(0 <= b < context.n_bands for b in chosen)


@pytest.mark.parametrize("name", sorted(bsmethods.METHODS))
def test_every_ranking_method_covers_the_full_band_count(name, context) -> None:
    """A ranking must reach k = C, or its curve cannot include the full cube.

    mRMR and SPA rank only the decorrelation pre-filter's survivors; without the
    completion step their orderings stop short and the full-budget reference —
    the one point that makes an elbow falsifiable — would be unreachable.
    """
    outcome = bsmethods.run_method(name, context, [N_BANDS], draws=2)
    if bsmethods.get(name).kind != "ranking":
        pytest.skip(f"{name} is per-budget, not a ranking")
    assert outcome.ranking is not None
    assert sorted(outcome.ranking) == list(range(context.n_bands))


@pytest.mark.parametrize(
    "name", ["mi", "mrmr", "fdr", "pls_vip", "tree_importance", "l1_path", "cluster_ward"]
)
def test_supervised_methods_find_the_planted_informative_window(name, context) -> None:
    """A supervised method that cannot find a planted signal is not working.

    The synthetic cube's classes differ *only* in bands 8–13; everything else is
    a shared continuum plus noise. A method with any discriminative power should
    put most of a small budget there. The nulls are excluded from this test
    precisely because they should not.
    """
    outcome = bsmethods.run_method(name, context, [6], draws=1)
    chosen = set(outcome.per_budget[6][0])
    planted = set(range(INFORMATIVE.start, INFORMATIVE.stop))
    assert len(chosen & planted) >= 3, f"{name} chose {sorted(chosen)}, planted {sorted(planted)}"


def test_a_failing_method_is_recorded_rather_than_raised(context, monkeypatch) -> None:
    """One broken method must not take the other eleven with it."""
    monkeypatch.setitem(
        bsmethods._IMPLS, "mi", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    outcome = bsmethods.run_method("mi", context, [4])
    assert outcome.failure is not None
    assert "boom" in outcome.failure
    assert not outcome.ok


def test_random_collapses_its_draws_at_the_full_band_count(context) -> None:
    """At k = C every draw is the same set, so the null's spread must not be faked."""
    outcome = bsmethods.run_method("random", context, [4, N_BANDS], draws=5)
    assert len(outcome.per_budget[4]) == 5
    assert len(outcome.per_budget[N_BANDS]) == 1


# ══════════════════════════════════════════════════════════════════════
#  Stability and redundancy
# ══════════════════════════════════════════════════════════════════════


def test_kuncheva_is_zero_at_chance_and_one_for_identical_sets() -> None:
    """Chance correction is what makes stability comparable across budgets.

    Two random k-subsets of C bands already share k²/C by construction, so a raw
    Jaccard at large k is mostly an arithmetic identity.
    """
    assert bsstability.kuncheva_index([1, 2, 3], [1, 2, 3], n_total=100) == pytest.approx(1.0)
    # Expected overlap for k=10 of 100 is 1.0; a pair sharing exactly one band
    # is therefore exactly at chance.
    assert bsstability.kuncheva_index(
        list(range(10)), [0] + list(range(20, 29)), n_total=100
    ) == pytest.approx(0.0, abs=1e-9)


def test_kuncheva_is_undefined_when_the_subset_is_the_whole_pool() -> None:
    """At k = C every subset is the pool; reporting 1.0 would be meaningless."""
    assert np.isnan(bsstability.kuncheva_index(range(10), range(10), n_total=10))


def test_stability_of_a_single_set_is_not_reported_as_perfect() -> None:
    """`replicates=1` must yield `nan`, not the most misleading 1.0 in the study."""
    report = bsstability.stability([[1, 2, 3]], n_total=50, budget=3)
    assert report.n_pairs == 0
    assert np.isnan(report.mean_jaccard)


def test_effective_rank_distinguishes_duplicate_bands_from_independent_ones() -> None:
    """The number that separates 40 informative bands from 40 copies of six."""
    rng = np.random.default_rng(0)
    independent = rng.normal(size=(400, 6))
    duplicated = np.repeat(independent, 2, axis=1) + rng.normal(size=(400, 12)) * 1e-3
    wl = np.linspace(400, 1000, 12)

    corr_dup = np.corrcoef(duplicated.T)
    corr_ind = np.corrcoef(np.concatenate([independent, rng.normal(size=(400, 6))], axis=1).T)

    dup = bsstability.redundancy(range(12), corr_dup, wl)
    ind = bsstability.redundancy(range(12), corr_ind, wl)
    assert dup.effective_rank == pytest.approx(6.0, abs=0.5)
    assert ind.effective_rank > dup.effective_rank + 3.0


def test_contiguous_regions_merge_adjacent_bands_but_not_distant_ones() -> None:
    wl = np.linspace(400.0, 1000.0, 61)  # 10 nm steps
    regions = bsstability.contiguous_regions([0, 1, 2, 40, 41], wl, max_gap_nm=15.0)
    assert len(regions) == 2
    assert regions[0][2] == 3
    assert regions[1][2] == 2


# ══════════════════════════════════════════════════════════════════════
#  Trend classification — the study must be able to say "no plateau"
# ══════════════════════════════════════════════════════════════════════


def _verdict(scores, budgets=(1, 2, 4, 8, 16, 32, 64), tol=0.01, sigma=0.002):
    return classify_trend(0, "m", "p", list(budgets), list(scores), tol=tol, sigma=sigma)


def test_a_still_rising_curve_is_not_reported_as_a_plateau() -> None:
    """CHANGES M-14, as a gate. A monotone curve must not yield a reduction."""
    verdict = _verdict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    assert verdict.shape == "monotone_increasing"
    assert verdict.plateau_demonstrable is False
    assert verdict.plateau_budget == 64


def test_a_saturating_curve_is_located_before_its_endpoint() -> None:
    verdict = _verdict([0.1, 0.3, 0.5, 0.60, 0.605, 0.607, 0.608])
    assert verdict.shape == "saturating"
    assert verdict.plateau_demonstrable is True
    assert verdict.plateau_budget == 8


def test_a_curve_that_declines_with_more_bands_is_named_as_such() -> None:
    """More bands hurting is a real outcome at ~40 samples per class."""
    verdict = _verdict([0.1, 0.3, 0.55, 0.70, 0.65, 0.55, 0.40])
    assert verdict.shape == "peaked_declining"
    assert verdict.peak_budget == 8


def test_a_flat_curve_yields_no_plateau_claim() -> None:
    verdict = _verdict([0.50, 0.501, 0.499, 0.502, 0.5, 0.498, 0.501])
    assert verdict.shape == "flat"


def test_the_recommendation_falls_back_to_uniform_when_nothing_beats_the_null() -> None:
    """No effective method is a legitimate finding and must be reportable."""
    import pandas as pd

    from spectralquadnet.bandstudy.analysis import recommend

    curves = pd.DataFrame(
        [
            {
                "fold": 0,
                "method": m,
                "proxy": "lda",
                "budget": k,
                "score": 0.5,
                "uncertainty": 0.01,
                "family": "univariate",
            }
            for m in ("mi", "random")
            for k in (4, 8, 16)
        ]
    )
    trends = pd.DataFrame(
        [
            {
                "fold": 0,
                "method": "mi",
                "proxy": "lda",
                "shape": "saturating",
                "plateau_budget": 8,
                "plateau_demonstrable": True,
                "max_budget": 16,
            }
        ]
    )
    ranking = pd.DataFrame([{"method": "mi", "effective": False}])
    out = recommend(
        BandStudyConfig(budgets=(4, 8, 16), folds=(0,)),
        curves,
        trends,
        ranking,
        pd.DataFrame(),
        [],
    )
    assert out["recommended_method"] == "uniform"
    assert out["fallback_reason"] == "no_effective_method"


# ══════════════════════════════════════════════════════════════════════
#  Configuration and resumability
# ══════════════════════════════════════════════════════════════════════


def test_a_config_without_the_full_band_count_is_refused() -> None:
    """A sweep truncated below the full cube cannot demonstrate any elbow."""
    problems = BandStudyConfig(budgets=(5, 10, 40)).validate(n_bands_available=256)
    assert any("full band count" in p for p in problems)


def test_a_config_without_the_random_null_is_refused() -> None:
    problems = BandStudyConfig(methods=("mrmr", "spa")).validate(n_bands_available=256)
    assert any("random" in p for p in problems)


def test_a_stratified_config_is_refused_as_a_default() -> None:
    problems = BandStudyConfig(split_scheme="stratified").validate()
    assert any("acquisition bundle" in p for p in problems)


def test_non_semantic_fields_do_not_change_the_fingerprint() -> None:
    """Resuming with `--verbose` must not look like a different experiment."""
    base = BandStudyConfig()
    assert base.fingerprint() == base.with_(verbose=True, jobs=8, note="x").fingerprint()
    assert base.fingerprint() != base.with_(budgets=(1, 2, 256)).fingerprint()


def test_resuming_into_a_different_study_is_refused(tmp_path) -> None:
    """Two configurations' cells must never end up in one table."""
    first = BandStudyConfig(output_root=str(tmp_path / "s"), budgets=(5, 256))
    check_or_write_manifest(first)
    second = first.with_(budgets=(10, 256))
    with pytest.raises(ValueError, match="different study configuration"):
        check_or_write_manifest(second)


def test_the_record_store_skips_cells_it_already_has(tmp_path) -> None:
    key = ("fold", "method", "budget")
    store = RecordStore(tmp_path / "recs", key)
    store.append({"fold": 0, "method": "mi", "budget": 4, "macro_f1": 0.3})
    store.close()

    resumed = RecordStore(tmp_path / "recs", key)
    assert resumed.has(fold=0, method="mi", budget=4)
    assert not resumed.has(fold=1, method="mi", budget=4)
    assert len(resumed.read()) == 1


def test_a_truncated_final_record_is_dropped_rather_than_fatal(tmp_path) -> None:
    """A killed run loses its last line and nothing else."""
    directory = tmp_path / "recs"
    directory.mkdir()
    (directory / "records.jsonl").write_text(
        json.dumps({"fold": 0, "method": "mi", "budget": 4}) + '\n{"fold": 1, "met'
    )
    store = RecordStore(directory, ("fold", "method", "budget"))
    assert len(store.read()) == 1


def test_the_cost_estimate_matches_the_pipeline_expansion(cfg) -> None:
    """A cost printed before a multi-hour run must be the cost that is paid."""
    from spectralquadnet.bandstudy.pipeline import _proxy_cells, load_inputs, stage_select

    stage_select(cfg)
    inputs = load_inputs(cfg, quiet=True)
    cells = _proxy_cells(cfg, inputs)
    estimate = cost_estimate(cfg)
    # The estimate deducts the full-budget duplicates that the proxy stage
    # de-duplicates at fit time; the cell list still contains them.
    assert estimate["band_sets"] * len(cfg.proxies) == len(cells)


# ══════════════════════════════════════════════════════════════════════
#  The band-slicing data path (BS-1)
# ══════════════════════════════════════════════════════════════════════


def test_band_indices_slice_the_patch_and_leave_everything_else_alone(tmp_path) -> None:
    """The mechanism that makes a k-band run cost a config change, not 14 GB."""
    from types import SimpleNamespace

    from spectralquadnet.data.datasets import RiceSeedDataset

    patches = np.arange(4 * 6 * 2 * 2, dtype=np.float32).reshape(4, 6, 2, 2)
    labels = np.array([0, 1, 0, 1], dtype=np.int64)
    bands = np.array([1, 3, 5], dtype=np.int64)
    band_path = tmp_path / "bands.npy"
    np.save(band_path, bands)

    store = SimpleNamespace(
        require_patches=lambda: patches,
        require_labels=lambda: labels,
        masks=None,
        patches_path=None,
        masks_path=None,
    )
    data_cfg = SimpleNamespace(
        band_indices_path=str(band_path),
        num_bands=3,
        max_cutout_bands=1,
        noise_std=0.0,
        cutmix_bands=0,
        cutmix_spatial=0,
    )
    dataset = RiceSeedDataset(np.array([0, 2]), store=store, data_cfg=data_cfg)
    patch, label = dataset[0]
    assert patch.shape == (3, 2, 2)
    assert np.allclose(patch.numpy(), patches[0][bands])
    assert int(label) == 0


def test_a_band_index_file_that_disagrees_with_num_bands_is_refused(tmp_path) -> None:
    """A silent mismatch builds a model whose λ vector describes other bands."""
    from types import SimpleNamespace

    from spectralquadnet.data.datasets import _band_selection

    path = tmp_path / "bands.npy"
    np.save(path, np.array([0, 1, 2], dtype=np.int64))
    with pytest.raises(ValueError, match="num_bands"):
        _band_selection(SimpleNamespace(band_indices_path=str(path), num_bands=40))


def test_no_band_index_path_reads_the_cube_unchanged() -> None:
    """The default must be byte-identical to the pre-band-study behaviour."""
    from types import SimpleNamespace

    from spectralquadnet.data.datasets import _band_selection

    assert _band_selection(SimpleNamespace(band_indices_path="", num_bands=40)) is None
    assert _band_selection(SimpleNamespace(num_bands=40)) is None
