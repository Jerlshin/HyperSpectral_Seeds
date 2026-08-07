"""T4-6 / M-14 — is the elbow at k = 40 demonstrable?

§2.2.11: ``dataset/band_selection_report.csv`` records seven band counts,
5 … 40, and 40 is the value the run *selected*. ``find_elbow`` returns the
smallest k reaching ``elbow_pct`` of the peak — but the peak is taken over the
counts supplied, so a curve that stops at its own elbow satisfies the criterion
by construction. The elbow claim in the paper is therefore unfalsifiable from
its own artifact.

:func:`test_the_shipped_curve_cannot_demonstrate_its_elbow` runs the verdict on
the real committed CSV, which is the measurement M-14 asserts, and the rest of
the module pins the machinery that makes the claim checkable next time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spectralquadnet.data.prep.band_selection import (
    ElbowVerdict,
    find_elbow,
    load_deployed_curve,
    verify_elbow,
)
from spectralquadnet.data.prep.config import BandSelectionConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
SHIPPED_REPORT = REPO_ROOT / "dataset" / "band_selection_report.csv"


@pytest.fixture
def cfg_bs() -> BandSelectionConfig:
    return BandSelectionConfig()


# ══════════════════════════════════════════════════════════════════════
#  The measurement on the shipped artifact
# ══════════════════════════════════════════════════════════════════════


def test_the_shipped_curve_cannot_demonstrate_its_elbow(cfg_bs) -> None:
    """M-14, measured on the committed report rather than argued.

    The recorded curve rises monotonically to its final point, which is the k
    it chose. Every larger band count is unevaluated, so "40 is where accuracy
    plateaus" is a claim the artifact contains no evidence for either way.
    """
    if not SHIPPED_REPORT.exists():
        pytest.skip(f"{SHIPPED_REPORT} missing — dataset/ is gitignored")

    report = pd.read_csv(SHIPPED_REPORT)
    counts, accs = report.n_bands.tolist(), report.spa_svc.tolist()

    assert find_elbow(counts, accs, cfg_bs) == 40
    verdict = verify_elbow(counts, accs, 40, cfg_bs)

    assert not verdict.demonstrable
    assert verdict.n_points_past_k == 0
    assert verdict.max_k_recorded == verdict.k == verdict.peak_k == 40
    assert "terminates at the chosen k" in verdict.reason
    # Monotone to the end: nothing in the curve even suggests a plateau.
    assert accs == sorted(accs)


def test_the_config_now_runs_the_curve_to_the_full_band_count(cfg_bs) -> None:
    """T4-6's fix, on the input side: rank and validate all 256 bands."""
    assert cfg_bs.n_select_max == 256
    assert max(cfg_bs.n_candidates) == 256
    assert 40 in cfg_bs.n_candidates
    assert sum(1 for k in cfg_bs.n_candidates if k > 40) >= 5


# ══════════════════════════════════════════════════════════════════════
#  verify_elbow
# ══════════════════════════════════════════════════════════════════════


def test_an_extended_flat_curve_demonstrates_its_elbow(cfg_bs) -> None:
    """The positive case: the curve goes past k and does not improve."""
    counts = [5, 10, 20, 40, 80, 160, 256]
    accs = [0.20, 0.35, 0.55, 0.80, 0.805, 0.807, 0.806]

    k = find_elbow(counts, accs, cfg_bs)
    verdict = verify_elbow(counts, accs, k, cfg_bs)

    assert k == 40
    assert verdict.demonstrable
    assert verdict.n_points_past_k == 3
    assert verdict.headroom_past_k == pytest.approx(0.007, abs=1e-9)


def test_a_curve_that_keeps_rising_is_reported_as_headroom(cfg_bs) -> None:
    """F-3's prediction, in the form the verdict would report it.

    If accuracy still climbs past 40, the elbow moves and the 40-band dataset
    is a choice that costs accuracy — which is exactly what F-3 says happens
    under the deployed estimator.
    """
    counts = [5, 10, 20, 40, 80, 160, 256]
    accs = [0.20, 0.35, 0.55, 0.60, 0.72, 0.83, 0.90]

    k = find_elbow(counts, accs, cfg_bs)
    verdict = verify_elbow(counts, accs, 40, cfg_bs)

    assert k == 256  # the elbow is nowhere near 40 on this curve
    assert not verdict.demonstrable
    assert verdict.headroom_past_k == pytest.approx(0.30, abs=1e-9)
    assert verdict.peak_k == 256


def test_the_verdict_reports_the_curve_it_saw(cfg_bs) -> None:
    counts = [10, 40, 100]
    accs = [0.5, 0.9, 0.91]
    verdict = verify_elbow(counts, accs, 40, cfg_bs)

    assert isinstance(verdict, ElbowVerdict)
    assert verdict.max_k_recorded == 100
    assert verdict.acc_at_k == pytest.approx(0.9)
    assert verdict.peak_acc == pytest.approx(0.91)
    assert any("k = 40" in line for line in verdict.lines())


def test_unordered_counts_are_handled(cfg_bs) -> None:
    """The decision curve may arrive from a CSV in any row order."""
    ordered = verify_elbow([10, 40, 100], [0.5, 0.9, 0.91], 40, cfg_bs)
    shuffled = verify_elbow([100, 10, 40], [0.91, 0.5, 0.9], 40, cfg_bs)
    assert ordered == shuffled


# ══════════════════════════════════════════════════════════════════════
#  The deployed estimator (F-3)
# ══════════════════════════════════════════════════════════════════════


def test_a_deployed_curve_is_read_by_band_count(tmp_path) -> None:
    """§4.2 T4-6: "re-run selection with the deployed estimator"."""
    path = tmp_path / "deployed.csv"
    pd.DataFrame({"n_bands": [40, 100, 256], "accuracy": [0.60, 0.75, 0.88]}).to_csv(
        path, index=False
    )
    assert load_deployed_curve(path) == {40: 0.60, 100: 0.75, 256: 0.88}


@pytest.mark.parametrize("column", ["accuracy", "acc", "macro_f1", "f1"])
def test_the_score_column_may_be_named_for_the_metric_it_is(tmp_path, column: str) -> None:
    """macro-F1 is this project's primary metric, not accuracy (§2 evaluate)."""
    path = tmp_path / f"{column}.csv"
    pd.DataFrame({"n_bands": [40], column: [0.5]}).to_csv(path, index=False)
    assert load_deployed_curve(path) == {40: 0.5}


def test_a_malformed_deployed_curve_is_rejected(tmp_path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame({"bands": [40], "score": [0.5]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="n_bands"):
        load_deployed_curve(path)


def test_a_missing_deployed_curve_is_rejected(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_deployed_curve(tmp_path / "nope.csv")


def test_the_deployed_curve_can_move_the_elbow(cfg_bs, tmp_path) -> None:
    """The two estimators need not agree, which is the whole of F-3.

    A linear model on 90 class-mean spectra saturates long before a four-branch
    network that reads spatial structure does, so an elbow measured on the
    proxy is not evidence about the deployed one.
    """
    proxy_counts = [5, 10, 20, 40, 80, 160, 256]
    proxy_accs = [0.20, 0.35, 0.45, 0.475, 0.478, 0.479, 0.479]

    path = tmp_path / "deployed.csv"
    pd.DataFrame(
        {"n_bands": proxy_counts, "macro_f1": [0.30, 0.50, 0.68, 0.78, 0.85, 0.89, 0.91]}
    ).to_csv(path, index=False)
    deployed = load_deployed_curve(path)

    proxy_k = find_elbow(proxy_counts, proxy_accs, cfg_bs)
    deployed_k = find_elbow(sorted(deployed), [deployed[k] for k in sorted(deployed)], cfg_bs)

    assert proxy_k == 40
    assert deployed_k > proxy_k
    assert cfg_bs.deployed_curve_path is None  # off unless a curve is supplied


def test_find_elbow_is_unchanged(cfg_bs) -> None:
    """T4-6 verifies the elbow; it does not redefine it.

    ``find_elbow`` still returns the smallest k reaching ``elbow_pct`` of the
    peak, so a re-run with the same curve selects the same bands and the
    difference is entirely in what gets *reported*.
    """
    counts = [5, 10, 20, 40]
    # Threshold is 0.98 x peak. 0.97 misses it, 0.99 clears it.
    assert find_elbow(counts, [0.2, 0.5, 0.97, 1.0], cfg_bs) == 40
    assert find_elbow(counts, [0.2, 0.5, 0.99, 1.0], cfg_bs) == 20
    assert find_elbow(counts, [0.2, 0.5, 0.90, 1.0], cfg_bs) == 40
    # Degenerate: nothing clears the threshold → the peak.
    assert find_elbow(counts, [np.nan, np.nan, np.nan, np.nan], cfg_bs) == 5
