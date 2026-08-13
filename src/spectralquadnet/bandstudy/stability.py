"""Is a selection reproducible, and is the set it returns redundant?

A band subset has two properties a score cannot express, and both change what
the subset means.

**Stability.** Re-run a selector on a different sample of the same training
split and it may return a different set. If two replicates of a k = 20
selection overlap in four bands, the method has not identified twenty
informative bands — it has found twenty of a much larger interchangeable pool,
and the specific twenty it named are an accident of the sample. That is not
necessarily a defect (correlated bands *are* interchangeable), but it is fatal
to any claim of the form "these wavelengths matter", and a study that reports
the score without the stability licenses exactly that claim.

Chance correction matters here. Two random k-subsets of 256 bands already share
``k²/256`` bands on average, so raw overlap at k = 128 is ~0.5 before any method
has done anything. :func:`kuncheva_index` subtracts that expectation; the raw
Jaccard is reported beside it because it is the quantity people recognise.

**Redundancy.** Twenty bands drawn from one 30 nm window carry less information
than twenty spread across the range, however well each scores alone.
:func:`redundancy` measures it three ways — mean absolute correlation, the
participation-ratio effective rank, and spectral coverage — because they
disagree in informative ways: a set can have low mean correlation and still
cover a quarter of the spectrum.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import numpy.typing as npt

# ══════════════════════════════════════════════════════════════════════
#  Set agreement
# ══════════════════════════════════════════════════════════════════════


def jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    """``|A ∩ B| / |A ∪ B|``. 1.0 for identical sets, 0.0 for disjoint ones."""
    sa, sb = set(a), set(b)
    union = sa | sb
    if not union:
        return 1.0
    return len(sa & sb) / len(union)


def kuncheva_index(a: Iterable[int], b: Iterable[int], n_total: int) -> float:
    """Chance-corrected agreement between two equal-size subsets.

    ``I = (r - k²/n) / (k - k²/n)`` where ``r`` is the observed overlap, ``k``
    the subset size and ``n`` the pool size (Kuncheva, Proc. IASTED AIA 2007).
    It is 1.0 for identical sets, 0.0 for the overlap two random subsets would
    have, and negative for sets that agree less than chance.

    Undefined at ``k == n`` — every subset is the whole pool, so there is
    nothing for a method to agree or disagree about — and returns ``nan`` there
    rather than a misleading 1.0.

    Args:
        n_total: Size of the pool the subsets were drawn from, i.e. the band
            count of the full cube.
    """
    sa, sb = set(a), set(b)
    k = len(sa)
    if len(sb) != k:
        # Only reachable if a method returned a short set; Jaccard still means
        # something there, this does not.
        return float("nan")
    if k == 0 or k >= n_total:
        return float("nan")
    expected = k * k / n_total
    denominator = k - expected
    if abs(denominator) < 1e-12:
        return float("nan")
    return (len(sa & sb) - expected) / denominator


@dataclass(frozen=True)
class StabilityReport:
    """Pairwise agreement over a group of selections at one budget."""

    n_sets: int
    n_pairs: int
    mean_jaccard: float
    min_jaccard: float
    mean_kuncheva: float
    #: Bands present in **every** set — the ones a "these wavelengths matter"
    #: claim can actually be made about.
    consensus: list[int]
    #: Bands present in at least one set. Its size against ``k`` is the size of
    #: the interchangeable pool the method is drawing from.
    union_size: int
    budget: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "budget": self.budget,
            "n_sets": self.n_sets,
            "n_pairs": self.n_pairs,
            "mean_jaccard": self.mean_jaccard,
            "min_jaccard": self.min_jaccard,
            "mean_kuncheva": self.mean_kuncheva,
            "consensus": list(self.consensus),
            "n_consensus": len(self.consensus),
            "union_size": self.union_size,
        }


def stability(sets: Sequence[Sequence[int]], n_total: int, budget: int) -> StabilityReport:
    """Pairwise Jaccard and Kuncheva over ``sets``, plus their consensus.

    A single set yields a report with ``n_pairs=0`` and ``nan`` agreements
    rather than a fabricated 1.0: one selection cannot disagree with itself,
    and reporting perfect stability for ``replicates=1`` would be the most
    misleading number in the study.
    """
    materialised = [list(s) for s in sets]
    if len(materialised) < 2:
        return StabilityReport(
            n_sets=len(materialised),
            n_pairs=0,
            mean_jaccard=float("nan"),
            min_jaccard=float("nan"),
            mean_kuncheva=float("nan"),
            consensus=sorted(materialised[0]) if materialised else [],
            union_size=len(set(materialised[0])) if materialised else 0,
            budget=budget,
        )

    jaccards = [jaccard(a, b) for a, b in combinations(materialised, 2)]
    kunchevas = [kuncheva_index(a, b, n_total) for a, b in combinations(materialised, 2)]
    finite = [v for v in kunchevas if np.isfinite(v)]

    consensus = set(materialised[0])
    union: set[int] = set()
    for s in materialised:
        consensus &= set(s)
        union |= set(s)

    return StabilityReport(
        n_sets=len(materialised),
        n_pairs=len(jaccards),
        mean_jaccard=float(np.mean(jaccards)),
        min_jaccard=float(np.min(jaccards)),
        mean_kuncheva=float(np.mean(finite)) if finite else float("nan"),
        consensus=sorted(consensus),
        union_size=len(union),
        budget=budget,
    )


# ══════════════════════════════════════════════════════════════════════
#  Redundancy and coverage
# ══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class RedundancyReport:
    """How much a selected set repeats itself, and how much spectrum it covers."""

    budget: int
    #: Mean ``|r|`` over distinct band pairs in the set. 1.0 would be k copies
    #: of one band.
    mean_abs_corr: float
    max_abs_corr: float
    #: ``(sum λ)² / sum λ²`` over the selection's correlation-matrix
    #: eigenvalues — the participation ratio. Reads as "how many genuinely
    #: independent directions is this set worth", and is the number that
    #: distinguishes 40 informative bands from 40 copies of 6.
    effective_rank: float
    #: Effective rank divided by the budget. 1.0 = mutually orthogonal.
    rank_efficiency: float
    #: Span of the selection in nm, over the span of the full cube.
    wavelength_coverage: float
    wavelength_min: float
    wavelength_max: float
    #: Largest gap between consecutive selected wavelengths, in nm. A large gap
    #: with high coverage means the set is two clusters at the ends.
    largest_gap_nm: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "budget": self.budget,
            "mean_abs_corr": self.mean_abs_corr,
            "max_abs_corr": self.max_abs_corr,
            "effective_rank": self.effective_rank,
            "rank_efficiency": self.rank_efficiency,
            "wavelength_coverage": self.wavelength_coverage,
            "wavelength_min": self.wavelength_min,
            "wavelength_max": self.wavelength_max,
            "largest_gap_nm": self.largest_gap_nm,
        }


def redundancy(
    bands: Sequence[int],
    corr: npt.NDArray[Any],
    wavelengths: npt.NDArray[Any],
) -> RedundancyReport:
    """Redundancy and spectral coverage of one selected set.

    Args:
        corr: ``(C, C)`` band correlation matrix computed on the **training**
            rows the selection saw. Computing it on all rows would put held-out
            spectra into a reported diagnostic; it would barely move the number
            and would make the diagnostic inadmissible.
        wavelengths: ``(C,)`` nm per stored band.
    """
    idx = np.asarray(sorted(set(int(b) for b in bands)), dtype=np.int64)
    k = len(idx)
    wl = np.asarray(wavelengths, dtype=np.float64)
    full_span = float(wl.max() - wl.min()) or 1.0
    selected_wl = np.sort(wl[idx])

    if k < 2:
        return RedundancyReport(
            budget=k,
            mean_abs_corr=0.0,
            max_abs_corr=0.0,
            effective_rank=float(k),
            rank_efficiency=1.0 if k else 0.0,
            wavelength_coverage=0.0,
            wavelength_min=float(selected_wl[0]) if k else float("nan"),
            wavelength_max=float(selected_wl[0]) if k else float("nan"),
            largest_gap_nm=0.0,
        )

    block = np.abs(np.asarray(corr, dtype=np.float64)[np.ix_(idx, idx)])
    off = block[~np.eye(k, dtype=bool)]

    eigenvalues = np.linalg.eigvalsh(np.asarray(corr, dtype=np.float64)[np.ix_(idx, idx)])
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = float(eigenvalues.sum())
    eff_rank = (total**2) / float((eigenvalues**2).sum()) if total > 0 else float("nan")

    return RedundancyReport(
        budget=k,
        mean_abs_corr=float(off.mean()),
        max_abs_corr=float(off.max()),
        effective_rank=float(eff_rank),
        rank_efficiency=float(eff_rank / k) if np.isfinite(eff_rank) else float("nan"),
        wavelength_coverage=float((selected_wl[-1] - selected_wl[0]) / full_span),
        wavelength_min=float(selected_wl[0]),
        wavelength_max=float(selected_wl[-1]),
        largest_gap_nm=float(np.diff(selected_wl).max()),
    )


def selection_frequency(sets: Iterable[Sequence[int]], n_total: int) -> npt.NDArray[np.float64]:
    """How often each band appears across a collection of selections.

    The basis of the "consistently useful wavelengths" answer: a band selected
    by nine methods across two folds and five replicates is evidence about rice;
    one selected by a single method once is evidence about that method.
    """
    counts = np.zeros(int(n_total), dtype=np.float64)
    n = 0
    for s in sets:
        n += 1
        counts[np.asarray(list(s), dtype=np.int64)] += 1.0
    return counts / max(n, 1)


def contiguous_regions(
    bands: Sequence[int], wavelengths: npt.NDArray[Any], max_gap_nm: float = 15.0
) -> list[tuple[float, float, int]]:
    """Group selected bands into contiguous wavelength regions.

    Reporting 40 individual wavelengths is unreadable and, worse, misleading:
    adjacent bands 2.4 nm apart are one feature, not two. This collapses them
    into ``(start_nm, end_nm, n_bands)`` runs, which is the form a paper's
    "the selected regions were …" sentence needs.

    Args:
        max_gap_nm: Bands further apart than this start a new region. 15 nm is
            about six of this cube's steps — wide enough to bridge a single
            skipped band, narrow enough not to merge distinct features.
    """
    wl = np.sort(
        np.asarray(wavelengths, dtype=np.float64)[np.asarray(sorted(set(bands)), dtype=int)]
    )
    if len(wl) == 0:
        return []
    regions: list[tuple[float, float, int]] = []
    start = wl[0]
    previous = wl[0]
    count = 1
    for value in wl[1:]:
        if value - previous > max_gap_nm:
            regions.append((float(start), float(previous), count))
            start, count = value, 0
        previous = value
        count += 1
    regions.append((float(start), float(previous), count))
    return regions
