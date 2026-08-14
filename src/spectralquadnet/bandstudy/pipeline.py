"""The stages, in the order they may legally run.

::

    prepare  ─ cache mean spectra; build and report every fold's splits
    select   ─ run every method on TRAIN rows only; write band sets + λ CSVs
    proxy    ─ fit every proxy at every budget; score on CALIB
    analyse  ─ trends, plateaus, stability, redundancy, flags, recommendation
    confirm  ─ score the recommendation on val ∪ test, once  (opt-in)
    neural   ─ emit/run the neural confirmation runs                (opt-in)
    report   ─ assemble everything into REPORT.md

The ordering is a data dependency, not a convention, and the module enforces
it: ``analyse`` refuses without proxy records, ``confirm`` refuses without a
recommendation, ``neural`` refuses without band files. The one ordering
constraint that is *not* a data dependency is the important one — ``confirm``
comes after ``analyse`` because the recommendation must exist before the
held-out split is touched. Running them the other way round would be the
project's original defect with extra steps.

Two selection scopes, doing different jobs
──────────────────────────────────────────
Each (fold, method) is selected twice over:

* once on the **whole** training split — the *canonical* selection, tagged
  ``rep="full"``. This is what a deployed pipeline would produce, it is what
  gets written to ``bands/`` for the neural stage, and its calib curve is the
  primary curve.
* once per **replicate**, on a stratified 80% subsample. These exist to answer
  "would a slightly different training sample have chosen the same bands, and
  scored the same?" — which the canonical selection cannot answer about itself.

``random`` is exempt from the replicates: its uncertainty is the spread across
draws, which it already has, and resampling a uniform random choice measures
the resampler.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from spectralquadnet.bandstudy import data as bsdata
from spectralquadnet.bandstudy import methods as bsmethods
from spectralquadnet.bandstudy import proxies as bsproxies
from spectralquadnet.bandstudy.config import BandStudyConfig, StageResult
from spectralquadnet.bandstudy.store import (
    RecordStore,
    check_or_write_manifest,
    console,
    progress,
    record_stage,
    summary_table,
)

_log = logging.getLogger("spectralquadnet.bandstudy.pipeline")

#: Tag for the selection made on the whole training split.
CANONICAL = "full"

#: Record key for the proxy stage. Every field is part of what identifies a
#: cell; drop one and a resume would skip a cell it has not run.
PROXY_KEY = ("fold", "method", "budget", "rep", "draw", "proxy", "split")

#: Record key for the confirm stage — same identity, different split.
CONFIRM_KEY = ("fold", "method", "budget", "rep", "draw", "proxy", "split")


# ══════════════════════════════════════════════════════════════════════
#  Shared loading
# ══════════════════════════════════════════════════════════════════════


@dataclass
class StudyInputs:
    """Everything the stages read, loaded once."""

    spectra: npt.NDArray[np.float32]
    features: npt.NDArray[np.float32]
    wavelengths: npt.NDArray[np.float64]
    labels: npt.NDArray[np.int64]
    folds: dict[int, bsdata.FoldData]
    n_bands: int
    n_classes: int

    def morphology_note(self) -> str:
        return f"{self.features.shape[1]} proxy features from {self.n_bands} bands"


def load_inputs(cfg: BandStudyConfig, quiet: bool = False) -> StudyInputs:
    """Cache/load features, build every fold, and validate the configuration.

    Raises:
        ValueError: The configuration has a design problem — see
            :meth:`BandStudyConfig.validate`. Raised *before* any compute,
            because every one of those problems produces a table that looks
            finished and answers the wrong question.
    """
    spectra, features = bsdata.extract_features(cfg)
    n_bands = int(spectra.shape[1])

    problems = cfg.validate(n_bands_available=n_bands)
    if problems:
        raise ValueError(
            "band study configuration is not runnable:\n  - " + "\n  - ".join(problems)
        )

    wavelengths = bsdata.load_wavelengths(cfg, n_bands)
    morph = bsdata.load_morphology(cfg, len(spectra))
    if morph is not None:
        features = np.concatenate([features, morph], axis=1)

    folds = {f: bsdata.fold_splits(cfg, f) for f in cfg.folds}
    labels = next(iter(folds.values())).labels
    n_classes = int(np.unique(labels).size)

    if not quiet:
        for fold in folds.values():
            _log.info("%s", fold.summary())
    return StudyInputs(
        spectra=np.asarray(spectra, dtype=np.float32),
        features=np.asarray(features, dtype=np.float32),
        wavelengths=wavelengths,
        labels=np.asarray(labels, dtype=np.int64),
        folds=folds,
        n_bands=n_bands,
        n_classes=n_classes,
    )


def replicate_tags(cfg: BandStudyConfig, method: str) -> list[str]:
    """Which selection scopes a method runs under.

    ``random`` gets the canonical scope only — see the module docstring.
    """
    if method == "random":
        return [CANONICAL]
    return [CANONICAL] + [str(r) for r in range(cfg.replicates)]


def _rows_for(cfg: BandStudyConfig, fold: bsdata.FoldData, rep: str) -> npt.NDArray[np.int64]:
    if rep == CANONICAL:
        return fold.train
    return bsdata.replicate_rows(
        fold.train, fold.labels, cfg.replicate_frac, seed=cfg.seed + 1000 * fold.fold + int(rep)
    )


def _bands_hash(bands: list[int]) -> str:
    return hashlib.sha1(",".join(str(b) for b in sorted(bands)).encode()).hexdigest()[:12]


# ══════════════════════════════════════════════════════════════════════
#  Stage: prepare
# ══════════════════════════════════════════════════════════════════════


def stage_prepare(cfg: BandStudyConfig) -> StageResult:
    """Cache the features and write every fold's split report.

    The split report is an artifact rather than a log line because it carries
    the study's central claim about itself: which rows the selectors saw. A
    reviewer's first question is "were the eval labels in scope when these
    bands were chosen?" and it must be answerable from a file.
    """
    started = time.perf_counter()
    check_or_write_manifest(cfg)
    inputs = load_inputs(cfg)

    payload = {
        "n_patches": int(len(inputs.labels)),
        "n_bands": inputs.n_bands,
        "n_classes": inputs.n_classes,
        "n_proxy_features": int(inputs.features.shape[1]),
        "wavelength_range_nm": [
            float(inputs.wavelengths.min()),
            float(inputs.wavelengths.max()),
        ],
        "features": cfg.features,
        "folds": {
            str(f): {
                "sizes": fold.sizes,
                "split_report": fold.report,
                "selection_scope": (
                    f"{fold.sizes['train']:,} TRAINING patches of fold {f} "
                    f"({cfg.split_scheme}); calib/val/test labels are out of scope"
                ),
            }
            for f, fold in inputs.folds.items()
        },
    }
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    (cfg.cache_dir / "splits.json").write_text(json.dumps(payload, indent=2))

    term = console()
    term.rule("[bold]prepare")
    term.print(
        summary_table(
            "Splits — what each partition is allowed to decide",
            ["fold", "train (selection)", "calib (decisions)", "held-out (confirm only)"],
            [
                [f, f"{fd.sizes['train']:,}", f"{fd.sizes['calib']:,}", f"{fd.sizes['heldout']:,}"]
                for f, fd in sorted(inputs.folds.items())
            ],
        )
    )
    term.print(
        f"[dim]{len(inputs.labels):,} patches · {inputs.n_bands} bands · "
        f"{inputs.n_classes} classes · {inputs.wavelengths.min():.0f}–"
        f"{inputs.wavelengths.max():.0f} nm · {inputs.morphology_note()}[/dim]"
    )

    result = StageResult(
        stage="prepare",
        n_done=len(inputs.folds),
        seconds=time.perf_counter() - started,
        notes=[
            f"features cached ({cfg.features}); splits written to {cfg.cache_dir/'splits.json'}"
        ],
    )
    record_stage(cfg, result)
    return result


# ══════════════════════════════════════════════════════════════════════
#  Stage: select
# ══════════════════════════════════════════════════════════════════════


def selection_path(cfg: BandStudyConfig, fold: int, rep: str, method: str) -> Path:
    return cfg.selections_dir / f"fold{fold}" / f"rep_{rep}" / f"{method}.json"


def load_selection(cfg: BandStudyConfig, fold: int, rep: str, method: str) -> dict[str, Any] | None:
    """One selection artifact, or ``None`` when it has not been produced."""
    path = selection_path(cfg, fold, rep, method)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        _log.warning("%s is corrupt — treating as missing", path)
        return None


def band_file(cfg: BandStudyConfig, fold: int, method: str, budget: int) -> Path:
    """Where the canonical index array for one (fold, method, k) lives.

    This is the file a training run points ``data.band_indices_path`` at, so
    the name has to be unambiguous about all three: a run that silently picked
    up another fold's bands would be exactly the leak this study exists to
    avoid, wearing the name of the fix.
    """
    return cfg.bands_dir / f"{method}_f{fold}_k{budget}.npy"


def wavelength_file(cfg: BandStudyConfig, fold: int, method: str, budget: int) -> Path:
    return cfg.bands_dir / f"{method}_f{fold}_k{budget}_wavelengths.csv"


def stage_select(cfg: BandStudyConfig) -> StageResult:
    """Run every method on every fold's training rows, and only those.

    Writes one JSON per (fold, replicate, method) holding the ranking, the
    per-budget sets, the timing and any failure; and, for the canonical
    selection only, one ``.npy`` index array plus a matching wavelength CSV per
    budget, which are the files the neural stage and any future training run
    consume.
    """
    started = time.perf_counter()
    check_or_write_manifest(cfg)
    inputs = load_inputs(cfg)
    budgets = sorted(cfg.budgets)
    term = console()
    term.rule("[bold]select")

    cells = [
        (fold, rep, method)
        for fold in sorted(cfg.folds)
        for method in cfg.methods
        for rep in replicate_tags(cfg, method)
    ]
    done = skipped = failed = 0
    failures: list[str] = []

    with progress(cfg, "selecting", len(cells)) as advance:
        # Group by (fold, rep) so the SelectionContext — and with it the one
        # expensive mutual-information estimate — is built once per training
        # matrix rather than once per method.
        for fold_id in sorted(cfg.folds):
            fold = inputs.folds[fold_id]
            for rep in [CANONICAL] + [str(r) for r in range(cfg.replicates)]:
                wanted = [m for m in cfg.methods if rep in replicate_tags(cfg, m)]
                pending = [
                    m for m in wanted if cfg.force or load_selection(cfg, fold_id, rep, m) is None
                ]
                skipped += len(wanted) - len(pending)
                advance(len(wanted) - len(pending))
                if not pending:
                    continue

                rows = _rows_for(cfg, fold, rep)
                ctx = bsmethods.SelectionContext(
                    x=np.asarray(inputs.spectra[rows], dtype=np.float64),
                    y=inputs.labels[rows],
                    seed=cfg.seed + 1000 * fold_id + (0 if rep == CANONICAL else int(rep) + 1),
                    corr_threshold=cfg.corr_threshold,
                    mi_neighbors=cfg.mi_neighbors,
                )

                for method in pending:
                    if cfg.dry_run:
                        _log.info(
                            "[dry-run] select fold=%d rep=%s method=%s (%d rows)",
                            fold_id,
                            rep,
                            method,
                            len(rows),
                        )
                        advance()
                        continue
                    outcome = bsmethods.run_method(method, ctx, budgets, draws=cfg.random_draws)
                    payload = {
                        **outcome.as_dict(),
                        "fold": fold_id,
                        "rep": rep,
                        "n_selection_rows": int(len(rows)),
                        "selection_scope": (
                            f"{len(rows):,} training rows of fold {fold_id}"
                            f"{'' if rep == CANONICAL else f' (replicate {rep}, {cfg.replicate_frac:.0%} stratified subsample)'}"
                            "; calib/val/test labels out of scope"
                        ),
                        "shared_timings": {k: round(v, 3) for k, v in ctx.timings.items()},
                        "spec": {
                            "kind": bsmethods.get(method).kind,
                            "supervised": bsmethods.get(method).supervised,
                            "family": bsmethods.get(method).family,
                        },
                    }
                    path = selection_path(cfg, fold_id, rep, method)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(payload, indent=2))

                    if outcome.ok:
                        done += 1
                        if rep == CANONICAL:
                            _write_band_files(cfg, inputs, fold_id, method, outcome)
                    else:
                        failed += 1
                        failures.append(f"fold {fold_id} rep {rep} {method}: {outcome.failure}")
                        _log.error("selection failed — %s", failures[-1])
                    advance()

    term.print(
        f"[green]{done}[/green] selections written, [yellow]{skipped}[/yellow] resumed, "
        f"[red]{failed}[/red] failed"
    )
    if failed:
        for line in failures[:10]:
            term.print(f"  [red]FAILED[/red] {line}")

    result = StageResult(
        stage="select",
        n_done=done,
        n_skipped=skipped,
        n_failed=failed,
        seconds=time.perf_counter() - started,
        notes=failures,
    )
    record_stage(cfg, result)
    return result


def _write_band_files(
    cfg: BandStudyConfig,
    inputs: StudyInputs,
    fold: int,
    method: str,
    outcome: bsmethods.SelectionOutcome,
) -> None:
    """Persist the canonical band sets as ``.npy`` + wavelength CSV.

    The CSV is written beside the indices and not derived on demand because a
    training run needs a k-row wavelength file to build its λ-aware operators,
    and a run that paired k bands with the 256-row CSV would produce a model
    whose wavelength vector describes bands its input does not contain.
    """
    cfg.bands_dir.mkdir(parents=True, exist_ok=True)
    for budget, sets in outcome.per_budget.items():
        # The canonical scope has exactly one set per budget for every method
        # except `random`, whose draws are a null distribution rather than a
        # recommendation — the first draw is written so the file exists and is
        # named, but nothing recommends it.
        bands = np.asarray(sorted(sets[0]), dtype=np.int64)
        np.save(band_file(cfg, fold, method, budget), bands)
        wl = inputs.wavelengths[bands]
        lines = ["index,Wavelength (nm)"]
        lines += [f"{int(b)},{float(v):.6f}" for b, v in zip(bands, wl, strict=True)]
        wavelength_file(cfg, fold, method, budget).write_text("\n".join(lines) + "\n")


# ══════════════════════════════════════════════════════════════════════
#  Stage: proxy
# ══════════════════════════════════════════════════════════════════════


def stage_proxy(cfg: BandStudyConfig) -> StageResult:
    """Fit every proxy at every band set and score it on **calib**.

    Calib and nothing else. It is held out from the rows the selector saw, so
    the score is not the score of the data the bands were chosen on; and it is
    not the reported split, so reading it as many times as this stage does
    costs no held-out evidence. That is the whole reason
    ``configs/data/hsi256_grouped.yaml`` carves it.
    """
    started = time.perf_counter()
    check_or_write_manifest(cfg)
    inputs = load_inputs(cfg)
    term = console()
    term.rule("[bold]proxy")

    cells = _proxy_cells(cfg, inputs)
    store = RecordStore(cfg.proxy_dir, PROXY_KEY, force=cfg.force)
    todo = [c for c in cells if not store.has(**{k: c[k] for k in PROXY_KEY})]
    skipped = len(cells) - len(todo)
    term.print(
        f"{len(cells):,} cells — [yellow]{skipped:,}[/yellow] already recorded, "
        f"[bold]{len(todo):,}[/bold] to run"
    )
    if cfg.dry_run:
        store.close()
        for cell in todo[:20]:
            term.print(
                f"  [dry-run] {cell['fold']} {cell['method']} k={cell['budget']} "
                f"rep={cell['rep']} draw={cell['draw']} {cell['proxy']}"
            )
        if len(todo) > 20:
            term.print(f"  … and {len(todo) - 20:,} more")
        return StageResult(stage="proxy", n_skipped=skipped, seconds=time.perf_counter() - started)

    done = failed = 0
    # Two identical band sets at one (fold, rep, proxy) are one fit. Every
    # method returns all 256 bands at k = 256, so without this the largest and
    # slowest budget is refitted once per method for no new information.
    cache: dict[tuple[Any, ...], bsproxies.ProxyScore] = {}

    with store, progress(cfg, "proxy fits", len(todo)) as advance:
        for cell in todo:
            fold = inputs.folds[cell["fold"]]
            rows = _rows_for(cfg, fold, cell["rep"])
            columns = bsdata.feature_columns(
                np.asarray(cell["bands"], dtype=np.int64), inputs.n_bands, cfg.features
            )
            cache_key = (cell["fold"], cell["rep"], cell["proxy"], cell["bands_hash"])
            score = cache.get(cache_key)
            if score is None:
                score = bsproxies.fit_and_score(
                    cell["proxy"],
                    inputs.features[np.ix_(rows, columns)],
                    inputs.labels[rows],
                    inputs.features[np.ix_(fold.calib, columns)],
                    inputs.labels[fold.calib],
                    num_classes=inputs.n_classes,
                    split="calib",
                    seed=cfg.seed,
                    n_boot=0,
                )
                cache[cache_key] = score
            store.append(
                {
                    **{k: cell[k] for k in PROXY_KEY},
                    "bands_hash": cell["bands_hash"],
                    "n_bands": len(cell["bands"]),
                    "family": bsmethods.get(cell["method"]).family,
                    "supervised": bsmethods.get(cell["method"]).supervised,
                    "selection_seconds": cell["selection_seconds"],
                    **score.as_dict(),
                }
            )
            done += 1
            failed += 0 if score.ok else 1
            advance()

    term.print(f"[green]{done:,}[/green] fits recorded, [red]{failed}[/red] failed")
    result = StageResult(
        stage="proxy",
        n_done=done,
        n_skipped=skipped,
        n_failed=failed,
        seconds=time.perf_counter() - started,
        notes=[f"records → {store.path}"],
    )
    record_stage(cfg, result)
    return result


def _proxy_cells(cfg: BandStudyConfig, inputs: StudyInputs) -> list[dict[str, Any]]:
    """Expand the grid into concrete cells, reading the band sets off disk.

    Raises:
        FileNotFoundError: No selections exist. The message says which stage
            writes them rather than leaving an empty grid to be interpreted as
            "nothing to do".
    """
    cells: list[dict[str, Any]] = []
    missing: list[str] = []
    for fold in sorted(cfg.folds):
        for method in cfg.methods:
            for rep in replicate_tags(cfg, method):
                payload = load_selection(cfg, fold, rep, method)
                if payload is None:
                    missing.append(f"fold {fold} rep {rep} {method}")
                    continue
                if payload.get("failure"):
                    continue
                seconds = float(payload.get("seconds", 0.0))
                for budget_str, sets in (payload.get("per_budget") or {}).items():
                    budget = int(budget_str)
                    if budget not in cfg.budgets:
                        continue
                    for draw, bands in enumerate(sets):
                        for proxy in cfg.proxies:
                            cells.append(
                                {
                                    "fold": fold,
                                    "method": method,
                                    "budget": budget,
                                    "rep": rep,
                                    "draw": draw,
                                    "proxy": proxy,
                                    "split": "calib",
                                    "bands": [int(b) for b in bands],
                                    "bands_hash": _bands_hash([int(b) for b in bands]),
                                    "selection_seconds": round(seconds, 4),
                                }
                            )
    if not cells:
        raise FileNotFoundError(
            "no band selections found. Run `python -m spectralquadnet.bandstudy.cli select` "
            f"first — it writes them under {cfg.selections_dir}."
        )
    if missing:
        _log.warning(
            "%d (fold, rep, method) selections are missing and were skipped: %s%s",
            len(missing),
            ", ".join(missing[:5]),
            " …" if len(missing) > 5 else "",
        )
    return cells


# ══════════════════════════════════════════════════════════════════════
#  Stage: confirm
# ══════════════════════════════════════════════════════════════════════


def stage_confirm(cfg: BandStudyConfig, extra: list[dict[str, Any]] | None = None) -> StageResult:
    """Score the **already-chosen** configurations on ``val ∪ test``, once.

    This is the only stage that touches the held-out split, it runs after the
    recommendation exists, and it scores a fixed list rather than a grid — the
    difference between confirming a choice and choosing on the confirmation
    set. Each cell carries a bootstrap interval, because here there is one
    number per configuration and the within-split sampling noise is exactly
    what a reader needs to not over-read it.

    Args:
        extra: Additional ``{"fold", "method", "budget"}`` configurations to
            score beside the recommendation — used to carry the repository's
            incumbents (40-band SPA, 100-band mRMR) into the same table.

    Raises:
        FileNotFoundError: No recommendation exists yet.
    """
    started = time.perf_counter()
    check_or_write_manifest(cfg)
    rec_path = cfg.analysis_dir / "recommendation.json"
    if not rec_path.exists():
        raise FileNotFoundError(
            f"{rec_path} does not exist, so there is nothing to confirm and no reason to "
            "look at the held-out split yet. Run `analyse` first — that ordering is the "
            "point of having a confirm stage at all."
        )
    recommendation = json.loads(rec_path.read_text())
    inputs = load_inputs(cfg)
    term = console()
    term.rule("[bold]confirm — the held-out split is scored here, once")

    configs = _confirm_configs(cfg, recommendation, extra)
    store = RecordStore(cfg.confirm_dir, CONFIRM_KEY, force=cfg.force)
    cells = [
        {**c, "proxy": proxy, "rep": CANONICAL, "draw": 0, "split": "val_test"}
        for c in configs
        for proxy in cfg.proxies
    ]
    todo = [c for c in cells if not store.has(**{k: c[k] for k in CONFIRM_KEY})]
    skipped = len(cells) - len(todo)

    term.print(
        f"{len(configs)} configurations × {len(cfg.proxies)} proxies = {len(cells)} cells "
        f"([yellow]{skipped}[/yellow] recorded, [bold]{len(todo)}[/bold] to run)"
    )
    if cfg.dry_run:
        store.close()
        for cell in todo:
            term.print(
                f"  [dry-run] fold {cell['fold']} {cell['method']} k={cell['budget']} "
                f"{cell['proxy']} → val∪test"
            )
        return StageResult(
            stage="confirm", n_skipped=skipped, seconds=time.perf_counter() - started
        )

    done = failed = 0
    with store, progress(cfg, "confirm fits", len(todo)) as advance:
        for cell in todo:
            fold = inputs.folds[cell["fold"]]
            path = band_file(cfg, cell["fold"], cell["method"], cell["budget"])
            if not path.exists():
                _log.error("no band file %s — skipping", path)
                failed += 1
                advance()
                continue
            bands = np.asarray(np.load(path), dtype=np.int64)
            columns = bsdata.feature_columns(bands, inputs.n_bands, cfg.features)
            heldout = fold.reveal_heldout(
                f"confirm stage scoring {cell['method']} k={cell['budget']} "
                f"({cell.get('why', 'recommended')})"
            )
            score = bsproxies.fit_and_score(
                cell["proxy"],
                inputs.features[np.ix_(fold.train, columns)],
                inputs.labels[fold.train],
                inputs.features[np.ix_(heldout, columns)],
                inputs.labels[heldout],
                num_classes=inputs.n_classes,
                split="val_test",
                seed=cfg.seed,
                n_boot=cfg.n_boot,
            )
            store.append(
                {
                    **{k: cell[k] for k in CONFIRM_KEY},
                    "why": cell.get("why", ""),
                    "n_bands": int(len(bands)),
                    "bands": [int(b) for b in bands],
                    "wavelengths_nm": [round(float(v), 2) for v in inputs.wavelengths[bands]],
                    **score.as_dict(),
                }
            )
            done += 1
            failed += 0 if score.ok else 1
            advance()

    result = StageResult(
        stage="confirm",
        n_done=done,
        n_skipped=skipped,
        n_failed=failed,
        seconds=time.perf_counter() - started,
        notes=["held-out split scored — every call is logged in logs/confirm_*.log"],
    )
    record_stage(cfg, result)
    return result


def _confirm_configs(
    cfg: BandStudyConfig, recommendation: dict[str, Any], extra: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    """The fixed list the confirm stage scores.

    Always includes the full band count under the ``uniform`` label (at k = C
    every method selects every band, so the label is a formality and the row is
    the reference every reduction is a reduction *from*), plus whatever the
    analysis recommended, plus the repository's two incumbents where their band
    files exist — so the table answers "is the recommendation better than what
    we shipped?" rather than only "how good is the recommendation?".
    """
    wanted: list[dict[str, Any]] = []
    seen: set[tuple[int, str, int]] = set()

    def add(fold: int, method: str, budget: int, why: str) -> None:
        key = (fold, method, budget)
        if key in seen:
            return
        if not band_file(cfg, fold, method, budget).exists():
            _log.warning(
                "confirm: no band file for fold %d %s k=%d — skipped", fold, method, budget
            )
            return
        seen.add(key)
        wanted.append({"fold": fold, "method": method, "budget": budget, "why": why})

    full = max(cfg.budgets)
    for fold in sorted(cfg.folds):
        add(fold, "uniform", full, "reference: the full band set")

    for item in recommendation.get("confirm_list", []):
        add(
            int(item["fold"]),
            str(item["method"]),
            int(item["budget"]),
            str(item.get("why", "recommended")),
        )

    for item in extra or []:
        add(
            int(item["fold"]),
            str(item["method"]),
            int(item["budget"]),
            str(item.get("why", "incumbent")),
        )

    # The repository's shipped choices, so the comparison is against what
    # exists rather than against nothing.
    for fold in sorted(cfg.folds):
        add(fold, "spa", 40, "incumbent: the shipped 40-band SPA subset")
        add(fold, "mrmr", 100, "incumbent: the per-fold 100-band mRMR subset")
    return wanted
