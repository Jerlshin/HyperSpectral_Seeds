"""S16 shared paths, loaders, the cell index and a hierarchical bootstrap. Run every script from the repository root:

    python docs/research/evidence/S16_replication_reading/code/<script>.py

Reads the S15 cells (``outputs/experiments_u430k32/s15/{Y3,Y5}``), the S13 seed-0 Y3 cells they complete
(``s13/Y3``) and the X1 references (``s11/X1``). Reuses S14's loaders (which reuse S12's and S09's row-aligned
metadata: labels, sessions, the 17 cross-session varieties, the pipeline's own split builders) and S14's integrity
helpers, rather than re-deriving anything. Small tables a claim rests on go to
``docs/research/evidence/S16_replication_reading/``.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[5]
S14_CODE = REPO / "docs/research/evidence/S14_screen_reading/code"
sys.path.insert(0, str(S14_CODE))
from s14common import (  # noqa: E402  (S14's loaders → S12's → S09's metadata and src/)
    _CELL_RE,
    CROSS,
    LABELS,
    N_CLASSES,
    SAME,
    SESSION,
    SWEEP,
    Cell,
    conf,
    f1_from_conf,
    grouped_rows,
    jsonl,
    load_logits,
    load_preds,
    recall_from_conf,
    recalls,
    run_json,
    s11_cells,
    s13_cells,
    scalars_by_epoch,
    softmax,
    train_session_by_class,
)


def _load(name: str, path: Path):
    """Import an earlier study's script under a distinct module name (its file names repeat here)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


S14_EXTRACT = _load("s14_extract", S14_CODE / "extract_cells.py")   # R1, DEFAULTS, CURVE_KEYS, heldout_block
S14_LEAN = _load("s14_lean", S14_CODE / "lean.py")                  # ece

S15 = SWEEP / "s15"
EVID = REPO / "docs" / "research" / "evidence" / "S16_replication_reading"
S14_EVID = REPO / "docs" / "research" / "evidence" / "S14_screen_reading"
S12_EVID = REPO / "docs" / "research" / "evidence" / "S12_frozen_arms_reading"
CACHE = REPO / "outputs" / "s16_reading"
CACHE.mkdir(parents=True, exist_ok=True)

S15_COMMIT = "52fba4f4698e0a8948b00a4a83f3893ae793e55b"
S13_COMMIT = "aed5257d71603edf4021e7763e325dd146c84170"
CODE_DIGEST = "fade41e5764c4d3313dcc8e9ab9e270f3ccfda591ea81423bd2ead42a650462f"   # training code at aed5257 (F82)
ENVIRONMENT = dict(torch="2.10.0+cu128", gpu="Tesla T4", world_size=2)          # S11/S13's runtime (guard 3)
PREREG = {
    "S12": REPO / "docs/research/evidence/S12_frozen_arms_reading/preregistration_s12.json",
    "S13": REPO / "docs/research/evidence/S13_representation_screening/preregistration_s13.json",
    "S14": REPO / "docs/research/evidence/S14_screen_reading/preregistration_s14.json",
}
PREREG_SHA = {
    "S12": "88b377c5bc32f31a9ccb95356a88eb0be35e8cb51216b035c88f1761919ae7f4",
    "S13": "ef5982131df48c2ba8105c751fcf21f527a01ea2c0addf4ad59380f1a0989460",
    "S14": "9e182670755e13a6ead29a761123841094a9b9fb6fae18392553893d65c1f3da",
}

# preregistration_s12.json → reference (X1, all seeds) and preregistration_s14.json → reference (fresh seeds, the
# dissection reference). Read, never re-derived; extract_cells.py checks the fresh-seed values against the X1 cells.
REF = dict(f1=0.530786, f1_sd=0.009933, same=0.648493, same_sd=0.013873, cross=0.146637, cross_sd=0.009823,
           attraction=0.463958, attraction_sd=0.020362, strat_f1=0.726984, strat_f1_sd=0.005806)
FRESH_REF = dict(f1=0.528496, cross=0.145363, attraction=0.467445, strat_f1=0.728593)
THRESH = dict(H21a=0.5108, H21b=0.7090, H21c_delta=0.020, H21d_cross=0.1666, H21d_attraction=0.4240,
              H21e_delta=0.018, H22=0.5508)
INTENT = {  # each arm's intent, stated independently of the runner (preregistration_s14.json → arms.*.change)
    "lean_grouped": dict(spectral_descriptor="snv_morph", spatial_tail_strides=[2, 2, 2, 1], cbam_min_hw=3),
    "lean_stratified": dict(spectral_descriptor="snv_morph", spatial_tail_strides=[2, 2, 2, 1], cbam_min_hw=3),
    "desc_only": dict(spectral_descriptor="snv_morph"),
    "spatial_repair": dict(spatial_tail_strides=[2, 2, 2, 1], cbam_min_hw=3),
}
FROZEN_SEED = {"Y3": (1, 2), "Y5": (0,)}
EXPECTED_PARAMS = {"lean_grouped": 2_725_700, "lean_stratified": 2_725_700, "desc_only": 2_808_230,
                   "spatial_repair": 2_766_948}

__all__ = [
    "CROSS", "LABELS", "N_CLASSES", "SAME", "SESSION", "SWEEP", "Cell", "conf", "f1_from_conf", "grouped_rows",
    "jsonl", "load_logits", "load_preds", "recall_from_conf", "recalls", "run_json", "s11_cells", "s13_cells",
    "scalars_by_epoch", "softmax", "train_session_by_class", "S14_EXTRACT", "S14_LEAN", "S15", "EVID", "S14_EVID",
    "S12_EVID", "CACHE", "S15_COMMIT", "S13_COMMIT", "CODE_DIGEST", "ENVIRONMENT", "REF", "FRESH_REF", "THRESH",
    "INTENT", "FROZEN_SEED", "EXPECTED_PARAMS", "s15_cells", "y3_cells", "x1_cells", "protocol",
    "verify_preregistrations", "save", "Scored", "hboot", "REPO",
]


def s15_cells() -> list[Cell]:
    out = []
    for arm in ("Y3", "Y5"):
        for d in sorted((S15 / arm).iterdir()):
            m = _CELL_RE.fullmatch(d.name)
            if m:
                out.append(Cell(arm, m.group(1), int(m.group(2)), int(m.group(3)), d))
    return out


def y3_cells(seeds: tuple[int, ...] = (0, 1, 2), variant: str = "lean_grouped") -> list[Cell]:
    """Y3 across studies: seed 0 from S13, seeds 1–2 from S15."""
    pool = [c for c in s13_cells() if c.arm == "Y3"] + [c for c in s15_cells() if c.arm == "Y3"]
    return sorted((c for c in pool if c.variant == variant and c.seed in seeds), key=lambda c: (c.fold, c.seed))


def x1_cells(seeds: tuple[int, ...] = (0, 1, 2), variant: str = "grouped") -> list[Cell]:
    return sorted((c for c in s11_cells(("X1",)) if c.variant == variant and c.seed in seeds),
                  key=lambda c: (c.fold, c.seed))


def protocol(c: Cell) -> str:
    return "stratified" if "stratified" in c.variant else "grouped"


def verify_preregistrations() -> None:
    """Refuse to aggregate a held-out row if any frozen file moved (WORKFLOW G1)."""
    for k, p in PREREG.items():
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        assert h == PREREG_SHA[k], f"{k} pre-registration hash moved: {h}"


def save(obj: pd.DataFrame | dict | list, name: str) -> None:
    EVID.mkdir(parents=True, exist_ok=True)
    p = EVID / name
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(p, index=False, float_format="%.6g")
    else:
        p.write_text(json.dumps(obj, indent=1, default=float))
    print(f"  wrote {p.relative_to(REPO)}")


class Scored:
    """Held-out predictions of one run with every statistic the reading needs, recomputable on a kernel resample."""

    def __init__(self, c: Cell, variant: str = "tta"):
        self.cell = c
        self.rows, self.p, self.t = load_preds(c, variant)
        self.grouped = protocol(c) == "grouped"
        if self.grouped:
            tr, _, _ = grouped_rows(c.fold)
            soc = train_session_by_class(tr)
            self.is_cross = np.isin(self.t, CROSS)
            self.attracted = soc[self.p] == SESSION[self.rows]

    def stats(self, idx: np.ndarray | None = None) -> dict[str, float]:
        i = slice(None) if idx is None else idx
        cm = conf(self.t[i], self.p[i])
        out = {"f1": f1_from_conf(cm), "same": recall_from_conf(cm, SAME if self.grouped else np.arange(N_CLASSES))}
        if self.grouped:
            out["cross"] = recall_from_conf(cm, CROSS)
            out["attraction"] = float(self.attracted[i][self.is_cross[i]].mean())
        return out


def _by_fold(runs: list[Scored]) -> dict[int, list[Scored]]:
    out: dict[int, list[Scored]] = {}
    for r in runs:
        out.setdefault(r.cell.fold, []).append(r)
    for f, rs in out.items():
        for r in rs[1:]:
            assert np.array_equal(r.rows, rs[0].rows), f"{r.cell.name}: rows differ within fold {f}"
    return out


def hboot(a: list[Scored], b: list[Scored] | None = None, n: int = 2000, seed: int = 16) -> dict[str, dict]:
    """Mean over runs of each statistic (arm a, minus arm b if given), with a hierarchical bootstrap as S12's
    ``hypotheses.py``: runs resampled with replacement within each fold and arm, held-out kernels resampled with
    replacement within each fold and shared by both arms. Carries seed variance and test-set sampling."""
    rng = np.random.default_rng(seed)
    A = _by_fold(a)
    B = _by_fold(b) if b is not None else None
    folds = sorted(A)
    if B is not None:
        assert set(folds) <= set(B), "reference lacks a fold of the arm"
        for f in folds:
            assert np.array_equal(A[f][0].rows, B[f][0].rows), f"fold {f}: arms scored on different rows"

    def arm_mean(G, idx, pick):
        vals = [G[f][j].stats(idx[f]) for f in folds for j in pick[f]]
        return {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}

    full = {f: None for f in folds}
    point = arm_mean(A, full, {f: range(len(A[f])) for f in folds})
    if B is not None:
        pb = arm_mean(B, full, {f: range(len(B[f])) for f in folds})
        point = {k: point[k] - pb[k] for k in point}
    draws: dict[str, list[float]] = {k: [] for k in point}
    for _ in range(n):
        idx = {f: rng.integers(0, len(A[f][0].rows), len(A[f][0].rows)) for f in folds}
        d = arm_mean(A, idx, {f: rng.integers(0, len(A[f]), len(A[f])) for f in folds})
        if B is not None:
            r = arm_mean(B, idx, {f: rng.integers(0, len(B[f]), len(B[f])) for f in folds})
            d = {k: d[k] - r[k] for k in d}
        for k in d:
            draws[k].append(d[k])
    return {k: dict(value=v, ci=[float(np.percentile(draws[k], 2.5)), float(np.percentile(draws[k], 97.5))],
                    se=float(np.std(draws[k], ddof=1)))
            for k, v in point.items()}
