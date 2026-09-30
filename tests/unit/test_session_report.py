"""Same- vs cross-session breakdown (``spectralquadnet.reporting.session``).

The fixture is a miniature of the real acquisition layout: six varieties, two
bundles each, three sessions — four varieties with both bundles in one session
(the dataset's 73) and two whose bundles span two sessions (its 17). The fold
trains on each variety's first bundle and scores the second, exactly as the
grouped protocol does.

Two prediction patterns are scored against it. A *session-collapsed* model gets
every same-session kernel right and assigns every cross-session kernel to a
variety trained in that kernel's own session — the failure the September 2026
band study measured on mean-spectrum models. A *perfect* model is the reference.
Every number the report produces is checked against its hand-derived value.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from spectralquadnet.data.datasets import RiceSeedDataset
from spectralquadnet.data.loaders import eval_row_order
from spectralquadnet.reporting.session import SessionMap, session_report

PER_BUNDLE = 5
N_CLASSES = 6

#: (class, bundle) -> session. Classes 0-3 keep both bundles in one session;
#: classes 4 and 5 are imaged in two sessions.
LAYOUT = {
    (0, 0): 0, (0, 1): 0,
    (1, 0): 0, (1, 1): 0,
    (2, 0): 1, (2, 1): 1,
    (3, 0): 2, (3, 1): 2,
    (4, 0): 0, (4, 1): 1,
    (5, 0): 1, (5, 1): 2,
}  # fmt: skip
SAME_CLASSES = [0, 1, 2, 3]
CROSS_CLASSES = [4, 5]


@pytest.fixture
def layout():
    """``(scan_table, groups, labels, train_rows, eval_rows)`` for the miniature."""
    scans, groups, labels, bundle_of = [], [], [], []
    for scan_id, ((c, b), s) in enumerate(sorted(LAYOUT.items())):
        scans.append({"scan_id": scan_id, "session_id": s, "session": f"S{s}", "label": c})
        groups += [scan_id] * PER_BUNDLE
        labels += [c] * PER_BUNDLE
        bundle_of += [b] * PER_BUNDLE
    groups_a, labels_a, bundle_a = np.array(groups), np.array(labels), np.array(bundle_of)
    train = np.flatnonzero(bundle_a == 0)
    held = np.flatnonzero(bundle_a == 1)
    return pd.DataFrame(scans), groups_a, labels_a, train, held


@pytest.fixture
def smap(layout) -> SessionMap:
    table, groups, labels, train, _ = layout
    return SessionMap.build(table, groups, labels, train, N_CLASSES)


def collapsed_predictions(smap: SessionMap, rows: np.ndarray) -> np.ndarray:
    """Right on same-session kernels; a class trained in the kernel's own session otherwise."""
    preds = smap.labels[rows].copy()
    for i, r in enumerate(rows):
        y = int(smap.labels[r])
        s = int(smap.session_of_row[r])
        if s not in smap.train_sessions[y]:
            preds[i] = min(c for c in range(N_CLASSES) if s in smap.train_sessions[c] and c != y)
    return preds


# ══════════════════════════════════════════════════════════════════════
#  The map
# ══════════════════════════════════════════════════════════════════════


def test_training_sessions_come_from_the_training_rows_alone(smap) -> None:
    """Class 4 was trained in session 0 only, although its held-out bundle is in session 1."""
    assert smap.train_sessions[4] == frozenset({0})
    assert smap.train_sessions[5] == frozenset({1})
    assert smap.train_sessions[2] == frozenset({1})
    assert smap.n_sessions == 3


def test_an_unknown_scan_id_is_refused(layout) -> None:
    table, groups, labels, train, _ = layout
    with pytest.raises(ValueError, match="not in the scan table"):
        SessionMap.build(table[table.scan_id != 0], groups, labels, train, N_CLASSES)


def test_from_config_explains_why_it_is_off(layout, tmp_path) -> None:
    table, groups, labels, train, held = layout
    splits = SimpleNamespace(
        labels=labels, train=train, calib=np.array([], dtype=np.int64), groups=groups
    )

    smap, reason = SessionMap.from_config(SimpleNamespace(scan_table_path=""), splits, N_CLASSES)
    assert smap is None and "empty" in reason

    missing = SimpleNamespace(scan_table_path=str(tmp_path / "nope.csv"))
    smap, reason = SessionMap.from_config(missing, splits, N_CLASSES)
    assert smap is None and "does not exist" in reason

    path = tmp_path / "scan_table.csv"
    table.to_csv(path, index=False)
    no_groups = SimpleNamespace(**{**vars(splits), "groups": None})
    smap, reason = SessionMap.from_config(
        SimpleNamespace(scan_table_path=str(path)), no_groups, N_CLASSES
    )
    assert smap is None and "groups" in reason

    smap, _ = SessionMap.from_config(SimpleNamespace(scan_table_path=str(path)), splits, N_CLASSES)
    assert smap is not None and smap.train_sessions[4] == frozenset({0})


# ══════════════════════════════════════════════════════════════════════
#  The report
# ══════════════════════════════════════════════════════════════════════


def test_a_session_collapsed_model_is_exposed(layout, smap) -> None:
    *_, held = layout
    preds = collapsed_predictions(smap, held)
    report = session_report(preds, smap.labels[held], held, smap, split="val_test_tta")

    assert report.same.classes == SAME_CLASSES
    assert report.cross.classes == CROSS_CLASSES
    assert report.same.macro_recall == pytest.approx(1.0)
    assert report.cross.macro_recall == pytest.approx(0.0)
    assert report.recall_gap == pytest.approx(1.0)
    assert report.attraction_cross == pytest.approx(1.0)
    # Class 4's held-out bundle is in session 1 (trained there: classes 2 and 5 → 2/6);
    # class 5's is in session 2 (trained there: class 3 → 1/6).
    assert report.attraction_cross_chance == pytest.approx((2 / 6 + 1 / 6) / 2)
    # Every held-out kernel is predicted into its own session, so the predicted
    # session is fully determined by the acquisition session — while the truth
    # is not (sessions 1 and 2 also hold cross-session varieties).
    assert report.entropy_all.predicted == pytest.approx(0.0)
    assert report.entropy_all.predicted < report.entropy_all.oracle
    flags = " ".join(report.flags())
    assert "at or below chance" in flags and "collapse onto session identity" in flags


def test_a_perfect_model_sits_on_the_oracle(layout, smap) -> None:
    *_, held = layout
    y = smap.labels[held]
    report = session_report(y, y, held, smap)

    assert report.same.macro_recall == pytest.approx(1.0)
    assert report.cross.macro_recall == pytest.approx(1.0)
    assert report.attraction_cross == pytest.approx(0.0)
    assert report.attraction_all == pytest.approx(report.attraction_all_oracle)
    assert report.entropy_all.predicted == pytest.approx(report.entropy_all.oracle)
    assert report.flags() == []


def test_the_chance_entropy_is_the_uniform_guess(smap, layout) -> None:
    """Training sessions per class: S0 ×3, S1 ×2, S2 ×1 → H = H(1/2, 1/3, 1/6)."""
    *_, held = layout
    y = smap.labels[held]
    report = session_report(y, y, held, smap)
    p = np.array([3, 2, 1]) / 6
    assert report.entropy_all.chance == pytest.approx(float(-(p * np.log2(p)).sum()))
    assert report.entropy_all.maximum == pytest.approx(np.log2(3))


def test_a_wrong_row_order_is_refused_not_scored(layout, smap) -> None:
    *_, held = layout
    y = smap.labels[held]
    with pytest.raises(ValueError, match="does not reproduce the targets"):
        session_report(y, y, held[::-1], smap)


def test_ddp_padding_duplicates_are_dropped(layout, smap) -> None:
    *_, held = layout
    rows = np.concatenate([held, held[:2]])
    y = smap.labels[rows]
    report = session_report(y, y, rows, smap)
    assert report.n_duplicates_dropped == 2
    assert report.n_kernels == len(held)


def test_the_report_serialises_and_tags(layout, smap) -> None:
    *_, held = layout
    report = session_report(collapsed_predictions(smap, held), smap.labels[held], held, smap)
    payload = json.loads(json.dumps(report.as_dict()))
    assert payload["cross_session"]["n_classes"] == 2
    assert len(payload["per_session"]) == 3
    tags = report.scalars("val_test_tta")
    assert tags["val_test_tta/session/cross_macro_recall"] == pytest.approx(0.0)
    assert all(isinstance(v, float) for v in tags.values())
    assert any("cross-session" in line for line in report.lines())


def test_without_cross_session_classes_the_cross_fields_are_absent(layout, smap) -> None:
    """A stratified split has no cross-session kernels; nothing is invented for them."""
    table, groups, labels, _, _ = layout
    everything = np.arange(len(labels))
    smap_all = SessionMap.build(table, groups, labels, everything, N_CLASSES)
    report = session_report(labels, labels, everything, smap_all)
    assert report.cross.n_classes == 0
    assert report.cross.macro_recall is None
    assert report.attraction_cross is None
    assert "val_test/session/cross_macro_recall" not in report.scalars("val_test")


# ══════════════════════════════════════════════════════════════════════
#  Which row each prediction belongs to
# ══════════════════════════════════════════════════════════════════════


class _Store:
    def __init__(self, n: int) -> None:
        self.patches = np.zeros((n, 2, 2, 2), dtype=np.float32)
        self.labels = np.zeros(n, dtype=np.int64)

    def require_patches(self) -> np.ndarray:
        return self.patches

    def require_labels(self) -> np.ndarray:
        return self.labels


def _dataset(indices: np.ndarray) -> RiceSeedDataset:
    cfg = SimpleNamespace(max_cutout_bands=1, noise_std=0.0, cutmix_bands=0, cutmix_spatial=0)
    return RiceSeedDataset(indices, store=_Store(200), data_cfg=cfg)


def test_a_sequential_loader_scores_rows_in_index_order() -> None:
    idx = np.array([7, 3, 42, 11, 0])
    assert np.array_equal(eval_row_order(DataLoader(_dataset(idx), batch_size=2)), idx)


@pytest.mark.parametrize("n, world", [(10, 3), (9, 3), (5, 4), (1, 2)])
def test_the_ddp_order_matches_what_the_gather_returns(n: int, world: int) -> None:
    """Rank-interleaved, padded, joined in rank order — exactly as ``gather_concat`` joins."""
    idx = np.arange(100, 100 + n)
    ds = _dataset(idx)
    shards = [
        list(DistributedSampler(ds, num_replicas=world, rank=r, shuffle=False, drop_last=False))
        for r in range(world)
    ]
    expected = idx[np.concatenate(shards)]
    rank0 = DataLoader(
        ds, sampler=DistributedSampler(ds, num_replicas=world, rank=0, shuffle=False)
    )
    assert np.array_equal(eval_row_order(rank0), expected)


def test_an_unknown_order_is_none_not_a_guess() -> None:
    idx = np.arange(6)
    assert eval_row_order(DataLoader(TensorDataset(torch.zeros(6, 1)))) is None
    assert eval_row_order(DataLoader(_dataset(idx), shuffle=True)) is None
    ddp = DistributedSampler(_dataset(idx), num_replicas=2, rank=0, shuffle=True)
    assert eval_row_order(DataLoader(_dataset(idx), sampler=ddp)) is None


# ══════════════════════════════════════════════════════════════════════
#  End to end through final_evaluation
# ══════════════════════════════════════════════════════════════════════


class _ValueModel(nn.Module):
    """Predicts the class written into the patch — so the test decides every prediction."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self._use_arcface = False

    def use_arcface(self, flag: bool) -> None:
        self._use_arcface = flag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls = x[:, 0, 0, 0].round().long()
        return nn.functional.one_hot(cls, N_CLASSES).float() * 10.0 + self.anchor


def test_final_evaluation_writes_the_session_breakdown(cfg, tmp_path, layout, smap) -> None:
    from spectralquadnet.engine.checkpoint import save_ckpt
    from spectralquadnet.engine.stages.final_eval import final_evaluation
    from spectralquadnet.models.ema import ModelEMA

    *_, held = layout
    store = _Store(len(smap.labels))
    store.labels = smap.labels.copy()
    preds_wanted = collapsed_predictions(smap, held)
    store.patches[held] = preds_wanted[:, None, None, None].astype(np.float32)
    data_cfg = SimpleNamespace(max_cutout_bands=1, noise_std=0.0, cutmix_bands=0, cutmix_spatial=0)
    loader = DataLoader(RiceSeedDataset(held, store=store, data_cfg=data_cfg), batch_size=4)

    eval_cfg = cfg.copy()
    eval_cfg.output_dir = str(tmp_path)
    eval_cfg.data.num_classes = N_CLASSES
    eval_cfg.evaluation.bootstrap_samples = 0
    eval_cfg.evaluation.tta = False
    eval_cfg.evaluation.save_artifacts = False
    eval_cfg.evaluation.report_split = "val_test"

    model = _ValueModel()
    ema = ModelEMA(_ValueModel(), decay=0.99)
    ckpt = tmp_path / "best.pth"
    save_ckpt(eval_cfg, str(ckpt), 1, "Single", model, ema, val_f1=0.5, val_acc=0.5)

    final_evaluation(eval_cfg, model, ema, loader, torch.device("cpu"), str(ckpt), sessions=smap)

    results = tmp_path / "results"
    assert np.array_equal(np.load(results / "rows_val_test_no_tta.npy"), held)
    breakdown = json.loads((results / "session_val_test_no_tta.json").read_text())
    assert breakdown["cross_session"]["macro_recall"] == pytest.approx(0.0)
    assert breakdown["attraction"]["cross"] == pytest.approx(1.0)
    manifest = json.loads((results / "run.json").read_text())
    assert manifest["results"]["no_tta"]["session"]["same_session"]["macro_recall"] == 1.0
    assert (results / "session_val_test_no_tta.csv").read_text().startswith("session_id,")


def test_final_evaluation_without_a_map_writes_no_breakdown(cfg, tmp_path) -> None:
    """The default path is byte-for-byte what it was: no rows, no session files."""
    from spectralquadnet.engine.checkpoint import save_ckpt
    from spectralquadnet.engine.stages.final_eval import final_evaluation
    from spectralquadnet.models.ema import ModelEMA

    eval_cfg = cfg.copy()
    eval_cfg.output_dir = str(tmp_path)
    eval_cfg.data.num_classes = N_CLASSES
    eval_cfg.evaluation.bootstrap_samples = 0
    eval_cfg.evaluation.tta = False
    eval_cfg.evaluation.save_artifacts = False
    model, ema = _ValueModel(), ModelEMA(_ValueModel(), decay=0.99)
    ckpt = tmp_path / "best.pth"
    save_ckpt(eval_cfg, str(ckpt), 1, "Single", model, ema, val_f1=0.5, val_acc=0.5)
    loader = DataLoader(TensorDataset(torch.ones(4, 2, 2, 2), torch.ones(4).long()), batch_size=2)

    final_evaluation(eval_cfg, model, ema, loader, torch.device("cpu"), str(ckpt))

    results = tmp_path / "results"
    assert not list(results.glob("rows_*.npy"))
    assert not list(results.glob("session_*"))
    manifest = json.loads((results / "run.json").read_text())
    assert "session" not in manifest["results"]["no_tta"]
