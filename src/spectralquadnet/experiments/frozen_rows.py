"""Load explicit, checksummed research partitions into the training pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spectralquadnet.data.loaders import SplitReport, Splits
from spectralquadnet.data.prep.multimodal import sha256


def load_frozen_rows(plan_path: Path, labels_path: Path, groups_path: Path, fold: int) -> Splits:
    if sha256(plan_path) != plan_path.with_suffix(".sha256").read_text().strip():
        raise ValueError("Frozen partition plan hash mismatch")
    plan = json.loads(plan_path.read_text())
    data = Path(plan["data"])
    for name, path in [("labels.npy", labels_path), ("groups.npy", groups_path)]:
        if sha256(path) != plan["input_hashes"][str(data / name)]:
            raise ValueError("Frozen partition dataset identity mismatch")
    labels, groups = np.load(labels_path), np.load(groups_path)
    if str(fold) not in plan["splits"]:
        raise ValueError("Unknown frozen fold")
    rows = {k: np.asarray(v, dtype=np.int64) for k, v in plan["splits"][str(fold)].items()}
    classes = np.unique(labels).astype(int).tolist()
    return Splits(
        labels=labels,
        groups=groups,
        **rows,
        report=SplitReport(
            scheme="frozen_complementary",
            fold=fold,
            n_patches=len(labels),
            n_classes=len(classes),
            n_groups=len(np.unique(groups)),
            sizes={k: len(v) for k, v in rows.items()},
            train_eval_group_disjoint=True,
            val_test_group_disjoint=False,
            calib_group_disjoint=False,
            groups_per_class_min=2,
            groups_per_class_max=2,
            classes_sharing_groups_val_test=classes,
            classes_sharing_groups_calib=classes,
        ),
    )
