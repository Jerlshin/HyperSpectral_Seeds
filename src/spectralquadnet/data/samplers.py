"""Custom batch/index samplers.

Relocated verbatim from ``HSI_modality_training/hsi_training.py`` @ ``886560f``:

========================================  ==============
Symbol                                    Baseline lines
========================================  ==============
:class:`ClassBalancedBatchSampler`        362-391
:class:`HardClassOversampledSampler`      394-446
========================================  ==============

Neither class reads ``CONFIG``, so both relocate with zero translation.

Note (REFACTOR_PLAN.md §1.3 / §3.6): ``ClassBalancedBatchSampler.__iter__``
creates an **unseeded** ``np.random.default_rng()`` on every epoch. That is one
of the two pre-existing non-deterministic islands in this codebase — it means a
full training run was never bit-reproducible even before the refactor. It is
preserved as-is; seeding it would be a behaviour change, not a relocation.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Sampler

# `Sampler` is generic, and both classes below inherit it bare — exactly as the
# baseline wrote them. Parameterising the base (`Sampler[list[int]]`) would be
# the better annotation, but it is the one typing change that is *not* erased by
# the AST no-op-move check (§3.2.1 normalises annotations, not base classes), so
# it would register as drift on a relocated class header. The ignore keeps the
# relocation bit-identical and the check at full strength.


class ClassBalancedBatchSampler(Sampler):  # type: ignore[type-arg]
    """Draws n_cls classes per batch, n_spc samples per class, with optional CDWS weighting."""

    def __init__(
        self,
        train_labels: npt.NDArray[Any],
        n_cls: int = 16,
        n_spc: int = 8,
        class_weights: dict[int, float] | None = None,
    ) -> None:
        self.n_cls = n_cls
        self.n_spc = n_spc
        self.classes = np.unique(train_labels)
        self.cls_idx = {c: np.where(train_labels == c)[0] for c in self.classes}
        self._n = len(train_labels) // (n_cls * n_spc)
        if class_weights is not None:
            raw = np.array([class_weights.get(int(c), 1.0) for c in self.classes])
            self.probs = raw / raw.sum()
        else:
            self.probs = None

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng()
        for _ in range(self._n):
            chosen = rng.choice(self.classes, self.n_cls, replace=False, p=self.probs)
            batch = []
            for c in chosen:
                pool = self.cls_idx[c]
                batch.extend(rng.choice(pool, self.n_spc, replace=len(pool) < self.n_spc).tolist())
            yield batch

    def __len__(self) -> int:
        return self._n


class HardClassOversampledSampler(Sampler):  # type: ignore[type-arg]
    def __init__(
        self,
        labels: npt.NDArray[Any],
        class_f1: dict[int, float],
        num_samples: int,
        oversample_power: float = 0.75,
        max_weight: float = 5.0,
        hard_f1_thresh: float = 0.50,
        eps: float = 0.05,
    ) -> None:
        self.num_samples = num_samples

        # ── Build per-class weights ────────────────────────────────────
        num_classes = int(np.max(labels)) + 1
        raw_weights: dict[int, float] = {}
        for c in range(num_classes):
            f1 = float(class_f1.get(c, 0.0))
            w = (1.0 / (f1 + eps)) ** oversample_power
            raw_weights[c] = min(w, max_weight)

        # Normalise so the mean weight stays at 1.0 (no overall rate change)
        mean_w = float(np.mean(list(raw_weights.values())))
        norm_weights = {c: w / mean_w for c, w in raw_weights.items()}

        # ── Assign per-sample weights ──────────────────────────────────
        sample_weights = np.array(
            [norm_weights.get(int(lbl), 1.0) for lbl in labels], dtype=np.float32
        )
        self._weights = torch.from_numpy(sample_weights)

        # ── Diagnostics ───────────────────────────────────────────────
        n_hard = sum(1 for f in class_f1.values() if f < hard_f1_thresh)
        hard_classes = sorted(
            [c for c, f in class_f1.items() if f < hard_f1_thresh], key=lambda c: class_f1[c]
        )
        print(
            f"[INFO] Phase-3 oversampling: {n_hard}/{num_classes} hard classes "
            f"(F1 < {hard_f1_thresh})  |  power={oversample_power:.2f}  "
            f"max_w={max_weight:.1f}  n_samples={num_samples:,}"
        )
        if hard_classes:
            worst5 = [(c, class_f1[c]) for c in hard_classes[:5]]
            print(f"[INFO] Hardest classes (class_id, F1): {worst5}")

    def __iter__(self) -> Iterator[int]:
        return iter(torch.multinomial(self._weights, self.num_samples, replacement=True).tolist())

    def __len__(self) -> int:
        return self.num_samples
