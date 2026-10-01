"""Clean training fit on a fixed subset — the measurement H13 and H16 are about.

S09 F34/F35 had to read "how well does the network fit its training data" off
the one epoch where mixup had stopped and the margin had not yet started,
because the only training accuracy the loop logs is scored against one of two
mixup labels (capped at ≈ 0.5) or against margin-penalised logits. S10 measured
it offline on the selected checkpoints (F45: 0.87–0.95) and wrote the definition
D18 asks for down precisely (S10 P0.2):

* a **fixed**, class-stratified subset of the run's training rows — 1,000 kernels
  by default, drawn by a private RNG from ``single.clean_fit_seed`` alone, so
  every seed of a fold measures the *same* kernels;
* **eval mode** (no dropout, BatchNorm on running statistics), **no
  augmentation**, **no margin**, **no label smoothing**: the plain scaled-cosine
  logits a deployed model produces, and the cross-entropy against them;
* the **live** and the **EMA** weights, both.

Measuring must not perturb what is measured. The probe's loader carries its own
``torch.Generator`` (starting any loader iterator otherwise draws a seed from the
global torch RNG and would shift every later dropout mask and mixup pairing), it
runs under ``inference_mode`` with autocast off, and it restores each model's
train/eval flag. S11's gate G-neutral checks the whole claim end to end.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader

from spectralquadnet.data.datasets import RiceSeedDataset
from spectralquadnet.data.loaders import build_eval_loader, dedup_index
from spectralquadnet.engine.batch import side_inputs, unpack_batch
from spectralquadnet.utils.distributed import DistContext, gather_concat

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch.nn as nn

    from spectralquadnet.data.mmap_store import DataStore
    from spectralquadnet.utils.device import RuntimePlan

#: Written into ``output_dir`` by the single-stage curriculum.
CLEAN_FIT_FILE = "clean_fit.json"

#: The definition, recorded verbatim in every ``clean_fit.json``.
DEFINITION = (
    "fixed class-stratified training subset; eval mode; no augmentation; margin 0; "
    "no label smoothing; accuracy and mean cross-entropy of the plain scaled-cosine logits"
)


def clean_fit_subset(
    train_rows: npt.NDArray[Any], labels: npt.NDArray[Any], n: int, seed: int
) -> npt.NDArray[np.int64]:
    """A fixed, class-stratified subset of ``train_rows``, as sorted patch rows.

    Every class's rows are permuted by one private generator, then taken
    round-robin — one per class per round, the class order of each round itself
    permuted — until ``n`` are chosen. So class counts differ by at most one,
    and the subset depends on the training rows and ``seed`` only: never on the
    run's seed, never on the global RNG.

    Returns all of ``train_rows`` (sorted) when ``n >= len(train_rows)``.
    """
    rows = np.unique(np.asarray(train_rows, dtype=np.int64))
    if n <= 0:
        return rows[:0]
    if n >= rows.size:
        return rows
    rng = np.random.default_rng(int(seed))
    row_labels = np.asarray(labels)[rows]
    classes = np.unique(row_labels)
    pools = {int(c): list(rng.permutation(rows[row_labels == c])) for c in classes}
    chosen: list[int] = []
    depth = 0
    while len(chosen) < n:
        progressed = False
        for c in rng.permutation(classes):
            pool = pools[int(c)]
            if depth < len(pool):
                chosen.append(int(pool[depth]))
                progressed = True
                if len(chosen) == n:
                    break
        if not progressed:  # pragma: no cover - n < rows.size guarantees progress
            break
        depth += 1
    return np.sort(np.asarray(chosen, dtype=np.int64))


@dataclass(frozen=True)
class FitMeasure:
    """One clean-fit measurement of one weight set."""

    acc: float
    ce: float
    n: int

    def as_dict(self) -> dict[str, float | int]:
        return {"acc": self.acc, "ce": self.ce, "n": self.n}


@torch.inference_mode()
def measure_fit(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    dist: DistContext | None = None,
) -> FitMeasure:
    """Accuracy and mean CE of ``model``'s eval-mode logits over ``loader``.

    fp32 regardless of the training autocast, gathered across ranks and
    de-duplicated, so every kernel of the subset counts exactly once.
    """
    dist = dist or DistContext()
    was_training = model.training
    model.eval()
    ce_parts: list[torch.Tensor] = []
    hit_parts: list[torch.Tensor] = []
    try:
        with autocast(device_type=device.type, enabled=False):
            for batch in loader:
                x, y, mask, morph = unpack_batch(batch, device)
                logits = model(x, **side_inputs(mask, morph)).float()
                ce_parts.append(F.cross_entropy(logits, y, reduction="none"))
                hit_parts.append((logits.argmax(1) == y).float())
                del x, mask, morph, logits
    finally:
        model.train(was_training)
    ce = gather_concat(dist, torch.cat(ce_parts)).cpu().numpy()
    hit = gather_concat(dist, torch.cat(hit_parts)).cpu().numpy()
    keep = dedup_index(loader) if dist.enabled else None
    if keep is not None:
        ce, hit = ce[keep], hit[keep]
    return FitMeasure(acc=float(hit.mean()), ce=float(ce.mean()), n=int(hit.size))


@dataclass
class CleanFitProbe:
    """The fixed subset, its loader, and the run's record of every measurement."""

    rows: npt.NDArray[np.int64]
    loader: DataLoader[Any]
    seed: int
    labels: npt.NDArray[Any]
    history: list[dict[str, Any]] = field(default_factory=list)
    at_best: dict[str, Any] | None = None

    @classmethod
    def build(
        cls,
        *,
        store: DataStore,
        data_cfg: Any,
        device: torch.device,
        train_rows: npt.NDArray[Any],
        labels: npt.NDArray[Any],
        morph: npt.NDArray[Any] | None,
        n: int,
        seed: int,
        plan: RuntimePlan | None = None,
        dist: DistContext | None = None,
    ) -> CleanFitProbe | None:
        """The probe for one run, or ``None`` when ``n <= 0`` (telemetry off)."""
        if n <= 0:
            return None
        rows = clean_fit_subset(train_rows, labels, n, seed)
        dataset = RiceSeedDataset(rows, store=store, data_cfg=data_cfg, device=device, morph=morph)
        # A private generator: see the module docstring. Seeded from the subset
        # seed so the loader's own bookkeeping is reproducible too.
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        loader = build_eval_loader(dataset, plan=plan, dist=dist, generator=generator)
        return cls(rows=rows, loader=loader, seed=int(seed), labels=np.asarray(labels)[rows])

    def measure(
        self,
        live: nn.Module,
        ema: nn.Module,
        device: torch.device,
        dist: DistContext | None = None,
    ) -> dict[str, FitMeasure]:
        """Both weight sets, live first."""
        return {
            "live": measure_fit(live, self.loader, device, dist),
            "ema": measure_fit(ema, self.loader, device, dist),
        }

    @staticmethod
    def scalars(fit: dict[str, FitMeasure]) -> dict[str, float]:
        """``fit/clean_train_{acc,ce}_{live,ema}`` — the tracker series."""
        out: dict[str, float] = {}
        for source, m in fit.items():
            out[f"fit/clean_train_acc_{source}"] = m.acc
            out[f"fit/clean_train_ce_{source}"] = m.ce
        return out

    def record(self, epoch: int, fit: dict[str, FitMeasure], *, improved: bool) -> dict[str, Any]:
        """Append one measurement; remember it if it describes the saved checkpoint."""
        entry = {"epoch": int(epoch), **{k: v.as_dict() for k, v in fit.items()}}
        self.history.append(entry)
        if improved:
            self.at_best = entry
        return entry

    def payload(self, *, final_epoch: int) -> dict[str, Any]:
        """The ``clean_fit.json`` document: definition, subset, final and at-best values."""
        counts = np.bincount(self.labels.astype(np.int64)) if self.labels.size else np.zeros(0)
        final = self.history[-1] if self.history else None
        return {
            "definition": DEFINITION,
            "subset": {
                "n": int(self.rows.size),
                "seed": self.seed,
                "per_class_min": int(counts[counts > 0].min()) if counts.size else 0,
                "per_class_max": int(counts.max()) if counts.size else 0,
                "rows": [int(r) for r in self.rows],
            },
            "final_epoch": int(final_epoch),
            "final": final if final is not None and final["epoch"] == final_epoch else None,
            "at_best_checkpoint": self.at_best,
            "history": self.history,
        }

    def write(self, output_dir: str | Path, *, final_epoch: int) -> Path:
        path = Path(output_dir) / CLEAN_FIT_FILE
        path.write_text(json.dumps(self.payload(final_epoch=final_epoch), indent=1) + "\n")
        return path
