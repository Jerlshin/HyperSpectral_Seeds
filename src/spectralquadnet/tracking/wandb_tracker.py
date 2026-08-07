"""Weights & Biases backend.

Selected via ``configs/tracking/wandb.yaml`` and requires the optional extra::

    pip install -e ".[tracking]"

``wandb`` is imported inside :meth:`WandbTracker.__init__`, not at module scope,
so importing :mod:`spectralquadnet.tracking` never pulls in a dependency the core
install does not declare.

**The human channel is intentionally inert here.** Banners, ``[INFO]`` lines and
epoch rows have no useful W&B representation — the numbers behind them already
arrive through :meth:`log_scalars` as plottable series. Selecting this backend
alone therefore gives a quiet terminal; to keep a readable console output
*and* stream to W&B, use the composite::

    python train.py tracking.backend=multi tracking.backends=[console,wandb]
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from spectralquadnet.tracking.base import flatten_hyperparams

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch.nn as nn


class WandbTracker:
    """Stream metrics to a W&B run.

    Args:
        project: W&B project name.
        entity: W&B team/user; ``None`` uses the account default.
        run_name: Display name for the run — the experiment's ``run_name``.
        watch_model: Enable ``wandb.watch`` gradient histograms in :meth:`watch`.
        config: Composed run config, logged at init so it is attached even if the
            run crashes before :meth:`log_hyperparams`.
    """

    def __init__(
        self,
        project: str | None = None,
        entity: str | None = None,
        run_name: str | None = None,
        watch_model: bool = False,
        config: dict[str, Any] | None = None,
    ) -> None:
        try:
            import wandb
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "tracking.backend=wandb requires the optional 'tracking' extra: "
                'pip install -e ".[tracking]"'
            ) from exc

        self._wandb = wandb
        self._watch_model = watch_model
        self._closed = False
        self._run = wandb.init(
            project=project or "spectralquadnet",
            entity=entity,
            name=run_name,
            config=flatten_hyperparams(config or {}),
        )

    # ── Machine channel ───────────────────────────────────────────────

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.log_scalars({tag: value}, step)

    def log_scalars(self, tags: dict[str, float], step: int) -> None:
        if tags:
            self._wandb.log(dict(tags), step=step)

    def log_table(self, tag: str, rows: list[dict[str, Any]], step: int) -> None:
        if not rows:
            return
        columns = list(rows[0])
        table = self._wandb.Table(
            columns=columns, data=[[row.get(c) for c in columns] for row in rows]
        )
        self._wandb.log({tag: table}, step=step)

    def log_hyperparams(self, cfg: dict[str, Any]) -> None:
        # `wandb.config.update` carries no annotations in the shipped SDK.
        self._wandb.config.update(  # type: ignore[no-untyped-call]
            flatten_hyperparams(cfg), allow_val_change=True
        )

    def watch(self, model: nn.Module) -> None:
        if self._watch_model:
            self._wandb.watch(model, log="gradients", log_freq=100)

    def close(self) -> None:
        if not self._closed:
            self._wandb.finish()
            self._closed = True

    # ── Human channel — no W&B equivalent, see the module docstring ───

    def banner(self, title: str, lines: Sequence[str] = ()) -> None:
        return None

    def log_message(self, text: str, level: str = "info") -> None:
        return None

    def log_row(self, tag: str, cells: dict[str, str], step: int) -> None:
        return None
