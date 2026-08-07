"""TensorBoard backend.

Selected via ``configs/tracking/tensorboard.yaml``, which points ``log_dir`` at
``${output_dir}/tensorboard``. Requires the optional extra::

    pip install -e ".[tracking]"

``torch.utils.tensorboard`` is imported inside
:meth:`TensorBoardTracker.__init__` so the core install never needs it.

As with the W&B backend, the human channel is inert — see
:mod:`spectralquadnet.tracking.wandb_tracker` for the rationale and for the
``tracking.backend=multi`` recipe that keeps console output alongside it. Tables
are rendered as Markdown via ``add_text``, which is the only tabular primitive
TensorBoard offers.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from spectralquadnet.tracking.base import flatten_hyperparams

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch.nn as nn


class TensorBoardTracker:
    """Write metrics to a TensorBoard event file.

    Args:
        log_dir: Event-file directory; created if missing.
    """

    def __init__(self, log_dir: str | Path = "runs") -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "tracking.backend=tensorboard requires the optional 'tracking' extra: "
                'pip install -e ".[tracking]"'
            ) from exc

        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(log_dir))
        self._closed = False

    # ── Machine channel ───────────────────────────────────────────────

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self._writer.add_scalar(tag, value, global_step=step)

    def log_scalars(self, tags: dict[str, float], step: int) -> None:
        for tag, value in tags.items():
            self._writer.add_scalar(tag, value, global_step=step)

    def log_table(self, tag: str, rows: list[dict[str, Any]], step: int) -> None:
        if not rows:
            return
        columns = list(rows[0])
        header = "| " + " | ".join(str(c) for c in columns) + " |"
        divider = "| " + " | ".join("---" for _ in columns) + " |"
        body = "\n".join("| " + " | ".join(str(r.get(c, "")) for c in columns) + " |" for r in rows)
        self._writer.add_text(tag, f"{header}\n{divider}\n{body}", global_step=step)

    def log_hyperparams(self, cfg: dict[str, Any]) -> None:
        flat = flatten_hyperparams(cfg)
        # `add_hparams` insists on a metric dict and creates a nested run; the
        # config is more useful as plain text alongside the metrics.
        lines = "\n".join(f"- **{k}**: `{v}`" for k, v in sorted(flat.items()))
        self._writer.add_text("config", lines, global_step=0)

    def watch(self, model: nn.Module) -> None:
        # Gradient histograms would need a per-step hook; the parameter count is
        # the cheap, always-correct summary.
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self._writer.add_text("model/params_M", f"{n_params / 1e6:.2f}", global_step=0)

    def close(self) -> None:
        if not self._closed:
            self._writer.flush()
            self._writer.close()
            self._closed = True

    # ── Human channel — no TensorBoard equivalent ─────────────────────

    def banner(self, title: str, lines: Sequence[str] = ()) -> None:
        return None

    def log_message(self, text: str, level: str = "info") -> None:
        return None

    def log_row(self, tag: str, cells: dict[str, str], step: int) -> None:
        return None
