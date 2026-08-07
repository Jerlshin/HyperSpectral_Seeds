"""One place that knows what a batch looks like — FE-2 / T3-7, FU-4 / P-4.

Tier 3 gave the model two optional side inputs: the persisted fill map
(``mask``) and the eight morphometrics (``morph``). Both are *optional* in the
strict sense — the model has an exact fallback for each, and the arrays that
carry them do not exist until ``scripts/prepare_dataset.py`` is re-run — so
``RiceSeedDataset`` yields a two-tuple when neither is configured and a
four-tuple when either is.

Seven loops across ``engine/`` iterate a loader. Rather than teach each of them
that arity, they call :func:`unpack_batch`, which also accepts the plain
``TensorDataset(x, y)`` the golden capture builds. The alternative — a fixed
four-tuple with sentinel tensors — would have made "no mask" indistinguishable
from "an all-zero mask", and an all-zero mask is a real and very different
thing: a patch with no foreground at all.
"""

from __future__ import annotations

from typing import Any

import torch


def _present(t: torch.Tensor | None) -> bool:
    """A side input is present when it is not the zero-length placeholder.

    ``RiceSeedDataset.ABSENT`` is a zero-length tensor because ``None`` cannot
    survive the default collate, and ``(B, 0)`` is the one shape no real mask or
    morphometric vector can take.
    """
    return t is not None and t.numel() > 0


def unpack_batch(
    batch: Any, device: torch.device | str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """``(x, y, mask, morph)`` on ``device``; the last two are ``None`` when absent.

    Args:
        batch: Whatever the loader yielded — ``(x, y)`` or ``(x, y, mask, morph)``.
        device: Target device; every returned tensor is moved to it.
    """
    x, y, *rest = batch
    side: list[torch.Tensor | None] = [None, None]
    for slot, value in enumerate(rest[:2]):
        if _present(value):
            side[slot] = value.to(device, non_blocking=True)
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True), side[0], side[1]


def side_inputs(mask: torch.Tensor | None, morph: torch.Tensor | None) -> dict[str, torch.Tensor]:
    """The ``**kwargs`` form, so a call site never passes ``mask=None`` explicitly.

    Keeps every ``model(...)`` call in ``engine/`` identical to its pre-Tier-3
    form when no side arrays are configured, which is what makes the golden
    forward-pass comparison a comparison of the same call.
    """
    out: dict[str, torch.Tensor] = {}
    if mask is not None:
        out["mask"] = mask
    if morph is not None:
        out["morph"] = morph
    return out
