"""One place that turns ``cfg.model.arch`` into a network (IC-10).

Two architectures ship, and both are load-bearing:

===================  ============================================================
``model.arch``       What it is
===================  ============================================================
``spectral_quadnet`` The audited v4 model — four branches, gated bilinear
                     fusion, four auxiliary heads, 5.19 M parameters. Kept
                     **unchanged**. A3 (is the four-branch design justified?)
                     and A8 (do Stages 2–3 add anything?) are both comparisons
                     *against* it; deleting the control arm would make every
                     removal in CHANGES §14 unfalsifiable.
``spectral_seed_net``CHANGES §16.2 — two pathways, concat + MLP, K=1 head, one
                     auxiliary head, ≈2.8 M parameters.
===================  ============================================================

Everything downstream — the stages, the checkpoint layer, the evaluator, TTA —
is written against the *shape contract* the two share (a dict in training mode
with ``main`` and some number of ``aux_*`` keys, bare logits in eval mode) and
never against a concrete class. That is what lets an ablation switch
architectures with one config override.

Schema identity travels with the model, not with the module
───────────────────────────────────────────────────────────
``SCHEMA_VERSION`` used to be a module constant in ``engine/checkpoint.py``,
which works while there is one architecture and silently misdescribes bundles
once there are two. Each class now declares its own :attr:`SCHEMA_VERSION` and
:attr:`ARCH`, :func:`describe` reads them, and ``save_ckpt`` records both — so a
bundle says what produced it and ``load_ckpt`` can refuse a cross-architecture
load instead of matching two thirds of the tensors by coincidence of naming.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple

import torch
import torch.nn as nn

from spectralquadnet.models.spectral_quadnet import SpectralQuadNet
from spectralquadnet.models.spectral_seed_net import SpectralSeedNet

if TYPE_CHECKING:  # pragma: no cover - typing only
    from spectralquadnet.config.schema import ExperimentConfig

#: ``arch`` name → constructor taking ``(cfg, physical_wl)``.
MODEL_BUILDERS: dict[str, Callable[[Any, torch.Tensor], nn.Module]] = {
    SpectralQuadNet.ARCH: SpectralQuadNet.from_config,
    SpectralSeedNet.ARCH: SpectralSeedNet.from_config,
}

#: Every architecture name a config may name, for error messages and for the
#: ablation registry's validation.
ARCHITECTURES: tuple[str, ...] = tuple(MODEL_BUILDERS)


class ModelIdentity(NamedTuple):
    """What a bundle has to record to be loadable without guessing."""

    arch: str
    schema_version: int


def describe(model: nn.Module) -> ModelIdentity:
    """The architecture name and checkpoint schema a model writes.

    Falls back to ``("spectral_quadnet", 3)`` for a module that declares
    neither, which is exactly what every bundle written before IC-10 was.
    """
    return ModelIdentity(
        arch=str(getattr(model, "ARCH", SpectralQuadNet.ARCH)),
        schema_version=int(getattr(model, "SCHEMA_VERSION", 3)),
    )


def build_model(cfg: ExperimentConfig | Any, physical_wl: torch.Tensor) -> nn.Module:
    """Construct the architecture named by ``cfg.model.arch``.

    Args:
        cfg: Composed experiment config.
        physical_wl: Min-max normalised wavelength vector, ``(num_bands,)``.

    Raises:
        ValueError: ``cfg.model.arch`` is not a known architecture.

    Note:
        The caller must have set the RNG seed **immediately** before this call.
        Every ``kaiming_normal_``/``trunc_normal_`` inside the sub-modules draws
        from the global torch stream in construction order, so the seed, this
        call and the ``ModelEMA`` deep-copy are one ordered sequence — see
        ``train.py``'s RNG-ordering block.
    """
    arch = str(getattr(cfg.model, "arch", SpectralQuadNet.ARCH))
    builder = MODEL_BUILDERS.get(arch)
    if builder is None:
        raise ValueError(
            f"Unknown model.arch {arch!r}. Expected one of: {', '.join(ARCHITECTURES)}."
        )
    return builder(cfg, physical_wl)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Total parameter count — the headline number every ablation arm reports."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def parameter_breakdown(model: nn.Module) -> dict[str, int]:
    """Parameters per top-level child, plus ``total``.

    The table CHANGES §2.3 tabulates by hand. Produced automatically so an
    ablation arm's parameter budget lands in the run's results JSON rather than
    in a comment that drifts.
    """
    breakdown = {
        name: sum(p.numel() for p in child.parameters()) for name, child in model.named_children()
    }
    breakdown["total"] = count_parameters(model, trainable_only=False)
    return breakdown
