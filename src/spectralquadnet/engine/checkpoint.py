"""Stage checkpoints, JSON meta sidecars and auto-resume detection.

The following conventions are load-bearing and must not drift:

* **Filenames** — ``best_stage{1,2,3}.pth`` and ``stage{1,2,3}_meta.json``.
* **Bundle schema** — ``{"epoch", "stage", "model", "ema", "val_f1", "val_acc",
  "use_arcface", "schema_version", **metadata}``, with the sidecar dropping
  ``model``/``ema`` and keeping only what :func:`_is_json_serialisable`
  accepts.
* **``schema_version``** — :data:`SCHEMA_VERSION`. Version 1 is every bundle
  written before Tier 2; version 2 is the HD-1 unified head (``linear_head``
  gone) plus the HD-3 ``arcface_head.confusion`` buffer.
  :func:`remap_state_dict` is the migration, applied by :func:`load_ckpt`, so
  the archived ``output_v12_spa40`` checkpoints keep loading.
* **``use_arcface``** — retained in the bundle at a constant ``True`` so
  existing readers (``scripts/phase0_*.py``, the resume banner) keep working.
  Since HD-1 there is only one head and nothing selects between two.
* **``best_source``** — every stage records, as metadata, *which* of the two
  weight sets its ``val_f1`` was measured on: ``"live"``, ``"ema"`` or (Stage 3)
  ``"swa"``. ``val_f1`` is a max over both, and ``final_eval`` used to evaluate
  the EMA shadow unconditionally, so a checkpoint selected on the live model's
  score was reported from a model that never scored it (§2.1.4). Bundles
  written before Tier 1 have no such key and default to ``"ema"``, which is the
  behaviour they were produced under.
* **Resume order** — :func:`latest_completed_stage` probes 3 → 2 → 1 and
  requires *both* the ``.pth`` and its ``.json`` to call a stage complete, so a
  crash between the two writes resumes rather than skipping.
* **Selection rule** — :func:`_pick_best_checkpoint` ranks by ``val_f1`` from the
  sidecar, falling back to the bundle, then to 0.0.
* **Wrapper transparency** — every ``state_dict()`` taken here goes through
  :func:`~spectralquadnet.utils.device.unwrap_model` first. ``torch.compile``
  prefixes every key with ``_orig_mod.`` and DDP with ``module.``, so a
  compiled or two-GPU run would otherwise write a bundle that no single-device
  run could load, and ``SCHEMA_VERSION`` would be describing the wrong thing.
  Only rank 0 writes: every rank holds identical weights, so a second writer
  would only race the first.

:func:`load_stage_meta` re-integerises dict keys because JSON stringifies them:
``class_f1``/``cdws_weights`` are ``{int: float}`` in memory and must come back
that way for :func:`~spectralquadnet.losses.cdws.build_cdws_weights` and the
samplers to index them.
"""

from __future__ import annotations

# Aliased as `_json` (not `json`) so this module stays symbol-for-symbol
# comparable against the reference implementation `scripts/check_ast_no_op_move.py`
# checks it against.
import json as _json
import os
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from spectralquadnet.engine.batch import side_inputs, unpack_batch
from spectralquadnet.models.ema import ModelEMA
from spectralquadnet.models.registry import describe
from spectralquadnet.utils.device import unwrap_model
from spectralquadnet.utils.distributed import DistContext

if TYPE_CHECKING:  # pragma: no cover - typing only
    from spectralquadnet.config.schema import ExperimentConfig

#: Checkpoint-schema version written by :func:`save_ckpt`.
#:
#: 1 → pre-Tier-2: a ``linear_head`` for Stage 1 and an ``arcface_head`` for
#: Stages 2-3, selected by a persisted ``use_arcface`` flag; no
#: ``arcface_head.confusion``.
#: 2 → HD-1 (T2-10) + HD-3 (T2-8): one head for every stage, and the pairwise
#: confusion buffer.
#: 3 → Tier 3 (T3-1 … T3-7): every branch's *input* changed, so most branch
#: tensors changed shape or ceased to exist. This one is not migratable — see
#: :func:`remap_state_dict`.
#: 4 → IC-10: ``SpectralSeedNet``. Not a migration of 3 at all but a *different
#: architecture*, which is why :data:`SCHEMA_VERSION` is no longer the whole
#: story — see below.
SCHEMA_VERSION: int = 3

#: Since IC-10 the schema version belongs to the **model class**, not to this
#: module: ``SpectralQuadNet.SCHEMA_VERSION == 3`` and
#: ``SpectralSeedNet.SCHEMA_VERSION == 4``, and a bundle records both the version
#: and the :attr:`ARCH` that wrote it. The module constant above is retained as
#: the value for a model that declares neither, which is exactly what every
#: pre-IC-10 bundle was, so the archived ``output_v12_spa40`` checkpoints keep
#: loading and every existing reader of this name keeps working.
#:
#: The version alone was never sufficient and now visibly is not: two
#: architectures can both be "current" simultaneously, and their state dicts
#: share no tensor names. A v3 bundle handed to ``SpectralSeedNet`` is not an
#: old checkpoint to migrate, it is the wrong model — :func:`load_ckpt` raises
#: rather than loading whatever happens to match.


class ArchitectureMismatchError(RuntimeError):
    """A bundle was written by a different architecture than the one loading it."""


#: State-dict prefixes that exist only in schema v1.
_V1_ONLY_PREFIXES: tuple[str, ...] = ("linear_head.",)

#: Buffers introduced by schema v2, with the value a v1 bundle is upgraded to.
#: An all-zero ``confusion`` makes the pairwise margin term identically zero,
#: which is exactly what a v1 checkpoint was trained under.
_V2_ADDED_BUFFERS: tuple[str, ...] = ("arcface_head.confusion",)

#: The last schema whose bundles can be upgraded into the live model. Anything
#: at or below this is migrated; anything below :data:`SCHEMA_VERSION` and above
#: it does not exist yet.
_MIGRATABLE_THROUGH: int = 2


class SchemaTooOldError(RuntimeError):
    """A bundle predates Tier 3 and cannot be upgraded into the current model."""


def remap_state_dict(
    sd: dict[str, Any],
    reference: dict[str, torch.Tensor],
    version: int,
    target_version: int = SCHEMA_VERSION,
) -> dict[str, Any]:
    """Upgrade a checkpoint's ``state_dict`` from ``version`` to :data:`SCHEMA_VERSION`.

    Two things changed in v2 and each is mechanical:

    * ``linear_head.*`` is dropped. HD-1 replaced the two-head design with one
      cosine head used at every stage, so those tensors have no home and
      nothing reads them.
    * ``arcface_head.confusion`` is added, zero-filled from ``reference``. Zero
      is not a placeholder: it makes the pairwise margin term vanish, so a
      migrated v1 checkpoint scores exactly what it scored before.

    **v1 and v2 cannot be migrated to v3, and this function says so rather than
    trying.** Tier 3 changed what three of the four branches *receive*: Branch B
    reads a 64-index bank where it read nine moments, Branch C's stem is 3-D
    where it was two 1x1 convolutions, Branch D runs at ``d_model = 192`` where
    it ran at 256, and the fusion has no attention at all. A weight trained to
    read one of those inputs is not a weight for the other, whatever its shape,
    so the only honest migration is a refusal: partially loading two thirds of a
    checkpoint and reporting the result would be a number about nothing.

    Args:
        sd: The bundle's ``model`` or ``ema`` state dict.
        reference: The live model's ``state_dict()``, used only for the shape
            and dtype of any buffer being added.
        version: The bundle's recorded ``schema_version`` (absent → 1).
        target_version: The schema the loading model writes. Defaults to
            :data:`SCHEMA_VERSION` so pre-IC-10 callers are unaffected;
            :func:`load_ckpt` passes the model's own.

    Returns:
        A new dict loadable with ``strict=True``.

    Raises:
        SchemaTooOldError: ``version`` predates the Tier-3 architecture.
    """
    if version >= target_version:
        return sd
    if version <= _MIGRATABLE_THROUGH:
        raise SchemaTooOldError(
            f"checkpoint schema v{version} predates Tier 3 (v{target_version}). Branches B, C "
            "and D changed what they consume, not just how many parameters they consume it "
            "with (IMPROVEMENT_PLAN §3.3 BR-1/BR-3/BR-4), so no tensor-level remap exists. "
            "Retrain, or check out the Tier-2 tree to re-score the archived checkpoints."
        )
    if version < target_version:
        # v3 → v4 is `SpectralQuadNet` → `SpectralSeedNet`, which `load_ckpt`
        # has already rejected by architecture. Reaching here means a caller
        # bypassed that check, so say the same thing rather than silently
        # returning a state dict that will fail `strict=True` with a wall of
        # missing keys.
        raise SchemaTooOldError(
            f"checkpoint schema v{version} cannot be upgraded to v{target_version}: they are "
            "different architectures (CHANGES IC-10), not two versions of one. Load it into "
            "the model that wrote it."
        )
    out = {k: v for k, v in sd.items() if not k.startswith(_V1_ONLY_PREFIXES)}
    for key in _V2_ADDED_BUFFERS:
        if key not in out and key in reference:
            out[key] = torch.zeros_like(reference[key])
    return out


#: Modules whose randomness :func:`update_bn_stats` switches off for its pass.
#:
#: ``nn.MultiheadAttention`` is in the list because its dropout is a plain
#: float attribute applied when ``self.training`` — ``SpectralQuadNet.set_dropout``
#: iterates ``nn.Dropout`` instances and cannot reach it (N-2).
_STOCHASTIC_MODULES: tuple[type[nn.Module], ...] = (
    nn.Dropout,
    nn.Dropout1d,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.AlphaDropout,
    nn.FeatureAlphaDropout,
    nn.MultiheadAttention,
)


def stage_ckpt_path(cfg: ExperimentConfig | Any, s: int) -> str:
    """Path to stage ``s``'s ``.pth`` checkpoint under ``cfg.output_dir``."""
    return os.path.join(cfg.output_dir, f"best_stage{s}.pth")


def stage_meta_path(cfg: ExperimentConfig | Any, s: int) -> str:
    """Path to stage ``s``'s JSON meta sidecar under ``cfg.output_dir``."""
    return os.path.join(cfg.output_dir, f"stage{s}_meta.json")


def stage_exists(cfg: ExperimentConfig | Any, s: int) -> bool:
    """True only if both stage ``s``'s checkpoint and its meta sidecar are present."""
    return os.path.isfile(stage_ckpt_path(cfg, s)) and os.path.isfile(stage_meta_path(cfg, s))


def latest_completed_stage(cfg: ExperimentConfig | Any) -> int:
    """Highest stage (3, 2 or 1) that has a complete checkpoint, or 0 if none do.

    Used to decide where auto-resume should pick up training.
    """
    for s in (3, 2, 1):
        if stage_exists(cfg, s):
            return s
    return 0


def save_ckpt(
    cfg: ExperimentConfig | Any,
    path: str,
    epoch: int,
    stage: str,
    model: nn.Module,
    ema: ModelEMA,
    val_f1: float,
    val_acc: float,
    dist: DistContext | None = None,
    **metadata: Any,
) -> None:
    """Persist the full checkpoint bundle plus a JSON-only meta sidecar.

    The sidecar carries every JSON-serialisable bundle key except ``model``
    and ``ema`` (their tensors are only in the ``.pth``), so downstream code
    can inspect ``val_f1``/``val_acc``/metadata without deserialising torch
    state.

    A no-op on every rank but rank 0 — see the module docstring.

    Args:
        cfg: Composed experiment config, read for ``cfg.output_dir``.
        path: Destination path for the ``.pth`` bundle.
        epoch: Epoch number to record.
        stage: Stage label (e.g. ``"stage 2"``); its trailing token must be
            the stage's digit, since the sidecar path is derived from it.
        model: Model whose ``state_dict()`` is saved.
        ema: EMA shadow whose ``state_dict()`` is saved under ``"ema"``.
        val_f1: Validation macro-F1 to record.
        val_acc: Validation accuracy to record.
        dist: DDP context; only the main rank writes.
        **metadata: Extra JSON-serialisable fields merged into the bundle
            (and, where serialisable, into the sidecar).
    """
    if dist is not None and not dist.is_main:
        return
    core = unwrap_model(model)
    identity = describe(core)
    bundle = {
        "epoch": epoch,
        "stage": stage,
        # Unwrapped, so a compiled or DDP-wrapped run writes the same key names
        # a plain one does and the bundle stays loadable anywhere.
        "model": core.state_dict(),
        "ema": ema.state_dict(),
        "val_f1": val_f1,
        "val_acc": val_acc,
        # Constant since HD-1 removed the second head; kept in the schema so
        # every existing reader of this key keeps working.
        "use_arcface": True,
        # Read from the model rather than from a module constant (IC-10): with
        # two architectures live, a bundle that records only a version number
        # does not identify what wrote it.
        "schema_version": identity.schema_version,
        "arch": identity.arch,
        **metadata,
    }
    torch.save(bundle, path)
    sidecar = {
        k: v for k, v in bundle.items() if k not in ("model", "ema") and _is_json_serialisable(v)
    }
    sn = int(stage.split()[-1]) if stage.split()[-1].isdigit() else 0
    with open(stage_meta_path(cfg, sn), "w") as f:
        _json.dump(sidecar, f, indent=2)


def _is_json_serialisable(v: Any) -> bool:
    """Whether ``v`` survives a round trip through ``json.dumps``."""
    try:
        _json.dumps(v)
        return True
    except (TypeError, ValueError):
        return False


def load_stage_meta(cfg: ExperimentConfig | Any, s: int) -> dict[str, Any]:
    """Load stage ``s``'s JSON sidecar, re-integerising any string-keyed dict values.

    JSON forces object keys to strings, so any top-level dict value (e.g.
    ``class_f1``, ``cdws_weights``) has its keys coerced back to ``int``
    where possible, restoring the ``{int: float}`` shape callers expect.

    Returns:
        The sidecar contents, or ``{}`` if the file does not exist.
    """
    p = stage_meta_path(cfg, s)
    if not os.path.isfile(p):
        return {}
    with open(p) as f:
        raw = _json.load(f)
    out = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            try:
                out[k] = {int(kk): vv for kk, vv in v.items()}
                continue
            except (ValueError, TypeError):
                pass
        out[k] = v
    return out


def load_ckpt(path: str, model: nn.Module, ema: ModelEMA, device: torch.device) -> dict[str, Any]:
    """Load a checkpoint bundle into ``model`` and its EMA shadow in place.

    Bundles older than the target model's schema are passed through
    :func:`remap_state_dict` first, so the three archived ``output_v12_spa40``
    checkpoints — schema v1, with a ``linear_head`` this model no longer has —
    still load ``strict=True``.

    Raises:
        ArchitectureMismatchError: The bundle names a different ``arch`` than
            the model being loaded into. This is checked *before* the version,
            because "old" and "wrong" are different failures with different
            fixes and a v3 bundle handed to ``SpectralSeedNet`` is the second.
        SchemaTooOldError: The bundle's schema predates what the target model
            can be migrated from.

    Returns:
        The full deserialised bundle (including ``val_f1``/``val_acc``/metadata).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    version = int(ckpt.get("schema_version", 1))
    # The eager module, for the same reason `save_ckpt` unwraps: bundles are
    # written without a wrapper prefix, so they must be loaded without one.
    target = unwrap_model(model)
    identity = describe(target)
    # A bundle with no `arch` key predates IC-10 and is therefore a
    # `SpectralQuadNet` one — the only architecture that existed when it was
    # written.
    bundle_arch = str(ckpt.get("arch", "spectral_quadnet"))
    if bundle_arch != identity.arch:
        raise ArchitectureMismatchError(
            f"{path} was written by {bundle_arch!r} (schema v{version}) but is being loaded "
            f"into {identity.arch!r} (schema v{identity.schema_version}). The two share no "
            "tensor names, so there is no remap — point `model.arch` at the architecture that "
            "produced the checkpoint, or train this one from scratch."
        )
    reference = target.state_dict()
    target.load_state_dict(
        remap_state_dict(ckpt["model"], reference, version, identity.schema_version)
    )
    ema.load_state_dict(remap_state_dict(ckpt["ema"], reference, version, identity.schema_version))
    return ckpt  # type: ignore[no-any-return]


def update_bn_stats(loader: DataLoader[Any], model: nn.Module, device: torch.device) -> None:
    """Recompute BatchNorm running statistics over ``loader`` (used after SWA averaging).

    Resets every BatchNorm layer's running mean/var and momentum to ``None``
    (cumulative-average mode), then runs one pass over ``loader`` with the
    BatchNorm layers in ``train()`` mode, so the running stats reflect the
    SWA-averaged weights rather than any individual snapshot's stats.

    **Every stochastic module is forced to ``eval()`` for the pass**
    (:data:`_STOCHASTIC_MODULES` — ``nn.Dropout`` and the internal dropout of
    ``nn.MultiheadAttention``, which ``SpectralQuadNet.set_dropout`` cannot
    reach). Inverted dropout preserves the mean but inflates the variance by
    ``p/(1-p) * E[a^2]``, so a train-mode pass records a per-channel σ larger
    than the eval-time σ and every downstream activation comes out attenuated
    — non-uniformly, so the fusion LayerNorm cannot undo it. Branch D's
    ``BatchNorm1d`` sits below four transformer blocks with two dropout sites
    each, which is where this bites hardest (IMPROVEMENT_PLAN §2.5.6(i),
    §3.6 OP-4.6).

    With the dropout off the pass is also **device-independent**: it was the
    Metal fused-attention kernel's refusal to run dropout under ``no_grad``
    that previously forced grad to stay enabled on MPS and only there, making
    the SWA checkpoint's BN buffers differ between accelerators (N-12). Nothing
    stochastic remains, so plain ``no_grad`` is correct everywhere.

    The caller owns the *class prior*: pass a shuffled, unweighted loader, not
    a CDWS-weighted balanced one, or the statistics are estimated under a
    re-weighted prior and applied under the natural one (§2.5.6(ii)). See
    :func:`~spectralquadnet.data.loaders.build_natural_prior_loader`.
    """
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.reset_running_stats()
            m.momentum = None
        elif isinstance(m, _STOCHASTIC_MODULES):
            m.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                x, _y, mask, morph = unpack_batch(batch, device)
                model(x, **side_inputs(mask, morph))
    finally:
        # Restores every module's training flag, including the ones forced to
        # eval above, so the model is left in a consistent state either way.
        model.eval()


def _pick_best_checkpoint(cfg: ExperimentConfig | Any, *ckpt_paths: str) -> str:
    """Select the checkpoint with the highest recorded ``val_f1`` across stages.

    For each path, ``val_f1`` is read from the JSON sidecar first (cheap);
    if absent, falls back to ``val_acc``, then to loading the full ``.pth``
    bundle, then to 0.0 if even that fails.

    Args:
        cfg: Composed experiment config, used to locate each path's sidecar.
        *ckpt_paths: Candidate checkpoint paths to rank.

    Returns:
        The path with the highest score, or the last path given if none
        exist on disk.
    """
    best_val, best_path = -1.0, ckpt_paths[-1]
    for p in ckpt_paths:
        if not os.path.isfile(p):
            continue
        try:
            sn = int(os.path.basename(p).replace("best_stage", "").replace(".pth", ""))
            meta = load_stage_meta(cfg, sn)
            v = meta.get("val_f1", meta.get("val_acc", None))
        except (ValueError, KeyError):
            v = None
        if v is None:
            try:
                v = torch.load(p, map_location="cpu", weights_only=False).get("val_f1", 0.0)
            except Exception:
                v = 0.0
        if v > best_val:
            best_val, best_path = v, p
    return best_path
