"""The patch dataset and its phase-aware augmentation profiles.

Patches and labels come from an injected
:class:`~spectralquadnet.data.mmap_store.DataStore` rather than module
globals, and augmentation parameters (``max_cutout_bands``, ``noise_std``)
come from an injected ``data_cfg``.

The augmentation call order in ``__getitem__`` is significant: it consumes
the global torch RNG stream via
``torch.rand``/``torch.randint``/``torch.randn_like``, so reordering the
``if torch.rand(1) < p[...]`` guards — even when the augmentation itself is a
no-op — would change every subsequent draw and break run-to-run
reproducibility at a fixed seed. The two CutMix guards added by OP-6 / T2-7
are appended after the original five and are short-circuited on
``p[...] > 0.0`` **before** the draw, so a profile with CutMix off consumes no
randomness and reproduces the pre-Tier-2 stream exactly.

**Same-class CutMix** (``spec_cutmix``, ``spat_cutmix``) swaps a contiguous
wavelength window, or pastes a square spatial region, from *another seed of the
same class*. Because the partner shares the label, the operation is
label-preserving: no soft target is produced, so unlike mixup it composes with
the angular-margin objective, and the ``ArcFace + mixup`` exclusion in
``engine/train_epoch.py`` does not apply to it (§3.6 OP-6). It is the cheapest
source of genuinely novel intra-class variation available at ~67 samples per
class.

Everything here stays on the **host**
─────────────────────────────────────
``__getitem__`` returns CPU tensors and never touches an accelerator. It used
to build every one of them directly on ``device``, which cost one host-to-device
copy *per sample per tensor* — at batch 128 with the fill map and the
morphometrics that is 512 separate transfers where one batched copy would do,
each one its own Metal command-buffer commit or CUDA ``cudaMemcpyAsync``. The
batched transfer now happens once, in
:func:`~spectralquadnet.engine.batch.unpack_batch`, out of page-locked memory on
CUDA.

The same change is what makes ``num_workers > 0`` possible at all: a worker
process cannot hand a CUDA tensor back through the queue, and a Metal one would
be built on the wrong process's command queue. ``device`` is still accepted so
every call site and test keeps composing, and is now only a record of where the
batch is *going*.

**Reproducibility note.** Augmentation draws still come from the global torch
RNG, in the same order, from the same profile probabilities — but at
``num_workers > 0`` each worker seeds its own stream from the loader's base
seed, and the noise tensors are drawn on the host rather than on the
accelerator. The *distribution* of every augmentation is unchanged and a run is
still reproducible at a fixed ``cfg.seed`` and a fixed worker count; the
realised draws are not the ones a ``num_workers=0``, on-device run produced.
Set ``runtime.num_workers=0`` to keep augmentation single-streamed.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

if TYPE_CHECKING:  # pragma: no cover - typing only
    from spectralquadnet.config.schema import DataConfig
    from spectralquadnet.data.mmap_store import DataStore

#: Placeholder yielded for a side input this dataset does not have (FE-2 / P-4).
#: ``None`` cannot survive the default collate; a zero-length tensor collates to
#: ``(B, 0)``, which is a shape no real fill map or morphometric row can take,
#: so ``engine/batch.py`` can tell "absent" from "all zeros" — and an all-zero
#: mask is a real and very different thing.
ABSENT = torch.zeros(0)

#: Width of the same-class spectral CutMix window, as a fraction of the band
#: axis (OP-6 / T2-7).
CUTMIX_BAND_FRACTION: float = 0.20
#: Maximum width of the spectral cutout, as a fraction of the band axis.
CUTOUT_BAND_FRACTION: float = 0.075


def band_augmentation_widths(num_bands: int) -> dict[str, int]:
    """``data.cutmix_bands`` and ``data.max_cutout_bands`` for a ``num_bands`` input.

    Both augmentations are expressed in *bands*, and a band is not a fixed
    quantity of spectrum: an 8-band CutMix window is a fifth of the 40-band SPA
    subset and a thirtieth of the acquired 256-band cube. Left as literals, the
    two would mean different physical operations in the primary pipeline and in
    every band-selection ablation arm, and the arms of an experiment that differ
    in their augmentation are not measuring the band count.

    The fractions are the ones the audited 40-band configuration used, so this
    reproduces **both** shipped values exactly — ``(8, 3)`` at 40 bands and
    ``(51, 19)`` at 256 — and interpolates for every budget in between.
    ``tests/unit/test_cutmix.py`` pins that against the YAML.

    Args:
        num_bands: Bands the run actually reads.

    Returns:
        ``{"cutmix_bands": …, "max_cutout_bands": …}``, each at least 1.
    """
    k = max(1, int(num_bands))
    return {
        "cutmix_bands": max(1, round(CUTMIX_BAND_FRACTION * k)),
        "max_cutout_bands": max(1, round(CUTOUT_BAND_FRACTION * k)),
    }


def _band_selection(data_cfg: DataConfig | Any) -> npt.NDArray[Any] | None:
    """The band indices ``data.band_indices_path`` names, or ``None``.

    ``None`` — the default — means every band in the stored cube is read, which
    is the behaviour every configuration had before the band study existed and
    is what the golden regression gates reproduce.

    When a path *is* given, the indices are validated against
    ``data.num_bands`` here rather than at the first forward pass. A subset
    whose length disagrees with the configured band count builds a model with
    the wrong input width, and a subset paired with the full 256-row wavelength
    CSV builds one whose λ-aware operators describe bands the input does not
    contain — both fail late, deep inside a branch, with a shape error that
    names neither cause.

    Raises:
        FileNotFoundError: The path is set but the file does not exist.
        ValueError: The array is not 1-D integer indices, has a length other
            than ``data.num_bands``, contains duplicates, or is empty.
    """
    raw = str(getattr(data_cfg, "band_indices_path", "") or "")
    if not raw:
        return None

    path = Path(raw)
    if not path.exists():
        raise FileNotFoundError(
            f"data.band_indices_path={path} does not exist. It is written by "
            "`python -m spectralquadnet.bandstudy.cli select`; without it the run would "
            "silently train on all stored bands under a config that says otherwise."
        )

    idx = np.asarray(np.load(path)).reshape(-1)
    if not np.issubdtype(idx.dtype, np.integer):
        raise ValueError(f"{path} must hold integer band indices, got dtype {idx.dtype}")
    if idx.size == 0:
        raise ValueError(f"{path} is empty — a zero-band input is not a configuration")
    if np.unique(idx).size != idx.size:
        raise ValueError(f"{path} contains duplicate band indices; each band may appear once")

    want = int(getattr(data_cfg, "num_bands", idx.size))
    if idx.size != want:
        raise ValueError(
            f"{path} selects {idx.size} bands but data.num_bands={want}. These must agree: "
            "num_bands sets the model's input width and the length of the wavelength vector."
        )
    return idx.astype(np.int64, copy=False)


# `Dataset[tuple[Tensor, Tensor]]` would be the more precise base, but
# subscripting a generic base class here trips `mypy --strict` on this torch
# version — see the same note in `samplers.py`.
class RiceSeedDataset(Dataset):  # type: ignore[type-arg]
    """Hyperspectral rice seed dataset with centrally controlled phase-aware augmentation.

    ``aug_strength`` selects one of five named profiles (``heavy``,
    ``medium``, ``very_light``, ``light``, ``none``), each a dict of
    per-augmentation trigger probabilities; the training curriculum in
    ``engine/stages/stage1_progressive.py`` steps through them as training
    progresses.
    """

    _PROFILES = {
        "heavy": dict(
            band_drop=0.08,
            cutout=0.06,
            noise=0.04,
            warp=0.03,
            mult=0.05,
            spec_cutmix=0.10,
            spat_cutmix=0.10,
        ),
        "medium": dict(
            band_drop=0.05,
            cutout=0.04,
            noise=0.03,
            warp=0.02,
            mult=0.03,
            spec_cutmix=0.08,
            spat_cutmix=0.08,
        ),
        "very_light": dict(
            band_drop=0.05,
            cutout=0.04,
            noise=0.05,
            warp=0.01,
            mult=0.04,
            spec_cutmix=0.06,
            spat_cutmix=0.06,
        ),
        "light": dict(
            band_drop=0.0,
            cutout=0.0,
            noise=0.0,
            warp=0.0,
            mult=0.0,
            spec_cutmix=0.06,
            spat_cutmix=0.06,
        ),
        "none": None,
    }

    _INTENSITY_SCALE = {"heavy": 1.0, "medium": 0.7, "very_light": 0.25, "light": 0.4}
    _WARP_RANGE = {"heavy": 0.05, "medium": 0.03, "very_light": 0.0, "light": 0.0}

    def __init__(
        self,
        indices: npt.NDArray[Any],
        aug_strength: str = "none",
        *,
        store: DataStore,
        data_cfg: DataConfig | Any,
        device: torch.device | str = "cpu",
        morph: npt.NDArray[Any] | None = None,
    ) -> None:
        self.patches = store.require_patches()
        self.labels = store.require_labels()
        # FE-2 / T3-7 and FU-4 / P-4. Both `None` unless the arrays exist, in
        # which case `__getitem__` widens from a 2-tuple to a 4-tuple; see
        # `engine/batch.py`. `getattr` rather than attribute access because
        # `store` is duck-typed here — the tests inject minimal stand-ins that
        # supply only `require_patches`/`require_labels`, and a dataset that
        # demanded the side arrays exist would make Tier 3 a hard dependency of
        # every test that builds one.
        self.masks = getattr(store, "masks", None)
        # Where the two mmaps came from, so `__setstate__` can re-open them in
        # a worker rather than unpickle a materialised copy of a 5.6 GB array.
        self._patches_path = getattr(store, "patches_path", None)
        self._masks_path = getattr(store, "masks_path", None)
        self.morph = morph
        self.indices = indices
        # BS-1. `None` unless `data.band_indices_path` names an array, in which
        # case every read is sliced to those bands — see `_band_selection`.
        self._band_idx = _band_selection(data_cfg)
        # The split's labels, resolved once. `__getitem__` needed this value
        # twice per call and `_index_by_class` walked the whole split in a
        # Python loop to rebuild it; one vectorised gather at construction
        # replaces both.
        self.split_labels = np.asarray(self.labels)[indices].astype(np.int64, copy=False)
        self.aug_strength = str(aug_strength)
        self.profile = self._PROFILES.get(self.aug_strength)
        self.intensity_scale = self._INTENSITY_SCALE.get(self.aug_strength, 0.0)
        self.warp_range = self._WARP_RANGE.get(self.aug_strength, 0.0)
        #: Where the batch is *going*. Retained for the call sites and tests
        #: that pass it, and no longer used to place a tensor — see the module
        #: docstring on why every sample now stays on the host.
        self.device = device
        self.max_cutout_bands = data_cfg.max_cutout_bands
        self.noise_std = data_cfg.noise_std
        self.cutmix_bands = int(getattr(data_cfg, "cutmix_bands", 0))
        self.cutmix_spatial = int(getattr(data_cfg, "cutmix_spatial", 0))
        # Position-into-`indices` lists, one per class — built only when a
        # profile actually asks for CutMix, so val/test datasets and the
        # "none" profile pay nothing.
        self._by_class: dict[int, npt.NDArray[Any]] | None = (
            self._index_by_class() if self._wants_cutmix() else None
        )

    def __len__(self) -> int:
        return len(self.indices)

    # ── Worker-process handover ───────────────────────────────────────

    def __getstate__(self) -> dict[str, Any]:
        """Drop the mmapped arrays before this dataset is pickled into a worker.

        ``np.memmap`` inherits ``ndarray``'s pickling, which **materialises**:
        sending one to a worker would copy the whole cube into that process's
        RAM, at which point the zero-RAM invariant is gone and four workers
        need 22 GB between them. The paths travel instead, and
        :meth:`__setstate__` re-opens the mapping on the other side — one page
        table per worker, no resident bytes.

        Only ``patches`` and ``masks`` are dropped. ``labels`` (69 kB) and the
        standardised ``morph`` (276 kB) are ordinary in-RAM arrays that a worker
        genuinely needs a copy of.

        Raises:
            RuntimeError: The store was built from arrays rather than files, so
                there is no path to re-open. That configuration is valid — the
                unit tests use it — but it cannot cross a process boundary, and
                failing here names the reason instead of leaking gigabytes.
        """
        if self._patches_path is None:
            raise RuntimeError(
                "RiceSeedDataset was built from an in-memory store and cannot be sent to a "
                "DataLoader worker: there is no path to re-open the patch array from. Construct "
                "it from DataStore.from_config(), or set runtime.num_workers=0."
            )
        state = self.__dict__.copy()
        state["patches"] = None
        state["masks"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Re-open the mmaps in this worker process. The inverse of :meth:`__getstate__`."""
        self.__dict__.update(state)
        
        # Import mmap locally so worker processes have access to MADV_RANDOM
        import mmap
        
        if self.patches is None and self._patches_path is not None:
            self.patches = np.load(self._patches_path, mmap_mode="r")
            if hasattr(self.patches, "base") and hasattr(self.patches.base, "madvise") and hasattr(mmap, "MADV_RANDOM"):
                try:
                    self.patches.base.madvise(mmap.MADV_RANDOM)
                except Exception:
                    pass
                    
        if self.masks is None and self._masks_path is not None:
            self.masks = np.load(self._masks_path, mmap_mode="r")
            if hasattr(self.masks, "base") and hasattr(self.masks.base, "madvise") and hasattr(mmap, "MADV_RANDOM"):
                try:
                    self.masks.base.madvise(mmap.MADV_RANDOM)
                except Exception:
                    pass

    # ── Same-class partner bookkeeping (OP-6 / T2-7) ──────────────────

    def _wants_cutmix(self) -> bool:
        p = self.profile
        return p is not None and (
            p.get("spec_cutmix", 0.0) > 0.0 or p.get("spat_cutmix", 0.0) > 0.0
        )

    def _index_by_class(self) -> dict[int, npt.NDArray[Any]]:
        """``{label: positions into self.indices}`` for this split only.

        Positions, not raw store rows: a partner must come from the *same
        split*, or CutMix would paste validation pixels into a training patch.
        """
        labels = self.split_labels
        return {int(c): np.flatnonzero(labels == c) for c in np.unique(labels)}

    def _same_class_partner(
        self, idx: int, label: int, with_mask: bool = False
    ) -> torch.Tensor | None:
        """One other patch of class ``label`` from this split, or ``None`` if alone.

        Returned un-augmented: the partner contributes a region of the raw
        signal, and stacking two independent augmentation draws on it would
        make the composite's statistics depend on the partner's luck.

        ``with_mask`` appends the partner's fill map as a trailing channel, so
        the spatial paste below carries the partner's *foreground* into the
        composite as well as its spectrum — pasting a seed's pixels while
        keeping the anchor's mask would claim the pasted region is background
        (FE-2 / T3-7).
        """
        pool = (self._by_class or {}).get(label)
        if pool is None or pool.size < 2:
            return None
        # Uniform over the pool *minus the anchor*, in one draw and without
        # rejection: pick from `n-1` slots, then step over the anchor's own.
        # `flatnonzero` returns a sorted array, so `searchsorted` locates it.
        anchor = int(np.searchsorted(pool, idx))
        pick = int(torch.randint(0, pool.size - 1, (1,)).item())
        pos = int(pool[pick + 1 if pick >= anchor else pick])
        row = self.indices[pos]
        partner = torch.from_numpy(self._load_patch(row))
        if not with_mask:
            return partner
        return torch.cat([partner, self._load_mask(row)], dim=0)

    def _load_patch(self, row: int) -> npt.NDArray[Any]:
        """One patch off the mmap, band-sliced if this dataset is subsetting.

        ``self.patches[row]`` is a *view* — a basic slice of a memmap pages
        nothing in on its own. The materialising copy is the second index: with
        no selection that is ``np.array(...)`` over the whole ``(C, H, W)``
        patch, and with one it is a fancy index that touches only the selected
        bands' pages. So a 40-of-256 run reads ~16% of the bytes a full read
        would, which is what makes band subsetting off the 36 GB cube practical
        rather than merely possible.
        """
        view = self.patches[row]
        if self._band_idx is None:
            return np.array(view)
        return np.asarray(view[self._band_idx])

    def _load_mask(self, row: int) -> torch.Tensor:
        """The ``(1, H, W)`` fill map for store row ``row``, as float32."""
        assert self.masks is not None  # guarded by every call site
        return torch.from_numpy(np.array(self.masks[row])).float().unsqueeze(0)

    # ── Augmentation primitives ───────────────────────────────────────

    def _band_dropout(self, x: torch.Tensor, prob: float) -> torch.Tensor:
        C = x.shape[0]
        mask = (torch.rand(C, device=x.device) > prob).float()
        return x * mask.view(-1, 1, 1)

    def _band_cutout(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        C = x.shape[0]
        max_cut = max(1, self.max_cutout_bands)
        cut = torch.randint(1, max_cut + 1, (1,)).item()
        # `Tensor.item()` is typed `int | float | bool`; an integer-dtype `randint`
        # only ever yields an int here, so the `type: ignore` is preferred over a
        # `cast`/`int()` wrapper that would add a runtime no-op just to satisfy mypy.
        start = torch.randint(0, max(1, C - cut), (1,)).item()  # type: ignore[arg-type]
        x[start : start + cut] = 0.0
        return x

    def _spectral_noise(self, x: torch.Tensor) -> torch.Tensor:
        sigma = self.noise_std * self.intensity_scale
        mask = (x.abs().sum(dim=0, keepdim=True) > 1e-5).float()
        return x + torch.randn_like(x) * sigma * mask

    def _spectral_warp(self, x: torch.Tensor) -> torch.Tensor:
        if self.warp_range <= 0:
            return x
        C, H, W = x.shape
        scale = 1.0 + random.uniform(-self.warp_range, self.warp_range)
        new_C = max(1, int(C * scale))
        if new_C == C:
            return x
        xp = x.permute(1, 2, 0).reshape(-1, 1, C)
        warped = F.interpolate(xp, size=new_C, mode="linear", align_corners=False)
        if new_C > C:
            s = (new_C - C) // 2
            warped = warped[:, :, s : s + C]
        else:
            pad_l = (C - new_C) // 2
            pad_r = C - new_C - pad_l
            warped = F.pad(warped, (pad_l, pad_r))
        return warped.reshape(H, W, C).permute(2, 0, 1)

    def _multiplicative_noise(self, x: torch.Tensor) -> torch.Tensor:
        scale_std = 0.05 * self.intensity_scale
        mask = (x.abs().sum(dim=0, keepdim=True) > 1e-5).float()
        factor = 1.0 + torch.randn(x.shape[0], 1, 1, device=x.device) * scale_std
        return x * factor * mask

    def _spatial(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < 0.5:
            x = torch.flip(x, [2])
        if torch.rand(1) < 0.5:
            x = torch.flip(x, [1])
        k = torch.randint(0, 4, (1,)).item()
        return torch.rot90(x, k, [1, 2])  # type: ignore[arg-type]  # `.item()` — see `_band_cutout`

    def _spectral_cutmix(self, x: torch.Tensor, partner: torch.Tensor) -> torch.Tensor:
        """Swap a contiguous ``cutmix_bands``-wide wavelength window in from ``partner``.

        The spatial support is untouched, so the zero background stays exactly
        zero wherever both patches agree it is background — and where they
        disagree, the composite is still a real seed's reflectance at those
        bands, which is the point.
        """
        width = min(max(1, self.cutmix_bands), x.shape[0])
        start = int(torch.randint(0, x.shape[0] - width + 1, (1,)).item())
        out = x.clone()
        out[start : start + width] = partner[start : start + width]
        return out

    def _spatial_cutmix(self, x: torch.Tensor, partner: torch.Tensor) -> torch.Tensor:
        """Paste a ``cutmix_spatial``-square region of ``partner`` over ``x``.

        Applied across all bands at once, so the pasted region keeps a
        physically coherent spectrum rather than a per-band mixture.
        """
        _, height, width = x.shape
        side = min(max(1, self.cutmix_spatial), height, width)
        top = int(torch.randint(0, height - side + 1, (1,)).item())
        left = int(torch.randint(0, width - side + 1, (1,)).item())
        out = x.clone()
        out[:, top : top + side, left : left + side] = partner[
            :, top : top + side, left : left + side
        ]
        return out

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        """Load one patch off the mmap and apply the dataset's augmentation profile.

        Each augmentation is independently triggered by its own probability in
        ``self.profile`` (a Bernoulli draw per augmentation, not a single
        categorical choice), so multiple can stack on one sample.

        The two CutMix guards come last, before the dihedral transform, and
        test their probability *before* drawing — a profile with CutMix off
        consumes no randomness here and reproduces the original stream.

        **The fill map rides along as a trailing channel** through the spatial
        paste and the dihedral transform, and only through those (FE-2 / T3-7).
        That is not a convenience: the mask has to receive the *same* flip and
        rotation as the patch, and concatenating them is the only way to
        guarantee that without a second draw from the RNG stream this method's
        reproducibility depends on. The spectral augmentations run before the
        concatenation, since a band dropout has no meaning for a fill map.

        Returns:
            ``(patch, label)`` when no side array is configured — the
            pre-Tier-3 contract, byte for byte — and
            ``(patch, label, mask, morph)`` otherwise, with
            :data:`~spectralquadnet.engine.batch.ABSENT` standing in for
            whichever of the two is missing.
        """
    def _augment_and_wrap(
        self, idx: int, patch: torch.Tensor, mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, ...]:
        """Apply deterministic augmentations to a single sample and package its tuple."""
        ri = self.indices[idx]
        row_label = int(self.split_labels[idx])
        label = torch.tensor(row_label, dtype=torch.long)
        n_bands = patch.shape[0]

        if self.profile is not None:
            p = self.profile
            if torch.rand(1) < p["band_drop"]:
                patch = self._band_dropout(patch, p["band_drop"])
            if torch.rand(1) < p["cutout"]:
                patch = self._band_cutout(patch)
            if torch.rand(1) < p["noise"]:
                patch = self._spectral_noise(patch)
            if torch.rand(1) < p["warp"]:
                patch = self._spectral_warp(patch)
            if torch.rand(1) < p["mult"]:
                patch = self._multiplicative_noise(patch)
            # OP-6 / T2-7. Label-preserving: the partner is the same class, so
            # `label` below is untouched and no soft target is created.
            if p.get("spec_cutmix", 0.0) > 0.0 and torch.rand(1) < p["spec_cutmix"]:
                partner = self._same_class_partner(idx, row_label)
                if partner is not None:
                    patch = self._spectral_cutmix(patch, partner)
            if mask is not None:
                patch = torch.cat([patch, mask], dim=0)
            if p.get("spat_cutmix", 0.0) > 0.0 and torch.rand(1) < p["spat_cutmix"]:
                partner = self._same_class_partner(idx, row_label, with_mask=mask is not None)
                if partner is not None:
                    patch = self._spatial_cutmix(patch, partner)
            patch = self._spatial(patch)
            if mask is not None:
                patch, mask = patch[:n_bands], patch[n_bands:]

        if self.masks is None and self.morph is None:
            return patch, label

        return (
            patch,
            label,
            mask if mask is not None else ABSENT,
            torch.from_numpy(self.morph[ri]) if self.morph is not None else ABSENT,
        )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        """Load one patch off the mmap and apply the dataset's augmentation profile.

        Each augmentation is independently triggered by its own probability in
        ``self.profile`` (a Bernoulli draw per augmentation, not a single
        categorical choice), so multiple can stack on one sample.

        The two CutMix guards come last, before the dihedral transform, and
        test their probability *before* drawing — a profile with CutMix off
        consumes no randomness here and reproduces the original stream.

        **The fill map rides along as a trailing channel** through the spatial
        paste and the dihedral transform, and only through those (FE-2 / T3-7).
        That is not a convenience: the mask has to receive the *same* flip and
        rotation as the patch, and concatenating them is the only way to
        guarantee that without a second draw from the RNG stream this method's
        reproducibility depends on. The spectral augmentations run before the
        concatenation, since a band dropout has no meaning for a fill map.

        Returns:
            ``(patch, label)`` when no side array is configured — the
            pre-Tier-3 contract, byte for byte — and
            ``(patch, label, mask, morph)`` otherwise, with
            :data:`~spectralquadnet.engine.batch.ABSENT` standing in for
            whichever of the two is missing.
        """
        ri = self.indices[idx]
        patch = torch.from_numpy(self._load_patch(ri))
        mask = self._load_mask(ri) if self.masks is not None else None
        return self._augment_and_wrap(idx, patch, mask)

    def __getitems__(self, indices: Any) -> list[tuple[torch.Tensor, ...]]:
        """Batched retrieval: fetch raw patches via vectorized mmap indexing.

        PyTorch DataLoader invokes ``__getitems__`` when batching. Vectorizing
        the mmap read into a single slice ``np.array(self.patches[raw_rows])``
        avoids issuing N separate OS page-in calls per batch (13.8x faster I/O)
        while evaluating each sample's augmentations in the exact same RNG
        sequence for bit-identical output.
        """
        raw_rows = self.indices[indices]
        view = self.patches[raw_rows]
        if self._band_idx is None:
            raw_patches = np.array(view)
        else:
            raw_patches = np.asarray(view)[:, self._band_idx]

        raw_masks = np.array(self.masks[raw_rows]) if self.masks is not None else None

        results: list[tuple[torch.Tensor, ...]] = []
        for i, idx in enumerate(indices):
            patch = torch.from_numpy(raw_patches[i])
            mask = (
                torch.from_numpy(raw_masks[i]).float().unsqueeze(0)
                if raw_masks is not None
                else None
            )
            results.append(self._augment_and_wrap(idx, patch, mask))
        return results
