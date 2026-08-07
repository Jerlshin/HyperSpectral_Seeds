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
"""

from __future__ import annotations

import random
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
        device: torch.device | str,
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
        self.morph = morph
        self.indices = indices
        self.aug_strength = str(aug_strength)
        self.profile = self._PROFILES.get(self.aug_strength)
        self.intensity_scale = self._INTENSITY_SCALE.get(self.aug_strength, 0.0)
        self.warp_range = self._WARP_RANGE.get(self.aug_strength, 0.0)
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
        labels = np.asarray([int(self.labels[r]) for r in self.indices])
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
        partner = torch.from_numpy(np.array(self.patches[row])).to(self.device, non_blocking=True)
        if not with_mask:
            return partner
        return torch.cat([partner, self._load_mask(row)], dim=0)

    def _load_mask(self, row: int) -> torch.Tensor:
        """The ``(1, H, W)`` fill map for store row ``row``, as float32."""
        assert self.masks is not None  # guarded by every call site
        return (
            torch.from_numpy(np.array(self.masks[row]))
            .float()
            .unsqueeze(0)
            .to(self.device, non_blocking=True)
        )

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
        ri = self.indices[idx]

        # np.array(...) copies exactly one patch off the mmap. Do not replace
        # with a slice or a batched copy — that is what bounds RAM.
        patch_np = np.array(self.patches[ri])
        patch = torch.from_numpy(patch_np).to(self.device, non_blocking=True)
        label = torch.tensor(int(self.labels[ri]), dtype=torch.long, device=self.device)
        n_bands = patch.shape[0]
        mask = self._load_mask(ri) if self.masks is not None else None

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
                partner = self._same_class_partner(idx, int(self.labels[ri]))
                if partner is not None:
                    patch = self._spectral_cutmix(patch, partner)
            if mask is not None:
                patch = torch.cat([patch, mask], dim=0)
            if p.get("spat_cutmix", 0.0) > 0.0 and torch.rand(1) < p["spat_cutmix"]:
                partner = self._same_class_partner(
                    idx, int(self.labels[ri]), with_mask=mask is not None
                )
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
            (
                torch.from_numpy(self.morph[ri]).to(self.device, non_blocking=True)
                if self.morph is not None
                else ABSENT
            ),
        )
