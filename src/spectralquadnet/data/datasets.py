"""The patch dataset and its phase-aware augmentation profiles.

Relocated from ``HSI_modality_training/hsi_training.py`` @ ``886560f``:

=============================  ==============
Symbol                         Baseline lines
=============================  ==============
:class:`RiceSeedDataset`       253-356
=============================  ==============

Three declared deviations, all mechanical (REFACTOR_PLAN.md §5 Phase 2 —
"verbatim function bodies; only ``CONFIG`` dict access becomes config-object
field access"):

* ``_GPU_PATCHES`` / ``_GLOBAL_LABELS`` globals → the injected
  :class:`~spectralquadnet.data.mmap_store.DataStore`.
* ``CONFIG["max_cutout_bands"]`` / ``CONFIG["noise_std"]`` → ``data_cfg`` fields,
  read once in ``__init__`` and cached on ``self``.
* ``CONFIG["device"]`` → the injected ``device``.

The augmentation call order in ``__getitem__`` is unchanged. It consumes the
global torch RNG stream via ``torch.rand``/``torch.randint``/``torch.randn_like``,
so reordering the five ``if torch.rand(1) < p[...]`` guards — even when the
augmentation itself is a no-op — would change every subsequent draw
(REFACTOR_PLAN.md §3.6).
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

if TYPE_CHECKING:  # pragma: no cover - typing only
    from spectralquadnet.config.schema import DataConfig
    from spectralquadnet.data.mmap_store import DataStore


class RiceSeedDataset(Dataset):
    """
    Hyperspectral Rice Seed Dataset with centrally controlled
    phase-aware spectral + spatial augmentation.
    """

    _PROFILES = {
        "heavy": dict(band_drop=0.08, cutout=0.06, noise=0.04, warp=0.03, mult=0.05),
        "medium": dict(band_drop=0.05, cutout=0.04, noise=0.03, warp=0.02, mult=0.03),
        "very_light": dict(band_drop=0.05, cutout=0.04, noise=0.05, warp=0.01, mult=0.04),
        "light": dict(band_drop=0.0, cutout=0.0, noise=0.0, warp=0.0, mult=0.0),
        "none": None,
    }

    _INTENSITY_SCALE = {"heavy": 1.0, "medium": 0.7, "very_light": 0.25, "light": 0.4}
    _WARP_RANGE = {"heavy": 0.05, "medium": 0.03, "very_light": 0.0, "light": 0.0}

    def __init__(
        self,
        indices: np.ndarray,
        aug_strength: str = "none",
        *,
        store: DataStore,
        data_cfg: DataConfig | Any,
        device: torch.device | str,
    ) -> None:
        self.patches = store.require_patches()
        self.labels = store.require_labels()
        self.indices = indices
        self.aug_strength = str(aug_strength)
        self.profile = self._PROFILES.get(self.aug_strength)
        self.intensity_scale = self._INTENSITY_SCALE.get(self.aug_strength, 0.0)
        self.warp_range = self._WARP_RANGE.get(self.aug_strength, 0.0)
        self.device = device
        self.max_cutout_bands = data_cfg.max_cutout_bands
        self.noise_std = data_cfg.noise_std

    def __len__(self) -> int:
        return len(self.indices)

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
        start = torch.randint(0, max(1, C - cut), (1,)).item()
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
        return torch.rot90(x, k, [1, 2])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ri = self.indices[idx]

        # §3.4: np.array(...) copies exactly one patch off the mmap. Do not
        # replace with a slice or a batched copy — that is what bounds RAM.
        patch_np = np.array(self.patches[ri])
        patch = torch.from_numpy(patch_np).to(self.device, non_blocking=True)
        label = torch.tensor(int(self.labels[ri]), dtype=torch.long, device=self.device)

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
            patch = self._spatial(patch)

        return patch, label
