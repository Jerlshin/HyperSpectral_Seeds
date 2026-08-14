"""Same-class spectral and spatial CutMix (**T2-7** / OP-6).

Mixup and an angular margin are mutually exclusive: the margin is indexed by a
single label, and mixup produces two. The documents note that and then leave a
gap — Stage 2 and Stage 3 train with no mixing regulariser at all, on ~67
samples per class.

Same-class CutMix closes it without touching the label. Swapping a contiguous
wavelength window, or pasting a square region, between two seeds *of the same
class* produces genuinely novel intra-class variation whose label is the label
it started with. T2-7's stated criterion is exactly that: "the new augmentation
is label-preserving, so the ArcFace ``ValueError`` guard is untouched".

The other property under test is RNG discipline. ``__getitem__`` consumes the
global torch stream, so a new draw anywhere in it shifts every subsequent one.
The two CutMix guards short-circuit on their probability *before* drawing, so a
profile with CutMix off reproduces the pre-Tier-2 stream exactly.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from spectralquadnet.data.datasets import RiceSeedDataset
from spectralquadnet.engine.train_epoch import train_one_epoch

pytestmark = pytest.mark.regression

BANDS, SPATIAL = 40, 16
PER_CLASS, CLASSES = 6, 4


class FakeStore:
    """A ``DataStore``-shaped object over in-memory arrays.

    Each patch is filled with its own row index, so any pixel in a composite
    names the seed it came from — which is what makes "did a partner region
    actually land here" a decidable question rather than a statistical one.
    """

    def __init__(self) -> None:
        n = PER_CLASS * CLASSES
        self.patches = np.zeros((n, BANDS, SPATIAL, SPATIAL), dtype=np.float32)
        for i in range(n):
            self.patches[i] = float(i)
        self.labels = np.repeat(np.arange(CLASSES), PER_CLASS).astype(np.int64)

    def require_patches(self) -> np.ndarray:
        return self.patches

    def require_labels(self) -> np.ndarray:
        return self.labels


def data_cfg(**overrides) -> SimpleNamespace:
    base = dict(max_cutout_bands=3, noise_std=0.02, cutmix_bands=8, cutmix_spatial=6)
    return SimpleNamespace(**{**base, **overrides})


def make_dataset(profile: str, **cfg_overrides) -> RiceSeedDataset:
    store = FakeStore()
    return RiceSeedDataset(
        np.arange(len(store.labels)),
        aug_strength=profile,
        store=store,
        data_cfg=data_cfg(**cfg_overrides),
        device="cpu",
    )


# ══════════════════════════════════════════════════════════════════════
#  The two operators
# ══════════════════════════════════════════════════════════════════════


def test_spectral_cutmix_swaps_a_contiguous_band_window() -> None:
    """A window of exactly ``cutmix_bands`` bands, contiguous, and nothing else."""
    dataset = make_dataset("light", cutmix_bands=8)
    anchor = torch.zeros(BANDS, SPATIAL, SPATIAL)
    partner = torch.ones(BANDS, SPATIAL, SPATIAL)

    torch.manual_seed(0)
    mixed = dataset._spectral_cutmix(anchor, partner)

    from_partner = (mixed.reshape(BANDS, -1) == 1.0).all(dim=1)
    assert int(from_partner.sum()) == 8
    swapped = from_partner.nonzero().flatten()
    assert torch.equal(swapped, torch.arange(int(swapped[0]), int(swapped[0]) + 8))


def test_spatial_cutmix_pastes_one_square_across_every_band() -> None:
    """A ``k x k`` region, the same region in every band — a coherent spectrum, not a mixture."""
    dataset = make_dataset("light", cutmix_spatial=6)
    anchor = torch.zeros(BANDS, SPATIAL, SPATIAL)
    partner = torch.ones(BANDS, SPATIAL, SPATIAL)

    torch.manual_seed(0)
    mixed = dataset._spatial_cutmix(anchor, partner)

    assert int(mixed.sum()) == 6 * 6 * BANDS
    per_band = (mixed == 1.0).sum(dim=(1, 2))
    assert torch.equal(per_band, torch.full((BANDS,), 36))


def test_the_operators_do_not_mutate_the_anchor() -> None:
    """``__getitem__`` reuses the tensor it loaded; an in-place paste would corrupt the mmap copy."""
    dataset = make_dataset("light")
    anchor = torch.zeros(BANDS, SPATIAL, SPATIAL)
    partner = torch.ones(BANDS, SPATIAL, SPATIAL)

    torch.manual_seed(0)
    dataset._spectral_cutmix(anchor, partner)
    dataset._spatial_cutmix(anchor, partner)

    assert not anchor.any()


def test_a_window_wider_than_the_cube_is_clamped() -> None:
    """``cutmix_bands >= num_bands`` must degenerate to "take the partner", not index out."""
    dataset = make_dataset("light", cutmix_bands=999, cutmix_spatial=999)
    anchor = torch.zeros(BANDS, SPATIAL, SPATIAL)
    partner = torch.ones(BANDS, SPATIAL, SPATIAL)

    torch.manual_seed(0)
    assert dataset._spectral_cutmix(anchor, partner).eq(1.0).all()
    assert dataset._spatial_cutmix(anchor, partner).eq(1.0).all()


# ══════════════════════════════════════════════════════════════════════
#  The partner is always same-class, and never the anchor
# ══════════════════════════════════════════════════════════════════════


def test_the_partner_is_drawn_from_the_anchors_own_class() -> None:
    dataset = make_dataset("heavy")
    torch.manual_seed(0)

    for idx in range(len(dataset)):
        expected = int(dataset.labels[dataset.indices[idx]])
        for _ in range(20):
            partner = dataset._same_class_partner(idx, expected)
            assert partner is not None
            assert int(dataset.labels[int(partner.flatten()[0])]) == expected


def test_the_partner_is_never_the_anchor_itself() -> None:
    """CutMix with yourself is a no-op that still costs a patch read and an RNG draw."""
    dataset = make_dataset("heavy")
    torch.manual_seed(1)

    for idx in range(len(dataset)):
        for _ in range(30):
            partner = dataset._same_class_partner(idx, int(dataset.labels[dataset.indices[idx]]))
            assert int(partner.flatten()[0]) != dataset.indices[idx]


def test_every_other_member_of_the_class_is_reachable() -> None:
    """The one-draw "step over the anchor" trick must stay uniform, not just legal."""
    dataset = make_dataset("heavy")
    torch.manual_seed(2)

    seen = {int(dataset._same_class_partner(0, 0).flatten()[0]) for _ in range(300)}

    assert seen == set(range(1, PER_CLASS))


def test_a_singleton_class_declines_rather_than_deadlocks() -> None:
    """With one sample of a class there is no partner; the augmentation must skip."""
    store = FakeStore()
    dataset = RiceSeedDataset(
        np.array([0, PER_CLASS]),  # one sample each of classes 0 and 1
        aug_strength="heavy",
        store=store,
        data_cfg=data_cfg(),
        device="cpu",
    )

    assert dataset._same_class_partner(0, 0) is None


def test_the_partner_pool_is_scoped_to_this_split() -> None:
    """A train-split dataset must never reach a val-split row."""
    store = FakeStore()
    train_rows = np.array([0, 1, 2])
    dataset = RiceSeedDataset(
        train_rows, aug_strength="heavy", store=store, data_cfg=data_cfg(), device="cpu"
    )
    torch.manual_seed(3)

    reached = {int(dataset._same_class_partner(0, 0).flatten()[0]) for _ in range(100)}

    assert reached <= set(train_rows.tolist())


# ══════════════════════════════════════════════════════════════════════
#  Label preservation — T2-7's stated criterion
# ══════════════════════════════════════════════════════════════════════


def test_cutmix_never_changes_the_label() -> None:
    """The whole reason this composes with an angular margin and mixup does not."""
    dataset = make_dataset("heavy")
    torch.manual_seed(4)

    for idx in range(len(dataset)):
        _, label = dataset[idx]
        assert int(label) == int(dataset.labels[dataset.indices[idx]])


def test_the_arcface_mixup_guard_is_untouched(cfg, physical_wl) -> None:
    """The guard still fires for mixup, which is what "untouched" has to mean.

    T2-7's criterion is that CutMix does not need the exclusion relaxed. The
    exclusion itself moved with HD-1 — from "which head is selected" to "is the
    margin non-zero" — because under one head the question "is this ArcFace" no
    longer has an answer, but its behaviour for mixup is the same.
    """
    from spectralquadnet.models.spectral_quadnet import SpectralQuadNet

    torch.manual_seed(0)
    model = SpectralQuadNet.from_config(cfg, physical_wl)

    with pytest.raises(ValueError, match="Mixup"):
        train_one_epoch(
            cfg,
            model,
            [],
            torch.optim.AdamW(model.parameters()),
            torch.nn.CrossEntropyLoss(),
            None,
            None,
            torch.device("cpu"),
            use_mixup=True,
            arc_m=0.35,
        )


def test_a_zero_margin_admits_mixup(cfg, physical_wl) -> None:
    """The other side of the same guard: Stage 1's cosine head takes soft targets fine."""
    from spectralquadnet.models.spectral_quadnet import SpectralQuadNet

    torch.manual_seed(0)
    model = SpectralQuadNet.from_config(cfg, physical_wl)

    loss, acc = train_one_epoch(
        cfg,
        model,
        [],
        torch.optim.AdamW(model.parameters()),
        torch.nn.CrossEntropyLoss(),
        None,
        None,
        torch.device("cpu"),
        use_mixup=True,
        arc_m=0.0,
    )

    assert (loss, acc) == (0.0, 0.0), "an empty loader should return zeros, not raise"


# ══════════════════════════════════════════════════════════════════════
#  RNG discipline
# ══════════════════════════════════════════════════════════════════════


def test_a_profile_without_cutmix_draws_nothing_extra() -> None:
    """The short-circuit: ``p > 0.0`` is tested before the Bernoulli draw.

    A ``torch.rand(1)`` that is drawn and then compared against zero would
    still never trigger CutMix while shifting every downstream draw — and with
    it the whole augmentation stream the archived run was produced under.
    """
    off = make_dataset("none")
    plain = RiceSeedDataset(
        np.arange(PER_CLASS * CLASSES),
        aug_strength="light",
        store=FakeStore(),
        data_cfg=data_cfg(),
        device="cpu",
    )
    plain.profile = {**plain.profile, "spec_cutmix": 0.0, "spat_cutmix": 0.0}
    plain._by_class = None

    torch.manual_seed(7)
    plain[0]
    after_plain = torch.random.get_rng_state()

    torch.manual_seed(7)
    off_profile = dict(plain.profile)
    off_profile.pop("spec_cutmix")
    off_profile.pop("spat_cutmix")
    plain.profile = off_profile
    plain[0]

    assert torch.equal(torch.random.get_rng_state(), after_plain)
    assert off.profile is None


def test_the_shipped_profiles_all_enable_cutmix() -> None:
    """The augmentation has to actually be on, or T2-7 is a no-op with tests."""
    for name in ("heavy", "medium", "very_light", "light"):
        profile = RiceSeedDataset._PROFILES[name]
        assert profile["spec_cutmix"] > 0.0, name
        assert profile["spat_cutmix"] > 0.0, name


def test_the_class_index_is_only_built_when_it_is_needed() -> None:
    """Val/test datasets and the ``none`` profile must not pay for the bookkeeping."""
    assert make_dataset("none")._by_class is None
    assert make_dataset("heavy")._by_class is not None


# ══════════════════════════════════════════════════════════════════════
#  Band-count scaling — the augmentation must mean one thing across the grid
# ══════════════════════════════════════════════════════════════════════


def test_the_band_widths_are_a_fixed_share_of_the_spectral_axis() -> None:
    """A CutMix window is a share of the spectrum, not a count of bands.

    Left as literals, ``cutmix_bands = 8`` is a fifth of the 40-band SPA subset
    and a thirtieth of the acquired 256-band cube — so the primary pipeline and
    every band-selection ablation arm would be running a different augmentation
    while claiming to differ only in the band count.
    """
    from spectralquadnet.data.datasets import band_augmentation_widths

    assert band_augmentation_widths(40) == {"cutmix_bands": 8, "max_cutout_bands": 3}
    assert band_augmentation_widths(256) == {"cutmix_bands": 51, "max_cutout_bands": 19}

    # Monotone, and never degenerate: a window of 0 bands is not an augmentation
    # and a window of every band is a relabelling.
    for k in (1, 2, 3, 5, 8, 16, 32, 64, 100, 128, 256):
        widths = band_augmentation_widths(k)
        assert 1 <= widths["cutmix_bands"] <= k, k
        assert 1 <= widths["max_cutout_bands"] <= k, k


@pytest.mark.parametrize(
    ("config", "bands"),
    [
        ("data=hsi256_grouped", 256),
        ("data=hsi256_stratified", 256),
        ("data=ablation/spa40_grouped", 40),
        ("data=ablation/spa40_stratified", 40),
        ("data=ablation/spa40_audited", 40),
    ],
)
def test_every_shipped_data_config_uses_the_scaled_widths(config: str, bands: int) -> None:
    """The YAML and the rule are checked against each other, in both directions."""
    from spectralquadnet.config.compose import load_experiment_config
    from spectralquadnet.data.datasets import band_augmentation_widths

    cfg = load_experiment_config(overrides=[config])
    assert cfg.data.num_bands == bands
    expected = band_augmentation_widths(bands)
    assert cfg.data.cutmix_bands == expected["cutmix_bands"], config
    assert cfg.data.max_cutout_bands == expected["max_cutout_bands"], config
