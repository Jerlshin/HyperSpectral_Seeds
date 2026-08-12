"""Worker residency: who keeps a pool alive, who gives one back, and what it costs.

``persistent_workers`` is the one DataLoader setting whose cost is a *process*
rather than a number, so it is invisible to every other test in this suite: a
loader that leaks its pool computes exactly the right answer while holding a
`spawn`-ed interpreter that will never serve another batch. On Metal those pages
come out of the same pool the activations do, and the profile that motivated
this file measured ~1.8 GB of resident RSS per worker pair against a machine
whose whole working set is 12.7 GB.

The two claims here are lifetime claims, and they point in opposite directions:

* An evaluation loader that is iterated **once** must not keep workers — the
  default, and what every throwaway loader in ``data/loaders.py`` gets.
* An evaluation loader that is iterated **twice per epoch for 600 epochs** must,
  because at ``persistent_workers=False`` each of those passes builds a worker
  pool and tears it down again. Measured on an M5 over the 1,294-patch
  validation split: **11.47 s per pass** non-persistent against **0.08 s**
  persistent, on six batches of actual work. That is not a tuning margin; it is
  the pass.

Neither can move a number. An evaluation dataset runs the ``none`` augmentation
profile, so ``RiceSeedDataset.__getitem__`` takes no draw from any RNG stream —
which is what makes worker count and worker residency unobservable in the data,
and it is asserted below rather than asserted about.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from spectralquadnet.data.datasets import RiceSeedDataset
from spectralquadnet.data.loaders import build_eval_loader, loader_options
from spectralquadnet.engine.stages.stage1_progressive import _release_stale_phase_loaders
from spectralquadnet.utils.device import RuntimePlan


def _plan(workers: int) -> RuntimePlan:
    return RuntimePlan(
        device=torch.device("cpu"),
        num_workers=workers,
        eval_num_workers=workers,
        pin_memory=False,
        persistent_workers=workers > 0,
        prefetch_factor=4,
        compile_enabled=False,
        compile_backend="inductor",
        compile_mode="default",
        channels_last=False,
        fused_optimizer=False,
        decompose_conv3d=False,
        checkpoint_branch_a=False,
        empty_cache_interval=0,
        diagnostics_interval=50,
        progress="off",
    )


class _AugCfg:
    max_cutout_bands = 2
    noise_std = 0.02
    cutmix_bands = 2
    cutmix_spatial = 4


class _Store:
    patches = np.zeros((8, 4, 8, 8), dtype=np.float32)
    labels = np.arange(8, dtype=np.int64) % 3
    masks = None
    morphology = None
    patches_path = None
    masks_path = None

    def require_patches(self):  # type: ignore[no-untyped-def]
        return self.patches

    def require_labels(self):  # type: ignore[no-untyped-def]
        return self.labels


# ══════════════════════════════════════════════════════════════════════
#  Residency is the caller's choice, and defaults to "no"
# ══════════════════════════════════════════════════════════════════════


def test_evaluation_loaders_are_throwaway_by_default() -> None:
    """The default has to stay "no resident pool": most eval loaders run once."""
    assert _plan(4).eval_loader_kwargs["persistent_workers"] is False
    assert loader_options(_plan(4), evaluation=True)["persistent_workers"] is False


def test_a_long_lived_evaluation_loader_can_keep_its_workers() -> None:
    """...and the val/calib loaders, which are not throwaway, must be able to opt in."""
    assert _plan(4).eval_loader_kwargs_for(persistent=True)["persistent_workers"] is True
    assert loader_options(_plan(4), evaluation=True, persistent=True)["persistent_workers"] is True


def test_residency_is_never_requested_without_workers_to_make_resident() -> None:
    """``persistent_workers`` at ``num_workers=0`` makes torch raise, not shrug."""
    assert "persistent_workers" not in _plan(0).eval_loader_kwargs_for(persistent=True)
    assert loader_options(None, evaluation=True, persistent=True) == {
        "num_workers": 0,
        "pin_memory": False,
    }


def test_training_loaders_ignore_the_evaluation_residency_flag() -> None:
    """``persistent`` is an evaluation concept; a train loader takes the plan's."""
    assert loader_options(_plan(4), evaluation=False, persistent=False)["persistent_workers"] is True


def test_build_eval_loader_threads_the_flag_through() -> None:
    """The keyword has to reach the DataLoader, not stop at the builder's signature."""
    dataset = RiceSeedDataset(
        np.arange(8), aug_strength="none", store=_Store(), data_cfg=_AugCfg(), device="cpu"
    )
    plan = _plan(2)
    assert build_eval_loader(dataset, plan=plan, persistent=True).persistent_workers is True
    assert build_eval_loader(dataset, plan=plan).persistent_workers is False


# ══════════════════════════════════════════════════════════════════════
#  Why residency cannot move a number
# ══════════════════════════════════════════════════════════════════════


def test_an_evaluation_dataset_draws_no_randomness() -> None:
    """The premise the whole change rests on, asserted rather than assumed.

    Worker count and worker residency change *which process* calls
    ``__getitem__`` and in what order the results are seeded — both of which are
    observable for a training dataset and neither of which is for an evaluation
    one, because the ``none`` profile takes no draw at all. If a profile ever
    grows an unconditional augmentation, this fails before any metric moves.
    """
    dataset = RiceSeedDataset(
        np.arange(8), aug_strength="none", store=_Store(), data_cfg=_AugCfg(), device="cpu"
    )
    assert dataset.profile is None

    torch.manual_seed(0)
    first = [dataset[i][0].clone() for i in range(len(dataset))]
    state_after = torch.random.get_rng_state()

    torch.manual_seed(0)
    torch.rand(17)  # advance the stream to somewhere else entirely
    second = [dataset[i][0].clone() for i in range(len(dataset))]

    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))
    torch.manual_seed(0)
    [dataset[i] for i in range(len(dataset))]
    assert torch.equal(torch.random.get_rng_state(), state_after), (
        "an evaluation pass consumed RNG, so worker layout is observable in the data"
    )


# ══════════════════════════════════════════════════════════════════════
#  Stage 1 gives a phase's pool back when the curriculum leaves it
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    ("phase", "expect_released", "expect_left"),
    [(1, [], [1, 2, 3]), (2, [1], [2, 3]), (3, [1, 2], [3])],
)
def test_stale_phase_loaders_are_released_at_the_boundary(
    phase: int, expect_released: list[int], expect_left: list[int]
) -> None:
    """Only phases strictly behind the curriculum go, and Phase 3's entry stays.

    Phase 3 keeps its entry because ``build_phase3_loader`` builds the
    oversampled loader over *that entry's dataset*; releasing it would pull the
    dataset out from under the loader actually in use.
    """
    loaders = {1: object(), 2: object(), 3: object()}
    released = _release_stale_phase_loaders(loaders, phase, torch.device("cpu"))  # type: ignore[arg-type]
    assert sorted(released) == expect_released
    assert sorted(loaders) == expect_left


def test_releasing_drops_the_last_reference_to_the_loader() -> None:
    """Popping the entry has to actually free the loader — that is the whole point.

    If anything else still held it, the worker pool would stay up and the
    release would be bookkeeping. A weak reference is the only way to assert
    "nothing holds this any more" without reaching into torch's shutdown path.
    """
    import weakref

    class _Loader:
        pass

    loaders = {1: _Loader(), 2: _Loader()}
    ref = weakref.ref(loaders[1])
    _release_stale_phase_loaders(loaders, 2, torch.device("cpu"))  # type: ignore[arg-type]
    assert ref() is None, "something other than the dict still holds the Phase-1 loader"
