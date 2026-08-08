"""The throughput refactor's contract: faster, and *identical*.

Every optimisation on the training loop's hot path replaced a Python-level loop
with a multi-tensor or vectorised equivalent. "Equivalent" is a claim about
floating-point arithmetic, not about mathematics — ``a*b + c*d`` and
``fma(a, b, c*d)`` are the same function and different floats — so each rewrite
is pinned here against a verbatim copy of the code it replaced, and the
assertion is ``torch.equal``, not ``allclose``.

The reference implementations below are deliberately duplicated rather than
imported. They are the *old* code; if they were shared with the new code the
test would compare something to itself.

The second half pins the behaviours the new execution paths have to have —
worker-safe pickling, refused uneven DDP shards, wrapper-transparent
state_dicts — which are correctness properties rather than numerical ones.
"""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from spectralquadnet.data.loaders import _shard_batch, loader_options
from spectralquadnet.engine.diagnostics import (
    BRANCH_PREFIXES,
    branch_grad_norm_tensors,
    should_render_details,
)
from spectralquadnet.losses.contrastive import ProtoNCELoss
from spectralquadnet.models.ema import ModelEMA
from spectralquadnet.models.stats_ops import extract_grid_spectra, extract_grid_spectra_multi
from spectralquadnet.optim.sam import SAM
from spectralquadnet.utils.device import RuntimePlan, unwrap_model
from spectralquadnet.utils.distributed import DistContext

SEED = 7


def _mlp() -> nn.Module:
    torch.manual_seed(SEED)
    return nn.Sequential(nn.Linear(48, 64), nn.GELU(), nn.BatchNorm1d(64), nn.Linear(64, 12))


# ══════════════════════════════════════════════════════════════════════
#  Numerical equivalence against the code each rewrite replaced
# ══════════════════════════════════════════════════════════════════════


def _reference_ema_update(shadow: nn.Module, model: nn.Module, decay: float) -> None:
    """``ModelEMA.update`` as it was: one Python iteration, four kernels per parameter."""
    with torch.no_grad():
        live = dict(model.named_parameters())
        for name, shadow_p in shadow.named_parameters():
            if name in live:
                shadow_p.copy_(decay * shadow_p + (1.0 - decay) * live[name])
        live_b = dict(model.named_buffers())
        for name, shadow_b in shadow.named_buffers():
            if name in live_b and shadow_b.dtype.is_floating_point:
                shadow_b.copy_(live_b[name])


def test_ema_multi_tensor_update_is_bit_identical() -> None:
    """The ``_foreach`` EMA must agree with the per-parameter loop on every bit.

    Over 50 steps, so a per-step rounding difference of one ulp would compound
    into something ``torch.equal`` sees. This is the update that runs on *every*
    optimiser step of all three stages, and it feeds the shadow that gets
    checkpointed — a drift here is a drift in the shipped weights.
    """
    model = _mlp()
    model(torch.randn(16, 48))  # populate the BatchNorm buffers
    ema = ModelEMA(model, decay=0.999)
    reference = copy.deepcopy(ema.shadow)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()
    torch.manual_seed(11)
    for step in range(50):
        optimizer.zero_grad()
        criterion(model(torch.randn(16, 48)), torch.randint(0, 12, (16,))).backward()
        optimizer.step()
        # `ModelEMA` increments its counter *before* reading `current_decay`.
        n = step + 1
        _reference_ema_update(reference, model, min(0.999, (1.0 + n) / (10.0 + n)))
        ema.update(model)

    actual, expected = ema.shadow.state_dict(), reference.state_dict()
    assert actual.keys() == expected.keys()
    drifted = [k for k in expected if not torch.equal(actual[k], expected[k])]
    assert not drifted, f"EMA drifted on {drifted}"


class _ReferenceSAM(torch.optim.Optimizer):
    """``SAM`` as it was: a clone, a scale and an add per parameter, per step."""

    def __init__(self, params, base_cls, rho=0.05, adaptive=False, **kw):  # type: ignore[no-untyped-def]
        super().__init__(params, dict(rho=rho, adaptive=adaptive, **kw))
        self.base_optimizer = base_cls(self.param_groups, **kw)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (norm + 1e-12)
            adaptive = bool(group.get("adaptive", False))
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = p.grad * scale.to(p)
                if adaptive:
                    e_w = e_w * p.data.pow(2)
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None and "old_p" in self.state[p]:
                    p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self) -> torch.Tensor:
        dev = self.param_groups[0]["params"][0].device
        norms = [
            ((p.data.abs() * p.grad) if g.get("adaptive", False) else p.grad).norm(p=2).to(dev)
            for g in self.param_groups
            for p in g["params"]
            if p.grad is not None
        ]
        return torch.norm(torch.stack(norms), p=2).clamp(min=1e-6)

    def step(self, closure=None):  # type: ignore[no-untyped-def,override]
        raise NotImplementedError


@pytest.mark.parametrize("adaptive", [False, True], ids=["sam", "asam"])
def test_sam_multi_tensor_step_is_bit_identical(adaptive: bool) -> None:
    """Both SAM modes must reproduce the per-parameter ascent/descent exactly.

    ``adaptive=True`` is the shipped Stage-3 setting (T2-4 / OP-5) and exercises
    the extra ``* theta**2`` term, which is where a reassociation would hide.
    """

    def groups(model: nn.Module) -> list[dict[str, object]]:
        return [
            {"params": [p for p in model.parameters() if p.ndim > 1], "weight_decay": 2e-4},
            {"params": [p for p in model.parameters() if p.ndim == 1], "weight_decay": 0.0},
        ]

    actual_model, reference_model = _mlp(), _mlp()
    reference_model.load_state_dict(copy.deepcopy(actual_model.state_dict()))
    kwargs = dict(rho=0.015, adaptive=adaptive, lr=4e-5, weight_decay=2e-4)
    actual = SAM(groups(actual_model), torch.optim.AdamW, **kwargs)  # type: ignore[arg-type]
    reference = _ReferenceSAM(groups(reference_model), torch.optim.AdamW, **kwargs)

    criterion = nn.CrossEntropyLoss()
    torch.manual_seed(11)
    for _ in range(15):
        x, y = torch.randn(24, 48), torch.randint(0, 12, (24,))
        for model, optimizer in ((actual_model, actual), (reference_model, reference)):
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.first_step(zero_grad=True)
            criterion(model(x), y).backward()
            optimizer.second_step(zero_grad=True)

    got, want = actual_model.state_dict(), reference_model.state_dict()
    drifted = [k for k in want if not torch.equal(got[k], want[k])]
    assert not drifted, f"SAM drifted on {drifted}"


def test_sam_restore_returns_the_exact_pre_ascent_weights() -> None:
    """The non-finite abort path must undo ``first_step`` bit-exactly.

    ``restore`` now copies out of a reused buffer instead of rebinding a fresh
    clone; if that buffer were ever stale the perturbation would be baked in
    permanently, which is the failure the method exists to prevent.
    """
    model = _mlp()
    optimizer = SAM(list(model.parameters()), torch.optim.AdamW, rho=0.02, adaptive=True, lr=1e-4)
    before = {n: p.clone() for n, p in model.named_parameters()}

    nn.CrossEntropyLoss()(model(torch.randn(8, 48)), torch.randint(0, 12, (8,))).backward()
    optimizer.first_step(zero_grad=True)
    assert any(not torch.equal(before[n], p) for n, p in model.named_parameters()), (
        "first_step did not move the weights, so restore proves nothing"
    )

    optimizer.restore(zero_grad=True)
    assert all(torch.equal(before[n], p) for n, p in model.named_parameters())


def _reference_branch_grad_norms(model: nn.Module) -> dict[str, torch.Tensor]:
    """The per-parameter, string-matching-every-step version."""
    squares: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        for prefix in BRANCH_PREFIXES:
            if name.startswith(prefix):
                sq = param.grad.detach().float().pow(2).sum()
                squares[prefix] = sq if prefix not in squares else squares[prefix] + sq
                break
    return {p.rstrip("."): v.sqrt() for p, v in squares.items()}


class _Branched(nn.Module):
    """A stand-in with ``SpectralQuadNet``'s top-level attribute names."""

    def __init__(self) -> None:
        super().__init__()
        for name in ("branch_a", "branch_b", "branch_c", "branch_d"):
            setattr(self, name, nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 64)))
        self.cross_interaction = nn.Linear(64, 64)
        self.arcface_head = nn.Linear(64, 90)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = ("branch_a", "branch_b", "branch_c", "branch_d")
        return self.arcface_head(self.cross_interaction(sum(getattr(self, n)(x) for n in branches)))


def test_branch_grad_norms_match_the_per_parameter_loop() -> None:
    """GradNorm reads these, so a reassociation here would move the aux weights."""
    torch.manual_seed(SEED)
    model = _Branched()
    nn.CrossEntropyLoss()(model(torch.randn(32, 64)), torch.randint(0, 90, (32,))).backward()

    actual, expected = branch_grad_norm_tensors(model), _reference_branch_grad_norms(model)
    assert actual.keys() == expected.keys()
    assert all(torch.equal(actual[k], expected[k]) for k in expected)


def test_a_branch_without_gradient_is_omitted_not_reported_as_zero() -> None:
    """A frozen owner must stay distinguishable from one that trains but is flat."""
    torch.manual_seed(SEED)
    model = _Branched()
    nn.CrossEntropyLoss()(model(torch.randn(32, 64)), torch.randint(0, 90, (32,))).backward()
    model.branch_b.zero_grad(set_to_none=True)
    assert "branch_b" not in branch_grad_norm_tensors(model)
    assert "branch_a" in branch_grad_norm_tensors(model)


@pytest.mark.parametrize(("batch", "n_classes"), [(128, 90), (24, 5), (7, 90), (64, 16)])
def test_protonce_label_mapping_is_the_dict_lookup_it_replaced(batch: int, n_classes: int) -> None:
    """``searchsorted`` over the sorted uniques is the same permutation the dict was.

    The dict cost one ``.item()`` per sample — 128 host synchronisations per
    step — so this is the single largest sync reduction in Stages 2 and 3, and
    it is only admissible because ``Tensor.unique`` returns sorted values.
    """
    torch.manual_seed(SEED)
    features = F.normalize(torch.randn(batch, 256), dim=1)
    labels = torch.randint(0, n_classes, (batch,))
    classes = labels.unique()
    if len(classes) < 2:
        pytest.skip("degenerate batch: the loss short-circuits")

    prototypes = F.normalize(torch.stack([features[labels == c].mean(0) for c in classes]), dim=1)
    similarity = torch.mm(features, prototypes.T) / 0.10
    lookup = {c.item(): i for i, c in enumerate(classes)}
    expected = F.cross_entropy(
        similarity, torch.tensor([lookup[v.item()] for v in labels], dtype=torch.long)
    )
    assert torch.equal(ProtoNCELoss(0.10)(features, labels), expected)


def test_shared_masked_cube_matches_two_independent_grid_extractions() -> None:
    """Branches A and D pool the same ``x * m``; that must change nothing.

    The product is a full ``(B, C, H, W)`` tensor — 80 MB at batch 128 — and it
    was built once per grid. Sharing it is only admissible because pooling is a
    pure function of its input.
    """
    torch.manual_seed(SEED)
    x = torch.randn(3, 40, 64, 64).abs()
    mask = torch.rand(3, 64, 64)

    (grid_a, mass_a), (grid_d, mass_d) = extract_grid_spectra_multi(x, (8, 4), mask=mask)
    ref_a, ref_mass_a = extract_grid_spectra(x, grid_size=8, mask=mask)
    ref_d, ref_mass_d = extract_grid_spectra(x, grid_size=4, mask=mask)

    assert torch.equal(grid_a, ref_a) and torch.equal(mass_a, ref_mass_a)
    assert torch.equal(grid_d, ref_d) and torch.equal(mass_d, ref_mass_d)


def test_branch_inputs_still_exposes_branch_c_input(cfg, physical_wl) -> None:
    """``forward`` skips Branch C's ``x * m``; the distinctness gate still needs it.

    The forward reaches Branch C through ``self.branch_c(x, m)`` and never read
    the copy ``branch_inputs`` built, so building it was a dead full-cube
    allocation. It has to stay *reachable*, because §4.3's distinctness test is
    a claim about what each branch sees.
    """
    from spectralquadnet.models.spectral_quadnet import SpectralQuadNet
    from spectralquadnet.utils.seed import set_seed

    set_seed(42)
    model = SpectralQuadNet.from_config(cfg, physical_wl).eval()
    x = torch.randn(2, cfg.data.num_bands, 64, 64).abs()

    inputs = model.branch_inputs(x)
    assert set(inputs) == {"a", "a_mass", "b", "c", "d"}
    assert inputs["c"].shape == x.shape


# ══════════════════════════════════════════════════════════════════════
#  Execution-path behaviour
# ══════════════════════════════════════════════════════════════════════


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
        empty_cache_interval=0,
        diagnostics_interval=50,
        progress="off",
    )


def test_loader_options_omit_worker_only_keys_at_zero_workers() -> None:
    """``prefetch_factor``/``persistent_workers`` at ``num_workers=0`` make torch raise."""
    assert _plan(0).loader_kwargs == {"num_workers": 0, "pin_memory": False}
    assert set(_plan(4).loader_kwargs) == {
        "num_workers",
        "pin_memory",
        "persistent_workers",
        "prefetch_factor",
    }


def test_no_plan_reproduces_the_pre_refactor_synchronous_feed() -> None:
    """Every loader built without a plan — as the unit tests do — stays single-process."""
    assert loader_options(None) == {"num_workers": 0, "pin_memory": False}
    assert loader_options(None, evaluation=True) == {"num_workers": 0, "pin_memory": False}


def test_eval_loaders_never_hold_workers_resident() -> None:
    """A persistent pool outliving its throwaway loader is a process leak."""
    assert _plan(4).eval_loader_kwargs["persistent_workers"] is False
    assert _plan(4).loader_kwargs["persistent_workers"] is True


def test_ddp_refuses_a_batch_that_does_not_divide_by_world_size() -> None:
    """Unequal shards make the all-reduced mean a weighted average, not the batch's.

    Refused rather than rounded: silently training at a different effective
    batch size is the failure mode this whole DDP path exists to avoid.
    """
    single = DistContext()
    assert _shard_batch(128, single) == 128

    two_ranks = DistContext(enabled=True, rank=0, world_size=2)
    assert _shard_batch(128, two_ranks) == 64
    with pytest.raises(ValueError, match="does not divide"):
        _shard_batch(127, two_ranks)


def test_dataset_pickles_without_materialising_the_mmap(tmp_path) -> None:
    """A worker must receive the *path*, not the cube.

    ``np.memmap`` inherits ``ndarray``'s pickling, which copies. Sending one to
    four workers would put four copies of a multi-GB array in RAM, which is
    exactly the zero-RAM invariant ``DataStore`` exists to hold.
    """
    from spectralquadnet.data.datasets import RiceSeedDataset
    from spectralquadnet.data.mmap_store import DataStore

    bands, size, n = 4, 8, 32
    rng = np.random.default_rng(0)
    patches_path, labels_path = tmp_path / "p.npy", tmp_path / "l.npy"
    np.save(patches_path, rng.standard_normal((n, bands, size, size)).astype(np.float32))
    np.save(labels_path, np.arange(n, dtype=np.int64) % 4)

    DataStore.reset()
    try:
        store = DataStore(patches_path=str(patches_path), labels_path=str(labels_path))
        # `none`, so the comparison below is of the *data* rather than of two
        # independent augmentation draws.
        dataset = RiceSeedDataset(
            np.arange(n), aug_strength="none", store=store, data_cfg=_AugCfg(), device="cpu"
        )
        state = dataset.__getstate__()
        assert state["patches"] is None, "the mmap must not cross the process boundary"

        payload = pickle.dumps(dataset)
        cube_bytes = n * bands * size * size * 4
        assert len(payload) < cube_bytes, (
            f"pickled dataset is {len(payload)} B against a {cube_bytes} B cube — "
            "the mmap is being materialised"
        )

        revived = pickle.loads(payload)
        assert revived.patches is not None, "__setstate__ must re-open the mapping"
        assert isinstance(revived.patches, np.memmap), "and re-open it as a mapping"
        assert torch.equal(revived[3][0], dataset[3][0]), "re-opened mapping reads other data"
        assert torch.equal(revived[3][1], dataset[3][1])
    finally:
        DataStore.reset()


class _AugCfg:
    """The four augmentation fields ``RiceSeedDataset`` reads off ``data_cfg``."""

    max_cutout_bands = 2
    noise_std = 0.02
    cutmix_bands = 2
    cutmix_spatial = 4


def test_dataset_yields_host_tensors() -> None:
    """Samples stay on the host: a worker cannot return an accelerator tensor.

    This is also what turned one host-to-device copy *per sample per tensor*
    into one per batch — at batch 128 with the fill map and the morphometrics,
    512 transfers became 4.
    """
    from spectralquadnet.data.datasets import RiceSeedDataset

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

    dataset = RiceSeedDataset(
        np.arange(8), aug_strength="heavy", store=_Store(), data_cfg=_AugCfg(), device="mps"
    )
    patch, label = dataset[0]
    assert patch.device.type == "cpu", "device placement belongs to engine/batch.py now"
    assert label.device.type == "cpu"


def test_ema_refuses_a_wrapped_module_instead_of_silently_stopping() -> None:
    """A prefixed ``named_parameters()`` must raise, not pair nothing.

    This is the failure mode the guard exists for: DDP renames every parameter
    to ``module.*``, the name-matched pairing finds no overlap, and the shadow
    quietly stops tracking while all three stages keep checkpointing it. The
    reported F1 would then come from weights that never moved — and nothing
    about the run would look wrong.
    """
    inner = _mlp()
    ema = ModelEMA(inner, decay=0.999)

    class _Wrapper(nn.Module):
        def __init__(self, module: nn.Module) -> None:
            super().__init__()
            self.module = module

    with pytest.raises(RuntimeError, match="unwrapped module"):
        ema.update(_Wrapper(inner))

    ema.update(unwrap_model(_Wrapper(inner)))  # the documented fix works


def test_unwrap_model_sees_through_wrapper_prefixes() -> None:
    """Checkpoints, EMA and the gradient split all address unprefixed names."""
    inner = _mlp()

    class _Wrapper(nn.Module):
        def __init__(self, module: nn.Module) -> None:
            super().__init__()
            self.module = module

    class _Compiled(nn.Module):
        def __init__(self, module: nn.Module) -> None:
            super().__init__()
            self._orig_mod = module

    assert unwrap_model(inner) is inner
    assert unwrap_model(_Wrapper(inner)) is inner
    assert unwrap_model(_Compiled(_Wrapper(inner))) is inner


@pytest.mark.parametrize(
    ("epoch", "interval", "expected"),
    [(1, 50, True), (2, 50, False), (49, 50, False), (50, 50, True), (100, 50, True), (7, 1, True)],
)
def test_diagnostics_gate(epoch: int, interval: int, expected: bool) -> None:
    """Epoch 1 always renders; after that the stride applies.

    A run whose first diagnostic arrived at epoch 50 would hide a broken setup
    for an hour, which is why the gate is not a plain modulo.
    """
    assert should_render_details(epoch, interval) is expected
