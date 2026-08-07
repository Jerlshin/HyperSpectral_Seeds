"""BatchNorm re-estimation after SWA averaging (**T1-5** / C-7e / N-12).

``update_bn_stats`` resets the two ``BatchNorm1d`` layers in the model —
``branch_c.proj`` and ``branch_d.proj``, the output projections of the two
highest-capacity branches — and re-estimates their running statistics for the
averaged weights. It was wrong three ways (IMPROVEMENT_PLAN §2.5.6):

1. **Dropout fired during the pass.** Inverted dropout preserves the mean and
   inflates the variance by ``p/(1-p)·E[a^2]``, so BN recorded a per-channel σ
   larger than the eval-time one and every downstream activation came out
   attenuated — non-uniformly, so the fusion LayerNorm could not undo it.
2. **The class prior was the CDWS-weighted training one**, not the natural one
   the statistics are applied under at test.
3. **The pass was device-dependent**: Metal's fused attention kernel rejects
   dropout under ``no_grad``, so the implementation kept grad enabled there and
   only there — which means the SWA checkpoint's BN buffers, the only stateful
   thing the SWA average does not itself compute, differed between accelerators.

All three have the same fix: switch the stochastic modules off. These tests
pin (1) and (3) directly and (2) through
:func:`~spectralquadnet.data.loaders.build_natural_prior_loader`.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from spectralquadnet.data.loaders import build_natural_prior_loader
from spectralquadnet.data.samplers import ClassBalancedBatchSampler
from spectralquadnet.engine.checkpoint import _STOCHASTIC_MODULES, update_bn_stats

pytestmark = pytest.mark.regression

FEATURES, SAMPLES, BATCH = 8, 32, 8


class Stochastic(nn.Module):
    """A model with the same stateful structure as ``branch_d.proj``'s neighbourhood.

    Attention with internal dropout, an ``nn.Dropout``, then ``BatchNorm1d`` —
    i.e. the arrangement where a train-mode pass inflates the variance BN
    records. Deliberately not ``SpectralQuadNet``: the point is the interaction
    of module types, and a 7.9 M-parameter forward would hide it behind noise.
    """

    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(FEATURES, 2, dropout=0.5, batch_first=True)
        self.drop = nn.Dropout(0.5)
        self.bn = nn.BatchNorm1d(FEATURES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.attn(x, x, x, need_weights=False)
        return self.bn(self.drop(h).mean(dim=1))


def make_loader(seed: int = 0) -> DataLoader:
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(SAMPLES, 4, FEATURES, generator=gen)
    y = torch.randint(0, 3, (SAMPLES,), generator=gen)
    return DataLoader(TensorDataset(x, y), batch_size=BATCH, shuffle=False)


def buffers(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: buf.detach().clone() for name, buf in model.named_buffers()}


def fresh_model(seed: int = 0) -> Stochastic:
    torch.manual_seed(seed)
    return Stochastic()


# ══════════════════════════════════════════════════════════════════════
#  Dropout is off for the pass
# ══════════════════════════════════════════════════════════════════════


def test_the_pass_is_reproducible() -> None:
    """Two passes over the same data give the same buffers, exactly.

    The sharpest statement of "no randomness left". With dropout live this
    fails on the first decimal — the estimated variance is a draw, not a
    statistic.
    """
    model, loader = fresh_model(), make_loader()

    update_bn_stats(loader, model, torch.device("cpu"))
    first = buffers(model)
    update_bn_stats(loader, model, torch.device("cpu"))

    for name, value in buffers(model).items():
        assert torch.equal(value, first[name]), name


def test_the_estimated_statistics_match_an_eval_mode_forward() -> None:
    """BN records the statistics of the activations it will actually see at eval.

    The reference is computed independently, from an ``eval()``-mode forward
    over the same batches, and averaged the way ``momentum=None`` averages:
    a cumulative mean of the per-batch statistics, variance included.
    """
    model, loader = fresh_model(), make_loader()
    update_bn_stats(loader, model, torch.device("cpu"))

    model.eval()
    with torch.no_grad():
        per_batch = [
            model.drop(model.attn(x, x, x, need_weights=False)[0]).mean(dim=1) for x, _ in loader
        ]
    means = torch.stack([a.mean(0) for a in per_batch]).mean(0)
    variances = torch.stack([a.var(0, unbiased=True) for a in per_batch]).mean(0)

    assert model.bn.running_mean == pytest.approx(means.numpy(), abs=1e-5)
    assert model.bn.running_var == pytest.approx(variances.numpy(), abs=1e-5)


def test_a_train_mode_pass_would_have_inflated_the_variance() -> None:
    """The defect, measured: this is what the buffers looked like before T1-5.

    Compared channel-averaged rather than per channel — the inflation is a
    systematic bias on top of an eight-sample-per-batch estimate, so individual
    channels are noisy while the aggregate is not.
    """
    model, loader = fresh_model(), make_loader()
    update_bn_stats(loader, model, torch.device("cpu"))
    clean_var = model.bn.running_var.clone()

    # The pre-Tier-1 pass: everything in `train()`, dropout included.
    model.train()
    model.bn.reset_running_stats()
    model.bn.momentum = None
    torch.manual_seed(0)
    with torch.no_grad():
        for x, _ in loader:
            model(x)

    inflation = float(model.bn.running_var.mean() / clean_var.mean())
    assert inflation > 1.5, f"dropout inflated σ² by only {inflation:.2f}x"


def test_every_stochastic_module_is_in_eval_during_the_pass() -> None:
    """Directly observed, not inferred from the numbers.

    Includes ``nn.MultiheadAttention``, whose dropout is a float attribute that
    ``SpectralQuadNet.set_dropout`` cannot reach (N-2).
    """
    model, loader = fresh_model(), make_loader()
    seen: list[bool] = []

    for module in model.modules():
        if isinstance(module, _STOCHASTIC_MODULES):
            module.register_forward_pre_hook(lambda m, _inp: seen.append(m.training))

    update_bn_stats(loader, model, torch.device("cpu"))

    assert seen, "the hooks must have fired"
    assert not any(seen), "no stochastic module may be in train() mode"
    assert nn.MultiheadAttention in _STOCHASTIC_MODULES


def test_batchnorm_is_in_train_mode_during_the_pass() -> None:
    """The converse — BN itself must stay in ``train()``, or nothing is estimated."""
    model, loader = fresh_model(), make_loader()
    seen: list[bool] = []
    model.bn.register_forward_pre_hook(lambda m, _inp: seen.append(m.training))

    update_bn_stats(loader, model, torch.device("cpu"))

    assert seen and all(seen)


def test_the_model_is_left_in_eval_mode() -> None:
    model, loader = fresh_model(), make_loader()
    update_bn_stats(loader, model, torch.device("cpu"))

    assert not any(m.training for m in model.modules())


def test_no_autograd_graph_is_built() -> None:
    """The pass runs under ``no_grad`` on every device now, not just off-Metal."""
    model, loader = fresh_model(), make_loader()
    seen: list[bool] = []
    model.bn.register_forward_pre_hook(lambda _m, _inp: seen.append(torch.is_grad_enabled()))

    update_bn_stats(loader, model, torch.device("cpu"))

    assert seen and not any(seen)


# ══════════════════════════════════════════════════════════════════════
#  N-12 · the buffers no longer depend on the accelerator
# ══════════════════════════════════════════════════════════════════════


def test_bn_stats_device_independent() -> None:
    """§4.3's gate: identical buffers on CPU and the accelerator, to 1e-5.

    Skips when no accelerator is present. Before T1-5 the two devices ran
    *different code* — grad enabled on Metal, disabled elsewhere — so the
    dropout noise differed by construction and this could not have held at any
    tolerance.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        pytest.skip("no accelerator on this machine")

    on_cpu = fresh_model()
    update_bn_stats(make_loader(), on_cpu, torch.device("cpu"))

    on_device = fresh_model().to(device)
    update_bn_stats(make_loader(), on_device, device)

    for name, value in buffers(on_device).items():
        assert value.cpu() == pytest.approx(buffers(on_cpu)[name].numpy(), abs=1e-5), name


# ══════════════════════════════════════════════════════════════════════
#  §2.5.6(ii) · the class prior
# ══════════════════════════════════════════════════════════════════════


def weighted_loader() -> DataLoader:
    """A stand-in for Stage 3's loader: CDWS-weighted class-balanced batches."""
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(SAMPLES, 4, FEATURES, generator=gen)
    labels = np.array([i % 4 for i in range(SAMPLES)])
    sampler = ClassBalancedBatchSampler(
        labels, n_cls=2, n_spc=4, class_weights={0: 3.0, 1: 1.0, 2: 1.0, 3: 1.0}
    )
    ds = TensorDataset(x, torch.from_numpy(labels))
    return DataLoader(ds, batch_sampler=sampler, num_workers=0)


def test_natural_prior_loader_drops_the_weighted_sampler() -> None:
    """BN statistics must be estimated under the prior they are applied under."""
    weighted = weighted_loader()
    natural = build_natural_prior_loader(weighted)

    assert natural.dataset is weighted.dataset
    assert natural.batch_sampler is not weighted.batch_sampler
    assert isinstance(natural.sampler, torch.utils.data.RandomSampler), "shuffled"
    assert natural.batch_size == 2 * 4, "inherits the balanced sampler's batch size"


def test_natural_prior_loader_visits_every_sample_once() -> None:
    """Unweighted means unweighted: one epoch, one visit, nothing dropped.

    The balanced sampler it replaces both over-samples class 0 and truncates
    the epoch to ``len // (n_cls*n_spc)`` batches.
    """
    natural = build_natural_prior_loader(weighted_loader())
    labels = torch.cat([y for _, y in natural])

    assert len(labels) == SAMPLES
    assert torch.equal(labels.bincount(), torch.full((4,), SAMPLES // 4))


def test_natural_prior_loader_honours_an_explicit_batch_size() -> None:
    natural = build_natural_prior_loader(make_loader(), batch_size=3)

    assert natural.batch_size == 3
    assert len(torch.cat([y for _, y in natural])) == SAMPLES, "drop_last is off"
