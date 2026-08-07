"""Stage 3's SAM step and its two averaging schemes (**T1-4**, **T1-9**, **T1-10**).

SAM minimises ``max_{||eps|| <= rho} L(theta + eps)`` for **one** ``L``. Stage 3
ascended along the gradient of ``L_focal + 0.02 L_supcon + 0.10 L_aux`` and then
descended along the gradient of ``L_focal`` alone, so the correction term was
``rho·H_D·g_A`` instead of ``rho·H_D·g_D``. Decomposing
``g_A = alpha·g_D + beta·u`` with ``u`` orthogonal to ``g_D``, only the
``alpha`` fraction is a sharpness penalty; the ``beta`` fraction is an
anisotropic perturbation with no descent guarantee, amplified by whatever the
Hessian's large eigendirections happen to be — exactly what SAM exists to avoid
(IMPROVEMENT_PLAN §2.5.1 C-6, Appendix A.3).

The two ``test_the_objective_mismatch_was_small_*`` tests measure ``alpha`` —
at initialisation and on the trained Stage-2 weights Stage 3 starts from. Both
come out **above 0.99**, so **F-9** ("``cos(g_A, g_D) < 0.8`` in Stage 3") is
refuted: the mismatch was real but tiny, and T1-4's +0.003…+0.008 estimate does
not follow from it. The fix stands on the definition — SAM is not defined for
two objectives — rather than on a predicted gain.

The other two items are wiring: ProtoNCE was constructed, weighted, logged
under ``sched/proto_weight`` and never added to any loss (N-1e), and the EMA
shadow was overwritten by the SWA average without either being scored (C-7f).
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from spectralquadnet.engine.checkpoint import SchemaTooOldError, load_ckpt
from spectralquadnet.engine.diagnostics import flat_grad, grad_cosine
from spectralquadnet.engine.stages.stage3_sam_swa import run_stage3_swa
from spectralquadnet.engine.train_epoch import train_one_epoch_sam
from spectralquadnet.losses.auxiliary import _compute_aux_loss
from spectralquadnet.losses.contrastive import ProtoNCELoss, SupConLoss
from spectralquadnet.losses.focal import FocalLoss
from spectralquadnet.models.ema import ModelEMA
from spectralquadnet.models.spectral_quadnet import SpectralQuadNet
from spectralquadnet.tracking.base import NullTracker

pytestmark = pytest.mark.regression

DEVICE = torch.device("cpu")
SAMPLES, BATCH, SPATIAL = 8, 4, 64
SUPCON_WEIGHT, PROTO_WEIGHT = 0.02, 0.01
ARC_M = 0.30


class CaptureTracker(NullTracker):
    """A ``NullTracker`` that keeps the machine channel's scalars."""

    def __init__(self) -> None:
        self.scalars: dict[str, float] = {}

    def log_scalars(self, tags: dict[str, float], step: int) -> None:
        self.scalars.update(tags)


@pytest.fixture
def stage3_cfg(cfg, tmp_path):
    """The composed config, shrunk to a two-epoch Stage 3 writing into ``tmp_path``."""
    small = cfg.copy()
    small.output_dir = str(tmp_path)
    small.stage3.epochs = 2
    small.stage3.cycle_len = 1
    return small


@pytest.fixture
def stage3_model(cfg, physical_wl, silence_dropout):
    """A Stage-3-configured model whose forward is deterministic.

    Stage 3 sets ``branch_drop_prob = 0`` at entry; with T1-6 wiring that in,
    the only randomness left is dropout, which ``silence_dropout`` closes.
    Without both, two forwards at the same weights differ and no statement
    about the *objective* could be separated from sampling noise.

    Since HD-1 (T2-10) there is no head to select: the sub-centre cosine head
    is the only one, so the ``use_arcface(True)`` this fixture used to call has
    nothing left to switch.
    """
    torch.manual_seed(0)
    model = SpectralQuadNet.from_config(cfg, physical_wl).to(DEVICE)
    silence_dropout(model)
    model.branch_drop_prob = 0.0
    return model.train()


@pytest.fixture(scope="module")
def synthetic_loader(cfg) -> DataLoader:
    """Two batches, two classes per batch — enough for SupCon and ProtoNCE to be non-zero."""
    gen = torch.Generator().manual_seed(3)
    x = torch.randn(SAMPLES, cfg.data.num_bands, SPATIAL, SPATIAL, generator=gen)
    y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    return DataLoader(TensorDataset(x, y), batch_size=BATCH, shuffle=False)


def run_one_sam_epoch(
    cfg: Any,
    model: SpectralQuadNet,
    loader: DataLoader,
    rho: float,
    tracker: CaptureTracker | None = None,
    proto_weight: float = PROTO_WEIGHT,
    ema: ModelEMA | None = None,
) -> tuple[float, float]:
    """``train_one_epoch_sam`` with Stage 3's exact loss set, at a chosen ``rho``."""
    from spectralquadnet.optim.param_groups import _wd_groups
    from spectralquadnet.optim.sam import SAM

    sam = SAM(
        list(_wd_groups(cfg, model.named_parameters(), cfg.stage3.swa_lr)),
        optim.AdamW,
        rho=rho,
        lr=cfg.stage3.swa_lr,
        weight_decay=cfg.weight_decay,
    )
    return train_one_epoch_sam(
        cfg,
        model,
        loader,
        sam,
        FocalLoss(gamma=1.0),
        DEVICE,
        supcon=SupConLoss(temperature=0.10),
        supcon_weight=SUPCON_WEIGHT,
        proto=ProtoNCELoss(temperature=0.10),
        proto_weight=proto_weight,
        arc_m=ARC_M,
        aux_weight=cfg.stage3.aux_loss_weight,
        current_ep=1,
        ema=ema,
        tracker=tracker,
    )


# ══════════════════════════════════════════════════════════════════════
#  T1-4 · the two SAM steps optimise one objective
# ══════════════════════════════════════════════════════════════════════


def test_sam_ascent_descent_objectives_identical(cfg, stage3_model, synthetic_loader) -> None:
    """§4.3's gate: ``cos(g_A, g_D) == 1`` within 1e-6.

    Run at ``rho = 0`` so the descent gradient is evaluated at the same weights
    as the ascent one — that isolates "same objective" from "same point", which
    is the property T1-4 is about. At the shipped ``rho = 0.015`` the cosine
    falls below 1 only by the curvature the perturbation traverses, which is
    the sharpness signal SAM is there to measure.
    """
    tracker = CaptureTracker()
    run_one_sam_epoch(cfg, stage3_model, synthetic_loader, rho=0.0, tracker=tracker)

    assert "sam/grad_cos" in tracker.scalars, "the diagnostic must be logged"
    assert tracker.scalars["sam/grad_cos"] == pytest.approx(1.0, abs=1e-6)


def legacy_objective_cosine(model: SpectralQuadNet, x: torch.Tensor, y: torch.Tensor) -> float:
    """``cos(g_A, g_D)`` for the *pre-Tier-1* pair of objectives, at one set of weights.

    The ascent objective was ``focal + 0.02 supcon + 0.01 proto + 0.10 aux``;
    the descent objective was ``focal`` alone. This is ``alpha`` — the fraction
    of SAM's perturbation that was a genuine sharpness penalty.
    """
    focal = FocalLoss(gamma=1.0)
    supcon = SupConLoss(temperature=0.10)
    proto = ProtoNCELoss(temperature=0.10)

    def gradient_of(compound: bool) -> torch.Tensor:
        model.zero_grad(set_to_none=True)
        out = model(x, labels=y, arc_m=ARC_M, return_embed=True)
        loss = focal(out["main"], y)
        if compound:
            loss = loss + SUPCON_WEIGHT * supcon(out["emb"], y)
            loss = loss + PROTO_WEIGHT * proto(out["emb"], y)
            loss = loss + 0.10 * _compute_aux_loss(focal, out, y, y, 1.0, use_mixup=False)
        loss.backward()
        return flat_grad(model).clone()

    return float(grad_cosine(gradient_of(compound=True), gradient_of(compound=False)))


def test_the_objective_mismatch_was_small_at_initialisation(stage3_model, synthetic_loader) -> None:
    """**F-9 is refuted**: the two objectives were already nearly parallel.

    §2.5.1 predicted ``cos(g_A, g_D) < 0.8``, reasoning that the auxiliary
    heads sit directly on the branch embeddings with large, well-conditioned
    gradients while the main path must traverse fusion, ``EmbedNet`` and the
    ``s = 48`` head. The measurement goes the other way: the ``s = 48``
    amplification makes the *focal* term dominate ``g_A`` as well, so dropping
    the auxiliaries on the descent step rotated the gradient by well under a
    degree.

    The T1-4 fix is still correct — SAM is not defined for two objectives — but
    its **+0.003…+0.008 estimate does not follow from this measurement**, and
    the number is pinned here so that is not quietly forgotten.
    """
    x, y = next(iter(synthetic_loader))

    assert legacy_objective_cosine(stage3_model, x, y) > 0.99


@pytest.mark.requires_dataset
def test_the_trained_weight_measurement_is_now_unreachable(
    cfg, checkpoint_paths, physical_wl
) -> None:
    """F-9's second measurement point — which Tier 3 put out of reach.

    Through Tier 2 this test loaded ``best_stage2.pth`` and re-measured
    ``cos(g_A, g_D)`` on the weights Stage 3 actually starts from, because
    initialisation is the weakest place to measure it: the auxiliary heads
    start at ``std = 0.02`` and contribute almost nothing. It came out at 0.9996
    there — ``||g_A|| = 1010`` against ``||g_D|| = 976`` — and that number is
    recorded in MIGRATION_PROGRESS under T1-4.

    It is not re-derivable. BR-1/BR-3/BR-4 changed what three branches consume,
    so a pre-Tier-3 checkpoint cannot be loaded into this model at all. The test
    is inverted rather than removed, so the record says the measurement is
    historical instead of leaving it looking current.
    """
    torch.manual_seed(0)
    model = SpectralQuadNet.from_config(cfg, physical_wl).to(DEVICE)
    ema = ModelEMA(model, decay=cfg.ema_decay)

    with pytest.raises(SchemaTooOldError, match="predates Tier 3"):
        load_ckpt(str(checkpoint_paths[2]), model, ema, DEVICE)


def test_sam_restores_the_weights_when_the_descent_loss_is_not_finite(
    cfg, stage3_model, synthetic_loader
) -> None:
    """The abort path: a skipped batch must not leave the ascent baked in.

    ``first_step`` moves the weights and caches the originals; the next
    ``first_step`` overwrites that cache. Skipping between the two — which the
    non-finite guard did — made the perturbation permanent.
    """
    from spectralquadnet.optim.param_groups import _wd_groups
    from spectralquadnet.optim.sam import SAM

    sam = SAM(
        list(_wd_groups(cfg, stage3_model.named_parameters(), cfg.stage3.swa_lr)),
        optim.AdamW,
        rho=0.5,
        lr=cfg.stage3.swa_lr,
        weight_decay=cfg.weight_decay,
    )
    x, y = next(iter(synthetic_loader))
    # Parameters only: a train-mode forward also advances the BatchNorm running
    # buffers, which `restore` neither can nor should undo.
    before = {k: v.detach().clone() for k, v in stage3_model.named_parameters()}

    out = stage3_model(x, labels=y, arc_m=ARC_M)
    FocalLoss(gamma=1.0)(out["main"], y).backward()
    sam.first_step(zero_grad=True)
    assert any(
        not torch.equal(v.detach(), before[k]) for k, v in stage3_model.named_parameters()
    ), "the ascent must have moved something"

    sam.restore(zero_grad=True)
    for key, value in stage3_model.named_parameters():
        assert torch.equal(value.detach(), before[key]), key


# ══════════════════════════════════════════════════════════════════════
#  T1-10 · ProtoNCE is applied, not merely constructed
# ══════════════════════════════════════════════════════════════════════


def test_protonce_weight_changes_the_objective(cfg, stage3_model, synthetic_loader) -> None:
    """``sched/proto_weight`` now reflects a term that exists.

    Two epochs from identical weights, differing only in ``proto_weight``. The
    argument was previously accepted and dropped, so both runs returned the
    same loss to the last bit.
    """
    with_proto = copy.deepcopy(stage3_model)
    without_proto = copy.deepcopy(stage3_model)

    loss_on, _ = run_one_sam_epoch(cfg, with_proto, synthetic_loader, 0.0, proto_weight=0.5)
    loss_off, _ = run_one_sam_epoch(cfg, without_proto, synthetic_loader, 0.0, proto_weight=0.0)

    assert loss_on != loss_off


def test_supcon_and_protonce_both_reach_the_loss(stage3_model, synthetic_loader) -> None:
    """Neither contrastive term is silently zero on a batch that has positives."""
    x, y = next(iter(synthetic_loader))
    out = stage3_model(x, labels=y, arc_m=ARC_M, return_embed=True)

    assert SupConLoss(temperature=0.10)(out["emb"], y).detach().item() > 0.0
    assert ProtoNCELoss(temperature=0.10)(out["emb"], y).detach().item() > 0.0


# ══════════════════════════════════════════════════════════════════════
#  T1-9 · the EMA shadow is maintained, scored and compared
# ══════════════════════════════════════════════════════════════════════


def test_the_sam_loop_updates_the_ema_shadow(cfg, stage3_model, synthetic_loader) -> None:
    """Stage 3's shadow tracks its trajectory, as Stages 1–2's do."""
    ema = ModelEMA(stage3_model, decay=cfg.ema_decay)
    before = copy.deepcopy(ema.shadow.state_dict())

    run_one_sam_epoch(cfg, stage3_model, synthetic_loader, rho=0.0, ema=ema)

    moved = [k for k, v in ema.shadow.state_dict().items() if not torch.equal(v, before[k])]
    assert moved, "the shadow was never updated"


@pytest.mark.slow
def test_stage3_scores_both_averages_and_records_the_winner(
    stage3_cfg, physical_wl, synthetic_loader, tmp_path
) -> None:
    """The end-to-end T1-9 gate, over a two-epoch Stage 3.

    Three things at once: ``val/f1_ema`` appears in the telemetry like Stages
    1–2 (the plan's stated criterion), both averages are scored and recorded in
    the sidecar, and ``best_source`` names the one the ``ema`` slot received.
    Before Tier 1 the shadow was overwritten by the SWA average with neither
    ever evaluated.
    """
    torch.manual_seed(0)
    model = SpectralQuadNet.from_config(stage3_cfg, physical_wl).to(DEVICE)
    ema = ModelEMA(model, decay=stage3_cfg.ema_decay)
    tracker = CaptureTracker()

    best_f1 = run_stage3_swa(
        stage3_cfg,
        model,
        ema,
        synthetic_loader,
        synthetic_loader,
        DEVICE,
        best_ckpt=str(tmp_path / "best_stage3.pth"),
        prev_best_f1=0.0,
        tracker=tracker,
    )

    assert "val/f1_ema" in tracker.scalars, "Stage 3 must score its shadow, like Stages 1-2"
    assert "val/acc_ema" in tracker.scalars
    assert {"swa/f1", "ema/f1"} <= set(tracker.scalars), "both averages scored at the end"

    meta = json.loads((tmp_path / "stage3_meta.json").read_text())
    assert meta["best_source"] in {"swa", "ema"}
    assert best_f1 == pytest.approx(max(meta["swa_val_f1"], meta["ema_val_f1"]))
    assert meta["val_f1"] == pytest.approx(best_f1)

    winner = meta["swa_val_f1"] if meta["best_source"] == "swa" else meta["ema_val_f1"]
    assert winner == pytest.approx(best_f1), "best_source must name the higher-scoring average"


@pytest.mark.slow
def test_stage3_bundle_ema_slot_holds_the_recorded_winner(
    stage3_cfg, physical_wl, synthetic_loader, tmp_path
) -> None:
    """``final_eval`` loads the ``ema`` slot, so it must hold whichever average won."""
    torch.manual_seed(1)
    model = SpectralQuadNet.from_config(stage3_cfg, physical_wl).to(DEVICE)
    ema = ModelEMA(model, decay=stage3_cfg.ema_decay)
    path = tmp_path / "best_stage3.pth"

    run_stage3_swa(
        stage3_cfg,
        model,
        ema,
        synthetic_loader,
        synthetic_loader,
        DEVICE,
        best_ckpt=str(path),
        prev_best_f1=0.0,
    )

    bundle = torch.load(path, map_location="cpu", weights_only=False)
    swa_matches_ema_slot = all(
        torch.equal(v, bundle["ema"][k])
        for k, v in bundle["model"].items()
        if v.dtype.is_floating_point
    )
    assert swa_matches_ema_slot == (bundle["best_source"] == "swa")
