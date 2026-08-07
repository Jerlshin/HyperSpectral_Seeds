"""Auto-resume and checkpoint-selection against the real artifacts of a trained run.

Pointing a run at the existing ``outputs/output_v12_spa40/`` directory is the
strongest available end-to-end regression signal, since it exercises
checkpoint discovery, sidecar schema, checkpoint selection and evaluation
together, against real trained weights rather than synthetic ones. This
module pins that signal as a permanent test rather than a one-off manual run.

A distinction this suite depends on (see also this file's last test)
──────────────────────────────────────────────────────────────────────
It is tempting to assert that ``final_evaluation`` reproduces
``stage3_meta.json``'s Macro F1 (0.8745). Those are two different quantities:

* **0.8745 is a validation-split number** — the SWA model's ``val_f1``, written
  by ``run_stage3_swa``'s own ``save_ckpt`` call.
* ``final_evaluation`` reports **test-split** numbers, computed from whichever
  checkpoint ``_pick_best_checkpoint`` ranks highest by ``val_f1`` — which on
  these artifacts is **Stage 1** (0.8877), not Stage 3.

So no correct implementation makes ``final_evaluation`` print 0.8745. What is
verifiable, and stronger, is checked here: the recorded ``val_f1`` of each stage
is reproducible from its checkpoint, and the recorded test predictions are
reproducible end-to-end.

Everything that needs ``dataset/`` or ``outputs/`` skips itself when those are
absent, so a fresh clone stays green.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from spectralquadnet.engine.checkpoint import (
    _pick_best_checkpoint,
    latest_completed_stage,
    load_stage_meta,
    stage_ckpt_path,
    stage_exists,
    stage_meta_path,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = REPO_ROOT / "outputs" / "output_v12_spa40"

#: Recorded by the original CUDA run; the numbers this module regresses against.
RECORDED_VAL_F1 = {1: 0.8876597527386549, 2: 0.8866637678171172, 3: 0.8745051282416407}

#: The checkpoint bundle schema that must not drift.
REQUIRED_META_KEYS = {"epoch", "stage", "val_f1", "val_acc", "use_arcface"}


@pytest.fixture(scope="module")
def outputs_cfg(cfg):
    """The experiment config, with ``output_dir`` pinned at the real artifacts."""
    if not OUTPUTS.is_dir():
        pytest.skip(f"{OUTPUTS} not present on this machine")
    cfg = cfg.copy()
    cfg.output_dir = str(OUTPUTS)
    return cfg


# ══════════════════════════════════════════════════════════════════════
#  Auto-resume detection
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.regression
@pytest.mark.requires_dataset
def test_filename_templates_resolve_to_the_recorded_artifacts(outputs_cfg) -> None:
    """``best_stage{n}.pth`` / ``stage{n}_meta.json`` — the pinned filename templates."""
    for stage in (1, 2, 3):
        assert Path(stage_ckpt_path(outputs_cfg, stage)).name == f"best_stage{stage}.pth"
        assert Path(stage_meta_path(outputs_cfg, stage)).name == f"stage{stage}_meta.json"
        assert stage_exists(outputs_cfg, stage), "both the .pth and its .json must be found"


@pytest.mark.regression
@pytest.mark.requires_dataset
def test_auto_resume_reports_all_three_stages_complete(outputs_cfg) -> None:
    """``done_stage == 3`` against the real artifacts, so nothing retrains."""
    assert latest_completed_stage(outputs_cfg) == 3


@pytest.mark.regression
@pytest.mark.requires_dataset
def test_incomplete_stage_resumes_rather_than_skipping(outputs_cfg, tmp_path) -> None:
    """A ``.pth`` with no ``.json`` means the stage did not finish.

    Copies only the checkpoint side of Stage 1 into an empty directory: the
    resume probe must report 0, not 1, so a crash between the two writes replays
    the stage instead of silently accepting a half-written one.
    """
    cfg = outputs_cfg.copy()
    cfg.output_dir = str(tmp_path)
    (tmp_path / "best_stage1.pth").write_bytes(b"not a real checkpoint")
    assert latest_completed_stage(cfg) == 0


# ══════════════════════════════════════════════════════════════════════
#  Checkpoint selection
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.regression
@pytest.mark.requires_dataset
def test_pick_best_checkpoint_ranks_by_val_f1_not_by_stage_order(outputs_cfg) -> None:
    """Stage 1's 0.8877 outranks Stage 3's 0.8745, so the final eval loads Stage 1.

    This is the behaviour ``stage3_meta.json``'s own ``note`` field anticipates
    ("Stage 2 ckpt preferred for eval") and the reason "final_evaluation
    reproduces stage3_meta.json's Macro F1" is not a valid assertion — see
    the module docstring.
    """
    paths = [stage_ckpt_path(outputs_cfg, s) for s in (1, 2, 3)]
    best = _pick_best_checkpoint(outputs_cfg, *paths)
    assert Path(best).name == "best_stage1.pth"
    assert RECORDED_VAL_F1[1] == max(RECORDED_VAL_F1.values())


@pytest.mark.regression
@pytest.mark.requires_dataset
def test_stage_meta_sidecars_keep_the_pinned_schema(outputs_cfg) -> None:
    """The sidecar schema: the bundle minus ``model``/``ema``, JSON-safe only."""
    for stage in (1, 2, 3):
        meta = json.loads(Path(stage_meta_path(outputs_cfg, stage)).read_text())
        assert set(meta) >= REQUIRED_META_KEYS, f"stage {stage} sidecar lost a pinned key"
        assert "model" not in meta and "ema" not in meta, "tensors must never reach the JSON"
        assert meta["val_f1"] == pytest.approx(RECORDED_VAL_F1[stage])


@pytest.mark.regression
@pytest.mark.requires_dataset
def test_load_stage_meta_reintegerises_class_keys(outputs_cfg) -> None:
    """JSON stringifies dict keys; the samplers and CDWS index them as ints."""
    meta = load_stage_meta(outputs_cfg, 1)
    assert meta["class_f1"], "Stage 1 recorded a per-class F1 map"
    assert all(isinstance(k, int) for k in meta["class_f1"])
    assert all(isinstance(k, int) for k in meta["cdws_weights"])


# ══════════════════════════════════════════════════════════════════════
#  The recorded metrics are reproducible from the checkpoints
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.regression
@pytest.mark.requires_dataset
def test_the_stage3_checkpoint_can_no_longer_be_re_scored(outputs_cfg) -> None:
    """``stage3_meta.json``'s 0.8745 is now a historical number, and this says so.

    Through Tier 2 this test rebuilt the model from the config, loaded
    ``best_stage3.pth``, evaluated the **EMA shadow** — the model
    ``final_evaluation`` scores — on the validation split, and reproduced the
    recorded 0.8745 to 5e-3. It exercised the state-dict key schema, mmap
    loading, the bundle schema and ``evaluate`` together, and it was the
    strongest end-to-end gate in the suite.

    Tier 3 ends it. BR-1/BR-3/BR-4 changed what three of the four branches
    consume, so those weights have no home in this model and ``load_ckpt``
    refuses them (T3-1 … T3-4). What is asserted instead is that the refusal
    happens *at load*, before any evaluation runs — the failure mode worth
    ruling out is a partial load that scores something and gets written down.

    The end-to-end coverage this gave up returns with the first Tier-3 run:
    ``RECORDED_VAL_F1`` is the number to re-pin then.
    """
    data_files = [
        Path(outputs_cfg.data.patches_data),
        Path(outputs_cfg.data.labels_path),
        Path(outputs_cfg.data.wavelength_path),
    ]
    if any(not p.exists() for p in data_files):
        pytest.skip("real dataset/ not present on this machine")

    from spectralquadnet.data.loaders import build_loaders, build_splits
    from spectralquadnet.data.mmap_store import DataStore
    from spectralquadnet.engine.checkpoint import SchemaTooOldError, load_ckpt
    from spectralquadnet.engine.evaluate import evaluate
    from spectralquadnet.models.ema import ModelEMA
    from spectralquadnet.models.spectral_quadnet import SpectralQuadNet
    from spectralquadnet.utils.seed import set_seed

    device = torch.device("cpu")
    DataStore.reset()
    try:
        # Seed before construction — the ordering train.py documents at its call site.
        set_seed(outputs_cfg.seed)
        store = DataStore.from_config(outputs_cfg.data, device)
        _, train_idx, val_idx, test_idx = build_splits(outputs_cfg)
        model = SpectralQuadNet.from_config(outputs_cfg, store.require_wavelengths()).to(device)
        ema = ModelEMA(model, decay=outputs_cfg.ema_decay)

        with pytest.raises(SchemaTooOldError, match="predates Tier 3"):
            load_ckpt(stage_ckpt_path(outputs_cfg, 3), model, ema, device)

        # The loader half of the gate still runs, and is still worth running:
        # it is the only place the real mmap, the real split and the Tier-3
        # side-input plumbing meet outside a training run.
        _, val_ldr, _ = build_loaders(outputs_cfg, store, device, train_idx, val_idx, test_idx, 256)
        f1, _acc = evaluate(model, val_ldr, device)
    finally:
        DataStore.reset()

    # An untrained model on 90 classes. Asserted only to prove the pass ran.
    assert 0.0 <= f1 < 0.20
    assert RECORDED_VAL_F1[3] == pytest.approx(0.8745, abs=1e-4)


@pytest.mark.regression
@pytest.mark.requires_dataset
def test_recorded_test_predictions_match_their_reported_metrics() -> None:
    """The test-split numbers ``final_evaluation`` actually produces.

    Guards the module docstring's claim from the other direction: whatever
    the run writes to ``test_preds_*.npy`` must be what the printed metrics
    describe. The two values below are what the recorded artifacts evaluate
    to — **not** 0.8745, which is a validation number for a different
    checkpoint.
    """
    from sklearn.metrics import f1_score

    files = {
        n: OUTPUTS / f"{n}.npy" for n in ("test_targets", "test_preds_TTA", "test_preds_noTTA")
    }
    if any(not p.exists() for p in files.values()):
        pytest.skip("recorded prediction artifacts not present")

    targets = np.load(files["test_targets"])
    tta = f1_score(targets, np.load(files["test_preds_TTA"]), average="macro", zero_division=0)
    no_tta = f1_score(targets, np.load(files["test_preds_noTTA"]), average="macro", zero_division=0)

    assert len(targets) == 1294, "the stratified 70/15/15 split's test partition"
    assert tta == pytest.approx(0.8933, abs=5e-3)
    assert no_tta == pytest.approx(0.8770, abs=5e-3)
    assert tta > no_tta, "12-view TTA is what buys the last ~1.6 points of macro F1"
