"""Training/evaluation engine: epoch loops, TTA, checkpointing, diagnostics."""

from spectralquadnet.engine.checkpoint import (
    latest_completed_stage,
    load_ckpt,
    load_stage_meta,
    save_ckpt,
    stage_ckpt_path,
    stage_exists,
    stage_meta_path,
    update_bn_stats,
)
from spectralquadnet.engine.diagnostics import compute_branch_influence, compute_class_difficulty
from spectralquadnet.engine.evaluate import evaluate, evaluate_per_class
from spectralquadnet.engine.train_epoch import train_one_epoch, train_one_epoch_sam
from spectralquadnet.engine.tta import tta_predict

__all__ = [
    "compute_branch_influence",
    "compute_class_difficulty",
    "evaluate",
    "evaluate_per_class",
    "latest_completed_stage",
    "load_ckpt",
    "load_stage_meta",
    "save_ckpt",
    "stage_ckpt_path",
    "stage_exists",
    "stage_meta_path",
    "tta_predict",
    "train_one_epoch",
    "train_one_epoch_sam",
    "update_bn_stats",
]
