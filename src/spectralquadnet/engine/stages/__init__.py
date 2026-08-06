"""The three-stage curriculum orchestrators plus final TTA evaluation."""

from spectralquadnet.engine.stages.final_eval import final_evaluation
from spectralquadnet.engine.stages.stage1_progressive import run_stage1
from spectralquadnet.engine.stages.stage2_arcface import run_stage2
from spectralquadnet.engine.stages.stage3_sam_swa import run_stage3_swa

__all__ = [
    "final_evaluation",
    "run_stage1",
    "run_stage2",
    "run_stage3_swa",
]
