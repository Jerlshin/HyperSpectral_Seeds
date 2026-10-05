"""S22 comparative interpretation and execution receipt from fixed artifacts."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from spectralquadnet.data.prep.multimodal import sha256, write_json
from spectralquadnet.experiments.rgb_probe import cluster_interval

ROOT = Path(__file__).resolve().parents[5]
BASE = ROOT / "docs/research/evidence/S22_complementary_v5"


def main() -> None:
    archive = BASE / "screen_results"
    manifest = json.loads((archive / "ARCHIVED.json").read_text())
    for name, digest in manifest["files"].items():
        if sha256(archive / name) != digest:
            raise ValueError("S22 archive changed")
    classes = pd.read_csv(archive / "all_per_class.csv")
    comparisons = []
    for candidate, control in (("v5_tta", "hsi_q32"), ("v5_rgb_equal", "fusion32_equal"),
                               ("v5_rgb_equal", "v5_tta"), ("fusion32_equal", "hsi_q32")):
        a = classes[classes.arm == candidate].set_index(["label", "fold"]).sort_index()
        b = classes[classes.arm == control].set_index(["label", "fold"]).sort_index()
        if not a.index.equals(b.index):
            raise ValueError("Comparator class identities differ")
        for metric, subset in (("f1", "all"), ("recall", "same"), ("recall", "cross")):
            delta = a[metric] - b[metric]
            if subset != "all":
                delta = delta[a.cross if subset == "cross" else ~a.cross]
            matrix = delta.unstack("fold").to_numpy()
            comparisons.append({"candidate": candidate, "control": control, "metric": metric,
                                "subset": subset, "mean": float(matrix.mean()),
                                "ci": cluster_interval(matrix), "fold_deltas": matrix.mean(0).tolist()})
    a = classes[(classes.arm == "v5_rgb_equal") & classes.cross].set_index(["label", "fold"]).sort_index()
    b = classes[(classes.arm == "v5_tta") & classes.cross].set_index(["label", "fold"]).sort_index()
    a["delta"] = a.recall - b.recall
    direction = []
    for label, mask in (("to_session8", a.destination_session == 8),
                        ("from_session8", a.destination_session != 8)):
        block = a[mask]
        if len(block) != 17 or len(block.index.get_level_values("label").unique()) != 17:
            raise ValueError("Acquisition direction coverage differs")
        direction.append({"direction": label, "fusion_minus_hsi_recall": float(block.delta.mean()),
                          "ci": cluster_interval(block.delta.to_numpy()),
                          "improved_classes": int((block.delta > 0).sum()),
                          "harmed_classes": int((block.delta < 0).sum()),
                          "unchanged_classes": int((block.delta == 0).sum())})
    cells = []
    for fold in (0, 1):
        root = ROOT / f"outputs/s22_complementary_v5/f{fold}_s0"
        run = json.loads((root / "results/run.json").read_text())
        marker = json.loads((root / "last_stage1.json").read_text())
        if not marker["finished"] or run["run"]["environment"]["world_size"] != 2:
            raise ValueError("CUDA cell did not finish the declared runtime")
        cells.append({"fold": fold, "seed": 0, "epochs_completed": marker["epoch"],
                      "selected_epoch": run["checkpoint"]["epoch"],
                      "selected_source": run["checkpoint"]["best_source"],
                      "calib_f1": run["checkpoint"]["selection_f1"],
                      "environment": run["run"]["environment"],
                      "checkpoint_sha256": sha256(root / "best_stage1.pth"),
                      "run_manifest_sha256": sha256(root / "results/run.json")})
    result = BASE / "screen_synthesis"
    result.mkdir(exist_ok=False)
    write_json(result / "comparisons.json", comparisons)
    write_json(result / "direction_deltas.json", direction)
    write_json(result / "execution.json", {"cells": cells,
               "cuda_bootstrap_wall_seconds": json.loads((ROOT / "outputs/CUDA_COMPLETE.json").read_text())["seconds"],
               "scope": "Two complete seed-0 CUDA fits. Earlier failed/interrupted local attempts remain preserved separately. Wall time is not a billed GPU-hour claim."})
    for name in ("CUDA_COMPLETE.json", "cuda_launch_receipt.json"):
        shutil.copy2(ROOT / "outputs" / name, result / name)
    for fold in (0, 1):
        root = ROOT / f"outputs/s22_complementary_v5/f{fold}_s0"
        for name in ("provenance.json", "last_stage1.json", "resolved_config.yaml"):
            shutil.copy2(root / name, result / f"f{fold}_{name}")
        shutil.copy2(root / "results/run.json", result / f"f{fold}_run.json")
    write_json(result / "ARCHIVED.json", {"training_plan_sha256": manifest["plan_sha256"],
               "source_sha256": sha256(Path(__file__)),
               "files": {p.name: sha256(p) for p in result.iterdir() if p.is_file()}})
    print(json.dumps({"comparisons": comparisons, "directions": direction}, indent=2))


if __name__ == "__main__":
    main()
