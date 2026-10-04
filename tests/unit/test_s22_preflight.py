"""A new GPU cell cannot overwrite results or resume a different experiment."""
import json
import runpy
from pathlib import Path

import pytest

from spectralquadnet.data.prep.multimodal import sha256


def test_s22_output_claim_and_resume_are_fail_closed(tmp_path):
    prepare = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/run_complementary_v5.py"))["prepare_output"]
    plan = tmp_path / "plan.json"
    plan.write_text("{}")
    out = tmp_path / "cell"
    prepare(out, plan, "train", 0, 1, False)
    with pytest.raises(FileExistsError):
        prepare(out, plan, "train", 0, 1, False)
    (out / "provenance.json").write_text(json.dumps({
        "training_plan_sha256": sha256(plan), "fold": 0, "seed": 1, "action": "train",
    }))
    prepare(out, plan, "train", 0, 1, True)
    with pytest.raises(ValueError, match="differs"):
        prepare(out, plan, "train", 1, 1, True)
    provenance = json.loads((out / "provenance.json").read_text())
    provenance["action"] = "profile"
    (out / "provenance.json").write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="differs"):
        prepare(out, plan, "train", 0, 1, True)
    (out / "results").mkdir()
    (out / "results/run.json").write_text("{}")
    with pytest.raises(FileExistsError):
        prepare(out, plan, "train", 0, 1, True)
