#!/usr/bin/env python
"""Offline frozen DINOv2-S/14 features with exact checkpoint/code/transform provenance.

No download or model fitting. CPU by default; explicit CUDA is supported. The model
repository and checkpoint must already be available locally. All views retain the
same masked, aspect-preserving crop geometry. Source-code hashes are recorded.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from spectralquadnet.data.prep.multimodal import sha256, write_json


def transform_rgb(images: torch.Tensor, view: str) -> torch.Tensor:
    """Create controlled RGB/grayscale/low-resolution/silhouette inputs in [0,1]."""
    x = images.float().permute(0, 3, 1, 2) / 255
    if view == "gray":
        weights = x.new_tensor([0.299, 0.587, 0.114])[None, :, None, None]
        x = (x * weights).sum(1, keepdim=True).expand(-1, 3, -1, -1)
    elif view == "rgb32":
        x = F.interpolate(
            F.interpolate(x, size=(32, 32), mode="area"),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )
    elif view == "silhouette":
        x = (x.sum(1, keepdim=True) > 0).float().expand(-1, 3, -1, -1)
    elif view != "rgb":
        raise ValueError(f"Unknown RGB view: {view}")
    mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return (x - mean) / std


def run(args: argparse.Namespace) -> None:
    import json

    manifest = json.loads((args.data / "MANIFEST.json").read_text())
    if not manifest["complete"]:
        raise ValueError("Incomplete dataset")
    if sha256(args.data / "rgb.npy") != manifest["files"]["rgb.npy"]["sha256"]:
        raise ValueError("RGB asset checksum mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    source = {
        str(p.relative_to(args.model_repo)): sha256(p)
        for p in sorted(args.model_repo.rglob("*.py"))
    }
    provenance: dict[str, Any] = {
        "model": "dinov2_vits14",
        "checkpoint_sha256": sha256(args.checkpoint),
        "source_hashes": source,
        "asset_manifest_sha256": sha256(args.data / "MANIFEST.json"),
        "script_sha256": sha256(Path(__file__)),
        "torch": torch.__version__,
        "device": args.device,
        "threads": args.threads,
        "batch_size": args.batch_size,
        "numpy": np.__version__,
        "license": "Apache-2.0",
        "model_card": "https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md",
        "training_data": "LVD-142M; source-level rice-dataset overlap not independently auditable",
        "embedding": "384-dimensional class token; eval; no TTA; no fitting; float32",
        "transform": "224 masked square crop; RGB / BT.601 grayscale / 32px area then bilinear / binary silhouette; ImageNet normalization",
    }
    torch.set_num_threads(args.threads)
    model = torch.hub.load(  # type: ignore[no-untyped-call]
        str(args.model_repo), "dinov2_vits14", source="local", pretrained=False
    )
    model.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=True
    )
    model = model.eval().to(args.device)
    images = np.load(args.data / "rgb.npy", mmap_mode="r")
    for view in args.views:
        target = args.output / (view + ".npy")
        meta = args.output / (view + ".json")
        if target.exists() or meta.exists():
            raise FileExistsError(f"{view} cache already exists; use a new output folder")
        start = time.monotonic()
        temporary = args.output / (view + ".partial.npy")
        features = np.lib.format.open_memmap(  # type: ignore[no-untyped-call]
            temporary, mode="w+", dtype=np.float32, shape=(len(images), 384)
        )
        with torch.inference_mode():
            for begin in range(0, len(images), args.batch_size):
                x = torch.from_numpy(
                    np.array(images[begin : begin + args.batch_size], copy=True)
                ).to(args.device)
                result = model(transform_rgb(x, view)).cpu().numpy()
                if not np.isfinite(result).all():
                    raise ValueError("Nonfinite embedding")
                features[begin : begin + len(result)] = result
                if begin % (args.batch_size * 64) == 0:
                    print(f"{view}: {begin}/{len(images)}", flush=True)
        features.flush()
        del features
        temporary.replace(target)
        write_json(
            meta,
            provenance
            | {
                "view": view,
                "sha256": sha256(target),
                "seconds": time.monotonic() - start,
                "n": len(images),
            },
        )
        print(f"{view} complete in {time.monotonic() - start:.1f}s", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("dataset_rgb_hsi_v3"))
    p.add_argument("--output", type=Path, default=Path("outputs/s20_rgb_features_v3"))
    p.add_argument("--model-repo", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--views",
        nargs="+",
        choices=["rgb", "gray", "rgb32", "silhouette"],
        default=["rgb", "gray", "rgb32", "silhouette"],
    )
    run(p.parse_args())


if __name__ == "__main__":
    main()
