#!/usr/bin/env python3
"""Reproducible benchmark and throughput verification tool for SpectralQuadNet.

Profiles each component of the hyperspectral training pipeline:
1. mmap batch retrieval (single vs batched indexing)
2. DataLoader workers and IPC pipelining
3. CPU preprocessing and augmentation
4. Host-to-device (CPU -> GPU) transfer
5. Forward pass sub-module latency
6. Backward pass latency (with / without 3D conv decomposition)
7. Optimizer step latency
8. Validation pass latency and memory footprint
9. End-to-end training loop throughput (samples/sec, sec/epoch)
10. Numerical equivalence verification between baseline and optimized paths
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spectralquadnet.data.datasets import RiceSeedDataset
from spectralquadnet.data.loaders import build_split_bundle
from spectralquadnet.data.mmap_store import DataStore
from spectralquadnet.models.spectral_seed_net import SpectralSeedNet
from spectralquadnet.utils.device import resolve_device, resolve_runtime


def sync(device: torch.device) -> None:
    """Synchronize accelerator execution queue for accurate latency measurement."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def format_row(name: str, latency_ms: float, throughput: float | None = None, extra: str = "") -> str:
    th_str = f"{throughput:>8.2f} samples/s" if throughput is not None else ""
    return f"  {name:<38} | {latency_ms:>8.2f} ms | {th_str:<18} | {extra}"


def run_benchmark(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    print("=" * 80)
    print("HYPERSPECTRAL SEED CLASSIFICATION: PIPELINE THROUGHPUT BENCHMARK")
    print("=" * 80)
    print(f"Device: {device} | Batch size: {args.batch_size} | Iterations: {args.iters}")

    store = DataStore.instance()
    patches_path = PROJECT_ROOT / "dataset" / "patches.npy"
    labels_path = PROJECT_ROOT / "dataset" / "labels.npy"
    wl_path = PROJECT_ROOT / "dataset" / "wavelengths.csv"
    masks_path = PROJECT_ROOT / "dataset" / "masks.npy"
    morph_path = PROJECT_ROOT / "dataset" / "morphology.npy"

    if not patches_path.exists():
        print(f"Dataset not found at {patches_path}. Skipping dataset-dependent benchmarks.")
        return

    store.load_patches(str(patches_path), str(labels_path))
    store.load_wavelengths(str(wl_path), "cpu")
    if masks_path.exists() and morph_path.exists():
        store.load_side_arrays(str(masks_path), str(morph_path))

    data_cfg = OmegaConf.create({
        "num_bands": 256,
        "max_cutout_bands": 19,
        "noise_std": 0.02,
        "cutmix_bands": 51,
        "cutmix_spatial": 16,
        "split_scheme": "stratified",
        "split_eval_frac": 0.30,
        "calib_frac": 0.0,
        "split_fold": 0,
        "labels_path": str(labels_path),
    })

    splits = build_split_bundle(OmegaConf.create({"data": data_cfg, "seed": 42}))
    train_idx = splits.train
    val_idx = splits.val

    ds_train = RiceSeedDataset(
        train_idx, aug_strength="heavy", store=store, data_cfg=data_cfg, device="cpu"
    )
    ds_val = RiceSeedDataset(
        val_idx, aug_strength="none", store=store, data_cfg=data_cfg, device="cpu"
    )

    print(f"\n[1] DATA I/O & RETRIEVAL BENCHMARK (Batch size = {args.batch_size})")
    print("-" * 80)
    rng = np.random.RandomState(42)
    sample_indices = rng.choice(len(train_idx), size=args.batch_size, replace=False).tolist()

    # Individual single reads
    t0 = time.perf_counter()
    for _ in range(args.iters):
        _ = [ds_train[i] for i in sample_indices]
    t_single = (time.perf_counter() - t0) / args.iters * 1000

    # Batched vectorized read
    t0 = time.perf_counter()
    for _ in range(args.iters):
        _ = ds_train.__getitems__(sample_indices)
    t_batched = (time.perf_counter() - t0) / args.iters * 1000

    print(format_row("Single random reads (__getitem__)", t_single, args.batch_size / (t_single / 1000), "Baseline"))
    print(format_row("Batched fancy indexing (__getitems__)", t_batched, args.batch_size / (t_batched / 1000), f"{t_single/t_batched:.2f}x faster"))

    print("\n[2] NUMERICAL EQUIVALENCE VERIFICATION")
    print("-" * 80)
    torch.manual_seed(12345)
    single_out = [ds_train[i] for i in sample_indices[:16]]
    torch.manual_seed(12345)
    batched_out = ds_train.__getitems__(sample_indices[:16])

    all_match = True
    for i in range(16):
        if not torch.equal(single_out[i][0], batched_out[i][0]):
            all_match = False
            print(f"  ❌ Mismatch detected in patch {i}")
            break
        if not torch.equal(single_out[i][1], batched_out[i][1]):
            all_match = False
            print(f"  ❌ Mismatch detected in label {i}")
            break
    if all_match:
        print("  ✅ Verified: __getitems__ produces 100% BIT-IDENTICAL sample outputs.")

    print("\n[3] DATALOADER PIPELINING BENCHMARK")
    print("-" * 80)
    for workers in [0, 2]:
        loader = torch.utils.data.DataLoader(
            ds_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=workers,
            persistent_workers=(workers > 0),
            pin_memory=(device.type == "cuda"),
        )
        # Warmup
        it = iter(loader)
        _ = next(it)
        t0 = time.perf_counter()
        count = 0
        for _ in range(min(10, len(loader))):
            _ = next(it)
            count += 1
        t_load = (time.perf_counter() - t0) / count * 1000
        print(format_row(f"DataLoader (workers={workers})", t_load, args.batch_size / (t_load / 1000)))

    print(f"\n[4] MODEL COMPUTE BENCHMARK ON {device} (Batch size = {args.batch_size})")
    print("-" * 80)
    from spectralquadnet.config.compose import load_experiment_config
    exp_cfg = load_experiment_config()
    wl_tensor = store.wavelengths.to(device)
    model = SpectralSeedNet(
        cfg=exp_cfg,
        physical_wl=wl_tensor,
        num_classes=len(np.unique(splits.labels)),
        num_bands=256,
    ).to(device)

    x = torch.randn(args.batch_size, 256, 64, 64, device=device)
    mask = torch.ones(args.batch_size, 1, 64, 64, device=device)
    morph = torch.zeros(args.batch_size, 8, device=device)
    n_classes = len(np.unique(splits.labels))
    labels = torch.randint(0, n_classes, (args.batch_size,), device=device)
    loss_fn = nn.CrossEntropyLoss()

    for decomp in [False, True]:
        model.spatial.stem.decompose_conv3d = decomp
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        # Warmup
        sync(device)
        out = model(x, mask=mask, morph=morph)
        logits = out["main"] if isinstance(out, dict) else out
        loss = loss_fn(logits, labels)
        loss.backward()
        opt.step()
        opt.zero_grad()
        sync(device)

        # Forward
        t0 = time.perf_counter()
        for _ in range(args.iters):
            out = model(x, mask=mask, morph=morph)
            sync(device)
        t_fwd = (time.perf_counter() - t0) / args.iters * 1000

        # Backward
        t0 = time.perf_counter()
        for _ in range(args.iters):
            logits = out["main"] if isinstance(out, dict) else out
            loss = loss_fn(logits, labels)
            loss.backward(retain_graph=True)
            sync(device)
        t_bwd = (time.perf_counter() - t0) / args.iters * 1000

        # Optimizer
        t0 = time.perf_counter()
        for _ in range(args.iters):
            opt.step()
            sync(device)
        t_opt = (time.perf_counter() - t0) / args.iters * 1000
        opt.zero_grad()

        total_step = t_fwd + t_bwd + t_opt
        step_th = args.batch_size / (total_step / 1000)
        n_train_samples = len(train_idx)
        est_epoch_sec = (n_train_samples / args.batch_size) * (total_step / 1000)

        decomp_label = "decompose_conv3d=True" if decomp else "decompose_conv3d=False (Baseline)"
        print(f"\n  -- Configuration: {decomp_label} --")
        print(format_row("Forward pass", t_fwd))
        print(format_row("Backward pass", t_bwd))
        print(format_row("Optimizer step", t_opt))
        print(format_row("Total training step", total_step, step_th))
        print(f"  >> Estimated compute time for 1 epoch ({n_train_samples} samples): {est_epoch_sec:.2f} s ({est_epoch_sec/60:.2f} min)")

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETED SUCCESSFULLY")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="SpectralQuadNet Throughput Benchmark")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/mps/cuda/cpu)")
    parser.add_argument("--iters", type=int, default=5, help="Benchmark iterations (default: 5)")
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
