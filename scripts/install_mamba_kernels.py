#!/usr/bin/env python3
"""Compile ``causal-conv1d`` and ``mamba-ssm`` against the active conda env.

Relocated from the root ``installation.py`` — REFACTOR_PLAN.md §2.1's last row,
"unchanged behavior, relocated only".

**This is not part of the training path.** The audit (§1.1) confirmed the current
``SpectralQuadNet`` imports neither ``mamba_ssm`` nor ``causal_conv1d``; the
script is kept for a future SpecMamba branch, and §7's open question 3 asks
whether to keep it at all. Nothing in ``src/spectralquadnet`` imports it.

It also assumes a CUDA Linux toolchain (``targets/x86_64-linux/{include,lib}``
under ``$CONDA_PREFIX``) and will not do anything useful on macOS.

Declared deviations from the original, neither of which changes behaviour:

* type annotations added, so ``mypy --strict`` covers ``scripts/`` (§4.4);
* the verification imports carry ``noqa: F401`` — they are import-availability
  probes, not usages, and were always dead-looking by design.

Usage
─────
    python scripts/install_mamba_kernels.py
"""

from __future__ import annotations

import os
import subprocess
import sys


def run_command(command: str) -> None:
    print(f"Executing: {command}")
    try:
        # We use shell=True to allow for environment variable expansion
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        sys.exit(1)


def install_mamba() -> None:
    # 1. Dynamically find the Conda Prefix
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        print("Error: CONDA_PREFIX not found. Please run this inside your conda environment.")
        return

    print(f"Found Conda Prefix: {conda_prefix}")

    # 2. Configure Environment Variables for the Compiler
    # This points specifically to the headers we found in your 'targets' folder
    cuda_include = os.path.join(conda_prefix, "targets/x86_64-linux/include")
    cuda_lib = os.path.join(conda_prefix, "targets/x86_64-linux/lib")

    os.environ["CUDA_HOME"] = conda_prefix
    os.environ["CPATH"] = f"{cuda_include}:{os.environ.get('CPATH', '')}"
    os.environ["C_INCLUDE_PATH"] = f"{cuda_include}:{os.environ.get('C_INCLUDE_PATH', '')}"
    os.environ["CPLUS_INCLUDE_PATH"] = f"{cuda_include}:{os.environ.get('CPLUS_INCLUDE_PATH', '')}"
    os.environ["LIBRARY_PATH"] = f"{cuda_lib}:{os.environ.get('LIBRARY_PATH', '')}"
    os.environ["PATH"] = f"{os.path.join(conda_prefix, 'bin')}:{os.environ.get('PATH', '')}"

    print("Environment variables configured for CUDA 12.x compilation.")

    # 3. Install Prerequisites
    print("Step 1/3: Installing build tools...")
    run_command("pip install wheel setuptools check-manifest packaging ninja --no-cache-dir")

    # 4. Install Causal Conv1d
    # This is the heavy lifting for the 1D spectral convolutions
    print("Step 2/3: Building causal-conv1d (this takes 2-5 minutes)...")
    run_command("pip install causal-conv1d>=1.4.0 --no-build-isolation --no-cache-dir")

    # 5. Install Mamba SSM
    print("Step 3/3: Building mamba-ssm...")
    run_command("pip install mamba-ssm --no-build-isolation --no-cache-dir")

    # 6. Final Verification
    print("\n" + "=" * 40)
    print("Verification: Testing Mamba and Causal Conv1d Imports")
    try:
        import causal_conv1d_cuda  # noqa: F401 - import-availability probe
        import torch  # noqa: F401 - import-availability probe
        from mamba_ssm import Mamba  # noqa: F401 - import-availability probe

        print("SUCCESS: All Mamba kernels are successfully compiled and linked!")
    except ImportError as e:
        print(f"VERIFICATION FAILED: {e}")
        print("Tip: Try restarting your terminal and activating your environment again.")


if __name__ == "__main__":
    install_mamba()
