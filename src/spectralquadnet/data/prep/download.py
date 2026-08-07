"""Dataset acquisition."""

from __future__ import annotations

import subprocess

from spectralquadnet.data.prep.config import PrepConfig


def download(cfg: PrepConfig | None = None) -> None:
    """Download the dataset zip via resumable ``curl``, skipping if it already exists.

    Raises:
        RuntimeError: The ``curl`` invocation exited non-zero.
    """
    cfg = cfg or PrepConfig()
    cfg.ensure_root()
    zip_file = cfg.zip_file

    if zip_file.exists():
        print("Zip exists. Skipping download.")
        return

    print("Downloading dataset with curl...")

    cmd = [
        "curl",
        "-L",
        "--fail",
        "-C",
        "-",
        "--progress-bar",
        "-o",
        str(zip_file),
        cfg.data_url,
    ]

    if subprocess.run(cmd).returncode != 0:  # noqa: PLW1510 - `check=True` would raise CalledProcessError instead of this message
        raise RuntimeError("Download failed.")

    print("Download complete.")
