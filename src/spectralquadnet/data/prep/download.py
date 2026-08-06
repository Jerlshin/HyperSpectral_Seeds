"""Dataset acquisition.

Relocated from ``data_setup_v3.py`` @ ``886560f``:

=====================  ==============
Symbol                 Baseline lines
=====================  ==============
:func:`download`       47-67
``DATA_URL``           30-33  (now :data:`spectralquadnet.data.prep.config.DATA_URL`)
=====================  ==============

Declared deviation: the baseline read the module-level ``ZIP_FILE``/``DATA_URL``
globals; both now arrive via :class:`~spectralquadnet.data.prep.config.PrepConfig`.
The ``curl`` invocation — including ``-C -`` for resumable downloads — is verbatim.
"""

from __future__ import annotations

import subprocess

from spectralquadnet.data.prep.config import PrepConfig


def download(cfg: PrepConfig | None = None) -> None:
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

    if subprocess.run(cmd).returncode != 0:  # noqa: PLW1510 - verbatim from baseline
        raise RuntimeError("Download failed.")

    print("Download complete.")
