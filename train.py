#!/usr/bin/env python3
"""Training entrypoint.

Usage
─────
    python train.py                                   # the default experiment
    python train.py data.split_fold=1 seed=1          # another protocol cell
    python train.py single.max_lr=1e-4                # single override
    python train.py -m seed=0,1,2                     # sweep
    python train.py tracking.backend=multi tracking.backends=[console,wandb]

    python train.py --config-name experiment/quadnet_audited   # the control arm

Multi-GPU (NVIDIA)::

    torchrun --standalone --nproc_per_node=2 train.py

DDP with ``SyncBatchNorm``, so the two-GPU run computes the **same** function as
the one-GPU run: the batch is split, but the normalisation statistics are
all-reduced back to the global batch's, and the gradient average over equal
shards is the global-batch gradient. Nothing about it engages under a plain
``python train.py``.

What this file is
─────────────────
A composition root and nothing else. It joins the process group, sets up
logging, builds the tracker, and hands a
:class:`~spectralquadnet.engine.pipelines.context.RunContext` to whichever
curriculum ``cfg.pipeline`` names. Everything that used to live here — the
stage sequencing, the loader construction, the resume logic — is in
``spectralquadnet.engine.pipelines``, because there are now two curricula and
four A8 arms and they have to share their setup exactly or the comparison
between them measures the setup instead.

The protocol is a config choice and is **printed at startup**, because the same
code produces two numbers that mean different things: a patch-level split
reports how well the model recognises acquisition bundles it has already seen,
and a bundle-disjoint one reports how well it recognises rice varieties. The
default composition is the second. See CHANGES.md §4 and §19.

Console output is append-only: one line per epoch, written once, mirrored into
``output_dir/training.log``.
"""

from __future__ import annotations

# First, and before `torch`: importing this module installs the warning filters,
# and the loudest warning it silences is emitted by `import torch` itself
# (`torch/cuda/__init__.py` imports `pynvml`, which warns about its own rename).
# A filter installed further down the file cannot unprint it.
# isort: off
import spectralquadnet.utils.warning_filters  # noqa: F401

# isort: on
import logging
import os
import sys
import traceback
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from spectralquadnet.config.schema import register_configs
from spectralquadnet.engine.pipelines import build_run_context, resolve_pipeline
from spectralquadnet.tracking import build_tracker
from spectralquadnet.tracking.base import ExperimentTracker, NullTracker
from spectralquadnet.utils.device import empty_cache, resolve_device
from spectralquadnet.utils.distributed import DistContext, init_distributed, shutdown
from spectralquadnet.utils.warning_filters import route_warnings_to_logging

# Makes the dataclass schemas available to Hydra's composition, so a typo in a
# YAML field fails at startup instead of hours into training.
register_configs()

#: The logger the console tracker mirrors its human-channel lines to.
_CONSOLE_MIRROR = "spectralquadnet.console"


def _setup_logging(cfg: DictConfig, dist: DistContext) -> None:
    """Point stdout, ``training.log`` and the warnings machinery at one another.

    Three streams have to end up in two places without any of them duplicating
    or interleaving:

    * **Module logs** (``utils/device.py``, ``data/mmap_store.py``, the fatal
      traceback) go to the root logger → terminal *and* file.
    * **The tracker's own lines** — banners, epoch lines, diagnostic blocks —
      are already written to stdout by the console backend, which formats and
      flushes them itself. They are mirrored to the ``spectralquadnet.console``
      logger, which carries **only** the file handler and does not propagate, so
      the file gets the run and the terminal does not get it twice.
    * **Warnings** that survive ``silence_known_warnings`` are routed through
      ``logging`` rather than raw stderr, so an unexpected one arrives as its
      own formatted line instead of splicing itself into an epoch line.

    Only the main rank writes a file: ``training.log`` is rank 0's record, and
    every rank appending to it would interleave copies of the same run.
    """
    file_handler: logging.Handler | None = None
    handlers: list[logging.Handler] = []
    if dist.is_main:
        file_handler = logging.FileHandler(os.path.join(cfg.output_dir, "training.log"))
        handlers.append(file_handler)
    handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=logging.INFO if dist.is_main else logging.WARNING,
        format=(
            ("%(asctime)s | %(levelname)s | %(message)s")
            if not dist.enabled
            else (f"%(asctime)s | rank{dist.rank} | %(levelname)s | %(message)s")
        ),
        handlers=handlers,
        force=True,  # @hydra.main installs its own handlers first
    )
    route_warnings_to_logging()

    mirror = logging.getLogger(_CONSOLE_MIRROR)
    # Cleared rather than appended to: `basicConfig(force=True)` closes the
    # previous root handlers, and a stale one left here would write to a closed
    # file for the rest of the run.
    mirror.handlers.clear()
    mirror.setLevel(logging.INFO)
    mirror.propagate = False
    if file_handler is not None:
        mirror.addHandler(file_handler)


@hydra.main(config_path="configs", config_name="experiment/seednet_grouped", version_base=None)
def main(cfg: DictConfig) -> None:
    """Compose the config, set up logging and the tracker, then run the pipeline.

    Under ``torchrun`` the process group is joined **first** — before the model,
    the store or the tracker exist — because every rank has to own its CUDA
    device before anything allocates on it. Only rank 0 gets a tracker; the
    others get a :class:`~spectralquadnet.tracking.base.NullTracker`, so two
    GPUs do not write two interleaved copies of the same epoch line into one
    terminal, and rank 0's ``training.log`` stays the record of the run.

    Any exception escaping the pipeline is logged at ``critical`` and exits with
    status 1, so a crashed run is visible in both the log file and the process
    exit code — which is what lets ``scripts/run_protocol.py`` tell a failed
    cell from a finished one.
    """
    # `cfg.device` is the fallback, not a suggestion: without a launcher this is
    # the only thing that chooses, and `device=cpu` has to keep meaning CPU.
    # Under `torchrun` the local rank wins, because a rank must own exactly one
    # device and nothing else can be true.
    dist = init_distributed(cfg.runtime, fallback=resolve_device(cfg.device))

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    _setup_logging(cfg, dist)

    # Built before `build_run_context`, which is where `set_seed` lives: a
    # tracker backend that consumed the global RNG between the seed and the
    # model construction would move every initial weight.
    tracker: ExperimentTracker = build_tracker(cfg) if dist.is_main else NullTracker()
    if dist.is_main:
        tracker.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]
    try:
        run_pipeline = resolve_pipeline(str(cfg.pipeline))
        run_pipeline(build_run_context(cfg, tracker, dist))
    except Exception:
        logging.critical("FATAL:\n" + traceback.format_exc())
        sys.exit(1)
    finally:
        tracker.close()
        empty_cache()
        shutdown(dist)


if __name__ == "__main__":
    main()
