"""Global RNG seeding.

``set_seed`` must run before model construction — every
``kaiming_normal_``/``trunc_normal_``/``xavier_uniform_`` call inside the
branches draws from the same global torch RNG stream, in the order
``SpectralQuadNet.__init__`` constructs them, so a fixed weight
initialisation depends on calling ``set_seed(cfg.seed)`` in this exact
position:

    load config → set_seed(cfg.seed) → DataStore → SpectralQuadNet → ModelEMA

Note that ``set_seed`` deliberately leaves cuDNN **non**-deterministic
(``benchmark=True``), trading bit-exact reproducibility for the faster
autotuned convolution algorithms. Regression tests that need bit-exact
results opt into determinism separately via
:func:`enable_deterministic_algorithms`.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and torch's (CPU + all-GPU) RNGs.

    Leaves cuDNN's algorithm selection non-deterministic (``benchmark=True``)
    for speed; see the module docstring for the reproducibility trade-off
    this implies and for the call-order requirement relative to model
    construction.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def enable_deterministic_algorithms(warn_only: bool = False) -> None:
    """Force deterministic kernels — for test harnesses only, never for training.

    Turning this on during training would change (and slow) cuDNN's
    algorithm selection relative to a normal training run, so it is kept
    strictly opt-in and separate from :func:`set_seed`.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
