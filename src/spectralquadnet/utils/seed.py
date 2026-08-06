"""Global RNG seeding.

Relocated verbatim from ``HSI_modality_training/hsi_training.py`` @ ``886560f``:

====================  ==============
Symbol                Baseline lines
====================  ==============
:func:`set_seed`      194-200
====================  ==============

The baseline called this **at module import** (line 200, immediately after the
``CONFIG`` literal), so it always ran before ``main()`` — and therefore before
model construction. REFACTOR_PLAN.md §3.6 identifies that as the migration's
highest-risk item: every ``kaiming_normal_``/``trunc_normal_``/``xavier_uniform_``
call inside the branches draws from the same global torch RNG stream, in the
order ``SpectralQuadNet.__init__`` constructs them. Phase 4's ``train.py`` must
call ``set_seed(cfg.seed)`` in this exact position:

    load config → set_seed(cfg.seed) → DataStore → SpectralQuadNet → ModelEMA

Note that ``set_seed`` deliberately leaves cuDNN **non**-deterministic
(``benchmark=True``) — that is pre-existing behaviour, not a regression the
refactor introduces (§3.6, "honest scope"). Regression tests that need bit-exact
results opt into determinism separately via
:func:`enable_deterministic_algorithms`.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def enable_deterministic_algorithms(warn_only: bool = False) -> None:
    """Force deterministic kernels — for test harnesses only, never for training.

    REFACTOR_PLAN.md §3.6 sanctions this as "a legitimate, additive setting for a
    *test harness* rather than for training itself": the original script never
    set it, and turning it on during training would change (and slow) the cuDNN
    algorithm selection that produced the existing checkpoints.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
