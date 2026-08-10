"""Third-party warning noise, filtered at one place with a reason per entry.

A training run's console is a log of *this* run. A ``FutureWarning`` raised
inside ``pynvml`` because torch asked it for a memory figure is not about this
run, and it arrives at the worst possible moment: ``warnings.warn`` writes
straight to ``sys.stderr``, outside the tracker's line discipline, so it lands
in the middle of whichever epoch line stdout was flushing — the exact
interleaving the console backend's per-line flush exists to prevent.

Every filter here is an **exact, categorised, module-scoped** ignore with a
comment saying what emits it and why silencing it is safe. Nothing here does
``simplefilter("ignore")``: a blanket ignore would also swallow the warnings
that *are* about this run — a Metal operator falling back to the CPU, a loader
oversubscribing the machine, a checkpoint written by an older schema.

Two entry points, because a run is more than one process:

* :func:`silence_known_warnings` — run **at import of this module**, and again
  by each ``DataLoader`` worker through
  :func:`~spectralquadnet.data.loaders.seed_worker`. Under ``spawn`` (the
  default on macOS) a worker starts with a fresh warnings registry, so filters
  installed in the parent do not reach it.
* :func:`route_warnings_to_logging` — makes anything *not* silenced arrive as a
  formatted log record on the same handlers as everything else, instead of as a
  raw ``stderr`` write. That is what keeps an unexpected warning on its own
  line, in both the terminal and ``training.log``.

Why the import side effect
──────────────────────────
The filters have to be installed **before ``torch`` is imported**, because the
loudest of them is emitted by that import: ``torch/cuda/__init__.py`` does
``import pynvml`` at module scope, and pynvml's ``FutureWarning`` about its
rename is raised there — before any statement in ``train.py``'s body has run.
A filter installed afterwards cannot unprint it. Importing this module is
therefore the installation, so an entry point only has to import it *first*
(see the top of ``train.py``) rather than remember to call something.

The functions are idempotent: ``warnings.filterwarnings`` prepends, and calling
it twice with the same arguments leaves two equivalent entries rather than
changing behaviour.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence

#: ``(category, message-regex)`` — each an anchored, deliberate silence.
#: ``message`` is matched case-insensitively against the *start* of the warning
#: text (``re.match`` semantics, which is what ``warnings`` uses).
#:
#: Deliberately **not** scoped by ``module``: ``warnings`` matches that argument
#: against the raising frame's *source path minus* ``.py`` — something like
#: ``/opt/…/site-packages/torch/cuda/__init__`` — and not against the dotted
#: name, so a natural-looking ``module=r"torch\\.cuda.*"`` is an anchored regex
#: that can never match and a filter that silently never fires. The message
#: patterns below are specific enough to stand on their own.
_KNOWN: tuple[tuple[type[Warning], str], ...] = (
    # `pynvml` renamed itself to `nvidia-ml-py` and warns from its own import,
    # which torch performs at `import torch` (and again from any CUDA memory
    # probe). Once per process, and about the environment rather than the run.
    (FutureWarning, r"The pynvml package is deprecated"),
    (UserWarning, r"The pynvml package is deprecated"),
    (DeprecationWarning, r"The pynvml package is deprecated"),
    # `torch.cuda.amp.GradScaler`/`autocast` deprecations. We already call the
    # `torch.amp` spelling with an explicit `device=`; the shim still warns from
    # inside torch when a dependency touches the old path.
    (FutureWarning, r"`?torch\.cuda\.amp\.\w+"),
    # `torch.load` defaulting to `weights_only=False`. Every checkpoint this
    # code loads is one it wrote itself, in `engine/checkpoint.py`.
    (FutureWarning, r"You are using `torch\.load` with `weights_only=False`"),
    (UserWarning, r"TypedStorage is deprecated"),
    # `torch.utils.checkpoint` without an explicit `use_reentrant`. Branch A's
    # activation checkpointing (`runtime.checkpoint_branch_a`) reaches this on
    # every forward pass of every step — the noisiest entry in the list.
    (UserWarning, r"torch\.utils\.checkpoint: the use_reentrant parameter"),
    (UserWarning, r"None of the inputs have requires_grad=True"),
    # `LambdaLR`/`SGDR` warn that `scheduler.step()` was called before
    # `optimizer.step()` — which is what an epoch whose first batches were all
    # skipped as non-finite looks like — and about the removed `epoch` argument.
    # Both loops step in the documented order; `stage1_progressive.py` already
    # suppresses the construction-time variant locally.
    (UserWarning, r"Detected call of `lr_scheduler\.step\(\)`"),
    (UserWarning, r"The epoch parameter in `scheduler\.step\(\)`"),
    # Raised per `DataLoader` construction when the worker count is one the
    # machine can technically host but did not suggest; `RuntimePlan` chooses it
    # deliberately.
    (UserWarning, r"This DataLoader will create \d+ worker processes"),
)


def silence_known_warnings(extra: Sequence[tuple[type[Warning], str]] = ()) -> None:
    """Install the filters in :data:`_KNOWN` (plus ``extra``) on this process.

    Safe to call from a ``DataLoader`` worker, a spawned rank, or twice in the
    same process. Called once at import — see the module docstring for why that
    is the only ordering that catches the ``pynvml`` warning.

    Args:
        extra: Additional ``(category, message-regex)`` pairs, for a caller that
            knows about noise this module does not.
    """
    for category, message in (*_KNOWN, *extra):
        warnings.filterwarnings("ignore", message=message, category=category)


class _OneLine(logging.Filter):
    """Strip the trailing newline ``warnings.formatwarning`` bakes into its text.

    ``captureWarnings`` logs the *formatted* warning, which ends in ``\\n``
    because it was built to be written to a stream directly. Logged as-is it
    emits a blank line after every warning — in the terminal *and* in
    ``training.log`` — which is precisely the ragged output this refactor
    removes everywhere else.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = record.msg.rstrip()
        return True


def route_warnings_to_logging() -> None:
    """Send surviving warnings through ``logging`` instead of raw ``stderr``.

    ``logging.captureWarnings`` redirects :func:`warnings.showwarning` to the
    ``py.warnings`` logger, so a warning this module did **not** silence is
    formatted like every other record, lands on the same handlers (terminal and
    ``training.log``), and occupies its own line rather than splicing itself
    into whatever the console was writing.

    The logger is pinned to ``WARNING`` explicitly: ``captureWarnings`` leaves
    it at ``NOTSET``, which would inherit a root level a caller may have raised.
    """
    logging.captureWarnings(True)
    logger = logging.getLogger("py.warnings")
    logger.setLevel(logging.WARNING)
    if not any(isinstance(f, _OneLine) for f in logger.filters):
        logger.addFilter(_OneLine())


# The import *is* the installation — see the module docstring. An entry point
# imports this module before `torch` and needs to do nothing else; a worker
# process re-runs the call, because `spawn` gives it an empty filter registry.
silence_known_warnings()
