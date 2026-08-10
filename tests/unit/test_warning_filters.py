"""Warning noise control.

Two properties, and the second matters as much as the first:

* the *known* third-party noise is silenced — pynvml's rename, torch's
  ``use_reentrant`` notice — because it arrives from ``warnings.warn``'s raw
  ``stderr`` write, in the middle of whichever epoch line the console was
  flushing;
* everything else still gets through. A blanket ``simplefilter("ignore")``
  would also swallow the warnings that are about *this* run — a Metal operator
  falling back to the CPU, a checkpoint written by an older schema — which is
  why the module lists messages one at a time.
"""

from __future__ import annotations

import logging
import warnings

import pytest

from spectralquadnet.utils.warning_filters import (
    route_warnings_to_logging,
    silence_known_warnings,
)

_PYNVML = (
    "The pynvml package is deprecated. Please install nvidia-ml-py instead. "
    "If you did not install pynvml directly, please report this to the maintainers "
    "of the package that installed pynvml for you."
)


def test_known_noise_is_silenced_and_nothing_else_is() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        silence_known_warnings()
        warnings.warn(_PYNVML, FutureWarning, stacklevel=1)
        warnings.warn(
            "torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly.",
            UserWarning,
            stacklevel=1,
        )
        warnings.warn(
            "The operator 'aten::_foo' is not currently supported on the MPS backend",
            UserWarning,
            stacklevel=1,
        )

    survived = [str(w.message) for w in caught]
    assert len(survived) == 1, survived
    assert survived[0].startswith("The operator 'aten::_foo'")


def test_an_extra_filter_can_be_supplied_by_the_caller() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        silence_known_warnings(extra=[(UserWarning, r"a locally known annoyance")])
        warnings.warn("a locally known annoyance, once per step", UserWarning, stacklevel=1)

    assert caught == []


def test_a_surviving_warning_is_logged_as_one_line(caplog: pytest.LogCaptureFixture) -> None:
    """Routed through ``logging``, and without the blank line ``formatwarning`` adds.

    ``captureWarnings`` logs the *formatted* warning, which ends in a newline
    because it was built to be written to a stream directly — logged as-is it
    leaves a blank line after every warning, in the terminal and in
    ``training.log`` both.
    """
    route_warnings_to_logging()
    try:
        with (
            caplog.at_level(logging.WARNING, logger="py.warnings"),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("always")
            warnings.warn("something worth reading", UserWarning, stacklevel=1)
    finally:
        logging.captureWarnings(False)

    assert caplog.records, "the warning reached logging rather than stderr"
    message = caplog.records[-1].getMessage()
    assert "something worth reading" in message
    assert message == message.rstrip(), "no trailing newline, so no blank line"
