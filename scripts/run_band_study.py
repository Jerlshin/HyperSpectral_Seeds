#!/usr/bin/env python3
"""How much of the 256-band cube does this task need? — thin CLI wrapper.

Forwards to :mod:`spectralquadnet.bandstudy.cli`, which holds the whole study;
this file exists so the entry point is discoverable next to
``scripts/select_bands.py`` rather than only as a ``-m`` invocation.

    python scripts/run_band_study.py list
    python scripts/run_band_study.py all
    python scripts/run_band_study.py neural --execute

How this differs from ``scripts/select_bands.py``
─────────────────────────────────────────────────
``select_bands.py`` **produces** a reduced cube: it runs mRMR and SPA, picks one
winner and one band count by a 98%-of-peak rule, and writes a new ``.npy``. It
is a build step.

This is the **experiment** that says whether that band count is the right one.
It compares twelve methods across twenty budgets up to the full 256, quantifies
how reproducible each selection is, checks every method against a random-subset
null, and reports a recommendation with its uncertainty and its limitations.
It writes band *index* files rather than cubes — a k = 100 reduced cube is
14 GB, and this study evaluates hundreds of band sets.

Neither replaces the other. Run this to decide what the budget should be; run
``select_bands.py`` if you want a materialised cube at a budget you have already
decided on.
"""

from __future__ import annotations

import sys

from spectralquadnet.bandstudy.cli import main

if __name__ == "__main__":
    sys.exit(main())
