"""How much of the 256-band cube does this task actually need?

The repository ships a 40-band SPA subset and, since IC-4, a per-fold 100-band
one. Neither number was chosen by an experiment that could have returned a
different answer: the 40-band curve terminates at k = 40 and the 100-band curve
terminates at k = 100, so in both cases the "elbow" is the endpoint of the
curve and the 98%-of-peak criterion is satisfied vacuously (CHANGES M-14). This
package is the experiment that can return a different answer.

What it measures
────────────────
Three questions, kept separate because they have different answers:

1. **How many bands?** A budget sweep from k = 1 to the full k = 256, so the
   curve extends past every candidate elbow by construction. Saturation,
   monotone improvement and outright deterioration are all representable
   outcomes and the analysis names whichever one the data shows.
2. **Which bands?** Selection frequency over folds, replicates and methods,
   reported in nanometres rather than indices, with the redundancy of each
   chosen set quantified.
3. **Which method?** Eleven selection strategies including two nulls — evenly
   spaced bands, and random subsets. On 256 bands with neighbour correlations
   above 0.99, a method that cannot beat a random draw of the same size has not
   been shown to select anything.

Leakage discipline
──────────────────
The same discipline the rest of the project runs under, applied one level up.
Band selection is a hyperparameter of the input representation, so choosing it
on data that is later scored is the same defect as C-1 wearing different
clothes (CHANGES §4.1, Ambroise & McLachlan PNAS 2002).

* Every selector sees **training rows of one grouped fold** and nothing else —
  not calib, not val, not test.
* Every *decision* — which k, which method — reads **calib** only.
  :mod:`spectralquadnet.bandstudy.analysis` is the module that makes
  recommendations and it is structurally incapable of reading a held-out score:
  the records it consumes are filtered by split before it sees them.
* ``val ∪ test`` is scored in a separate, opt-in ``confirm`` stage, after the
  recommendation exists, and is reported beside it as confirmation rather than
  folded into the choice.

Proxy models are not the deployed model
───────────────────────────────────────
Almost everything here is measured with classical models on foreground-masked
mean spectra, because that is what makes ~1,300 selection × budget cells
affordable at all. That representation discards spatial structure entirely, and
CHANGES F-3 is the standing prediction that its curve saturates earlier than a
network reading spatial-spectral joint structure does. So every proxy
conclusion is labelled a proxy conclusion, and the ``neural`` stage exists to
test the specific claim the proxies cannot: that the budget chosen on mean
spectra is also the budget the network wants.
"""

from __future__ import annotations

from spectralquadnet.bandstudy.config import BandStudyConfig

__all__ = ["BandStudyConfig"]
