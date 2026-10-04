# S19 · Next-generation rice recognition: evidence, prior art and strategy

| | |
|---|---|
| Status | **Complete research synthesis and design; proposed models untrained** |
| Dates | 2026-10-03 → 2026-10-04; literature cutoff 2026-10-03 |
| Repository state | HEAD `52fba4f4698e0a8948b00a4a83f3893ae793e55b`; pre-existing S16 documentation changes preserved |
| Data audited | 8,624 kernels, 90 varieties, 180 scans,nine sessions; reflectance 215 metadata, local k32 cube, raw archive/RGB headers |
| Historical scope | S00–S16; S17 execution and S18 reading remain reserved by S16 |
| New computation | Descriptive CPU metadata/archive/calibration audits; no fitting or new held-out scoring |
| Evidence | [tables, code and bibliography](../../evidence/S19_next_generation_strategy/) |
| Registers | F90–F96; D38–D40; H26–H31 draft; FW-39–FW-43 |
| Master plan / handoff | [plan](../../MASTER_RESEARCH_PLAN.md), [progress](../../RESEARCH_PROGRESS.md) |

## 1. Question and trigger

Which combination of information, architecture, acquisition design and evaluation
has the strongest realistic path to better90-variety recognition and a substantial
paper? The user's request follows S16's replicated v5 gain but persistent failure
on most cross-session destinations. This study reassesses F17–F19 (band findings),
F63–F71 (representation/shift), F84–F89 (v5), D27 (route), D35–D37 (reference/queue)
and FW-12/18/27–29. It does not presume that the existing architecture should win.

## 2. Executive conclusion

**Keep v5 as the reference; develop a substantially redesigned candidate that
combines high-resolution RGB with compact, wavelength-aware HSI representations.
Pair the method work with acquisition diversity and a locked new-session test.**
The critical scientific opportunity is identifying and preserving transferable seed
information while controlling acquisition effects. Generic attention, more capacity
or more training are poorly justified main directions.

- **Established:** early high scores were compromised by protocol/selection problems.
  Fit-first training alone did not solve generalisation. Lean spatial repair did:
  v5 grouped TTA macro-F1 **.570816**, stratified **.744788**; same/cross-session
  macro-recall **.682841/.199401**, training-session attraction **.412773**. Fresh-seed
  grouped improvement over X1 is **+.046839**; its interval excludes zero. The
  within-acquisition improvement did not pass the frozen fresh-seed hypothesis.
- **Dominant unresolved problem:** representation under acquisition shift. Every
  grouped training variety has one session; all 17 cross-session varieties bridge
  session8. Better overall scores do not establish all-session or biological-lot
  robustness. The dataset does not identify a numeric attainable performance ceiling.
- **New opportunities are concrete:** the raw archive is now present and all 180 HSI
  scans have matching 4896×3264 RGB image headers. Full 215 cube extraction remains
  needed. RGB object correspondence is not yet validated.
- **Band choice remains open:** historical 32 and 64 overlap at only 8 bands; full 215
  also adds 20 bands below 430 nm. Test nested budgets and 195-vs215 with a fixed spectral
  representation; use the smallest noninferior deployment input. Do not infer that
  all bands are useless from one proxy or a differently strided v5 stem.
- **Proposed system:** masked pretrained RGB features plus a compact physical-
  wavelength encoder of HSI spectra/regions, preserved morphology and simple fusion.
  Add measured-nuisance correction/consistency only after independent calibration
  evidence. Retain HSI-only and RGB-only as serious possible winners.
- **Paper route:** a reproducible acquisition-aware benchmark plus one validated
  representation/nuisance mechanism and locked external acquisition evidence.
  Current literature already covers RGB–HSI fusion and gated networks; superiority,
  method novelty and Q1 acceptance are not established by this proposal.

## 3. Reading map

| Deliverable | Content |
|---|---|
| [Repository review](repository_review.md) | Acquisition/calibration, every study S00–S16, current/historical models, failures, band/evaluation audit and identifiability argument |
| [Prior art](prior_art.md) |39 primary-source records interpreted by comparability and mechanism; current work through September2026; access limits explicit |
| [Architecture proposal](architecture_proposal.md) | Mathematical/data-flow specification, controls, transfer options, rejected/deferred ideas and reversal conditions |
| [Experiments and paper](experiments_and_paper.md) | Data gates, bounded compute portfolio, S17 relationship, crossed acquisition plan, statistics, claims and manuscript route |
| [Source ledger](../../evidence/S19_next_generation_strategy/literature.csv) | Bibliographic identity, verified link/date, protocol scope and applicability for each source |
| [BibTeX](../../evidence/S19_next_generation_strategy/references.bib) | Working citations for later writing; some large author lists abbreviated explicitly |

## 4. New evidence and its strength

**Directly checked metadata/source facts:**180 paired scan stems and RGB dimensions;
73/17 same/cross support; every cross edge touches8; train/full additive design ranks
90/97 of 99; non-nested 32/64; 20 blue-edge bands; a 102.667 nm valid-axis hole; three
retained session-filled white references in 64/215 and none in 32. See
[audit_summary](../../evidence/S19_next_generation_strategy/audit_summary.json),
[calibration sources](../../evidence/S19_next_generation_strategy/calibration_sources.json),
[split audit](../../evidence/S19_next_generation_strategy/split_audit.json) and
[band geometry](../../evidence/S19_next_generation_strategy/band_geometry.csv).

**Existing confirmed predictive evidence:** S16's saved aggregate results and its
integrity analysis, not a new independent rerun. **Mathematical consequence:** under
current training support S=g(Y), unconditional representation independence from
session conflicts with perfect class prediction. **Hypotheses:** usefulness of RGB,
full spectra, physical wavelength encoding and standards-based nuisance handling.
Do not label those hypotheses E3 model evidence or describe the proposed architecture
as already outperforming another method.

## 5. Decisions and reversal rules

D38 selects complementary representation research over a long incremental SeedNet
sweep. Reverse if verified RGB/full-spectrum probes and a bounded tuning trial show
no useful gain, or correspondence cannot be made reliable. D39 requires a controlled
band-information test outside the native-v5 continuation gate; return to32 if added
measurements repeatedly fail. D40 makes independent acquisition evidence central
to broad robustness claims; without it narrow the paper's estimand, not its honesty.
See detailed gates and metrics in the experiment plan; H26–H31 are draft, not frozen.

## 6. Threats and open facts

Current held-out acquisitions have informed multiple research rounds. Fresh seeds
support stochastic replication, not a globally fresh test. Cross-session varieties
are a selected subset and may differ biologically; there is no randomized session
counterfactual for all 90. Raw zip checksum/payload validation, individual RGB pairing,
full-band rebuild, external checkpoint licenses and final journal quartile verification
remain implementation/submission tasks. Some prior-art methods are paywalled or
poorly specified; this review marks those gaps rather than inventing protocols.
No current data establishes generalisation across instruments, years or seed lots.
No prediction of an absolute95% accuracy target is justified.

## 7. Reproduce and resume

From the repository root, with numpy/pandas/Pillow/matplotlib and project dependencies:

```sh
python docs/research/evidence/S19_next_generation_strategy/code/audit_repository.py
python docs/research/evidence/S19_next_generation_strategy/code/build_literature.py
python docs/research/evidence/S19_next_generation_strategy/code/draw_figures.py
python docs/research/evidence/S19_next_generation_strategy/code/validate_study.py
```

Audit took about3 seconds after font-cache creation on this machine; it reads metadata
and archive image headers, not all 17GB of payloads. Literature generation is offline:
it materializes manually verified records, not a live re-search. Figures regenerate
from tracked evidence without the original dataset; `tools/build_assets.py` now calls
`fig_s19` for whole-log figure builds. Audit/ledger scripts write directly into this
study's evidence directory, like the existing S09 analysis workflow.

The next executable work is the pairing/full-band identity gate and E0/E1 feature
probes, alongside the already-frozen bounded S17 baseline round. Before any new
confirmation, freeze a new study manifest with exact assets, split IDs, controls and
selection rules. Resume instructions are in the progress file. No training code,
model defaults, old study text or frozen experiment specification was changed by S19.
