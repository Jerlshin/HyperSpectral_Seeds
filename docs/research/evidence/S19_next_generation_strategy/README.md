# S19 evidence manifest

Descriptive audit observed2026-10-03; synthesis completed2026-10-04. No new model
fitting or held-out prediction scoring. Provenance and input hashes are in
`provenance.json`, `inventory.csv`, `run_inventory.csv`, `frozen_plan_hashes.csv`.

| Files | Meaning |
|---|---|
| `audit_summary.json`, `array_availability.json` | Counts, local availability and explicitly conditional recall arithmetic |
| `class_session_support.csv`, `session_summary.csv`, `variety_support.csv`, `identifiability.json` | Acquisition support, missing kernels and additive-design rank |
| `split_audit.json` | Actual grouped split reports, train/calib scan sharing, rank checks |
| `band_geometry.csv`, `band_axes.json` | Actual wavelengths, band indices, coverage, gaps and cube sizes |
| `calibration_sources.json` | Own versus session-filled white references on each input axis |
| `archive_inventory.csv`, `archive_availability.json`, `rgb_scan_pairs.csv` | Zip directory and RGB-header checks; no archive-wide checksum assertion |
| `s16_summary_reference.csv` | Exact copy of already-reported S16 aggregate results, not a re-analysis |
| `literature.csv`, `literature.json`, `references.bib`, `search_log.md` |39-source primary literature ledger, bibliography and verification limits |
| `validation.json` | Artifact checks; never a model-validation result |
| `code/` | Audit, offline bibliography generation, snapshot-only figure drawing and validation |

Figure source values are fully persisted, so reproducing plots does not require the
raw archive or full cube. The archive entries' stored CRC values are metadata, not
checksums independently recomputed by S19. Running the audit again refreshes the
working-tree inventory; preserve this snapshot if comparing later repository states.
