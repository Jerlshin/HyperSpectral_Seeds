# S## · <Study title>

| | |
|---|---|
| **Status** | planned · running · complete · superseded by S## |
| **Dates** | YYYY-MM-DD → YYYY-MM-DD |
| **Commits** | `abc1234` (code), `def5678` (results) |
| **Data** | which cube and band axis (refl-215 / SNV-256), which folds |
| **Code** | `src/...`, `scripts/...`, or `evidence/S##_.../code/` |
| **Raw outputs** | `outputs/<study>/` (git-ignored) |
| **Evidence snapshot** | [`evidence/S##_.../`](../../evidence/) |
| **Findings** | F## … · **Decisions** D## … · **Hypotheses** H## … |

## 1 · Question
One sentence. Then: which findings it tests, which decisions it could reverse (gate G0).

## 2 · Why we did this
The trigger, quoted with its source: the anomaly, the audit item, or the FUTURE_WORK entry.

## 3 · Hypotheses and decision rules
Copied from `HYPOTHESES.md` as frozen. If held-out data is touched: the pre-registration file and
its SHA-256.

## 4 · Method
Partitions (what each split was allowed to do), arms, nulls, the full reference, metric, seeds ×
folds, proxies or models, compute.

## 5 · Results
Figures (from `figures/S##_.../`, each with its source line) and tables. Name the split and band
axis of every number. Include the same- vs cross-session breakdown for held-out scores.

## 6 · Findings
Bulleted, each linked to its `F##` entry with its evidence strength.

## 7 · Decisions this led to
Each linked to its `D##` entry. Record any deviation from the frozen rule.

## 8 · Threats to validity
What could make the findings wrong: proxies vs the real model, n = 2 folds, calib optimism, the
session confound, unsaved intermediate numbers.

## 9 · What would change these conclusions
Concrete results, each mapped to the decision it would reverse.

## 10 · Reproduce
Exact commands, in order, with expected runtime.

## 11 · Provenance
Where every raw artifact lives and what in it is snapshotted under `evidence/`.
