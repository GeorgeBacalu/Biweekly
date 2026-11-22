---
id: SPEC-biweekly
companions:
  - validation.md
  - ../../../requirements/COMPETITION.md
sources:
  - ../../planning-artifacts/briefs/brief-Biweekly-2026-11-22/brief.md
---

# Biweekly Season 2 Competition Entry

## Why

Enter AlphaNova’s Season 2 Cycle 1 with a reproducible signal that can add distinct information and earn positive net Sharpe after turnover costs. Cycle 1 submissions close on 1 December 2026 at 00:00 UTC. [AlphaNova Season 2](https://www.alphanova.tech/competition/biweekly) · [Competition timeline](../../../requirements/COMPETITION.md#timeline)

## Capabilities

- **CAP-1**
  - **intent:** Produce hourly positions for all 20 assets from the six anonymized Season 2 features.
  - **success:** The shipped runner loads the Python submission, and its `Predictor` returns an aligned, finite, cross-sectionally de-meaned position vector for each input timestamp.

- **CAP-2**
  - **intent:** Decide whether a candidate clears the local performance bar.
  - **success:** A full walk-forward comparison over matching periods and settings shows positive candidate net Sharpe and strictly higher net Sharpe than the shipped `demo_engineered.py`. The run record supports reproducing the result.

- **CAP-3**
  - **intent:** Assess a candidate’s distinctiveness before submission.
  - **success:** Record its nearest-city distance against the current Season 2 map using `--gauge-fix`, with 60° or more as the target. Record an unavailable comparison when no valid map is present.

- **CAP-4**
  - **intent:** Track the official outcome of the selected Cycle 1 entry.
  - **success:** The contest records the entry as accepted. Preserve the submitted version and record the settled Sharpe and quality-set result so the evidence can inform later cycles.

## Constraints

- Use the Season 2 package and shipped `Predictor` API. Leave the kit’s Python source and base signatures intact.
- Forecast causally from the supplied features. Treat ticker identities as period-specific and keep cities outside the submitted model.
- Keep training within 240 seconds and prediction within 60 seconds per call on CPU, with peak memory below 8 GB. Keep repeated retraining feasible under the official gate.
- Meet the organizer’s submission structure, row-position, flat-period, and overfitting rules. See `validation.md` for the local evidence and its coverage.

## Non-goals

- Notebook submissions, changes to the organizer’s scoring pipeline, and reconstruction of the anonymized features, target, or hidden scoring data.
- Building later-cycle entries under this spec. The evaluation workflow can be reused for a distinct signal.

## Success signal

A Cycle 1 entry that clears the local net-Sharpe bar passes official checks and appears in the settled quality set.

## Open Questions

- If the local performance bar is met but the current city comparison is unavailable or comes in below 60°, should the candidate still be submitted?
