---
id: SPEC-biweekly
companions:
  - validation.md
  - ../../../requirements/COMPETITION.md
  - ../../planning-artifacts/architecture/architecture-Biweekly-2026-11-22/ARCHITECTURE-SPINE.md
sources:
  - ../../planning-artifacts/briefs/brief-Biweekly-2026-11-22/brief.md
  - ../../forge/cycle-1-signal-search/forged-idea.md
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
  - **success:** A complete full walk-forward compares the candidate, matched control, and shipped `demo_engineered.py` under matching settings. Candidate net Sharpe is positive and strictly above the shipped baseline. Record the difference between the candidate and matched control as experiment evidence.

- **CAP-3**
  - **intent:** Determine whether a candidate clears the local novelty requirement.
  - **success:** A complete comparison against the official Season 2 map current at decision time produces a finite, unrounded nearest-city distance of at least 60°. Hold the candidate if the distance is below 60° or the comparison is unavailable.

- **CAP-4**
  - **intent:** Track the official outcome of the selected Cycle 1 entry.
  - **success:** The contest records the entry as accepted. Preserve the submitted version and record the settled Sharpe and quality-set result so the evidence can inform later cycles.

## Constraints

- Use the Season 2 package and shipped `Predictor` API. Leave the kit’s Python source and base signatures intact.
- Forecast causally from the supplied features. Treat ticker identities as period-specific and keep cities outside the submitted model.
- Keep training within 240 seconds and prediction within 60 seconds per call on CPU, with peak memory below 8 GB. Keep repeated retraining feasible under the official gate.
- Meet the organizer’s submission structure, row-position, flat-period, and overfitting rules. See `validation.md` for the local evidence and its coverage.
- Test weighted feature agreement first and use the full walk-forward for every experiment comparison. Follow the architecture spine for the matched experiment and the offline submission decision.

## Non-goals

- Notebook submissions, changes to the organizer’s scoring pipeline, and reconstruction of the anonymized features, target, or hidden scoring data.
- Building later-cycle entries under this spec. The evaluation workflow can be reused for a distinct signal.

## Success signal

A Cycle 1 entry clears the local net Sharpe and novelty requirements, passes official checks, and appears in the settled quality set.
