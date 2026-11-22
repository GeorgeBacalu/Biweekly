---
title: Biweekly Season 2 Cycle 1 Entry
status: final
created: 2026-11-23
updated: 2026-11-23
---

# Biweekly Season 2 Cycle 1 Entry

## Purpose and outcome

George needs a reproducible decision on whether to submit a Cycle 1 predictor. The implementer needs acceptance criteria for the predictor, experiment, and local validation command. The first experiment tests whether weighted feature agreement adds useful signal. Competitive success is admission to AlphaNova’s settled quality set.

Cycle 1 submissions close **1 December 2026 at 00:00 UTC** (02:00 in Bucharest). The live window closes **1 January 2027 at 00:00 UTC**. [AlphaNova Season 2 overview](https://www.alphanova.tech/competition/biweekly) · [Competition timeline](../../../../requirements/COMPETITION.md#timeline)

## Contestant workflow

**UJ-1 — Decide whether to submit.** George runs the exact candidate file through offline validation and reviews the preflight, behavior, resource use, full walk-forward comparison, and novelty evidence. If the command reports ready, he uploads that file to AlphaNova and links the entry verdict and settlement to the file digest. If it reports hold, he saves the attempt, addresses the cause, and reruns validation before submitting.

## Terms

- **Net Sharpe** — Sharpe after the shipped scorer’s one-hour position lag and 5 bp L1 turnover charge on the supplied returns.
- **Matched control** — The same predictor with the weighted vote disabled.
- **Nearest-city distance** — The candidate’s angular distance to the nearest published city on the official Season 2 map.
- **Quality set** — The signals AlphaNova admits at settlement under its Sharpe and pairwise correlation rules.

## Functional requirements

### Candidate and experiment

- **FR-1 — Produce a valid submission.** One Python predictor consumes the six supplied hourly features and returns a finite, zero-sum position vector for all 20 assets at each supplied hour. The output preserves the input timestamps, row order, and asset column order.
- **FR-2 — Isolate the weighted vote.** Compare the weighted feature-agreement candidate, matched control, and shipped `demo_engineered.py` baseline on the full walk-forward. The candidate and matched control share ranks, learner settings, seed, and position shaping. The result includes the unrounded net Sharpe for each artifact and the candidate-minus-control difference.

### Local submission decision

- **FR-3 — Verify comparable evidence.** One offline command requires a quick preflight that confirms the candidate loads and returns finite, input-aligned, zero-sum positions. It requires passing model-behavior and resource checks. Each full walk-forward run must include every expected period with its expected observation count for the data snapshot. All three full runs use the same kit, data snapshot, resolved runtime, and scoring settings.
- **FR-4 — Verify novelty.** Require all expected candidate gauge rows and a finite, unrounded nearest-city distance against the official map current at decision time. Record the map snapshot and nearest city.
- **FR-5 — Decide whether to submit.** The command reports ready when FR-3 and FR-4 pass, the candidate’s finite net Sharpe is positive and strictly above the shipped baseline, and nearest-city distance is at least **60°**. Otherwise it reports hold and identifies failed checks or missing evidence.

### Evidence and handoff

- **FR-6 — Preserve each attempt.** Record completed and failed attempts with exact artifact digests, kit and input snapshots, runtime and settings, and map snapshot when used. Preserve distinct quick, full, and novelty results, with any observed coverage, per-call timings, and peak memory. Retain the failure reason on failed runs. Link all candidate checks to the same digest.
- **FR-7 — Record the contest outcome.** The contestant submits and retains the exact checked candidate file. Link the official entry verdict and any available settled Sharpe and quality-set membership to its digest and Cycle 1.

## Quality and resource requirements

- **NFR-1 — CPU budget.** Each training call must finish within 240 seconds and each prediction call within 60 seconds. Keep measured peak memory below 8 GB. Repeated retraining must remain feasible under the official gate.
- **NFR-2 — Causality and repeatability.** Fit from the supplied Season 2 training features and labels. Forecast with observations available by the scored hour. Treat ticker identities as period-specific and use city maps only for offline validation. Predictions agree across repeated calls, after reload, and on shared timestamps when padding preserves the available history. The predictor tolerates permuted gate rows, variable lengths, and inference frames that overlap training. The local check for extended or frequent near-zero exposure must pass under a documented criterion. AlphaNova determines the official verdict. [Entry checks](../../../../requirements/COMPETITION.md#entry-checks-and-the-overfitting-gate)

## Scope

The first release covers the weighted feature-agreement experiment, local readiness, and manual Cycle 1 handoff and outcome recording. Non-goals include notebook submissions, hosted orchestration, changes to the shipped kit or organizer scoring, reconstruction of concealed data, later-cycle signal builds, rank-change and turnover-only experiments, and a separate local holdout.

## Success measures

- **SM-1 — Local readiness.** The exact candidate passes FR-3 through FR-5 before the deadline.
- **SM-2 — Official outcome.** AlphaNova accepts the entry and admits it to the settled quality set (FR-7). Admission requires positive official Sharpe and absolute time-averaged cross-sectional correlation of at most 0.5 with every signal already selected for the cycle and every signal in the legacy pot. [Quality signal selection](../../../../requirements/COMPETITION.md#quality-signal-selection)
- **SM-C1 — Coverage and attempt history.** Track missing scoring coverage and every trial, including failures (FR-3, FR-6). The trial history shows how often the same data informed candidate selection.

[Product brief](../../briefs/brief-Biweekly-2026-11-22/brief.md) · [Spec](../../../specs/spec-biweekly/SPEC.md) · [Validation checklist](../../../specs/spec-biweekly/validation.md) · [Implementation addendum](addendum.md)
