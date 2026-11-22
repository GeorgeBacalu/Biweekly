---
title: Biweekly Season 2 Competition Entry
status: final
created: 2026-11-22
updated: 2026-11-22
---

# Biweekly Season 2 Competition Entry

## Goal

Build a reproducible Python `Predictor` for an AlphaNova Biweekly Season 2 contestant. It will use the six anonymized hourly features to produce a zero-sum position vector across 20 assets. The entry needs to pass AlphaNova’s checks, earn positive Sharpe after turnover costs, and add a distinct signal to the season’s field. [Season 2 rules](../../../../requirements/COMPETITION.md)

The first entry targets Cycle 1, which closes on 1 December 2026 at 00:00 UTC. [AlphaNova Season 2](https://www.alphanova.tech/competition/biweekly) · [Season 2 rules](../../../../requirements/COMPETITION.md#timeline)

## Validation constraints

The target’s construction is undisclosed, ticker identities rotate across periods, and a modest forecasting edge can disappear after the 5 bp turnover charge. The kit supports local walk-forward validation. AlphaNova scores additional hidden history and the cycle’s live window, may apply additional execution-cost adjustments, and runs a shuffled-label gate on submissions. [Season 2 rules](../../../../requirements/COMPETITION.md)

## First-entry scope

- Fit on the Season 2 package and use causal features that generalize across assets and periods. Submit one Python file implementing the provided `Predictor` interface. Keep the shipped kit intact. [Competition data](../../../../requirements/COMPETITION.md#competition-data) · [Submission format](../../../../requirements/COMPETITION.md#submission-format)
- Use the quick runner to verify that the file loads and positions sum to zero at each timestamp it checks. Then use `--full` to compare candidates against the shipped `demo_engineered.py` under matching settings. Record the code version, inputs, and results so the comparison can be repeated. [Starter kit](../../../../requirements/README.md#quick-start)
- Use `--gauge-fix` with the current Season 2 city map from the contest page to assess novelty offline. Record the nearest published city and its measured distance. Reserve the map for candidate selection and feed the predictor the competition features. [Signal cities and novelty](../../../../requirements/COMPETITION.md#signal-cities-and-novelty)

## Success criteria

| Milestone | Evidence |
| --- | --- |
| **Local submission decision** | Require positive net Sharpe on the full walk-forward run and a higher net Sharpe than `demo_engineered.py` on the same data and settings. Also require a measured nearest-city distance of at least 60° using the current Season 2 map. Hold the candidate if the distance is below 60° or the comparison is unavailable. Confirm zero-sum output and stay within the CPU limits of 4 minutes for training and 60 seconds per prediction call, with peak memory under 8 GB. |
| **First official milestone** | The submission passes the entry checks, including the shuffled-label overfitting gate and the row-position and flat-period checks. |
| **Competitive goal** | The entry reaches the settled quality set. The contest ranks positive-Sharpe signals and admits one only if the absolute value of its time-averaged cross-sectional correlation with each signal already in that cycle’s quality set or the season’s legacy pot is at most 0.5. The top three quality signals earn prizes. |

[Local validation](../../../../requirements/COMPETITION.md#local-validation) · [Entry checks](../../../../requirements/COMPETITION.md#entry-checks-and-the-overfitting-gate) · [Computation limits](../../../../requirements/COMPETITION.md#computation-limits) · [Quality selection](../../../../requirements/COMPETITION.md#quality-signal-selection)

## Later cycles

After a cycle closes, every entry that passed the checks joins the season’s legacy pot. Reuse the evaluation workflow and develop a distinct candidate for any later cycle. Record the official verdict and score before choosing the next candidate. [Legacy pot rules](../../../../requirements/COMPETITION.md#the-legacy-pot--why-each-cycle-needs-new-alpha)
