# PRD Quality Review — Biweekly Season 2 Cycle 1 Entry

## Overall verdict

The PRD gives one contestant a coherent workflow from the weighted-vote experiment through the official result. It identifies the comparison artifacts, the two numeric readiness thresholds, and the evidence needed to associate a result with the checked file. Clarify how local behavior flags affect readiness so implementers can reach the same decision from the same evidence.

## Decision-readiness — adequate

UJ-1 and FR-5 connect ready and hold to a contestant action. The matched-control difference remains evidence for the experiment, while the shipped baseline sets the performance threshold. The scope excludes a separate local holdout and SM-C1 captures repeated trials so readers can interpret the selected result in context.

## Substance over theater — strong

The requirements focus on the contest’s six-feature experiment, file identity, shared scoring coverage, and finite metrics. NFR-1 uses the timing and memory bounds that affect eligibility. The single named user journey establishes ownership of submission and recovery.

## Strategic coherence — strong

The thesis is a distinct, reproducible signal worth submitting. FR-2 isolates the weighted vote, FR-3 through FR-5 determine readiness, and SM-2 tests the competitive outcome. SM-C1 guards against partial coverage and gives context for repeated model selection.

## Done-ness clarity — adequate

FR-1, FR-2, FR-4, and NFR-1 have observable outputs or thresholds. FR-5 combines those checks into a concrete decision. FR-3 still needs an unambiguous definition of a local behavior failure.

### Findings

- **High — Local behavior flags have no stated decision rule** (FR-3, FR-5, NFR-2). FR-3 requires “passing model-behavior” and NFR-2 says local diagnostics “flag extended or frequent near-zero exposure.” The PRD leaves two plausible outputs for a finite candidate that passes the numeric gates and produces a flat-exposure flag. *Fix:* State which deterministic behavior failures trigger hold and define the effect of a flat-exposure flag. If the team uses this flag as a blocker, define its local acceptance rule before implementation. Keep the organizer’s separate unpublished flat-period check under official verification.
- **Medium — Complete coverage needs an explicit reference** (FR-3). “Complete matching full walk-forward periods and per-period observation counts” could lead to a comparison of the returned rows alone. The runner can omit periods, and all three artifacts can omit the same period. *Fix:* Require all three results to match every expected period and its observation count for the selected dataset, as the validation checklist specifies.
- **Low — Partial attempt and outcome records need clear states** (FR-6, FR-7, SM-C1). A failed load has no score, and pending or rejected entries may lack settled metrics. *Fix:* Retain the observed failure or current official state and record scores and settled membership when available.

## Scope honesty — strong

The first release is one experiment, local validation, and manual handoff. Scope identifies tempting expansion paths, and the addendum assigns the remaining implementation choices to the implementer. The unresolved local behavior decision is narrow and belongs in the readiness policy.

## Downstream usability — strong

FR-1 through FR-7 are distinct and can map to work in the predictor, offline command, and manual handoff. The glossary distinguishes the benchmark, control, nearest-city distance, and quality set. The addendum gives the implementer the settled architecture and handoff references with room to choose the model internals.

## Shape fit — strong

One named journey and a small set of functional requirements fit a single contestant and implementer. The product text keeps scientific intent and readiness visible while the addendum holds dependencies, fit-state details, and artifact construction.

## Mechanical notes

- FR, NFR, UJ, and SM identifiers are unique. The requirement references resolve locally, and UJ-1 names George.
- UJ-1 uses “lift” where FR-2 uses “candidate-minus-control difference.” Use “difference” so negative results read naturally.
