# Reconciliation with the Cycle 1 architecture spine

**Verdict** Aligned. The PRD and addendum carry all six architecture decisions through direct requirements or explicit source references. There are no conflicting acceptance criteria or missing architecture decisions.

| Architecture decision | Coverage |
| --- | --- |
| AD-1 Submission boundary | FR-1 and Scope define the single eligible predictor and intact kit. The addendum links the packaging and dependency rules. |
| AD-2 Matched comparison | FR-2 and the first-experiment addendum define the baseline, matched settings, single maintained pipeline, and vote-only difference. FR-6 and FR-7 bind the checked artifact to submission. |
| AD-3 Fit state and causal inference | NFR-2 and the addendum cover training-only fit, equivalent-input behavior, and the read-only inference contract. |
| AD-4 Labeled feature and position contract | FR-1 covers output alignment. The addendum defines the vote and links training feature/label alignment details. |
| AD-5 Local decision | FR-3 through FR-5 and NFR-1 cover preflight, complete comparison, unrounded metrics, current map, exact thresholds, and hold on failure or incomplete evidence. |
| AD-6 Evidence and verdict | FR-6 and FR-7 cover attempt evidence, digest continuity, and official outcome recording. The addendum links runtime and map staging details. |

Two clarifications would make the addendum easier to use on its own. State that the submission excludes city maps, run records, and the offline command. Name the no-BOM requirement alongside UTF-8. AD-1 already binds both details through the addendum link.

[Architecture spine](../../architecture/architecture-Biweekly-2026-11-22/ARCHITECTURE-SPINE.md) · [PRD](prd.md) · [Addendum](addendum.md)
