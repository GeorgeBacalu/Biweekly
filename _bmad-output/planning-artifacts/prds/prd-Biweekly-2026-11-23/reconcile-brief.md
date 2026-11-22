# Reconciliation of the contestant brief

**Verdict** The PRD preserves the brief’s workflow and success criteria. Two input boundaries merit an explicit line for the implementer.

- The brief says ticker identities rotate across periods. NFR-2 covers fitting and causal inference, and the addendum keeps learned state per fit. State explicitly that ticker identities are period-specific and fitted state must avoid carrying them across periods. This also matches the finalized spec and AD-3.
- The brief reserves the city map for offline candidate selection. FR-4 assigns the novelty check to validation. Add that the city map stays outside the submission and is unavailable as a predictor input, consistent with the finalized spec and AD-1.
- FR-3 through FR-5 preserve the local Sharpe, coverage, novelty, and resource gates. The PRD uses the more precise later decisions by requiring a single offline command, complete like-for-like runs, the unrounded distance, and the official map current at decision time.
- SM-2 and FR-7 preserve official acceptance, settled Sharpe, and quality-set membership. The Cycle 1 scope and official outcome record carry forward the brief’s reusable evaluation and handoff. Prize rankings and legacy-pot timing remain organizer context in the linked rules.

[Contestant brief](../../briefs/brief-Biweekly-2026-11-22/brief.md) · [PRD](prd.md) · [Implementation addendum](addendum.md)
