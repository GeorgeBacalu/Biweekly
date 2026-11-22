# Reconciliation of the Cycle 1 experiment

**Verdict** The PRD and addendum preserve the experiment scope, matched comparison, and local submission gate.

- The purpose section and FR-2 prioritize weighted feature agreement. The addendum specifies same-hour ranks, orientation learned within each training split, and the arithmetic mean of signed rank-distance votes.
- FR-2 and FR-3 require the candidate, matched control, and shipped baseline to use matching full walk-forward coverage and settings. The addendum fixes the vote selector as the only difference between the candidate and matched control.
- The scope section excludes a separate local holdout and defers rank-change and turnover-only experiments.
- FR-4 and FR-5 require the current official map, a finite unrounded nearest-city distance of at least 60°, and positive net Sharpe strictly above `demo_engineered.py`. Failed checks or incomplete evidence lead to hold.

[Forged experiment](../../../forge/cycle-1-signal-search/forged-idea.md) · [PRD](prd.md) · [Implementation addendum](addendum.md)
