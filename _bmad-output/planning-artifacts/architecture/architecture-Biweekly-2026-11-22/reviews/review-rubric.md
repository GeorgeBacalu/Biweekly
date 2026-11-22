# Good-spine rubric review

**Verdict** Pass with three bounded rule clarifications. No critical or high findings.

## Findings

- **Medium · AD-2** Require the candidate and control source contents to differ only at the vote selector. The current rule names settings to share but could still allow two artifacts with different preprocessing, learned-orientation code, or other unlisted changes from the same source revision. Check the pair before scoring it. Leave the comparison method to implementation.
- **Medium · AD-4** State that the score is the arithmetic mean of signed contributions after applying rank-distance weights. “Averages signed votes weighted by rank distance” could also describe a normalized weighted mean with the sum of weights in the denominator. The forge choice preserves rank strength, so specify the former. Tie and missing-value handling can remain deferred to the shared pipeline.
- **Medium · AD-5** Require recorded pass/fail evidence for the candidate's behavior and resource checks before assigning readiness. The companion describes checks and thresholds. The runner can warn on a timing violation and still yield a score, and it does not establish peak-memory compliance. Name the evidence required, including per-call CPU timing and peak memory, and leave instrumentation to implementation.

## Rubric coverage

- The submission, fit-state, labeled-output, evaluation, and official verdict boundaries cover CAP-1 through CAP-4. The revised rules tolerate overlapping, permuted and variable-length frames. They separate the exact submitted artifact and official verdict from local experiment results.
- The single-file format, shipped interface, runner, cumulative period checks, unrounded novelty output and gauge coverage are compatible with the existing kit. AD-5's completeness requirements close the known omission paths in the runner. The kit remains the scorer.
- The architecture binds the kit's documented Python 3.12 compatibility reference and records the actual local environment. Python's linked release status and the run's technology research support this choice. The spine introduces no new third-party dependency decision. The organizer's runtime remains a separate execution environment.
- The operational envelope covers isolated local CPU evaluation, the shared work directory, map-snapshot serialization, recorded evidence, and submission to AlphaNova. Later-cycle models and hosted orchestration have explicit deferral conditions.
- The shared pipeline and offline owner make model tuning, rank tie and missing-value behavior, variant names, and evidence storage format safe to defer. No inherited parent spine is in scope.
