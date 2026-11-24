---
title: 'Story 4 follow-up — Harden submission handoffs and outcome records'
type: 'bugfix'
created: '2026-11-24'
status: 'done'
route: 'dispatch'
baseline_commit: '77fc4e01addaacb418dd56817b848424c3422e39'
review_loop_iteration: 0
context: []
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

Keep the offline Cycle 1 handoff tied to the exact candidate and coherent Story 3 readiness evidence. A published outcome must stay identifiable and count as saved even if cleanup or console output fails.

## Boundaries & Constraints

- Preserve the `prepare` and `record` CLI, canonical archive layout, version-1 JSON, and existing candidates, decisions, and observations. Continue recording against valid archives after the source attempt is removed or the archive moves between Windows and POSIX. The operator supplies the official facts.
- Validate the linked evidence produced by Story 3 without rerunning the validator. Reject duplicate JSON keys. Require finite candidate Sharpe above zero and baseline, an unrounded distance from 60° to 180° with a named nearest city, and agreement between the relevant metrics in the attempt, novelty, and decision. Require confirmed map evidence, valid matching scored-map and input digests, and nondecreasing CAP-2, novelty, and decision completion timestamps. Keep checks that need original reports in `prepare`. During `record`, verify the retained candidate and decision against the handoff.
- Preserve explicit official entry IDs for multiple uploads of the same digest. An omitted ID remains `null`. When the archive already has multiple distinct IDs, require an explicit ID for new observations. Treat unidentified observations as unassigned and serialize the history check with publication across CLI writers.
- Publish only complete archives and observations. Preserve an existing observation at the same path. Support safe same-volume publication on Windows without hard links and check filesystem capability before offering an upload when safe publication is unavailable. Treat cleanup and output errors after publication as diagnostics. Flush success output and use stderr for the saved path when stdout is unavailable.

## I/O & Edge-Case Matrix

| Scenario | Input or state | Expected behavior |
| --- | --- | --- |
| Finalized attempt | Consistent Story 3 reports, including equal UTC timestamps or exactly 60° | Prepare or reuse the byte-identical archive. |
| Inconsistent attempt | Conflicting or missing metrics, map confirmation, digests, completion order, links, or duplicate JSON keys | Reject with the cause before publishing a handoff. |
| Existing handoff | Source unavailable or from another OS, with intact version-1 archive | Append to the verified archive. Reject an altered archive ID or retained digest. |
| Invalid observation | Rejected verdict with membership `yes`, non-finite or underflowing Sharpe, or missing entry ID in a multi-entry history | Reject before publication. Preserve legitimate zero, false, and unknown values. |
| Publication failure | Concurrent writers, forced collision, interrupted write, or output or cleanup failure after commit | Preserve prior observations. Report success after a committed write and show its path when an output stream is available. |

</frozen-after-approval>

## Code Map

- `scripts/validate_walk_forward.py` supplies `CITY_MIN_DISTANCE`, `atomic_json`, and `utc_now`. `comparison` and `decision_prerequisites` define the metrics. `record_decision` writes the final confirmation, metrics, and digests. Compare `attempt.inputs_before` and `novelty.input_manifest` and `inputs_at_decision` with `decision.inputs_sha256`. Compare the novelty phase's scored map, staged map, and confirmation digests with `decision.scored_map_sha256`.
- `scripts/record_cycle_outcome.py` owns `read_ready_attempt`, `verify_handoff`, `prepare`, `record`, and their CLI reporting. `source_attempt` is archival metadata. Validate its syntax with the source path flavor and retain the canonical handoff-root check.
- `tests/test_record_cycle_outcome.py` has synthetic reports and collision, interruption, and concurrency tests. `tests/test_validate_walk_forward.py` exercises the producer-to-handoff path at the threshold. `_bmad-output/specs/spec-biweekly/validation.md` documents manual operation.

## Tasks & Acceptance

- [x] `scripts/record_cycle_outcome.py` — Enforce evidence and observation checks, preserve existing archive compatibility, and handle concurrent publication and post-commit errors.
- [x] `tests/test_record_cycle_outcome.py` — Match fixtures to the producer, exercise the matrix, pause preparation to observe the publication boundary, mutate all reverse report links and the handoff ID independently, and assert exact JSON identity field types. Retain the existing concurrent multi-entry case.
- [x] `_bmad-output/specs/spec-biweekly/validation.md` — Explain entry IDs for repeated uploads, filesystem support, and how the command reports a saved observation when output fails.

- Given a producer-generated ready attempt, when the contestant prepares the handoff and later records an outcome without the source directory, then the original candidate digest remains consistent across the handoff, decision, and observation.
- Given concurrent record commands for a handoff, when their history checks and writes complete, then each valid submission is published as one complete observation and no preexisting observation changes.

## Implementation Notes

- The 24 recorder tests passed on Windows, including the publication boundary, cross-process entry history check, and closed stdout pipe. The Story 3 producer integration passed at 60°. A producer-to-record check also passed after removing the source attempt and verified that all three records retained the candidate digest.
- Windows uses same-volume rename for observation publication. POSIX uses hard-link publication. The archive probes its publication and locking support before preparation reports success. Native POSIX execution was unavailable in this environment.
- After review fixes, all 34 recorder tests and the Story 3 producer integration passed. The additional tests covered contradictory worker evidence, unavailable publication on archive reuse, Windows publication without hard links, and lock contention confirmed by a separate process.

## Spec Change Log

## Review Triage Log

| Reviewer | Finding | Verdict and evidence | Route |
| --- | --- | --- | --- |
| Blind Hunter | Scored candidate artifact can disagree with the retained candidate | Medium. The full and novelty phases carry artifact digests, but preparation checked only the top-level candidate references. | Patch |
| Blind Hunter | Equal Sharpe can hide different coverage | Medium. The producer compares complete runner results, while the recorder allowed mismatched observation counts. | Patch |
| Blind Hunter | A ready report can contain failed phase evidence | Medium. A fixture with a failed novelty phase and populated failures still published. | Patch |
| Blind Hunter | Post-scoring and worker manifests can disagree | Medium. The producer retains the after-scoring and phase input digests. Inconsistent copies escaped the recorder's comparisons. | Patch |
| Blind Hunter | Post-scoring map digest can disagree | Medium. A conflicting `staged_map_after` digest passed preparation in a controlled fixture. | Patch |
| Blind Hunter | Blank explicit entry ID bypasses disambiguation | Medium. The guard only checks whether the flag value is `None`. | Patch |
| Blind Hunter | Unrelated JSON can affect the entry history | Medium. The scan counted any JSON without checking the observation filename or handoff identity. | Patch |
| Blind Hunter | Failed probe cleanup can leave probe files in a new archive | Low. An injected deletion failure left a probe in the published archive. Preparation can require probe cleanup before publication. | Patch |
| Blind Hunter | Omission guidance is ambiguous for pending verdicts | Low. The guide explains the ID rule separately from the pending example. Tie the rule to the pending case. | Patch |
| Edge Case Hunter | Enumeration failure can appear to be empty history | Medium. Python 3.13 glob can suppress permission errors even when the directory allows file creation. | Patch |
| Edge Case Hunter | Windows lock acquisition expires while a writer is active | Medium. A contender failed after the native retry window while a separate process still held the lock. | Patch |
| Edge Case Hunter | Blank entry IDs can distort history | Medium. Explicit whitespace avoids the omitted-ID guard and is counted as a known ID. | Patch |
| Edge Case Hunter | A surrogate path can fail to print on both streams | Low. UTF-8 reconfiguration set both streams to strict error handling. Preserve escaped diagnostics. | Patch |
| Edge Case Hunter | A novelty score match can conceal a failed coverage gate | Medium. A fixture with different coverage and a failed score-agreement check still archived. | Patch |
| Verification Gap | Windows linkless publication lacks a direct regression test | Medium. Substituting hard-link publication for rename still passed the collision and interruption test. | Patch |
| Verification Gap | Reused archives lack a capability-failure test | Medium. Removing the reuse probe let the CLI offer upload on an unsupported publisher. | Patch |
| Verification Gap | Scheduling can hide a missing lock | Medium. The test passed with a no-op lock when the child yielded after signaling. | Patch |
| Verification Gap | The zero Sharpe floor is not isolated | Medium. The zero-score fixture also failed the positive-baseline check when the floor was removed. | Patch |
| Verification Gap | Equal but incorrect Sharpe deltas lack a negative test | Low. Mutating one delta fails equality even with the arithmetic check removed. A matching pair exercises the missing branch. | Patch |

## Verification

- Run `python -m unittest discover -v -s tests -p test_record_cycle_outcome.py` and the `DecisionTests.test_ready_preserves_exact_scores_map_and_candidate_digests` integration test.
- Run `git diff --check` and report any wider-suite environment failures separately from the recorder verification.
