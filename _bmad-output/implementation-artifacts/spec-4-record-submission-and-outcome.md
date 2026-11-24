---
title: 'Story 4 — Record the manual submission and official outcome'
type: 'feature'
created: '2026-11-24'
status: 'done'
route: 'dispatch'
baseline_commit: 'e3a2bad2c2c2a031a73be7a6eb034102ddb3ca40'
review_loop_iteration: 0
context: []
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

Provide a local Season 2 Cycle 1 handoff that preserves the exact candidate the validator marked `ready`. Let the contestant append AlphaNova outcomes against the digest of the file they manually submitted.

## Boundaries & Constraints

- `python scripts/record_cycle_outcome.py prepare ATTEMPT_DIR` requires the finalized Story 3 `ready` decision and consistent attempt IDs and candidate hashes across the three reports. Read the in-attempt candidate bytes once, verify their SHA-256, and retain those exact bytes with their original `.py` basename and a copy of the decision. Print the archive path and digest for manual upload. Preparing a handoff creates no submission observation.
- Archive each ready attempt under `_bmad-output/evidence/submissions/season-2-cycle-1/<attempt-id>/`. Keep a versioned `handoff.json` with the season, cycle, preparation time, source attempt, candidate filename and SHA-256, and copied decision SHA-256. Preserve the original attempt and prior handoffs. An already prepared, matching handoff is reusable.
- `python scripts/record_cycle_outcome.py record HANDOFF_DIR --submitted-sha256 SHA256 --verdict pending|accepted|rejected` verifies the supplied digest, retained candidate bytes, and archived decision against the handoff before appending a UTC-timestamped JSON observation. Bind the observation to the season, cycle, handoff, and candidate digest. Accept an optional official entry ID, verdict detail, finite settled Sharpe, and quality-set membership (`yes` or `no`). Record absent fields as `null` and keep zero Sharpe and `false` membership. Preserve observations across updates.
- The command stays offline. The contestant supplies official facts from AlphaNova. The current `hold` decision is suitable for rejection testing, while controlled fixtures exercise the ready and outcome paths.

## I/O & Edge-Case Matrix

| Scenario | Input or state | Expected behavior |
| --- | --- | --- |
| Prepare | Finalized ready attempt with matching identities and bytes | Archive the exact candidate and ready evidence. Repeating preparation returns the verified archive. |
| Invalid handoff source | Hold, incomplete or mismatched reports, escaping staged path, or changed candidate | Reject with a clear cause and publish no handoff. |
| Official observations | Pending, accepted or rejected verdict with partial fields, then more facts | Append each observation with the same provenance. Preserve prior records and distinguish unknown from zero or false. |
| Invalid outcome | Supplied digest mismatch, archive tampering, or non-finite settled Sharpe | Reject with a clear cause and append no observation. |

</frozen-after-approval>

## Code Map

- `scripts/validate_walk_forward.py::record_decision` writes `decision.json` with `attempt_id`, `status`, `candidate_sha256`, `candidate_at_decision.sha256` and sibling report links. `collect` links it from the completed `attempt.json`, which contains `artifacts.candidate.staged` and `artifacts.candidate.sha256`. `novelty.json` repeats the candidate identity.
- Reuse `scripts/validate_walk_forward.py::atomic_json` and `utc_now`. The root `.gitignore` covers generated evidence. Story 3's `tests/test_validate_walk_forward.py::DecisionTests` shows valid and held report shapes.
- `_bmad-output/specs/spec-biweekly/validation.md` describes the manual workflow. `requirements/COMPETITION.md` describes official entry checks and settlement.

## Tasks & Acceptance

- [x] `scripts/record_cycle_outcome.py` — Implement prepare and record with byte and report identity checks, safe archive publication, and append-only observations.
- [x] `tests/test_record_cycle_outcome.py` — Exercise both CLI entry points and every matrix row with controlled attempts.
- [x] `_bmad-output/specs/spec-biweekly/validation.md` — Document manual handoff and recording commands, including how to supply the digest and optional fields.

- Given a ready attempt, when a handoff is prepared and its outcome is later recorded, then the archive, decision, and observation identify the same candidate SHA-256, and earlier observations remain readable.
- Given repeated or concurrent writes, when preparation or recording succeeds, then the CLI exposes a complete archive or complete observation and keeps existing evidence intact.

## Implementation Notes

- Initial focused verification passed all seven CLI tests. The real Story 3 `hold` returned exit 1 with its decision reasons and created no handoff. [Held decision](../evidence/20261123T224733Z-1546b83d4229415aa7d6f5e4e81195e3/decision.json)
- Final verification passed all 13 recorder tests and the validator-to-handoff integration test. A transient Windows directory rename error observed during verification now has a bounded retry and a regression test. The real held attempt again exited 1 and created no handoff.

## Spec Change Log

## Review Triage Log

| Reviewer | Finding | Verdict and evidence | Route |
| --- | --- | --- | --- |
| Blind review | Reports without completion metadata can be accepted | Medium. Removing versions and completion times from matching reports still published a handoff. | Patch |
| Blind review | Staged candidate paths can disagree | Medium. Two files with identical bytes but different basenames could produce a handoff under a filename absent from the novelty report. | Patch |
| Blind review | Upload guide permits the digest of a different file | Medium. The original instructions let the operator use the printed hash without checking the file selected for upload. | Patch |
| Blind review | Malformed handoff identity can record an observation | Medium. JSON `true` compares equal to integer `1`, and arbitrary nonempty strings passed the source and timestamp checks. | Patch |
| Blind review | Detached archive can fork the outcome history | Medium. An archive copy with the same basename could receive observations outside the canonical directory. | Patch |
| Blind review | Negative exponent notation fails with a space separator | Medium. Python 3.13 rejects the space-separated form. Document and test the equals form supported by the CLI. | Patch |
| Edge cases | Negative exponent value is parsed as an option | Medium. The space-separated token is treated as an option. Use the explicitly documented equals form for finite scientific-notation values. | Patch |
| Edge cases | Redirected stdout fails after writing evidence | Medium. A valid Unicode path can exceed a Windows pipe's default encoding and fail after publishing evidence. | Patch |
| Edge cases | Missing completion timestamps do not block preparation | Medium. Reports without `finished_at` or `decided_at` could reach publication. | Patch |
| Edge cases | Boolean identity fields compare equal to integers | Medium. JSON `true` passed the schema or cycle identity check. | Patch |
| Edge cases | A filename collision can replace an observation | Medium. Atomic replacement publishes a complete record but overwrites an existing destination if the ID collides. | Patch |
| Verification gaps | Producer and consumer are tested separately | Medium. The synthetic reports omitted the distinct source and staged paths emitted by the collector. Exercise the real collector output. | Patch |
| Verification gaps | Archive collision guard has no regression test | Medium. The reuse tests never supply a different internally consistent attempt with the same basename. | Patch |
| Verification gaps | Interrupted write could expose partial JSON | Medium. Tests only read observations after successful writes. Add a controlled write failure. | Patch |
| Verification gaps | Tests miss loss of official score precision | Medium. The previous finite score case only tested zero, which also compares equal to a Boolean in Python. | Patch |

## Design Notes

Use fixed report basenames, require staged paths to resolve under the given attempt, and verify the copied decision's identity before recording an outcome. Publish a complete archive together and give each outcome observation its own unique file. Record the official Cycle 1 page as the source context for operator observations.

## Verification

- Run `python -m unittest discover -v -s tests -p test_record_cycle_outcome.py`.
- Run `prepare` against the real Story 3 hold. Verify nonzero exit with a clear hold reason and no handoff. Use fixtures for positive cases and official observations.
