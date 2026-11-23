---
title: 'Story 3 — Apply the submit-or-hold decision'
type: 'feature'
created: '2026-11-24'
status: 'done'
route: 'dispatch'
baseline_commit: '6874c61f53b0d6f5265b474bd50b8f6d25391ac4'
review_loop_iteration: 0
context: []
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

Extend the offline walk-forward command with a local `ready` or `hold` decision for the exact scored candidate. Retain a separate novelty record and explain each failed prerequisite. The contestant owns submission.

## Boundaries & Constraints

- Return `ready` only when all CAP-2 checks pass for the exact candidate, its unrounded net Sharpe is finite, positive, and strictly above the shipped baseline, and its unrounded nearest-city distance is finite and at least 60° with complete gauge coverage and a confirmed current official map. Retain the matched-control delta as experiment evidence.
- Use `--city-map PATH` to supply the extracted Season 2 Parquet for the isolated candidate novelty phase. Stage exact bytes per attempt and record the digest, source and verification time when known. The checked-in map has unknown freshness. A local file path or timestamp alone gives no current-map credit.
- Keep the command offline. With `--confirm-current-map`, after all other prerequisites pass, ask the operator to re-download the latest map from the official Season 2 Cycle 1 page and supply the extracted path. Record their confirmation and UTC time. At decision, recheck the staged candidate digest and require the confirmed map digest to match the map used for scoring. A superseded map yields `hold` until a fresh novelty run for the same candidate. An absent or declined confirmation yields `hold`.
- Use the shipped `validate_submission(..., full=True, gauge_fix=True)` with the CAP-2 runtime, candidate, and scoring inputs. Keep the kit and shared `signal_db` intact. Record before/after gauge digests, exact raw gauge dimensions and finite values, one successful fixed-vector result per expected period, and the aggregate row count.
- Retain CAP-2 reports and exit semantics. Write the novelty record and structured decision with reasons to the same attempt. Keep submission and official result tracking outside this workflow.

## I/O & Edge-Case Matrix

| Scenario | Input or state | Expected behavior |
| --- | --- | --- |
| Eligible | Passing digest-linked evidence, positive Sharpe strictly above baseline, current-map confirmation, distance `60.0` or higher | `ready` with raw metrics and candidate digest. |
| Numeric gate | Candidate Sharpe is zero, negative, tied with baseline, or non-finite, or distance is `59.96` with display rounding to `60.0` | `hold` with the relevant cause. Use raw metrics. |
| Incomplete novelty | Missing or malformed gauge/map, truncated coverage, non-finite distance, or unavailable nearest city | `hold` and preserve the failure and available evidence. |
| Provenance failure | CAP-2 failed, candidate/input digest drifted, map confirmation is missing, or the latest official download has a different digest | `hold` with each blocking cause. Retain the attempt. |

</frozen-after-approval>

## Code Map

- `scripts/validate_walk_forward.py::collect`, `run_worker`, `coverage_checks` and `comparison` own the attempt, worker dispatch and CAP-2 evidence. `scripts/_walk_forward_worker.py::score` already observes the shipped runner and current period.
- `requirements/walkforward.py::walkforward_full` can skip missing gauges or truncate rows while reporting complete Sharpe coverage. Observe `city_tools.compute_gauge_fixed`; native results include `gauge_fixed`, `city_comparison`, `city_novelty_deg` and `nearest_city`.
- `requirements/city_tools.py::compare_to_cities` accepts an explicit `city_db_path`. Bind it to the attempt snapshot inside the worker. `requirements/signal_db/NO_CITIES_IN_THIS_RELEASE.txt` identifies the hourly official download. The local map uses `small-4335-s2`.

## Tasks & Acceptance

- [x] `scripts/validate_walk_forward.py` — Add map options, gauge and map provenance, a separate novelty worker phase, and the decision record. Preserve partial failures and link paths and digests.
- [x] `scripts/_walk_forward_worker.py` — Validate the staged map and raw gauges, observe gauge-fix coverage, and preserve native nearest-city metrics with the shipped runner.
- [x] `tests/test_validate_walk_forward.py` — Cover the matrix, including one real shipped gauge/compare path with fixture data and mismatches that the runner would otherwise truncate.

- Given the shipped snapshot and a supplied map, when novelty runs, then its Sharpe coverage and score agree with candidate full evidence, it reports 24,800 gauge-fixed rows over 31 periods, and the comparison covers every validated map row.
- Given separate attempts that supply different maps, when both score, then each uses its own staged map and reports its digest without writing to the kit's `signal_db`.

## Implementation Notes

- Initial verification passed all 41 tests. The shipped run using the checked-in map exited zero with CAP-2 status `passed` and a decision of `hold`. Novelty coverage was 31 periods, 24,800 gauge-fixed rows, and 355 map rows. The unrounded distance was `34.512651845971135` degrees and the candidate Sharpe was `-0.031194688556763914`. [Initial decision](../evidence/20261123T222804Z-71cd4b69a6d54a5fad4dc77d656068ae/decision.json)

## Spec Change Log

## Review Triage Log

| Reviewer | Finding | Verdict and evidence | Route |
| --- | --- | --- | --- |
| Edge cases | Finite vectors can misstate the angle | High. The shipped norm calculation turns a parallel vector with underflow or overflow into an apparent 90° separation. | Patch |
| Edge cases | CAP-2 interrupt can start novelty | Medium. An interrupt outside worker dispatch enters the collection error list only, which the novelty gate previously ignored. | Patch |
| Edge cases | Confirmation read error skips decision | Medium. The prompt handler caught EOF and Ctrl+C but allowed I/O and decoding errors to escape. | Patch |
| Edge cases | A second gauge read can use different bytes | High. The native scorer opened each gauge after a separate validation read, so a restored replacement could pass the parent drift check. | Patch |
| Edge cases | Candidate hash can differ from executed bytes | High. The worker hashed the staged path before validation and the shipped loader copied it in a later read. | Patch |
| Blind review | Gauge digest can misidentify scoring inputs | High. The digest, shape validation, and scoring did not share a single snapshot. | Patch |
| Blind review | Candidate load can race its digest | High. Scoring accepted the source path a second time after the worker hashed it. | Patch |
| Blind review | Confirmation accepts its own staged map | High. The staged snapshot and a hard link to it satisfy a digest equality that gives no separate download evidence. | Patch |
| Blind review | Invalid normalization can clear the 60° gate | High. A collinear vector at `1e-100` passed schema checks and scored 90° through the shipped comparator. | Patch |
| Blind review | Novelty settings misstate gauge-fix | Medium. The novelty record pointed at the CAP-2 settings file with `gauge_fix` set to `false`. | Patch |
| Blind review | Native error loses structured partial coverage | Medium. The scorer emitted the summary only after a returned result, even when earlier gauge vectors were recorded. | Patch |
| Blind review | Prompt I/O failure can change command exit | Medium. The exception escaped before the decision and novelty reports were written. | Patch |
| Blind review | Decision hides worker failure details | Medium. The original `novelty_worker` reason omitted available failed checks and period labels. | Patch |
| Blind review | No test exercises the real novelty subprocess handoff | Medium. One fixture mocked the worker and another called the scorer directly, leaving serialization and parsing untested. | Patch |
| Verification gaps | Removing request serialization passes tests | Medium. The mutation passed all 31 validator tests. Exercise the real novelty worker through the dispatcher. | Patch |
| Verification gaps | Replacing score agreement with presence passes tests | Medium. Both original phase fixtures used the same score. Exercise distinct finite scores with equal coverage. | Patch |
| Verification gaps | Dropping map and gauge drift gates passes tests | Medium. The original confirmation drift test modified the candidate and returns only. Exercise both files separately. | Patch |
| Verification gaps | Changing hold exit semantics passes tests | Medium. Existing tests called `collect()` and did not assert the CLI status. Exercise a passing CAP-2 hold through `main()`. | Patch |

## Design Notes

Validate raw gauge shape `(expected_rows, 19, 20)` and fixed output `(expected_rows, 19)` before and after the native transform, respectively. Require a nonempty `small-4335-s2` map with unique identifiers and a finite, nonzero 19-dimensional vector per row. Check that the nearest city is in the staged map and the distance is within `[0°, 180°]`.

## Verification

- Run `python -m unittest discover -v -s tests`.
- Run the offline command on the shipped candidate with the checked-in map as an unconfirmed diagnostic fixture. Verify 31 periods and 24,800 gauge rows, the separate CAP-2 and novelty evidence, and `hold` for the observed negative Sharpe and absent current-map confirmation. Verify `ready` and boundary failures with controlled fixtures.
- The reviewed-code run reproduced the initial raw scores and nearest-city distance. CAP-2 passed and the decision stayed `hold`. All 31 consumed gauge digests matched the manifest, all 24,800 fixed rows and 355 city comparisons were present, and the hashed novelty settings enabled gauge-fix. [Verified decision](../evidence/20261123T224733Z-1546b83d4229415aa7d6f5e4e81195e3/decision.json) · [Novelty evidence](../evidence/20261123T224733Z-1546b83d4229415aa7d6f5e4e81195e3/novelty.json)
- The final `python -m unittest discover -v -s tests` run passed all 51 tests. Two confirmation tests failed intermittently on the prior run and passed three consecutive isolated reruns and the final full suite. The test assertions now show the decision and input checks if the issue recurs.
