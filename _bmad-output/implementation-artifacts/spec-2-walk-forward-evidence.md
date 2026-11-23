---
title: 'Story 2 — Collect comparable walk-forward evidence'
type: 'feature'
created: '2026-11-23'
status: 'done'
route: 'dispatch'
baseline_commit: '3945434e9c26391f464b87e1fcd2d6096accbdd6'
review_loop_iteration: 0
context: []
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

Provide one offline command for CAP-2 that checks the exact candidate and compares it with its matched control and the shipped baseline. Keep reproducible evidence for completed and failed attempts so the performance result can inform the later submission decision.

## Boundaries & Constraints

- Use the shipped quick and full runner from the kit directory with one resolved CPU environment and `COMP6_DISABLE_PIP_INSTALL=1`. Keep the scorer, kit, and submissions intact. Run the three full evaluations with identical inputs and settings and gauge-fix disabled.
- Require the shipped release's periods `001`–`031`. Derive expected rows per period from feature Parquet metadata and the shipped split function. Validate exact returned period keys, counts, and total. Keep the full-precision returned net Sharpe and compute candidate minus control only from complete comparable results. Report separately whether candidate Sharpe is finite, positive, and strictly above the baseline. A valid weak score still constitutes collected evidence.
- Check candidate quick success and de-mean, exact input axes, finite zero-sum positions, repeat and reload, equivalent padding, independent axis permutations, short frames, and training overlap. Use real early, middle, and latest panels for behavior probes.
- Measure each train and predict call and each worker's peak resident bytes. Limits are 240 s train, 60 s predict, and strictly below 8,000,000,000 bytes. Preserve failures, partial observations, and the phase that produced them. Story 3 owns novelty and the submit-or-hold decision.
- On candidate full predictions, flag a row when its gross exposure (sum of absolute positions) is at most `max(1e-12, 0.01 × median gross exposure across all scored rows)`. Fail the local diagnostic if more than 5% are flagged or a streak reaches 24 consecutive flagged hours within a period. Break streaks on hour gaps and period boundaries. All-flat output fails. Record the threshold and observed counters.

## I/O & Edge-Case Matrix

| Scenario | Input or state | Expected behavior |
| --- | --- | --- |
| Partial run | Missing input, runner `{}`, returned error, missing period, wrong count, or worker failure | Retain checks, errors, and available results in the attempt. Fail evidence collection even if all models skipped the same period. |
| Weak or invalid score | Complete negative or zero Sharpe, or a missing/non-finite Sharpe | Retain a finite weak score with the performance flag false. Invalid scores fail collection and use JSON null with a reason. |
| Contract failure | Misaligned or non-finite positions, failed invariance, flatness, or exceeded resource limit | Identify the failing check and retain available evidence. |
| Changed inputs or rerun | Changed kit or data during scoring, or same command run again | Mark drift as failure. Keep distinct attempts so earlier evidence remains available. |

</frozen-after-approval>

## Code Map

- `requirements/runner.py::validate_submission` returns unrounded results and can return `{}` on a failed load. `load_predictor_class_from_file` loads a standalone copy and registers it for by-value `cloudpickle` reload.
- `requirements/walkforward.py::compute_val_rows` is `min(800, int(0.2 * n))`. `walkforward_full` may skip failed periods. It returns `per_period`, `n_periods`, and `n_observations`. `evaluation.py::backtest` retains the first validation row.
- `requirements/data_loader.py::load_period` reads feature, target, returns, and available dispersion Parquets. The checked-in release expects 800 × 31 = 24,800 scored rows per artifact.
- Story 1 owns `submissions/weighted_vote.py`, `submissions/weighted_vote_control.py`, `scripts/materialize_control.py`, and the behavior examples in `tests/test_weighted_vote.py`. `requirements/demo_engineered.py` requires XGBoost 3.2 or newer. `tests/requirements.txt` includes the kit and `cloudpickle`.

## Tasks & Acceptance

- [x] `scripts/materialize_control.py` — Share the byte-level control derivation with the validator so it can verify the staged pair.
- [x] `scripts/validate_walk_forward.py` — Add the CLI, input manifests, unique attempt directories, atomic JSON reporting, coverage checks, and comparisons.
- [x] `scripts/_walk_forward_worker.py` — Run quick, representative behavior, and each full evaluation in isolated workers. Observe the shipped loader, period loader, and model calls without editing the kit. Capture full candidate exposure and phase failures.
- [x] `tests/test_validate_walk_forward.py` — Exercise every matrix row, exposure boundaries, precision, failed-call telemetry, and report linkage.
- [x] `_bmad-output/evidence/.gitignore` — Keep generated evidence local under unique attempt directories. Record the actual run's path and result in this spec.

- Given the shipped snapshot and matched artifacts, when `python scripts/validate_walk_forward.py` runs, then the three full records each cover every expected period and 24,800 observations, and each record identifies the exact artifact SHA-256, runtime, shared input manifest, timings, and peak memory.
- Given a valid comparison, when the command reports its result, then it keeps the unrounded values and control delta and exits zero for passing collection checks. Failed checks yield a nonzero exit and a retained attempt record.

## Implementation Notes

- The validator stages the exact submission bytes, checks the matched pair using the materializer, and runs the shipped scorer in separate CPU workers. Each phase links to the shared settings and the pre-run input manifest, with per-call telemetry and phase logs retained alongside the attempt summary.
- The Windows managed sandbox denied the shipped runner access to its private temporary directory. That failure was retained in the [managed attempt](../evidence/20261123T214105Z-df0bad805dc74ad6b94771848f85aedf/attempt.json). The completed run used the same offline command with access to the runner's temporary directory and the local Python 3.13.14, XGBoost 3.4.1, and cloudpickle 3.1.3 environment.
- The first post-review run hit its metadata worker deadline while rebuilding dependency bytecode. Its [failed attempt](../evidence/20261123T220035Z-dcfd54d58a064804852c15dbe4d01f7d/attempt.json) preserved the timeout. The validator now loads the shipped kit modules directly from the fingerprinted source files and uses normal import behavior for installed dependencies. A regression test confirms that stale kit bytecode is ignored.

## Spec Change Log

## Review Triage Log

| Reviewer | Finding | Verdict and evidence | Route |
| --- | --- | --- | --- |
| Edge cases | Hung worker blocks collection | Medium. `subprocess.run` had no deadline and a hung call never reached the checks for returned timings. | Patch |
| Edge cases | Interrupt omits the active phase | Medium. An interrupt bypassed the journal reads and reached the outer exception handler. | Patch |
| Edge cases | Reversible input swap can evade the manifests | Low. Two point-in-time manifests cannot identify an intervening swap that restores the original bytes and timestamps. An immutable copy of the full kit and data adds substantial storage for that scenario. | Reject |
| Edge cases | Malformed extra period disappears | Medium. The normalizer filtered out every non-dictionary entry before checking the exact key set. | Patch |
| Edge cases | In-place label changes can pass | Medium. The validator read requested labels after `predict` returned. | Patch |
| Edge cases | Reused output can overwrite repeat reference | Medium. The check held the frame by reference across multiple calls. | Patch |
| Edge cases | Conditional selector override is missed | Medium. The AST scan only visited assignments in the immediate class body. | Patch |
| Blind review | Managed Windows temp denies kit loader | Medium. Python 3.13 creates a private temp directory that the managed sandbox cannot open. The identical command passed with approved temp access. The failure originates in the shipped loader and its execution environment. | Defer |
| Blind review | Hung model call prevents final report | Medium. Worker execution had no finite deadline. | Patch |
| Blind review | Interrupt loses summarized partial observations | Medium. The live worker was absent from the completed phase map. | Patch |
| Blind review | Swap and restore defeats pre/post hashes | Low. Moving the original file away and back can preserve its metadata and bytes. The approved manifest design records the observed before and after states. | Reject |
| Blind review | Cached kit bytecode can evade source hashes | Medium. Disabling bytecode writes still allowed imports of existing caches. | Patch |
| Blind review | Optional dispersion blocks preflight | Medium. The kit loader accepts missing dispersion with scoring intact, while the manifest required every dispersion file. | Patch |
| Blind review | Reload shares the original class registry | Medium. Cloudpickle can reuse the same class object in the process that fitted it. | Patch |
| Blind review | Padding keeps the immediate future unchanged | Medium. The reference and padded inputs share future observations around every checked hour. Add an independent future perturbation probe. | Patch |
| Blind review | Momentarily flat short frame fails unrelated check | Medium. The extra nonzero assertion could reject valid short predictions even when full exposure met its approved limits. | Patch |
| Blind review | Unexpected malformed period can be dropped | Medium. A returned `032` with a null value was filtered before coverage checked the key set. | Patch |
| Blind review | Valid score comparison suppressed by diagnostics | Medium. The comparison depended on the combined full phase pass, including exposure and resources. | Patch |
| Blind review | Failed import loses available runtime details | Medium. The interpreter and dependency versions were written only after every import succeeded. | Patch |
| Verification gaps | Missing reload probes pass tests | Medium. The collection tests fabricated worker results. Add direct worker checks on every representative period. | Patch |
| Verification gaps | Position-to-exposure calculation is untested | Medium. The tests started with computed gross values, so dropping the absolute value did not break them. | Patch |
| Verification gaps | Expected counts could be hard-coded | Medium. Mocked fixture counts were always 800 and never exercised the metadata and split path. | Patch |
| Verification gaps | Worker could lose the offline setting | Medium. The mocked subprocess tests ignored the passed environment. | Patch |

## Design Notes

Stage artifact bytes in each attempt. Hash kit sources and score inputs before and after scoring. Record settings, loaded dependency versions, and phase logs separately from the versioned `attempt.json` summary. Use monotonic wall times around calls and OS peak RSS, with Windows working-set high-water or POSIX `ru_maxrss` normalized to bytes. Derive exposure only from aligned candidate full predictions.

## Verification

- `python -m unittest discover -s tests -p test_validate_walk_forward.py` — targeted checks pass.
- `python scripts/validate_walk_forward.py` — run the shipped snapshot once and inspect the attempt record for coverage, behavior, resources, and the exact three scores.
- The nine targeted tests passed, and `python scripts/materialize_control.py --check` verified the matched pair.
- The [completed shipped-snapshot attempt](../evidence/20261123T214306Z-1c050183510d47ceb353e8fb7db69f75/attempt.json) passed collection. The candidate passed the quick and representative behavior checks. Each full evaluation scored all 31 periods and 24,800 rows. The before and after input manifests matched. The slowest train and predict calls took approximately 3.221 s and 0.072 s, respectively. The highest worker peak was 813,064,192 bytes.
- The full net Sharpe values were `-0.031194688556763914` for candidate, `-0.030590596256508125` for control, and `-0.020093478151728007` for baseline. Candidate minus control was `-0.0006040923002557895`. The candidate's score was finite, negative, and below baseline. Candidate exposure passed with zero flagged rows and a longest streak of zero at a gross-exposure threshold of `0.0019433479222425375`.
- The [reviewed-code attempt](../evidence/20261123T220439Z-c7c531c47d554cbdb5300590d6f1c996/attempt.json) reproduced all three scores and the control delta exactly. The 31 periods and 24,800 observations matched the input manifest for each model. The three fresh-process reloads passed, and no candidate rows were near-flat. The slowest train and predict calls took 3.802 s and 0.189 s. The highest worker peak was 837,017,600 bytes.
- The final `python -m unittest discover -v -s tests` run passed all 31 tests, including the predictor suite and the stale kit bytecode regression. The shipped kit and scored artifact bytes matched the recorded digests.
