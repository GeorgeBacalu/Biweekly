---
title: 'Story 1 — Weighted-vote predictor and matched control'
type: 'feature'
created: '2026-11-23'
status: 'done'
route: 'dispatch'
baseline_commit: '237033a6ab08d4264b7ed6fcca0e68946742f971'
review_loop_iteration: 0
context: []
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

Build the Season 2 CAP-1 submission and its matched control. A single maintained pipeline produces two independently loadable Python files so later evaluation can isolate the effect of adding the weighted vote.

## Boundaries & Constraints

- Deliver UTF-8 `.py` files without a BOM. Import the shipped `Predictor` and use its `train(features, target)` and `predict(features)` signatures. Keep model logic inside the subclass and use kit-supported dependencies. Keep the kit intact.
- Rank the six supplied features across the 20 assets per hour. Fit orientations by feature label from the current split. The candidate averages the six signed votes, each weighted by distance from the median rank. Share rank inputs, learner settings, seed, and position shaping. Fix the selector inside each file.
- Align training targets by hour and asset before fitting. Treat ticker labels as local to the current panel. Keep predictions read-only and causal with finite trailing history anchored to hour labels. Preserve input index and row order and the asset order of first appearance. Return finite, zero-sum positions.
- Keep fit and prediction within the competition CPU limits of 240 seconds and 60 seconds per call, respectively, and peak memory below 8 GB. Full walk-forward comparisons, near-zero diagnostics, resource evidence, novelty and submission belong to later stories. Keep city and evaluation inputs outside the model and leave the full-panel feature hook unused.

## I/O & Edge-Case Matrix

| Scenario | Input | Expected behavior |
| --- | --- | --- |
| Gate ordering | Independently shuffled target axes, reordered feature blocks or tickers, permuted evaluation hours | Match by labels, sort internally by hour where needed, and restore output axes. |
| Causal history | Short or overlapping frames, equivalent padding, a gap in hour labels, or a changed future suffix | Score each supplied hour using fitted state and observations inside its trailing window. Shared hours with identical available history agree. |
| Ties and missing values | Tied or non-finite features, warm-up zeros in targets, partial missing targets | Use average ranks for ties and neutral values for missing features. Keep finite zero labels and mask missing targets after alignment. |

</frozen-after-approval>

## Code Map

- `requirements/predictor.py` defines the two-level feature columns and subclass signatures. Real kit panels use unique integer hour labels. Features and targets have different ticker order.
- `requirements/demo_engineered.py::_csrank` demonstrates centered normalized ranks. Its smoothing depends on frame start. Use finite index-time smoothing for this story.
- `requirements/data_loader.py::load_period` loads cumulative panels. `requirements/runner.py::load_predictor_class_from_file` loads a temporary copy, evicts it, and registers it by value when `cloudpickle` is present. `validate_submission` defaults to latest-period quick mode. Check for returned errors before checking `demean_check`.
- `requirements/evaluation.py::check_cross_sectional_demean` checks row means. Verify exact labels and finite positions separately.
- `_bmad-output/specs/spec-biweekly/stories.yaml` defines Story 1 and its checkpoints. The architecture spine's AD-1 through AD-4 govern the predictor.

## Tasks & Acceptance

**Execution**

- [x] `submissions/weighted_vote.py` — Implement the shared pipeline as the canonical, loadable candidate.
- [x] `scripts/materialize_control.py` and `submissions/weighted_vote_control.py` — Generate the checked-in control by replacing exactly one selector in the candidate bytes. Provide `--check` to detect a missing or stale control without rewriting it.
- [x] `tests/test_weighted_vote.py` — Cover rank and vote arithmetic, independent label permutations, both variants' inference and reload behavior, and the shipped quick runner.

**Acceptance criteria**

- Given the candidate and generated control, when checked and loaded independently through the shipped runner, then their bytes differ only at the in-class selector and both complete quick mode with a passing de-mean check and no returned error.
- Given known directional ranks and labels, when fitted and scored, then each orientation follows its aligned labels, the candidate vote equals the mean of the six signed rank-distance votes, and the control's vote input is zero.
- Given varied frame lengths and the input cases in the matrix, when either trained variant predicts, then it preserves exact axes, returns finite rows summing to zero within `1e-12`, and gives equivalent predictions on repeat, reload, and shared hours with the same trailing observations.

## Implementation Notes

- The learner uses 30 depth-2 CPU histogram trees with seed 42 and four threads. The vote uses the sign of the aligned rank and target covariance. Position smoothing counts available observations in the current and previous 23 labeled hours.
- All six tests passed on Python 3.13.14 with `cloudpickle` 3.1.3. The suite exercised each input case and ran the shipped quick check on 50,014 training rows and 800 holdout rows for each variant. Both variants passed de-meaning without a returned error. The selector check also passed on Python 3.12.14.
- After review fixes, all ten tests passed on Python 3.13.14 with XGBoost 3.4.1 and `cloudpickle` 3.1.3. The candidate and control trained in 2.4 s and 2.1 s in the period 031 quick run. Both passed the de-mean check. The runner displayed quick Sharpe of −0.0245 for the candidate and −0.0255 for the control.
- SHA-256 of the candidate checked after review is `51e4c2c67a802dd0877d09c185e8c4c476ab91d94379b4107e8ba04733002aa8`. SHA-256 of the control is `b2ffa8056a5f2a46ff253c82a244666a9a1db0e3df553896b2c684b1e592b0d9`.

## Spec Change Log

## Review Triage Log

| Reviewer | Finding | Verdict and evidence | Route |
| --- | --- | --- | --- |
| Edge cases | Hour subtraction overflows | Medium. An `int8` index with hours −100 and 100 wraps a 200-hour gap into the smoothing window. | Patch |
| Edge cases | Selector text in a docstring can pass | Medium. A selector-shaped docstring line can be replaced while a reformatted class assignment stays enabled. | Patch |
| Edge cases | Interrupted generation can truncate the control | Medium. Direct `write_bytes` opens the loadable control for truncation before the new bytes are written. | Patch |
| Edge cases | Prediction can see two fits | Medium. A retrain publishes the new booster before it publishes orientations, so a concurrent prediction can mix them. | Patch |
| Edge cases | Test discovery cannot import `cloudpickle` | Medium. The kit requirements omit it and the suite imports it during collection. | Patch |
| Blind review | Selector text in a docstring can pass | Medium. The matching logic examines literal lines without checking which one is the assignment. | Patch |
| Blind review | Zero covariance can get full orientation | Medium. A symmetric fixture produced tiny nonzero covariance and a different sign after ticker relabeling. | Patch |
| Blind review | Hour subtraction overflows | Medium. The original mask subtracts at the index dtype and changes a position across a 200-hour gap. | Patch |
| Blind review | No finite aligned labels abort training | Low. The explicit error is reachable on an entirely missing label set. Shipped and gate fits use labeled monthly slices, and an all-missing fallback would score a fabricated fit. | Reject |
| Blind review | Clean kit setup cannot collect tests | Medium. Importing `cloudpickle` requires a dependency outside the kit setup. | Patch |
| Blind review | Two-sided padding lacks a direct check | Low. Separate truncation and suffix perturbation tests cover the causal mechanisms. Adding an equivalent two-sided padded frame exercises the gate scenario directly. | Patch |
| Blind review | Reload runs in the training process | Medium. The original test shared Cloudpickle's class registry with the fit. A cold process exercises the persistence boundary. | Patch |
| Blind review | Quick evidence lacks exact digests | Low. Initial notes name the files and check result, but lack digests. This finding concerns only the spec. | Reject |
| Verification gaps | Both actual learner calls can drop the vote | Medium. Both calls were altered to zero the vote and the original six tests still passed. The added check must exercise real fit and inference. | Patch |
| Verification gaps | Sparse fixture permits eight-row smoothing | Medium. An eight-observation cap passed the original suite. Dense hourly data needs a test spanning a complete window. | Patch |

## Design Notes

Use the sign of each centered rank's association with its aligned training target. Use a shallow fixed-seed CPU XGBoost regressor with six base ranks plus the vote input. A nonlinear learner can split on agreement even though the vote is a linear combination of the ranks. Shape positions with a trailing 24-hour mean and row de-meaning. Use hour labels for window membership, including when the input has gaps.

## Verification

- `python -m pip install -r tests/requirements.txt` sets up the kit and the by-value reload dependency in a test environment.
- `python scripts/materialize_control.py --check` exits successfully and confirms the selector-only difference.
- `python -m unittest discover -s tests -p test_weighted_vote.py` passes with kit dependencies and `cloudpickle` for by-value reload. Run integration checks from the kit directory with `COMP6_DISABLE_PIP_INSTALL=1` and temporary output. Prefer kit-recommended Python 3.12 and record the interpreter used.
