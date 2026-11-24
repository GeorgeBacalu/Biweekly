# Submission and validation

## Inputs and submission

- Train against the supplied target. Each period is cumulative, so keep the runner’s period splits and handle target warm-up zeros as expected input. Score against the supplied proprietary returns using the organizer’s one-hour position lag and 5 bp L1 turnover charge.
- Deliver a single UTF-8 `.py` file without a BOM. Implement `train(features, target)` and `predict(features)` on a subclass of the shipped `Predictor`. Keep submission logic in the subclass and declare any extra dependencies through PEP 723.
- Preserve the features’ timestamp index and 20 asset columns in the prediction output. Design inference to be deterministic across repeated calls and after model reload, to accept overlapping training and inference frames, and to tolerate gate inputs with permuted rows and variable lengths.

[Competition data](../../../requirements/COMPETITION.md#competition-data) · [Submission format](../../../requirements/COMPETITION.md#submission-format)

## Evidence

One offline validation command calls the shipped runner, collects the evidence below, and issues the local submit-or-hold decision. See the architecture spine for ownership and execution rules. [Architecture spine](../../planning-artifacts/architecture/architecture-Biweekly-2026-11-22/ARCHITECTURE-SPINE.md)

| Phase | Required evidence |
| --- | --- |
| Quick preflight | Run the shipped runner in quick mode for the exact candidate from the kit directory. Require a successful result and a de-mean pass. Inspect returned errors and diagnostics because failed loads can return without a failing process exit. |
| Model behavior | Verify aligned, finite, de-meaned predictions across representative periods, including exact input row and asset column order. Compare repeated and reloaded predictions and outputs on shared timestamps when padding preserves the available history. Check for extended or frequent near-zero exposure, permuted gate rows, variable lengths, and inference frames that overlap training. These are local diagnostics. AlphaNova determines the official verdict. |
| Full benchmark | Use the shipped full walk-forward for the shipped baseline, candidate, and matched control with the same kit, data snapshot, runtime, and scoring settings. Require every expected period and per-period observation count because the runner can skip failed periods. Retain unrounded Sharpe, per-call elapsed time on CPU, and measured peak memory. Record the candidate-minus-control difference to assess the vote. |
| Reproducibility | Record each attempt with the exact artifact digests, kit and input snapshots, resolved Python and dependencies, settings, period and observation counts, and returned scores. Preserve separate quick, full, and novelty results. Apply all candidate checks to the same artifact digest. |
| Novelty screen | Stage the official Season 2 map at `signal_db/signal_cities_ambient_coords.parquet` in the kit directory. Require all expected candidate gauge rows and verify the map is current at decision time. Record the snapshot, nearest city, and unrounded `city_novelty_deg` returned by `validate_submission(..., gauge_fix=True)`. The console and CSV round the distance to one decimal place. Require a finite distance of at least 60°. Hold below 60° or when evidence is incomplete or unavailable, including `nan` or `N/A`. |
| Official outcome | The contestant submits the exact file checked locally. Record the official entry verdict, settled Sharpe, and quality-set membership against the submitted digest and Cycle 1. AlphaNova evaluates hidden history and the live window, may apply additional execution costs, and runs the shuffled-label gate. |

[Runner](../../../requirements/runner.py) · [Walk-forward evaluator](../../../requirements/walkforward.py) · [Entry checks](../../../requirements/COMPETITION.md#entry-checks-and-the-overfitting-gate)

## Manual Season 2 Cycle 1 handoff

Run these commands from the repository root. Replace `ATTEMPT_DIR` with the validator's evidence directory for a completed `ready` attempt.

```bash
python scripts/record_cycle_outcome.py prepare "ATTEMPT_DIR"
```

Select the archived `.py` file for upload to AlphaNova. Before submitting, compute the SHA-256 of the file you actually selected and confirm it matches the digest printed by `prepare`. In PowerShell, run `Get-FileHash -Algorithm SHA256 "UPLOADED_FILE.py"`. Use the printed handoff directory as `HANDOFF_DIR`. The handoff is stored under `_bmad-output/evidence/submissions/season-2-cycle-1/`. [Official Cycle 1 page](https://www.alphanova.tech/competition/biweekly/season-2/cycle-1)

`prepare` checks that the handoff filesystem supports locking and publication of a complete observation without overwriting an existing file. The check uses same-volume rename on Windows and hard links on POSIX. If the check fails, use a filesystem that supports those operations and rerun `prepare` before uploading.

After uploading, replace `SUBMITTED_SHA256` with the SHA-256 of the uploaded file and enter the verdict reported by AlphaNova (`pending`, `accepted`, or `rejected`).

```bash
python scripts/record_cycle_outcome.py record "HANDOFF_DIR" --submitted-sha256 SUBMITTED_SHA256 --verdict pending
```

Use the official entry ID for each upload, including repeated uploads of the same file. Once the handoff contains two distinct official entry IDs, add `--official-entry-id "ENTRY_ID"` to every new observation, including the pending command above. In a handoff with fewer than two known IDs, you can omit an unknown ID. The observation stores `null` and stays unassigned.

When AlphaNova reports more facts, run `record` again with the current verdict and any available fields. Use `--official-entry-id` for the AlphaNova entry ID, `--verdict-detail` for the verdict message, `--settled-sharpe=VALUE` for the finite settled score, and `--quality-set-membership yes` or `--quality-set-membership no` for the settled selection result. The equals form also accepts negative scientific notation, such as `--settled-sharpe=-1.23e-3`.

```bash
python scripts/record_cycle_outcome.py record "HANDOFF_DIR" --submitted-sha256 SUBMITTED_SHA256 --verdict accepted --official-entry-id "ENTRY_ID" --settled-sharpe=0.0 --quality-set-membership no
```

Omit optional fields that AlphaNova has yet to report, following the entry-ID requirement above. Each successful `record` call writes a new UTC-timestamped JSON file under `HANDOFF_DIR/observations/`. The command runs offline and records the facts you supply. Once the file is published, `record` exits successfully even if temporary cleanup or output fails. It flushes the saved path to standard output and reports the path on standard error if standard output is unavailable. If both streams are unavailable, check the observation directory for the saved JSON.

## Quality-set selection

- AlphaNova ranks positive-Sharpe signals. Admission requires the absolute time-averaged cross-sectional correlation to be at most 0.5 against each already selected signal and each signal in the season’s legacy pot. The top three signals in the set earn prizes.
- Every entry that passes the checks joins the season’s legacy pot when its cycle closes. Use a distinct signal for a later cycle.

[Quality signal selection](../../../requirements/COMPETITION.md#quality-signal-selection) · [Legacy pot rules](../../../requirements/COMPETITION.md#the-legacy-pot--why-each-cycle-needs-new-alpha)
