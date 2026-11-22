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

## Quality-set selection

- AlphaNova ranks positive-Sharpe signals. Admission requires the absolute time-averaged cross-sectional correlation to be at most 0.5 against each already selected signal and each signal in the season’s legacy pot. The top three signals in the set earn prizes.
- Every entry that passes the checks joins the season’s legacy pot when its cycle closes. Use a distinct signal for a later cycle.

[Quality signal selection](../../../requirements/COMPETITION.md#quality-signal-selection) · [Legacy pot rules](../../../requirements/COMPETITION.md#the-legacy-pot--why-each-cycle-needs-new-alpha)
