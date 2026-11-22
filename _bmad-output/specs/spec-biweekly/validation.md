# Submission and validation

## Inputs and submission

- Train against the supplied target. Each period is cumulative, so keep the runner’s period splits and handle target warm-up zeros as expected input. Score against the supplied proprietary returns using the organizer’s one-hour position lag and 5 bp L1 turnover charge.
- Deliver a single UTF-8 `.py` file without a BOM. Implement `train(features, target)` and `predict(features)` on a subclass of the shipped `Predictor`. Keep submission logic in the subclass and declare any extra dependencies through PEP 723.
- Preserve the features’ timestamp index and 20 asset columns in the prediction output. Design inference to be deterministic across repeated calls and after model reload, to accept overlapping training and inference frames, and to tolerate gate inputs with permuted rows and variable lengths.

[Competition data](../../../requirements/COMPETITION.md#competition-data) · [Submission format](../../../requirements/COMPETITION.md#submission-format)

## Evidence

| Phase | Required evidence |
| --- | --- |
| Quick preflight | Run `python runner.py <candidate.py>` from the kit directory. Require the de-mean check to pass and confirm the run produced a valid result. Inspect diagnostics as well as the result, since a failed load can return without a failing process exit. |
| Model behavior | Verify aligned, finite, de-meaned predictions across representative periods. Compare repeated and reloaded predictions and outputs on shared timestamps when padding preserves their available history. Check for extended or frequent near-zero exposure and exercise retraining on permuted rows. These are local diagnostics. AlphaNova determines the official verdict. |
| Full benchmark | Run `python runner.py demo_engineered.py --full` and `python runner.py <candidate.py> --full` from the kit directory. Verify complete matching period and observation coverage because the runner can skip failed periods while still reporting aggregate Sharpe. Check timing diagnostics and retain evidence of peak memory. |
| Reproducibility | Record candidate and baseline code versions, kit and input data version, environment, command options, evaluated periods and observation counts, and both net Sharpe results. Keep the quick, full, and gauge-fix outputs distinguishable. |
| Novelty screen | Run `python runner.py <candidate.py> --gauge-fix` with the current Season 2 city map at `signal_db/signal_cities_ambient_coords.parquet` in the kit directory. Record the map snapshot, nearest city, and distance. `nan` or `N/A` leaves the comparison unavailable. |
| Official outcome | Preserve the exact submitted file. Record the entry verdict and, when settled, official Sharpe and quality-set membership. The organizer evaluates additional hidden history and the live window, may apply additional execution costs, and runs the shuffled-label gate. |

[Runner](../../../requirements/runner.py) · [Walk-forward evaluator](../../../requirements/walkforward.py) · [Entry checks](../../../requirements/COMPETITION.md#entry-checks-and-the-overfitting-gate)

## Quality-set selection

- AlphaNova ranks positive-Sharpe signals. Admission requires the absolute time-averaged cross-sectional correlation to be at most 0.5 against each already selected signal and each signal in the season’s legacy pot. The top three signals in the set earn prizes.
- Every entry that passes the checks joins the season’s legacy pot when its cycle closes. Use a distinct signal for a later cycle.

[Quality signal selection](../../../requirements/COMPETITION.md#quality-signal-selection) · [Legacy pot rules](../../../requirements/COMPETITION.md#the-legacy-pot--why-each-cycle-needs-new-alpha)
