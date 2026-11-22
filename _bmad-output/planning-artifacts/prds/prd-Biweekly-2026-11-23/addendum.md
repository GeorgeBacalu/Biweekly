# Implementation context

## First experiment

- For each hour, rank each of the six supplied features across the 20 assets. Learn feature orientation within each training split. Weight each signed vote by its rank distance from the median, then take the arithmetic mean of those weighted votes. [Forged experiment](../../../forge/cycle-1-signal-search/forged-idea.md)
- One maintained predictor pipeline produces the candidate and matched control. The self-contained files differ only at the vote selector. Base ranks, learner settings, seed, and position shaping are shared. Use the full walk-forward for each comparison. [Architecture spine, AD-2 through AD-4](../../architecture/architecture-Biweekly-2026-11-22/ARCHITECTURE-SPINE.md)

## Submission and validation boundaries

- The checked submission is one UTF-8 `.py` file without a BOM. Keep logic inside the `Predictor` subclass and use PEP 723 for extra dependencies. City maps, run records, and the offline command stay outside the submission. [Architecture spine, AD-1](../../architecture/architecture-Biweekly-2026-11-22/ARCHITECTURE-SPINE.md)
- Keep training state per fit, inference read-only, and trailing history finite. Leave the full-panel feature hook unused. Follow the kit’s cumulative period splits and treat target warm-up zeros as expected input. [Architecture spine, AD-3](../../architecture/architecture-Biweekly-2026-11-22/ARCHITECTURE-SPINE.md) · [Validation checklist](../../../specs/spec-biweekly/validation.md)
- Resolve dependencies for the baseline, candidate, and control in one CPU environment before scoring. Include XGBoost for the shipped baseline. Set `COMP6_DISABLE_PIP_INSTALL=1` for the attempt. The offline command delegates scoring to the kit and uses its returned full-precision metrics. [Architecture spine, AD-5 and execution environments](../../architecture/architecture-Biweekly-2026-11-22/ARCHITECTURE-SPINE.md) · [Validation checklist](../../../specs/spec-biweekly/validation.md)
- Keep the city map snapshot fixed during the run and refresh novelty evidence for the same candidate if the map changes before the decision. [Architecture spine, AD-5 and AD-6](../../architecture/architecture-Biweekly-2026-11-22/ARCHITECTURE-SPINE.md)

## Deferred implementation choices

During the first build, choose the orientation estimator, learner and smoothing settings, evidence format, and artifact filenames. Decide how to handle rank ties and missing values. Document the local near-zero exposure criterion before implementing its check. The shared-pipeline and validation requirements govern those choices. [Architecture spine, Deferred](../../architecture/architecture-Biweekly-2026-11-22/ARCHITECTURE-SPINE.md#deferred) · [Entry checks](../../../../requirements/COMPETITION.md#entry-checks-and-the-overfitting-gate)
