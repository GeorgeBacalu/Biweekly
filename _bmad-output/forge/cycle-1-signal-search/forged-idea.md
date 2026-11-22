# Cycle 1 signal search

- Test feature agreement first. Orient the six same-hour cross-sectional ranks using each training split alone. Weight each directional vote by its rank distance from the median and average the signed votes for each asset.
- Compare the candidate with a control that shares the rank inputs, learner settings, and position smoothing. Add only the weighted vote. Score both against the shipped baseline on the full walk-forward, with the same period coverage and settings.
- Use the full walk-forward for every comparison. The separate one-time holdout was set aside. Defer rank-change and turnover-only experiments to isolate feature agreement.
- Submit only with positive net Sharpe above `demo_engineered.py` and an unrounded nearest-city distance of at least 60° on the current Season 2 map. Hold below 60° or when the comparison is unavailable.
