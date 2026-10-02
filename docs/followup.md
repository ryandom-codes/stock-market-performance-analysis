# Can a model beat "always up"?

**Historical experiment:** these rules describe the first follow-up run on October 1, 2026. The 2026 results have since been examined; rerunning this script reproduces them and does not create a fresh holdout. Subsequent changes need a new prospective evaluation.

The first experiment focused on balanced accuracy and didn't beat the baseline. This follow-up asks a different, narrower question: can the same features improve overall next-day direction accuracy?

## Rules set before looking at 2026

- Keep the original eight stocks, SPY, 15 features, and next-day target.
- Treat 2015–2025 as development data now. The previous 2024–2025 test has already been inspected and is no longer a fresh holdout.
- Compare the same Logistic Regression and shallow Random Forest configurations, but remove balanced class weights to match the accuracy objective.
- Try only probability thresholds 0.45, 0.50, and 0.55 for each model. Predict up at or above the threshold.
- Validate on each of 2022, 2023, 2024, and 2025, training on all earlier years. Purge labels that reach each validation year's first session or leave that year.
- Choose the candidate and threshold with the highest mean yearly accuracy. Within 0.1 percentage point, prefer Logistic Regression, then threshold 0.50, 0.45, 0.55. Require a development gain above 0.1 percentage point over always-up before calling it the preferred approach.
- Lock that candidate before downloading the new holdout. Refit using retained data through 2025.
- Evaluate the locked candidate and always-up once on January–September 2026 (download end October 1, exclusive). No further changes based on this holdout.
- Report every validation configuration, the chosen candidate's holdout metrics, prediction coverage, and a paired 95% block-bootstrap interval for its accuracy difference from always-up. Use 10-session circular blocks, all stocks kept together within dates, 2,000 draws, and seed 42.

A higher score on 2026 than the old 54.69% score isn't enough: the model must beat always-up on the same 2026 observations. A small observed gain alone doesn't establish a dependable forecasting advantage. The bootstrap interval is approximate and assumes the observed period and chosen block length are informative.

Run with `.\.venv\Scripts\python.exe src\beat_the_baseline.py`. Outputs go to `results/followup/`. The original experiment and results stay available for comparison.
