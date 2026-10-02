"""A fixed follow-up: can accuracy-focused models beat always-up on fresh dates?"""
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from market_analysis import (ROOT, STOCKS, TICKERS, OHLCV, FEATURES, SEED,
                             build_market_signals, get_market_history,
                             check_price_history, measure_predictions, write_json)

HOLDOUT_START = pd.Timestamp("2026-01-01")
HOLDOUT_END = "2026-10-01"
OUT = ROOT / "results" / "followup"


def choose_before_looking(panel):
    """Use four earlier years to choose one model and one decision threshold."""
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, max_iter=2000, random_state=SEED)),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=5,
            min_samples_leaf=100, random_state=SEED, n_jobs=-1),
    }
    rows = []
    for year in [2022, 2023, 2024, 2025]:
        start, end = pd.Timestamp(year, 1, 1), pd.Timestamp(year + 1, 1, 1)
        train = panel[(panel.Date < start) & (panel.target_date < start)]
        valid = panel[(panel.Date >= start) & (panel.Date < end) & (panel.target_date < end)]
        assert train.target_date.max() < valid.Date.min()
        baseline_accuracy = float(valid.target.mean())
        for name, template in models.items():
            model = clone(template).fit(train[FEATURES], train.target)
            probability = model.predict_proba(valid[FEATURES])[:, 1]
            for threshold in [0.50, 0.45, 0.55]:
                prediction = (probability >= threshold).astype(int)
                rows.append({"year": year, "model": name, "threshold": threshold,
                             "rows": len(valid), "accuracy": float((prediction == valid.target).mean()),
                             "always_up_accuracy": baseline_accuracy})
    folds = pd.DataFrame(rows)
    folds.to_csv(OUT / "yearly_validation.csv", index=False)
    ranking = folds.groupby(["model", "threshold"], sort=False)[["accuracy", "always_up_accuracy"]].mean().reset_index()
    ranking.to_csv(OUT / "validation_summary.csv", index=False)
    best = ranking.accuracy.max()
    eligible = ranking[ranking.accuracy >= best - 0.001].copy()
    eligible["simplicity"] = eligible.model.map({"Logistic Regression": 0, "Random Forest": 1})
    eligible["threshold_order"] = eligible.threshold.map({0.50: 0, 0.45: 1, 0.55: 2})
    winner = eligible.sort_values(["simplicity", "threshold_order"]).iloc[0]
    choice = {"model": winner.model, "threshold": float(winner.threshold),
              "validation_mean_accuracy": float(winner.accuracy),
              "validation_mean_always_up_accuracy": float(winner.always_up_accuracy),
              "preferred_on_validation": bool(winner.accuracy > winner.always_up_accuracy + 0.001)}
    # Write the choice before the holdout download. No picking a winner after seeing 2026.
    write_json(OUT / "locked_choice.json", choice)
    print("Locked choice:", choice, flush=True)
    return clone(models[winner.model]), choice


def get_fresh_sessions():
    """Include a few warm-up months, but evaluate only the new 2026 dates."""
    cache = ROOT / "data" / "followup_prices.csv"
    metadata = ROOT / "data" / "followup_source.json"
    if cache.exists() and metadata.exists():
        source = json.loads(metadata.read_text(encoding="utf-8"))
        if hashlib.sha256(cache.read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError("Follow-up cache fingerprint changed.")
        return pd.read_csv(cache, parse_dates=["Date"]), source
    yf.set_tz_cache_location(str(ROOT / "data" / "yfinance-cache"))
    parts = []
    for ticker in TICKERS:
        print(f"Downloading fresh dates: {ticker}", flush=True)
        frame = yf.download(ticker, start="2025-10-01", end=HOLDOUT_END,
                            auto_adjust=True, multi_level_index=False,
                            threads=False, progress=False, timeout=30)
        if frame is None or frame.empty:
            raise ValueError(f"No follow-up data for {ticker}.")
        frame = frame[OHLCV].copy()
        frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
        frame.index.name = "Date"
        parts.append(frame.reset_index().assign(Ticker=ticker))
    data = pd.concat(parts, ignore_index=True).sort_values(["Date", "Ticker"])
    data.to_csv(cache, index=False, date_format="%Y-%m-%d")
    source = {"provider": "Yahoo Finance via yfinance", "start": "2025-10-01",
              "end_exclusive": HOLDOUT_END, "auto_adjust": True,
              "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
              "sha256": hashlib.sha256(cache.read_bytes()).hexdigest()}
    write_json(metadata, source)
    return data, source


def check_fresh_sessions(data):
    if data.duplicated(["Date", "Ticker"]).any() or data[["Date", "Ticker"] + OHLCV].isna().any().any():
        raise ValueError("Duplicate or missing follow-up observations.")
    if not np.isfinite(data[OHLCV].to_numpy()).all() or (data[OHLCV] <= 0).any().any():
        raise ValueError("Invalid follow-up prices or volume.")
    if ((data.High + 1e-5 < data[["Open", "Close", "Low"]].max(axis=1)) |
        (data.Low - 1e-5 > data[["Open", "Close", "High"]].min(axis=1))).any():
        raise ValueError("Inconsistent follow-up OHLC values.")
    if set(data.Ticker) != set(TICKERS):
        raise ValueError("Missing follow-up tickers.")
    dates = pd.Index(data.loc[data.Ticker == "SPY", "Date"])
    for ticker, frame in data.groupby("Ticker"):
        if not pd.Index(frame.Date).equals(dates):
            raise ValueError(f"Missing follow-up sessions for {ticker}.")
    if data.Date.min() > pd.Timestamp("2025-10-10") or data.Date.max() != pd.Timestamp("2026-09-30"):
        raise ValueError("Incomplete follow-up history.")
    if not ((data.Date >= "2025-10-01") & (data.Date < HOLDOUT_END)).all():
        raise ValueError("Follow-up dates outside the fixed window.")


def paired_accuracy_interval(predictions):
    """Resample whole blocks of market days, not supposedly independent stocks."""
    difference = ((predictions.prediction == predictions.target).astype(float) -
                  (predictions.target == 1).astype(float))
    daily = predictions.assign(difference=difference).groupby("Date").difference.mean().to_numpy()
    rng = np.random.default_rng(SEED)
    length = len(daily)
    samples = []
    for _ in range(2000):
        starts = rng.integers(0, length, size=int(np.ceil(length / 10)))
        indices = ((starts[:, None] + np.arange(10)) % length).ravel()[:length]
        samples.append(daily[indices].mean())
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    old, old_source = get_market_history()
    old, _ = check_price_history(old)
    development = build_market_signals(old).dropna(subset=FEATURES + ["target", "target_date"])
    development = development[development.target_date < HOLDOUT_START]
    model, choice = choose_before_looking(development)
    model.fit(development[FEATURES], development.target)
    fresh, source = get_fresh_sessions()
    check_fresh_sessions(fresh)
    test = build_market_signals(fresh).dropna(subset=FEATURES + ["target", "target_date"])
    test = test[(test.Date >= HOLDOUT_START) & (test.target_date < HOLDOUT_END)]
    assert development.target_date.max() < test.Date.min()
    assert (test.groupby("Date").Ticker.nunique() == len(STOCKS)).all()
    probability = model.predict_proba(test[FEATURES])[:, 1]
    prediction = (probability >= choice["threshold"]).astype(int)
    output = test[["Date", "Ticker", "target_date", "target"]].copy()
    output["prediction"], output["probability_up"] = prediction, probability
    output.to_csv(OUT / "holdout_predictions.csv", index=False, float_format="%.8f")
    candidate = measure_predictions(test.target, prediction, probability)
    baseline = measure_predictions(test.target, np.ones(len(test)), np.ones(len(test)))
    metrics = pd.DataFrame([{"model": choice["model"], **candidate}, {"model": "Always up", **baseline}])
    metrics.to_csv(OUT / "holdout_metrics.csv", index=False)
    report = {"choice": choice, "development_source": old_source, "holdout_source": source,
              "feature_start": str(test.Date.min().date()), "feature_end": str(test.Date.max().date()),
              "last_label_date": str(test.target_date.max().date()), "rows": len(test),
              "market_days": int(test.Date.nunique()), "coverage": 1.0,
              "predicted_down_fraction": float((prediction == 0).mean()),
              "accuracy_gain": candidate["accuracy"] - baseline["accuracy"],
              "accuracy_gain_95pct_block_interval": paired_accuracy_interval(output),
              "bootstrap": {"block_sessions": 10, "draws": 2000, "seed": SEED}}
    write_json(OUT / "holdout_report.json", report)
    print(metrics.to_string(index=False))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
