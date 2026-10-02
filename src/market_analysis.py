"""Can yesterday's market tell us much about tomorrow?

Run from any directory: python path/to/src/market_analysis.py
Compare eight stocks with SPY, then put a few simple direction models to the test.
The prediction happens after today's close, using only what is known by then.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data" / "matplotlib-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

STOCKS = ["AAPL", "MSFT", "AMZN", "GOOGL", "JPM", "JNJ", "XOM", "WMT"]
TICKERS = STOCKS + ["SPY"]
START, END = "2015-01-01", "2026-01-01"  # end is exclusive
VALIDATION_START, TEST_START = pd.Timestamp("2022-01-01"), pd.Timestamp("2024-01-01")
OHLCV = ["Open", "High", "Low", "Close", "Volume"]
FEATURES = ["return_1d", "return_lag1", "return_lag2", "momentum_5d",
            "momentum_20d", "ma_gap_10d", "ma_gap_20d", "volatility_20d",
            "volume_change", "volume_relative_20d", "intraday_range",
            "intraday_return", "spy_return_1d", "spy_momentum_5d",
            "excess_return_1d"]
SEED = 42


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def get_market_history(refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    """Download once so reruns compare models using the same price history."""
    cache = ROOT / "data" / "prices.csv"
    provenance = ROOT / "data" / "source.json"
    if cache.exists() and provenance.exists() and not refresh:
        data = pd.read_csv(cache, parse_dates=["Date"])
        source = json.loads(provenance.read_text(encoding="utf-8"))
        if source["sha256"] != hashlib.sha256(cache.read_bytes()).hexdigest():
            raise ValueError("Cached data hash mismatch. Use --refresh to download again.")
        return data, source
    yf.set_tz_cache_location(str(ROOT / "data" / "yfinance-cache"))
    parts = []
    for ticker in TICKERS:
        print(f"Downloading {ticker}...", flush=True)
        frame = yf.download(ticker, start=START, end=END, auto_adjust=True,
                            progress=False, threads=False, multi_level_index=False,
                            timeout=30)
        if frame is None or frame.empty:
            raise RuntimeError(f"No data for {ticker}. Check connectivity and retry.")
        frame = frame[OHLCV].copy()
        frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
        frame.index.name = "Date"
        parts.append(frame.reset_index().assign(Ticker=ticker))
    data = pd.concat(parts, ignore_index=True).sort_values(["Date", "Ticker"])
    data.to_csv(cache, index=False, date_format="%Y-%m-%d")
    source = {"provider": "Yahoo Finance via yfinance", "auto_adjust": True,
              "start_inclusive": START, "end_exclusive": END, "tickers": TICKERS,
              "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
              "sha256": hashlib.sha256(cache.read_bytes()).hexdigest()}
    write_json(provenance, source)
    return data, source


def check_price_history(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Catch missing days and strange prices before they become model inputs."""
    data = data.sort_values(["Date", "Ticker"]).reset_index(drop=True).copy()
    if set(data.Ticker.unique()) != set(TICKERS):
        raise ValueError("Unexpected or missing tickers.")
    if data.duplicated(["Date", "Ticker"]).any():
        raise ValueError("Duplicate ticker/date observations.")
    if data[["Date", "Ticker"] + OHLCV].isna().any().any():
        raise ValueError("Missing OHLCV values. Inspect the source; no filling is applied.")
    if not np.isfinite(data[OHLCV].to_numpy()).all():
        raise ValueError("Non-finite OHLCV values.")
    if (data[["Open", "High", "Low", "Close"]] <= 0).any().any() or (data.Volume <= 0).any():
        raise ValueError("Nonpositive prices or volume.")
    tolerance = 1e-5
    if ((data.High + tolerance < data[["Open", "Close", "Low"]].max(axis=1)) |
        (data.Low - tolerance > data[["Open", "Close", "High"]].min(axis=1))).any():
        raise ValueError("OHLC price relationships are inconsistent.")
    if not ((data.Date >= START) & (data.Date < END)).all():
        raise ValueError("Dates fall outside the requested interval.")
    benchmark_dates = pd.Index(data.loc[data.Ticker == "SPY", "Date"])
    for ticker, group in data.groupby("Ticker"):
        if not pd.Index(group.Date).equals(benchmark_dates):
            raise ValueError(f"{ticker}: dates do not align with SPY; inspect missing sessions.")
    if data.Date.min() > pd.Timestamp("2015-01-10") or data.Date.max() < pd.Timestamp("2025-12-24"):
        raise ValueError("History is incomplete at the beginning or end.")
    audit = {"rows": len(data), "trading_dates": int(data.Date.nunique()),
             "tickers": len(TICKERS), "first_date": str(data.Date.min().date()),
             "last_date": str(data.Date.max().date()), "duplicate_rows": 0,
             "missing_ohlcv_cells": 0, "calendar_aligned_to_spy": True,
             "missing_value_policy": "fail; no forward or backward fill"}
    return data, audit


def build_market_signals(data: pd.DataFrame) -> pd.DataFrame:
    """Turn each stock's history into momentum, activity, and risk signals."""
    parts = []
    for ticker, frame in data.groupby("Ticker", sort=True):
        # Each stock gets its own rolling windows. Mixing tickers would scramble returns.
        frame = frame.sort_values("Date").copy()
        close, volume = frame.Close, frame.Volume
        returns = close.pct_change(fill_method=None)
        frame["return_1d"] = returns
        frame["return_lag1"] = returns.shift(1)
        frame["return_lag2"] = returns.shift(2)
        for window in [5, 20]:
            frame[f"momentum_{window}d"] = close.pct_change(window, fill_method=None)
        for window in [10, 20]:
            frame[f"ma_gap_{window}d"] = close / close.rolling(window).mean() - 1
        frame["volatility_20d"] = returns.rolling(20).std()
        frame["volume_change"] = volume.pct_change(fill_method=None)
        frame["volume_relative_20d"] = volume / volume.rolling(20).mean() - 1
        frame["intraday_range"] = (frame.High - frame.Low) / close
        frame["intraday_return"] = close / frame.Open - 1
        # Looking one day ahead is only allowed here, where we make the answer key.
        frame["target_date"] = frame.Date.shift(-1)
        frame["next_return"] = close.shift(-1) / close - 1
        # We don't know tomorrow for the last row. NaN > 0 would quietly call it a down day.
        frame["target"] = (frame.next_return > 0).astype(float).where(frame.next_return.notna())
        parts.append(frame)
    panel = pd.concat(parts, ignore_index=True)
    spy = panel.loc[panel.Ticker == "SPY", ["Date", "return_1d", "momentum_5d"]]
    spy = spy.rename(columns={"return_1d": "spy_return_1d", "momentum_5d": "spy_momentum_5d"})
    panel = panel.merge(spy, on="Date", how="left", validate="many_to_one")
    panel["excess_return_1d"] = panel.return_1d - panel.spy_return_1d
    return panel.loc[panel.Ticker != "SPY"].sort_values(["Date", "Ticker"]).reset_index(drop=True)


def split_without_peeking(panel: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict]:
    """Keep the calendar in order, including the dates used to make each label."""
    usable = panel.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + ["target", "target_date"]).copy()
    usable["target"] = usable.target.astype(int)
    # A December row can have a January answer. Drop it if that crosses the cutoff.
    train = usable[(usable.Date < VALIDATION_START) & (usable.target_date < VALIDATION_START)]
    validation = usable[(usable.Date >= VALIDATION_START) & (usable.Date < TEST_START) &
                        (usable.target_date < TEST_START)]
    test = usable[usable.Date >= TEST_START]
    parts = {"train": train, "validation": validation, "test": test}
    for name, frame in parts.items():
        if frame.empty or frame.target.nunique() != 2:
            raise ValueError(f"{name} needs observations from both classes.")
    assert train.target_date.max() < validation.Date.min()
    assert validation.target_date.max() < test.Date.min()
    audit = {name: {"rows": len(frame), "feature_start": str(frame.Date.min().date()),
                    "feature_end": str(frame.Date.max().date()),
                    "last_label_date": str(frame.target_date.max().date()),
                    "up_rate": float(frame.target.mean())} for name, frame in parts.items()}
    audit["excluded_warmup_or_unknown_label_rows"] = len(panel) - len(usable)
    audit["purged_boundary_rows"] = len(usable) - sum(len(p) for p in parts.values())
    return parts, audit


def measure_predictions(y, prediction, probability) -> dict:
    return {"accuracy": float(accuracy_score(y, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
            "precision_up": float(precision_score(y, prediction, zero_division=0)),
            "recall_up": float(recall_score(y, prediction, zero_division=0)),
            "f1_up": float(f1_score(y, prediction, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, probability))}


def run_direction_experiment(parts: dict) -> tuple[pd.DataFrame, pd.DataFrame, str, pd.DataFrame]:
    """Let validation pick a model, then see how that choice holds up later."""
    # 'Always up' is surprisingly hard to beat on accuracy. Start there.
    models = {
        "Majority baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": make_pipeline(StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=SEED)),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=5,
            min_samples_leaf=100, class_weight="balanced", random_state=SEED, n_jobs=-1),
    }
    train, validation = parts["train"], parts["validation"]
    rows = []
    for name, model in models.items():
        model.fit(train[FEATURES], train.target)
        rows.append({"split": "validation", "model": name,
                     **measure_predictions(validation.target, model.predict(validation[FEATURES]),
                             model.predict_proba(validation[FEATURES])[:, 1])})
    results = pd.DataFrame(rows)
    # Don't choose a more complicated model for a gain of 0.1 percentage point or less.
    best = results.balanced_accuracy.max()
    selected = next(name for name in models
                    if results.loc[results.model == name, "balanced_accuracy"].iloc[0] >= best - 0.001)
    print(f"Validation-selected model: {selected}", flush=True)
    # The choice is locked in now. Give it the earlier data, then open the test set.
    development = pd.concat([train, validation])
    test = parts["test"]
    predictions, by_ticker = [], []
    for name in dict.fromkeys(["Majority baseline", selected]):
        model = clone(models[name]).fit(development[FEATURES], development.target)
        predicted = model.predict(test[FEATURES])
        probability = model.predict_proba(test[FEATURES])[:, 1]
        rows.append({"split": "test", "model": name, **measure_predictions(test.target, predicted, probability)})
        output = test[["Date", "Ticker", "target_date", "target"]].copy()
        output["model"], output["prediction"], output["probability_up"] = name, predicted, probability
        predictions.append(output)
        for ticker, group in output.groupby("Ticker"):
            by_ticker.append({"model": name, "Ticker": ticker, "rows": len(group),
                              **measure_predictions(group.target, group.prediction, group.probability_up)})
    return pd.DataFrame(rows), pd.concat(predictions), selected, pd.DataFrame(by_ticker)


def save_chart(fig, name: str) -> None:
    fig.savefig(ROOT / "figures" / name, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def compare_risk_and_growth(data: pd.DataFrame) -> pd.DataFrame:
    """Look at the ride as well as the finish: growth, volatility, and drawdowns."""
    prices = data.pivot(index="Date", columns="Ticker", values="Close")
    returns = prices.pct_change(fill_method=None).dropna()
    growth = prices / prices.iloc[0]
    years = (prices.index[-1] - prices.index[0]).days / 365.25
    drawdown = prices / prices.cummax() - 1
    summary = pd.DataFrame({"total_return": growth.iloc[-1] - 1,
                            "cagr": growth.iloc[-1] ** (1 / years) - 1,
                            "annualized_volatility": returns.std() * np.sqrt(252),
                            "max_drawdown": drawdown.min(),
                            "correlation_with_spy": returns.corr()["SPY"]})
    summary.index.name = "Ticker"
    summary.to_csv(ROOT / "results" / "performance_summary.csv", float_format="%.6f")
    fig, ax = plt.subplots(figsize=(11, 6))
    for ticker in TICKERS:
        ax.plot(growth.index, growth[ticker] * 100,
                label=ticker, linewidth=2.8 if ticker == "SPY" else 1.4,
                color="#111827" if ticker == "SPY" else None)
    ax.set(title="Historical growth of $100 | adjusted daily closes", ylabel="Indexed value ($, log scale)")
    ax.set_yscale("log")
    ax.legend(ncol=3, loc="upper left")
    ax.grid(alpha=0.2)
    fig.text(0.1, 0.01, "2015–2025 • Descriptive performance; not model trading returns", fontsize=9)
    save_chart(fig, "historical_growth.png")
    fig, ax = plt.subplots(figsize=(9, 6))
    for ticker, row in summary.iterrows():
        ax.scatter(row.annualized_volatility * 100, row.cagr * 100,
                   s=110 if ticker == "SPY" else 65, color="#111827" if ticker == "SPY" else "#2563eb")
        ax.annotate(ticker, (row.annualized_volatility * 100, row.cagr * 100),
                    xytext=(6, 6), textcoords="offset points", fontsize=9)
    ax.margins(0.18)
    ax.set(title="Return versus risk | 2015–2025", xlabel="Annualized daily volatility (%)", ylabel="Compound annual growth rate (%)")
    ax.grid(alpha=0.2)
    save_chart(fig, "risk_return.png")
    fig, ax = plt.subplots(figsize=(9, 7))
    corr = returns[TICKERS].corr()
    img = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(TICKERS)), TICKERS, rotation=45)
    ax.set_yticks(range(len(TICKERS)), TICKERS)
    for i in range(len(TICKERS)):
        for j in range(len(TICKERS)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(corr.iloc[i, j]) > 0.65 else "black")
    ax.set_title("Daily return correlations | 2015–2025")
    fig.colorbar(img, ax=ax, shrink=0.75)
    save_chart(fig, "return_correlations.png")
    return summary


def plot_prediction_results(metrics: pd.DataFrame, predictions: pd.DataFrame, selected: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, partition in zip(axes, ["validation", "test"]):
        subset = metrics[metrics.split == partition]
        x = np.arange(len(subset))
        ax.bar(x - 0.18, subset.accuracy, 0.36, label="Accuracy", color="#94a3b8")
        ax.bar(x + 0.18, subset.balanced_accuracy, 0.36, label="Balanced accuracy", color="#2563eb")
        ax.axhline(0.5, color="#111827", ls="--", lw=1)
        ax.set_xticks(x, [n.replace(" ", "\n") for n in subset.model])
        ax.set(ylim=(0, 0.7), title=partition.title(), ylabel="Score")
        ax.legend(fontsize=8)
    fig.suptitle("Next-day direction | validation selects, test evaluates")
    fig.tight_layout()
    save_chart(fig, "model_comparison.png")
    winner = predictions[predictions.model == selected]
    matrix = confusion_matrix(winner.target, winner.prediction, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max())
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center", fontsize=20,
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Down / flat", "Up"])
    ax.set_yticks([0, 1], ["Down / flat", "Up"])
    ax.set(xlabel="Predicted", ylabel="Actual", title=f"{selected} | test confusion matrix")
    save_chart(fig, "confusion_matrix.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Replace the local market-data snapshot.")
    args = parser.parse_args()
    for folder in ["data", "figures", "results"]:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titlesize": 13})
    data, source = get_market_history(args.refresh)
    data, data_audit = check_price_history(data)
    print(f"Validated {len(data):,} OHLCV rows across {len(TICKERS)} instruments.", flush=True)
    panel = build_market_signals(data)
    parts, split_audit = split_without_peeking(panel)
    metrics, predictions, selected, by_ticker = run_direction_experiment(parts)
    metrics.to_csv(ROOT / "results" / "model_metrics.csv", index=False, float_format="%.6f")
    predictions.to_csv(ROOT / "results" / "test_predictions.csv", index=False, float_format="%.6f")
    by_ticker.to_csv(ROOT / "results" / "test_metrics_by_stock.csv", index=False, float_format="%.6f")
    # These full-history charts describe what happened; they don't pick the model.
    compare_risk_and_growth(data)
    plot_prediction_results(metrics, predictions, selected)
    manifest = {"source": source, "data_validation": data_audit, "splits": split_audit,
                "features": FEATURES, "selected_model": selected,
                "selection_rule": "Highest validation balanced accuracy; within 0.001 favor simplicity: baseline, logistic, forest.",
                "classification_threshold": 0.5, "seed": SEED,
                "python": platform.python_version(),
                "packages": {p: importlib.metadata.version(p) for p in
                             ["numpy", "pandas", "matplotlib", "scikit-learn", "yfinance"]}}
    write_json(ROOT / "results" / "run_manifest.json", manifest)
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved figures and results to {ROOT}")


if __name__ == "__main__":
    main()
