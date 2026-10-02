"""A few checks for the mistakes that could make these results look too good."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from market_analysis import (FEATURES, TICKERS, build_market_signals,
                             split_without_peeking, check_price_history)


def synthetic_data():
    dates = pd.bdate_range("2015-01-02", "2025-12-31")
    parts = []
    for i, ticker in enumerate(TICKERS):
        close = 100 * np.exp(np.cumsum(np.sin(np.arange(len(dates)) + i) * 0.01))
        parts.append(pd.DataFrame({"Date": dates, "Ticker": ticker, "Open": close,
                                   "High": close * 1.01, "Low": close * 0.99,
                                   "Close": close, "Volume": 100000 + np.arange(len(dates))}))
    return pd.concat(parts, ignore_index=True).sort_values(["Date", "Ticker"]).reset_index(drop=True)


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.data = synthetic_data()

    def test_future_prices_do_not_change_earlier_features(self):
        cutoff = pd.Timestamp("2023-06-01")
        original = build_market_signals(self.data)
        changed = self.data.copy()
        changed.loc[changed.Date > cutoff, ["Open", "High", "Low", "Close", "Volume"]] *= 3
        revised = build_market_signals(changed)
        pd.testing.assert_frame_equal(original.loc[original.Date <= cutoff, FEATURES],
                                      revised.loc[revised.Date <= cutoff, FEATURES])

    def test_unknown_final_labels_are_missing(self):
        panel = build_market_signals(self.data)
        self.assertTrue(panel.groupby("Ticker").tail(1).target.isna().all())

    def test_target_is_exactly_next_session_direction(self):
        panel = build_market_signals(self.data)
        stock = panel[panel.Ticker == "AAPL"].iloc[25]
        original = self.data[self.data.Ticker == "AAPL"].reset_index(drop=True)
        expected = int(original.loc[26, "Close"] > original.loc[25, "Close"])
        self.assertEqual(stock.target, expected)
        self.assertEqual(stock.target_date, original.loc[26, "Date"])

    def test_labels_and_dates_cannot_cross_partitions(self):
        parts, audit = split_without_peeking(build_market_signals(self.data))
        self.assertLess(parts["train"].target_date.max(), parts["validation"].Date.min())
        self.assertLess(parts["validation"].target_date.max(), parts["test"].Date.min())
        self.assertEqual(audit["purged_boundary_rows"], 16)
        self.assertFalse(set(parts["train"].Date) & set(parts["validation"].Date))
        self.assertTrue((parts["test"].groupby("Date").Ticker.nunique() == 8).all())

    def test_invalid_data_is_rejected(self):
        check_price_history(self.data)
        for corrupted in [pd.concat([self.data, self.data.iloc[:1]]),
                          self.data.drop(index=0),
                          self.data.assign(High=0),
                          self.data.assign(Close=np.nan)]:
            with self.assertRaises(ValueError):
                check_price_history(corrupted)


if __name__ == "__main__":
    unittest.main()
