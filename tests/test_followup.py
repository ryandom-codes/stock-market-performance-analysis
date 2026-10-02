"""Keep the follow-up's date boundaries and uncertainty calculation honest."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from beat_the_baseline import paired_accuracy_interval, check_fresh_sessions
from test_integrity import synthetic_data


class FollowupTests(unittest.TestCase):
    def test_followup_rejects_missing_sessions_or_incomplete_history(self):
        dates = pd.bdate_range("2025-10-01", "2026-09-30")
        data = synthetic_data().groupby("Ticker").tail(len(dates)).copy()
        data["Date"] = dates[data.groupby("Ticker").cumcount()]
        data = data.sort_values(["Date", "Ticker"]).reset_index(drop=True)
        check_fresh_sessions(data)
        for broken in [data.iloc[1:], data[data.Date < "2026-09-30"]]:
            with self.assertRaises(ValueError):
                check_fresh_sessions(broken)

    def test_matching_always_up_has_exactly_zero_gain(self):
        days = pd.bdate_range("2025-01-01", periods=30)
        frame = pd.DataFrame({"Date": np.repeat(days, 8),
                              "target": np.tile([0, 1], 120), "prediction": 1})
        self.assertEqual(paired_accuracy_interval(frame), [0.0, 0.0])

    def test_resampling_keeps_a_markets_stocks_together(self):
        days = pd.bdate_range("2025-01-01", periods=30)
        # Every market day has the same mixture. Resampling whole dates must give
        # a fixed gain even though the individual stock outcomes differ.
        frame = pd.DataFrame({"Date": np.repeat(days, 8),
                              "target": np.tile([0, 0, 0, 0, 1, 1, 1, 1], 30),
                              "prediction": np.tile([0, 1, 1, 1, 1, 1, 1, 1], 30)})
        self.assertEqual(paired_accuracy_interval(frame), [0.125, 0.125])


if __name__ == "__main__":
    unittest.main()
