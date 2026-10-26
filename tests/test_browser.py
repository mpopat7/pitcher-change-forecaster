import json
import unittest

import pandas as pd

from pcf.browser import season_payload


class BrowserExportTests(unittest.TestCase):
    def setUp(self):
        self.watchlist = pd.DataFrame({
            "pitcher": [1, 2], "forecast_date": ["2025-04-07"] * 2,
            "y_bf": [0, 100], "y_k": [0, 25], "y_bb": [0, 5],
            "name": ["Missing, Followup", "Present, Followup"],
            "k_minus_bb_B": [0.30, 0.21], "performance_rank": [1, 2],
            "alert": [False, True], "p_k_B_raw": [0.40, 0.30],
        })
        self.context = pd.DataFrame({
            "pitcher": [1, 2], "forecast_date": ["2025-04-07"] * 2,
            "fb_velo_cur": [None, 94.0],
        })
        self.alerts = pd.DataFrame({
            "pitcher": [2], "forecast_date": ["2025-04-07"], "resolution": ["confirmed"],
        })

    def test_no_followup_is_retained_without_fabricated_zero_rates(self):
        payload = season_payload(self.watchlist, self.context, self.alerts)
        records = [dict(zip(payload["columns"], row)) for row in payload["data"]]
        self.assertEqual(len(records), 2)
        self.assertIsNone(records[0]["observed_k"])
        self.assertIsNone(records[0]["fb_velo_cur"])
        self.assertAlmostEqual(records[1]["observed_k_minus_bb"], 0.20)
        self.assertNotIn("p_k_B_raw", payload["columns"])
        json.dumps(payload, allow_nan=False)

    def test_missing_context_cannot_silently_drop_a_forecast(self):
        with self.assertRaisesRegex(ValueError, "different eligible populations"):
            season_payload(self.watchlist, self.context.iloc[1:], self.alerts)

    def test_duplicate_forecast_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate forecast"):
            season_payload(pd.concat([self.watchlist, self.watchlist.iloc[[0]]]),
                           self.context, self.alerts)

    def test_alert_resolution_must_come_from_the_saved_audit(self):
        with self.assertRaisesRegex(ValueError, "missing its saved resolution"):
            season_payload(self.watchlist, self.context, self.alerts.iloc[:0])


if __name__ == "__main__":
    unittest.main()
