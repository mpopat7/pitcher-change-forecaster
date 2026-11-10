import unittest

import numpy as np
import pandas as pd

from pcf.injury_warning import label_rows, model_columns, parse_placements, verdict


def _t(text, date="2024-06-10", pid=7):
    return {"description": text, "date": date, "person": {"id": pid}}


class PlacementTests(unittest.TestCase):
    def test_parses_placements_and_retroactive_dates(self):
        tx = [_t("Rays placed LHP A on the 15-day injured list retroactive to June 7, 2024. Elbow."),
              _t("Cubs placed RHP B on the 60-day injured list.", pid=8),
              _t("Mets transferred RHP C to the 60-day injured list.", pid=9),
              _t("Mets placed RHP D on the COVID-19 injured list.", pid=10),
              _t("Mets activated RHP E from the 15-day injured list.", pid=11),
              {"description": "Twins placed RHP F on the 15-day injured list.", "date": "2024-06-10"}]
        p = parse_placements(tx)
        self.assertEqual(p.pitcher.tolist(), [7, 8])
        self.assertEqual(p.event.tolist(), [pd.Timestamp("2024-06-07"), pd.Timestamp("2024-06-10")])


class LabelTests(unittest.TestCase):
    def setUp(self):
        dates = pd.to_datetime(["2024-06-03", "2024-06-10", "2024-06-17"])
        self.rows = pd.DataFrame({"pitcher": [1, 1, 2], "forecast_date": dates, "season": 2024,
                                  "days_since_last": [3, 5, 2]})

    def test_labels_windows_history_and_exclusion(self):
        placements = pd.DataFrame({"pitcher": [1, 1, 2], "event": pd.to_datetime(
            ["2024-06-08", "2023-08-01", "2024-07-20"])})
        out = label_rows(self.rows, placements).set_index(["pitcher", "forecast_date"])
        first = out.loc[(1, pd.Timestamp("2024-06-03"))]
        self.assertTrue(first.y_primary)            # 5 days ahead
        self.assertTrue(first.gap_lead_time)        # inside days 0-6
        self.assertEqual(first.il_365, 1.0)         # Aug 2023 placement
        self.assertEqual(first.il_prev_season, 1.0)
        # 2024-06-10: placed two days before, after last appearance (5 days ago): already injured.
        self.assertNotIn((1, pd.Timestamp("2024-06-10")), out.index)
        second = out.loc[(2, pd.Timestamp("2024-06-17"))]
        self.assertFalse(second.y_primary)          # 33 days ahead
        self.assertEqual(second.il_365, 0.0)

    def test_lead_time_window(self):
        placements = pd.DataFrame({"pitcher": [2], "event": pd.to_datetime(["2024-07-20"])})
        out = label_rows(self.rows, placements)
        r = out[out.pitcher == 2].iloc[0]
        self.assertFalse(r.y_primary)               # 33 days ahead
        self.assertTrue(r.y_lead_time)
        self.assertFalse(r.gap_lead_time)


class ModelTests(unittest.TestCase):
    def test_nested_feature_sets(self):
        cols = ["k_logit_r30", "fb_velo_cur", "chg_fb_velo", "chg_fb_arm"]
        m = model_columns(cols)
        self.assertLess(set(m["W"]), set(m["V"]))
        self.assertLess(set(m["V"]), set(m["X"]))
        self.assertIn("il_365", m["W"])
        self.assertIn("chg_fb_arm", m["X"])
        self.assertNotIn("chg_fb_arm", m["V"])

    def test_verdict(self):
        s = pd.DataFrame([{"label": "primary", "season": "pooled", "comparison": "X-W",
                           "diff": -0.001, "ci_low": -0.002, "ci_high": -0.0001}])
        self.assertEqual(verdict(s)["changes_warn"]["verdict"], "supported")


if __name__ == "__main__":
    unittest.main()
