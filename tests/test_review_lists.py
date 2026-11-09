import unittest

import numpy as np
import pandas as pd

from pcf.review_lists import build_lists, eligible, method_scores, retention, summarize


def _row(pitcher, pitch_type, metric, delta, **kw):
    base = {"pitcher": pitcher, "name": str(pitcher), "team": "DET", "season": 2022,
            "forecast_date": pd.Timestamp("2022-06-01"), "pitch_type": pitch_type, "metric": metric,
            "recent": 10 + delta, "baseline": 10.0, "delta": delta, "shrunk_delta": delta / 2,
            "prior_variance": 1.0, "type_outings_recent": 3, "type_outings_baseline": 3,
            "future": 10 + delta, "future_outings": 3}
    base.update(kw)
    return base


class ReviewListTests(unittest.TestCase):
    def test_support_floor_and_unknown_labels(self):
        frame = pd.DataFrame([_row(1, "FF", "velocity_mph", 1.0),
                              _row(2, "UN", "velocity_mph", 3.0),
                              _row(3, "SL", "velocity_mph", 2.0, type_outings_baseline=1),
                              _row(4, "CH", "velocity_mph", 2.0, season=2025)])
        self.assertEqual(eligible(frame, [2022]).pitcher.tolist(), [1])

    def test_each_anchor_keeps_its_strongest_signal(self):
        frame = eligible(pd.DataFrame([_row(1, "FF", "velocity_mph", 1.0),
                                       _row(1, "SL", "velocity_mph", -2.0),
                                       _row(1, "SL", "usage", 0.3)]), [2022])
        velo = method_scores(frame, "velocity_only")
        self.assertEqual(len(velo), 1)
        self.assertEqual(velo.pitch_type.item(), "SL")
        self.assertEqual(method_scores(frame, "usage_only").metric.item(), "usage")

    def test_physical_score_standardizes_by_prior_and_needs_positive_prior(self):
        frame = eligible(pd.DataFrame([_row(1, "FF", "velocity_mph", 1.0, shrunk_delta=0.5, prior_variance=0.25),
                                       _row(1, "FF", "hb_in", 2.0, shrunk_delta=0.8, prior_variance=4.0),
                                       _row(2, "FF", "hb_in", 5.0, shrunk_delta=5.0, prior_variance=0.0)]), [2022])
        physical = method_scores(frame, "physical")
        self.assertEqual(physical.pitcher.tolist(), [1])
        self.assertEqual(physical.metric.item(), "velocity_mph")
        self.assertAlmostEqual(physical.score.item(), 1.0)

    def test_retention_is_signed_and_unresolved_is_missing(self):
        rows = pd.DataFrame({"delta": [2.0, -2.0, 2.0, 0.0], "baseline": [10.0] * 4,
                             "future": [11.0, 11.0, np.nan, 10.0],
                             "resolved": [True, True, False, True]})
        out = retention(rows)
        self.assertEqual(out.iloc[0], 0.5)
        self.assertEqual(out.iloc[1], -0.5)
        self.assertTrue(np.isnan(out.iloc[2]))
        self.assertTrue(np.isnan(out.iloc[3]))

    def test_equal_capacity_and_rates_keep_unresolved(self):
        rows = [_row(p, "FF", m, d, future=f, future_outings=o)
                for p, d, f, o in [(1, 3.0, 13.0, 3), (2, 2.0, 10.0, 3), (3, 1.0, np.nan, 0), (4, 0.5, 10.5, 3)]
                for m in ("velocity_mph", "usage")]
        frame = eligible(pd.DataFrame(rows), [2022])
        lists = build_lists(frame, [0.5], [2022])
        top = lists[lists.capacity == 0.5]
        self.assertEqual(top.groupby("method").size().tolist(), [2, 2, 2])
        summary = summarize(lists, [0.5], draws=50, seed=1)
        velo = summary[(summary.capacity == 1.0) & (summary.method == "velocity_only")].iloc[0]
        self.assertEqual((velo.flagged, velo.resolved, velo.persisted), (4, 3, 2))
        self.assertAlmostEqual(velo.rate_flagged, 0.5)

    def test_insufficient_candidates_fail_loudly(self):
        frame = eligible(pd.DataFrame([_row(1, "FF", "velocity_mph", 1.0)]), [2022])
        frame.loc[len(frame)] = frame.iloc[0]
        frame.loc[len(frame) - 1, "pitcher"] = 2
        frame.loc[len(frame) - 1, "metric"] = "spin"
        with self.assertRaises(ValueError):
            build_lists(frame, [0.9], [2022])


if __name__ == "__main__":
    unittest.main()
