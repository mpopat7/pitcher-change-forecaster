import unittest

import numpy as np
import pandas as pd

from pcf.watchlist import RULES, evaluate_alerts, select_watchlist


class WatchlistTests(unittest.TestCase):
    def forecasts(self):
        return pd.DataFrame({
            "pitcher": [1, 1, 1, 1, 2, 3, 4, 5],
            "forecast_date": pd.to_datetime([
                "2024-04-01", "2024-04-08", "2024-04-29", "2024-05-06",
                "2024-04-01", "2024-04-01", "2024-04-01", "2024-04-01"]),
            "p_k_A": [0.20] * 8, "p_bb_A": [0.08] * 8,
            "p_k_B": [0.25] * 8, "p_bb_B": [0.08] * 8,
            "p_k_C": [0.27, 0.28, 0.28, 0.28, 0.23, 0.28, 0.28, 0.28],
            "p_bb_C": [0.08] * 8,
            "repertoire_fallback": [False] * 7 + [True],
            "n_pitches_cur": [150] * 5 + [149, 200, 200],
            "chg_base_pitches": [150] * 6 + [np.nan, 200],
            "bf_r30": [100] * 8, "k_r30": [25] * 8, "bb_r30": [8] * 8,
            "bf_ps": [200] * 8, "k_ps": [50] * 8, "bb_ps": [16] * 8,
            "y_bf": [0, 100, 100, 49, 100, 100, 100, 100],
            "y_k": [0, 30, 30, 15, 20, 30, 30, 30],
            "y_bb": [0, 8, 8, 3, 8, 8, 8, 8],
        }, index=range(10, 18))

    def test_selection_ignores_all_future_outcomes(self):
        rows = self.forecasts()
        before = select_watchlist(rows, RULES)
        rows[["y_bf", "y_k", "y_bb"]] = 999
        after = select_watchlist(rows, RULES)
        for col in ("alert_candidate", "alert", "performance_watchlist", "performance_rank",
                    "recent_results_status"):
            pd.testing.assert_series_equal(before[col], after[col])
        self.assertTrue(before.loc[10, "alert"])

    def test_cooldown_is_per_pitcher_and_covers_the_outcome_window(self):
        selected = select_watchlist(self.forecasts(), RULES)
        self.assertEqual(selected.alert.tolist(), [True, False, False, True, True, False, False, False])

    def test_small_or_missing_past_samples_and_fallback_do_not_alert(self):
        selected = select_watchlist(self.forecasts(), RULES)
        self.assertFalse(selected.loc[[15, 16, 17], "alert_candidate"].any())

    def test_resolution_includes_unresolved_alerts_in_the_denominator(self):
        alerts, summary = evaluate_alerts(select_watchlist(self.forecasts(), RULES), RULES)
        self.assertEqual(alerts.resolution.tolist(), ["unresolved", "unresolved", "confirmed"])
        self.assertEqual(summary[0]["alerts"], 3)
        self.assertEqual(summary[0]["unresolved"], 2)
        self.assertAlmostEqual(summary[0]["unconfirmed_fraction_all"], 2 / 3)

    def test_false_alert_requires_a_resolved_outcome(self):
        rows = self.forecasts()
        rows.loc[14, "y_k"] = 24
        alerts, summary = evaluate_alerts(select_watchlist(rows, RULES), RULES)
        self.assertEqual(alerts.loc[14, "resolution"], "false_alert")
        self.assertEqual(summary[0]["false_fraction_resolved"], 1)

    def test_ranking_uses_forecasts_with_deterministic_ties(self):
        rules = {**RULES, "ranking_size": 1}
        selected = select_watchlist(self.forecasts().sample(frac=1, random_state=1), rules)
        first = selected[selected.forecast_date == "2024-04-01"]
        self.assertEqual(first.loc[first.performance_watchlist, "pitcher"].tolist(), [1])

    def test_recent_results_comparison_requires_past_exposure(self):
        rows = self.forecasts()
        rows.loc[10, "k_r30"] = 30
        rows.loc[11, "bf_ps"] = 0
        selected = select_watchlist(rows, RULES)
        self.assertEqual(selected.loc[10, "recent_results_status"], "already_evident")
        self.assertEqual(selected.loc[11, "recent_results_status"], "insufficient_past_results")
        self.assertEqual(selected.loc[12, "recent_results_status"], "not_yet_evident")

    def test_no_alerts_produce_counts_and_no_rate_estimate(self):
        rows = self.forecasts()
        rows["p_k_C"] = rows.p_k_B
        alerts, summary = evaluate_alerts(select_watchlist(rows, RULES), RULES)
        self.assertTrue(alerts.empty)
        self.assertEqual(summary[0]["alerts"], 0)
        self.assertIsNone(summary[0]["false_fraction_resolved"])


if __name__ == "__main__":
    unittest.main()
