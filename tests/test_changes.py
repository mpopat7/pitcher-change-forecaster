import unittest

import numpy as np
import pandas as pd

from pcf.changes import aggregate_outings, build_changes, eligible_forecasts


def game(date, game_pk, pitches, **overrides):
    rows = []
    for number, (pitch_type, speed) in enumerate(pitches, 1):
        row = {
            "game_type": "R", "game_pk": game_pk, "game_date": date,
            "at_bat_number": number, "pitch_number": 1, "pitcher": 7,
            "pitch_type": pitch_type, "player_name": "Example, Pitcher",
            "p_throws": "R", "pfx_x": -0.5, "pfx_z": 1.0,
            "release_speed": speed, "inning_topbot": "Top",
            "home_team": "DET", "away_team": "CIN", "events": "field_out",
        }
        row.update(overrides)
        rows.append(row)
    return rows


def forecast():
    return pd.DataFrame([{
        "pitcher": 7, "forecast_date": pd.Timestamp("2024-06-03"),
        "season": 2024, "team": "DET", "name": "Example, Pitcher",
    }])


def changes(rows, forecasts=None, **kwargs):
    return build_changes(aggregate_outings(pd.DataFrame(rows)),
                         forecast() if forecasts is None else forecasts,
                         n_boot=100, **kwargs)


def measurement(table, pitch_type, metric):
    rows = table[(table.pitch_type == pitch_type) & (table.metric == metric)]
    if len(rows) != 1:
        raise AssertionError(f"Expected one {pitch_type}/{metric} row; got {len(rows)}")
    return rows.iloc[0]


def calendar_rows():
    return (game("2024-03-20", 1000, [("FF", 90)], pitcher=999, events=None)
            + game("2024-07-31", 1001, [("FF", 90)], pitcher=999, events=None))


def june_forecasts(rows):
    table = eligible_forecasts(pd.DataFrame(rows))
    return table[table.forecast_date == pd.Timestamp("2024-06-03")].reset_index(drop=True)


class ChangesTests(unittest.TestCase):
    def test_windows_exclude_cutoff_and_keep_exact_boundaries(self):
        rows = []
        for number, (date, speed) in enumerate([
            ("2024-02-03", 20), ("2024-02-04", 90), ("2024-05-03", 92),
            ("2024-05-04", 94), ("2024-06-02", 96), ("2024-06-03", 120),
        ], 1):
            rows += game(date, number, [("FF", speed)])
        value = measurement(changes(rows), "FF", "velocity_mph")
        self.assertEqual(value.baseline_source, "earlier_window")
        self.assertEqual(value.baseline, 91)
        self.assertEqual(value.recent, 95)
        self.assertEqual(value.delta, 4)
        self.assertEqual(value.n_recent, 2)
        self.assertEqual(value.n_baseline, 2)

    def test_later_pitches_and_future_outcomes_cannot_change_measurements(self):
        rows = (game("2024-04-10", 1, [("FF", 90)])
                + game("2024-04-20", 2, [("FF", 92)])
                + game("2024-05-10", 3, [("FF", 94)])
                + game("2024-05-20", 4, [("FF", 96)]))
        before = changes(rows, seed=19)
        later = (game("2024-06-03", 5, [("ST", 120)])
                 + game("2024-08-10", 6, [("FF", 10)])
                 + game("2025-04-10", 7, [("FF", 150)]))
        future = forecast().assign(y_bf=0, y_k=999, y_bb=999)
        after = changes(rows + later, future, seed=19)
        pd.testing.assert_frame_equal(before, after)

    def test_sparse_earlier_window_uses_prior_season(self):
        rows = (game("2023-07-10", 1, [("FF", 88)])
                + game("2023-07-20", 2, [("FF", 92)])
                + game("2024-04-20", 3, [("FF", 50)])
                + game("2024-05-10", 4, [("FF", 94)])
                + game("2024-05-20", 5, [("FF", 96)]))
        value = measurement(changes(rows), "FF", "velocity_mph")
        self.assertEqual(value.baseline_source, "prior_season")
        self.assertEqual(value.baseline, 90)
        self.assertEqual(value.recent, 95)
        self.assertEqual(value.n_baseline, 2)

    def test_usage_counts_unknown_types_and_missing_velocity(self):
        rows = []
        for number, date in enumerate(["2024-04-10", "2024-04-20"], 1):
            rows += game(date, number, [("FF", 90), ("FF", np.nan), (None, np.nan)])
        for number, date in enumerate(["2024-05-10", "2024-05-20"], 3):
            rows += game(date, number, [("FF", 94), ("ST", 85), (None, np.nan)])
        table = changes(rows)
        usage = measurement(table, "FF", "usage")
        self.assertAlmostEqual(usage.baseline, 2 / 3)
        self.assertAlmostEqual(usage.recent, 1 / 3)
        self.assertEqual(usage.n_baseline, 6)
        self.assertEqual(usage.n_recent, 6)
        self.assertEqual(usage.outings_recent, 2)
        unknown = measurement(table, "UN", "usage")
        self.assertAlmostEqual(unknown.recent, 1 / 3)
        velocity = measurement(table, "FF", "velocity_mph")
        self.assertEqual(velocity.n_baseline, 2)
        self.assertEqual(velocity.baseline, 90)
        self.assertEqual(velocity.recent, 94)

    def test_absent_type_has_zero_usage_and_missing_physical_measurements(self):
        rows = []
        for number, date in enumerate(["2024-04-10", "2024-04-20"], 1):
            rows += game(date, number, [("FF", 90), ("SL", 84)])
        for number, date in enumerate(["2024-05-10", "2024-05-20"], 3):
            rows += game(date, number, [("FF", 94), ("CH", 85)])
        table = changes(rows)
        self.assertEqual(set(table.pitch_type), {"FF", "SL", "CH"})
        added = measurement(table, "CH", "usage")
        self.assertEqual(added.type_status, "recent_only")
        self.assertEqual(added.baseline, 0)
        self.assertEqual(added.recent, 0.5)
        removed = measurement(table, "SL", "usage")
        self.assertEqual(removed.type_status, "baseline_only")
        self.assertEqual(removed.recent, 0)
        self.assertEqual(removed.delta, -0.5)
        for pitch_type in ["CH", "SL"]:
            physical = measurement(table, pitch_type, "velocity_mph")
            self.assertTrue(pd.isna(physical.delta))
            self.assertTrue(pd.isna(physical.ci_low))
            self.assertTrue(pd.isna(physical.ci_high))

    def test_one_outing_does_not_produce_an_interval(self):
        rows = (game("2024-04-10", 1, [("FF", 90)] * 100)
                + game("2024-04-20", 2, [("FF", 92)] * 100)
                + game("2024-05-10", 3, [("FF", 96)] * 100))
        table = changes(rows)
        for metric in ["velocity_mph", "ivb_in", "hb_in", "usage"]:
            value = measurement(table, "FF", metric)
            self.assertEqual(value.outings_recent, 1)
            self.assertTrue(pd.isna(value.ci_low))
            self.assertTrue(pd.isna(value.ci_high))

    def test_missing_metric_does_not_count_as_a_measured_outing(self):
        rows = (game("2024-04-10", 1, [("FF", 90)])
                + game("2024-04-20", 2, [("FF", 92)])
                + game("2024-05-10", 3, [("FF", 94)])
                + game("2024-05-20", 4, [("FF", np.nan)]))
        table = changes(rows)
        velocity = measurement(table, "FF", "velocity_mph")
        self.assertEqual(velocity.recent, 94)
        self.assertEqual(velocity.n_recent, 1)
        self.assertEqual(velocity.outings_recent, 1)
        self.assertTrue(pd.isna(velocity.ci_low))
        usage = measurement(table, "FF", "usage")
        self.assertEqual(usage.n_recent, 2)
        self.assertEqual(usage.outings_recent, 2)
        self.assertEqual(usage.recent, 1)

    def test_movement_uses_inches_and_arm_side_positive_for_either_hand(self):
        rows = (game("2024-05-10", 1, [("FF", 90)], p_throws="R", pfx_x=-0.5)
                + game("2024-05-11", 2, [("FF", 90)], p_throws="L", pfx_x=0.5))
        outings = aggregate_outings(pd.DataFrame(rows))
        np.testing.assert_allclose(outings.hb_in_sum / outings.hb_in_count, 6)
        np.testing.assert_allclose(outings.ivb_in_sum / outings.ivb_in_count, 12)

    def test_means_weight_pitches_instead_of_weighting_outings_equally(self):
        rows = (game("2024-04-10", 1, [("FF", 90)] * 9)
                + game("2024-04-20", 2, [("FF", 100)])
                + game("2024-05-10", 3, [("FF", 95)] * 9)
                + game("2024-05-20", 4, [("FF", 105)]))
        value = measurement(changes(rows), "FF", "velocity_mph")
        self.assertEqual(value.baseline, 91)
        self.assertEqual(value.recent, 96)
        self.assertEqual(value.delta, 5)

    def test_resampling_is_deterministic_for_a_fixed_seed(self):
        rows = []
        for number, (date, speed) in enumerate([
            ("2024-04-10", 88), ("2024-04-20", 92),
            ("2024-05-10", 94), ("2024-05-20", 98),
        ], 1):
            rows += game(date, number, [("FF", speed)])
        first = changes(rows, seed=42)
        second = changes(rows, seed=42)
        pd.testing.assert_frame_equal(first, second)
        value = measurement(first, "FF", "velocity_mph")
        self.assertTrue(np.isfinite(value.ci_low))
        self.assertTrue(np.isfinite(value.ci_high))
        self.assertLess(value.ci_low, value.ci_high)

    def test_eligibility_uses_past_bf_window_and_last_seen(self):
        rows = calendar_rows()
        rows += game("2024-04-04", 1, [("FF", 90)] * 49, pitcher=7)
        rows += game("2024-05-20", 2, [("FF", 90)], pitcher=7)
        rows += game("2024-04-03", 3, [("FF", 90)] * 49, pitcher=8)
        rows += game("2024-05-20", 4, [("FF", 90)], pitcher=8)
        rows += game("2024-05-19", 5, [("FF", 90)] * 50, pitcher=9)
        rows += game("2024-05-20", 6, [("FF", 90)] * 50, pitcher=10)
        self.assertEqual(set(june_forecasts(rows).pitcher), {7, 10})

    def test_cutoff_pa_intentional_walks_and_truncated_pas_do_not_add_past_bf(self):
        rows = calendar_rows()
        rows += game("2024-05-20", 1, [("FF", 90)] * 49, pitcher=7)
        rows += game("2024-06-03", 2, [("FF", 90)], pitcher=7)
        rows += game("2024-05-20", 3, [("FF", 90)] * 48, pitcher=8)
        rows += game("2024-05-20", 4, [("FF", 90)], pitcher=8, events="intent_walk")
        rows += game("2024-05-20", 5, [("FF", 90)], pitcher=8, events="truncated_pa")
        rows += game("2024-05-20", 6, [("FF", 90)] * 49, pitcher=9)
        rows += game("2024-05-20", 7, [("FF", 90)], pitcher=9, events="walk")
        self.assertEqual(set(june_forecasts(rows).pitcher), {9})

    def test_eligibility_does_not_require_future_workload_or_read_outcome_columns(self):
        rows = calendar_rows() + game("2024-05-20", 1, [("FF", 90)] * 50, pitcher=7)
        before = june_forecasts(rows)
        self.assertEqual(before.pitcher.tolist(), [7])
        future = (game("2024-06-03", 2, [("FF", 90)] * 100, pitcher=8)
                  + game("2024-06-20", 3, [("FF", 90)] * 100, pitcher=8))
        altered = pd.DataFrame(rows + future).assign(y_bf=999, y_k=999, y_bb=999)
        table = eligible_forecasts(altered)
        after = table[table.forecast_date == pd.Timestamp("2024-06-03")].reset_index(drop=True)
        pd.testing.assert_frame_equal(before, after)

    def test_revised_experiment_rejects_2025_forecasts(self):
        rows = game("2024-05-20", 1, [("FF", 90)])
        heldout = forecast().assign(season=2025, forecast_date=pd.Timestamp("2025-06-03"))
        with self.assertRaisesRegex(ValueError, "development seasons"):
            changes(rows, heldout)


if __name__ == "__main__":
    unittest.main()
