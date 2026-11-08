import unittest

import numpy as np
import pandas as pd

from pcf.changes import build_changes
from pcf.reliability import (build_persistence, independent_splits, measure_noise, select_anchors,
                             shrink_changes, synthetic_null)


def outing(date, game_pk, pitch_type="FF", velocity=94, n=1):
    return {
        "pitcher": 7, "game_pk": game_pk, "game_date": pd.Timestamp(date),
        "season": pd.Timestamp(date).year, "pitch_team": "DET", "pitch_type": pitch_type,
        "n_pitches": n, "velocity_mph_sum": velocity * n, "velocity_mph_count": n,
        "ivb_in_sum": 12 * n, "ivb_in_count": n,
        "hb_in_sum": 6 * n, "hb_in_count": n,
    }


def past_outings():
    return pd.DataFrame([
        outing("2024-04-10", 1, velocity=90), outing("2024-04-20", 2, velocity=90),
        outing("2024-05-10", 3), outing("2024-05-20", 4),
    ])


def features():
    forecasts = pd.DataFrame([{
        "pitcher": 7, "forecast_date": pd.Timestamp("2024-06-03"), "season": 2024,
        "team": "DET", "name": "Example, Pitcher",
    }])
    frame = build_changes(past_outings(), forecasts, n_boot=100)
    return frame.assign(noise_variance=1.0, shrunk_delta=1.0)


def prior_row(pitcher, season, delta, noise=1.0, pitch_type="FF"):
    return {
        "pitcher": pitcher, "season": season, "metric": "velocity_mph",
        "pitch_type": pitch_type, "delta": delta, "noise_variance": noise,
        "recent": 90 + delta, "baseline": 90,
    }


class ReliabilityTests(unittest.TestCase):
    def test_anchor_dates_are_separated_per_pitcher(self):
        rows = []
        for pitcher, dates in [(7, ["2024-04-01", "2024-04-08", "2024-04-29",
                                    "2024-05-01", "2024-05-31"]),
                               (8, ["2024-04-08", "2024-05-01", "2024-06-01"])]:
            rows += [{"pitcher": pitcher, "forecast_date": pd.Timestamp(date)} for date in dates]
        selected = select_anchors(pd.DataFrame(rows).sample(frac=1, random_state=3), gap_days=30)
        kept = selected.groupby("pitcher").forecast_date.apply(
            lambda dates: sorted(dates.dt.strftime("%Y-%m-%d"))).to_dict()
        self.assertEqual(kept[7], ["2024-04-01", "2024-05-01", "2024-05-31"])
        self.assertEqual(kept[8], ["2024-04-08", "2024-06-01"])

    def test_split_halves_partition_both_windows(self):
        current = np.array([1.0, 2.0, 4.0, 8.0])
        baseline = np.array([16.0, 32.0, 64.0, 128.0])
        counts = np.ones(4)
        for seed in range(5):
            split = independent_splits(current, counts, baseline, counts, np.random.default_rng(seed))
            self.assertAlmostEqual(split["delta_a"] + split["delta_b"], -112.5)
            self.assertEqual(split["half_min_pitches"], 2)

    def test_split_means_weight_valid_pitches_and_drop_unmeasured_outings(self):
        counts = np.array([1.0, 3.0, 7.0, 9.0, 0.0])
        split = independent_splits(94 * counts, counts, 90 * counts, counts,
                                   np.random.default_rng(42))
        self.assertAlmostEqual(split["delta_a"], 4)
        self.assertAlmostEqual(split["delta_b"], 4)

    def test_split_requires_four_measured_outings_in_each_window(self):
        split = independent_splits(np.array([90.0, 90, 90, 0]), np.array([1.0, 1, 1, 0]),
                                   np.full(4, 90.0), np.ones(4), np.random.default_rng(7))
        for key in ["delta_a", "delta_b", "half_min_pitches"]:
            self.assertTrue(np.isnan(split[key]))

    def test_constant_synthetic_null_has_no_zero_exclusions(self):
        result = synthetic_null(np.full(5, 900.0), np.full(5, 10.0), 4, 4,
                                np.random.default_rng(11), outer=20, inner=100)
        self.assertEqual(result["trials"], 20)
        self.assertEqual(result["zero_exclusions"], 0)
        self.assertEqual(result["exclusion_rate_all"], 0)

    def test_synthetic_null_records_unsupported_short_windows(self):
        result = synthetic_null(np.array([90.0, 92]), np.ones(2), 1, 4,
                                np.random.default_rng(11), outer=20, inner=100)
        self.assertEqual(result["trials"], 20)
        self.assertEqual(result["unsupported_trials"], 20)
        self.assertEqual(result["intervals"], 0)
        self.assertEqual(result["zero_exclusions"], 0)

    def test_synthetic_null_suppresses_usage_boundary_intervals(self):
        result = synthetic_null(np.zeros(4), np.full(4, 10.0), 4, 4,
                                np.random.default_rng(11), outer=20, inner=100, usage=True)
        self.assertEqual(result["intervals"], 0)
        self.assertEqual(result["unsupported_trials"], 20)
        self.assertEqual(result["zero_exclusions"], 0)

    def test_synthetic_null_resampling_repeats_with_fixed_seed(self):
        args = (np.array([88.0, 90, 92, 94]), np.ones(4), 4, 6)
        first = synthetic_null(*args, np.random.default_rng(42), outer=20, inner=100)
        second = synthetic_null(*args, np.random.default_rng(42), outer=20, inner=100)
        self.assertEqual(first, second)

    def test_shrinkage_uses_only_earlier_seasons_and_two_distinct_pitchers(self):
        frame = pd.DataFrame([prior_row(1, 2021, 2), prior_row(2, 2021, 4),
                              prior_row(3, 2022, 3, noise=3)])
        shrunk, _ = shrink_changes(frame)
        opening = shrunk[shrunk.season == 2021]
        self.assertTrue(opening.prior_variance.isna().all())
        evaluated = shrunk[shrunk.season == 2022].iloc[0]
        self.assertAlmostEqual(evaluated.prior_variance, 9)
        self.assertAlmostEqual(evaluated.shrinkage_weight, 0.75)
        self.assertAlmostEqual(evaluated.shrunk_delta, 2.25)
        self.assertLess(evaluated.prior_train_end, 2022)
        extended = pd.concat([frame, pd.DataFrame([prior_row(4, 2024, 1000)])], ignore_index=True)
        later, _ = shrink_changes(extended)
        columns = ["prior_variance", "shrinkage_weight", "shrunk_delta", "prior_train_end"]
        pd.testing.assert_frame_equal(shrunk[columns], later.loc[:len(frame) - 1, columns])

    def test_shrinkage_falls_back_by_metric_and_excludes_unknown_labels(self):
        frame = pd.DataFrame([prior_row(1, 2021, 2, noise=0),
                              prior_row(2, 2021, 4, noise=0, pitch_type="SL"),
                              prior_row(3, 2021, 1000, noise=0, pitch_type="UN"),
                              prior_row(4, 2022, 3, noise=2)])
        shrunk, _ = shrink_changes(frame)
        evaluated = shrunk[shrunk.season == 2022].iloc[0]
        self.assertAlmostEqual(evaluated.prior_variance, 10)
        self.assertAlmostEqual(evaluated.shrinkage_weight, 10 / 12)
        self.assertTrue(shrunk.loc[shrunk.pitch_type == "UN", "prior_variance"].isna().all())

    def test_zero_signal_variance_shrinks_to_zero_even_with_zero_noise(self):
        frame = pd.DataFrame([prior_row(1, 2021, 1, noise=2),
                              prior_row(2, 2021, -1, noise=2),
                              prior_row(3, 2022, 5, noise=0)])
        shrunk, _ = shrink_changes(frame)
        evaluated = shrunk[shrunk.season == 2022].iloc[0]
        self.assertEqual(evaluated.prior_variance, 0)
        self.assertEqual(evaluated.shrinkage_weight, 0)
        self.assertEqual(evaluated.shrunk_delta, 0)

    def test_repeated_rows_from_one_pitcher_do_not_supply_a_prior(self):
        frame = pd.DataFrame([prior_row(1, 2021, 2), prior_row(1, 2021, 4),
                              prior_row(3, 2022, 3)])
        shrunk, _ = shrink_changes(frame)
        self.assertTrue(shrunk.loc[shrunk.season == 2022, "shrunk_delta"].isna().all())

    def test_invalid_training_noise_is_excluded_and_weights_reduce_deltas(self):
        frame = pd.DataFrame([
            prior_row(1, 2021, 2), prior_row(2, 2021, 4),
            prior_row(3, 2021, 1000, noise=-1), prior_row(4, 2021, 1000, noise=np.nan),
            prior_row(5, 2022, 3, noise=0), prior_row(6, 2022, -3, noise=100),
        ])
        shrunk, _ = shrink_changes(frame)
        evaluated = shrunk[shrunk.season == 2022]
        np.testing.assert_allclose(evaluated.prior_variance, 9)
        self.assertTrue(evaluated.shrinkage_weight.between(0, 1).all())
        self.assertTrue((evaluated.shrunk_delta.abs() <= evaluated.delta.abs()).all())

    def test_persistence_uses_cutoff_through_day_29_and_compares_equal_rows(self):
        future = pd.DataFrame([
            outing("2024-06-02", 5, velocity=200), outing("2024-06-03", 6, velocity=92),
            outing("2024-07-02", 7, velocity=96), outing("2024-07-03", 8, velocity=200),
        ])
        table = build_persistence(pd.concat([past_outings(), future], ignore_index=True), features())
        value = table[table.metric == "velocity_mph"].iloc[0]
        self.assertEqual(value.future, 94)
        self.assertEqual(value.future_n, 2)
        self.assertEqual(value.future_outings, 2)
        self.assertTrue(value.paired)
        self.assertEqual(value.baseline_sq_error, 16)
        self.assertEqual(value.recent_sq_error, 0)
        self.assertEqual(value.shrunk_sq_error, 9)

    def test_no_future_appearance_is_missing_for_usage_and_physical_metrics(self):
        table = build_persistence(past_outings(), features())
        self.assertTrue(table.future.isna().all())
        self.assertFalse(table.paired.any())

    def test_future_absent_type_has_zero_usage_without_adding_future_only_types(self):
        future = pd.DataFrame([
            outing("2024-06-03", 5, pitch_type="CH", velocity=85),
            outing("2024-06-10", 6, pitch_type="CH", velocity=85),
        ])
        table = build_persistence(pd.concat([past_outings(), future], ignore_index=True), features())
        self.assertEqual(set(table.pitch_type), {"FF"})
        self.assertEqual(len(table), len(features()))
        usage = table[table.metric == "usage"].iloc[0]
        self.assertEqual(usage.future, 0)
        self.assertEqual(usage.future_n, 2)
        self.assertEqual(usage.future_outings, 2)
        physical = table[table.metric != "usage"]
        self.assertTrue(physical.future.isna().all())
        self.assertFalse(physical.paired.any())

    def test_one_future_outing_is_not_a_paired_comparison(self):
        future = pd.DataFrame([outing("2024-06-03", 5, velocity=92, n=100)])
        table = build_persistence(pd.concat([past_outings(), future], ignore_index=True), features())
        value = table[table.metric == "velocity_mph"].iloc[0]
        self.assertEqual(value.future, 92)
        self.assertEqual(value.future_n, 100)
        self.assertFalse(value.paired)

    def test_missing_prior_excludes_all_models_from_the_paired_comparison(self):
        future = pd.DataFrame([
            outing("2024-06-03", 5, velocity=92), outing("2024-06-10", 6, velocity=94),
        ])
        unsupported = features().assign(shrunk_delta=np.nan)
        table = build_persistence(pd.concat([past_outings(), future], ignore_index=True), unsupported)
        self.assertFalse(table.paired.any())
        for model in ["baseline", "recent", "shrunk"]:
            self.assertTrue(table[f"{model}_sq_error"].isna().all())

    def test_noise_and_persistence_reject_2025_features(self):
        heldout = features().assign(season=2025, forecast_date=pd.Timestamp("2025-06-03"))
        for function in [measure_noise, build_persistence]:
            with self.subTest(function=function.__name__):
                with self.assertRaisesRegex(ValueError, "2021-2024"):
                    function(past_outings(), heldout)

    def test_date_year_cannot_bypass_the_development_season_guard(self):
        mislabeled = features().assign(forecast_date=pd.Timestamp("2025-06-03"))
        for function in [measure_noise, build_persistence]:
            with self.subTest(function=function.__name__):
                with self.assertRaises(ValueError):
                    function(past_outings(), mislabeled)


if __name__ == "__main__":
    unittest.main()
