import unittest

import numpy as np
import pandas as pd
from scipy import stats

from pcf.calibration import (compare_intervals, ratio_variance, sample_known,
                             simulate_context, summarize, welch_interval)


class CalibrationTests(unittest.TestCase):
    def test_ratio_variance_uses_pitch_weighted_outing_residuals(self):
        mean, variance, outings = ratio_variance(np.array([2.0, 12]), np.array([1.0, 3]))
        self.assertEqual(mean, 3.5)
        self.assertEqual(variance, 0.5625)
        self.assertEqual(outings, 2)

    def test_ratio_variance_ignores_nonfinite_and_nonpositive_count_rows(self):
        sums = np.array([2.0, 12, np.nan, np.inf, 7, 8, 9, 10])
        counts = np.array([1.0, 3, 1, 1, 0, -1, np.nan, np.inf])
        mean, variance, outings = ratio_variance(sums, counts)
        self.assertEqual(mean, 3.5)
        self.assertEqual(variance, 0.5625)
        self.assertEqual(outings, 2)

    def test_ratio_variance_does_not_estimate_noise_from_one_outing(self):
        mean, variance, outings = ratio_variance(np.array([8.0]), np.array([2.0]))
        self.assertEqual(mean, 4)
        self.assertTrue(np.isnan(variance))
        self.assertEqual(outings, 1)
        mean, variance, outings = ratio_variance(np.array([8.0]), np.array([0.0]))
        self.assertTrue(np.isnan(mean))
        self.assertTrue(np.isnan(variance))
        self.assertEqual(outings, 0)

    def test_ratio_variance_respects_affine_measurement_changes(self):
        sums, counts = np.array([2.0, 12, 20]), np.array([1.0, 3, 4])
        mean, variance, outings = ratio_variance(sums, counts)
        transformed = ratio_variance(2.5 * sums + 7 * counts, counts)
        self.assertAlmostEqual(transformed[0], 2.5 * mean + 7)
        self.assertAlmostEqual(transformed[1], 2.5 ** 2 * variance)
        self.assertEqual(transformed[2], outings)

    def test_noninteger_constant_outing_ratios_have_exactly_zero_variance(self):
        total, count = 7300.065656855437, 74.0
        for n_outings in [2, 10]:
            with self.subTest(n_outings=n_outings):
                mean, variance, outings = ratio_variance(np.full(n_outings, total),
                                                         np.full(n_outings, count))
                self.assertEqual(mean, total / count)
                self.assertEqual(variance, 0)
                self.assertEqual(outings, n_outings)
        low, high, reason = welch_interval(np.full(10, total), np.full(10, count),
                                           np.full(2, total), np.full(2, count))
        self.assertEqual(reason, "zero_estimated_variance")
        self.assertTrue(np.isnan(low))
        self.assertTrue(np.isnan(high))

    def test_small_real_outing_variation_is_not_rounded_to_constant(self):
        means = np.array([90 - 1e-8, 90 + 1e-8, 90 - 1e-8, 90 + 1e-8])
        counts = np.array([1.0, 10, 37, 74])
        mean, variance, outings = ratio_variance(means * counts, counts)
        self.assertTrue(np.isfinite(mean))
        self.assertGreater(variance, 0)
        self.assertEqual(outings, 4)
        low, high, reason = welch_interval(means * counts, counts, means * counts, counts)
        self.assertEqual(reason, "ok")
        self.assertLess(low, 0)
        self.assertGreater(high, 0)

    def test_equal_pitch_counts_match_scipy_welch_interval(self):
        recent = np.array([90.0, 91, 96, 92, 99])
        baseline = np.array([88.0, 90, 90, 92])
        expected = stats.ttest_ind(recent, baseline, equal_var=False).confidence_interval(
            confidence_level=0.95)
        low, high, reason = welch_interval(recent * 20, np.full(5, 20.0),
                                           baseline * 20, np.full(4, 20.0))
        self.assertEqual(reason, "ok")
        self.assertAlmostEqual(low, expected.low)
        self.assertAlmostEqual(high, expected.high)

    def test_welch_interval_is_affine_equivariant(self):
        recent_sums, recent_counts = np.array([3.0, 10, 21]), np.array([1.0, 2, 3])
        baseline_sums, baseline_counts = np.array([4.0, 12, 30, 20]), np.array([2.0, 3, 5, 2])
        low, high, reason = welch_interval(recent_sums, recent_counts, baseline_sums, baseline_counts)
        shifted = welch_interval(2.5 * recent_sums + 4 * recent_counts, recent_counts,
                                  2.5 * baseline_sums - baseline_counts, baseline_counts)
        self.assertEqual(reason, "ok")
        self.assertEqual(shifted[2], "ok")
        self.assertAlmostEqual(shifted[0], 2.5 * low + 5)
        self.assertAlmostEqual(shifted[1], 2.5 * high + 5)

    def test_one_constant_window_still_has_variance_from_the_other_window(self):
        recent, baseline = np.array([94.0, 94]), np.array([88.0, 90, 94])
        expected = stats.t.interval(0.95, df=2, loc=recent.mean() - baseline.mean(),
                                    scale=stats.sem(baseline))
        low, high, reason = welch_interval(recent, np.ones(2), baseline, np.ones(3))
        self.assertEqual(reason, "ok")
        self.assertAlmostEqual(low, expected[0])
        self.assertAlmostEqual(high, expected[1])

    def test_both_methods_retain_sparse_comparisons_as_unavailable(self):
        result = compare_intervals(np.array([90.0]), np.ones(1),
                                   np.array([90.0, 92]), np.ones(2),
                                   np.random.default_rng(42), draws=100)
        self.assertEqual(set(result), {"percentile", "welch_t"})
        for low, high, reason in result.values():
            self.assertEqual(reason, "fewer_than_two_outings")
            self.assertTrue(np.isnan(low))
            self.assertTrue(np.isnan(high))

    def test_zero_variance_is_unavailable_for_welch_and_recorded_for_percentile(self):
        result = compare_intervals(np.array([950.0, 950]), np.full(2, 10.0),
                                   np.array([900.0, 900]), np.full(2, 10.0),
                                   np.random.default_rng(42), draws=100)
        low, high, reason = result["welch_t"]
        self.assertEqual(reason, "zero_estimated_variance")
        self.assertTrue(np.isnan(low))
        self.assertTrue(np.isnan(high))
        self.assertEqual(result["percentile"], (5, 5, "ok"))

    def test_usage_at_either_boundary_has_no_interval_for_either_method(self):
        counts = np.full(2, 10.0)
        mixed = np.array([2.0, 8])
        for boundary in [np.zeros(2), counts]:
            for recent, baseline in [(boundary, mixed), (mixed, boundary)]:
                result = compare_intervals(recent, counts, baseline, counts,
                                           np.random.default_rng(42), draws=100, usage=True)
                for low, high, reason in result.values():
                    self.assertEqual(reason, "usage_boundary")
                    self.assertTrue(np.isnan(low))
                    self.assertTrue(np.isnan(high))

    def test_interval_comparison_repeats_with_a_fixed_seed(self):
        args = (np.array([88.0, 90, 92, 94]), np.ones(4),
                np.array([84.0, 88, 92, 96]), np.ones(4))
        first = compare_intervals(*args, np.random.default_rng(42), draws=100)
        second = compare_intervals(*args, np.random.default_rng(42), draws=100)
        self.assertEqual(first, second)

    def test_known_shift_keeps_informative_counts_fixed_before_shifting(self):
        for distribution in ["normal", "lognormal", "two_point"]:
            with self.subTest(distribution=distribution):
                sums, counts = sample_known(distribution, "informative", 128,
                                            np.random.default_rng(42))
                shifted_sums, shifted_counts = sample_known(distribution, "informative", 128,
                                                            np.random.default_rng(42), shift=0.75)
                np.testing.assert_array_equal(counts, shifted_counts)
                self.assertEqual(set(counts), {5, 50})
                np.testing.assert_allclose(shifted_sums, sums + 0.75 * counts)
                self.assertAlmostEqual(shifted_sums.sum() / shifted_counts.sum()
                                       - sums.sum() / counts.sum(), 0.75)

    def test_fixed_count_design_uses_twenty_pitches_per_outing(self):
        for distribution in ["normal", "lognormal", "two_point"]:
            with self.subTest(distribution=distribution):
                sums, counts = sample_known(distribution, "fixed", 32, np.random.default_rng(42))
                self.assertEqual(len(sums), 32)
                np.testing.assert_array_equal(counts, np.full(32, 20))

    def test_null_simulation_pairs_methods_on_identical_trials_and_estimates(self):
        result = simulate_context(np.array([88.0, 90, 92, 94]), np.ones(4), 4, 6,
                                  np.random.default_rng(42), trials=20, draws=100)
        self.assertEqual(len(result), 40)
        self.assertEqual(set(result.method), {"percentile", "welch_t"})
        self.assertTrue(result.groupby("trial").size().eq(2).all())
        self.assertTrue(result.groupby("trial").estimate.nunique().eq(1).all())
        self.assertTrue(result.truth.eq(0).all())
        for _, pair in result.groupby("trial"):
            self.assertTrue(pair.common_available.eq(pair.available.all()).all())
        available = result[result.available]
        expected_coverage = available.low.le(0) & available.high.ge(0)
        np.testing.assert_array_equal(available.covered, expected_coverage)
        np.testing.assert_array_equal(available.zero_excluded, ~expected_coverage)
        np.testing.assert_allclose(available.width, available.high - available.low)

    def test_null_simulation_keeps_all_unavailable_sparse_trials(self):
        result = simulate_context(np.array([88.0, 90, 92, 94]), np.ones(4), 1, 4,
                                  np.random.default_rng(42), trials=20, draws=100)
        self.assertEqual(len(result), 40)
        self.assertFalse(result.available.any())
        self.assertFalse(result.common_available.any())
        self.assertFalse(result.zero_excluded.any())
        self.assertTrue(result.reason.eq("fewer_than_two_outings").all())

    def test_constant_pool_distinguishes_degenerate_and_unavailable_methods(self):
        result = simulate_context(np.full(4, 900.0), np.full(4, 10.0), 4, 4,
                                  np.random.default_rng(42), trials=20, draws=100)
        self.assertEqual(len(result), 40)
        percentile = result[result.method == "percentile"]
        welch = result[result.method == "welch_t"]
        self.assertTrue(percentile.available.all())
        self.assertTrue(percentile.width.eq(0).all())
        self.assertTrue(percentile.covered.all())
        self.assertFalse(welch.available.any())
        self.assertTrue(welch.reason.eq("zero_estimated_variance").all())
        self.assertFalse(result.common_available.any())
        self.assertFalse(result.zero_excluded.any())

    def test_noninteger_constant_pool_roundoff_is_not_a_zero_exclusion(self):
        total, count = 7300.065656855437, 74.0
        result = simulate_context(np.full(4, total), np.full(4, count), 10, 2,
                                  np.random.default_rng(42), trials=20, draws=100)
        self.assertFalse(result.zero_excluded.any())
        self.assertTrue(result.numerical_tolerance.gt(0).all())
        percentile = result[result.method == "percentile"]
        self.assertTrue(percentile.available.all())
        self.assertTrue(percentile.covered.all())
        self.assertTrue(percentile.width.le(percentile.numerical_tolerance).all())
        welch = result[result.method == "welch_t"]
        self.assertFalse(welch.available.any())
        self.assertTrue(welch.reason.eq("zero_estimated_variance").all())
        summary = summarize(result, ["method"]).set_index("method")
        self.assertEqual(summary.loc["percentile", "zero_width"], 20)
        self.assertEqual(summary.loc["percentile", "exclusion_rate_all"], 0)

    def test_null_simulation_is_reproducible(self):
        args = (np.array([88.0, 90, 92, 94]), np.ones(4), 4, 6)
        first = simulate_context(*args, np.random.default_rng(42), trials=20, draws=100)
        second = simulate_context(*args, np.random.default_rng(42), trials=20, draws=100)
        pd.testing.assert_frame_equal(first, second)

    def test_empty_pool_retains_every_trial_without_claiming_coverage(self):
        result = simulate_context(np.array([np.nan]), np.ones(1), 4, 4,
                                  np.random.default_rng(42), trials=20, draws=100)
        self.assertEqual(len(result), 40)
        self.assertFalse(result.available.any())
        self.assertFalse(result.covered.any())
        self.assertFalse(result.zero_excluded.any())
        self.assertTrue(result.estimate.isna().all())

    def test_summary_separates_all_trial_available_and_common_denominators(self):
        rows = []
        intervals = {
            "percentile": [(-1, 1), (1, 1), (-0.5, 0.5), (np.nan, np.nan)],
            "welch_t": [(-1, 1), (-2, 2), (np.nan, np.nan), (np.nan, np.nan)],
        }
        for method, bounds in intervals.items():
            for trial, (low, high) in enumerate(bounds):
                rows.append({
                    "method": method, "trial": trial, "available": np.isfinite([low, high]).all(),
                    "common_available": trial < 2, "covered": low <= 0 <= high,
                    "zero_excluded": low > 0 or high < 0, "width": high - low,
                })
        result = summarize(pd.DataFrame(rows), ["method"]).set_index("method")
        percentile, welch = result.loc["percentile"], result.loc["welch_t"]
        self.assertEqual(percentile.trials, 4)
        self.assertEqual(percentile.available, 3)
        self.assertEqual(percentile.unsupported, 1)
        self.assertEqual(percentile.zero_width, 1)
        self.assertAlmostEqual(percentile.coverage_available, 2 / 3)
        self.assertEqual(percentile.covered_and_available_all, 0.5)
        self.assertEqual(percentile.exclusion_rate_all, 0.25)
        self.assertAlmostEqual(percentile.exclusion_rate_available, 1 / 3)
        self.assertEqual(percentile.common_trials, 2)
        self.assertEqual(percentile.common_coverage, 0.5)
        self.assertEqual(percentile.common_exclusion_rate, 0.5)
        self.assertEqual(percentile.width_median, 1)
        self.assertEqual(percentile.common_width_median, 1)
        self.assertEqual(welch.trials, 4)
        self.assertEqual(welch.available, 2)
        self.assertEqual(welch.unsupported, 2)
        self.assertEqual(welch.coverage_available, 1)
        self.assertEqual(welch.covered_and_available_all, 0.5)
        self.assertEqual(welch.common_trials, 2)
        self.assertEqual(welch.common_coverage, 1)
        self.assertEqual(welch.common_width_median, 3)

    def test_summary_does_not_count_unavailable_intervals_as_successful_coverage(self):
        trials = simulate_context(np.array([88.0, 90, 92]), np.ones(3), 1, 4,
                                  np.random.default_rng(42), trials=20, draws=100)
        result = summarize(trials, ["method"])
        self.assertTrue(result.trials.eq(20).all())
        self.assertTrue(result.unsupported.eq(20).all())
        self.assertTrue(result.availability.eq(0).all())
        self.assertTrue(result.covered_and_available_all.eq(0).all())
        self.assertTrue(result.coverage_available.isna().all())
        self.assertTrue(result.common_coverage.isna().all())
        self.assertTrue(result.exclusion_mc_lo.ge(0).all())
        self.assertTrue(result.exclusion_mc_hi.le(1).all())


if __name__ == "__main__":
    unittest.main()
