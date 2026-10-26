import unittest

import numpy as np
import pandas as pd

from pcf.model import USAGE, RepertoireSupport, fallback_predictions


class RepertoireTests(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame([
            [0.20, 0.10, 0.50, 0.20, 0.00],
            [0.65, 0.00, 0.20, 0.14, 0.01],
            [0.50, 0.05, 0.30, 0.125, 0.025],
        ], columns=USAGE, index=[12, 45, 80])
        self.support = RepertoireSupport().fit(self.train)

    def test_training_boundaries_are_supported(self):
        self.assertFalse(self.support.flags(self.train).to_numpy().any())

    def test_rare_group_is_flagged(self):
        waldron = self.train.iloc[[0]].copy()
        waldron.loc[12, USAGE] = [0.20, 0.10, 0.20, 0.15, 0.35]
        flags = self.support.flags(waldron)
        self.assertEqual(flags.loc[12, flags.loc[12]].index.tolist(), ["use_other_cur"])

    def test_both_sides_of_range_and_missing_usage_are_flagged(self):
        rows = self.train.copy()
        rows.loc[12, "use_fb_cur"] = 0.19
        rows.loc[45, "use_fb_cur"] = 0.66
        rows.loc[80, "use_off_cur"] = np.nan
        self.assertEqual(self.support.flags(rows).any(axis=1).tolist(), [True, True, True])

    def test_no_training_measurements_means_no_support(self):
        self.train["use_other_cur"] = np.nan
        support = RepertoireSupport().fit(self.train)
        self.assertTrue(support.flags(self.train)["use_other_cur"].all())

    def test_evaluating_future_rows_does_not_expand_training_bounds(self):
        future = self.train.copy()
        future["use_other_cur"] = 0.40
        self.support.flags(future)
        self.assertEqual(self.support.bounds()["use_other_cur"]["max"], 0.025)

    def test_fallback_keeps_every_row_and_only_replaces_flagged_predictions(self):
        pitch_model = np.array([0.01, 0.30, 0.32])
        results_model = np.array([0.25, 0.24, 0.22])
        flags = np.array([True, False, True])
        np.testing.assert_array_equal(
            fallback_predictions(pitch_model, results_model, flags), [0.25, 0.30, 0.22])


if __name__ == "__main__":
    unittest.main()
