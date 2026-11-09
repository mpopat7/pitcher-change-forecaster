import unittest

import numpy as np
import pandas as pd

from pcf.forecast_value import directional, fit_priors, per_type_features

DATE = pd.Timestamp("2022-06-01")


def _rows(pitcher, pitch_type, status, usage, physical, season=2022, outings=3):
    """usage = (recent, baseline); physical = {metric: (delta, ci_half_width)}."""
    out = [{"pitcher": pitcher, "season": season, "forecast_date": DATE, "pitch_type": pitch_type,
            "metric": "usage", "type_status": status, "recent": usage[0], "baseline": usage[1],
            "delta": (usage[0] or 0) - (usage[1] or 0), "ci_low": np.nan, "ci_high": np.nan,
            "type_outings_recent": outings, "type_outings_baseline": outings}]
    for metric, (delta, half) in physical.items():
        out.append({"pitcher": pitcher, "season": season, "forecast_date": DATE, "pitch_type": pitch_type,
                    "metric": metric, "type_status": status, "recent": 90 + delta, "baseline": 90.0,
                    "delta": delta, "ci_low": delta - half, "ci_high": delta + half,
                    "type_outings_recent": outings, "type_outings_baseline": outings})
    return out


class ForecastValueTests(unittest.TestCase):
    def test_priors_use_training_seasons_and_fall_back_to_metric_pool(self):
        rows = (_rows(1, "FF", "both", (0.6, 0.6), {"velocity_mph": (2.0, 0.0)})
                + _rows(2, "FF", "both", (0.6, 0.6), {"velocity_mph": (0.0, 0.0)})
                + _rows(3, "SL", "both", (0.3, 0.3), {"velocity_mph": (5.0, 0.0)}, season=2023))
        priors = fit_priors(pd.DataFrame(rows), [2022]).set_index(["metric", "pitch_type"])
        self.assertEqual(priors.loc[("velocity_mph", "FF"), "tau"], 2.0)
        self.assertEqual(priors.loc[("velocity_mph", "FF"), "source"], "pitch_type")
        self.assertEqual(priors.loc[("velocity_mph", "SL"), "source"], "metric_pool")
        self.assertEqual(priors.loc[("velocity_mph", "SL"), "tau"], 2.0)

    def test_features_shrink_pick_primary_fastball_and_summarize_usage(self):
        rows = (_rows(1, "FF", "both", (0.5, 0.4), {"velocity_mph": (1.0, 1.96), "ivb_in": (2.0, 1.96),
                                                     "hb_in": (-1.0, 1.96)})
                + _rows(1, "SI", "both", (0.1, 0.3), {"velocity_mph": (3.0, 1.96)})
                + _rows(1, "SL", "both", (0.2, 0.3), {"velocity_mph": (-2.0, 1.96), "ivb_in": (1.0, 1.96),
                                                     "hb_in": (1.0, 1.96)})
                + _rows(1, "CH", "recent_only", (0.2, None), {}))
        priors = pd.DataFrame([{"metric": m, "pitch_type": t, "tau": 1.0, "source": "pitch_type"}
                               for m in ("velocity_mph", "ivb_in", "hb_in") for t in ("FF", "SI", "SL")])
        f = per_type_features(pd.DataFrame(rows), priors).iloc[0]
        # Noise variance (2 * 1.96 / 3.92)^2 = 1, so shrinkage halves every change.
        self.assertAlmostEqual(f.pt_fb_velo, 0.5)
        self.assertAlmostEqual(f.pt_fb_ivb, 1.0)
        self.assertAlmostEqual(f.pt_fb_hb, -0.5)
        self.assertAlmostEqual(f.pt_other_velo, -1.0)
        self.assertAlmostEqual(f.pt_other_move, 1.0)
        self.assertAlmostEqual(f.pt_usage_shift, 0.5 * (0.1 + 0.2 + 0.1 + 0.2))
        self.assertAlmostEqual(f.pt_new_usage, 0.2)
        self.assertAlmostEqual(f.pt_max_z, 1.5)

    def test_unsupported_types_do_not_create_physical_features(self):
        rows = _rows(1, "FF", "both", (0.5, 0.5), {"velocity_mph": (4.0, 1.96)}, outings=1)
        priors = pd.DataFrame([{"metric": "velocity_mph", "pitch_type": "FF", "tau": 1.0, "source": "pitch_type"}])
        f = per_type_features(pd.DataFrame(rows), priors).iloc[0]
        self.assertTrue(np.isnan(f.pt_fb_velo))
        self.assertTrue(np.isnan(f.pt_max_z))
        self.assertEqual(f.pt_usage_shift, 0.0)

    def test_directional_share_is_bf_weighted_and_skips_unchanged_rows(self):
        share, moved = directional(np.array([0.3, 0.1, 0.2]), np.array([0.2, 0.2, 0.2]),
                                   np.array([3, 1, 5]), np.array([10, 30, 10]))
        self.assertEqual(moved, 2)
        self.assertAlmostEqual(share, 10 / 40 + 30 / 40)


if __name__ == "__main__":
    unittest.main()
