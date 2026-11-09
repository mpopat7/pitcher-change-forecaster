import unittest

from pcf.evaluate_2026 import ZONE, model_columns
from pcf.forecast_value import PT_FEATURES
from pcf.model import RESULTS_A


class Evaluation2026Tests(unittest.TestCase):
    def test_models_match_development_and_zone_sensitivity_drops_only_zone(self):
        columns = RESULTS_A + ["use_fb_cur", "zone_rate_cur", "chg_fb_velo", "chg_zone_rate"]
        m = model_columns(columns)
        self.assertEqual(m["A"], RESULTS_A)
        self.assertEqual(m["B"], RESULTS_A + ["use_fb_cur", "zone_rate_cur"])
        self.assertEqual(m["V"], m["B"] + ["chg_fb_velo"])
        self.assertEqual(m["P"], m["B"] + PT_FEATURES)
        self.assertEqual(m["C"], m["B"] + ["chg_fb_velo", "chg_zone_rate"])
        self.assertEqual(set(m["B"]) - set(m["Bnz"]), {"zone_rate_cur"})
        self.assertEqual(set(m["P"]) - set(m["Pnz"]), {"zone_rate_cur"})
        self.assertTrue(set(ZONE).isdisjoint(m["Pnz"]))


if __name__ == "__main__":
    unittest.main()
