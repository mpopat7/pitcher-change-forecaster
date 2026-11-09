import unittest

import numpy as np
import pandas as pd

from pcf.whiff_value import group_of, window_counts

T = pd.Timestamp("2023-06-05")


def _daily(rows):
    d = pd.DataFrame(rows, columns=["pitcher", "pitch_type", "game_date", "swings", "whiffs"])
    d["game_date"] = pd.to_datetime(d.game_date)
    d["season"] = d.game_date.dt.year
    return d


class WhiffValueTests(unittest.TestCase):
    def test_groups(self):
        self.assertEqual([group_of(t) for t in ("FF", "SI", "FC", "ST", "CH", "KN", "UN")],
                         ["fb", "fb", "cut", "brk", "off", "other", "other"])

    def test_windows_separate_future_from_cutoff_history(self):
        daily = _daily([
            (1, "SL", "2023-06-05", 10, 4),   # forecast day counts as future
            (1, "SL", "2023-07-04", 5, 1),    # day 29: future
            (1, "SL", "2023-07-05", 50, 50),  # day 30: outside
            (1, "SL", "2023-06-04", 8, 2),    # recent and season to date
            (1, "SL", "2023-04-01", 6, 3),    # season to date only
            (1, "SL", "2022-08-01", 20, 5),   # prior season
            (1, "FF", "2023-06-10", 7, 0),
        ])
        overall, typed = window_counts(daily, pd.DataFrame({"pitcher": [1, 2], "forecast_date": [T, T]}))
        o = overall.set_index("pitcher")
        self.assertEqual((o.loc[1, "y_swings"], o.loc[1, "y_whiffs"]), (22, 5))
        self.assertEqual((o.loc[2, "y_swings"], o.loc[2, "y_whiffs"]), (0, 0))
        sl = typed[(typed.pitcher == 1) & (typed.pitch_type == "SL")].iloc[0]
        self.assertEqual((sl.y_sw, sl.y_wh, sl.r30_sw, sl.std_sw, sl.ps_sw), (15, 5, 8, 14, 20))


if __name__ == "__main__":
    unittest.main()
