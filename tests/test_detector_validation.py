import unittest

import numpy as np
import pandas as pd

from pcf.detector_validation import (METHODS, cutoff, flag, method_scores, outcome, shrink,
                                     split_baseline, summarize, verdicts)


def _row(pitcher, season, metric, delta, **kw):
    base = {"pitcher": pitcher, "name": str(pitcher), "team": "DET", "season": season,
            "forecast_date": pd.Timestamp(f"{season}-06-01"), "pitch_type": "FF", "metric": metric,
            "recent": 10 + delta, "base_a": 10.0, "base_b": 10.0, "delta": delta, "noise_variance": 0.1,
            "type_outings_recent": 3, "type_outings_a": 2, "type_outings_b": 2,
            "future": 10 + delta, "future_outings": 3}
    base.update(kw)
    return base


class SplitTests(unittest.TestCase):
    def test_alternates_outings_by_date(self):
        b = pd.DataFrame({"game_pk": [5, 1, 3, 2, 3], "game_date": pd.to_datetime(
            ["2024-05-05", "2024-05-01", "2024-05-03", "2024-05-02", "2024-05-03"]), "pitch_type": "FF"})
        a, h = split_baseline(b)
        self.assertEqual(sorted(a.game_pk.unique()), [1, 3])
        self.assertEqual(sorted(h.game_pk.unique()), [2, 5])
        self.assertEqual(len(a) + len(h), len(b))


class OutcomeTests(unittest.TestCase):
    def test_persistence_is_judged_against_half_b(self):
        rows = pd.DataFrame([
            _row(1, 2023, "velocity_mph", 2.0, future=11.2),               # held 60% of 2.0 vs B: persists
            _row(2, 2023, "velocity_mph", 2.0, base_b=11.5, future=12.0),   # vs B lead 0.5, kept 0.5: persists
            _row(3, 2023, "velocity_mph", 2.0, base_b=12.5, future=12.6),   # recent below B: not persisted
            _row(4, 2023, "velocity_mph", -2.0, future=9.5),               # kept 25%: not persisted
            _row(5, 2023, "velocity_mph", 2.0, future_outings=1),          # unresolved
        ])
        r = outcome(rows)
        self.assertEqual(r.persisted.tolist(), [True, True, False, False, False])
        self.assertEqual(r.resolved.tolist(), [True, True, True, True, False])
        # Shared-baseline rule would call pitcher 3 persisted (future 12.6 vs base_a 10).
        self.assertTrue(r.persisted_shared.iloc[2])


class ShrinkTests(unittest.TestCase):
    def test_priors_use_earlier_seasons_only(self):
        rows = [_row(p, 2021, "velocity_mph", d) for p, d in ((1, 1.0), (2, -1.0))]
        rows += [_row(p, 2022, "velocity_mph", d) for p, d in ((3, 5.0), (4, -5.0))]
        d, priors = shrink(pd.DataFrame(rows))
        p22 = priors[priors.season == 2022].iloc[0]
        self.assertAlmostEqual(p22.prior_variance, 1.0 - 0.1)
        self.assertEqual(p22.training_end, 2021)
        self.assertTrue(d[d.season == 2021].prior_variance.isna().all())
        self.assertAlmostEqual(d[d.pitcher == 3].shrunk_delta.iloc[0], 5.0 * 0.9 / 1.0)

    def test_unsupported_rows_are_not_shrunk(self):
        rows = [_row(p, 2021, "velocity_mph", 1.0) for p in (1, 2)]
        rows += [_row(3, 2022, "velocity_mph", 3.0, type_outings_b=1)]
        d, _ = shrink(pd.DataFrame(rows))
        self.assertTrue(np.isnan(d[d.pitcher == 3].shrunk_delta.iloc[0]))


class ScoreAndFlagTests(unittest.TestCase):
    def setUp(self):
        rows = []
        for season in (2022, 2023):
            for p in range(1, 11):
                rows.append(_row(p, season, "velocity_mph", p / 10, prior_variance=1.0, shrunk_delta=p / 10))
                rows.append(_row(p, season, "hb_in", 0.05, prior_variance=1.0, shrunk_delta=p / 5))
                rows.append(_row(p, season, "usage", 0.01 * p, prior_variance=np.nan, shrunk_delta=np.nan))
        self.d = pd.DataFrame(rows)
        self.anchors = self.d[["pitcher", "forecast_date", "season"]].drop_duplicates()

    def test_physical_takes_strongest_metric(self):
        s = method_scores(self.d, "physical")
        top = s[s.season == 2022].iloc[0]
        self.assertEqual((top.pitcher, top.metric), (10, "hb_in"))
        v = method_scores(self.d, "velocity_shrunk")
        self.assertTrue(v.metric.eq("velocity_mph").all())

    def test_cutoff_counts_all_anchors(self):
        self.assertEqual(cutoff([5, 4, 3, 2, 1], 20, 0.1), 4)
        with self.assertRaises(ValueError):
            cutoff([5], 20, 0.1)

    def test_prospective_cutoff_uses_past_seasons_only(self):
        scored = {m: outcome(method_scores(self.d, m)) for m in METHODS}
        # Make 2023 scores huge; the 2023 cutoff must still come from 2022.
        flags, cutoffs = flag(scored, self.anchors, [0.2], [2023], [2022])
        c = cutoffs[(cutoffs.method == "velocity_raw")].iloc[0]
        self.assertEqual(c.past_seasons, "2022")
        self.assertAlmostEqual(c.cutoff, 0.9)
        retro = flags[(flags.scheme == "retrospective") & (flags.method == "velocity_raw")]
        self.assertEqual(sorted(retro.pitcher), [9, 10])

    def test_summary_and_verdicts(self):
        scored = {m: outcome(method_scores(self.d, m)) for m in METHODS}
        flags, _ = flag(scored, self.anchors, [0.1, 0.2], [2023], [2022, 2023])
        summary = summarize(flags, [2022, 2023], draws=50, seed=1)
        phys = summary[(summary.method == "physical") & (summary.scheme == "prospective")
                       & (summary.capacity == 0.1) & (summary.seasons == "pooled")].iloc[0]
        self.assertEqual(phys.rate_resolved, 1.0)
        design = {"capacities": {"primary": 0.1}}
        v = verdicts(summary, design)
        self.assertEqual(set(v), {"beats_velocity_alerts", "movement_adds"})


if __name__ == "__main__":
    unittest.main()
