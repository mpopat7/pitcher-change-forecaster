"""Fit models A, B, and C for next-30-day K% and BB% and compare them on held-out seasons.

Default: train 2021-2023, tune regularization and report on 2024 validation.
--final: refit on 2021-2024 with the validation-chosen settings and score the 2025 holdout once.
Run: python3 -m pcf.model [--final]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROWS = Path("data/processed/forecast_rows.parquet")
ARTIFACTS = Path("artifacts")
C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
BOOTSTRAPS = 1000

RESULTS_A = [
    *(f"{s}_{w}" for s in ("k_logit", "bb_logit", "whiff", "log_bf") for w in ("r30", "std", "ps")),
    "opp_k_r30", "opp_bb_r30", "opp_k_std", "opp_bb_std",
    "sched_opp_k", "sched_opp_bb", "sched_games",
    "start_share", "bf_per_app", "apps_60", "days_since_last", "month",
]
TARGETS = {"k": "y_k", "bb": "y_bb"}


def feature_sets(cols: list[str]) -> dict[str, list[str]]:
    current = [c for c in cols if c.endswith("_cur")]
    changes = [c for c in cols if c.startswith("chg_")]
    return {"A": RESULTS_A, "B": RESULTS_A + current, "C": RESULTS_A + current + changes}


class Design:
    """Median imputation with missing-value indicators, then standardization; fit on training rows."""

    def fit(self, X: pd.DataFrame):
        self.cols = [c for c in X.columns if X[c].notna().any()]
        self.med = X.median()
        self.flag = [c for c in self.cols if X[c].isna().mean() > 0.01]
        Z = self._raw(X)
        self.mu, self.sd = Z.mean(), Z.std().replace(0, 1)
        return self

    def _raw(self, X):
        Z = X[self.cols].fillna(self.med)
        flags = X[self.flag].isna().astype(float).add_suffix("_missing")
        return pd.concat([Z, flags], axis=1)

    def transform(self, X):
        return ((self._raw(X) - self.mu) / self.sd).to_numpy()


def binomial_fit(X, k, n, C):
    """Logistic regression on aggregated counts: each row contributes k successes and n-k failures."""
    Xs = np.vstack([X, X])
    ys = np.r_[np.ones(len(X)), np.zeros(len(X))]
    w = np.r_[k, n - k]
    keep = w > 0
    m = LogisticRegression(C=C, max_iter=2000)
    m.fit(Xs[keep], ys[keep], sample_weight=w[keep])
    return m


def row_loss(p, k, n):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(k * np.log(p) + (n - k) * np.log(1 - p))


def clustered_ci(diff_loss, n, groups, seed=0):
    """Bootstrap over pitchers: the per-BF loss difference, resampling whole pitchers."""
    rng = np.random.default_rng(seed)
    g = pd.DataFrame({"d": diff_loss, "n": n, "g": groups}).groupby("g")[["d", "n"]].sum()
    d, w = g.d.to_numpy(), g.n.to_numpy()
    idx = rng.integers(0, len(g), size=(BOOTSTRAPS, len(g)))
    stats = d[idx].sum(1) / w[idx].sum(1)
    return float(d.sum() / w.sum()), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def calibration(p, k, n, bins=10):
    df = pd.DataFrame({"p": p, "k": k, "n": n})
    df["bin"] = pd.qcut(df.p, bins, labels=False, duplicates="drop")
    out = df.groupby("bin").apply(
        lambda s: pd.Series({"pred": np.average(s.p, weights=s.n), "obs": s.k.sum() / s.n.sum(),
                             "bf": s.n.sum()}), include_groups=False)
    return out.reset_index(drop=True)


def run(final: bool) -> dict:
    rows = pd.read_parquet(ROWS)
    rows["month"] = rows.forecast_date.dt.month
    coverage = (rows.groupby("season")
                .apply(lambda s: pd.Series({"eligible": len(s), "no_followup": int((s.y_bf == 0).sum())}),
                       include_groups=False))
    rows = rows[rows.y_bf > 0]

    train_seasons = [2021, 2022, 2023] + ([2024] if final else [])
    test_season = 2025 if final else 2024
    train, test = rows[rows.season.isin(train_seasons)], rows[rows.season == test_season]
    sets = feature_sets(list(rows.columns))
    chosen_path = ARTIFACTS / "chosen_C.json"
    chosen = json.loads(chosen_path.read_text()) if final else {}

    report = {"test_season": test_season, "train_seasons": train_seasons,
              "coverage": coverage.reset_index().to_dict("records"),
              "test_rows": len(test), "test_pitchers": int(test.pitcher.nunique()),
              "test_bf": int(test.y_bf.sum()), "targets": {}}
    preds = test[["pitcher", "name", "forecast_date", "y_bf", "y_k", "y_bb"]].copy()

    for tgt, ycol in TARGETS.items():
        k_tr, n_tr = train[ycol].to_numpy(), train.y_bf.to_numpy()
        k_te, n_te = test[ycol].to_numpy(), test.y_bf.to_numpy()
        base_rate = k_tr.sum() / n_tr.sum()
        res = {"league_rate_loss": float(row_loss(base_rate, k_te, n_te).sum() / n_te.sum()),
               "models": {}}
        losses = {}
        for name, cols in sets.items():
            design = Design().fit(train[cols])
            X_tr, X_te = design.transform(train[cols]), design.transform(test[cols])
            if final:
                grid = [chosen[tgt][name]]
            else:
                grid = C_GRID
            best = None
            for C in grid:
                m = binomial_fit(X_tr, k_tr, n_tr, C)
                p = m.predict_proba(X_te)[:, 1]
                loss = row_loss(p, k_te, n_te)
                if best is None or loss.sum() < best[2].sum():
                    best = (C, p, loss)
            C, p, loss = best
            losses[name] = loss
            preds[f"p_{tgt}_{name}"] = p
            res["models"][name] = {"C": C, "n_features": X_tr.shape[1],
                                   "loss_per_bf": float(loss.sum() / n_te.sum()),
                                   "calibration": calibration(p, k_te, n_te).round(4).to_dict("records")}
        groups = test.pitcher.to_numpy()
        for a, b in (("A", "B"), ("B", "C"), ("A", "C")):
            est, lo, hi = clustered_ci(losses[b] - losses[a], n_te, groups)
            # Negative means the second model has lower loss (better).
            res[f"{b}_minus_{a}"] = {"per_bf": est, "ci95": [lo, hi]}
        report["targets"][tgt] = res

    ARTIFACTS.mkdir(exist_ok=True)
    tag = "holdout_2025" if final else "validation_2024"
    if not final:
        chosen_path.write_text(json.dumps(
            {t: {m: r["models"][m]["C"] for m in r["models"]} for t, r in report["targets"].items()},
            indent=2))
    (ARTIFACTS / f"report_{tag}.json").write_text(json.dumps(report, indent=2, default=str))
    preds.to_parquet(ARTIFACTS / f"predictions_{tag}.parquet", index=False)
    return report


def summarize(report: dict) -> None:
    print(f"Test season {report['test_season']}: {report['test_rows']} forecasts, "
          f"{report['test_pitchers']} pitchers, {report['test_bf']} batters faced")
    for c in report["coverage"]:
        print(f"  {int(c['season'])}: {int(c['eligible'])} eligible, {int(c['no_followup'])} with no follow-up")
    for tgt, r in report["targets"].items():
        print(f"\n{tgt.upper()}%  log loss per BF (league-rate baseline {r['league_rate_loss']:.5f})")
        for m, v in r["models"].items():
            print(f"  {m}: {v['loss_per_bf']:.5f}  (C={v['C']}, {v['n_features']} features)")
        for key in ("B_minus_A", "C_minus_B", "C_minus_A"):
            v = r[key]
            print(f"  {key}: {v['per_bf']:+.6f}  95% CI [{v['ci95'][0]:+.6f}, {v['ci95'][1]:+.6f}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true", help="score the 2025 holdout (run once)")
    summarize(run(ap.parse_args().final))
