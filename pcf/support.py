"""Check the repertoire fallback on 2022 and 2023 without selecting settings on either season.

Run: python3 -m pcf.support
"""

import json

import pandas as pd

from pcf.dataset import HORIZON
from pcf.model import (
    ARTIFACTS, C_GRID, ROWS, TARGETS, USAGE, RepertoireSupport,
    fallback_predictions, feature_sets, fit_candidates, row_loss,
)


def chronological_checks():
    rows = pd.read_parquet(ROWS, filters=[("season", "<=", 2023)])
    rows["month"] = rows.forecast_date.dt.month
    sets = feature_sets(list(rows.columns))
    folds, metrics, audits = [], [], []
    for season in (2022, 2023):
        eligible_train = rows[rows.season < season]
        eligible_test = rows[rows.season == season]
        last_outcome = eligible_train.forecast_date.max() + pd.Timedelta(days=HORIZON - 1)
        if last_outcome >= eligible_test.forecast_date.min():
            raise ValueError("Training outcome windows overlap the evaluation season")
        support = RepertoireSupport().fit(eligible_train)
        flags = support.flags(eligible_test)
        unsupported = flags.any(axis=1)
        train = eligible_train[eligible_train.y_bf > 0]
        test = eligible_test[eligible_test.y_bf > 0]
        fallback = unsupported.loc[test.index].to_numpy()
        folds.append({
            "test_season": season, "train_seasons": sorted(train.season.unique().tolist()),
            "last_training_outcome": str(last_outcome.date()),
            "first_test_forecast": str(eligible_test.forecast_date.min().date()),
            "bounds": support.bounds(), "eligible": len(eligible_test), "scored": len(test),
            "flagged_eligible": int(unsupported.sum()), "flagged_scored": int(fallback.sum()),
            "flagged_pitchers": int(eligible_test.loc[unsupported, "pitcher"].nunique()),
            "flags_by_feature": flags.sum().astype(int).to_dict(),
        })
        audit = eligible_test[["pitcher", "name", "forecast_date", "y_bf", *USAGE]].copy()
        audit["repertoire_fallback"] = unsupported
        audit["repertoire_reasons"] = flags.apply(lambda r: ",".join(r.index[r]), axis=1)
        audits.append(audit)
        for target, ycol in TARGETS.items():
            predictions = {
                name: dict(fit_candidates(train, test, cols, ycol, C_GRID)[1])
                for name, cols in sets.items()
            }
            k, n = test[ycol].to_numpy(), test.y_bf.to_numpy()
            for C in C_GRID:
                losses = {}
                for name in sets:
                    raw = predictions[name][C]
                    guarded = (raw if name == "A" else
                               fallback_predictions(raw, predictions["A"][C], fallback))
                    raw_loss, guarded_loss = row_loss(raw, k, n), row_loss(guarded, k, n)
                    losses[name] = (raw_loss.sum() / n.sum(), guarded_loss.sum() / n.sum())
                    metrics.append({
                        "test_season": season, "target": target, "C": C, "model": name,
                        "raw_loss_per_bf": float(losses[name][0]),
                        "guarded_loss_per_bf": float(losses[name][1]),
                        "guarded_minus_raw": float((guarded_loss - raw_loss).sum() / n.sum()),
                        "flagged_bf": int(n[fallback].sum()),
                        "flagged_raw_loss_per_bf": (float(raw_loss[fallback].sum() / n[fallback].sum())
                                                    if fallback.any() else None),
                        "flagged_guarded_loss_per_bf": (
                            float(guarded_loss[fallback].sum() / n[fallback].sum())
                            if fallback.any() else None),
                    })
                print(f"{season} {target.upper()} C={C:g}: "
                      f"B fallback−raw {losses['B'][1] - losses['B'][0]:+.6f}; "
                      f"C fallback−raw {losses['C'][1] - losses['C'][0]:+.6f}", flush=True)
    report = {
        "rule": "current_usage_training_min_max_fallback_A",
        "regularization": "Every C_GRID setting reported; no selection on these test seasons",
        "folds": folds, "metrics": metrics,
    }
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "support_chronological.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame(metrics).to_csv(ARTIFACTS / "support_chronological.csv", index=False)
    pd.concat(audits, ignore_index=True).to_csv(ARTIFACTS / "repertoire_chronological.csv", index=False)
    return report


if __name__ == "__main__":
    chronological_checks()
