"""Rank projected performance and audit experimental C-versus-B disagreement alerts."""

import argparse
import json

import numpy as np
import pandas as pd

from pcf.model import ARTIFACTS, ROWS, row_loss

RULES = {
    "ranking_model": "B", "ranking_size": 10,
    "alert_delta": 0.02, "min_current_pitches": 150, "min_baseline_pitches": 150,
    "cooldown_days": 30, "min_resolution_bf": 50,
    "min_past_results_bf": 50,
}
CONTEXT = ["pitcher", "forecast_date", "n_pitches_cur", "chg_base_pitches",
           "bf_r30", "k_r30", "bb_r30", "bf_ps", "k_ps", "bb_ps", "start_share"]


def select_watchlist(forecasts, rules):
    rows = forecasts.copy()
    for model in ("A", "B", "C"):
        rows[f"k_minus_bb_{model}"] = rows[f"p_k_{model}"] - rows[f"p_bb_{model}"]
    rows["change_delta"] = rows.k_minus_bb_C - rows.k_minus_bb_B
    ranked = rows.sort_values(["forecast_date", f"k_minus_bb_{rules['ranking_model']}", "pitcher"],
                              ascending=[True, False, True])
    rows["performance_rank"] = ranked.groupby("forecast_date").cumcount().add(1).reindex(rows.index)
    rows["performance_watchlist"] = rows.performance_rank <= rules["ranking_size"]
    rows["alert_candidate"] = (
        ~rows.repertoire_fallback
        & rows.n_pitches_cur.ge(rules["min_current_pitches"])
        & rows.chg_base_pitches.ge(rules["min_baseline_pitches"])
        & rows.change_delta.abs().ge(rules["alert_delta"] - 1e-12)
    )
    rows["alert"] = False
    last_alert = {}
    candidates = rows[rows.alert_candidate].sort_values(["forecast_date", "pitcher"])
    for index, row in candidates.iterrows():
        prior = last_alert.get(row.pitcher)
        if prior is None or (row.forecast_date - prior).days >= rules["cooldown_days"]:
            rows.loc[index, "alert"] = True
            last_alert[row.pitcher] = row.forecast_date
    rows["alert_direction"] = np.where(rows.change_delta >= 0, "improvement", "decline")
    rows["recent_results_delta"] = ((rows.k_r30 - rows.bb_r30) / rows.bf_r30.replace(0, np.nan)
                                     - (rows.k_ps - rows.bb_ps) / rows.bf_ps.replace(0, np.nan))
    past_comparable = (rows.bf_r30.ge(rules["min_past_results_bf"])
                       & rows.bf_ps.ge(rules["min_past_results_bf"]))
    evident = np.sign(rows.change_delta) * rows.recent_results_delta >= rules["alert_delta"] - 1e-12
    rows["recent_results_status"] = np.select(
        [~past_comparable, evident], ["insufficient_past_results", "already_evident"],
        default="not_yet_evident")
    return rows


def evaluate_alerts(selected, rules):
    alerts = selected[selected.alert].copy()
    alerts["resolution"] = "unresolved"
    resolved = alerts.y_bf >= rules["min_resolution_bf"]
    direction = np.where(alerts.change_delta >= 0, 1, -1)
    observed = (alerts.y_k - alerts.y_bb) / alerts.y_bf.replace(0, np.nan)
    alerts["observed_k_minus_bb"] = observed
    alerts["signed_residual_from_B"] = direction * (observed - alerts.k_minus_bb_B)
    confirmed = alerts.signed_residual_from_B >= rules["alert_delta"] - 1e-12
    alerts.loc[resolved & confirmed, "resolution"] = "confirmed"
    alerts.loc[resolved & ~confirmed, "resolution"] = "false_alert"
    summaries = []
    for label, subset in [("all", alerts), ("improvement", alerts[alerts.alert_direction == "improvement"]),
                          ("decline", alerts[alerts.alert_direction == "decline"])]:
        counts = subset.resolution.value_counts()
        total = len(subset)
        false, confirmed = int(counts.get("false_alert", 0)), int(counts.get("confirmed", 0))
        adjudicated = false + confirmed
        record = {
            "direction": label, "alerts": total, "pitchers": int(subset.pitcher.nunique()),
            "confirmed": confirmed, "false_alerts": false,
            "unresolved": int(counts.get("unresolved", 0)),
            "false_fraction_resolved": false / adjudicated if adjudicated else None,
            "unconfirmed_fraction_all": (total - confirmed) / total if total else None,
            "not_yet_evident": int((subset.recent_results_status == "not_yet_evident").sum()),
            "confirmed_not_yet_evident": int(((subset.recent_results_status == "not_yet_evident")
                                               & (subset.resolution == "confirmed")).sum()),
            "insufficient_past_results": int(
                (subset.recent_results_status == "insufficient_past_results").sum()),
        }
        scored = subset[subset.y_bf > 0]
        record["scored_alerts"] = len(scored)
        record["bf"] = int(scored.y_bf.sum())
        for model in ("B", "C"):
            for target in ("k", "bb"):
                record[f"{model}_{target}_loss_per_bf"] = (
                    float(row_loss(scored[f"p_{target}_{model}"], scored[f"y_{target}"],
                                   scored.y_bf).sum() / scored.y_bf.sum())
                    if len(scored) else None)
        summaries.append(record)
    return alerts, summaries


def run(final=False):
    if final:
        from pcf.freeze import load_frozen_spec
        rules = load_frozen_spec()["watchlist"]
    else:
        rules = RULES
    season, tag = (2025, "holdout_2025") if final else (2024, "validation_2024")
    forecasts = pd.read_parquet(ARTIFACTS / f"forecasts_{tag}.parquet")
    context = pd.read_parquet(ROWS, columns=CONTEXT, filters=[("season", "==", season)])
    if len(context) != len(forecasts):
        raise ValueError("Watchlist forecasts must cover every eligible row")
    forecasts = forecasts.merge(context, on=["pitcher", "forecast_date"], validate="one_to_one")
    if len(forecasts) != len(context):
        raise ValueError("Watchlist forecast keys differ from the eligible population")
    selected = select_watchlist(forecasts, rules)
    alerts, summaries = evaluate_alerts(selected, rules)
    top = selected[selected.performance_watchlist]
    future = top.y_bf > 0
    report = {
        "season": season, "rules": rules, "eligible": len(selected),
        "eligible_no_followup": int((selected.y_bf == 0).sum()),
        "alert_candidates": int(selected.alert_candidate.sum()),
        "cooldown_suppressed": int(selected.alert_candidate.sum() - selected.alert.sum()),
        "performance_watchlist": {
            "forecasts": len(top), "pitchers": int(top.pitcher.nunique()),
            "no_followup": int((~future).sum()), "bf": int(top.y_bf.sum()),
            "predicted_k_minus_bb_bf_weighted": (
                float(np.average(top.loc[future, "k_minus_bb_B"], weights=top.loc[future, "y_bf"]))
                if future.any() else None),
            "observed_k_minus_bb_bf_weighted": (
                float((top.y_k.sum() - top.y_bb.sum()) / top.y_bf.sum()) if future.any() else None),
        },
        "alerts": summaries,
    }
    selected.to_csv(ARTIFACTS / f"watchlist_{tag}.csv", index=False)
    alerts.to_csv(ARTIFACTS / f"alerts_{tag}.csv", index=False)
    (ARTIFACTS / f"watchlist_{tag}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true", help="evaluate the already-scored 2025 holdout")
    run(parser.parse_args().final)
