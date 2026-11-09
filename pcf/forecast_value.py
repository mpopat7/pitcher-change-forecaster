"""Test whether per-type change features improve next-30-day K% and BB% forecasts in development.

Run: python3 -m pcf.forecast_value
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

from pcf.freeze import file_hash, load_frozen_spec
from pcf.model import (C_GRID, ROWS, TARGETS, Design, RepertoireSupport, binomial_fit, clustered_ci,
                       feature_sets, row_loss)

DESIGN = Path("experiments/forecast_value.json")
CHANGES = Path("artifacts/experiments/per_type_changes")
OUT = Path("artifacts/experiments/forecast_value")
FASTBALLS = ("FF", "SI", "FA")
PHYSICAL = ("velocity_mph", "hb_in", "ivb_in")
KEY = ["pitcher", "forecast_date"]
PT_FEATURES = ["pt_fb_velo", "pt_fb_ivb", "pt_fb_hb", "pt_other_velo", "pt_other_move",
               "pt_usage_shift", "pt_new_usage", "pt_max_z"]
ITER_GRID = [100, 200, 400]
COMPARISONS = [("P", "B"), ("V", "B"), ("C", "B"), ("P", "V"), ("NP", "NB"), ("NP", "P")]


def _supported(changes):
    return (changes.pitch_type.ne("UN") & changes.type_status.eq("both")
            & changes.type_outings_recent.ge(2) & changes.type_outings_baseline.ge(2)
            & np.isfinite(changes.delta))


def fit_priors(changes, train_seasons):
    d = changes[changes.season.isin(train_seasons) & _supported(changes)
                & changes.metric.isin(PHYSICAL)].copy()
    d["noise_variance"] = ((d.ci_high - d.ci_low) / 3.92) ** 2
    d = d[np.isfinite(d.noise_variance)]
    d["excess"] = d.delta ** 2 - d.noise_variance
    by_type = d.groupby(["metric", "pitch_type"]).agg(tau=("excess", "mean"), pitchers=("pitcher", "nunique"))
    pooled = d.groupby("metric").agg(tau=("excess", "mean"), pitchers=("pitcher", "nunique"))
    rows = []
    for metric in PHYSICAL:
        for pitch_type in sorted(changes.loc[changes.metric == metric, "pitch_type"].unique()):
            if (metric, pitch_type) in by_type.index and by_type.loc[(metric, pitch_type), "pitchers"] >= 2:
                tau, source = by_type.loc[(metric, pitch_type), "tau"], "pitch_type"
            elif metric in pooled.index and pooled.loc[metric, "pitchers"] >= 2:
                tau, source = pooled.loc[metric, "tau"], "metric_pool"
            else:
                continue
            rows.append({"metric": metric, "pitch_type": pitch_type, "tau": max(float(tau), 0.0),
                         "source": source})
    return pd.DataFrame(rows)


def per_type_features(changes, priors):
    """One row per forecast. Uses only cutoff measurements and outcome-free priors."""
    usage = changes[changes.metric == "usage"].copy()
    usage["recent_u"] = usage.recent.fillna(0.0)
    usage["baseline_u"] = usage.baseline.fillna(0.0)
    usage["abs_shift"] = (usage.recent_u - usage.baseline_u).abs()
    usage["new_u"] = np.where(usage.type_status.eq("recent_only"), usage.recent_u, 0.0)
    out = usage.groupby(KEY).agg(pt_usage_shift=("abs_shift", "sum"), pt_new_usage=("new_u", "sum"))
    out["pt_usage_shift"] *= 0.5

    phys = changes[changes.metric.isin(PHYSICAL) & _supported(changes)].merge(
        priors[["metric", "pitch_type", "tau"]], on=["metric", "pitch_type"], how="inner")
    v = ((phys.ci_high - phys.ci_low) / 3.92) ** 2
    denominator = phys.tau + v
    phys["shrunk"] = np.where(np.isfinite(v) & (denominator > 0), phys.delta * phys.tau / denominator, np.nan)
    phys = phys[np.isfinite(phys.shrunk)].copy()
    phys["z"] = np.where(phys.tau > 0, phys.shrunk.abs() / np.sqrt(phys.tau.where(phys.tau > 0)), np.nan)
    out["pt_max_z"] = phys.groupby(KEY).z.max()

    wide = phys.pivot_table(index=KEY + ["pitch_type"], columns="metric", values="shrunk")
    wide = wide.reindex(columns=list(PHYSICAL)).reset_index()
    wide = wide.merge(usage[KEY + ["pitch_type", "baseline_u"]], on=KEY + ["pitch_type"], how="left")

    fb_usage = usage[usage.pitch_type.isin(FASTBALLS)].sort_values(
        KEY + ["baseline_u", "pitch_type"], ascending=[True, True, False, True])
    primary = fb_usage[fb_usage.baseline_u > 0].drop_duplicates(KEY)[KEY + ["pitch_type"]]
    fb = primary.merge(wide, on=KEY + ["pitch_type"], how="left").set_index(KEY)
    out["pt_fb_velo"], out["pt_fb_ivb"], out["pt_fb_hb"] = fb.velocity_mph, fb.ivb_in, fb.hb_in

    other = wide[~wide.pitch_type.isin(FASTBALLS) & wide.baseline_u.gt(0)].copy()
    velo = other[np.isfinite(other.velocity_mph)]
    weights = velo.groupby(KEY).baseline_u.sum()
    out["pt_other_velo"] = (velo.velocity_mph * velo.baseline_u).groupby([velo.pitcher, velo.forecast_date]).sum() / weights
    move = other[np.isfinite(other.hb_in) & np.isfinite(other.ivb_in)].copy()
    move["m"] = move.hb_in.abs() + move.ivb_in.abs()
    weights = move.groupby(KEY).baseline_u.sum()
    out["pt_other_move"] = (move.m * move.baseline_u).groupby([move.pitcher, move.forecast_date]).sum() / weights
    return out[PT_FEATURES].reset_index()


def _hgb_fit(X, k, n, max_iter, fixed, seed):
    Xs, ys, w = np.vstack([X, X]), np.r_[np.ones(len(X)), np.zeros(len(X))], np.r_[k, n - k]
    keep = w > 0
    model = HistGradientBoostingClassifier(max_iter=max_iter, early_stopping=False, random_state=seed, **fixed)
    model.fit(Xs[keep], ys[keep], sample_weight=w[keep])
    return model


def _loss(p, k, n):
    return row_loss(p, k, n).sum() / n.sum()


def fit_predict(kind, cols, train, tune_train, tune_valid, test, ycol, design):
    """Choose one setting on the inner season, refit on all training rows, predict test rows."""
    def frames(fit_rows, pred_rows):
        transform = Design().fit(fit_rows[cols])
        return transform.transform(fit_rows[cols]), transform.transform(pred_rows[cols])

    X_tr, X_va = frames(tune_train, tune_valid)
    k_tr, n_tr = tune_train[ycol].to_numpy(), tune_train.y_bf.to_numpy()
    k_va, n_va = tune_valid[ycol].to_numpy(), tune_valid.y_bf.to_numpy()
    nl = design["nonlinear"]
    grid = C_GRID if kind == "linear" else ITER_GRID
    scores = []
    for setting in grid:
        if kind == "linear":
            p = binomial_fit(X_tr, k_tr, n_tr, setting).predict_proba(X_va)[:, 1]
        else:
            p = _hgb_fit(X_tr, k_tr, n_tr, setting, nl["fixed"], design["seed"]).predict_proba(X_va)[:, 1]
        scores.append(_loss(p, k_va, n_va))
    chosen = grid[int(np.argmin(scores))]
    X_tr, X_te = frames(train, test)
    k_tr, n_tr = train[ycol].to_numpy(), train.y_bf.to_numpy()
    if kind == "linear":
        p = binomial_fit(X_tr, k_tr, n_tr, chosen).predict_proba(X_te)[:, 1]
    else:
        p = _hgb_fit(X_tr, k_tr, n_tr, chosen, nl["fixed"], design["seed"]).predict_proba(X_te)[:, 1]
    return chosen, p


def run_fold(rows, changes, fold, design):
    seasons = fold["train"]
    priors = fit_priors(changes, seasons)
    features = per_type_features(changes[changes.season.isin(seasons + [fold["test"]])], priors)
    d = rows[rows.season.isin(seasons + [fold["test"]])].merge(features, on=KEY, how="left", validate="one_to_one")
    sets = feature_sets([c for c in rows.columns])
    models = {"A": sets["A"], "B": sets["B"], "V": sets["B"] + ["chg_fb_velo"], "C": sets["C"],
              "P": sets["B"] + PT_FEATURES}
    eligible_train = d[d.season.isin(seasons)]
    eligible_test = d[d.season == fold["test"]]
    unsupported = RepertoireSupport().fit(eligible_train).flags(eligible_test).any(axis=1).to_numpy()
    scored = d.y_bf.gt(0)
    train = d[d.season.isin(seasons) & scored]
    tune_train = d[d.season.isin(fold["tune_train"]) & scored]
    tune_valid = d[(d.season == fold["tune_valid"]) & scored]
    keep = eligible_test.y_bf.gt(0).to_numpy()
    test = eligible_test[keep]
    unsupported = unsupported[keep]
    out = test[KEY + ["season", "name", "y_bf", "y_k", "y_bb", "start_share", "pt_fb_velo"]].copy()
    out["repertoire_fallback"] = unsupported
    chosen = {}
    for target, ycol in TARGETS.items():
        for name, cols in models.items():
            chosen[(target, name)], p = fit_predict("linear", cols, train, tune_train, tune_valid, test, ycol, design)
            out[f"p_{target}_{name}"] = p
        for name, cols in (("NB", models["B"]), ("NP", models["P"])):
            chosen[(target, name)], p = fit_predict("hgb", cols, train, tune_train, tune_valid, test, ycol, design)
            out[f"p_{target}_{name}"] = p
        base = out[f"p_{target}_A"].to_numpy()
        for name in ("B", "V", "C", "P", "NB", "NP"):
            out[f"p_{target}_{name}"] = np.where(unsupported, base, out[f"p_{target}_{name}"])
    meta = {"test": fold["test"], "train": seasons, "eligible_test": len(eligible_test),
            "no_followup": int((~keep).sum()), "scored": len(test), "fallback": int(unsupported.sum()),
            "chosen": {f"{t}_{m}": v for (t, m), v in chosen.items()},
            "priors": priors.to_dict("records"),
            "feature_coverage": {c: float(d.loc[d.season == fold["test"], c].notna().mean()) for c in PT_FEATURES}}
    return out, meta


def directional(p_new, p_base, k, n):
    move = p_new - p_base
    gap = k / n - p_base
    moved = move != 0
    toward = np.sign(move) == np.sign(gap)
    return float(np.sum(n[moved & toward]) / np.sum(n[moved])) if moved.any() else np.nan, int(moved.sum())


def compare(pred, label):
    rows = []
    for target, ycol in TARGETS.items():
        k, n = pred[ycol].to_numpy(), pred.y_bf.to_numpy()
        losses = {m: row_loss(pred[f"p_{target}_{m}"].to_numpy(), k, n)
                  for m in ("A", "B", "V", "C", "P", "NB", "NP")}
        for new, old in COMPARISONS:
            est, lo, hi = clustered_ci(losses[new] - losses[old], n, pred.pitcher.to_numpy())
            share, moved = directional(pred[f"p_{target}_{new}"].to_numpy(), pred[f"p_{target}_{old}"].to_numpy(), k, n)
            rows.append({"group": label, "target": target, "comparison": f"{new}-{old}",
                         "rows": len(pred), "pitchers": pred.pitcher.nunique(), "bf": int(n.sum()),
                         "loss_new": losses[new].sum() / n.sum(), "loss_old": losses[old].sum() / n.sum(),
                         "diff_per_bf": est, "ci_low": lo, "ci_high": hi,
                         "toward_share": share, "rows_moved": moved})
    return rows


def write_report(summary, metas, out):
    def table(group):
        lines = ["| Target | Comparison | Loss new | Loss old | Difference per BF (95% pitcher CI) | Moves toward outcome |",
                 "|---|---|---:|---:|---|---:|"]
        for r in summary[summary.group == group].itertuples():
            lines.append(f"| {r.target.upper()}% | {r.comparison} | {r.loss_new:.5f} | {r.loss_old:.5f} | "
                         f"{r.diff_per_bf:+.6f} ({r.ci_low:+.6f} to {r.ci_high:+.6f}) | {r.toward_share:.1%} |")
        return lines
    lines = ["# Forecasting value of per-type changes", "",
             "Development test folds 2023 and 2024. Negative differences mean the first model has lower "
             "log loss per batter faced. Move share is the BF-weighted fraction of changed forecasts whose "
             "move away from the second model points toward the observed rate.", ""]
    for m in metas:
        lines.append(f"- {m['test']}: {m['eligible_test']:,} eligible, {m['no_followup']:,} without follow-up, "
                     f"{m['scored']:,} scored, {m['fallback']:,} repertoire fallbacks to A. "
                     f"Per-type coverage: " + ", ".join(f"{k} {v:.0%}" for k, v in m["feature_coverage"].items()))
    lines += ["", "## Pooled 2023-2024", ""] + table("pooled")
    for group in [g for g in summary.group.unique() if g != "pooled"]:
        lines += ["", f"## {group}", ""] + table(group)
    lines += ["", "Subgroups are exploratory. Lists of chosen settings are in config.json.", ""]
    (out / "report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DESIGN)
    parser.add_argument("--changes", type=Path, default=CHANGES)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    out, root = args.output.resolve(), Path("artifacts/experiments").resolve()
    if not out.is_relative_to(root) or out == root or (out.exists() and any(out.iterdir())):
        parser.error("choose an empty output subdirectory of artifacts/experiments")
    design = json.loads(args.design.read_text())
    if any(max(f["train"] + [f["test"]]) > 2024 for f in design["folds"]):
        parser.error("development folds must end by 2024")
    load_frozen_spec()
    old = json.loads((args.changes / "config.json").read_text())
    hashes = {p: file_hash(p) for p in ["pcf/forecast_value.py", "pcf/model.py", str(args.design),
                                         str(args.changes / "config.json"),
                                         str(args.changes / "changes.parquet"), str(ROWS),
                                         "research_spec.json"]}
    if old["original_study_sha256"] != hashes["research_spec.json"]:
        parser.error("per-type measurements belong to a different original study")
    rows = pd.read_parquet(ROWS, filters=[("season", "<=", 2024)])
    rows["month"] = rows.forecast_date.dt.month
    changes = pd.read_parquet(args.changes / "changes.parquet")
    if changes.season.max() > 2024 or rows.season.max() > 2024:
        raise ValueError("Development experiment read a non-development season")
    preds, metas = [], []
    for fold in design["folds"]:
        print(f"Fold: test {fold['test']}", flush=True)
        p, m = run_fold(rows, changes, fold, design)
        preds.append(p)
        metas.append(m)
    pred = pd.concat(preds, ignore_index=True)
    summary = compare(pred, "pooled")
    for season in sorted(pred.season.unique()):
        summary += compare(pred[pred.season == season], f"Test {season}")
    summary += compare(pred[pred.start_share >= 0.5], "Starters (exploratory)")
    summary += compare(pred[pred.start_share < 0.5], "Relievers (exploratory)")
    summary += compare(pred[pred.pt_fb_velo.notna()], "Supported primary-fastball change (exploratory)")
    summary += compare(pred[pred.pt_fb_velo.isna()], "No supported primary-fastball change (exploratory)")
    summary = pd.DataFrame(summary)
    for path, expected in hashes.items():
        if file_hash(path) != expected:
            raise RuntimeError(f"Input changed during analysis: {path}; results were not saved")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(
        {"design": design, "folds": metas, "input_sha256": hashes,
         "versions": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__}},
        indent=2, default=str) + "\n")
    pred.to_parquet(out / "predictions.parquet", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    write_report(summary, metas, out)
    print((out / "report.md").read_text())


if __name__ == "__main__":
    main()
