"""Test whether recent pitch changes improve next-30-day whiffs-per-swing forecasts.

    python3 -m pcf.whiff_value              # development folds 2023 and 2024
    python3 -m pcf.whiff_value --freeze     # record sources and the 2024-fold settings
    python3 -m pcf.whiff_value --evaluate   # score 2025 and 2026 once
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from pcf.dataset import BREAKING, FASTBALLS, OFFSPEED, SWINGS, WHIFFS
from pcf.forecast_value import (KEY, PT_FEATURES, Design, _hgb_fit, _supported, directional, fit_priors,
                                per_type_features)
from pcf.freeze import file_hash, load_frozen_spec
from pcf.model import C_GRID, ROWS, RepertoireSupport, binomial_fit, clustered_ci, feature_sets, row_loss

DESIGN = Path("experiments/whiff_value.json")
NONLINEAR = Path("experiments/forecast_value.json")
FREEZE = Path("experiments/whiff_value_freeze.json")
DEV_CHANGES = Path("artifacts/experiments/per_type_changes/changes.parquet")
ROWS_2026 = Path("data/processed/forecast_rows_2026.parquet")
RAW = Path("data/raw/statcast")
DEV_OUT = Path("artifacts/experiments/whiff_value")
EVAL_OUT = Path("artifacts/experiments/whiff_value_evaluation")
ITER_GRID = [100, 200, 400]
GROUPS = ("fb", "cut", "brk", "off")
PHYS = {"velocity_mph": "velo", "ivb_in": "ivb", "hb_in": "hb"}
TYPE_BASE = [f"g_{g}" for g in GROUPS] + ["t_velo", "t_ivb", "t_hb", "t_usage",
                                         "t_wh_r30", "t_wh_std", "t_wh_ps", "t_lsw_r30", "t_lsw_std", "t_lsw_ps"]
TYPE_CHANGE = [f"c_{m}_{g}" for m in PHYS.values() for g in GROUPS] + ["t_usage_delta", "t_z"]
PRIOR_SWINGS = 50
SOURCES = ["pcf/whiff_value.py", "pcf/forecast_value.py", "pcf/model.py", "pcf/dataset.py", "pcf/changes.py",
           "pcf/evaluate_2026.py", str(DESIGN), str(NONLINEAR), "pyproject.toml", "uv.lock", "research_spec.json"]


def group_of(pitch_type):
    if pitch_type in FASTBALLS:
        return "fb"
    if pitch_type == "FC":
        return "cut"
    if pitch_type in BREAKING:
        return "brk"
    if pitch_type in OFFSPEED:
        return "off"
    return "other"


def daily_swings(through_season):
    files = sorted(f for f in RAW.glob("*/*.parquet") if int(f.parent.name) <= through_season)
    cols = ["game_type", "game_pk", "game_date", "at_bat_number", "pitch_number", "pitcher", "pitch_type",
            "description"]
    p = pd.concat((pd.read_parquet(f, columns=cols) for f in files), ignore_index=True)
    p = p[p.game_type == "R"].drop_duplicates(["game_pk", "at_bat_number", "pitch_number"])
    p["game_date"] = pd.to_datetime(p.game_date)
    p["season"] = p.game_date.dt.year
    p["pitch_type"] = p.pitch_type.fillna("UN")
    p["swing"] = p.description.isin(SWINGS)
    p["whiff"] = p.description.isin(WHIFFS)
    return (p.groupby(["pitcher", "pitch_type", "season", "game_date"])
            .agg(swings=("swing", "sum"), whiffs=("whiff", "sum")).reset_index())


def window_counts(daily, keys):
    """Future totals and cutoff-only type histories for each forecast key. Keys: pitcher, forecast_date."""
    overall, typed = [], []
    for t, group in keys.groupby("forecast_date"):
        season = t.year
        pitchers = group.pitcher.unique()
        d = daily[daily.pitcher.isin(pitchers)]
        same = d[d.season == season]
        future = same[(same.game_date >= t) & (same.game_date <= t + pd.Timedelta(days=29))]
        windows = {
            "y": future,
            "r30": same[(same.game_date < t) & (same.game_date >= t - pd.Timedelta(days=30))],
            "std": same[same.game_date < t],
            "ps": d[d.season == season - 1],
        }
        o = future.groupby("pitcher")[["swings", "whiffs"]].sum().rename(
            columns={"swings": "y_swings", "whiffs": "y_whiffs"})
        overall.append(o.reindex(pitchers, fill_value=0).rename_axis("pitcher").reset_index().assign(forecast_date=t))
        parts = []
        for name, frame in windows.items():
            parts.append(frame.groupby(["pitcher", "pitch_type"])[["swings", "whiffs"]].sum()
                         .rename(columns={"swings": f"{name}_sw", "whiffs": f"{name}_wh"}))
        typed.append(pd.concat(parts, axis=1).fillna(0).reset_index().assign(forecast_date=t))
    return pd.concat(overall, ignore_index=True), pd.concat(typed, ignore_index=True)


def type_frame(changes, priors, rows, history, league):
    """One row per supported pitcher-date-type with cutoff-only features and its future target."""
    supported = changes[_supported(changes)]
    phys = supported[supported.metric.isin(PHYS)].merge(priors[["metric", "pitch_type", "tau"]],
                                                         on=["metric", "pitch_type"], how="left")
    v = ((phys.ci_high - phys.ci_low) / 3.92) ** 2
    denominator = phys.tau + v
    phys["shrunk"] = np.where(np.isfinite(v) & (denominator > 0), phys.delta * phys.tau / denominator, np.nan)
    phys["z"] = np.where(phys.tau > 0, phys.shrunk.abs() / np.sqrt(phys.tau.where(phys.tau > 0)), np.nan)
    ids = KEY + ["pitch_type"]
    recent = phys.pivot_table(index=ids, columns="metric", values="recent").rename(
        columns={m: f"t_{s}" for m, s in PHYS.items()})
    shrunk = phys.pivot_table(index=ids, columns="metric", values="shrunk", dropna=False).rename(
        columns={m: f"s_{s}" for m, s in PHYS.items()})
    z = phys.groupby(ids).z.max().rename("t_z")
    usage = supported[supported.metric == "usage"].set_index(ids)[["recent", "delta"]].rename(
        columns={"recent": "t_usage", "delta": "t_usage_delta"})
    d = usage.join([recent, shrunk, z], how="left").reset_index()
    d = d.merge(history, on=ids, how="left")
    for w in ("r30", "std", "ps"):
        sw, wh = d[f"{w}_sw"].fillna(0), d[f"{w}_wh"].fillna(0)
        d[f"t_wh_{w}"] = (wh + PRIOR_SWINGS * league) / (sw + PRIOR_SWINGS)
        d[f"t_lsw_{w}"] = np.log1p(sw)
    d["y_swings_t"], d["y_whiffs_t"] = d.y_sw.fillna(0), d.y_wh.fillna(0)
    for c in ("t_velo", "t_ivb", "t_hb", *(f"s_{s}" for s in PHYS.values())):
        if c not in d:
            d[c] = np.nan
    d["group"] = d.pitch_type.map(group_of)
    for g in GROUPS:
        d[f"g_{g}"] = (d.group == g).astype(float)
        for s in PHYS.values():
            d[f"c_{s}_{g}"] = np.where(d.group == g, d[f"s_{s}"], 0.0)
    pitcher = rows[list(dict.fromkeys(KEY + ["season", "start_share", "y_bf"] + feature_sets(list(rows.columns))["B"]))]
    return d.merge(pitcher, on=KEY, how="inner", validate="many_to_one")


def fit(kind, cols, train, test, kcol, ncol, setting, fixed, seed):
    transform = Design().fit(train[cols])
    X_tr, X_te = transform.transform(train[cols]), transform.transform(test[cols])
    k, n = train[kcol].to_numpy(), train[ncol].to_numpy()
    if kind == "linear":
        return binomial_fit(X_tr, k, n, setting).predict_proba(X_te)[:, 1]
    return _hgb_fit(X_tr, k, n, setting, fixed, seed).predict_proba(X_te)[:, 1]


def choose(kind, cols, tune_train, tune_valid, kcol, ncol, fixed, seed):
    grid = C_GRID if kind == "linear" else ITER_GRID
    k, n = tune_valid[kcol].to_numpy(), tune_valid[ncol].to_numpy()
    losses = [row_loss(fit(kind, cols, tune_train, tune_valid, kcol, ncol, s, fixed, seed), k, n).sum() for s in grid]
    return grid[int(np.argmin(losses))]


def overall_models(rows):
    sets = feature_sets(list(rows.columns))
    return {"A": ("linear", sets["A"]), "B": ("linear", sets["B"]), "C": ("linear", sets["C"]),
            "V": ("linear", sets["B"] + ["chg_fb_velo"]), "P": ("linear", sets["B"] + PT_FEATURES),
            "NB": ("hgb", sets["B"]), "NP": ("hgb", sets["B"] + PT_FEATURES)}


def type_models(rows):
    b = feature_sets(list(rows.columns))["B"] + TYPE_BASE
    return {"Bt": ("linear", b), "Pt": ("linear", b + TYPE_CHANGE),
            "NBt": ("hgb", b), "NPt": ("hgb", b + TYPE_CHANGE)}


def prepare(rows, changes, daily, train_seasons, test_season):
    seasons = train_seasons + [test_season]
    priors = fit_priors(changes, train_seasons)
    rows = rows[rows.season.isin(seasons)].copy()
    rows["month"] = rows.forecast_date.dt.month
    rows = rows.merge(per_type_features(changes[changes.season.isin(seasons)], priors), on=KEY, how="left",
                      validate="one_to_one")
    overall, history = window_counts(daily, rows[KEY])
    rows = rows.merge(overall, on=KEY, how="left", validate="one_to_one")
    train_hist = history[history.forecast_date.dt.year.isin(train_seasons)]
    league = train_hist.y_wh.sum() / train_hist.y_sw.sum()
    typed = type_frame(changes[changes.season.isin(seasons)], priors, rows, history, league)
    return rows, typed, {"league_whiff_rate": float(league), "prior_sources": priors.source.value_counts().to_dict()}


def score_fold(rows, typed, train_seasons, test_season, settings, fixed, seed, tune=None):
    """settings: dict model->setting, or None to tune on `tune` = (tune_train_seasons, tune_valid_season)."""
    chosen = {}
    out = {}
    for label, frame, models, kcol, ncol in (
            ("overall", rows, overall_models(rows), "y_whiffs", "y_swings"),
            ("per_type", typed, type_models(rows), "y_whiffs_t", "y_swings_t")):
        scored = frame[frame[ncol] > 0]
        train = scored[scored.season.isin(train_seasons)]
        test = scored[scored.season == test_season]
        pred = test[[c for c in KEY + ["pitch_type", "group", "t_z", "season", "start_share", kcol, ncol]
                     if c in test]].copy()
        for name, (kind, cols) in models.items():
            if settings is None:
                tt = scored[scored.season.isin(tune[0])]
                tv = scored[scored.season == tune[1]]
                setting = choose(kind, cols, tt, tv, kcol, ncol, fixed, seed)
            else:
                setting = settings[f"{label}_{name}"]
            chosen[f"{label}_{name}"] = setting
            pred[f"p_{name}"] = fit(kind, cols, train, test, kcol, ncol, setting, fixed, seed)
        if label == "overall":
            eligible_train = frame[frame.season.isin(train_seasons)]
            flags = RepertoireSupport().fit(eligible_train).flags(test).any(axis=1).to_numpy()
            pred["repertoire_fallback"] = flags
            for name in models:
                if name != "A":
                    pred[f"p_{name}"] = np.where(flags, pred.p_A, pred[f"p_{name}"])
        out[label] = pred
    return out, chosen


def compare(pred, pairs, kcol, ncol, label):
    k, n = pred[kcol].to_numpy(), pred[ncol].to_numpy()
    rows = []
    for a, b in pairs:
        la, lb = row_loss(pred[f"p_{a}"].to_numpy(), k, n), row_loss(pred[f"p_{b}"].to_numpy(), k, n)
        est, lo, hi = clustered_ci(la - lb, n, pred.pitcher.to_numpy())
        share, moved = directional(pred[f"p_{a}"].to_numpy(), pred[f"p_{b}"].to_numpy(), k, n)
        rows.append({"group": label, "comparison": f"{a}-{b}", "rows": len(pred), "swings": int(n.sum()),
                     "loss_new": la.sum() / n.sum(), "loss_old": lb.sum() / n.sum(), "diff_per_swing": est,
                     "ci_low": lo, "ci_high": hi, "toward_share": share})
    return rows


def summarize(overall, typed, design):
    pairs_o = [tuple(c.split("-")) for c in design["comparisons"]["overall"]]
    pairs_t = [tuple(c.split("-")) for c in design["comparisons"]["per_type"]]
    rows = []
    for season in ["pooled", *sorted(overall.season.unique())]:
        o = overall if season == "pooled" else overall[overall.season == season]
        t = typed if season == "pooled" else typed[typed.season == season]
        tag = "pooled" if season == "pooled" else f"Test {season}"
        rows += compare(o, pairs_o, "y_whiffs", "y_swings", f"Overall · {tag}")
        rows += compare(t, pairs_t, "y_whiffs_t", "y_swings_t", f"Per type · {tag}")
    for g in GROUPS:
        rows += compare(typed[typed.group == g], pairs_t, "y_whiffs_t", "y_swings_t", f"Per type · {g} (exploratory)")
    big = typed[typed.t_z >= 2]
    rows += compare(big, pairs_t, "y_whiffs_t", "y_swings_t", "Per type · change score >= 2 (exploratory)")
    return pd.DataFrame(rows)


def write_report(summary, meta, out, title):
    lines = [f"# {title}", "", "Log loss per swing; negative differences favor the first model. Move share is the "
             "swing-weighted fraction of changed forecasts whose move points toward the observed rate.", ""]
    for m in meta:
        lines.append(f"- Test {m['test']}: overall {m['overall_eligible']:,} forecasts, {m['overall_scored']:,} with "
                     f"swings; per type {m['type_rows']:,} pitcher-type rows, {m['type_scored']:,} with swings; "
                     f"league whiff rate {m['league_whiff_rate']:.3f}.")
    for group, g in summary.groupby("group", sort=False):
        lines += ["", f"## {group}", "", "| Comparison | Rows | Loss new | Loss old | Difference per swing (95% pitcher CI) | Moves toward outcome |",
                  "|---|---:|---:|---:|---|---:|"]
        for r in g.itertuples():
            lines.append(f"| {r.comparison} | {r.rows:,} | {r.loss_new:.5f} | {r.loss_old:.5f} | {r.diff_per_swing:+.6f} "
                         f"({r.ci_low:+.6f} to {r.ci_high:+.6f}) | {r.toward_share:.1%} |")
    lines.append("")
    (out / "report.md").write_text("\n".join(lines))


def save(out, design, meta, chosen, hashes, overall, typed, summary, title):
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(
        {"design": design, "folds": meta, "chosen": chosen, "input_sha256": hashes,
         "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
         "versions": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__}},
        indent=2, default=str) + "\n")
    overall.to_parquet(out / "predictions_overall.parquet", index=False)
    typed.to_parquet(out / "predictions_type.parquet", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    write_report(summary, meta, out, title)
    print((out / "report.md").read_text())


def _fold_meta(rows, typed, test, info, preds):
    return {"test": test, "overall_eligible": int((rows.season == test).sum()),
            "overall_scored": len(preds["overall"]), "type_rows": int((typed.season == test).sum()),
            "type_scored": len(preds["per_type"]), **info}


def development(design, nonlinear):
    hashes = {p: file_hash(p) for p in SOURCES + [str(DEV_CHANGES), str(ROWS)]}
    load_frozen_spec()
    rows = pd.read_parquet(ROWS, filters=[("season", "<=", 2024)])
    changes = pd.read_parquet(DEV_CHANGES)
    daily = daily_swings(2024)
    if rows.season.max() > 2024 or changes.season.max() > 2024 or daily.season.max() > 2024:
        raise ValueError("Development read a non-development season")
    overall, typed, meta, chosen = [], [], [], {}
    for fold in design["development_folds"]:
        print(f"Fold: test {fold['test']}", flush=True)
        r, t, info = prepare(rows, changes, daily, fold["train"], fold["test"])
        preds, ch = score_fold(r, t, fold["train"], fold["test"], None, nonlinear["fixed"], nonlinear_seed(),
                               tune=(fold["tune_train"], fold["tune_valid"]))
        chosen[str(fold["test"])] = ch
        overall.append(preds["overall"])
        typed.append(preds["per_type"])
        meta.append(_fold_meta(r, t, fold["test"], info, preds))
    if {p: file_hash(p) for p in hashes} != hashes:
        raise RuntimeError("Inputs changed during the run; results were not saved")
    overall, typed = pd.concat(overall, ignore_index=True), pd.concat(typed, ignore_index=True)
    save(DEV_OUT, design, meta, chosen, hashes, overall, typed, summarize(overall, typed, design),
         "Whiff forecasts: development folds 2023-2024")


def nonlinear_seed():
    return json.loads(NONLINEAR.read_text())["seed"]


def freeze():
    if not (DEV_OUT / "config.json").exists():
        raise SystemExit("Run development first")
    if FREEZE.exists():
        raise SystemExit("Already frozen")
    chosen = json.loads((DEV_OUT / "config.json").read_text())["chosen"]["2024"]
    record = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "settings": chosen, "settings_source": "development fold testing 2024",
              "source_sha256": {p: file_hash(p) for p in SOURCES + [str(DEV_CHANGES), str(ROWS), str(ROWS_2026)]},
              "development_report_sha256": file_hash(DEV_OUT / "report.md")}
    FREEZE.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Frozen: {FREEZE}")


def evaluate(design, nonlinear):
    from pcf.evaluate_2026 import new_changes
    record = json.loads(FREEZE.read_text())
    changed = [p for p, h in record["source_sha256"].items() if file_hash(p) != h]
    if changed:
        raise SystemExit(f"Sources changed after freezing: {changed}")
    load_frozen_spec()
    rows = pd.read_parquet(ROWS_2026)
    dev = pd.read_parquet(DEV_CHANGES)
    print("Measuring per-type changes for 2025-2026", flush=True)
    fresh, _ = new_changes([2025, 2026])
    changes = pd.concat([dev[dev.season <= 2024], fresh], ignore_index=True)
    daily = daily_swings(2026)
    overall, typed, meta = [], [], []
    for test in (2025, 2026):
        train = list(range(2021, test))
        print(f"Scoring {test}", flush=True)
        r, t, info = prepare(rows, changes, daily, train, test)
        preds, _ = score_fold(r, t, train, test, record["settings"], nonlinear["fixed"], nonlinear_seed())
        overall.append(preds["overall"])
        typed.append(preds["per_type"])
        meta.append(_fold_meta(r, t, test, info, preds))
    overall, typed = pd.concat(overall, ignore_index=True), pd.concat(typed, ignore_index=True)
    save(EVAL_OUT, design, meta, record["settings"], record["source_sha256"], overall, typed,
         summarize(overall, typed, design), "Whiff forecasts: frozen evaluation 2025-2026")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    design = json.loads(DESIGN.read_text())
    nonlinear = json.loads(NONLINEAR.read_text())["nonlinear"]
    if args.freeze:
        return freeze()
    out = EVAL_OUT if args.evaluate else DEV_OUT
    if out.exists() and any(out.iterdir()):
        parser.error(f"{out} already holds results; this run happens once")
    if args.evaluate:
        if not FREEZE.exists():
            parser.error("freeze first")
        return evaluate(design, nonlinear)
    if FREEZE.exists():
        parser.error("development must precede the freeze")
    development(design, nonlinear)


if __name__ == "__main__":
    main()
