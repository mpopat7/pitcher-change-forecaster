"""Test whether recent pitch changes help predict injured-list placements in the next 30 days.

Run, in order (see experiments/injury_warning.json):
    python3 -m pcf.injury_warning develop
    python3 -m pcf.injury_warning freeze
    python3 -m pcf.injury_warning download --seasons 2025 2026
    python3 -m pcf.injury_warning evaluate
"""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import sklearn
from sklearn.metrics import roc_auc_score

from pcf.forecast_value import KEY, PT_FEATURES, _hgb_fit, fit_priors, per_type_features
from pcf.freeze import file_hash, load_frozen_spec
from pcf.model import C_GRID, Design, binomial_fit, clustered_ci, feature_sets, row_loss

DESIGN = Path("experiments/injury_warning.json")
FREEZE = Path("experiments/injury_warning_freeze.json")
NONLINEAR = Path("experiments/forecast_value.json")
TRANSACTIONS = Path("data/raw/transactions")
ROWS = Path("data/processed/forecast_rows_2026.parquet")
DEV_CHANGES = Path("artifacts/experiments/per_type_changes/changes.parquet")
DEV_OUT = Path("artifacts/experiments/injury_warning_dev")
EVAL_OUT = Path("artifacts/experiments/injury_warning_evaluation")
SOURCES = ["pcf/injury_warning.py", "pcf/forecast_value.py", "pcf/evaluate_2026.py", "pcf/model.py",
           "pcf/changes.py", "pcf/dataset.py", "pcf/freeze.py", str(DESIGN), str(NONLINEAR),
           "pyproject.toml", "uv.lock", "research_spec.json", str(DEV_CHANGES)]
HISTORY = ["il_365", "il_prev_season"]
LABELS = {"primary": (0, 29, None), "lead_time": (7, 36, (0, 6))}
ITER_GRID = [100, 200, 400]
COMPARISONS = [("X", "W"), ("V", "W"), ("X", "V"), ("NX", "NW")]
PLACED = re.compile(r"placed .* on the (10|15|60)-day injured list", re.I)
RETRO = re.compile(r"retroactive to ([A-Z][a-z]+ \d{1,2}, \d{4})")


def download(seasons):
    TRANSACTIONS.mkdir(parents=True, exist_ok=True)
    for season in seasons:
        out = TRANSACTIONS / f"{season}.json"
        if out.exists():
            continue
        r = requests.get("https://statsapi.mlb.com/api/v1/transactions", timeout=120,
                         params={"startDate": f"{season}-01-01", "endDate": f"{season}-12-31", "sportId": 1})
        r.raise_for_status()
        out.write_text(r.text)
        print(f"{season}: {len(r.json()['transactions']):,} transactions", flush=True)


def parse_placements(transactions):
    rows = []
    for t in transactions:
        text = t.get("description", "")
        lower = text.lower()
        if not PLACED.search(text) or "transferred" in lower or "covid" in lower or "person" not in t:
            continue
        retro = RETRO.search(text)
        event = pd.Timestamp(retro.group(1)) if retro else pd.Timestamp(t["date"])
        rows.append({"pitcher": int(t["person"]["id"]), "event": event})
    return pd.DataFrame(rows, columns=["pitcher", "event"]).drop_duplicates()


def load_placements(last_season):
    files = sorted(TRANSACTIONS.glob("*.json"))
    seasons = [int(f.stem) for f in files]
    missing = set(range(2020, last_season + 1)) - set(seasons)
    if missing:
        raise FileNotFoundError(f"Missing transactions for {sorted(missing)}")
    use = [f for f in files if int(f.stem) <= last_season]
    frames = [parse_placements(json.loads(f.read_text())["transactions"]) for f in use]
    return pd.concat(frames, ignore_index=True).drop_duplicates(), {str(f): file_hash(f) for f in use}


def label_rows(rows, placements):
    """Attach labels, injury history and the already-injured exclusion to each forecast row."""
    r = rows[KEY + ["days_since_last"]].copy()
    m = r.merge(placements, on="pitcher", how="left")
    days = (m.event - m.forecast_date).dt.days
    flags = pd.DataFrame({"pitcher": m.pitcher, "forecast_date": m.forecast_date})
    flags["already"] = (days < 0) & (days >= -m.days_since_last)
    flags["il_365"] = ((days < 0) & (days >= -365)).astype(float)
    flags["il_prev_season"] = (m.event.dt.year == m.forecast_date.dt.year - 1).astype(float)
    for name, (lo, hi, gap) in LABELS.items():
        flags[f"y_{name}"] = days.between(lo, hi)
        flags[f"gap_{name}"] = days.between(*gap) if gap else False
    agg = flags.groupby(KEY).agg(already=("already", "any"), il_365=("il_365", "sum"),
                                 il_prev_season=("il_prev_season", "max"),
                                 **{f"y_{n}": (f"y_{n}", "any") for n in LABELS},
                                 **{f"gap_{n}": (f"gap_{n}", "any") for n in LABELS}).reset_index()
    out = rows.merge(agg, on=KEY, how="left", validate="one_to_one")
    return out[~out.already].reset_index(drop=True)


def model_columns(columns):
    sets = feature_sets(list(columns))
    w = sets["B"] + HISTORY
    changes = [c for c in sets["C"] if c not in sets["B"]]
    return {"W": w, "V": w + ["chg_fb_velo"], "X": w + changes + PT_FEATURES}


def with_features(rows, changes, prior_seasons):
    feats = per_type_features(changes, fit_priors(changes, prior_seasons))
    return rows.merge(feats, on=KEY, how="left", validate="one_to_one")


def _fit(kind, cols, train, ycol, setting, fixed, seed):
    design = Design().fit(train[cols])
    X, y = design.transform(train[cols]), train[ycol].to_numpy(float)
    ones = np.ones(len(y))
    model = (binomial_fit(X, y, ones, setting) if kind == "linear"
             else _hgb_fit(X, y, ones, setting, fixed, seed))
    return lambda rows: model.predict_proba(design.transform(rows[cols]))[:, 1]


def _loss(p, y):
    return row_loss(p, y, np.ones(len(y))).mean()


def label_frame(d, name):
    return d[~d[f"gap_{name}"]]


def run_fold(rows, changes, train, test, tune, settings, design, fixed):
    """Predict one test season for both labels. `tune` = (tune_train, tune_valid) or None with `settings`."""
    models = model_columns(rows.columns)
    specs = {**{m: ("linear", cols) for m, cols in models.items()},
             "NW": ("hgb", models["W"]), "NX": ("hgb", models["X"])}
    final = with_features(rows[rows.season.isin(train + [test])], changes, train)
    chosen, preds = {}, []
    if tune:
        inner = with_features(rows[rows.season.isin(tune[0] + [tune[1]])], changes, tune[0])
    for name in LABELS:
        ycol = f"y_{name}"
        fit_rows = label_frame(final[final.season.isin(train)], name)
        test_rows = label_frame(final[final.season == test], name)
        pred = test_rows[KEY + ["season", ycol]].rename(columns={ycol: "y"}).assign(label=name)
        for model, (kind, cols) in specs.items():
            if tune:
                it, iv = label_frame(inner[inner.season.isin(tune[0])], name), label_frame(inner[inner.season == tune[1]], name)
                grid = C_GRID if kind == "linear" else ITER_GRID
                scores = [_loss(_fit(kind, cols, it, ycol, s, fixed, design["seed"])(iv), iv[ycol].to_numpy(float))
                          for s in grid]
                setting = grid[int(np.argmin(scores))]
            else:
                setting = settings[name][model]
            chosen.setdefault(name, {})[model] = setting
            pred[f"p_{model}"] = _fit(kind, cols, fit_rows, ycol, setting, fixed, design["seed"])(test_rows)
        preds.append(pred)
    return pd.concat(preds, ignore_index=True), chosen


def _cells(pred):
    """Per-season cells, then both evaluation seasons pooled."""
    pooled = pred.assign(season="pooled")
    return [*pred.groupby(["label", "season"]), *pooled.groupby(["label", "season"])]


def summarize(pred):
    rows = []
    for (label, season), g in _cells(pred):
        y = g.y.to_numpy(float)
        ones = np.ones(len(y))
        losses = {m: row_loss(g[f"p_{m}"].to_numpy(), y, ones) for m in ("W", "V", "X", "NW", "NX")}
        for a, b in COMPARISONS:
            est, lo, hi = clustered_ci(losses[a] - losses[b], ones, g.pitcher.to_numpy())
            rows.append({"label": label, "season": str(season), "comparison": f"{a}-{b}", "rows": len(g),
                         "events": int(y.sum()), "pitchers": g.pitcher.nunique(),
                         "loss_a": losses[a].mean(), "loss_b": losses[b].mean(),
                         "diff": est, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def ranking(pred):
    rows = []
    for (label, season), g in _cells(pred):
        for m in ("W", "V", "X", "NW", "NX"):
            p = g[f"p_{m}"]
            top = g[p >= p.quantile(0.9)]
            rows.append({"label": label, "season": str(season), "model": m, "auc": roc_auc_score(g.y, p),
                         "base_rate": g.y.mean(), "top10_rate": top.y.mean(), "mean_pred": p.mean()})
    return pd.DataFrame(rows)


def verdict(summary):
    r = summary[(summary.label == "primary") & (summary.season == "pooled") & (summary.comparison == "X-W")].iloc[0]
    v = "supported" if r.ci_high < 0 else "harmful" if r.ci_low > 0 else "not supported"
    return {"changes_warn": {"diff": r["diff"], "lo": r.ci_low, "hi": r.ci_high, "verdict": v}}


def write_report(summary, ranks, settings, stage, out, result=None):
    title = "development (2023-2024 folds)" if stage == "develop" else "evaluation (2025-2026)"
    lines = [f"# Injury warning: {title}", "",
             "Log-loss differences per forecast with 95% pitcher-bootstrap intervals; negative favors the "
             "first model. Primary label: injured-list placement 0-29 days after the forecast date. "
             "Lead-time label: 7-36 days, rows with a placement in days 0-6 removed.", ""]
    if result:
        v = result["changes_warn"]
        lines += [f"**Pre-registered claim (X-W, primary label, pooled): {v['verdict']}**, "
                  f"{v['diff']:+.6f} ({v['lo']:+.6f} to {v['hi']:+.6f})", ""]
    lines += ["| Label | Season | Comparison | Rows | Events | Difference (95% CI) |", "|---|---|---|---:|---:|---|"]
    for r in summary.itertuples():
        lines.append(f"| {r.label} | {r.season} | {r.comparison} | {r.rows:,} | {r.events:,} | "
                     f"{r.diff:+.6f} ({r.ci_low:+.6f} to {r.ci_high:+.6f}) |")
    lines += ["", "## Ranking (descriptive)", "", "| Label | Season | Model | AUC | Base rate | Top-10% rate |",
              "|---|---|---|---:|---:|---:|"]
    for r in ranks.itertuples():
        lines.append(f"| {r.label} | {r.season} | {r.model} | {r.auc:.3f} | {r.base_rate:.1%} | {r.top10_rate:.1%} |")
    lines += ["", "## Settings", "", "```json", json.dumps(settings, indent=2), "```", ""]
    (out / "report.md").write_text("\n".join(lines))


def prepare(last_season):
    placements, raw = load_placements(last_season)
    rows = pd.read_parquet(ROWS)
    rows = rows[rows.season <= last_season].copy()
    rows["month"] = rows.forecast_date.dt.month
    return label_rows(rows, placements), raw


def develop(design, fixed):
    hashes = {p: file_hash(p) for p in SOURCES}
    load_frozen_spec()
    rows, raw = prepare(2024)
    changes = pd.read_parquet(DEV_CHANGES)
    preds, settings = [], {}
    for fold in design["development_folds"]:
        print(f"Fold test {fold['test']}", flush=True)
        pred, chosen = run_fold(rows, changes, fold["train"], fold["test"],
                                (fold["tune_train"], fold["tune_valid"]), None, design, fixed)
        preds.append(pred)
        settings[str(fold["test"])] = chosen
    return pd.concat(preds, ignore_index=True), settings, hashes | raw, None


def evaluate(design, fixed):
    from pcf.evaluate_2026 import new_changes
    record = json.loads(FREEZE.read_text())
    hashes = {p: file_hash(p) for p in SOURCES}
    load_frozen_spec()
    rows, raw = prepare(2026)
    dev = pd.read_parquet(DEV_CHANGES)
    print("Measuring 2025-2026 per-type changes", flush=True)
    fresh, _ = new_changes([2025, 2026])
    changes = pd.concat([dev[dev.season <= 2024], fresh], ignore_index=True)
    preds = []
    for test in (2025, 2026):
        print(f"Scoring {test}", flush=True)
        pred, _ = run_fold(rows, changes, list(range(2021, test)), test, None, record["settings"], design, fixed)
        preds.append(pred)
    return pd.concat(preds, ignore_index=True), record["settings"], hashes | raw, True


def freeze():
    if FREEZE.exists():
        raise SystemExit("Already frozen")
    if EVAL_OUT.exists():
        raise SystemExit("Evaluation outputs exist")
    if any(int(f.stem) >= 2025 for f in TRANSACTIONS.glob("*.json")):
        raise SystemExit("2025+ transactions are already downloaded; the freeze would not precede them")
    settings = json.loads((DEV_OUT / "settings.json").read_text())["2024"]
    record = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "settings": settings, "settings_source": "development fold testing 2024",
              "source_sha256": {p: file_hash(p) for p in SOURCES},
              "development_report_sha256": file_hash(DEV_OUT / "report.md")}
    FREEZE.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Frozen: {FREEZE}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["develop", "freeze", "download", "evaluate"])
    parser.add_argument("--seasons", type=int, nargs="+", default=list(range(2020, 2025)))
    args = parser.parse_args()
    if args.stage == "download":
        if any(s >= 2025 for s in args.seasons) and not FREEZE.exists():
            parser.error("freeze before downloading evaluation-season transactions")
        return download(args.seasons)
    if args.stage == "freeze":
        return freeze()
    design, fixed = json.loads(DESIGN.read_text()), json.loads(NONLINEAR.read_text())["nonlinear"]["fixed"]
    out = DEV_OUT if args.stage == "develop" else EVAL_OUT
    if out.exists():
        parser.error(f"{out} exists; each stage runs once")
    if args.stage == "develop":
        if FREEZE.exists():
            parser.error("frozen; development cannot rerun")
        pred, settings, hashes, final = develop(design, fixed)
    else:
        if not FREEZE.exists():
            parser.error("freeze first")
        changed = [p for p, h in json.loads(FREEZE.read_text())["source_sha256"].items() if file_hash(p) != h]
        if changed:
            parser.error(f"sources changed after freezing: {changed}")
        if file_hash(DEV_OUT / "report.md") != json.loads(FREEZE.read_text())["development_report_sha256"]:
            parser.error("development report changed after freezing")
        pred, settings, hashes, final = evaluate(design, fixed)
    summary, ranks = summarize(pred), ranking(pred)
    result = verdict(summary) if final else None
    for path, expected in hashes.items():
        if file_hash(path) != expected:
            raise RuntimeError(f"Input changed during the run: {path}; results were not saved")
    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps(
        {"experiment": design["experiment"], "stage": args.stage,
         "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "input_sha256": hashes,
         "versions": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__}},
        indent=2) + "\n")
    (out / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    pred.to_parquet(out / "predictions.parquet", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    ranks.to_csv(out / "ranking.csv", index=False)
    if result:
        (out / "verdict.json").write_text(json.dumps(result, indent=2) + "\n")
    write_report(summary, ranks, settings, args.stage, out, result)
    print((out / "report.md").read_text())


if __name__ == "__main__":
    main()
