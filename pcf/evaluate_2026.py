"""Score the frozen development comparisons once on 2026.

    python3 -m pcf.evaluate_2026 --rehearsal   # pipeline check on 2025 (already inspected)
    python3 -m pcf.evaluate_2026 --freeze      # record hashes; refuses once 2026 data exists
    python3 -m pcf.ingest --seasons 2026
    python3 -m pcf.evaluate_2026               # the single 2026 scoring run
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from pcf.changes import COLUMNS, SEED, aggregate_outings, build_changes, eligible_forecasts
from pcf.dataset import build
from pcf.forecast_value import (KEY, PT_FEATURES, Design, _hgb_fit, directional, fit_priors,
                                per_type_features)
from pcf.freeze import file_hash, load_frozen_spec
from pcf.model import ROWS, TARGETS, RepertoireSupport, binomial_fit, calibration, clustered_ci, feature_sets, row_loss

DESIGN = Path("experiments/evaluation_2026.json")
FREEZE = Path("experiments/evaluation_2026_freeze.json")
DEV_CHANGES = Path("artifacts/experiments/per_type_changes/changes.parquet")
RAW = Path("data/raw/statcast")
ROWS_2026 = Path("data/processed/forecast_rows_2026.parquet")
REHEARSAL = Path("artifacts/experiments/rehearsal_2025")
OUT = Path("artifacts/experiments/evaluation_2026")
ZONE = ["zone_rate_cur", "chg_zone_rate"]
LINEAR_C = 0.001
HGB_ITER = 100
COMPARISONS = [("P", "B"), ("V", "B"), ("P", "V"), ("NP", "NB"), ("B", "A"), ("C", "B"),
               ("Pnz", "Bnz"), ("Bnz", "B")]
SOURCES = ["pcf/evaluate_2026.py", "pcf/forecast_value.py", "pcf/model.py", "pcf/dataset.py",
           "pcf/changes.py", "pcf/ingest.py", "pcf/freeze.py", str(DESIGN), "pyproject.toml", "uv.lock",
           "research_spec.json", "experiments/forecast_value.json", str(DEV_CHANGES), str(ROWS)]


def model_columns(columns):
    sets = feature_sets(list(columns))
    p = sets["B"] + PT_FEATURES
    return {"A": sets["A"], "B": sets["B"], "C": sets["C"], "V": sets["B"] + ["chg_fb_velo"], "P": p,
            "Bnz": [c for c in sets["B"] if c not in ZONE], "Pnz": [c for c in p if c not in ZONE]}


def load_rows(test_season):
    saved = pd.read_parquet(ROWS, filters=[("season", "<=", 2025)])
    if test_season == 2025:
        return saved
    rows = build(through_season=test_season)
    order = lambda f: f.sort_values(KEY).reset_index(drop=True)
    old = order(rows[rows.season <= 2025])[saved.columns]
    pd.testing.assert_frame_equal(old, order(saved), check_dtype=False, check_exact=False, rtol=1e-9)
    ROWS_2026.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(ROWS_2026, index=False)
    return rows


def new_changes(seasons):
    files = sorted(f for f in RAW.glob("*/*.parquet") if min(seasons) - 1 <= int(f.parent.name) <= max(seasons))
    pitches = pd.concat((pd.read_parquet(f, columns=COLUMNS) for f in files), ignore_index=True)
    pitches = pitches[pitches.game_type == "R"].drop_duplicates(["game_pk", "at_bat_number", "pitch_number"])
    pitches["season"] = pd.to_datetime(pitches.game_date).dt.year
    labels = (pitches[pitches.season.isin(seasons)].assign(pitch_type=lambda p: p.pitch_type.fillna("UN"))
              .groupby("season").pitch_type.value_counts(normalize=True).rename("share").reset_index())
    forecasts = eligible_forecasts(pitches)
    forecasts = forecasts[pd.to_datetime(forecasts.forecast_date).dt.year.isin(seasons)]
    outings = aggregate_outings(pitches)
    del pitches
    return build_changes(outings, forecasts, n_boot=400, seed=SEED,
                         seasons=(min(seasons), max(seasons))), labels


def predict(train, test, cols, ycol, kind, design):
    transform = Design().fit(train[cols])
    X_tr, X_te = transform.transform(train[cols]), transform.transform(test[cols])
    k, n = train[ycol].to_numpy(), train.y_bf.to_numpy()
    if kind == "linear":
        return binomial_fit(X_tr, k, n, LINEAR_C).predict_proba(X_te)[:, 1]
    return _hgb_fit(X_tr, k, n, HGB_ITER, design["nonlinear"]["fixed"], design["seed"]).predict_proba(X_te)[:, 1]


def evaluate(rows, changes, train_seasons, test_season, nonlinear_design):
    rows = rows.copy()
    rows["month"] = rows.forecast_date.dt.month
    priors = fit_priors(changes, train_seasons)
    features = per_type_features(changes, priors)
    d = rows[rows.season.isin(train_seasons + [test_season])].merge(features, on=KEY, how="left",
                                                                      validate="one_to_one")
    test_keys = set(map(tuple, d.loc[d.season == test_season, KEY].astype(str).to_numpy()))
    change_keys = set(map(tuple, changes.loc[changes.season == test_season, KEY].drop_duplicates()
                          .astype(str).to_numpy()))
    if test_keys != change_keys:
        raise ValueError(f"Per-type forecasts differ from study rows: {len(test_keys ^ change_keys)} keys")
    models = model_columns(rows.columns)
    eligible_train, eligible_test = d[d.season.isin(train_seasons)], d[d.season == test_season]
    unsupported = RepertoireSupport().fit(eligible_train).flags(eligible_test).any(axis=1).to_numpy()
    train = eligible_train[eligible_train.y_bf > 0]
    keep = eligible_test.y_bf.gt(0).to_numpy()
    test, unsupported = eligible_test[keep], unsupported[keep]
    pred = test[KEY + ["season", "name", "team", "y_bf", "y_k", "y_bb", "start_share", *ZONE, *PT_FEATURES]].copy()
    pred["repertoire_fallback"] = unsupported
    for target, ycol in TARGETS.items():
        for name, cols in models.items():
            pred[f"p_{target}_{name}"] = predict(train, test, cols, ycol, "linear", nonlinear_design)
        for name, cols in (("NB", models["B"]), ("NP", models["P"])):
            pred[f"p_{target}_{name}"] = predict(train, test, cols, ycol, "hgb", nonlinear_design)
        base = pred[f"p_{target}_A"].to_numpy()
        for name in [m for m in [*models, "NB", "NP"] if m != "A"]:
            pred[f"p_{target}_{name}"] = np.where(unsupported, base, pred[f"p_{target}_{name}"])
    meta = {"train_seasons": train_seasons, "test_season": test_season, "eligible": len(eligible_test),
            "no_followup": int((~keep).sum()), "scored": len(test), "pitchers": int(test.pitcher.nunique()),
            "fallback": int(unsupported.sum()),
            "feature_coverage": {c: float(eligible_test[c].notna().mean()) for c in PT_FEATURES},
            "prior_sources": priors.source.value_counts().to_dict()}
    return pred, meta, d


def comparisons(pred):
    rows = []
    for target, ycol in TARGETS.items():
        k, n = pred[ycol].to_numpy(), pred.y_bf.to_numpy()
        losses = {}
        for a, b in COMPARISONS:
            for m in (a, b):
                losses.setdefault(m, row_loss(pred[f"p_{target}_{m}"].to_numpy(), k, n))
            est, lo, hi = clustered_ci(losses[a] - losses[b], n, pred.pitcher.to_numpy())
            share, moved = directional(pred[f"p_{target}_{a}"].to_numpy(), pred[f"p_{target}_{b}"].to_numpy(), k, n)
            rows.append({"target": target, "comparison": f"{a}-{b}", "loss_new": losses[a].sum() / n.sum(),
                         "loss_old": losses[b].sum() / n.sum(), "diff_per_bf": est, "ci_low": lo,
                         "ci_high": hi, "toward_share": share, "rows_moved": moved})
    return pd.DataFrame(rows)


def audits(pred, d, labels):
    seasons = d[d.y_bf > 0].groupby("season").agg(k=("y_k", "sum"), bb=("y_bb", "sum"), bf=("y_bf", "sum"),
                                                  zone_rate_cur=("zone_rate_cur", "mean"))
    seasons["k_rate"], seasons["bb_rate"] = seasons.k / seasons.bf, seasons.bb / seasons.bf
    totals = []
    for target, ycol in TARGETS.items():
        observed = pred[ycol].sum()
        for m in [c.removeprefix(f"p_{target}_") for c in pred.columns if c.startswith(f"p_{target}_")]:
            expected = float((pred[f"p_{target}_{m}"] * pred.y_bf).sum())
            totals.append({"target": target, "model": m, "expected": expected, "observed": int(observed),
                           "expected_minus_observed_pct": 100 * (expected - observed) / observed})
    deciles = []
    for target, ycol in TARGETS.items():
        for m in ("A", "B", "P"):
            c = calibration(pred[f"p_{target}_{m}"].to_numpy(), pred[ycol].to_numpy(), pred.y_bf.to_numpy())
            deciles.append(c.assign(target=target, model=m, decile=np.arange(1, len(c) + 1)))
    return seasons.reset_index(), pd.DataFrame(totals), pd.concat(deciles, ignore_index=True), labels


def write_report(summary, meta, seasons, totals, labels, out, title):
    t = meta["test_season"]
    lines = [f"# {title}", "",
             f"Training {meta['train_seasons'][0]}-{meta['train_seasons'][-1]}; test {t}. "
             f"{meta['eligible']:,} eligible forecasts, {meta['no_followup']:,} without follow-up, "
             f"{meta['scored']:,} scored across {meta['pitchers']:,} pitchers; {meta['fallback']:,} repertoire "
             "fallbacks to A. Negative differences favor the first model.", "",
             "| Target | Comparison | Loss new | Loss old | Difference per BF (95% pitcher CI) | Moves toward outcome |",
             "|---|---|---:|---:|---|---:|"]
    for r in summary.itertuples():
        lines.append(f"| {r.target.upper()}% | {r.comparison} | {r.loss_new:.5f} | {r.loss_old:.5f} | "
                     f"{r.diff_per_bf:+.6f} ({r.ci_low:+.6f} to {r.ci_high:+.6f}) | {r.toward_share:.1%} |")
    lines += ["", "## Audit: forecast-window rates by season", "", "| Season | K% | BB% | Mean zone rate |",
              "|---:|---:|---:|---:|"]
    for r in seasons.itertuples():
        lines.append(f"| {r.season} | {r.k_rate:.2%} | {r.bb_rate:.2%} | {r.zone_rate_cur:.3f} |")
    lines += ["", "## Audit: expected minus observed totals", "", "| Target | Model | Expected minus observed |",
              "|---|---|---:|"]
    for r in totals.itertuples():
        lines.append(f"| {r.target.upper()}% | {r.model} | {r.expected_minus_observed_pct:+.2f}% |")
    lines += ["", "## Audit: per-type coverage", ""]
    lines += [f"- {k}: {v:.1%}" for k, v in meta["feature_coverage"].items()]
    shares = labels.pivot_table(index="pitch_type", columns="season", values="share").fillna(0)
    lines += ["", "## Audit: pitch-type label shares", "",
              "| Type | " + " | ".join(map(str, shares.columns)) + " |", "|---|" + "---:|" * len(shares.columns)]
    for pitch_type, r in shares.sort_values(shares.columns[-1], ascending=False).iterrows():
        lines.append(f"| {pitch_type} | " + " | ".join(f"{v:.2%}" for v in r) + " |")
    lines.append("")
    (out / "report.md").write_text("\n".join(lines))


def _hashes():
    return {p: file_hash(p) for p in SOURCES}


def run(test_season, out, title):
    design = json.loads(DESIGN.read_text())
    nonlinear = json.loads(Path("experiments/forecast_value.json").read_text())
    hashes = _hashes()
    load_frozen_spec()
    train_seasons = [s for s in design["train_seasons"] if s < test_season]
    rows = load_rows(test_season)
    dev = pd.read_parquet(DEV_CHANGES)
    print(f"Measuring per-type changes for {list(range(2025, test_season + 1))}", flush=True)
    fresh, labels = new_changes(list(range(2025, test_season + 1)))
    changes = pd.concat([dev[dev.season <= 2024], fresh], ignore_index=True)
    print("Fitting and scoring", flush=True)
    pred, meta, d = evaluate(rows, changes, train_seasons, test_season, nonlinear)
    summary = comparisons(pred)
    seasons, totals, deciles, labels = audits(pred, d, labels)
    if _hashes() != hashes:
        raise RuntimeError("Inputs changed during the run; results were not saved")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(
        {"design": design, "meta": meta, "input_sha256": hashes,
         "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
         "versions": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__}},
        indent=2, default=str) + "\n")
    pred.to_parquet(out / "predictions.parquet", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    totals.to_csv(out / "totals.csv", index=False)
    deciles.to_csv(out / "calibration.csv", index=False)
    seasons.to_csv(out / "seasons.csv", index=False)
    labels.to_csv(out / "labels.csv", index=False)
    write_report(summary, meta, seasons, totals, labels, out, title)
    print((out / "report.md").read_text())


def freeze():
    if not (REHEARSAL / "report.md").exists():
        raise SystemExit("Run --rehearsal before freezing")
    if FREEZE.exists():
        raise SystemExit("The 2026 evaluation is already frozen")
    if (RAW / "2026").exists() and any((RAW / "2026").iterdir()):
        raise SystemExit("2026 data is already present; the evaluation can no longer be frozen cleanly")
    record = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "source_sha256": _hashes(),
              "rehearsal_report_sha256": file_hash(REHEARSAL / "report.md")}
    FREEZE.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Frozen: {FREEZE}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--rehearsal", action="store_true")
    mode.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        return freeze()
    out = REHEARSAL if args.rehearsal else OUT
    if out.exists() and any(out.iterdir()):
        parser.error(f"{out} already holds results; this run happens once")
    if args.rehearsal:
        if FREEZE.exists():
            parser.error("rehearsal must precede the freeze")
        return run(2025, out, "Rehearsal on 2025 (already inspected; descriptive)")
    if not FREEZE.exists():
        parser.error("freeze the evaluation first")
    frozen = json.loads(FREEZE.read_text())["source_sha256"]
    changed = [p for p, h in frozen.items() if file_hash(p) != h]
    if changed:
        parser.error(f"sources changed after freezing: {changed}")
    if not list((RAW / "2026").glob("*.parquet")):
        parser.error("download 2026 first: python3 -m pcf.ingest --seasons 2026")
    run(2026, out, "2026 temporal and domain test")


if __name__ == "__main__":
    main()
