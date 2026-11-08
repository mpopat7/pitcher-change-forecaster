"""Development-only pitch-type measurements; no future outcomes or model fitting.

Run: python3 -m pcf.changes
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pcf.dataset import ELIG_BF, ELIG_LAST_SEEN, ELIG_WINDOW, forecast_dates
from pcf.freeze import file_hash, load_frozen_spec

RAW = Path("data/raw/statcast")
OUT = Path("artifacts/experiments/per_type_changes")
METRICS = {"velocity_mph": "release_speed", "ivb_in": "ivb", "hb_in": "hb"}
COLUMNS = ["game_type", "game_pk", "game_date", "at_bat_number", "pitch_number",
           "pitcher", "pitch_type", "player_name", "p_throws", "pfx_x", "pfx_z",
           "release_speed", "inning_topbot", "home_team", "away_team", "events"]
SEED = 20261108
SOURCES = ["pcf/changes.py", "pcf/dataset.py", "pcf/freeze.py", "pyproject.toml", "uv.lock"]


def aggregate_outings(pitches):
    p = pitches.loc[pitches.game_type == "R"].drop_duplicates(
        ["game_pk", "at_bat_number", "pitch_number"]).copy()
    p["game_date"] = pd.to_datetime(p.game_date)
    p["season"] = p.game_date.dt.year
    p["pitch_type"] = p.pitch_type.fillna("UN")
    p["pitch_team"] = np.where(p.inning_topbot == "Top", p.home_team, p.away_team)
    p["ivb"] = p.pfx_z * 12
    arm = p.p_throws.map({"R": -1.0, "L": 1.0})
    p["hb"] = p.pfx_x * 12 * arm
    aggregations = {"n_pitches": ("pitcher", "size")}
    for metric, column in METRICS.items():
        aggregations[f"{metric}_sum"] = (column, "sum")
        aggregations[f"{metric}_count"] = (column, "count")
    return (p.groupby(["pitcher", "game_pk", "game_date", "season", "pitch_team", "pitch_type"])
            .agg(**aggregations).reset_index())


def eligible_forecasts(pitches):
    """Reconstruct the study population from past BF, without loading saved outcomes."""
    p = pitches.copy()
    p["game_date"] = pd.to_datetime(p.game_date)
    p["season"] = p.game_date.dt.year
    p["pitch_team"] = np.where(p.inning_topbot == "Top", p.home_team, p.away_team)
    pa = p[p.events.notna() & ~p.events.isin(["truncated_pa", "intent_walk"])]
    daily = pa.groupby(["pitcher", "game_date"]).size().rename("bf").reset_index()
    appearances = (p.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"])
                   .drop_duplicates(["pitcher", "game_pk"], keep="last"))
    rows = []
    for cutoff in forecast_dates(p):
        past = daily[(daily.game_date < cutoff)
                     & (daily.game_date >= cutoff - pd.Timedelta(days=ELIG_WINDOW))]
        counts = past.groupby("pitcher").agg(bf=("bf", "sum"), last=("game_date", "max"))
        ids = counts.index[(counts.bf >= ELIG_BF)
                           & (counts["last"] >= cutoff - pd.Timedelta(days=ELIG_LAST_SEEN))]
        context = (appearances[appearances.game_date < cutoff].drop_duplicates("pitcher", keep="last")
                   .set_index("pitcher").reindex(ids))
        f = pd.DataFrame({"pitcher": ids, "forecast_date": cutoff, "season": cutoff.year,
                          "team": context.pitch_team.to_numpy(),
                          "name": context.player_name.to_numpy()})
        rows.append(f)
    return pd.concat(rows, ignore_index=True)


def _bootstrap_means(sums, counts, rng, n_boot):
    valid = counts > 0
    sums, counts = sums[valid], counts[valid]
    if len(counts) < 2:
        return None
    weights = rng.multinomial(len(counts), np.full(len(counts), 1 / len(counts)), size=n_boot)
    return (weights @ sums) / (weights @ counts)


def _window_summary(window, types, rng, n_boot):
    summary = {}
    by_type = dict(tuple(window.groupby("pitch_type")))
    all_outings = window.groupby("game_pk").n_pitches.sum()
    n_all = int(all_outings.sum())
    weights = None
    if len(all_outings) >= 2:
        weights = rng.multinomial(len(all_outings), np.full(len(all_outings), 1 / len(all_outings)),
                                  size=n_boot)
        usage_den = weights @ all_outings.to_numpy(dtype=float)
    for pitch_type in types:
        p = by_type.get(pitch_type, window.iloc[:0])
        type_n = int(p.n_pitches.sum())
        usage_draws = None
        if weights is not None:
            per_game = p.groupby("game_pk").n_pitches.sum().reindex(all_outings.index, fill_value=0)
            usage_draws = (weights @ per_game.to_numpy(dtype=float)) / usage_den
        summary[(pitch_type, "usage")] = {
            "mean": type_n / n_all if n_all else np.nan,
            "n": n_all, "outings": len(all_outings), "type_n": type_n,
            "type_outings": p.game_pk.nunique(), "draws": usage_draws,
        }
        for metric in METRICS:
            counts = p[f"{metric}_count"].to_numpy(dtype=float)
            sums = p[f"{metric}_sum"].to_numpy(dtype=float)
            n = int(counts.sum())
            summary[(pitch_type, metric)] = {
                "mean": sums.sum() / n if n else np.nan,
                "n": n, "outings": int((counts > 0).sum()), "type_n": type_n,
                "type_outings": p.game_pk.nunique(),
                "draws": _bootstrap_means(sums, counts, rng, n_boot),
            }
    return summary


def _row_seed(seed, pitcher, cutoff):
    key = f"{seed}/{pitcher}/{cutoff.date()}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little")


def build_changes(outings, forecasts, *, n_boot=400, seed=SEED):
    if n_boot < 100:
        raise ValueError("Use at least 100 bootstrap draws")
    if not forecasts.season.between(2021, 2024).all():
        raise ValueError("This experiment is restricted to development seasons 2021-2024")
    by_pitcher = dict(tuple(outings.groupby("pitcher")))
    rows = []
    for f in forecasts.sort_values(["forecast_date", "pitcher"]).itertuples(index=False):
        cutoff = pd.Timestamp(f.forecast_date)
        if cutoff.year != f.season:
            raise ValueError("Forecast season must match its date")
        p = by_pitcher[f.pitcher]
        current_season = p[(p.season == f.season) & (p.game_date < cutoff)]
        recent = current_season[current_season.game_date >= cutoff - pd.Timedelta(days=30)]
        earlier = current_season[(current_season.game_date >= cutoff - pd.Timedelta(days=120))
                                 & (current_season.game_date < cutoff - pd.Timedelta(days=30))]
        prior = p[p.season == f.season - 1]
        source = "earlier_window"
        baseline = earlier
        if earlier.game_pk.nunique() < 2 and not prior.empty:
            baseline, source = prior, "prior_season"
        types = sorted(set(recent.pitch_type) | set(baseline.pitch_type))
        rng = np.random.default_rng(_row_seed(seed, f.pitcher, cutoff))
        cur = _window_summary(recent, types, rng, n_boot)
        base = _window_summary(baseline, types, rng, n_boot)
        for pitch_type in types:
            nc, nb = cur[(pitch_type, "usage")]["type_n"], base[(pitch_type, "usage")]["type_n"]
            status = "both" if nc and nb else "recent_only" if nc else "baseline_only"
            for metric in (*METRICS, "usage"):
                c, b = cur[(pitch_type, metric)], base[(pitch_type, metric)]
                lo = hi = np.nan
                # Absent types still have meaningful zero usage, but no physical mean.
                # Suppress boundary intervals: resampling cannot discover an unseen type.
                supported = metric != "usage" or (0 < nc < c["n"] and 0 < nb < b["n"])
                if supported and c["draws"] is not None and b["draws"] is not None:
                    lo, hi = np.quantile(c["draws"] - b["draws"], [0.025, 0.975])
                rows.append({
                    "pitcher": f.pitcher, "name": f.name, "team": f.team,
                    "season": f.season, "forecast_date": cutoff,
                    "pitch_type": pitch_type, "metric": metric, "type_status": status,
                    "baseline_source": source,
                    "recent_first": recent.game_date.min(), "recent_last": recent.game_date.max(),
                    "baseline_first": baseline.game_date.min(), "baseline_last": baseline.game_date.max(),
                    "recent": c["mean"], "baseline": b["mean"],
                    "delta": c["mean"] - b["mean"], "ci_low": lo, "ci_high": hi,
                    "n_recent": c["n"], "n_baseline": b["n"],
                    "outings_recent": c["outings"], "outings_baseline": b["outings"],
                    "type_pitches_recent": nc, "type_pitches_baseline": nb,
                    "type_outings_recent": c["type_outings"], "type_outings_baseline": b["type_outings"],
                    "label_caveat": pitch_type in {"SL", "ST", "UN"},
                })
    return pd.DataFrame(rows)


def interval_widths(changes):
    d = changes.copy()
    d["min_pitches"] = d[["n_recent", "n_baseline"]].min(axis=1)
    d["count_bin"] = pd.cut(d.min_pitches, [-1, 0, 24, 49, 99, 199, np.inf],
                            labels=["0", "1-24", "25-49", "50-99", "100-199", "200+"])
    d["width"] = d.ci_high - d.ci_low
    return (d.groupby(["metric", "count_bin"], observed=True)
            .agg(rows=("delta", "size"), measured_deltas=("delta", "count"),
                 intervals=("width", "count"), median_interval_width=("width", "median"))
            .reset_index())


def write_report(changes, forecasts, out):
    widths = interval_widths(changes)
    widths.to_csv(out / "interval_widths.csv", index=False)
    teams = changes[changes.team.isin(["CIN", "DET"])]
    teams.to_csv(out / "cin_det_changes.csv", index=False)
    usage = changes[changes.metric == "usage"]
    physical = changes[changes.metric != "usage"]
    coverage = {
        "forecasts": len(forecasts), "pitchers": int(forecasts.pitcher.nunique()),
        "forecast_dates": int(forecasts.forecast_date.nunique()), "measurement_rows": len(changes),
        "pitch_type_rows": len(usage), "unknown_type_rows": int((usage.pitch_type == "UN").sum()),
        "type_status_counts": usage.type_status.value_counts().to_dict(),
        "baseline_source_counts": usage.baseline_source.value_counts().to_dict(),
        "physical_rows_with_delta": int(physical.delta.notna().sum()),
        "physical_rows_with_interval": int(physical.ci_low.notna().sum()),
        "forecast_counts_by_season": forecasts.groupby("season").size().to_dict(),
    }
    (out / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
    lines = ["# Development pitch-type changes", "",
             f"{len(forecasts):,} eligible pitcher forecasts; {forecasts.pitcher.nunique():,} pitchers; "
             f"{len(usage):,} pitch-type comparisons. No future outcomes are used to calculate "
             "changes or select eligible pitchers; no prediction targets are computed.", "",
             "These are unshrunk descriptive measurements. Reliability, synthetic-null calibration, "
             "persistence, and forecast relevance have not been tested.", "",
             "Recent pitches are from days 1-30 before each forecast. The baseline uses days 31-120 "
             "in that season when at least two outings are available; otherwise it uses the prior "
             "season, or sparse earlier data when no prior season exists. This outing rule permits "
             "resampling; it is not a validated reliability threshold. Baseline dates and source are "
             "included in every row. This policy differs from the original pooled-pitch fallback.", "",
             "Means weight pitches equally. Physical intervals independently resample type-bearing "
             "outings with valid measurements in each window; usage resamples all outings with their "
             "whole repertoires. Intervals are pointwise percentile bootstrap summaries conditional on "
             "observed outings, not simultaneous tests or evidence of a skill change. One-outing "
             "comparisons and usage at zero or one have no interval. Few outings and dependence "
             "between outings can make intervals unreliable. Weekly windows overlap.", "",
             "Usage includes every recorded pitch in the denominator, including unknown types and "
             "missing velocity. Physical counts include only valid measurements for that metric. "
             "Movement is in inches; positive horizontal movement is arm-side for either hand. "
             "Batter mix and pitch reclassification can affect means. SL/ST and unknown labels carry "
             "a review caveat; no automatic reclassification is inferred.", "",
             "See `interval_widths.csv` for count-dependent interval widths and availability. "
             "This is a description of bootstrap output, not a reliability calibration.", "",
             "## Detroit and Cincinnati examples", "",
             "The full team subset is in `cin_det_changes.csv`, including types seen in only one "
             "window. The examples below select the largest absolute velocity differences among "
             "comparisons with intervals, with at most one example per pitcher per team. They "
             "illustrate measurements, not persistence or performance gains.", ""]
    for team in ("DET", "CIN"):
        candidates = teams[(teams.team == team) & (teams.metric == "velocity_mph")
                           & teams.ci_low.notna()].copy()
        candidates["abs_delta"] = candidates.delta.abs()
        candidates = candidates.sort_values("abs_delta", ascending=False).drop_duplicates("pitcher").head(4)
        lines.extend([f"### {team}", "", "| Pitcher | Date | Type | Delta mph | 95% interval | Pitches recent / baseline | Outings recent / baseline |",
                      "|---|---|---|---:|---|---:|---:|"])
        for r in candidates.itertuples():
            lines.append(f"| {r.name} | {r.forecast_date.date()} | {r.pitch_type} | {r.delta:+.2f} | "
                         f"{r.ci_low:+.2f} to {r.ci_high:+.2f} | {r.n_recent} / {r.n_baseline} | "
                         f"{r.outings_recent} / {r.outings_baseline} |")
        lines.append("")
    (out / "report.md").write_text("\n".join(lines))
    return coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through-season", type=int, choices=range(2021, 2025), default=2024)
    parser.add_argument("--bootstrap", type=int, default=400)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error("--bootstrap must be at least 100")
    out = args.output.resolve()
    root = Path("artifacts/experiments").resolve()
    if not out.is_relative_to(root) or out == root:
        parser.error("--output must name a subdirectory of artifacts/experiments")
    if out.exists() and any(out.iterdir()):
        parser.error("output directory is not empty; choose a new experiment directory")
    load_frozen_spec()
    files = sorted(f for f in RAW.glob("*/*.parquet")
                   if 2020 <= int(f.parent.name) <= args.through_season)
    if not files:
        parser.error("no downloaded development pitches found")
    source_hashes = {p: file_hash(p) for p in SOURCES}
    raw_hashes = {str(f): file_hash(f) for f in files}
    study_hash = file_hash("research_spec.json")
    print(f"Loading {len(files)} development/history chunks (through {args.through_season})", flush=True)
    pitches = pd.concat((pd.read_parquet(f, columns=COLUMNS) for f in files), ignore_index=True)
    pitches = pitches[pitches.game_type == "R"].drop_duplicates(
        ["game_pk", "at_bat_number", "pitch_number"])
    forecasts = eligible_forecasts(pitches)
    outings = aggregate_outings(pitches)
    raw_count = len(pitches)
    del pitches
    print(f"Measuring {len(forecasts):,} eligible forecasts with {args.bootstrap} bootstrap draws", flush=True)
    changes = build_changes(outings, forecasts, n_boot=args.bootstrap, seed=args.seed)
    for path, expected in (source_hashes | raw_hashes | {"research_spec.json": study_hash}).items():
        if file_hash(path) != expected:
            raise RuntimeError(f"Input changed during analysis: {path}; results were not saved")
    load_frozen_spec()
    config = {
        "experiment": "per_type_measurements_v1", "development_seasons": [2021, args.through_season],
        "history_season": 2020, "raw_pitches": raw_count,
        "recent_days": [1, 30], "earlier_days": [31, 120],
        "baseline_policy": "earlier with >=2 pitcher outings; else prior season if available; else earlier",
        "means": "unshrunk, pitch weighted", "metrics": list(METRICS) + ["usage"],
        "bootstrap": {"draws": args.bootstrap, "seed": args.seed, "interval": [0.025, 0.975],
                      "unit": "whole outing (game_pk)", "minimum_outings_per_window": 2,
                      "physical": "conditional on type and metric observed in outing",
                      "usage": "whole repertoire; no interval at boundary usage"},
        "uses_future_outcomes": False, "calibrated_detector": False,
        "versions": {"numpy": np.__version__, "pandas": pd.__version__},
        "source_sha256": source_hashes, "original_study_sha256": study_hash,
        "raw_sha256": raw_hashes,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    changes.to_parquet(out / "changes.parquet", index=False)
    forecasts.to_csv(out / "eligible_forecasts.csv", index=False)
    coverage = write_report(changes, forecasts, out)
    load_frozen_spec()
    print(json.dumps(coverage, indent=2), flush=True)
    print(f"Report: {args.output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
