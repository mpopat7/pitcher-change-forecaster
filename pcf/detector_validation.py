"""Validate the physical change score against velocity alerts with past-only cutoffs and a split baseline.

Run, in order (see experiments/detector_validation.json):
    python3 -m pcf.detector_validation develop
    python3 -m pcf.detector_validation freeze
    python3 -m pcf.detector_validation evaluate
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from pcf.changes import COLUMNS, METRICS, _window_summary, aggregate_outings, eligible_forecasts
from pcf.freeze import file_hash, load_frozen_spec
from pcf.reliability import _seed, select_anchors

DESIGN = Path("experiments/detector_validation.json")
FREEZE = Path("experiments/detector_validation_freeze.json")
RAW = Path("data/raw/statcast")
DEV_OUT = Path("artifacts/experiments/detector_validation_dev")
EVAL_OUT = Path("artifacts/experiments/detector_validation_evaluation")
SOURCES = ["pcf/detector_validation.py", "pcf/changes.py", "pcf/reliability.py", "pcf/dataset.py",
           "pcf/freeze.py", str(DESIGN), "pyproject.toml", "uv.lock", "research_spec.json"]
METHODS = ("physical", "velocity_shrunk", "velocity_raw", "usage_raw")
PHYSICAL = ("velocity_mph", "hb_in", "ivb_in")
ANCHOR = ["pitcher", "forecast_date"]
STAGES = {"develop": {"seasons": (2021, 2024), "prospective": [2023, 2024], "retrospective": [2022, 2023, 2024]},
          "evaluate": {"seasons": (2021, 2026), "prospective": [2025, 2026], "retrospective": [2025, 2026]}}
N_BOOT = 200


def split_baseline(baseline):
    """Alternate baseline outings by date and game id into halves A and B."""
    games = baseline[["game_pk", "game_date"]].drop_duplicates().sort_values(["game_date", "game_pk"])
    first = set(games.game_pk.iloc[0::2])
    second = set(games.game_pk.iloc[1::2])
    return baseline[baseline.game_pk.isin(first)], baseline[baseline.game_pk.isin(second)]


def _levels(window, types):
    """Means and outing counts per pitch type and metric, without resampling."""
    out = {}
    total = window.groupby("game_pk").n_pitches.sum()
    n_all = int(total.sum())
    by_type = dict(tuple(window.groupby("pitch_type")))
    for pitch_type in types:
        p = by_type.get(pitch_type, window.iloc[:0])
        type_n = int(p.n_pitches.sum())
        out[(pitch_type, "usage")] = (type_n / n_all if n_all else np.nan, len(total))
        for metric in METRICS:
            counts = p[f"{metric}_count"].to_numpy(dtype=float)
            n = counts.sum()
            out[(pitch_type, metric)] = (p[f"{metric}_sum"].sum() / n if n else np.nan,
                                         int((counts > 0).sum()))
    return out


def measure(outings, anchors, seed):
    by_pitcher = dict(tuple(outings.groupby("pitcher")))
    rows = []
    for a in anchors.sort_values(ANCHOR).itertuples(index=False):
        cutoff = pd.Timestamp(a.forecast_date)
        p = by_pitcher[a.pitcher]
        current = p[(p.season == a.season) & (p.game_date < cutoff)]
        recent = current[current.game_date >= cutoff - pd.Timedelta(days=30)]
        earlier = current[(current.game_date >= cutoff - pd.Timedelta(days=120))
                          & (current.game_date < cutoff - pd.Timedelta(days=30))]
        prior = p[p.season == a.season - 1]
        baseline, source = earlier, "earlier_window"
        if earlier.game_pk.nunique() < 2 and not prior.empty:
            baseline, source = prior, "prior_season"
        half_a, half_b = split_baseline(baseline)
        future = p[(p.season == a.season) & (p.game_date >= cutoff)
                   & (p.game_date < cutoff + pd.Timedelta(days=30))]
        types = sorted((set(recent.pitch_type) | set(half_a.pitch_type)) - {"UN"})
        if not types:
            continue
        rng = np.random.default_rng(_seed(seed, "detector", a.pitcher, cutoff.date()))
        cur = _window_summary(recent, types, rng, N_BOOT)
        base = _window_summary(half_a, types, rng, N_BOOT)
        held, ahead = _levels(half_b, types), _levels(future, types)
        for pitch_type in types:
            for metric in (*METRICS, "usage"):
                c, b = cur[(pitch_type, metric)], base[(pitch_type, metric)]
                noise = np.nan
                if c["draws"] is not None and b["draws"] is not None:
                    noise = np.var(c["draws"], ddof=1) + np.var(b["draws"], ddof=1)
                fut, fut_outings = ahead[(pitch_type, metric)]
                rows.append({
                    "pitcher": a.pitcher, "name": a.name, "team": a.team, "season": a.season,
                    "forecast_date": cutoff, "pitch_type": pitch_type, "metric": metric,
                    "baseline_source": source, "recent": c["mean"], "base_a": b["mean"],
                    "base_b": held[(pitch_type, metric)][0], "delta": c["mean"] - b["mean"],
                    "noise_variance": noise, "type_outings_recent": c["type_outings"],
                    "type_outings_a": b["type_outings"],
                    "type_outings_b": int(half_b.loc[half_b.pitch_type == pitch_type, "game_pk"].nunique()),
                    "future": fut, "future_outings": fut_outings,
                })
    return pd.DataFrame(rows)


def supported(frame):
    return (frame.pitch_type.ne("UN") & np.isfinite(frame.delta) & frame.type_outings_recent.ge(2)
            & frame.type_outings_a.ge(2) & frame.type_outings_b.ge(2) & np.isfinite(frame.base_b))


def shrink(frame):
    """Past-seasons-only priors: season s uses supported rows from 2021 through s-1."""
    d = frame.copy()
    d["prior_variance"], d["shrunk_delta"] = np.nan, np.nan
    usable = supported(d) & np.isfinite(d.noise_variance) & d.noise_variance.ge(0)
    priors = []
    for season in sorted(d.season.unique()):
        training = d[usable & (d.season < season)]
        for (metric, pitch_type), index in d[d.season == season].groupby(["metric", "pitch_type"]).groups.items():
            train, source = training[(training.metric == metric) & (training.pitch_type == pitch_type)], "pitch_type"
            if train.pitcher.nunique() < 2:
                train, source = training[training.metric == metric], "metric_pool"
            if train.pitcher.nunique() < 2:
                continue
            tau = max(float((train.delta ** 2 - train.noise_variance).mean()), 0.0)
            ids = index[usable.loc[index].to_numpy()]
            noise = d.loc[ids, "noise_variance"].to_numpy()
            weight = np.divide(tau, tau + noise, out=np.zeros(len(ids)), where=(tau + noise) > 0)
            d.loc[ids, "prior_variance"] = tau
            d.loc[ids, "shrunk_delta"] = d.loc[ids, "delta"].to_numpy() * weight
            priors.append({"season": int(season), "metric": metric, "pitch_type": pitch_type,
                           "prior_variance": tau, "source": source, "training_rows": len(train),
                           "training_end": int(train.season.max())})
    return d, pd.DataFrame(priors)


def method_scores(d, method):
    rows = d[supported(d)]
    if method in ("physical", "velocity_shrunk"):
        metrics = PHYSICAL if method == "physical" else ("velocity_mph",)
        rows = rows[rows.metric.isin(metrics) & rows.prior_variance.gt(0) & np.isfinite(rows.shrunk_delta)].copy()
        rows["score"] = rows.shrunk_delta.abs() / np.sqrt(rows.prior_variance)
    elif method == "velocity_raw":
        rows = rows[rows.metric.eq("velocity_mph")].copy()
        rows["score"] = rows.delta.abs()
    elif method == "usage_raw":
        rows = rows[rows.metric.eq("usage")].copy()
        rows["score"] = rows.delta.abs()
    else:
        raise ValueError(method)
    rows = rows[np.isfinite(rows.score)].sort_values(
        ["score", "pitcher", "pitch_type", "metric"], ascending=[False, True, True, True], kind="mergesort")
    return rows.drop_duplicates(ANCHOR, keep="first")


def outcome(rows):
    """Resolution and persistence against held-out half B, plus the shared-baseline diagnostic."""
    r = rows.copy()
    r["resolved"] = r.future_outings.ge(2) & np.isfinite(r.future)
    sign = np.sign(r.delta)
    lead = (r.recent - r.base_b) * sign
    kept = (r.future - r.base_b) * sign
    r["persisted"] = r.resolved & np.isfinite(lead) & lead.gt(0) & kept.ge(0.5 * lead)
    shared = (r.future - r.base_a) * sign
    r["persisted_shared"] = r.resolved & r.delta.ne(0) & shared.ge(0.5 * r.delta.abs())
    return r


def cutoff(past_scores, n_anchors, capacity):
    k = int(round(capacity * n_anchors))
    ranked = np.sort(np.asarray(past_scores, dtype=float))[::-1]
    if k < 1 or k > len(ranked):
        raise ValueError(f"Cannot place {k} flags among {len(ranked)} past scores")
    return ranked[k - 1]


def flag(scored, anchors, capacities, prospective, retrospective):
    per_season = anchors.groupby("season").size()
    out, cutoffs = [], []
    for method, s in scored.items():
        for capacity in capacities:
            for season in prospective:
                past = [y for y in per_season.index if 2022 <= y < season]
                if not past:
                    raise ValueError(f"No past seasons for prospective cutoffs in {season}")
                value = cutoff(s[s.season.isin(past)].score, per_season[past].sum(), capacity)
                cutoffs.append({"method": method, "capacity": capacity, "season": season,
                                "cutoff": value, "past_seasons": ",".join(map(str, past))})
                top = s[(s.season == season) & s.score.ge(value)].copy()
                top["scheme"], top["capacity"] = "prospective", capacity
                out.append(top.assign(method=method))
            for season in retrospective:
                k = int(round(capacity * per_season[season]))
                top = s[s.season == season].head(k).copy()
                top["scheme"], top["capacity"] = "retrospective", capacity
                out.append(top.assign(method=method))
    return pd.concat(out, ignore_index=True), pd.DataFrame(cutoffs)


def summarize(flags, seasons, *, draws, seed):
    rows = []
    for (scheme, capacity), sub in flags.groupby(["scheme", "capacity"]):
        for label, keep in [("pooled", seasons)] + [(str(y), [y]) for y in seasons]:
            cell = sub[sub.season.isin(keep)]
            if cell.empty:
                continue
            pitchers = np.array(sorted(cell.pitcher.unique()))
            index = {p: i for i, p in enumerate(pitchers)}
            totals = np.zeros((len(pitchers), len(METHODS), 3))
            for j, m in enumerate(METHODS):
                g = cell[cell.method == m]
                ids = g.pitcher.map(index).to_numpy()
                np.add.at(totals[:, j, 0], ids, g.persisted.to_numpy(float))
                np.add.at(totals[:, j, 1], ids, g.resolved.to_numpy(float))
                np.add.at(totals[:, j, 2], ids, 1.0)
            rng = np.random.default_rng(_seed(seed, "detector-boot", scheme, capacity, label))
            weights = rng.multinomial(len(pitchers), np.full(len(pitchers), 1 / len(pitchers)), size=draws)
            boot = np.einsum("bp,pmk->bmk", weights, totals)
            with np.errstate(divide="ignore", invalid="ignore"):
                rate = boot[..., 0] / boot[..., 1]
                rate_flagged = boot[..., 0] / boot[..., 2]
            physical = None
            for j, m in enumerate(METHODS):
                g = cell[cell.method == m]
                persisted, resolved = int(g.persisted.sum()), int(g.resolved.sum())
                row = {"scheme": scheme, "capacity": capacity, "seasons": label, "method": m,
                       "flagged": len(g), "resolved": resolved, "persisted": persisted,
                       "rate_resolved": persisted / resolved if resolved else np.nan,
                       "rate_flagged": persisted / len(g) if len(g) else np.nan,
                       "rate_shared_baseline": g.persisted_shared.sum() / resolved if resolved else np.nan}
                row["rate_lo"], row["rate_hi"] = np.nanquantile(rate[:, j], [0.025, 0.975])
                if m == "physical":
                    physical = row
                else:
                    row["physical_minus"] = physical["rate_resolved"] - row["rate_resolved"]
                    row["physical_minus_lo"], row["physical_minus_hi"] = np.nanquantile(
                        rate[:, 0] - rate[:, j], [0.025, 0.975])
                    row["physical_minus_flagged"] = physical["rate_flagged"] - row["rate_flagged"]
                    row["physical_minus_flagged_lo"], row["physical_minus_flagged_hi"] = np.nanquantile(
                        rate_flagged[:, 0] - rate_flagged[:, j], [0.025, 0.975])
                rows.append(row)
    return pd.DataFrame(rows)


def verdicts(summary, design):
    primary = summary[(summary.scheme == "prospective") & (summary.capacity == design["capacities"]["primary"])
                      & (summary.seasons == "pooled")].set_index("method")
    out = {}
    for claim, method in (("beats_velocity_alerts", "velocity_raw"), ("movement_adds", "velocity_shrunk")):
        r = primary.loc[method]
        lo, hi = r.physical_minus_lo, r.physical_minus_hi
        out[claim] = {"difference": r.physical_minus, "lo": lo, "hi": hi,
                      "verdict": "supported" if lo > 0 else "reversed" if hi < 0 else "not replicated"}
    return out


def _pct(x):
    return "n/a" if pd.isna(x) else f"{100 * x:.1f}%"


def _pts(x, lo, hi):
    return f"{100 * x:+.1f} pts ({100 * lo:+.1f} to {100 * hi:+.1f})"


def write_report(summary, cutoffs, verdict, stage, design, out):
    title = "development (2022-2024; not a test)" if stage == "develop" else "evaluation (2025-2026)"
    lines = [f"# Change-detector validation: {title}", "",
             "Flags persist when the next 30 days stay at least halfway from held-out baseline half B "
             "toward the recent level. Selection used only baseline half A. Prospective cutoffs come "
             "from earlier seasons only.", ""]
    if stage == "evaluate":
        lines += ["## Pre-registered claims (10% prospective, pooled)", ""]
        for claim, v in verdict.items():
            lines.append(f"- **{claim}**: {v['verdict']}, physical minus comparator "
                         f"{_pts(v['difference'], v['lo'], v['hi'])}")
        lines.append("")
    for scheme in ("prospective", "retrospective"):
        lines += [f"## {scheme.title()} flags", "",
                  "| Capacity | Seasons | Method | Flagged | Resolved | Persisted / resolved (95% CI) | "
                  "Persisted / flagged | Shared-baseline rate | Physical minus method (95% CI) |",
                  "|---:|---|---|---:|---:|---|---:|---:|---|"]
        for r in summary[summary.scheme == scheme].itertuples():
            diff = "" if r.method == "physical" else _pts(r.physical_minus, r.physical_minus_lo, r.physical_minus_hi)
            lines.append(f"| {r.capacity:.0%} | {r.seasons} | {r.method} | {r.flagged:,} | {r.resolved:,} | "
                         f"{_pct(r.rate_resolved)} ({_pct(r.rate_lo)} to {_pct(r.rate_hi)}) | "
                         f"{_pct(r.rate_flagged)} | {_pct(r.rate_shared_baseline)} | {diff} |")
        lines.append("")
    lines += ["## Prospective cutoffs", "", "| Method | Capacity | Season | Cutoff | From seasons |",
              "|---|---:|---:|---:|---|"]
    for r in cutoffs.itertuples():
        lines.append(f"| {r.method} | {r.capacity:.0%} | {r.season} | {r.cutoff:.3f} | {r.past_seasons} |")
    lines += ["", "## Design", "", design["purpose"], "", design["stop_rule"], ""]
    (out / "report.md").write_text("\n".join(lines))


def load(seasons):
    files = sorted(f for f in RAW.glob("*/*.parquet") if seasons[0] - 1 <= int(f.parent.name) <= seasons[1])
    if any(int(f.parent.name) > seasons[1] for f in files):
        raise ValueError("Raw data beyond the stage's last season")
    pitches = pd.concat((pd.read_parquet(f, columns=COLUMNS) for f in files), ignore_index=True)
    pitches = pitches[pitches.game_type == "R"].drop_duplicates(["game_pk", "at_bat_number", "pitch_number"])
    pitches["season"] = pd.to_datetime(pitches.game_date).dt.year
    forecasts = eligible_forecasts(pitches)
    forecasts = forecasts[forecasts.season.between(*seasons)]
    outings = aggregate_outings(pitches)
    return outings, forecasts, {str(f): file_hash(f) for f in files}


def run(stage, design, out):
    cfg = STAGES[stage]
    load_frozen_spec()
    hashes = {p: file_hash(p) for p in SOURCES}
    outings, forecasts, raw = load(cfg["seasons"])
    anchors = pd.concat(select_anchors(g) for _, g in forecasts.groupby("season"))
    anchors = anchors.sort_values(ANCHOR).reset_index(drop=True)
    print(f"{stage}: {len(anchors):,} anchors {anchors.groupby('season').size().to_dict()}", flush=True)
    measured, priors = shrink(measure(outings, anchors, design["seed"]))
    scored = {m: outcome(method_scores(measured, m)) for m in METHODS}
    capacities = [design["capacities"]["primary"], *design["capacities"]["secondary"]]
    flags, cutoffs = flag(scored, anchors, capacities, cfg["prospective"], cfg["retrospective"])
    seasons = sorted(set(cfg["prospective"]) | set(cfg["retrospective"]))
    summary = summarize(flags, seasons, draws=2000, seed=design["seed"])
    verdict = verdicts(summary, design)
    for path, expected in (hashes | raw).items():
        if file_hash(path) != expected:
            raise RuntimeError(f"Input changed during analysis: {path}; results were not saved")
    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps(
        {"experiment": design["experiment"], "stage": stage, "run_at_utc": datetime.now(timezone.utc).isoformat(),
         "anchors": anchors.groupby("season").size().to_dict(), "input_sha256": hashes | raw,
         "versions": {"numpy": np.__version__, "pandas": pd.__version__}}, indent=2, default=int) + "\n")
    measured.to_parquet(out / "measurements.parquet", index=False)
    priors.to_csv(out / "priors.csv", index=False)
    flags.to_csv(out / "flags.csv", index=False)
    cutoffs.to_csv(out / "cutoffs.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "verdicts.json").write_text(json.dumps(verdict, indent=2) + "\n")
    write_report(summary, cutoffs, verdict, stage, design, out)
    print((out / "report.md").read_text())


def freeze(design):
    if FREEZE.exists():
        raise FileExistsError("Already frozen")
    if EVAL_OUT.exists():
        raise FileExistsError("Evaluation outputs exist; cannot freeze")
    report = DEV_OUT / "report.md"
    if not report.exists():
        raise FileNotFoundError("Run the development stage first")
    record = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": {p: file_hash(p) for p in SOURCES},
              "development_report_sha256": file_hash(report)}
    FREEZE.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Froze {len(record['source_sha256'])} sources in {FREEZE}")


def check_frozen():
    if not FREEZE.exists():
        raise FileNotFoundError("Freeze the design before evaluating")
    record = json.loads(FREEZE.read_text())
    for path, expected in record["source_sha256"].items():
        if file_hash(path) != expected:
            raise ValueError(f"Changed after freezing: {path}")
    if file_hash(DEV_OUT / "report.md") != record["development_report_sha256"]:
        raise ValueError("Development report changed after freezing")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["develop", "freeze", "evaluate"])
    args = parser.parse_args()
    design = json.loads(DESIGN.read_text())
    if args.stage == "freeze":
        freeze(design)
        return
    out = DEV_OUT if args.stage == "develop" else EVAL_OUT
    if out.exists():
        parser.error(f"{out} exists; each stage runs once")
    if args.stage == "evaluate":
        check_frozen()
    elif FREEZE.exists():
        parser.error("Design is frozen; development cannot rerun")
    run(args.stage, design, out)


if __name__ == "__main__":
    main()
