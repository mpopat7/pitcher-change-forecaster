"""Check development pitch-type measurement noise and next-month physical prediction.

Run: python3 -m pcf.reliability
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pcf.changes import COLUMNS, METRICS, SEED, _window_summary, aggregate_outings
from pcf.freeze import file_hash, load_frozen_spec

INPUT = Path("artifacts/experiments/per_type_changes")
OUT = Path("artifacts/experiments/reliability")
KEYS = ["pitcher", "forecast_date", "pitch_type", "metric"]
COUNT_BINS = ["0", "1-24", "25-49", "50-99", "100-199", "200+"]
SOURCES = ["pcf/reliability.py", "pcf/changes.py", "pcf/dataset.py", "pcf/freeze.py",
           "pyproject.toml", "uv.lock"]


def _seed(seed, *parts):
    return int.from_bytes(hashlib.sha256("/".join(map(str, (seed, *parts))).encode()).digest()[:8], "little")


def count_bin(counts):
    return pd.cut(counts, [-1, 0, 24, 49, 99, 199, np.inf], labels=COUNT_BINS)


def _development_dates(features):
    if not features.season.between(2021, 2024).all():
        raise ValueError("Development is restricted to 2021-2024")
    if not pd.to_datetime(features.forecast_date).dt.year.eq(features.season).all():
        raise ValueError("Forecast season must match its date")


def select_anchors(forecasts, gap_days=30):
    keep, last = [], {}
    for r in forecasts.sort_values(["forecast_date", "pitcher"]).itertuples():
        date = pd.Timestamp(r.forecast_date)
        if r.pitcher not in last or (date - last[r.pitcher]).days >= gap_days:
            keep.append(r.Index)
            last[r.pitcher] = date
    return forecasts.loc[keep].reset_index(drop=True)


def independent_splits(recent_sums, recent_counts, baseline_sums, baseline_counts, rng):
    halves = []
    for sums, counts in ((recent_sums, recent_counts), (baseline_sums, baseline_counts)):
        sums, counts = np.asarray(sums, dtype=float), np.asarray(counts, dtype=float)
        valid = counts > 0
        sums, counts = sums[valid], counts[valid]
        if len(counts) < 4:
            return {"delta_a": np.nan, "delta_b": np.nan, "half_min_pitches": np.nan}
        a, b = np.array_split(rng.permutation(len(counts)), 2)
        halves.append((sums[a].sum() / counts[a].sum(), sums[b].sum() / counts[b].sum(),
                       counts[a].sum(), counts[b].sum()))
    c, b = halves
    return {"delta_a": c[0] - b[0], "delta_b": c[1] - b[1],
            "half_min_pitches": min(c[2], c[3], b[2], b[3])}


def _draw_means(sums, counts, rng, draws):
    weights = rng.multinomial(len(counts), np.full(len(counts), 1 / len(counts)), size=draws)
    return (weights @ sums) / (weights @ counts)


def synthetic_null(pool_sums, pool_counts, n_recent, n_baseline, rng, *, outer=100, inner=200,
                   usage=False):
    sums, counts = np.asarray(pool_sums, dtype=float), np.asarray(pool_counts, dtype=float)
    valid = counts > 0
    sums, counts = sums[valid], counts[valid]
    intervals = excluded = degenerate = 0
    if len(counts) >= 2 and min(n_recent, n_baseline) >= 2:
        for _ in range(outer):
            a = rng.integers(len(counts), size=n_recent)
            b = rng.integers(len(counts), size=n_baseline)
            if usage:
                means = [sums[idx].sum() / counts[idx].sum() for idx in (a, b)]
                if not all(0 < mean < 1 for mean in means):
                    continue
            delta = (_draw_means(sums[a], counts[a], rng, inner)
                     - _draw_means(sums[b], counts[b], rng, inner))
            lo, hi = np.quantile(delta, [0.025, 0.975])
            intervals += 1
            excluded += int(lo > 0 or hi < 0)
            degenerate += int(hi == lo)
    rate = excluded / outer
    z = 1.96
    center = (rate + z ** 2 / (2 * outer)) / (1 + z ** 2 / outer)
    half = z * np.sqrt(rate * (1 - rate) / outer + z ** 2 / (4 * outer ** 2)) / (1 + z ** 2 / outer)
    return {"trials": outer, "intervals": intervals, "zero_exclusions": excluded,
            "unsupported_trials": outer - intervals, "degenerate_intervals": degenerate,
            "exclusion_rate_all": rate, "monte_carlo_lo": max(center - half, 0),
            "monte_carlo_hi": min(center + half, 1),
            "exclusion_rate_supported": excluded / intervals if intervals else np.nan,
            "monte_carlo_se": np.sqrt(rate * (1 - rate) / outer)}


def metric_pairs(window, pitch_type, metric):
    p = window[window.pitch_type == pitch_type]
    if metric == "usage":
        counts = window.groupby("game_pk").n_pitches.sum()
        sums = p.groupby("game_pk").n_pitches.sum().reindex(counts.index, fill_value=0)
        return sums.to_numpy(dtype=float), counts.to_numpy(dtype=float)
    counts = p[f"{metric}_count"].to_numpy(dtype=float)
    sums = p[f"{metric}_sum"].to_numpy(dtype=float)
    return sums[counts > 0], counts[counts > 0]


def null_sample_keys(features, budget, seed):
    d = features[(features.pitch_type != "UN") & features.ci_low.notna()].copy()
    d["count_bin"] = count_bin(d[["n_recent", "n_baseline"]].min(axis=1))
    d["order"] = [_seed(seed, "sample", *key) for key in d[KEYS].itertuples(index=False, name=None)]
    groups = [g.sort_values("order").index.tolist()
              for _, g in d.groupby(["season", "metric", "count_bin"], observed=True)]
    chosen = []
    for position in range(max(map(len, groups), default=0)):
        for group in groups:
            if position < len(group):
                chosen.append(group[position])
                if len(chosen) == budget:
                    return set(d.loc[chosen, KEYS].itertuples(index=False, name=None))
    return set(d.loc[chosen, KEYS].itertuples(index=False, name=None))


def measure_noise(outings, features, *, n_boot=400, null_contexts=400, outer=100, inner=200, seed=SEED):
    _development_dates(features)
    selected = null_sample_keys(features, null_contexts, seed)
    by_pitcher = dict(tuple(outings.groupby("pitcher")))
    rows, null_rows = [], []
    previous_season = None
    for (pitcher, cutoff), group in features.groupby(["pitcher", "forecast_date"], sort=False):
        season = cutoff.year
        if season != previous_season:
            print(f"Measuring noise: {season}", flush=True)
            previous_season = season
        p = by_pitcher[pitcher]
        past = p[(p.season == season) & (p.game_date < cutoff)]
        recent = past[past.game_date >= cutoff - pd.Timedelta(days=30)]
        baseline = past[(past.game_date >= cutoff - pd.Timedelta(days=120))
                        & (past.game_date < cutoff - pd.Timedelta(days=30))]
        if group.baseline_source.iloc[0] == "prior_season":
            baseline = p[p.season == season - 1]
        types = sorted(group.pitch_type.unique())
        rng = np.random.default_rng(_seed(seed, "variance", pitcher, cutoff))
        cur = _window_summary(recent, types, rng, n_boot)
        base = _window_summary(baseline, types, rng, n_boot)
        for r in group.to_dict("records"):
            pitch_type, metric = r["pitch_type"], r["metric"]
            c, b = cur[(pitch_type, metric)], base[(pitch_type, metric)]
            # Reconstructed windows must agree with the saved cutoff-only features.
            np.testing.assert_allclose([c["mean"], b["mean"]], [r["recent"], r["baseline"]],
                                       atol=1e-9, rtol=1e-9, equal_nan=True)
            r["noise_variance"] = np.nan
            if pd.notna(r["ci_low"]):
                r["noise_variance"] = np.var(c["draws"], ddof=1) + np.var(b["draws"], ddof=1)
            cs, cn = metric_pairs(recent, pitch_type, metric)
            bs, bn = metric_pairs(baseline, pitch_type, metric)
            split_rng = np.random.default_rng(_seed(seed, "split", pitcher, cutoff, pitch_type, metric))
            r.update(independent_splits(cs, cn, bs, bn, split_rng))
            rows.append(r)
            if (pitcher, cutoff, pitch_type, metric) in selected:
                null_rng = np.random.default_rng(_seed(seed, "null", pitcher, cutoff, pitch_type, metric))
                result = synthetic_null(bs, bn, len(cn), len(bn), null_rng, outer=outer,
                                        inner=inner, usage=metric == "usage")
                null_rows.append({k: r[k] for k in KEYS + ["season", "n_recent", "n_baseline",
                                                          "outings_recent", "outings_baseline"]}
                                 | {"pool_outings": len(bn), "pool_source": r["baseline_source"]}
                                 | result)
    return pd.DataFrame(rows), pd.DataFrame(null_rows)


def shrink_changes(frame):
    d = frame.copy()
    if not d.season.between(2021, 2024).all():
        raise ValueError("Shrinkage development is restricted to 2021-2024")
    for c in ("prior_variance", "shrinkage_weight", "shrunk_delta", "prior_train_end"):
        d[c] = np.nan
    d["prior_source"] = "unavailable"
    priors = []
    valid = (d.pitch_type.ne("UN") & np.isfinite(d.delta) & np.isfinite(d.noise_variance)
             & d.noise_variance.ge(0))
    for season in sorted(d.season.unique()):
        training = d[valid & (d.season < season)]
        for (metric, pitch_type), index in d[d.season == season].groupby(["metric", "pitch_type"]).groups.items():
            if pitch_type == "UN":
                continue
            train = training[(training.metric == metric) & (training.pitch_type == pitch_type)]
            source = "pitch_type"
            if train.pitcher.nunique() < 2:
                train, source = training[training.metric == metric], "metric_pool"
            if train.pitcher.nunique() < 2:
                continue
            tau = max(float(((train.delta ** 2) - train.noise_variance).mean()), 0.0)
            priors.append({"season": int(season), "metric": metric, "pitch_type": pitch_type,
                           "prior_variance": tau, "prior_source": source,
                           "training_rows": len(train), "training_pitchers": train.pitcher.nunique(),
                           "training_seasons": ",".join(map(str, sorted(train.season.unique()))),
                           "mean_noise_variance": train.noise_variance.mean()})
            ids = d.index.intersection(index)
            ids = ids[valid.loc[ids]]
            variance = d.loc[ids, "noise_variance"]
            denominator = tau + variance
            weight = np.divide(tau, denominator.to_numpy(), out=np.zeros(len(ids)),
                               where=denominator.to_numpy() > 0)
            d.loc[ids, "prior_variance"] = tau
            d.loc[ids, "shrinkage_weight"] = weight
            d.loc[ids, "shrunk_delta"] = d.loc[ids, "delta"] * weight
            d.loc[ids, "prior_train_end"] = int(train.season.max())
            d.loc[ids, "prior_source"] = source
    return d, pd.DataFrame(priors)


def build_persistence(outings, features):
    _development_dates(features)
    by_pitcher = dict(tuple(outings.groupby("pitcher")))
    rows = []
    for (pitcher, cutoff), group in features.groupby(["pitcher", "forecast_date"], sort=False):
        p = by_pitcher[pitcher]
        future = p[(p.season == cutoff.year) & (p.game_date >= cutoff)
                   & (p.game_date < cutoff + pd.Timedelta(days=30))]
        for r in group.to_dict("records"):
            sums, counts = metric_pairs(future, r["pitch_type"], r["metric"])
            n = int(counts.sum())
            r.update(future=sums.sum() / n if n else np.nan, future_n=n, future_outings=len(counts))
            r["paired"] = (r["season"] >= 2022 and r["pitch_type"] != "UN" and len(counts) >= 2
                           and all(np.isfinite(r[c]) for c in
                                   ["future", "recent", "baseline", "shrunk_delta", "noise_variance"]))
            for name, prediction in (("baseline", r["baseline"]), ("recent", r["recent"]),
                                     ("shrunk", r["baseline"] + r["shrunk_delta"])):
                r[f"{name}_sq_error"] = (r["future"] - prediction) ** 2 if r["paired"] else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def _correlation(totals):
    n, sx, sy, sxx, syy, sxy = np.asarray(totals).T
    numerator = sxy - sx * sy / n
    denominator = np.sqrt(np.maximum(sxx - sx * sx / n, 0) * np.maximum(syy - sy * sy / n, 0))
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)


def split_summary(frame, *, draws=400, seed=SEED):
    d = frame[(frame.pitch_type != "UN") & frame.delta_a.notna() & frame.delta_b.notna()].copy()
    d["count_bin"] = count_bin(d.half_min_pitches)
    rows = []
    for (metric, bin_name), group in d.groupby(["metric", "count_bin"], observed=True):
        values = pd.DataFrame({"pitcher": group.pitcher, "n": 1.0, "x": group.delta_a,
                               "y": group.delta_b, "xx": group.delta_a ** 2,
                               "yy": group.delta_b ** 2, "xy": group.delta_a * group.delta_b})
        totals = values.groupby("pitcher")[["n", "x", "y", "xx", "yy", "xy"]].sum().to_numpy()
        correlation = float(_correlation(totals.sum(axis=0)))
        lo = hi = np.nan
        if len(totals) >= 2:
            rng = np.random.default_rng(_seed(seed, "correlation", metric, bin_name))
            weights = rng.multinomial(len(totals), np.full(len(totals), 1 / len(totals)), size=draws)
            corr = _correlation(weights @ totals)
            if np.isfinite(corr).any():
                lo, hi = np.nanquantile(corr, [0.025, 0.975])
        rows.append({"metric": metric, "count_bin": str(bin_name), "pairs": len(group),
                     "pitchers": len(totals), "correlation": correlation, "ci_low": lo, "ci_high": hi,
                     "disagreement_rmse": np.sqrt(np.mean((group.delta_a - group.delta_b) ** 2))})
    return pd.DataFrame(rows)


def persistence_summary(frame, *, draws=400, seed=SEED):
    rows = []
    known = frame[(frame.pitch_type != "UN") & (frame.season >= 2022)]
    for metric, group in known.groupby("metric"):
        paired = group[group.paired]
        row = {"metric": metric, "selected_rows": len(group), "resolved_rows": len(paired),
               "unresolved_rows": len(group) - len(paired), "pitchers": paired.pitcher.nunique()}
        for model in ("baseline", "recent", "shrunk"):
            row[f"{model}_mse"] = paired[f"{model}_sq_error"].mean()
        for reference in ("baseline", "recent"):
            difference = paired.shrunk_sq_error - paired[f"{reference}_sq_error"]
            totals = pd.DataFrame({"pitcher": paired.pitcher, "difference": difference, "n": 1.0})
            totals = totals.groupby("pitcher")[["difference", "n"]].sum().to_numpy()
            lo = hi = np.nan
            if len(totals) >= 2:
                rng = np.random.default_rng(_seed(seed, "persistence", metric, reference))
                weights = rng.multinomial(len(totals), np.full(len(totals), 1 / len(totals)), size=draws)
                sample = weights @ totals
                lo, hi = np.quantile(sample[:, 0] / sample[:, 1], [0.025, 0.975])
            row.update({f"shrunk_minus_{reference}": difference.mean(),
                        f"shrunk_minus_{reference}_lo": lo, f"shrunk_minus_{reference}_hi": hi})
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(features, nulls, priors, out):
    splits = split_summary(features)
    persistence = persistence_summary(features)
    nulls = nulls.copy()
    nulls["count_bin"] = count_bin(nulls[["n_recent", "n_baseline"]].min(axis=1))
    null_summary = (nulls.groupby(["metric", "count_bin"], observed=True)
                    .agg(contexts=("trials", "size"), trials=("trials", "sum"),
                         intervals=("intervals", "sum"), zero_exclusions=("zero_exclusions", "sum"),
                         unsupported_trials=("unsupported_trials", "sum"),
                         degenerate_intervals=("degenerate_intervals", "sum"),
                         mean_monte_carlo_se=("monte_carlo_se", "mean")).reset_index())
    null_summary["exclusion_rate_all"] = null_summary.zero_exclusions / null_summary.trials
    null_summary["exclusion_rate_supported"] = (null_summary.zero_exclusions
                                                / null_summary.intervals.replace(0, np.nan))
    splits.to_csv(out / "split_reliability.csv", index=False)
    nulls.to_csv(out / "null_contexts.csv", index=False)
    null_summary.to_csv(out / "synthetic_null.csv", index=False)
    persistence.to_csv(out / "persistence.csv", index=False)
    coverage = (features.groupby(["season", "metric"]).agg(
        rows=("delta", "size"), known_deltas=("delta", "count"),
        noise_estimates=("noise_variance", "count"), independent_splits=("delta_a", "count"),
        shrinkage_estimates=("shrunk_delta", "count"), paired_followup=("paired", "sum"))).reset_index()
    coverage.to_csv(out / "coverage.csv", index=False)
    lines = ["# Reliability and physical persistence", "",
             "This is exploratory development analysis on 2021-2024. Shrinkage comparisons replay "
             "2022-2024 with priors fitted on earlier seasons only. No 2025 or 2026 data is used, "
             "and these results do not test strikeout or walk forecasting.", "",
             "Each pitcher contributes the first eligible date and then dates at least 30 days "
             "apart. Selection uses only the original cutoff metadata. Baselines can still overlap. "
             "Coverage includes unknown labels, missing baselines, unsupported comparisons, and "
             "missing follow-up. Unknown labels are excluded from interpretation and scoring.", "",
             "## Independent halves", "",
             "Both recent and baseline outings are independently split into disjoint halves, with "
             "at least two contributing outings in each half. Correlations compare the two change "
             "estimates, not pitcher levels. Count bins use the smallest half-window pitch count. "
             "Halves can have unequal exposure. Whole-pitcher bootstrap intervals do not fully "
             "capture shared calendar effects or sensitivity to another random split.", "",
             "| Metric | Minimum half-window pitches | Pairs | Correlation | 95% pitcher interval |",
             "|---|---|---:|---:|---|"]
    for r in splits.itertuples():
        lines.append(f"| {r.metric} | {r.count_bin} | {r.pairs:,} | {r.correlation:.3f} | {r.ci_low:.3f} to {r.ci_high:.3f} |")
    lines += ["", "## Synthetic null", "",
              "A bounded hash-selected sample is balanced across season, metric, and count bins, "
              "among known-label contexts with an available interval in the original measurement "
              "report. Thus it is conditional on observed measurement support. "
              "Each context generates two independent pseudo windows by sampling from its same "
              "baseline-outing pool, then bootstraps each pseudo window. The true conditional "
              "mean difference is zero under that exchangeable-outing simulation. This cannot "
              "validate time trends, reclassification, real-world false-alert probability, or "
              "the choice between earlier-season and prior-season baselines. Its aggregate "
              "rates describe the sampled contexts, not population-weighted estimates.", "",
              "Unsupported simulations and zero-width intervals remain in the output. Exclusions "
              "are divided by all trials and by available intervals separately. Per-context "
              "Monte Carlo standard errors and Wilson intervals quantify simulation noise, conditional on the pool; "
              "they are not uncertainty about real-world alert rates.", "",
              "| Metric | Window pitches | Contexts | Intervals / trials | Zero excluded / intervals |",
              "|---|---|---:|---:|---:|"]
    for r in null_summary.itertuples():
        lines.append(f"| {r.metric} | {r.count_bin} | {r.contexts} | {r.intervals:,} / {r.trials:,} | {r.exclusion_rate_supported:.1%} |")
    lines += ["", "## Past-only shrinkage and future physical measurements", "",
              "The prior is centered on no change. Signal variance is the nonnegative mean of "
              "past squared changes minus their outing-bootstrap noise variance. A type-specific "
              "prior requires two training pitchers; otherwise the metric-wide prior is used "
              "when it has two pitchers. This is a computational support rule, not a reliability "
              "threshold. Priors and training seasons are saved in `priors.csv`. The weight is "
              "signal variance divided by signal plus measurement variance; zero/zero gives zero "
              "weight. Bootstrap noise can be underestimated with few outings, so this shrinkage "
              "also needs evaluation rather than being assumed optimal.", "",
              "Predict the next 30 days' physical mean with the unchanged baseline, unchanged "
              "recent state, or baseline plus shrunk change. Compare all three on identical rows "
              "with an earlier-season prior and at least two future contributing outings. "
              "A missing type has no physical follow-up; its usage is zero only if future outings "
              "exist. No future outings means missing usage too. Rows lacking a usable baseline "
              "or prior are also unresolved for this paired comparison. These are squared errors "
              "in each metric's own units, not comparisons across unlike scales.", "",
              "| Metric | Paired / selected | Baseline MSE | Recent MSE | Shrunk MSE | Shrunk minus recent, 95% pitcher interval |",
              "|---|---:|---:|---:|---:|---|"]
    for r in persistence.itertuples():
        lines.append(f"| {r.metric} | {r.resolved_rows:,} / {r.selected_rows:,} | {r.baseline_mse:.5f} | "
                     f"{r.recent_mse:.5f} | {r.shrunk_mse:.5f} | {r.shrunk_minus_recent:+.5f} "
                     f"({r.shrunk_minus_recent_lo:+.5f} to {r.shrunk_minus_recent_hi:+.5f}) |")
    lines += ["", "Lower error is better. Shared baseline noise can inflate direction-retention "
              "statistics, so sign agreement is not used as the primary persistence result. "
              "Missing follow-up and future measurement noise limit interpretation. Results "
              "remain development diagnostics; no minimum sample size, alert threshold, or "
              "forecasting benefit is established by this report.", ""]
    (out / "report.md").write_text("\n".join(lines))
    return persistence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--bootstrap", type=int, default=400)
    parser.add_argument("--null-contexts", type=int, default=400)
    parser.add_argument("--null-trials", type=int, default=100)
    parser.add_argument("--null-bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    root, out = Path("artifacts/experiments").resolve(), args.output.resolve()
    if not out.is_relative_to(root) or out == root or (out.exists() and any(out.iterdir())):
        parser.error("choose an empty output subdirectory of artifacts/experiments")
    if min(args.bootstrap, args.null_bootstrap) < 100 or args.null_trials < 20 or args.null_contexts < 1:
        parser.error("use >=100 bootstrap draws, >=20 null trials, and >=1 null context")
    load_frozen_spec()
    old = json.loads((args.input / "config.json").read_text())
    if old["development_seasons"][1] > 2024 or old["uses_future_outcomes"]:
        parser.error("expected cutoff-only development measurements through 2024")
    if old["original_study_sha256"] != file_hash("research_spec.json"):
        parser.error("measurements belong to a different original study fingerprint")
    input_files = [args.input / f for f in ("config.json", "changes.parquet", "eligible_forecasts.csv")]
    hashes = {p: file_hash(p) for p in SOURCES}
    hashes.update({str(p): file_hash(p) for p in input_files})
    hashes.update(old["raw_sha256"])
    hashes["research_spec.json"] = file_hash("research_spec.json")
    for path, expected in (old["source_sha256"] | old["raw_sha256"]).items():
        if file_hash(path) != expected:
            raise ValueError(f"Measurement inputs changed: {path}")
    files = [Path(p) for p in old["raw_sha256"]]
    if any(not 2020 <= int(p.parent.name) <= 2024 for p in files):
        parser.error("raw inputs must be development/history seasons 2020-2024")
    forecasts = pd.read_csv(args.input / "eligible_forecasts.csv", parse_dates=["forecast_date"])
    anchors = select_anchors(forecasts)
    original = pd.read_parquet(args.input / "changes.parquet")
    features = original.merge(anchors[["pitcher", "forecast_date"]], on=["pitcher", "forecast_date"],
                              how="inner", validate="many_to_one")
    features = features.sort_values(["forecast_date", "pitcher", "pitch_type", "metric"]).reset_index(drop=True)
    print(f"Loading {len(files)} development/history chunks; {len(anchors):,} anchored forecasts", flush=True)
    pitches = pd.concat((pd.read_parquet(p, columns=COLUMNS) for p in files), ignore_index=True)
    outings = aggregate_outings(pitches)
    del pitches, original
    measured, nulls = measure_noise(outings, features, n_boot=args.bootstrap,
                                   null_contexts=args.null_contexts, outer=args.null_trials,
                                   inner=args.null_bootstrap, seed=args.seed)
    print("Fitting past-seasons-only shrinkage", flush=True)
    shrunk, priors = shrink_changes(measured)
    print("Evaluating next-30-day physical measurements", flush=True)
    evaluated = build_persistence(outings, shrunk)
    for path, expected in hashes.items():
        if file_hash(path) != expected:
            raise RuntimeError(f"Input changed during analysis: {path}; results were not saved")
    load_frozen_spec()
    config = {"experiment": "reliability_persistence_v1", "development_only": True,
              "anchor_gap_days": 30, "eligible_forecasts": len(forecasts), "anchors": len(anchors),
              "bootstrap_draws": args.bootstrap, "null_context_budget": args.null_contexts,
              "null_outer_trials": args.null_trials, "null_inner_draws": args.null_bootstrap,
              "seed": args.seed, "split": "both windows; >=4 contributing outings before splitting",
              "null_pool": "baseline outings; independent pseudo windows; conditional exchangeability",
              "shrinkage": "zero-centered; mean(delta^2 - sampling variance), clipped >=0; past years only",
              "prior_minimum_training_pitchers": 2, "future_window": [0, 29],
              "future_minimum_outings": 2, "unknown_labels": "retained in coverage; excluded from scoring",
              "prediction_targets": "physical means and usage only; no K/BB",
              "versions": {"numpy": np.__version__, "pandas": pd.__version__}, "input_sha256": hashes}
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    evaluated.to_parquet(out / "measurements.parquet", index=False)
    anchors.to_csv(out / "anchors.csv", index=False)
    priors.to_csv(out / "priors.csv", index=False)
    persistence = write_report(evaluated, nulls, priors, out)
    print(persistence.to_string(index=False), flush=True)
    print(f"Report: {args.output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
