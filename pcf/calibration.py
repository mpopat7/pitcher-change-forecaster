"""Compare a fixed pair of interval methods on known truth and development outing pools.

Run: python3 -m pcf.calibration
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import t

from pcf.changes import COLUMNS, aggregate_outings
from pcf.freeze import file_hash, load_frozen_spec
from pcf.reliability import _development_dates, _draw_means, _seed, metric_pairs

DESIGN = Path("experiments/interval_calibration.json")
INPUT = Path("artifacts/experiments/reliability")
OUT = Path("artifacts/experiments/interval_calibration")
METHODS = ("percentile", "welch_t")
OUTING_BINS = ["2", "3", "4-5", "6-9", "10+"]


def _valid_pairs(sums, counts):
    sums, counts = np.asarray(sums, dtype=float), np.asarray(counts, dtype=float)
    valid = np.isfinite(sums) & np.isfinite(counts) & (counts > 0)
    return sums[valid], counts[valid]


def ratio_variance(sums, counts):
    sums, counts = _valid_pairs(sums, counts)
    m = len(counts)
    if not m:
        return np.nan, np.nan, 0
    ratios = sums / counts
    tolerance = 8 * np.finfo(float).eps * np.max(np.abs(ratios))
    if np.ptp(ratios) <= tolerance:
        return ratios[0], 0.0 if m > 1 else np.nan, m
    mean = sums.sum() / counts.sum()
    variance = m / (m - 1) * np.sum((sums - mean * counts) ** 2) / counts.sum() ** 2 if m > 1 else np.nan
    return mean, variance, m


def welch_interval(recent_sums, recent_counts, baseline_sums, baseline_counts, *, usage=False):
    recent, vr, nr = ratio_variance(recent_sums, recent_counts)
    baseline, vb, nb = ratio_variance(baseline_sums, baseline_counts)
    if min(nr, nb) < 2:
        return np.nan, np.nan, "fewer_than_two_outings"
    if usage and not (0 < recent < 1 and 0 < baseline < 1):
        return np.nan, np.nan, "usage_boundary"
    variance = vr + vb
    if variance <= 0:
        return np.nan, np.nan, "zero_estimated_variance"
    df = variance ** 2 / (vr ** 2 / (nr - 1) + vb ** 2 / (nb - 1))
    radius = t.ppf(0.975, df) * np.sqrt(variance)
    delta = recent - baseline
    return delta - radius, delta + radius, "ok"


def compare_intervals(recent_sums, recent_counts, baseline_sums, baseline_counts, rng, *,
                      draws=400, usage=False):
    cs, cn = _valid_pairs(recent_sums, recent_counts)
    bs, bn = _valid_pairs(baseline_sums, baseline_counts)
    welch = welch_interval(cs, cn, bs, bn, usage=usage)
    if min(len(cn), len(bn)) < 2:
        percentile = (np.nan, np.nan, "fewer_than_two_outings")
    elif usage and not (0 < cs.sum() / cn.sum() < 1 and 0 < bs.sum() / bn.sum() < 1):
        percentile = (np.nan, np.nan, "usage_boundary")
    else:
        delta = _draw_means(cs, cn, rng, draws) - _draw_means(bs, bn, rng, draws)
        lo, hi = np.quantile(delta, [0.025, 0.975])
        percentile = (lo, hi, "ok")
    return {"percentile": percentile, "welch_t": welch}


def sample_known(distribution, count_mode, n_outings, rng, shift=0):
    if distribution == "normal":
        state = rng.normal(size=n_outings)
    elif distribution == "lognormal":
        state = (np.exp(rng.normal(size=n_outings)) - np.exp(0.5)) / np.sqrt((np.e - 1) * np.e)
    elif distribution == "two_point":
        state = rng.choice([-1.0, 1.0], size=n_outings)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    if count_mode == "fixed":
        counts = np.full(n_outings, 20.0)
    elif count_mode == "informative":
        counts = np.where(state < 0, 5.0, 50.0)
    else:
        raise ValueError(f"Unknown count mode: {count_mode}")
    return (state + shift) * counts, counts


def _trial_rows(cs, cn, bs, bn, rng, truth, trial, draws, usage=False):
    intervals = compare_intervals(cs, cn, bs, bn, rng, draws=draws, usage=usage)
    recent, baseline = ratio_variance(cs, cn)[0], ratio_variance(bs, bn)[0]
    estimate = recent - baseline
    tolerance = 8 * np.finfo(float).eps * max(abs(recent), abs(baseline), abs(truth))
    common = all(np.isfinite([lo, hi]).all() for lo, hi, _ in intervals.values())
    return [{"trial": trial, "method": method, "estimate": estimate, "truth": truth,
             "low": lo, "high": hi, "reason": reason, "available": np.isfinite([lo, hi]).all(),
             "covered": bool(lo - tolerance <= truth <= hi + tolerance),
             "zero_excluded": bool(lo > tolerance or hi < -tolerance),
             "width": hi - lo, "numerical_tolerance": tolerance,
             "zero_width": bool(hi - lo <= tolerance), "common_available": common}
            for method, (lo, hi, reason) in intervals.items()]


def simulate_context(pool_sums, pool_counts, n_recent, n_baseline, rng, *, trials=100,
                     draws=400, usage=False):
    sums, counts = _valid_pairs(pool_sums, pool_counts)
    rows = []
    for trial in range(trials):
        if len(counts):
            a, b = rng.integers(len(counts), size=n_recent), rng.integers(len(counts), size=n_baseline)
            rows.extend(_trial_rows(sums[a], counts[a], sums[b], counts[b], rng, 0.0, trial, draws, usage))
        else:
            rows.extend(_trial_rows([], [], [], [], rng, 0.0, trial, draws, usage))
    return pd.DataFrame(rows)


def known_trials(design):
    frames = []
    grid = itertools.product(design["distributions"], design["count_modes"],
                             design["outing_pairs"], design["effects"])
    for distribution, mode, (nr, nb), effect in grid:
        print(f"Known truth: {distribution}, {mode}, {nr}/{nb} outings, shift {effect:g}", flush=True)
        rows = []
        rng = np.random.default_rng(_seed(design["seed"], "known", distribution, mode, nr, nb, effect))
        for trial in range(design["known_trials_per_cell"]):
            cs, cn = sample_known(distribution, mode, nr, rng, shift=effect)
            bs, bn = sample_known(distribution, mode, nb, rng)
            rows.extend(_trial_rows(cs, cn, bs, bn, rng, effect, trial, design["bootstrap_draws"]))
        frames.append(pd.DataFrame(rows).assign(distribution=distribution, count_mode=mode,
                                                n_recent=nr, n_baseline=nb, effect=effect))
    return pd.concat(frames, ignore_index=True)


def empirical_trials(outings, contexts, design):
    _development_dates(contexts)
    by_pitcher = dict(tuple(outings.groupby("pitcher")))
    frames = []
    for i, r in enumerate(contexts.itertuples(index=False)):
        if i % 50 == 0:
            print(f"Empirical context {i + 1}/{len(contexts)}", flush=True)
        p = by_pitcher[r.pitcher]
        if r.pool_source == "prior_season":
            pool = p[p.season == r.season - 1]
        elif r.pool_source == "earlier_window":
            pool = p[(p.season == r.season)
                     & (p.game_date >= r.forecast_date - pd.Timedelta(days=120))
                     & (p.game_date < r.forecast_date - pd.Timedelta(days=30))]
        else:
            raise ValueError(f"Unknown pool source: {r.pool_source}")
        sums, counts = metric_pairs(pool, r.pitch_type, r.metric)
        if len(counts) != r.pool_outings:
            raise ValueError("Outing pool no longer matches saved calibration context")
        rng = np.random.default_rng(_seed(design["seed"], "calibration", r.pitcher,
                                         r.forecast_date, r.pitch_type, r.metric))
        d = simulate_context(sums, counts, r.outings_recent, r.outings_baseline, rng,
                             trials=design["empirical_trials_per_context"],
                             draws=design["bootstrap_draws"], usage=r.metric == "usage")
        for column in ("pitcher", "forecast_date", "pitch_type", "metric", "season", "pool_source"):
            d[column] = getattr(r, column)
        d["n_recent"], d["n_baseline"] = r.outings_recent, r.outings_baseline
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def summarize(frame, group_keys):
    rows = []
    for key, group in frame.groupby(group_keys, observed=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        available, common = group[group.available], group[group.common_available]
        n, na, nc = len(group), len(available), len(common)
        covered, exclusions = int(available.covered.sum()), int(available.zero_excluded.sum())
        p = exclusions / n
        z = 1.96
        center = (p + z * z / (2 * n)) / (1 + z * z / n)
        radius = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        row = dict(zip(group_keys, key)) | {
            "trials": n, "available": na, "unsupported": n - na, "availability": na / n,
            "zero_width": int(available.zero_width.sum()) if "zero_width" in available
                          else int(available.width.eq(0).sum()), "covered_and_available_all": covered / n,
            "coverage_available": covered / na if na else np.nan, "zero_exclusions": exclusions,
            "exclusion_rate_all": p, "exclusion_rate_available": exclusions / na if na else np.nan,
            "exclusion_mc_lo": max(center - radius, 0), "exclusion_mc_hi": min(center + radius, 1),
            "width_median": available.width.median(), "width_p90": available.width.quantile(0.9),
            "common_trials": nc, "common_coverage": common.covered.mean(),
            "common_exclusion_rate": common.zero_excluded.mean(), "common_width_median": common.width.median(),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(known, empirical, out):
    known_keys = ["distribution", "count_mode", "n_recent", "n_baseline", "effect", "method"]
    known_summary = summarize(known, known_keys)
    empirical = empirical.copy()
    empirical["outing_bin"] = pd.cut(empirical[["n_recent", "n_baseline"]].min(axis=1),
                                     [1, 2, 3, 5, 9, np.inf], labels=OUTING_BINS)
    empirical_summary = summarize(empirical, ["metric", "outing_bin", "method"])
    by_outings = summarize(empirical, ["outing_bin", "method"])
    reasons = (pd.concat([known.assign(cohort="known"), empirical.assign(cohort="empirical")])
               .groupby(["cohort", "method", "reason"]).size().rename("trials").reset_index())
    known_summary.to_csv(out / "known_summary.csv", index=False)
    empirical_summary.to_csv(out / "empirical_summary.csv", index=False)
    by_outings.to_csv(out / "empirical_by_outings.csv", index=False)
    reasons.to_csv(out / "support_reasons.csv", index=False)
    lines = ["# Interval-method calibration", "",
             "This is a fixed development comparison of percentile bootstrap and finite-cluster "
             "Welch-style t intervals. It does not select an alert cutoff or certify either method "
             "for real-world change detection. All earlier results remain intact; no 2025/2026 "
             "data or K/BB targets enter this experiment.", "",
             "The known-truth grid varies outing balance, distribution, and informative pitch "
             "counts. Both methods use identical samples. The positive shift is one SD of the "
             "unweighted latent outing mean, applied after counts are generated; it is a sensitivity "
             "illustration rather than a practical baseball threshold.", "",
             "Welch uses the pitch-weighted ratio mean, a finite-outing residual variance, and "
             "Welch-Satterthwaite degrees of freedom. It is an approximation under independent "
             "outings, not a theorem for two-outing baseball windows. Zero estimated variance "
             "makes that interval unavailable, whereas percentile point intervals stay available. "
             "Both methods suppress observed boundary-usage intervals. Eight machine epsilons, "
             "scaled to the window measurement level, handle effectively constant ratios and "
             "numerical coverage boundaries equally for both methods. This is not an effect-size "
             "threshold. Intervals are not clipped "
             "to parameter bounds; width is part of the tradeoff.", "",
             "## Known-truth null stress tests", "",
             "Each row gives the range across the nine outing-count pairs. This exposes bad "
             "cases rather than reporting only an average. The full cell-level Monte Carlo "
             "results and positive-shift sensitivity are in `known_summary.csv`.", "",
             "| Distribution | Counts | Method | Available coverage range | Lowest availability |",
             "|---|---|---|---:|---:|"]
    null = known_summary[known_summary.effect == 0]
    for (distribution, mode, method), g in null.groupby(["distribution", "count_mode", "method"]):
        lines.append(f"| {distribution} | {mode} | {method} | "
                     f"{g.coverage_available.min():.1%} to {g.coverage_available.max():.1%} | {g.availability.min():.1%} |")
    lines += ["", "## Empirical baseline-outing pools", "",
              "The same 400 previously selected development contexts are reused, conditional on "
              "observed interval support. This is not an untouched evaluation. Pseudo windows "
              "are independent draws from the same baseline pool, not real adjacent periods. "
              "This run uses 400 inner bootstrap draws, versus 200 in the earlier null report, "
              "with a new fixed seed stream. The earlier failure remains the reference result.", "",
              "The table pools the balanced sample's metrics, not a population-weighted cohort. "
              "`empirical_summary.csv` separates each metric and supplies widths in its own units. "
              "Monte Carlo intervals quantify simulation variation only, conditional on these "
              "pools, not uncertainty about real-world false-alert rates.", "",
              "| Minimum outings | Method | Available / trials | Zero excluded / available | Coverage on common-supported trials |",
              "|---|---|---:|---:|---:|"]
    for r in by_outings.itertuples():
        lines.append(f"| {r.outing_bin} | {r.method} | {r.available:,} / {r.trials:,} | "
                     f"{r.exclusion_rate_available:.1%} | {r.common_coverage:.1%} |")
    lines += ["", "Unavailable intervals are failure to provide an interval, not successful "
              "coverage. `covered_and_available_all` reports the joint fraction separately from "
              "coverage among available intervals. Common-support results use exactly the same "
              "trials for both methods. Support reasons and zero-width counts remain visible.", "",
              "Wider intervals or additional unsupported cases can lower zero exclusions without "
              "making a method useful. Review the known-shift sensitivity and common-supported "
              "widths alongside calibration. Neither synthetic design validates temporal drift, "
              "pitch reclassification, serial dependence, or operational false-alert control. "
              "The declared stopping rule is to finish this grid and preserve every failure, "
              "with no automatic method choice or sample-size threshold.", "",
              "Methods background: [SciPy t distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html), "
              "[Welch reference implementation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html), "
              "[clustered-inference background](https://www.nber.org/papers/t0344). These references "
              "do not validate this application.", ""]
    (out / "report.md").write_text("\n".join(lines))
    return by_outings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DESIGN)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    out, root = args.output.resolve(), Path("artifacts/experiments").resolve()
    if not out.is_relative_to(root) or out == root or (out.exists() and any(out.iterdir())):
        parser.error("choose an empty output subdirectory of artifacts/experiments")
    design = json.loads(args.design.read_text())
    if design["methods"] != list(METHODS) or design["confidence"] != 0.95:
        parser.error("implementation supports the fixed two-method 95% comparison")
    if (design["fixed_pitch_count"] != 20 or design["informative_pitch_counts"] != [5, 50]
            or design["bootstrap_draws"] < 100):
        parser.error("design counts must match the implemented generators; use >=100 bootstrap draws")
    if min(design["known_trials_per_cell"], design["empirical_trials_per_context"],
           design["empirical_contexts"]) < 1:
        parser.error("trial and context counts must be positive")
    load_frozen_spec()
    old = json.loads((args.input / "config.json").read_text())
    if old["experiment"] != "reliability_persistence_v1" or not old["development_only"]:
        parser.error("expected development-only reliability inputs")
    if old["input_sha256"]["research_spec.json"] != file_hash("research_spec.json"):
        parser.error("original study fingerprint does not match reliability inputs")
    hashes = old["input_sha256"].copy()
    for p in ["pcf/calibration.py", str(args.design), str(args.input / "config.json"),
              str(args.input / "null_contexts.csv")]:
        hashes[p] = file_hash(p)
    for path, expected in hashes.items():
        if file_hash(path) != expected:
            raise ValueError(f"Input changed since prior experiment: {path}")
    files = [Path(p) for p in old["input_sha256"] if Path(p).parent.parent.name == "statcast"]
    if not files or any(not 2020 <= int(p.parent.name) <= 2024 for p in files):
        parser.error("expected raw development/history inputs through 2024 only")
    contexts = pd.read_csv(args.input / "null_contexts.csv", parse_dates=["forecast_date"])
    _development_dates(contexts)
    keys = ["pitcher", "forecast_date", "pitch_type", "metric"]
    if len(contexts) != design["empirical_contexts"] or contexts.duplicated(keys).any():
        parser.error("context cohort must match the declared count, without duplicates")
    known = known_trials(design)
    print("Loading saved empirical outing pools", flush=True)
    pitches = pd.concat((pd.read_parquet(p, columns=COLUMNS) for p in files), ignore_index=True)
    outings = aggregate_outings(pitches)
    del pitches
    empirical = empirical_trials(outings, contexts, design)
    for path, expected in hashes.items():
        if file_hash(path) != expected:
            raise RuntimeError(f"Input changed during calibration: {path}; results were not saved")
    load_frozen_spec()
    out.mkdir(parents=True, exist_ok=True)
    config = {"design": design, "input_sha256": hashes,
              "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__}}
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    known.to_parquet(out / "known_trials.parquet", index=False)
    empirical.to_parquet(out / "empirical_trials.parquet", index=False)
    summary = write_report(known, empirical, out)
    print(summary[["outing_bin", "method", "availability", "exclusion_rate_available", "common_coverage"]].to_string(index=False))
    print(f"Report: {args.output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
