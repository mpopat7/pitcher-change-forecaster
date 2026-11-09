"""Compare physical-change review lists with velocity-only and usage-only lists at equal capacity.

Run: python3 -m pcf.review_lists
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pcf.freeze import file_hash, load_frozen_spec
from pcf.reliability import _seed

DESIGN = Path("experiments/review_lists.json")
INPUT = Path("artifacts/experiments/reliability")
OUT = Path("artifacts/experiments/review_lists")
METHODS = ("physical", "velocity_only", "usage_only")
PHYSICAL = ("velocity_mph", "hb_in", "ivb_in")
ANCHOR = ["pitcher", "forecast_date"]


def eligible(frame, seasons):
    d = frame[frame.season.isin(seasons) & frame.pitch_type.ne("UN") & np.isfinite(frame.delta)
              & frame.type_outings_recent.ge(2) & frame.type_outings_baseline.ge(2)].copy()
    d["resolved"] = d.future_outings.ge(2) & np.isfinite(d.future)
    return d


def method_scores(d, method):
    if method == "physical":
        rows = d[d.metric.isin(PHYSICAL) & d.prior_variance.gt(0) & np.isfinite(d.shrunk_delta)].copy()
        rows["score"] = rows.shrunk_delta.abs() / np.sqrt(rows.prior_variance)
    elif method == "velocity_only":
        rows = d[d.metric.eq("velocity_mph")].copy()
        rows["score"] = rows.delta.abs()
    elif method == "usage_only":
        rows = d[d.metric.eq("usage")].copy()
        rows["score"] = rows.delta.abs()
    else:
        raise ValueError(method)
    rows = rows[np.isfinite(rows.score)]
    rows = rows.sort_values(["score", "pitcher", "pitch_type", "metric"],
                            ascending=[False, True, True, True], kind="mergesort")
    return rows.drop_duplicates(ANCHOR, keep="first")


def retention(rows):
    sign = np.sign(rows.delta)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = (rows.future - rows.baseline) * sign / rows.delta.abs()
    return value.where(rows.resolved & rows.delta.ne(0))


def build_lists(d, capacities, seasons):
    if not all(0 < c < 1 for c in capacities):
        raise ValueError("capacities must be fractions below 1; 1.0 marks the all-eligible rows")
    anchors = d[ANCHOR + ["season"]].drop_duplicates()
    per_season = anchors.groupby("season").size()
    out = []
    for method in METHODS:
        scored = method_scores(d, method)
        scored["retention"] = retention(scored)
        for season in seasons:
            pool = scored[scored.season == season]
            for capacity in capacities:
                k = int(round(capacity * per_season[season]))
                if k > len(pool):
                    raise ValueError(f"{method} {season} has {len(pool)} candidates for {k} flags")
                top = pool.head(k).copy()
                top["method"], top["capacity"], top["rank"] = method, capacity, np.arange(1, k + 1)
                out.append(top)
        everyone = scored.copy()
        everyone["method"], everyone["capacity"], everyone["rank"] = method, 1.0, np.nan
        out.append(everyone)
    return pd.concat(out, ignore_index=True)


def _rates(group, threshold):
    persisted = group.retention.ge(threshold)
    resolved = int(group.resolved.sum())
    return {"flagged": len(group), "resolved": resolved, "unresolved": len(group) - resolved,
            "persisted": int(persisted.sum()),
            "rate_resolved": persisted.sum() / resolved if resolved else np.nan,
            "rate_flagged": persisted.sum() / len(group) if len(group) else np.nan,
            "median_retention": group.retention.median()}


def summarize(lists, thresholds, *, draws=400, seed=0):
    rows = []
    for (capacity, threshold), _ in pd.Series(0, index=pd.MultiIndex.from_product(
            [sorted(lists.capacity.unique()), thresholds])).items():
        sub = lists[lists.capacity == capacity]
        cells = {m: sub[sub.method == m] for m in METHODS}
        # Pitcher-cluster bootstrap with fixed lists: (persisted, resolved) totals per pitcher and method.
        pitchers = np.array(sorted(sub.pitcher.unique()))
        index = {p: i for i, p in enumerate(pitchers)}
        totals = np.zeros((len(pitchers), len(METHODS), 2))
        for j, m in enumerate(METHODS):
            g = cells[m]
            ids = g.pitcher.map(index).to_numpy()
            np.add.at(totals[:, j, 0], ids, g.retention.ge(threshold).to_numpy(float))
            np.add.at(totals[:, j, 1], ids, g.resolved.to_numpy(float))
        rng = np.random.default_rng(_seed(seed, "review", capacity, threshold))
        weights = rng.multinomial(len(pitchers), np.full(len(pitchers), 1 / len(pitchers)), size=draws)
        boot = np.einsum("bp,pmk->bmk", weights, totals)
        with np.errstate(divide="ignore", invalid="ignore"):
            boot_rate = boot[..., 0] / boot[..., 1]
        for j, m in enumerate(METHODS):
            row = {"capacity": capacity, "threshold": threshold, "method": m, **_rates(cells[m], threshold)}
            row["rate_lo"], row["rate_hi"] = np.nanquantile(boot_rate[:, j], [0.025, 0.975])
            if m != "physical":
                diff = boot_rate[:, 0] - boot_rate[:, j]
                base = _rates(cells["physical"], threshold)["rate_resolved"]
                row["physical_minus"] = base - row["rate_resolved"]
                row["physical_minus_lo"], row["physical_minus_hi"] = np.nanquantile(diff, [0.025, 0.975])
            rows.append(row)
    return pd.DataFrame(rows)


def overlap(lists):
    rows = []
    for capacity, sub in lists[lists.capacity < 1].groupby("capacity"):
        sets = {m: set(map(tuple, sub[sub.method == m][ANCHOR].astype(str).to_numpy())) for m in METHODS}
        for a, b in (("physical", "velocity_only"), ("physical", "usage_only"), ("velocity_only", "usage_only")):
            rows.append({"capacity": capacity, "pair": f"{a}/{b}", "shared": len(sets[a] & sets[b]),
                         "flagged": len(sets[a])})
    return pd.DataFrame(rows)


def triggers(lists):
    sub = lists[lists.capacity < 1]
    return (sub.groupby(["method", "capacity", "metric"]).size().rename("flags").reset_index())


def _pct(x):
    return "n/a" if pd.isna(x) else f"{100 * x:.1f}%"


def write_report(summary, overlaps, trigger_counts, design, out):
    primary = summary[summary.threshold == design["primary_threshold"]]
    lines = ["# Review lists at equal capacity", "",
             "Development seasons 2022-2024. Each method flags the same number of pitcher-dates per "
             "season. A flag persists when the next 30 days stay at least halfway from the old baseline "
             "toward the recent level on the triggering metric. The 100% row scores every eligible "
             "pitcher-date under that method's own trigger.", "",
             "| Capacity | Method | Flagged | Resolved | Persisted / resolved (95% pitcher CI) | Persisted / flagged | Physical minus method (95% CI) |",
             "|---:|---|---:|---:|---|---:|---|"]
    for r in primary.itertuples():
        diff = "" if r.method == "physical" else (
            f"{100 * r.physical_minus:+.1f} pts ({100 * r.physical_minus_lo:+.1f} to {100 * r.physical_minus_hi:+.1f})")
        lines.append(f"| {r.capacity:.0%} | {r.method} | {r.flagged:,} | {r.resolved:,} | "
                     f"{_pct(r.rate_resolved)} ({_pct(r.rate_lo)} to {_pct(r.rate_hi)}) | "
                     f"{_pct(r.rate_flagged)} | {diff} |")
    lines += ["", "## Threshold sensitivity (persisted / resolved)", "",
              "| Capacity | Method | " + " | ".join(f"retain >= {t}" for t in
                                                   [design["sensitivity_thresholds"][0], design["primary_threshold"],
                                                    design["sensitivity_thresholds"][1]]) + " |",
              "|---:|---|---:|---:|---:|"]
    wide = summary.pivot_table(index=["capacity", "method"], columns="threshold", values="rate_resolved")
    for (capacity, method), r in wide.iterrows():
        lines.append(f"| {capacity:.0%} | {method} | " + " | ".join(_pct(v) for v in r.to_numpy()) + " |")
    lines += ["", "## List overlap", "", "| Capacity | Pair | Shared / flagged |", "|---:|---|---:|"]
    for r in overlaps.itertuples():
        lines.append(f"| {r.capacity:.0%} | {r.pair} | {r.shared:,} / {r.flagged:,} |")
    lines += ["", "## Triggering metric for physical flags", "", "| Capacity | Metric | Flags |", "|---:|---|---:|"]
    for r in trigger_counts[trigger_counts.method == "physical"].itertuples():
        lines.append(f"| {r.capacity:.0%} | {r.metric} | {r.flags:,} |")
    lines += ["", "## Caveats", "", design["caveats"], "", design["uncertainty"], ""]
    (out / "report.md").write_text("\n".join(lines))


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
    if list(design["methods"]) != list(METHODS) or max(design["seasons"]) > 2024:
        parser.error("design must use the implemented methods and development seasons only")
    load_frozen_spec()
    old = json.loads((args.input / "config.json").read_text())
    if old["experiment"] != "reliability_persistence_v1" or not old["development_only"]:
        parser.error("expected development-only reliability inputs")
    hashes = {p: file_hash(p) for p in ["pcf/review_lists.py", str(args.design),
                                         str(args.input / "config.json"),
                                         str(args.input / "measurements.parquet"), "research_spec.json"]}
    if old["input_sha256"]["research_spec.json"] != hashes["research_spec.json"]:
        parser.error("original study fingerprint does not match reliability inputs")
    frame = pd.read_parquet(args.input / "measurements.parquet")
    d = eligible(frame, design["seasons"])
    lists = build_lists(d, design["capacities"], design["seasons"])
    thresholds = sorted([design["primary_threshold"], *design["sensitivity_thresholds"]])
    summary = summarize(lists, thresholds, seed=design["seed"])
    overlaps, trigger_counts = overlap(lists), triggers(lists)
    for path, expected in hashes.items():
        if file_hash(path) != expected:
            raise RuntimeError(f"Input changed during analysis: {path}; results were not saved")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({"design": design, "input_sha256": hashes,
                                                 "versions": {"numpy": np.__version__,
                                                              "pandas": pd.__version__}}, indent=2) + "\n")
    keep = ["method", "capacity", "rank", "season", "pitcher", "name", "team", "forecast_date",
            "pitch_type", "metric", "score", "recent", "baseline", "delta", "shrunk_delta", "future",
            "future_outings", "resolved", "retention"]
    lists[lists.capacity < 1][keep].to_csv(out / "lists.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    overlaps.to_csv(out / "overlap.csv", index=False)
    trigger_counts.to_csv(out / "triggers.csv", index=False)
    write_report(summary, overlaps, trigger_counts, design, out)
    print((out / "report.md").read_text())


if __name__ == "__main__":
    main()
