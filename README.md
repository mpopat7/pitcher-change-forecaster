# Pitcher Change Forecaster

Do recent changes in a pitcher's repertoire improve forecasts of his performance over the next month?

This project uses public MLB Statcast data to compare forecasts based on past results, current pitch characteristics, and recent changes in those characteristics. The pipeline produces a historical performance watchlist and an audit of experimental improvement and decline alerts.

## Status

The frozen specification was evaluated once on the 2025 holdout. Current pitch measurements modestly improve strikeout forecasts over the results-only baseline; adding recent-change features makes strikeout forecasts slightly worse and shows no clear walk benefit. The performance watchlist and experimental alert audit are implemented. All 2020-2025 data is downloaded and passes the game-level audit. There is no interface yet.

## Running it

Requires [uv](https://docs.astral.sh/uv/). Downloaded data and outputs stay in the gitignored `data/` and `artifacts/` folders.

```bash
uv run python3 -m pcf.ingest      # 2020 history plus 2021-2025; resumes saved chunks
uv run python3 -m pcf.audit --fetch-schedules  # compare chunks with completed MLB games
uv run python3 -m pcf.dataset --through-season 2024
uv run python3 -m pcf.support     # chronological repertoire-fallback checks on 2022 and 2023
uv run python3 -m pcf.model       # fit A/B/C, tune and score on 2024
uv run python3 -m pcf.report      # comparison charts, calibration, and failure reviews
uv run python3 -m pcf.watchlist   # all-eligible performance ranks and experimental alert audit
uv run python3 -m unittest discover -s tests -v
```

The full download is about 270 requests and takes roughly 90 minutes. Keep the machine awake (`caffeinate -i` on macOS); a sleeping laptop stalls it.

To include 2025 in the dataset, run `uv run python3 -m pcf.dataset` without a cutoff. After completing validation, `uv run python3 -m pcf.freeze` records the model and watchlist specification in `research_spec.json`, including source, dependency-lock, and validation-report hashes. It refuses to replace an existing freeze or run after holdout outputs exist. `uv run python3 -m pcf.model --final` checks the frozen specification, refits on 2021-2024, and scores 2025 once; it refuses to overwrite holdout outputs. Use `uv run python3 -m pcf.watchlist --final` and `uv run python3 -m pcf.report --final` to evaluate and plot those saved forecasts without refitting.

## Final evaluation: 2025

The model, feature lists, repertoire fallback, and watchlist rules were frozen in [research_spec.json](research_spec.json) at 2026-10-26 05:49 UTC, before scoring 2025. The specification SHA-256 is `b2038e9abd9b586b5621536b07528bb834de05d3af9da06968fa36c4a653cb91`; the holdout report records the same fingerprint. All models refit on 2021-2024 with regularization C=0.001. No settings were changed after the holdout results were read.

Of 6,734 eligible forecasts, 381 (5.7%) have no subsequent appearances. The remaining 6,353 scored forecasts cover 538 pitchers and 435,187 BF exposures across overlapping weekly target windows. All eligible forecasts are exported. The repertoire fallback applies to five forecasts from two pitchers (Tim Hill and Tommy Kahnle), all with follow-up.

Lower log loss per BF is better:

| Target | A: results/context | B: + current pitches | C: + recent changes |
|---|---:|---:|---:|
| Strikeouts | 0.526900 | 0.526715 | 0.526801 |
| Walks | 0.269622 | 0.269599 | 0.269611 |

The central C-minus-B difference is **+0.000086** for strikeouts (95% pitcher-bootstrap interval +0.000020 to +0.000154) and **+0.000012** for walks (−0.000053 to +0.000070). Recent-change features slightly worsen strikeout accuracy under this specification and provide no clear walk improvement. B-minus-A is −0.000185 for strikeouts (−0.000360 to −0.000008) and −0.000023 for walks (−0.000138 to +0.000091): a small strikeout gain and uncertain walk difference. These intervals resample whole pitchers and condition on the selected settings; they do not fully capture shared calendar-time effects. The scope of this finding is the tested features, windows, and linear model.

The frozen experimental alert rules produce **19 alerts across 19 pitchers: 8 confirmed, 5 false, 6 unresolved**. Five of the 13 resolved alerts are false (38.5%); 11 of all 19 are unconfirmed (57.9%) when unresolved alerts remain in the denominator. Five alerts are not yet evident in the past-results comparison, and three of those confirm under the operational definition below. The sample is small and does not overturn the full-population C-versus-B result. The main performance watchlist has 210 entries across 32 pitchers, including six without follow-up. Its BF-weighted predicted K% minus BB% is 25.36%, observed 26.43%; this diagnostic repeats exposure across weekly rankings.

Use B for the performance ranking and retain C's alerts as experimental review candidates. This study does not establish that tracking recent changes adds reliable next-month forecasting value. A new modeling experiment would need a new evaluation period; 2025 has now been used.

## Revised validation: 2024

Measured on 2026-10-26 with the same 2021-2023 training seasons and 2024 evaluation population as the initial run. Lower log loss per BF is better:

| Target | A: results/context | B: + current pitches | C: + recent changes |
|---|---:|---:|---:|
| Strikeouts | 0.533940 | 0.533732 | 0.533774 |
| Walks | 0.265933 | 0.265745 | 0.265729 |

With the fallback, C-minus-B is **+0.000042** for strikeouts (95% pitcher-bootstrap interval −0.000028 to +0.000108) and **−0.000016** for walks (−0.000071 to +0.000040). Both intervals include zero. B-minus-A is −0.000208 for strikeouts (−0.000411 to −0.000008) and −0.000189 for walks (−0.000300 to −0.000081). These remain development comparisons: the failure was discovered on 2024, and regularization was reselected on 2024 after adding the fallback. The intervals condition on those settings and do not fully capture shared calendar-time effects. All six model/target combinations select C=0.001 from the existing grid. The 2025 holdout is needed to assess generalization.

### Repertoire fallback

For each current pitch group (fastball, cutter, breaking, offspeed, other), record the minimum and maximum usage share across **all eligible training forecasts**, including rows without future appearances. Flag an evaluation forecast if any current share is missing or outside that range. Both B and C then use A's prediction for that target. The bounds use only training features, never evaluation outcomes; no pitchers or flagged forecasts are removed. The rule has no tuned margin or percentile. It checks each current usage share separately; it does not establish support for combinations of pitches, physical traits, or change features.

Before the revised 2024 run, the fixed rule was checked with expanding chronological splits: train 2021 → evaluate 2022; train 2021-2022 → evaluate 2023. Training outcome windows end before the evaluation forecasts. Each of the seven existing regularization settings is reported without selecting a winner on either evaluation season. To isolate the fallback at each setting, A, B, and C use the same regularization within that comparison.

| Evaluation season | Flagged eligible / scored | Pitchers | Reasons |
|---|---:|---:|---|
| 2022 | 11 / 11 | 2 | Cutter usage (10), other usage (1) |
| 2023 | 3 / 3 | 1 | Offspeed usage (3) |
| 2024 | 33 / 32 | 3 | Other usage (31), fastball usage (2) |

In those earlier splits, the fallback lowers C's loss for both targets at every tested setting. B's effect is mixed: its largest overall loss increase is 0.0000013 per BF. Only 14 earlier scored forecasts trigger the rule, so this is a limited check of a safety fallback, not evidence for a general accuracy improvement.

The 2024 flags are Matt Waldron (20 eligible forecasts, including one without follow-up), Adrian Morejon (11), and Tim Hill (2). For Waldron on 2024-04-29, B and C now use A's **18.94%** K forecast, versus the observed 37/135 = **27.41%**. The original B forecast was 1.05%; with revised regularization its unguarded forecast would be 0.99%. The fallback prevents that extrapolation but does not make the forecast accurate.

`pcf.support` writes every setting's results and all eligibility flags to `artifacts/support_chronological.{json,csv}` and `artifacts/repertoire_chronological.csv`. The model writes `repertoire_validation_2024.csv` for all eligible forecasts, including no-follow-up rows, and saves raw and fallback predictions for scored rows. Reports include training bounds, flag counts, and losses at the selected settings before and after fallback. The report command also compares forecasts within the usage ranges against forecasts using A. The default model run reads only seasons through 2024; support checks read only through 2023.

Validation saves `chosen_C.json` and `repertoire_spec.json`; the final path uses the settings in the frozen `research_spec.json` and refits usage bounds on all eligible 2021-2024 rows. The initial report, predictions, and settings from this development run are retained locally in `artifacts/before_repertoire_fallback/`.

## Watchlist and experimental alerts

The main watchlist ranks **B's projected K% minus BB%**, showing the top ten eligible pitchers each forecast date. Both components, A and C comparisons, past exposure, and repertoire fallback flags are retained. Forecasts cover all eligible rows, including pitchers with no future appearances. The separate `predictions_*` files retain only rows with scored outcomes for the model comparisons.

Experimental alerts flag a C-minus-B difference in projected K% minus BB% of at least **two percentage points**, in either direction. Require at least 150 current pitches and 150 pitches in the earlier comparison window, using the prior-season fallback already defined in the dataset. Forecasts using the repertoire fallback do not alert. Once a pitcher alerts, suppress further alerts for 30 days in either direction. This prevents overlapping target windows within a pitcher's alert series. These are fixed review rules, not thresholds optimized for 2024 success rate.

An alert is resolved only with at least 50 subsequent BF. It is **confirmed under this operational definition** if the observed next-month K% minus BB% differs from B's prediction by at least two points in the alerted direction; otherwise it is a false alert. Fewer than 50 BF, including zero, is unresolved. Report false alerts divided by resolved alerts, plus unconfirmed alerts divided by **all** alerts, including unresolved cases. These are realized outcome checks subject to sampling noise; they do not establish a change in underlying skill, causal coaching effects, or a conventional population false-positive rate.

To inspect whether the direction was already apparent in conventional results, compare past-30-day K% minus BB% with the prior-season rate, requiring at least 50 BF in each. A same-direction difference of two points is already evident; a smaller or opposite change is not yet evident. Missing past exposure is a separate category. This is a descriptive proxy, not an estimate of lead time or evidence that the physical change happened first. It does not affect alert selection.

On 2024 validation, 46 candidates become **18 alerts across 18 pitchers** after cooldown: **5 confirmed, 3 false, 10 unresolved**. Thus 37.5% of the eight resolved alerts are false, while 72.2% of all alerts are unconfirmed. Seven alerts are not yet evident in the past-results comparison; three of those confirm. The small resolved sample does not establish a reliable alert detector. The main performance watchlist has 210 entries across 47 pitchers and includes three entries without follow-up. Its BF-weighted predicted K% minus BB% is 23.79%, observed 22.34%; repeated weekly exposure remains in this ranking diagnostic.

Outputs are `artifacts/watchlist_{validation_2024,holdout_2025}.{csv,json}` and `artifacts/alerts_{validation_2024,holdout_2025}.csv`. They remain local and gitignored. `research_spec.json` is the portable record of the model and watchlist rules fixed before final evaluation.

## Initial validation: 2024, before fallback

Measured on 2026-10-24, training on 2021-2023. There are 6,857 eligible 2024 forecasts; 429 (6.3%) have no subsequent appearances. Accuracy uses the remaining 6,428 forecasts across 538 pitchers. Their 434,953 BF exposures include repeated plate appearances in overlapping weekly target windows.

Lower log loss per BF is better:

| Target | A: results/context | B: + current pitches | C: + recent changes |
|---|---:|---:|---:|
| Strikeouts | 0.533940 | 0.536391 | 0.536214 |
| Walks | 0.265933 | 0.266126 | 0.266029 |

The central C-minus-B comparison is **−0.000177** for strikeouts (95% pitcher-bootstrap interval −0.000729 to +0.000086) and **−0.000097** for walks (−0.000325 to +0.000038). Both intervals include zero. These are development results: regularization was selected on the same 2024 data. The intervals condition on that selection, resample whole pitchers, and do not fully capture shared calendar-time effects across pitchers.

### Failure case: an unfamiliar repertoire

Matt Waldron accounts for most of B's excess loss relative to A. The training rows have at most 2.56% usage in the residual `other` pitch group. Waldron's 2024 repertoire is roughly 33-44% in that group, mainly his knuckleball. The downloaded data contains 936 knuckleballs, matching [MLB's player notes](https://img.mlbstatic.com/mlb-images/image/upload/fl_attachment/mlb/i1wm79f8ska4k2gupw8d.pdf). The linear models extrapolate far beyond the usage range seen in training.

For the 2024-04-29 forecast, B predicts a 1.05% strikeout rate for Waldron; he records 37 strikeouts in 135 BF (27.4%). C predicts 1.32%, so it reduces the loss while still making a poor forecast. All main results retain him. As a diagnostic only, excluding Waldron changes C-minus-B to +0.000032 for strikeouts and −0.000007 for walks; the apparent aggregate gain is therefore not broad evidence for recent changes.

The report command writes the central C-versus-B chart (PNG and PDF), all model comparisons, calibration plots, role/workload results, per-pitcher loss contributions, and the largest C-versus-B misses to `artifacts/`. Generated artifacts and downloaded data stay out of Git.

Hand checks for Aaron Nola and Josh Hader on 2024-06-03 match raw recent-window counts, future targets, fastball velocity, and velocity changes. Adding 2024 leaves all 20,687 existing 2021-2023 rows unchanged. The full 2021-2025 dataset contains 34,278 rows across 1,169 pitchers and 102 forecast dates; adding 2025 leaves all 27,544 earlier rows unchanged.

## Definitions in the code

- Batters faced excludes intentional walks and truncated plate appearances. K includes strikeout double plays. BB counts unintentional walks only.
- Forecasts are made each Monday whose full 30-day window ends inside the regular season. Eligible pitchers faced at least 50 batters in the prior 60 days and pitched within the prior 14.
- Opponent quality: batters already faced are rated by their shrunk K% and BB% as of the day before each game; upcoming opponents are the teams on the pitcher's schedule for the next 30 days, rated as of the forecast date. The schedule is taken from games actually played, so postponed games that were never made up inside the window are missing.
- Model B sees only the last 30 days of pitch characteristics. Model C adds the earlier baseline window (days 31-120, or the prior season when that window has fewer than 150 pitches) and the change from it. Otherwise a linear model given both windows could reconstruct the change itself.

## Baseball question

A pitcher might gain fastball velocity, change his slider's movement, or start using a secondary pitch more often before his strikeout and walk rates visibly change. The study asks whether those measurements provide useful evidence about future performance after accounting for information already available in recent results and current pitch quality.

This connects to player evaluation and development monitoring. Forecasting an improvement does not establish that a particular grip, delivery, or pitch-selection adjustment caused it. Coaching recommendations would require additional evidence.

## Forecast targets

At each historical forecast date, estimate:

- Strikeout rate (K%): strikeouts divided by batters faced over the following 30 calendar days.
- Walk rate (BB%): walks divided by batters faced over the same period.
- K% - BB% as a summary, with both component forecasts shown separately.

Forecasts describe performance conditional on subsequent appearances. Report future exposure, missing follow-up, and uncertainty. Define eligibility using information available at the forecast date, so injuries, demotions, or low future workload do not silently remove difficult cases.

Whiff rate (misses per swing) and other pitch outcomes can support the analysis, but they do not replace the main targets.

## Data

Source: [MLB Baseball Savant / Statcast](https://baseballsavant.mlb.com/statcast_search), using its [documented CSV fields](https://baseballsavant.mlb.com/csv-docs).

Start with 2021-2025 MLB data. Relevant fields include game dates, pitcher and batter IDs, pitch type, velocity, movement, spin, release position, extension, plate location, batter handedness, count, pitch outcomes, and plate-appearance outcomes.

Candidate inputs, computed only from preceding appearances:

| Group | Examples |
|---|---|
| Recent results | Strikeout and walk rates, whiff rate, swing behavior |
| Current repertoire | Pitch-type usage, velocity and movement distributions |
| Delivery measurements | Release position, extension, dispersion within pitch types |
| Relationships between pitches | Velocity and movement differences relative to the primary fastball |
| Recent changes | Differences between recent and earlier windows, with sample counts |
| Context | Pitcher role, workload, historical opponent quality and handedness mix |

MLB documents changes to plate-location and strike-zone definitions in 2026. Treat that season separately until measurements can be compared appropriately. Keep downloaded data and generated artifacts out of Git; record retrieval dates, filters, and processing steps when ingestion is implemented.

## Model comparisons

| Model | Information available |
|---|---|
| A: results baseline | Historical and recent outcomes plus context; shrink noisy rates toward longer history and population averages |
| B: current pitches | Model A inputs plus current physical pitch characteristics and usage |
| C: recent changes | Model B inputs plus changes in the repertoire and delivery measurements |

The central test is C versus B. Improvement over A alone would show that pitch measurements help forecasting, without isolating the value of tracking changes. Give B a credible recency-weighted specification so C does not win against an artificially weak comparator.

The frozen study uses binomial ridge logistic regression with the windows and thresholds documented above. Alternative model families or feature definitions would be a separate experiment with a new evaluation period.

## Evaluation

- Split: train on 2021-2023, tune on 2024, and evaluate the frozen specification on 2025 after refitting on 2021-2024.
- Replay forecast dates chronologically. Training outcomes and every input must be available before the relevant cutoff; keep overlapping target windows from crossing evaluation boundaries.
- Compare probability accuracy and calibration, with uncertainty estimates that account for repeated pitchers and time dependence.
- Report performance across workloads and pitcher roles, coverage, and missing follow-up.
- Set watchlist rules using validation data, then measure false alerts and whether useful signals precede changes in conventional results.
- Evaluate the full eligible population before selecting case studies; include failed forecasts.

A finding that change features add no predictive value is a valid research result. Report it alongside the comparison that produced it.

## Planned interface

A ranked pitcher watchlist with next-month forecasts, uncertainty, and the difference from the baseline. Each pitcher view would show pitch measurements over time, supporting outcomes, sample sizes, and historical forecast errors. Model explanations describe predictive associations.

The current pipeline exports point forecasts and uncertainty intervals for **model comparisons**. Pitcher-level forecast intervals are not implemented; the interface must distinguish those from the bootstrap comparison intervals.

## Next session

1. Build a historical forecast browser around B's performance ranking, with A/C comparisons, sample sizes, repertoire flags, and failed forecasts.
2. Show experimental alerts with their confirmed/false/unresolved status and the full denominator; do not present them as a validated change detector.
3. Keep the frozen study and its 2025 result intact when exploring future model changes.

## Related work

- [Stuff+, Location+, and Pitching+](https://library.fangraphs.com/pitching/stuff-location-and-pitching-primer/): existing pitch-quality and performance models.
- [Unified Pitch Graphs](https://arxiv.org/abs/2609.03810): retrospective analysis of pitching strategy and execution changes.
- [Baseball Savant pitch arsenals](https://baseballsavant.mlb.com/pitchmix): existing repertoire visualizations.

The project investigates the incremental forecasting value of recent changes. It makes no claim that pitcher forecasting or change detection is new.
