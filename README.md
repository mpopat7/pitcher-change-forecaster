# Pitcher Change Forecaster

Do recent changes in a pitcher's repertoire improve forecasts of his performance over the next month?

This project uses public MLB Statcast data to compare forecasts based on past results, current pitch characteristics, and recent changes in those characteristics. The pipeline produces a historical performance watchlist and an audit of experimental improvement and decline alerts.

## Status

The frozen specification was evaluated once on the 2025 holdout. Current pitch measurements modestly improve strikeout forecasts over the results-only baseline; adding recent-change features makes strikeout forecasts slightly worse and shows no clear walk benefit. A local historical forecast browser now covers every eligible 2024 and 2025 forecast, with pitch histories, model comparisons, and experimental alert outcomes. All 2020-2025 data is downloaded and passes the game-level audit.

The per-type experiment measures 27,544 eligible development forecasts from 2021-2024, with sample counts and Detroit/Cincinnati examples. Reliability checks found poor interval calibration. A fixed comparison found that Welch-style t intervals improve some cases but leave substantial failures. Neither method establishes a reliable change alert. At equal review capacity, a shrunk physical-change list keeps its recent level somewhat more often than raw velocity or usage lists. Shrinkage gives a small development gain for next-month velocity prediction; movement and usage gains remain uncertain. In 2023-2024 development folds, per-type change features, a simple velocity change, and a gradient-boosted model all left next-month K% and BB% forecasts statistically unchanged. The original specification and saved results stay frozen.

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

## Browse the saved forecasts

From the repository root, with the saved study outputs already present:

```bash
uv run python3 -m pcf.browser             # build and serve at http://127.0.0.1:8765
uv run python3 -m pcf.browser --build-only # export without starting a server
node --test tests/browser.test.mjs        # frontend filtering, export, and chart checks
```

The browser reads saved forecasts, watchlist audits, and as-of pitch measurements. It checks the frozen study fingerprint and does not fit models, select new thresholds, or rerun holdout scoring. Source files are in `web/`; the builder is `pcf/browser.py`. It adds no dependencies to the frozen environment.

The generated `artifacts/browser/standalone.html` embeds the data, styles, and JavaScript for all **13,591 eligible forecasts** across both seasons. It can be copied or opened as a single file without a running server. Web fonts are optional requests to Google Fonts; forecast data stays in the file. The same folder also contains the modular static site and season-specific JSON files. All exports remain gitignored.

Choose a season and forecast date, filter by pitcher or past role, and switch between the frozen top-ten watchlist, all eligible pitchers, experimental alerts, and largest misses. Dates can be replayed with previous/next controls. Ranks stay those of the original study, including when filters narrow the list. Alerts and misses initially span the full season. Largest misses are absolute B-versus-observed K% minus BB% gaps among forecasts with at least 50 future BF; this view explicitly uses later outcomes.

Select a pitcher for A/B/C forecasts, current and earlier-window pitch measurements, pitch mix, and history through the selected date. History charts can show projected K% minus BB%, fastball velocity, vertical movement, or usage, with a table alternative. Later forecast dates are excluded from those histories. The **Reveal outcomes** control shows following-month results and saved confirmed/false/unresolved alert statuses. Zero follow-up is shown as missing observation, never as a zero rate. Turning outcomes off exits the misses view. Table and history CSV exports omit outcomes while they are hidden; copied forecast links retain the season, date, pitcher, filters, view, and outcome visibility.

The UI uses the chosen soft-editorial style: Newsreader headings, Geist interface text, a cool gray page, white panels, and a deep-green accent. It is keyboard accessible and adapts to phone widths. Individual forecast uncertainty intervals remain unestimated; only the study's model-comparison intervals are displayed. These are historical forecasts, not live player projections. The browser has not been published.

For a short demonstration, follow [the browser walkthrough](DEMO.md), which pairs a confirmed alert with a false alert and an unresolved case before returning to the full study result. A 2026-10-27 review verified desktop and phone layouts, outcome visibility, and switching embedded seasons with the preview server stopped.

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

With the fallback, C-minus-B is **+0.000042** for strikeouts (95% pitcher-bootstrap interval −0.000028 to +0.000108) and **−0.000016** for walks (−0.000071 to +0.000040). Both intervals include zero. B-minus-A is −0.000208 for strikeouts (−0.000411 to −0.000008) and −0.000189 for walks (−0.000300 to −0.000081). These remain development comparisons: the failure was discovered on 2024, and regularization was reselected on 2024 after adding the fallback. The intervals condition on those settings and do not fully capture shared calendar-time effects. All six model/target combinations select C=0.001 from the existing grid. The completed 2025 evaluation above measures generalization for that frozen specification.

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

MLB documents changes to plate-location and strike-zone definitions in 2026. Treat that season separately until measurements can be compared appropriately. Downloaded data and generated artifacts stay out of Git. The ingestion script records retrieval times and chunk row counts in `data/raw/statcast/retrieval_log.jsonl`; its source defines the filters and retained fields.

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

## Interface limits

The browser displays point forecasts, comparisons, pitch measurements over time, sample sizes, and historical outcomes. Model explanations describe predictive associations. Change alerts remain experimental and show unresolved cases alongside confirmed and false alerts.

The current pipeline exports point forecasts and uncertainty intervals for **model comparisons**. Pitcher-level forecast intervals are not implemented; the interface must distinguish those from the bootstrap comparison intervals.

## Research plan: changes within pitch types

The original features combine some pitch types into broad groups. A group's average movement can change because a pitcher throws more sinkers and fewer four-seam fastballs, even when neither pitch changes. Those averages still describe the repertoire, and the original forecasting result stands. The next experiment separates pitch selection from changes within a pitch type.

### Measure the change

Start with velocity, vertical and horizontal movement, and usage for each pitcher and recorded pitch type. Report recent and baseline pitch counts, outings, missing measurements, and uncertainty from resampling whole outings. Keep types that appear in only one window and distinguish missing measurements from zero usage. A type absent from the baseline may reflect a classification change or a pitch returning to use; it does not establish that the pitcher learned a new pitch.

The measurement table covers the full eligible MLB population for the 2021-2024 development years, with 2020 available only as prior-season history. It includes summaries of interval width by sample size and Detroit/Cincinnati examples. Release measurements and batter-handedness breakdowns can follow once these measurements pass the reliability checks. All features use pitches thrown before the forecast date.

The first measurement stage runs with the existing downloaded data:

```bash
uv run python3 -m pcf.changes
```

It writes `changes.parquet`, eligible forecast metadata, coverage counts, interval-width summaries, a Detroit/Cincinnati CSV, and `report.md` under `artifacts/experiments/per_type_changes/`. Each run records its settings and source/data fingerprints. Use `--output artifacts/experiments/<run-name>` for another run; the command refuses to overwrite an existing output directory. It accepts development seasons only and does not fit models or rescore the original study.

This stage reports unshrunk, pitch-weighted means. Recent pitches come from days 1-30 before the forecast; the baseline uses days 31-120 within that season when there are at least two pitcher outings, otherwise the prior season if available. If neither is available, it retains the sparse earlier data. Each row records the baseline source and observed date range. This is a separate measurement policy from the frozen study's pooled-pitch fallback; two outings permit resampling but do not establish reliability.

Physical intervals resample whole outings where that pitch type and measurement were observed. Usage intervals resample complete repertoires across all outings, with unknown types and missing velocity kept in the denominator. Intervals are unavailable with fewer than two contributing outings in either window, and usage at zero or one also has no interval. The synthetic-null checks below found poor calibration, so these pointwise summaries remain descriptive; they do not establish that a change is real or lasting.

### Check reliability and persistence

Estimate noise by per-type sample size and compare measurements across separate outings, then test whether changes remain in the following 30 days. Develop shrinkage and practical thresholds using chronological development folds. Choose effect sizes before evaluation, state their rationale, and report sensitivity across thresholds. Synthetic-null checks describe behavior under the simulated null; their flag rate is not an observed real-world false-alert probability.

The development reliability and physical-persistence analysis runs after the measurement stage:

```bash
uv run python3 -m pcf.reliability
```

It uses each pitcher's first eligible date and then dates at least 30 days apart, leaving 6,594 analysis dates. For the independent-halves check, it splits both recent and baseline outings into separate halves before comparing change estimates. Splitting only the recent window would leave shared baseline noise in both estimates. The output includes correlations, disagreement, sample counts, and pitcher-bootstrap intervals by half-window pitch count.

The synthetic-null check samples up to 400 contexts, balanced across development year, metric, and sample-size bin, among known pitch labels with an available measurement interval. For each context, it generates 100 pairs of pseudo windows from the same baseline-outing pool and uses 200 bootstrap draws to form each interval. This tests the interval procedure under an exchangeable-outing simulation, conditional on observed support. Unsupported simulations and zero-width intervals stay visible, along with Monte Carlo uncertainty. These settings define a bounded diagnostic, not a success threshold or a population false-alert estimate.

Shrinkage pulls a measured change toward zero according to its estimated outing-bootstrap noise and the variation in changes from earlier seasons. The chronological comparisons are 2021 to 2022, 2021-2022 to 2023, and 2021-2023 to 2024; 2021 supplies calibration data only. A type-specific prior needs two distinct training pitchers, otherwise it falls back to the metric-wide pool when available. That requirement permits estimation and does not establish reliability. The saved prior table records its source and training seasons.

Physical persistence is evaluated by predicting the next 30 days' measurements with the unchanged baseline, the unchanged recent state, and the baseline plus shrunk change. All three use the same supported rows. Missing future appearances remain unresolved; future appearances without a particular type imply zero usage but no physical measurement for that type. Direction agreement against a shared baseline is not the main result because baseline noise can inflate it.

Outputs live under `artifacts/experiments/reliability/`: per-row measurements and follow-up, anchor dates, priors, coverage, independent-halves results, synthetic-null results, paired physical prediction errors, and a report. The command checks source and input fingerprints and refuses to overwrite an existing run. Use `--output artifacts/experiments/<run-name>` for another run. These are development diagnostics; they do not evaluate strikeout or walk forecasts or establish a minimum sample size for an alert.

### Development findings

The nominal 95% measurement intervals excluded zero in 6,367 of 39,441 available synthetic-null intervals (16.1%), with 559 unavailable trials retained in coverage. Among sampled contexts with only two outings in the smaller window, 33.1% of available intervals excluded zero; with ten or more, the rate was 8.8%. These are conditional simulation results from a balanced sample, not population estimates or real-world false-alert rates. They show that the current interval procedure needs calibration, and large pitch counts alone do not resolve the problem. No safe outing cutoff follows from this test.

For 2022-2024, the comparison below uses 16,022 paired rows per physical metric and 17,381 usage rows, out of 23,077 known-label candidate rows per metric. Missing baselines, measurement support, priors, and future observations remain in coverage. Lower squared error is better; each row has its own units.

| Future measurement | Recent-state MSE | Shrunk MSE | Shrunk minus recent, 95% pitcher interval |
|---|---:|---:|---|
| Velocity (mph squared) | 0.54504 | 0.53237 | -0.01267 (-0.02346 to -0.00080) |
| Horizontal movement (inches squared) | 1.29693 | 1.25900 | -0.03793 (-0.07632 to +0.00336) |
| Vertical movement (inches squared) | 1.34751 | 1.33241 | -0.01510 (-0.05052 to +0.02299) |
| Usage (fraction squared) | 0.004513 | 0.004494 | -0.000019 (-0.000120 to +0.000098) |

Velocity error falls by about 2.3% against carrying forward the recent mean. The other intervals include zero. These exploratory comparisons condition on the fitted priors, use pitcher resampling, and do not fully account for shared calendar effects or prior-estimation uncertainty. Better physical-mean prediction does not establish persistent underlying skill or better strikeout/walk forecasts.

The interval-method comparison below preserves this negative result and tests one alternative. Per-type intervals that exclude zero remain diagnostic.

### Compare interval methods

The fixed design is in [experiments/interval_calibration.json](experiments/interval_calibration.json). Run it after the reliability analysis:

```bash
uv run python3 -m pcf.calibration
```

The two methods are the existing percentile bootstrap and a Welch-style t interval using a finite-outing variance estimate for each pitch-weighted mean. With equal pitch counts per outing, the implementation matches SciPy's independent-sample Welch interval. Unequal counts use a ratio-mean approximation; the simulation tests that approximation rather than assuming it is valid. SciPy is already present in the frozen dependency lock; this experiment changes neither the original environment nor the frozen study.

The declared grid has nine balanced or unequal outing-count pairs, three distributions, fixed or informative pitch counts, and shifts of zero or one latent outing standard deviation. Counts are generated before the shift, so the true population pitch-weighted difference stays known. The nonzero shift measures sensitivity; it is not a baseball alert threshold. There are 500 repetitions per cell, giving 54,000 paired known-truth trials, plus 40,000 trials from the same 400 empirical contexts. Both methods see identical samples, with 400 bootstrap draws per interval. A machine-precision rule prevents floating-point roundoff from becoming a measured effect.

Welch intervals are unavailable when estimated variance is zero, while percentile point intervals remain visible. Both methods retain boundary-usage cases as unavailable. The saved results report availability, true-parameter coverage, zero exclusions, interval width, unavailable reasons, and comparisons on exactly the same supported trials. Interval widths are not clipped to the parameter range, so a very wide interval cannot look useful merely because it includes the truth.

In the empirical pools, the comparison was:

| Fewest outings in either window | Percentile available / trials | Welch available / trials | Zero excluded, percentile | Zero excluded, Welch |
|---|---:|---:|---:|---:|
| 2 | 4,401 / 4,500 | 4,112 / 4,500 | 33.6% | 25.3% |
| 3 | 7,694 / 7,800 | 7,686 / 7,800 | 19.4% | 9.3% |
| 4-5 | 14,884 / 15,000 | 14,884 / 15,000 | 13.4% | 6.2% |
| 6-9 | 8,099 / 8,200 | 8,099 / 8,200 | 9.7% | 6.0% |
| 10+ | 4,388 / 4,500 | 4,388 / 4,500 | 7.8% | 5.7% |

Zero-exclusion percentages use available intervals; unavailable trials stay in the adjacent counts and separate all-trial summaries. These are sample-specific conditional-null results, not real-world false-alert rates. On the 39,169 trials supported by both methods, Welch intervals were a median 1.29 times as wide as percentile intervals (90th percentile: 2.18 times).

The known-truth stress tests still expose failures: with ten recent and two baseline outings from a symmetric two-point distribution and fixed pitch counts, Welch coverage was only 56.4%, with all 500 intervals available. Under skewed data with informative counts, coverage fell to 61.0% in the 25/3-outing case. Better results in the empirical pools therefore do not establish a general calibration fix, and no safe outing cutoff or winning method is selected.

Outputs are under `artifacts/experiments/interval_calibration/`, including every simulated interval, cell-level coverage and sensitivity summaries, support reasons, the design and input fingerprints, and a report. The command refuses to overwrite a run. This completes the declared grid; further methods would need a separate design. Both interval methods remain descriptive. Background: [SciPy's Welch reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html) and [research on inference with clustered errors](https://www.nber.org/papers/t0344); neither validates this baseball application.

### Compare review lists

The fixed design is in [experiments/review_lists.json](experiments/review_lists.json):

```bash
uv run python3 -m pcf.review_lists
```

Each list ranks pitchers on the 2022-2024 reliability dates using only information before the forecast. The physical list uses the largest shrunk velocity or movement change, scaled by its earlier-season prior. The comparators use the largest raw velocity change and the largest raw usage change. All three need at least two outings with the pitch type in both windows and flag the same number of pitchers each season. A flag persists when the next 30 days stay at least halfway from the old baseline toward the recent level on the triggering metric. Flags without two later outings stay in the counts as unresolved.

| Review capacity | Physical, persisted / resolved | Velocity-only | Usage-only | Persisted / all flags (physical, velocity, usage) |
|---:|---:|---:|---:|---|
| 1% (46 per list) | 97.0% | 84.2% | 83.7% | 69.6%, 69.6%, 78.3% |
| 5% (231) | 91.6% | 85.6% | 77.7% | 75.3%, 69.7%, 72.3% |
| 10% (462) | 89.1% | 82.9% | 74.6% | 76.0%, 67.1%, 68.2% |
| 20% (925) | 83.5% | 80.3% | 72.6% | 71.6%, 66.8%, 65.4% |
| All eligible (4,624) | 72.4% | 69.3% | 62.0% | 63.4%, 60.2%, 56.4% |

At 5% and 10% capacity, the physical list's resolved rate was higher than velocity-only by about 6 points (pitcher-bootstrap intervals +0.4 to +12.5 and +2.2 to +9.9). At 1%, 2%, and 20% the intervals approach or cross zero. Part of the gap with usage-only is a difference between metrics: across all eligible dates, physical triggers already persist 10 points more often than usage triggers. Physical flags are also unresolved more often, because a pitcher who stops throwing a pitch gives no physical measurement for it. When unresolved flags count as failures, usage-only leads at 1% and physical ties velocity-only; at 5-20% the physical advantage narrows to about 3-9 points. The 0.25 and 0.75 thresholds mostly keep the same order, but velocity-only slightly edges the physical list at 2% (0.25 threshold) and 20% (0.75 threshold).

Retention is measured against the same baseline used to compute the change, so baseline noise inflates every rate. The numbers compare lists with each other; they are not probabilities that a change is real. Lists are held fixed in the bootstrap. A persistent physical change is a separate finding from an improvement in strikeouts or walks. Outputs are under `artifacts/experiments/review_lists/`.

### Test forecasting value

Compare per-type change features with a matched baseline that has the same current pitch measurements and results history. Include simple velocity-change and historical baselines. Examine paired forecast losses and directional errors, with workload, measurement support, and missing follow-up reported. Any subgroup findings are exploratory.

Run one predeclared nonlinear comparison to test thresholds and interactions, with a bounded tuning budget and the same baseline information. Report it even if the linear comparison finds no benefit. Future whiffs per swing can support the analysis, but their noise must be measured and they do not replace the K% and BB% targets. Historical whiff rate is already part of the results baseline.

The fixed design is in [experiments/forecast_value.json](experiments/forecast_value.json):

```bash
uv run python3 -m pcf.forecast_value
```

Model P adds eight per-type features to the original model B: shrunk velocity and movement changes for the primary fastball, usage-weighted velocity and movement changes for the other pitches, the total usage shift, usage of newly appearing pitches, and the review-list score. Shrinkage priors come from each fold's training seasons only. Model V adds only the original fastball velocity change to B. The nonlinear pair fits gradient-boosted trees on B's features (NB) and P's features (NP) with the same three-setting budget. Two chronological folds test 2023 and 2024; each model picks its setting on the preceding season and refits on all training seasons. Of 13,642 eligible forecasts, 859 had no follow-up and are not scored; 35 used the original repertoire fallback to model A.

Pooled over 12,783 scored 2023-2024 forecasts (log loss per batter faced; negative favors the first model):

| Target | Comparison | Difference (95% pitcher interval) | Changed forecasts moving toward the outcome |
|---|---|---|---:|
| K% | P - B | -0.000010 (-0.000038 to +0.000017) | 52.6% |
| K% | V - B | -0.000012 (-0.000029 to +0.000003) | 52.0% |
| K% | P - V | +0.000002 (-0.000020 to +0.000025) | 51.7% |
| K% | NP - NB | -0.000001 (-0.000023 to +0.000020) | 52.0% |
| BB% | P - B | +0.000006 (-0.000026 to +0.000037) | 50.1% |
| BB% | V - B | +0.000005 (-0.000015 to +0.000026) | 50.6% |
| BB% | NP - NB | +0.000009 (-0.000009 to +0.000028) | 49.9% |

Every interval includes zero, in both folds and in the exploratory starter, reliever, and fastball-support subgroups. Every pooled difference is below 0.01% of baseline loss. Per-type features did no better than the single fastball velocity change, and the nonlinear model found no thresholds or interactions that the linear model missed. The original group-level C was slightly worse than B for strikeouts, as in the 2025 holdout. Inner validation chose the strongest regularization and the smallest tree budget on the grid for nearly every model, which suggests the extra features carry little signal; the stop rule keeps the grid fixed. Whiffs per swing were not added as a supporting target. Outputs are under `artifacts/experiments/forecast_value/`.

### Evaluate on a separate season

Use 2021-2024 for development with chronological folds and a gap that keeps training outcome windows from crossing evaluation cutoffs. Revised analyses of 2025 are descriptive because that season's results have already been inspected. Preserve the original 2025 evaluation as the result of the original specification.

Before inspecting 2026 outcomes, freeze the features, targets, thresholds, model comparisons, and measurement-compatibility policy. Treat 2026 as a temporal and domain test: audit baseline calibration and coverage for both strikeouts and walks, then score the frozen comparisons once and report the results regardless of direction. If its outcomes have already been inspected during development, it cannot serve as an untouched test.

### Present the evidence

Evaluate MLB-wide before choosing examples. Add a browser team filter and Detroit/Cincinnati pages that distinguish the measured change, its uncertainty, later persistence, and forecast impact. Include unsuccessful and unresolved examples, and label cases selected after inspecting outcomes. Team attribution uses the last team the pitcher appeared for before each forecast.

New analysis outputs belong under `artifacts/experiments/`, separate from the frozen study, with their own configuration and source fingerprints. Individual forecast intervals remain deferred. The write-up should follow the measured results, including the possibility that changes are difficult to measure, do not persist, or add little forecasting value. Publication of the historical browser remains a separate task.

## Related work

- [Stuff+, Location+, and Pitching+](https://library.fangraphs.com/pitching/stuff-location-and-pitching-primer/): existing pitch-quality and performance models.
- [Unified Pitch Graphs](https://arxiv.org/abs/2609.03810): retrospective analysis of pitching strategy and execution changes.
- [Baseball Savant pitch arsenals](https://baseballsavant.mlb.com/pitchmix): existing repertoire visualizations.

The project investigates the incremental forecasting value of recent changes. It makes no claim that pitcher forecasting or change detection is new.
