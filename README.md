# Pitcher Change Forecaster

Do recent changes in a pitcher's repertoire improve forecasts of his performance over the next month?

This project uses public MLB Statcast data to compare forecasts based on past results, current pitch characteristics, and recent changes in those characteristics. The intended output is a watchlist that helps an analyst decide which apparent pitcher improvements or declines deserve closer review.

## Status

The first 2024 validation is complete. Recent-change features do not show a clear gain over current pitch characteristics, and the results-only model has the lowest overall loss. A rare-repertoire failure accounts for much of the difference; details below. All 2020-2025 data is downloaded and passes the game-level audit. The 2025 holdout remains unscored. There is no interface yet.

## Running it

Requires [uv](https://docs.astral.sh/uv/). Downloaded data and outputs stay in the gitignored `data/` and `artifacts/` folders.

```bash
uv run python3 -m pcf.ingest      # 2020 history plus 2021-2025; resumes saved chunks
uv run python3 -m pcf.audit --fetch-schedules  # compare chunks with completed MLB games
uv run python3 -m pcf.dataset --through-season 2024
uv run python3 -m pcf.model       # fit A/B/C, tune and score on 2024
uv run python3 -m pcf.report      # comparison charts, calibration, and failure reviews
```

The full download is about 270 requests and takes roughly 90 minutes. Keep the machine awake (`caffeinate -i` on macOS); a sleeping laptop stalls it.

To include 2025 in the dataset, run `uv run python3 -m pcf.dataset` without a cutoff. Run `uv run python3 -m pcf.model --final` only after the model specification and watchlist rules are fixed; it refits on 2021-2024 and scores 2025 with the validation-selected settings.

## First validation: 2024

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

Begin with regularized statistical models. Compare boosted trees if nonlinear relationships justify them. Model family, window lengths, and sample thresholds remain open until the data audit and validation work.

## Evaluation

- Proposed split: train on 2021-2023, tune on 2024, and reserve 2025 for the final evaluation.
- Replay forecast dates chronologically. Training outcomes and every input must be available before the relevant cutoff; keep overlapping target windows from crossing evaluation boundaries.
- Compare probability accuracy and calibration, with uncertainty estimates that account for repeated pitchers and time dependence.
- Report performance across workloads and pitcher roles, coverage, and missing follow-up.
- Set watchlist rules using validation data, then measure false alerts and whether useful signals precede changes in conventional results.
- Evaluate the full eligible population before selecting case studies; include failed forecasts.

A finding that change features add no predictive value is a valid research result. Report it alongside the comparison that produced it.

## Planned interface

A ranked pitcher watchlist with next-month forecasts, uncertainty, and the difference from the baseline. Each pitcher view would show pitch measurements over time, supporting outcomes, sample sizes, and historical forecast errors. Model explanations describe predictive associations.

## Next session

1. Define how to flag or handle repertoires outside training support; check the rule with chronological splits inside the training years.
2. Review the revised 2024 comparison and set any watchlist/false-alert rules before freezing the specification.
3. Score the 2025 holdout once. Build the interface around the measured results and remaining limitations.

## Related work

- [Stuff+, Location+, and Pitching+](https://library.fangraphs.com/pitching/stuff-location-and-pitching-primer/): existing pitch-quality and performance models.
- [Unified Pitch Graphs](https://arxiv.org/abs/2609.03810): retrospective analysis of pitching strategy and execution changes.
- [Baseball Savant pitch arsenals](https://baseballsavant.mlb.com/pitchmix): existing repertoire visualizations.

The project investigates the incremental forecasting value of recent changes. It makes no claim that pitcher forecasting or change detection is new.
