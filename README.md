# Pitcher Change Forecaster

Do recent changes in a pitcher's repertoire improve forecasts of his performance over the next month?

This project will use public MLB Statcast data to compare forecasts based on past results, current pitch characteristics, and recent changes in those characteristics. The intended output is a watchlist that helps an analyst decide which apparent pitcher improvements or declines deserve closer review.

## Status

The data pipeline and model code are written; no results yet. 2020-2023 Statcast data has been downloaded and the 2021-2023 forecast rows build and pass hand checks against published season totals. Next is the 2024-2025 download and the first 2024 validation run. The interface follows once there are results to explain.

## Running it

Requires [uv](https://docs.astral.sh/uv/). Downloaded data and outputs stay in the gitignored `data/` and `artifacts/` folders.

```bash
uv run python -m pcf.ingest      # Statcast pitches, 2020 (history only) through 2025; resumes from saved chunks
uv run python -m pcf.dataset     # one row per pitcher per Monday forecast date
uv run python -m pcf.model       # fit A/B/C, tune and score on 2024
uv run python -m pcf.model --final   # score the 2025 holdout once
```

The full download is about 270 requests and takes roughly 90 minutes. Keep the machine awake (`caffeinate -i` on macOS); a sleeping laptop stalls it.

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

1. Finish the 2024-2025 download and check every chunk's row count for gaps.
2. Build the full dataset and hand-check a few 2024 rows.
3. Run the 2024 validation comparison of A, B, and C; review calibration and failure cases.
4. Chart the C-versus-B result with its uncertainty.
5. Score the 2025 holdout once, then start the watchlist interface.

## Related work

- [Stuff+, Location+, and Pitching+](https://library.fangraphs.com/pitching/stuff-location-and-pitching-primer/): existing pitch-quality and performance models.
- [Unified Pitch Graphs](https://arxiv.org/abs/2609.03810): retrospective analysis of pitching strategy and execution changes.
- [Baseball Savant pitch arsenals](https://baseballsavant.mlb.com/pitchmix): existing repertoire visualizations.

The project investigates the incremental forecasting value of recent changes. It makes no claim that pitcher forecasting or change detection is new.
