# A short browser walkthrough

Open the forecast archive with outcomes hidden. Each row is a forecast made at a historical cutoff using information available by that date; the target is strikeouts and walks per batter faced over the following 30 calendar days.

## Start with the question

Do recent changes in velocity, movement, and pitch usage help predict next-month performance beyond recent results and current pitch characteristics? A uses recent results and context, B adds current pitches, and C adds recent changes. The main test is C versus B.

## Inspect a forecast

In the 2025 season, select **Alerts**, then **Logan Webb on April 14**. B projects 21.8% strikeouts and 5.9% walks, or 15.9% K% minus BB%. C projects 18.0% K% minus BB%, producing an experimental improvement alert of 2.1 percentage points. Inspect his pitch measurements and mix before revealing the outcome.

Turn on **Reveal outcomes**. Webb records 24.8% K% minus BB% over 129 batters faced, so this alert confirms under the study's definition. His recent results already pointed in the same direction; this example does not establish advance warning.

## Show a failure and an unresolved case

Select **Robbie Ray on April 14**. B projects 11.5% K% minus BB%, while C projects 9.1%, triggering a decline alert. His observed result is 17.7% over 147 batters faced, so the alert is false.

Then select **Matt Strahm on April 28**. His alert is unresolved because he faces only 45 batters over the target window, below the fixed 50-BF resolution requirement. Unresolved cases stay in the totals.

## Return to the full study

The 2025 holdout contains 6,734 eligible forecasts, including 381 with no follow-up. Adding recent-change features slightly worsens strikeout accuracy and gives no clear walk improvement. Current pitch characteristics provide a small strikeout gain over results alone.

The alert view has 19 cases: eight confirmed, five false, and six unresolved. These examples help inspect individual forecasts; the full-population comparison determines the research finding. The browser ranks pitchers using B, with C's alerts retained as experimental review candidates.

## Close with the limits

These are historical point forecasts. Individual forecast intervals are not estimated, weekly target windows overlap, and predictive associations do not establish a coaching effect. The specification was frozen before the 2025 evaluation. Any further model development needs a new evaluation period.
