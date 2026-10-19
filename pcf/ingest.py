"""Download regular-season Statcast pitches from Baseball Savant into per-chunk parquet files.

Savant returns at most 25,000 rows per request, and a full day is roughly 4,500 pitches, so the
season is fetched in 4-day windows. Finished chunks are skipped on rerun.
"""

import argparse
import io
import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

URL = "https://baseballsavant.mlb.com/statcast_search/csv"
ROW_CAP = 25_000
CHUNK_DAYS = 4
RAW = Path("data/raw/statcast")

# Regular-season windows, padded a few days on each side. 2020 (60 games) is downloaded only as
# prior-season history for 2021 forecasts.
SEASONS = {
    2020: ("2020-07-20", "2020-09-30"),
    2021: ("2021-03-28", "2021-10-05"),
    2022: ("2022-04-05", "2022-10-07"),
    2023: ("2023-03-28", "2023-10-03"),
    2024: ("2024-03-18", "2024-10-02"),
    2025: ("2025-03-16", "2025-09-30"),
}

COLUMNS = [
    "game_pk", "game_date", "game_year", "game_type", "at_bat_number", "pitch_number",
    "pitcher", "player_name", "batter", "p_throws", "stand", "home_team", "away_team",
    "inning_topbot", "balls", "strikes", "outs_when_up",
    "pitch_type", "release_speed", "release_spin_rate", "spin_axis", "pfx_x", "pfx_z",
    "release_pos_x", "release_pos_z", "release_extension", "arm_angle",
    "plate_x", "plate_z", "sz_top", "sz_bot", "zone",
    "description", "events", "type",
    "n_thruorder_pitcher", "pitcher_days_since_prev_game",
]


def fetch(start: date, end: date) -> pd.DataFrame:
    params = {
        "all": "true", "type": "details", "player_type": "pitcher", "hfGT": "R|",
        "game_date_gt": start.isoformat(), "game_date_lt": end.isoformat(),
    }
    # Patient backoff (about 25 minutes in total) so a dropped connection doesn't end the run.
    attempts = 8
    for attempt in range(attempts):
        try:
            r = requests.get(URL, params=params, timeout=180)
            r.raise_for_status()
            df = pd.read_csv(io.BytesIO(r.content), low_memory=False)
            break
        except (requests.RequestException, pd.errors.ParserError) as exc:
            if attempt == attempts - 1:
                raise
            print(f"  retry {start}: {exc}", flush=True)
            time.sleep(min(30 * 2 ** attempt, 600))
    if len(df) >= ROW_CAP:
        raise RuntimeError(f"{start}..{end} hit the {ROW_CAP}-row cap; shrink CHUNK_DAYS")
    return df[[c for c in COLUMNS if c in df.columns]]


def windows(first: str, last: str):
    cur, stop = date.fromisoformat(first), date.fromisoformat(last)
    while cur <= stop:
        end = min(cur + timedelta(days=CHUNK_DAYS - 1), stop)
        yield cur, end
        cur = end + timedelta(days=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+", default=list(SEASONS))
    args = ap.parse_args()

    log_path = RAW / "retrieval_log.jsonl"
    for season in args.seasons:
        out_dir = RAW / str(season)
        out_dir.mkdir(parents=True, exist_ok=True)
        for start, end in windows(*SEASONS[season]):
            out = out_dir / f"{start}_{end}.parquet"
            if out.exists():
                continue
            df = fetch(start, end)
            df.to_parquet(out, index=False)
            with log_path.open("a") as log:
                log.write(json.dumps({
                    "file": str(out), "rows": len(df), "game_type": "R",
                    "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }) + "\n")
            print(f"{season} {start}..{end}: {len(df):>6} pitches", flush=True)
            time.sleep(1)


if __name__ == "__main__":
    main()
