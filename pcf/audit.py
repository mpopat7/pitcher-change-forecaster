"""Check saved Statcast chunks against retrieval logs and completed MLB games."""

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import requests

from pcf.ingest import COLUMNS, RAW, ROW_CAP, SEASONS, windows

SCHEDULES = Path("data/raw/schedules")


def audit(seasons, fetch_schedules=False):
    log = {r["file"]: r for line in (RAW / "retrieval_log.jsonl").read_text().splitlines()
           if (r := json.loads(line))}
    report = {}
    for season in seasons:
        schedule_path = SCHEDULES / f"{season}.json"
        if fetch_schedules:
            response = requests.get("https://statsapi.mlb.com/api/v1/schedule",
                                    params={"sportId": 1, "season": season, "gameType": "R"},
                                    timeout=60)
            response.raise_for_status()
            SCHEDULES.mkdir(parents=True, exist_ok=True)
            schedule_path.write_text(json.dumps(response.json()))
        schedule = json.loads(schedule_path.read_text())
        expected_games = {g["gamePk"] for day in schedule["dates"] for g in day["games"]
                          if g["gameType"] == "R" and g["status"]["codedGameState"] == "F"}
        issues, chunks = [], []
        for start, end in windows(*SEASONS[season]):
            path = RAW / str(season) / f"{start}_{end}.parquet"
            if not path.exists():
                issues.append(f"Missing chunk: {path.name}")
                continue
            metadata = pq.ParquetFile(path)
            count = metadata.metadata.num_rows
            if log.get(str(path), {}).get("rows") != count:
                issues.append(f"Retrieval log mismatch: {path.name}")
            if count >= ROW_CAP:
                issues.append(f"Row cap reached: {path.name}")
            if not count:
                continue
            missing = sorted(set(COLUMNS) - set(metadata.schema_arrow.names))
            if missing:
                issues.append(f"Missing columns in {path.name}: {missing}")
            d = pd.read_parquet(path, columns=["game_pk", "game_date", "game_type",
                                               "at_bat_number", "pitch_number"])
            dates = pd.to_datetime(d.game_date)
            if not dates.between(pd.Timestamp(start), pd.Timestamp(end)).all():
                issues.append(f"Dates outside chunk bounds: {path.name}")
            if not d.game_type.eq("R").all():
                issues.append(f"Non-regular-season rows: {path.name}")
            chunks.append(d)
        keys = ["game_pk", "at_bat_number", "pitch_number"]
        pitches = (pd.concat(chunks, ignore_index=True) if chunks
                   else pd.DataFrame(columns=keys))
        duplicates = int(pitches.duplicated(keys).sum())
        if duplicates:
            issues.append(f"Duplicate pitch keys: {duplicates}")
        found_games = set(pitches.game_pk)
        missing_games = sorted(expected_games - found_games)
        extra_games = sorted(found_games - expected_games)
        if missing_games:
            issues.append(f"Missing completed games: {missing_games}")
        if extra_games:
            issues.append(f"Games absent from completed schedule: {extra_games}")
        report[season] = {"pitches": len(pitches), "games": len(found_games),
                          "expected_games": len(expected_games), "issues": issues}
        print(f"{season}: {len(pitches):,} pitches, {len(found_games)}/{len(expected_games)} games, "
              f"{len(issues)} issues", flush=True)
        for issue in issues:
            print(f"  {issue}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", type=int, nargs="+", default=list(SEASONS))
    parser.add_argument("--fetch-schedules", action="store_true")
    args = parser.parse_args()
    report = audit(args.seasons, args.fetch_schedules)
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/data_audit.json").write_text(json.dumps(report, indent=2))
    raise SystemExit(int(any(s["issues"] for s in report.values())))
