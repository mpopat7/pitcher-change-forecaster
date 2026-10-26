"""Record the validation-selected model and watchlist before reading holdout outcomes."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

SPEC = Path("research_spec.json")
SOURCES = ["pcf/dataset.py", "pcf/model.py", "pcf/watchlist.py", "pcf/freeze.py",
           "pyproject.toml", "uv.lock"]


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_frozen_spec():
    if not SPEC.exists():
        raise ValueError("Freeze the validated specification with python3 -m pcf.freeze first")
    spec = json.loads(SPEC.read_text())
    for path, expected in spec["source_sha256"].items():
        if file_hash(path) != expected:
            raise ValueError(f"Source changed after freezing: {path}")
    for path, expected in spec["validation_sha256"].items():
        if file_hash(path) != expected:
            raise ValueError(f"Validation changed after freezing: {path}")
    return spec


def freeze():
    from pcf.dataset import HORIZON
    from pcf.model import ARTIFACTS, ROWS, feature_sets
    from pcf.watchlist import RULES
    import pandas as pd

    if list(ARTIFACTS.glob("*holdout_2025*")):
        raise FileExistsError("Holdout outputs exist; the study specification cannot be refrozen")
    if SPEC.exists():
        raise FileExistsError("The specification is already frozen")
    model_path = ARTIFACTS / "report_validation_2024.json"
    watchlist_path = ARTIFACTS / "watchlist_validation_2024.json"
    report, watchlist = json.loads(model_path.read_text()), json.loads(watchlist_path.read_text())
    if report["test_season"] != 2024 or report["train_seasons"] != [2021, 2022, 2023]:
        raise ValueError("Expected the chronological 2024 validation")
    eligible = next(c["eligible"] for c in report["coverage"] if c["season"] == 2024)
    if watchlist["rules"] != RULES or watchlist["eligible"] != eligible:
        raise ValueError("Watchlist validation does not match the rules or eligible population")
    columns = list(pd.read_parquet(ROWS, filters=[("season", "<=", 2024)]).columns) + ["month"]
    spec = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_seasons_final": [2021, 2022, 2023, 2024], "holdout_season": 2025,
        "horizon_days": HORIZON,
        "regularization": {
            t: {m: v["C"] for m, v in r["models"].items()} for t, r in report["targets"].items()
        },
        "features": feature_sets(columns),
        "repertoire_rule": report["repertoire_support"]["rule"],
        "repertoire_bounds_final": "refit on all eligible 2021-2024 features",
        "watchlist": RULES,
        "source_sha256": {path: file_hash(path) for path in SOURCES},
        "validation_sha256": {str(path): file_hash(path) for path in (model_path, watchlist_path)},
    }
    SPEC.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"Froze model, features, repertoire rule, watchlist rules, and validation hashes in {SPEC}")
    return spec


if __name__ == "__main__":
    freeze()
