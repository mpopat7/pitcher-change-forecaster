"""Export the saved study to a static browser and serve it on localhost.

Run: python3 -m pcf.browser [--build-only] [--port 8765]
"""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import shutil

import pandas as pd

from pcf.freeze import SPEC, file_hash, load_frozen_spec

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "artifacts/browser"
TRAITS = ["team", "days_since_last", "fb_n_cur", "fb_velo_cur", "chg_fb_velo",
          "fb_ivb_cur", "chg_fb_ivb", "fb_hb_cur", "chg_fb_hb", "chg_baseline_is_ps",
          *(f"use_{g}_cur" for g in ("fb", "cut", "brk", "off", "other")),
          *(f"chg_use_{g}" for g in ("fb", "cut", "brk", "off", "other"))]


def season_payload(watchlist, context, alerts):
    if watchlist.duplicated(["pitcher", "forecast_date"]).any():
        raise ValueError("Duplicate forecast keys in the saved watchlist")
    merged = watchlist.merge(context, on=["pitcher", "forecast_date"],
                             how="outer", validate="one_to_one", indicator=True)
    if not merged._merge.eq("both").all():
        raise ValueError("Saved forecasts and as-of context have different eligible populations")
    merged = merged.drop(columns="_merge")
    resolutions = alerts[["pitcher", "forecast_date", "resolution"]]
    merged = merged.merge(resolutions, on=["pitcher", "forecast_date"],
                           how="left", validate="one_to_one")
    if merged.loc[merged.alert, "resolution"].isna().any():
        raise ValueError("An alert is missing its saved resolution")
    # The browser computes display rates only; forecast values and ranks come from saved outputs.
    merged["observed_k"] = merged.y_k.div(merged.y_bf.where(merged.y_bf > 0))
    merged["observed_bb"] = merged.y_bb.div(merged.y_bf.where(merged.y_bf > 0))
    merged["observed_k_minus_bb"] = merged.observed_k - merged.observed_bb
    merged["error_pp"] = (merged.k_minus_bb_B - merged.observed_k_minus_bb).abs() * 100
    merged = merged.drop(columns=[c for c in merged if c.endswith("_raw")])
    merged = merged.sort_values(["forecast_date", "performance_rank"])
    payload = json.loads(merged.to_json(orient="split", index=False, double_precision=10))
    payload["dates"] = sorted(merged.forecast_date.unique().tolist())
    return payload


def build(output=OUTPUT):
    load_frozen_spec()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = output / "data"
    data.mkdir(exist_ok=True)
    manifest = {"spec_sha256": file_hash(SPEC), "seasons": {}}
    bundled = {}
    for season, tag in ((2024, "validation_2024"), (2025, "holdout_2025")):
        artifacts = ROOT / "artifacts"
        watchlist = pd.read_csv(artifacts / f"watchlist_{tag}.csv")
        alerts = pd.read_csv(artifacts / f"alerts_{tag}.csv")
        context = pd.read_parquet(ROOT / "data/processed/forecast_rows.parquet",
                                  columns=["pitcher", "forecast_date", *TRAITS],
                                  filters=[("season", "==", season)])
        context["forecast_date"] = context.forecast_date.dt.strftime("%Y-%m-%d")
        payload = season_payload(watchlist, context, alerts)
        bundled[f"data/{season}.json"] = payload
        (data / f"{season}.json").write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False))
        manifest["seasons"][str(season)] = {
            "label": "Held-out evaluation" if season == 2025 else "Development validation",
            "dates": payload["dates"], "rows": len(payload["data"]),
            "report": json.loads((artifacts / f"report_{tag}.json").read_text()),
            "watchlist": json.loads((artifacts / f"watchlist_{tag}.json").read_text()),
        }
    (data / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":"), allow_nan=False))
    for source in (ROOT / "web").iterdir():
        if source.is_file() and source.suffix in {".html", ".css", ".js", ".mjs"}:
            shutil.copy2(source, output / source.name)
    bundled["data/manifest.json"] = manifest
    html = (ROOT / "web/index.html").read_text()
    html = html.replace('<link rel="stylesheet" href="styles.css">',
                         f"<style>{(ROOT / 'web/styles.css').read_text()}</style>")
    html = html.replace('<script type="module" src="app.js"></script>', "")
    scripts = []
    for name in ("data.mjs", "charts.mjs", "app.js"):
        script = (ROOT / "web" / name).read_text()
        script = re.sub(r"^import .*?;\n", "", script, flags=re.MULTILINE)
        script = re.sub(r"^export (?=const |function )", "", script, flags=re.MULTILINE)
        scripts.append(script)
    inline_data = json.dumps(bundled, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    html = html.replace("</body>", f'<script type="application/json" id="saved-study">{inline_data}</script>'
                        + '<script type="module">' + "\n".join(scripts) + "</script></body>")
    (output / "standalone.html").write_text(html)
    print(f"Built {sum(s['rows'] for s in manifest['seasons'].values()):,} saved forecasts in {output}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-only", action="store_true", help="export static files without serving")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    build()
    if not args.build_only:
        handler = partial(SimpleHTTPRequestHandler, directory=str(OUTPUT))
        server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
        print(f"Local: http://127.0.0.1:{args.port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.server_close()
