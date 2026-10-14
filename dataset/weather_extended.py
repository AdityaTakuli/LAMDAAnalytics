#!/usr/bin/env python3
"""Extended panel, 2021-2024: NASA POWER daily temperature for the 24 added countries.

Uses ingest_weather._nasa (same request, same cache naming) and
ingest_weather._rows, then the same availability rule as
repair_processed_sources.repair_weather (observed + 2 days for NASA POWER).
The published weather table is copied unchanged and the new rows appended:
writes processed_v3/weather_daily.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ingest_weather as iw  # noqa: E402
from common import load_config, read_table, stable_id, vintage_now  # noqa: E402
from download_comtrade_2021_2024 import NEW_REPORTERS  # noqa: E402

BASE = Path(__file__).resolve().parent
ROOT = BASE / "data/four_year_2021_2024"


def main() -> int:
    config = load_config("config_extended_2021_2024.yaml")
    locations = [l for l in config["sources"]["weather"]["locations"] if str(l["comtrade_code"]) in NEW_REPORTERS]
    assert len(locations) == len(NEW_REPORTERS), (len(locations), len(NEW_REPORTERS))
    start = pd.Timestamp(config["analysis"]["start_date"], tz="UTC")
    end = pd.Timestamp(config["analysis"]["end_date"], tz="UTC")
    cache = BASE / config["sources"]["weather"]["cache_dir"] / "nasa_power"
    vintage = vintage_now()
    rows = []
    for loc in locations:
        raw = cache / f"{stable_id('location', loc['name'])}_{start:%Y%m%d}_{end:%Y%m%d}.json"
        payload = iw._nasa(loc, start, end, raw, force=False)
        got = iw._rows(loc, payload, "nasa_power", vintage)
        rows += got
        print(f"{loc['name']:12s} {len(got)} days", flush=True)
    new = pd.DataFrame(rows)
    observed = pd.to_datetime(new["observed_date"], utc=True)
    new["published_at"] = (observed + pd.Timedelta(days=2)).map(lambda v: v.isoformat())
    new["available_at"] = new["published_at"]
    old = read_table(ROOT / "processed/weather_daily.csv")
    out = pd.concat([old, new[old.columns]], ignore_index=True)
    dest = ROOT / "processed_v3/weather_daily.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    print(f"wrote {dest}: {len(old)} published + {len(new)} new rows, {out['node_id'].nunique()} locations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
