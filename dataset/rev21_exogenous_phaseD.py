#!/usr/bin/env python3
"""Phase D: GSCPI and NASA POWER weather for 2019-2025 (all 44 configured locations).

GSCPI: download the NY Fed spreadsheet, normalise with ingest_gscpi.normalize
(same month-end + 7 day availability rule). The published 2021-2024 values are
kept exactly as published; only months outside 2021-2024 come from the new
download, and any revision of 2021-2024 values is reported, not applied.
Writes processed_v4/gscpi_monthly.csv and processed_v4/gscpi_revision_check.csv.

Weather: ingest_weather._nasa / _rows for 2019-01-01..2025-12-31 per location
(new cache files; the 2021-2024 rows are taken from the published/Phase B
tables unchanged), availability = observed + 2 days as in repair_weather.
Writes processed_v4/weather_daily.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ingest_gscpi as ig  # noqa: E402
import ingest_weather as iw  # noqa: E402
from common import load_config, read_table, request_headers, stable_id, vintage_now  # noqa: E402

BASE = Path(__file__).resolve().parent
ROOT = BASE / "data/four_year_2021_2024"
V4 = ROOT / "processed_v4"
START, END = pd.Timestamp("2019-01-01", tz="UTC"), pd.Timestamp("2025-12-31", tz="UTC")


def gscpi(config) -> None:
    cache = ROOT / "cache/gscpi_phaseD"
    cache.mkdir(parents=True, exist_ok=True)
    xlsx = cache / "gscpi_data.xlsx"
    if not xlsx.exists():
        r = requests.get(config["sources"]["gscpi"]["excel_url"], headers=request_headers(), timeout=120)
        r.raise_for_status()
        xlsx.write_bytes(r.content)
    fresh = ig.normalize(xlsx, vintage_now())
    fresh = fresh[fresh["month"].between("2019-01", "2025-12")]
    pub = read_table(ROOT / "processed/gscpi_monthly.csv")
    pub["month"] = pub["month"].astype(str)
    chk = pub[["month", "global_risk"]].merge(fresh[["month", "global_risk"]], on="month", suffixes=("_published", "_current"))
    chk["revision"] = chk["global_risk_current"] - chk["global_risk_published"]
    chk.to_csv(V4 / "gscpi_revision_check.csv", index=False)
    print(f"GSCPI 2021-2024 revisions since publication: max |change| {chk['revision'].abs().max():.4f}")
    out = pd.concat([fresh[~fresh["month"].isin(set(pub["month"]))], pub], ignore_index=True).sort_values("month")
    out.to_csv(V4 / "gscpi_monthly.csv", index=False)
    print(f"GSCPI months: {out['month'].min()}..{out['month'].max()} ({len(out)})")


def weather(config) -> None:
    locations = config["sources"]["weather"]["locations"]
    cache = BASE / config["sources"]["weather"]["cache_dir"] / "nasa_power"
    vintage = vintage_now()
    rows = []
    for loc in locations:
        raw = cache / f"{stable_id('location', loc['name'])}_{START:%Y%m%d}_{END:%Y%m%d}.json"
        payload = iw._nasa(loc, START, END, raw, force=False)
        got = iw._rows(loc, payload, "nasa_power", vintage)
        rows += got
        print(f"{loc['name']:22s} {len(got)} days", flush=True)
    new = pd.DataFrame(rows)
    observed = pd.to_datetime(new["observed_date"], utc=True)
    new["published_at"] = (observed + pd.Timedelta(days=2)).map(lambda v: v.isoformat())
    new["available_at"] = new["published_at"]
    base = read_table(ROOT / "processed_v3/weather_daily.csv")   # published + Phase B rows for 2021-2024
    in_base = pd.to_datetime(new["observed_date"], utc=True).dt.year.between(2021, 2024)
    out = pd.concat([base, new.loc[~in_base, base.columns]], ignore_index=True)
    out.to_csv(V4 / "weather_daily.csv", index=False)
    print(f"weather: {len(base)} rows 2021-2024 kept + {int((~in_base).sum())} rows for 2019-2020/2025")


def main() -> int:
    V4.mkdir(parents=True, exist_ok=True)
    config = load_config("config_phaseB.yaml")
    gscpi(config)
    weather(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
