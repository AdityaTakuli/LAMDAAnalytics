#!/usr/bin/env python3
"""Phase D: fetch the GDELT 1.0 daily event files for 2019-2020 and 2025.

Same source and cache layout as ingest_gdelt.py
(https://data.gdeltproject.org/events/YYYYMMDD.export.CSV.zip ->
cache/gdelt/<year>/YYYYMMDD.export.CSV.zip). Existing valid files are kept;
downloads are written to a temporary name and validated as zip archives
before being moved into place. Direct connection (GDELT is not blocked).
"""
from __future__ import annotations

import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

BASE = Path(__file__).resolve().parent
CACHE = BASE / "data/four_year_2021_2024/cache/gdelt"
URL = "https://data.gdeltproject.org/events/{d}.export.CSV.zip"
DATES = [*pd.date_range("2019-01-01", "2020-12-31"), *pd.date_range("2025-01-01", "2025-12-31")]


def valid(p: Path) -> bool:
    try:
        with zipfile.ZipFile(p) as z:
            return z.testzip() is None and bool(z.namelist())
    except Exception:
        return False


def fetch(day: pd.Timestamp) -> str:
    stamp = day.strftime("%Y%m%d")
    dest = CACHE / day.strftime("%Y") / f"{stamp}.export.CSV.zip"
    if dest.exists() and valid(dest):
        return "cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".zip.tmp")
    for attempt in range(4):
        try:
            with requests.get(URL.format(d=stamp), stream=True, timeout=(20, 120)) as r:
                if r.status_code == 404:
                    return "missing"
                r.raise_for_status()
                with tmp.open("wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
            if valid(tmp):
                tmp.replace(dest)
                return "ok"
        except Exception:
            pass
        time.sleep(5 * (attempt + 1))
    tmp.unlink(missing_ok=True)
    return "failed"


def main() -> int:
    counts: dict[str, int] = {}
    failed = []
    with ThreadPoolExecutor(6) as pool:
        futures = {pool.submit(fetch, d): d for d in DATES}
        for i, fut in enumerate(as_completed(futures), 1):
            status = fut.result()
            counts[status] = counts.get(status, 0) + 1
            if status in ("failed", "missing"):
                failed.append(f"{futures[fut]:%Y-%m-%d} {status}")
            if i % 100 == 0:
                print(f"{i}/{len(DATES)} {counts}", flush=True)
    print(f"finished {counts}", flush=True)
    for line in failed:
        print("  ", line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
