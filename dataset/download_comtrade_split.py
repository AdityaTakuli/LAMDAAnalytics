#!/usr/bin/env python3
"""Fallback for reporter-years whose full-year Comtrade reply keeps dropping.

Large reporters (e.g. Canada, Turkiye, Switzerland) return replies big enough
that the proxied connection is closed mid-transfer. This fetches the same
request in quarters (and single months if a quarter still fails), merges the
records, and saves them under the same reporter-year cache file and format as
the full-year request, so the panel builders read them unchanged.

Usage: python download_comtrade_split.py extended_2021_2024|extended_2019_2025
"""
from __future__ import annotations

import sys
import time

import download_consumer_basket as dc
from download_comtrade_2021_2024 import HS_ALL, NEW_REPORTERS
from download_comtrade_2019_2025 import REPORTERS as PHASED_REPORTERS
from download_comtrade_2019_2025 import YEARS as PHASED_YEARS


def fetch_split(reporter: str, year: int) -> dict:
    months = [f"{year}{m:02d}" for m in range(1, 13)]
    data = []
    for q in range(4):
        part = months[3 * q:3 * q + 3]
        try:
            data += dc.request(reporter, part, tries=3)["data"]
        except RuntimeError:
            for m in part:  # last resort: one month at a time
                data += dc.request(reporter, [m], tries=4)["data"]
        time.sleep(1)
    return {"elapsedTime": "split", "count": len(data), "data": data, "error": ""}


def main() -> int:
    phase = sys.argv[1]
    dc.HS = HS_ALL
    dc.CACHE = dc.BASE / f"data/four_year_2021_2024/cache/comtrade/{phase}"
    reporters, years = (NEW_REPORTERS, dc.YEARS) if phase == "extended_2021_2024" else (PHASED_REPORTERS, PHASED_YEARS)
    todo = [(r, y) for r in reporters for y in years if not dc.done(dc.cache_file(r, str(y)))]
    print(f"{phase}: {len(todo)} reporter-years to fetch in pieces", flush=True)
    for r, y in todo:
        if dc.done(dc.cache_file(r, str(y))):
            continue
        try:
            payload = fetch_split(r, y)
            dc.save(dc.cache_file(r, str(y)), r, [f"{y}{m:02d}" for m in range(1, 13)], payload)
            print(f"{reporters[r]} ({r}) {y}: {len(payload['data'])} rows, "
                  f"{len({x['period'] for x in payload['data']})} months (split)", flush=True)
        except RuntimeError as exc:
            print(f"FAILED {r} {y}: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
