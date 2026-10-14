#!/usr/bin/env python3
"""Phase D: Comtrade imports for 2019, 2020 and 2025, all 41 panel countries, both baskets.

Same request as Phase B (one reporter-year per call, all 13 HS codes, the 20
original partners, imports), so 2019-2020 and 2025 extend every series with
the identical definition. Cached under cache/comtrade/phaseD/ (resumable);
REV21_REVERSE=1 / REV21_MIDDLE=1 let extra workers start elsewhere in the list.
Routes through the local Psiphon proxy.
"""
from __future__ import annotations

import os
import time

import rev21_download_consumer as dc
from rev21_download_phaseB import HS_ALL, NEW_REPORTERS

EXCLUDED = {"642"}  # Romania (no GDELT actor code), as in Phase B
REPORTERS = {**{r: r for r in dc.REPORTERS}, **{k: v for k, v in NEW_REPORTERS.items() if k not in EXCLUDED}}
YEARS = [2019, 2020, 2025]


def main() -> int:
    dc.CACHE = dc.BASE / "data/four_year_2021_2024/cache/comtrade/phaseD"
    dc.HS = HS_ALL
    jobs = [(r, [f"{y}{m:02d}" for m in range(1, 13)], str(y)) for r in REPORTERS for y in YEARS]
    todo = [j for j in jobs if not dc.done(dc.cache_file(j[0], j[2]))]
    if os.environ.get("REV21_REVERSE") == "1":
        todo = todo[::-1]
    if os.environ.get("REV21_MIDDLE") == "1":
        todo = todo[len(todo) // 2:] + todo[:len(todo) // 2]
    print(f"{len(jobs)} requests, {len(jobs) - len(todo)} cached, {len(todo)} to fetch", flush=True)
    failures = []
    for i, (reporter, periods, label) in enumerate(todo, 1):
        if dc.done(dc.cache_file(reporter, label)):
            continue
        try:
            payload = dc.request(reporter, periods)
            dc.save(dc.cache_file(reporter, label), reporter, periods, payload)
            got = sorted({r["period"] for r in payload["data"]})
            print(f"{i}/{len(todo)} {REPORTERS[reporter]} ({reporter}) {label}: {len(payload['data'])} rows, "
                  f"{len(got)} months", flush=True)
        except RuntimeError as exc:
            failures.append(str(exc))
            print(f"{i}/{len(todo)} FAILED {reporter} {label}: {exc}", flush=True)
            if "key rejected" in str(exc):
                break
        time.sleep(dc.PAUSE_S)
    print(f"finished: {len(todo) - len(failures)} fetched, {len(failures)} failed", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
