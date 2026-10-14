#!/usr/bin/env python3
"""Phase B: add 24 importing countries to the panel (both baskets, 2021-2024).

New reporters are India's next-largest import partners (ranked over both
baskets, 2021-2024; cache/comtrade/india_partner_ranking/) that report monthly
imports to Comtrade. They are added as *importers only*, with the same 20
partner countries as the existing panel, so the 18 existing series are
unchanged and the target keeps its meaning (imports from the original panel
countries). One request per reporter-year covers all 13 HS codes (2
semiconductor + 11 consumer); records are split by code when the panel is built.

Responses are cached under cache/comtrade/phaseB/ and reused (resumable).
Routes through the local Psiphon proxy (Comtrade is blocked on this network).
"""
from __future__ import annotations

import os
import time

import rev21_download_consumer as dc

NEW_REPORTERS = {
    "376": "Israel", "528": "Netherlands", "246": "Finland", "360": "Indonesia", "116": "Cambodia",
    "56": "Belgium", "348": "Hungary", "724": "Spain", "124": "Canada", "792": "Turkiye",
    "757": "Switzerland", "616": "Poland", "752": "Sweden", "642": "Romania", "703": "Slovakia",
    "36": "Australia", "233": "Estonia", "440": "Lithuania", "76": "Brazil", "208": "Denmark",
    "428": "Latvia", "104": "Myanmar", "470": "Malta", "579": "Norway",
}
HS_ALL = ["8541", "8542", *dc.HS]


def main() -> int:
    dc.CACHE = dc.BASE / "data/four_year_2021_2024/cache/comtrade/phaseB"
    dc.HS = HS_ALL
    jobs = [(r, [f"{y}{m:02d}" for m in range(1, 13)], str(y)) for r in NEW_REPORTERS for y in dc.YEARS]
    todo = [j for j in jobs if not dc.done(dc.cache_file(j[0], j[2]))]
    if os.environ.get("REV21_REVERSE") == "1":  # second worker takes the list from the other end
        todo = todo[::-1]
    if os.environ.get("REV21_MIDDLE") == "1":  # third worker starts in the middle
        todo = todo[len(todo) // 2:] + todo[:len(todo) // 2]
    print(f"{len(jobs)} requests, {len(jobs) - len(todo)} cached, {len(todo)} to fetch", flush=True)
    failures = []
    for i, (reporter, periods, label) in enumerate(todo, 1):
        if dc.done(dc.cache_file(reporter, label)):  # finished by the other worker
            continue
        try:
            payload = dc.request(reporter, periods)
            dc.save(dc.cache_file(reporter, label), reporter, periods, payload)
            got = sorted({r["period"] for r in payload["data"]})
            print(f"{i}/{len(todo)} {NEW_REPORTERS[reporter]} ({reporter}) {label}: "
                  f"{len(payload['data'])} rows, {len(got)} months", flush=True)
        except RuntimeError as exc:
            failures.append(str(exc))
            print(f"{i}/{len(todo)} FAILED {NEW_REPORTERS[reporter]} {label}: {exc}", flush=True)
            if "key rejected" in str(exc):
                break
        time.sleep(dc.PAUSE_S)
    print(f"finished: {len(todo) - len(failures)} fetched, {len(failures)} failed", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
