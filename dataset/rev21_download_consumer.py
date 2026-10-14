#!/usr/bin/env python3
"""Phase A: download the consumer-goods basket for the existing panel.

Same reporters (the 18 panel countries), same partner list (the 20-country
acquisition set), same flow (imports, M), same months (2021-01..2024-12) as
the semiconductor panel; only the HS codes change, to the eleven-code consumer
basket in config.yaml (8517, 8471, 8528, 8516, 8507, 9403, 6109, 6203, 6204,
9503, 3304).

One request per reporter per year (12 periods in one call) when the API
accepts it, otherwise one per reporter-month. Every response is cached under
data/four_year_2021_2024/cache/comtrade/consumer/ and reused, so the script can
be stopped and rerun. It never touches processed/ or the semiconductor cache.

Usage:
  python rev21_download_consumer.py --probe     # one request, prints what came back
  python rev21_download_consumer.py             # full download (resumable)
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import dotenv_values

BASE = Path(__file__).resolve().parent
CACHE = BASE / "data/four_year_2021_2024/cache/comtrade/consumer"
ENDPOINT = "https://comtradeapi.un.org/data/v1/get/C/M/HS"
HS = ["8517", "8471", "8528", "8516", "8507", "9403", "6109", "6203", "6204", "9503", "3304"]
REPORTERS = ["699", "156", "410", "344", "702", "704", "458", "392", "764", "372",
             "842", "484", "608", "276", "826", "251", "380", "203"]
PARTNERS = ["699", "156", "704", "842", "458", "702", "490", "764", "484", "410", "50",
            "392", "276", "344", "203", "251", "372", "380", "826", "608"]
YEARS = [2021, 2022, 2023, 2024]
PAUSE_S = 1.5
# This network blocks the Comtrade host directly; route through the local Psiphon proxy.
PROXY = __import__("os").environ.get("COMTRADE_PROXY", "http://127.0.0.1:8081")
PROXIES = {"http": PROXY, "https": PROXY} if PROXY else None


def keys() -> list[str]:
    env = dotenv_values(BASE / ".env")
    return [v for v in (env.get("COMTRADE_SUBSCRIPTION_KEY"), env.get("COMTRADE_SECONDARY_KEY")) if v]


def request(reporter: str, periods: list[str], tries: int = 5) -> dict:
    params = {"reportercode": reporter, "flowCode": "M", "period": ",".join(periods),
              "cmdCode": ",".join(HS), "partnerCode": ",".join(PARTNERS), "maxRecords": 250000,
              "format": "JSON", "breakdownMode": "classic", "includeDesc": "false"}  # descriptions are never used downstream
    last = None
    for key in keys():
        for attempt in range(tries):
            try:
                r = requests.get(ENDPOINT, params={**params, "subscription-key": key}, timeout=(30, 180), proxies=PROXIES)
                if r.status_code == 200:
                    payload = r.json()
                    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
                        return payload
                    last = RuntimeError(f"no data field: {str(payload)[:200]}")
                elif r.status_code in (401, 403):
                    last = RuntimeError(f"HTTP {r.status_code} (key rejected or quota exhausted)")
                    break
                elif r.status_code == 429:
                    last = RuntimeError("HTTP 429 (rate limited)")
                    time.sleep(30 * (attempt + 1))
                    continue
                else:
                    last = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            except (requests.RequestException, ValueError) as exc:
                last = exc
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Comtrade request failed for reporter {reporter} {periods[0]}..: {last}")


def cache_file(reporter: str, label: str) -> Path:
    return CACHE / f"reporter_{int(reporter):03d}_{label}.json"


def save(path: Path, reporter: str, periods: list[str], payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({
        "request": {"reporter_code": reporter, "partner_codes": PARTNERS, "periods": periods,
                    "flow_code": "M", "commodity_codes": HS},
        "response": payload,
        "downloaded_at": datetime.now(timezone.utc).isoformat()}, default=str), encoding="utf-8")
    tmp.replace(path)


def done(path: Path) -> bool:
    try:
        return isinstance(json.loads(path.read_text())["response"]["data"], list)
    except Exception:
        return False


def probe() -> None:
    periods = [f"2021{m:02d}" for m in range(1, 13)]
    payload = request("156", periods, tries=1)
    rows = payload["data"]
    print("rows", len(rows), "count", payload.get("count"), "error", payload.get("error"))
    print("periods returned", sorted({r["period"] for r in rows}))
    print("codes returned", sorted({r["cmdCode"] for r in rows}))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--per-month", action="store_true", help="one request per reporter-month")
    args = ap.parse_args()
    if args.probe:
        probe()
        return 0
    jobs = []
    for reporter in REPORTERS:
        for year in YEARS:
            months = [f"{year}{m:02d}" for m in range(1, 13)]
            if args.per_month:
                jobs += [(reporter, [p], p) for p in months]
            else:
                jobs.append((reporter, months, str(year)))
    todo = [j for j in jobs if not done(cache_file(j[0], j[2]))]
    print(f"{len(jobs)} requests, {len(jobs) - len(todo)} cached, {len(todo)} to fetch", flush=True)
    failures = []
    for i, (reporter, periods, label) in enumerate(todo, 1):
        try:
            payload = request(reporter, periods)
            save(cache_file(reporter, label), reporter, periods, payload)
            got = sorted({r["period"] for r in payload["data"]})
            print(f"{i}/{len(todo)} reporter {reporter} {label}: {len(payload['data'])} rows, "
                  f"{len(got)} months", flush=True)
        except RuntimeError as exc:
            failures.append(str(exc))
            print(f"{i}/{len(todo)} FAILED {exc}", flush=True)
            if "key rejected" in str(exc) or "quota" in str(exc):
                break
        time.sleep(PAUSE_S)
    print(f"finished: {len(todo) - len(failures)} fetched, {len(failures)} failed", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
