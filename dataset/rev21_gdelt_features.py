#!/usr/bin/env python3
"""Phase B: monthly GDELT features for the expanded country set, from the cached daily files.

Reproduces, without materialising the event tables (~100 GB for 42 countries):
  * ingest_gdelt._normalise_chunk   (row parsing, country explode, relevance and labour flags)
  * ingest_gdelt._stream_feature_arrays + _monthly_features
        news_vol_7d      relevant events whose availability day falls in the trailing 7 days
        neg_tone_frac_3d share of tone <= -2 among relevant events in the trailing 3 days
    read at month end;
  * repair_processed_sources.rebuild_gdelt_monthly_strike
        strike_flag_7d = 1 if any relevant event with base code 143/144 becomes
        available during the calendar month.
The daily zips are processed in parallel; per (country, availability day)
totals are additive, so summing worker outputs is exact.

Check: for the 18 published countries the output must equal
processed/gdelt_monthly_features.csv. Writes processed_v3/gdelt_monthly_features.csv.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from datetime import date
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ingest_gdelt as ig  # noqa: E402
from common import load_config  # noqa: E402

BASE = Path(__file__).resolve().parent
CACHE = BASE / "data/four_year_2021_2024/cache/gdelt"
import os

# Phase B: 2021-2024 -> processed_v3. Phase D: REV21_GDELT_YEARS=2019-2025 REV21_GDELT_OUT=processed_v4
_Y0, _Y1 = (int(y) for y in os.environ.get("REV21_GDELT_YEARS", "2021-2024").split("-"))
OUT = BASE / "data/four_year_2021_2024" / os.environ.get("REV21_GDELT_OUT", "processed_v3")
START, END = date(_Y0, 1, 1), date(_Y1, 12, 31)
EXISTING = ["IND", "CHN", "KOR", "HKG", "SGP", "VNM", "MYS", "JPN", "THA", "IRL", "USA", "MEX", "PHL",
            "DEU", "GBR", "FRA", "ITA", "CZE"]
NEW = ["ISR", "NLD", "FIN", "IDN", "KHM", "BEL", "HUN", "ESP", "CAN", "TUR", "CHE", "POL", "SWE", "ROU",
       "SVK", "AUS", "EST", "LTU", "BRA", "DNK", "LVA", "MMR", "MLT", "NOR"]
ALIASES = {"ROM": "ROU"}  # CAMEO uses ROM for Romania
SELECTED = set(EXISTING) | set(NEW) | set(ALIASES)
CONFIG = None


def _init():
    global CONFIG
    CONFIG = load_config("config.yaml")


def one_day(path: Path):
    stamp = path.name[:8]
    day = date(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:]))
    counts = defaultdict(lambda: [0, 0])   # (country, avail_day) -> [count, negative]
    strikes = set()                        # (month, country)
    for raw in ig._read_zip_chunks(path):
        chunk = ig._normalise_chunk(raw, SELECTED, day, CONFIG)
        if chunk.empty:
            continue
        chunk["country"] = chunk["country"].replace(ALIASES)
        rel = chunk[chunk["is_supply_chain_relevant"].astype(int).eq(1)]
        if rel.empty:
            continue
        avail = pd.to_datetime(rel["available_at"], utc=True, errors="coerce")
        neg = (pd.to_numeric(rel["avg_tone"], errors="coerce") <= -2.0).astype(int)
        g = pd.DataFrame({"country": rel["country"], "day": avail.dt.date, "neg": neg}).dropna(subset=["day"])
        for (c, d), grp in g.groupby(["country", "day"]):
            counts[(c, d)][0] += len(grp)
            counts[(c, d)][1] += int(grp["neg"].sum())
        lab = rel[rel["is_labor_unrest"].astype(int).eq(1)]
        if not lab.empty:
            months = pd.to_datetime(lab["available_at"], utc=True, errors="coerce").dt.strftime("%Y-%m")
            strikes |= set(zip(months, lab["country"]))
    return dict(counts), strikes


def main() -> int:
    files = []
    for d in pd.date_range(START, END, freq="D"):
        p = CACHE / f"{d.year}" / f"{d:%Y%m%d}.export.CSV.zip"
        if p.exists() and ig._valid_zip(p):
            files.append(p)
    print(f"{len(files)} daily files", flush=True)
    totals = defaultdict(lambda: [0, 0])
    strikes = set()
    with Pool(int(os.environ.get("REV21_GDELT_WORKERS", "8")), initializer=_init) as pool:
        for i, (c, s) in enumerate(pool.imap_unordered(one_day, files, chunksize=4), 1):
            for k, v in c.items():
                totals[k][0] += v[0]
                totals[k][1] += v[1]
            strikes |= s
            if i % 100 == 0:
                print(f"  {i}/{len(files)} files", flush=True)

    countries = sorted(set(EXISTING) | set(NEW))
    dates = pd.date_range(START, END, freq="D")
    idx = {d.date(): i for i, d in enumerate(dates)}
    cpos = {c: i for i, c in enumerate(countries)}
    vol = np.zeros((len(dates), len(countries)), dtype=np.int64)
    vol3 = np.zeros_like(vol)
    negs = np.zeros_like(vol)
    for (c, d), (n, ng) in totals.items():
        if c not in cpos or d not in idx:
            continue
        i, j = idx[d], cpos[c]
        vol[i:min(i + 6, len(dates) - 1) + 1, j] += n
        vol3[i:min(i + 2, len(dates) - 1) + 1, j] += n
        negs[i:min(i + 2, len(dates) - 1) + 1, j] += ng
    rows = []
    for i, d in enumerate(dates):
        if d != d + pd.offsets.MonthEnd(0):
            continue
        month = d.strftime("%Y-%m")
        for c, j in cpos.items():
            rows.append({"month": month, "date": d.strftime("%Y-%m-%d"), "country": c,
                         "news_vol_7d": int(vol[i, j]),
                         "neg_tone_frac_3d": negs[i, j] / vol3[i, j] if vol3[i, j] else 0.0,
                         "strike_flag_7d": int((month, c) in strikes)})
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "gdelt_monthly_features.csv", index=False)

    pub = pd.read_csv(BASE / "data/four_year_2021_2024/processed/gdelt_monthly_features.csv")
    m = pub.merge(out, on=["month", "country"], suffixes=("_pub", "_new"))
    for f in ("news_vol_7d", "neg_tone_frac_3d", "strike_flag_7d"):
        ok = np.isclose(m[f"{f}_pub"], m[f"{f}_new"], atol=1e-9)
        print(f"check {f}: {ok.mean():.4f} of {len(m)} published country-months reproduced")
    print(out[out["country"].isin(NEW)].groupby("country")["news_vol_7d"].mean().round(1).sort_values().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
