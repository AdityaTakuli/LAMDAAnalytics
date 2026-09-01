#!/usr/bin/env python3
"""Repair processed source tables after availability-timestamp fixes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import load_config, nested, read_table, write_table  # noqa: E402
from ingest_comtrade import select_country_universe  # noqa: E402


def _norm_comtrade_code(value: object) -> str:
    text = str(value).strip()
    if text.lower() in {"", "nan", "none"}:
        return ""
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def repair_gscpi(path: Path) -> None:
    frame = read_table(path)
    if frame.empty:
        return
    month_end = pd.to_datetime(frame["month"].astype(str) + "-01", utc=True) + pd.offsets.MonthEnd(0)
    available = month_end + pd.Timedelta(days=7)
    frame["published_at"] = available.map(lambda value: value.isoformat())
    frame["available_at"] = frame["published_at"]
    write_table(frame, path)
    print(f"Repaired GSCPI availability timestamps: {path}")


def repair_weather(path: Path) -> None:
    frame = read_table(path)
    if frame.empty:
        return
    observed = pd.to_datetime(frame["observed_date"], errors="coerce", utc=True)
    lag = frame.get("provider", pd.Series("nasa_power", index=frame.index)).astype(str).eq("nasa_power").map(
        {True: 2, False: 1}
    )
    available = observed + pd.to_timedelta(lag.fillna(2), unit="D")
    frame["published_at"] = available.map(lambda value: value.isoformat() if pd.notna(value) else "")
    frame["available_at"] = frame["published_at"]
    write_table(frame, path)
    print(f"Repaired weather availability timestamps: {path}")


def rebuild_country_universe(config: dict) -> None:
    comtrade_path = Path(__file__).resolve().parent / nested(
        config, "outputs", "comtrade", default="processed/comtrade.csv"
    )
    universe_path = Path(__file__).resolve().parent / nested(
        config, "outputs", "country_universe", default="processed/country_universe.csv"
    )
    frame = read_table(comtrade_path)
    universe = select_country_universe(frame, config)
    write_table(universe, universe_path)
    print(f"Rebuilt country universe ({len(universe)} countries): {universe_path}")


def rebuild_gdelt_monthly_strike(config: dict, chunk_size: int = 250_000) -> None:
    """Recompute monthly strike flags from stored GDELT events."""
    import csv

    base = Path(__file__).resolve().parent
    events_path = base / nested(config, "outputs", "gdelt", default="processed/gdelt_events.csv")
    monthly_path = base / nested(
        config,
        "outputs",
        "gdelt_monthly_features",
        default="processed/gdelt_monthly_features.csv",
    )
    if not events_path.exists():
        print(f"Skipping GDELT strike rebuild; missing {events_path}")
        return

    source = nested(config, "sources", "gdelt", default={})
    labor_base_codes = {
        str(value).strip().zfill(3)
        for value in source.get("labor_unrest_event_base_codes", ["143", "144"])
    }
    relevant_roots = {
        str(value).strip().zfill(2)
        for value in source.get("supply_chain_event_roots", ["14", "16", "17", "18", "19"])
    }

    labor_days: dict[tuple[str, str], int] = {}
    rows = 0
    with events_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        index = {name: position for position, name in enumerate(header)}
        required = ("country", "available_at", "event_base_code", "event_root_code")
        missing = [name for name in required if name not in index]
        if missing:
            raise RuntimeError(f"GDELT events missing columns: {missing}")
        for row in reader:
            rows += 1
            if rows % 1_000_000 == 0:
                print(f"  scanned {rows:,} GDELT rows...", flush=True)
            if len(row) <= max(index.values()):
                continue
            country = row[index["country"]].strip()
            if len(country) != 3 or not country.isalpha():
                continue
            root = row[index["event_root_code"]].strip().zfill(2)
            if root not in relevant_roots:
                continue
            base_code = row[index["event_base_code"]].strip().zfill(3)
            if base_code not in labor_base_codes:
                continue
            available = pd.to_datetime(row[index["available_at"]], errors="coerce", utc=True)
            if pd.isna(available):
                continue
            month = available.strftime("%Y-%m")
            labor_days[(month, country)] = 1

    monthly = read_table(monthly_path)
    if monthly.empty:
        print("GDELT monthly table missing; skip strike merge")
        return
    monthly["strike_flag_7d"] = [
        int(labor_days.get((str(month), str(country)), 0))
        for month, country in zip(monthly["month"], monthly["country"])
    ]
    write_table(monthly, monthly_path)
    rate = float(monthly["strike_flag_7d"].mean())
    print(
        f"Rebuilt GDELT monthly strike flags from {rows:,} event rows "
        f"(positive rate={rate:.3f}): {monthly_path}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--skip-gdelt", action="store_true", help="Skip the chunked GDELT strike rebuild")
    args = parser.parse_args()
    config = load_config(args.config)
    base = Path(__file__).resolve().parent

    repair_gscpi(base / nested(config, "outputs", "gscpi", default="processed/gscpi_monthly.csv"))
    repair_weather(base / nested(config, "outputs", "weather", default="processed/weather_daily.csv"))
    rebuild_country_universe(config)
    if not args.skip_gdelt:
        rebuild_gdelt_monthly_strike(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
