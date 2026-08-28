"""Ingest daily NASA POWER (or optional NOAA CDO) temperature observations.

Locations are configured in ``config.yaml`` and may represent country
centroids or known facility coordinates.  Raw JSON responses are cached per
location/provider.  The fusion stage applies the paper's seven-day anomaly
rule; this stage does not look ahead or normalize values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from common import (
    add_common_cli,
    ensure_parent,
    env_value,
    load_config,
    nested,
    request_headers,
    read_table,
    stable_id,
    vintage_now,
    write_table,
)


def _nasa(location: dict[str, Any], start: pd.Timestamp, end: pd.Timestamp, raw_path: Path, force: bool) -> dict:
    if force or not raw_path.exists():
        url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        params = {
            "parameters": "T2M",
            "community": "AG",
            "longitude": location["longitude"],
            "latitude": location["latitude"],
            "start": start.strftime("%Y%m%d"),
            "end": end.strftime("%Y%m%d"),
            "format": "JSON",
        }
        response = requests.get(url, params=params, headers=request_headers(), timeout=120)
        response.raise_for_status()
        ensure_parent(raw_path)
        raw_path.write_text(response.text, encoding="utf-8")
    return json.loads(raw_path.read_text(encoding="utf-8"))


def _noaa(
    location: dict[str, Any], start: pd.Timestamp, end: pd.Timestamp, raw_path: Path, config: dict, force: bool
) -> dict:
    token = env_value(config, nested(config, "sources", "weather", "noaa_token_env"), "")
    station = location.get("noaa_station_id")
    if not token or not station:
        raise RuntimeError("NOAA mode requires noaa_token_env and noaa_station_id in config")
    if force or not raw_path.exists():
        response = requests.get(
            "https://www.ncei.noaa.gov/cdo-web/api/v2/data",
            headers={**request_headers(), "token": token},
            params={
                "datasetid": "GHCND",
                "stationid": station,
                "startdate": start.strftime("%Y-%m-%d"),
                "enddate": end.strftime("%Y-%m-%d"),
                "datatypeid": "TAVG",
                "units": "metric",
                "limit": 1000,
            },
            timeout=120,
        )
        response.raise_for_status()
        ensure_parent(raw_path)
        raw_path.write_text(response.text, encoding="utf-8")
    return json.loads(raw_path.read_text(encoding="utf-8"))


def _rows(
    location: dict[str, Any],
    payload: dict,
    provider: str,
    available_at: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if provider == "nasa_power":
        values = nested(payload, "properties", "parameter", "T2M", default={})
        for day, temperature in values.items():
            try:
                numeric = float(temperature)
            except (TypeError, ValueError):
                continue
            if numeric <= -900:
                continue
            rows.append(
                {
                    "node_id": location.get("node_id") or stable_id("country", location["name"]),
                    "location_name": location["name"],
                    "latitude": location["latitude"],
                    "longitude": location["longitude"],
                    "observed_date": pd.to_datetime(day, format="%Y%m%d", utc=True).isoformat(),
                    "temperature_c": numeric,
                    "provider": provider,
                    "published_at": available_at,
                    "available_at": available_at,
                    "vintage_date": available_at,
                    "source": "nasa_power",
                }
            )
    else:
        for item in payload.get("results", []):
            value = pd.to_numeric(item.get("value"), errors="coerce")
            if pd.isna(value):
                continue
            # NOAA TAVG is tenths of a degree Celsius for GHCND.
            numeric = float(value) / 10.0 if abs(float(value)) > 100 else float(value)
            rows.append(
                {
                    "node_id": location.get("node_id") or stable_id("country", location["name"]),
                    "location_name": location["name"],
                    "latitude": location["latitude"],
                    "longitude": location["longitude"],
                    "observed_date": pd.Timestamp(item["date"], tz="UTC").isoformat(),
                    "temperature_c": numeric,
                    "provider": provider,
                    "published_at": available_at,
                    "available_at": available_at,
                    "vintage_date": available_at,
                    "source": "noaa_cdo",
                }
            )
    return rows


def ingest(config: dict, force: bool = False) -> Path:
    destination = Path(__file__).resolve().parent / nested(
        config, "outputs", "weather", default="processed/weather_daily.csv"
    )
    if destination.exists() and not force:
        return destination
    source = nested(config, "sources", "weather", default={})
    provider = source.get("provider", "nasa_power")
    start = pd.Timestamp(nested(config, "analysis", "start_date"), tz="UTC")
    configured_end = pd.Timestamp(nested(config, "analysis", "end_date"), tz="UTC")
    end = min(configured_end, pd.Timestamp.now(tz="UTC").normalize())
    cache_dir = Path(__file__).resolve().parent / source.get("cache_dir", "cache/weather")
    available_at = vintage_now()
    all_rows: list[dict[str, Any]] = []
    locations = source.get("locations", [])
    universe_path = Path(__file__).resolve().parent / nested(
        config, "outputs", "country_universe", default="processed/country_universe.csv"
    )
    if universe_path.exists() or universe_path.with_suffix(".csv").exists():
        universe = read_table(universe_path)
        selected_codes = set(universe.get("comtrade_code", pd.Series(dtype=str)).dropna().astype(str))
        locations = [
            location
            for location in locations
            if str(location.get("comtrade_code", "")) in selected_codes
        ]
        print(f"Weather universe restricted to {len(locations)} selected countries")
    for location in locations:
        location_key = stable_id("location", location["name"])
        raw_path = cache_dir / provider / f"{location_key}_{start:%Y%m%d}_{end:%Y%m%d}.json"
        if provider == "noaa":
            payload = _noaa(location, start, end, raw_path, config, force)
        else:
            payload = _nasa(location, start, end, raw_path, force)
        all_rows.extend(_rows(location, payload, provider, available_at))
    frame = pd.DataFrame(all_rows)
    return write_table(frame, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_cli(parser)
    args = parser.parse_args()
    output = ingest(load_config(args.config), force=args.force)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

