"""Download and process the official GDELT Event Database.

The raw distribution is a daily tab-delimited, ZIP-compressed export at
``https://data.gdeltproject.org/events/YYYYMMDD.export.CSV.zip``.  This
adapter keeps the event-level records needed by the research pipeline,
processes one day at a time, and writes daily and monthly feature tables.

GDELT's ``SQLDATE`` is used as the event/reference date and ``DATEADDED`` is
used as the availability timestamp.  Consequently, a feature for day ``d``
only includes events with both timestamps no later than the end of ``d``.
The labor-unrest indicator is a transparent CAMEO ``14`` (Protest) proxy; it
does not claim to identify every real-world strike.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import re
import time
import zipfile
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests
import numpy as np

from common import (
    add_common_cli,
    ensure_parent,
    load_config,
    nested,
    read_table,
    request_headers,
    vintage_now,
    write_json,
    write_table,
)


LOGGER = logging.getLogger("gdelt")
BASE_URL = "https://data.gdeltproject.org/events"
CHUNK_SIZE = 100_000

# GDELT Event Database fields.  The positional definitions are from the
# official V2 codebook; EventCode=26, EventBaseCode=27, EventRootCode=28,
# AvgTone=34, DATEADDED=59, and SOURCEURL=60 (zero-based positions).
GDELT_COLUMNS = [
    "GLOBALEVENTID",
    "SQLDATE",
    "MonthYear",
    "Year",
    "FractionDate",
    "Actor1Code",
    "Actor1Name",
    "Actor1CountryCode",
    "Actor1KnownGroupCode",
    "Actor1EthnicCode",
    "Actor1Religion1Code",
    "Actor1Religion2Code",
    "Actor1Type1Code",
    "Actor1Type2Code",
    "Actor1Type3Code",
    "Actor2Code",
    "Actor2Name",
    "Actor2CountryCode",
    "Actor2KnownGroupCode",
    "Actor2EthnicCode",
    "Actor2Religion1Code",
    "Actor2Religion2Code",
    "Actor2Type1Code",
    "Actor2Type2Code",
    "Actor2Type3Code",
    "IsRootEvent",
    "EventCode",
    "EventBaseCode",
    "EventRootCode",
    "QuadClass",
    "GoldsteinScale",
    "NumMentions",
    "NumSources",
    "NumArticles",
    "AvgTone",
    "Actor1Geo_Type",
    "Actor1Geo_FullName",
    "Actor1Geo_CountryCode",
    "Actor1Geo_ADM1Code",
    "Actor1Geo_ADM2Code",
    "Actor1Geo_Lat",
    "Actor1Geo_Long",
    "Actor1Geo_FeatureID",
    "Actor2Geo_Type",
    "Actor2Geo_FullName",
    "Actor2Geo_CountryCode",
    "Actor2Geo_ADM1Code",
    "Actor2Geo_ADM2Code",
    "Actor2Geo_Lat",
    "Actor2Geo_Long",
    "Actor2Geo_FeatureID",
    "ActionGeo_Type",
    "ActionGeo_FullName",
    "ActionGeo_CountryCode",
    "ActionGeo_ADM1Code",
    "ActionGeo_ADM2Code",
    "ActionGeo_Lat",
    "ActionGeo_Long",
    "ActionGeo_FeatureID",
    "DATEADDED",
    "SOURCEURL",
]

# The currently available historical daily ``events/`` export is a
# 58-column-compatible official layout. GDELT's 61-column Event Database
# layout is also accepted when a mirror or future archive provides it. The
# event, actor, and tone positions are stable; the historical layout omits
# three later geography fields, moving DATEADDED/SOURCEURL to 56/57.
RAW_POSITIONS_61 = {
    "GLOBALEVENTID": 0,
    "SQLDATE": 1,
    "Actor1Name": 6,
    "Actor1CountryCode": 7,
    "Actor2Name": 16,
    "Actor2CountryCode": 17,
    "EventCode": 26,
    "EventBaseCode": 27,
    "EventRootCode": 28,
    "AvgTone": 34,
    "DATEADDED": 59,
    "SOURCEURL": 60,
}
RAW_POSITIONS_58 = {
    "GLOBALEVENTID": 0,
    "SQLDATE": 1,
    "Actor1Name": 6,
    "Actor1CountryCode": 7,
    "Actor2Name": 16,
    "Actor2CountryCode": 17,
    "EventCode": 26,
    "EventBaseCode": 27,
    "EventRootCode": 28,
    "AvgTone": 34,
    "DATEADDED": 56,
    "SOURCEURL": 57,
}

EVENT_COLUMNS = [
    "event_id",
    "event_date",
    "reference_date",
    "published_at",
    "available_at",
    "country",
    "actor1_country",
    "actor2_country",
    "event_code",
    "event_base_code",
    "event_root_code",
    "avg_tone",
    "source_url",
    "is_labor_unrest",
    "is_supply_chain_relevant",
    "source_file_date",
    "source",
    "vintage_date",
]

FEATURE_COLUMNS = [
    "date",
    "country",
    "news_vol_7d",
    "neg_tone_frac_3d",
    "strike_flag_7d",
]

REQUIRED_EVENT_COLUMNS = set(EVENT_COLUMNS)
PART_DTYPES = {
    "event_id": "string",
    "country": "string",
    "actor1_country": "string",
    "actor2_country": "string",
    "event_code": "string",
    "event_base_code": "string",
    "event_root_code": "string",
}


def _config_dates(config: dict[str, Any], test_mode: bool = False) -> tuple[date, date]:
    source = nested(config, "sources", "gdelt", default={})
    start_value = source.get("start_date") or nested(config, "analysis", "start_date")
    end_value = source.get("end_date") or nested(config, "analysis", "end_date")
    if test_mode:
        start_value = source.get("test_start_date", "2020-01-01")
        end_value = source.get("test_end_date", "2020-01-03")
    start = pd.Timestamp(start_value).date()
    end = pd.Timestamp(end_value).date()
    if end < start:
        raise ValueError(f"GDELT end date {end} precedes start date {start}")
    return start, end


def _selected_iso3(config: dict[str, Any]) -> list[str]:
    """Use Comtrade's selected universe when available, otherwise config."""
    configured_output = nested(
        config, "outputs", "country_universe", default="processed/country_universe.csv"
    )
    universe_path = Path(__file__).resolve().parent / configured_output
    if universe_path.exists():
        universe = read_table(universe_path)
        if "iso3" in universe:
            values = universe["iso3"].dropna().astype(str).str.upper()
            selected = sorted({value for value in values if re.fullmatch(r"[A-Z]{3}", value)})
            if selected:
                return selected

    source = nested(config, "sources", "gdelt", default={})
    configured = source.get("countries", [])
    if configured:
        return sorted({str(value).strip().upper() for value in configured if value})
    return sorted(
        {
            str(location.get("iso3", "")).strip().upper()
            for location in nested(config, "sources", "weather", "locations", default=[])
            if location.get("iso3")
        }
    )


def _paths(config: dict[str, Any], test_mode: bool = False) -> dict[str, Path]:
    root = Path(__file__).resolve().parent
    source = nested(config, "sources", "gdelt", default={})
    cache = root / source.get("cache_dir", "cache/gdelt")
    output = root / nested(config, "outputs", "gdelt", default="processed/gdelt_events.csv")
    if test_mode:
        cache = cache / "test"
        output = output.parent / "test" / output.name
    processed = output.parent
    return {
        "cache": cache,
        "parts": processed / "gdelt_event_parts",
        "events": output,
        "features": root
        / nested(config, "outputs", "gdelt_features", default="processed/gdelt_features.csv"),
        "monthly": root
        / nested(
            config,
            "outputs",
            "gdelt_monthly_features",
            default="processed/gdelt_monthly_features.csv",
        ),
        "metadata": root
        / nested(
            config, "outputs", "gdelt_metadata", default="processed/gdelt_metadata.json"
        ),
        "validation": root
        / nested(
            config,
            "outputs",
            "gdelt_validation",
            default="processed/gdelt_validation.json",
        ),
        "failures": cache / "failed_downloads.json",
    }


def _redirect_test_outputs(paths: dict[str, Path], test_mode: bool) -> dict[str, Path]:
    if not test_mode:
        return paths
    result = dict(paths)
    for key in ("features", "monthly", "metadata", "validation"):
        result[key] = result[key].parent / "test" / result[key].name
    return result


def _source_url(config: dict[str, Any], stamp: str) -> str:
    source = nested(config, "sources", "gdelt", default={})
    template = source.get("url_template", f"{BASE_URL}/{{date}}.export.CSV.zip")
    return str(template).format(date=stamp)


def _valid_zip(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                return False
            return any(name.lower().endswith((".csv", ".tsv", ".txt")) for name in archive.namelist())
    except (OSError, zipfile.BadZipFile):
        return False


def _download_file(
    session: requests.Session,
    url: str,
    destination: Path,
    config: dict[str, Any],
    force: bool = False,
) -> tuple[bool, int, str | None]:
    """Download atomically with retries and post-download ZIP validation."""
    source = nested(config, "sources", "gdelt", default={})
    attempts = max(1, int(source.get("max_retries", 3)))
    timeout = int(source.get("timeout_seconds", 180))
    backoff = float(source.get("backoff_seconds", 2.0))
    delay = float(source.get("request_delay_seconds", 0.2))
    ensure_parent(destination)

    for temporary_suffix in (".tmp", ".part"):
        stale = destination.with_name(destination.name + temporary_suffix)
        if stale.exists():
            stale.unlink()
    if not force and destination.exists() and _valid_zip(destination):
        return True, 0, None
    if destination.exists() and not _valid_zip(destination):
        destination.unlink()

    last_error = "unknown download error"
    for attempt in range(1, attempts + 1):
        temporary = destination.with_name(destination.name + ".tmp")
        try:
            if temporary.exists():
                temporary.unlink()
            if delay:
                time.sleep(delay)
            with session.get(
                url,
                headers=request_headers(),
                timeout=(min(30, timeout), timeout),
                stream=True,
            ) as response:
                if response.status_code == 404:
                    return False, attempt, "HTTP 404 (daily export unavailable)"
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    wait = float(retry_after) if retry_after and retry_after.isdigit() else backoff * (2 ** (attempt - 1))
                    time.sleep(min(wait, 300))
                    raise requests.HTTPError(f"HTTP 429 rate limited for {url}")
                response.raise_for_status()
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if not _valid_zip(temporary):
                raise zipfile.BadZipFile("downloaded file is not a valid GDELT ZIP")
            temporary.replace(destination)
            return True, attempt, None
        except (OSError, requests.RequestException, zipfile.BadZipFile) as exc:
            last_error = str(exc)
            if temporary.exists():
                temporary.unlink()
            if attempt < attempts:
                time.sleep(min(backoff * (2 ** (attempt - 1)), 300))
        finally:
            if temporary.exists():
                temporary.unlink()
    return False, attempts, last_error


def _download_task(
    current: date,
    url: str,
    destination: Path,
    config: dict[str, Any],
    force: bool = False,
) -> tuple[date, bool, int, str | None]:
    """Run one isolated download so requests sessions are not shared by threads."""
    with requests.Session() as session:
        ok, attempts, error = _download_file(
            session, url, destination, config, force=force
        )
    return current, ok, attempts, error


def _read_zip_chunks(path: Path) -> Iterable[pd.DataFrame]:
    with zipfile.ZipFile(path) as archive:
        names = [
            name
            for name in archive.namelist()
            if name.lower().endswith((".csv", ".tsv", ".txt"))
        ]
        if not names:
            raise ValueError(f"No tabular member in {path}")
        with archive.open(names[0]) as handle:
            first_line = handle.readline()
            column_count = len(first_line.rstrip(b"\r\n").split(b"\t"))
            handle.seek(0)
            if column_count >= 61:
                positions = RAW_POSITIONS_61
            elif column_count >= 58:
                positions = RAW_POSITIONS_58
            else:
                raise ValueError(f"Unexpected GDELT column count: {column_count}")
            reader = pd.read_csv(
                handle,
                sep="\t",
                header=None,
                usecols=sorted(set(positions.values())),
                dtype=str,
                chunksize=CHUNK_SIZE,
                on_bad_lines="skip",
                low_memory=False,
            )
            rename = {index: name for name, index in positions.items()}
            for chunk in reader:
                yield chunk.rename(columns=rename)


def _normalise_chunk(
    raw: pd.DataFrame,
    selected: set[str],
    source_file_date: date,
    config: dict[str, Any],
) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    result = pd.DataFrame(index=raw.index)
    result["event_id"] = raw["GLOBALEVENTID"].fillna("").astype(str)
    result["event_date"] = pd.to_datetime(raw["SQLDATE"], format="%Y%m%d", errors="coerce", utc=True)
    result["reference_date"] = result["event_date"]
    date_added = raw["DATEADDED"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
    result["published_at"] = pd.to_datetime(date_added, format="%Y%m%d%H%M%S", errors="coerce", utc=True)
    result["published_at"] = result["published_at"].fillna(
        pd.to_datetime(date_added, format="%Y%m%d", errors="coerce", utc=True)
    )
    result["available_at"] = result["published_at"]
    actor1 = raw["Actor1CountryCode"].fillna("").astype(str).str.strip().str.upper()
    actor2 = raw["Actor2CountryCode"].fillna("").astype(str).str.strip().str.upper()
    result["actor1_country"] = actor1
    result["actor2_country"] = actor2
    result["event_code"] = raw["EventCode"].fillna("").astype(str).str.strip().str.zfill(3)
    base_code = raw["EventBaseCode"] if "EventBaseCode" in raw else raw["EventCode"]
    result["event_base_code"] = base_code.fillna("").astype(str).str.strip().str.zfill(3)
    result["event_root_code"] = raw["EventRootCode"].fillna("").astype(str).str.strip().str.zfill(2)
    result["avg_tone"] = pd.to_numeric(raw["AvgTone"], errors="coerce")
    result["source_url"] = raw["SOURCEURL"].fillna("").astype(str)
    valid_dates = result["event_date"].notna() & result["available_at"].notna()
    result = result[valid_dates].copy()
    if result.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    # The raw export contains one row per event.  Expand it to one row per
    # selected country involved in the event, without broadcasting unknown
    # country events to every node.
    result["country"] = [
        sorted({code for code in (a, b) if code in selected})
        for a, b in zip(result["actor1_country"], result["actor2_country"])
    ]
    result = result.explode("country", ignore_index=True).dropna(subset=["country"])
    if result.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    source = nested(config, "sources", "gdelt", default={})
    labor_roots = {
        str(value).strip().zfill(2)
        for value in source.get("labor_unrest_event_roots", ["14"])
    }
    relevant_roots = {
        str(value).strip().zfill(2)
        for value in source.get(
            "supply_chain_event_roots", ["14", "16", "17", "18", "19"]
        )
    }
    result["is_labor_unrest"] = result["event_root_code"].isin(labor_roots).astype(int)
    result["is_supply_chain_relevant"] = result["event_root_code"].isin(relevant_roots).astype(int)
    result["source_file_date"] = source_file_date.isoformat()
    result["source"] = "gdelt_event_database"
    result["vintage_date"] = vintage_now()
    # Preserve the raw value for validation.  GDELT AvgTone is normally in
    # [-100, 100], and silently clipping an impossible value would hide a
    # malformed input file.
    result["avg_tone"] = result["avg_tone"].fillna(0.0)
    return result[EVENT_COLUMNS].reset_index(drop=True)


def _valid_part(path: Path) -> bool:
    try:
        sample = pd.read_csv(path, nrows=1)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return False
    return REQUIRED_EVENT_COLUMNS.issubset(sample.columns)


def _process_day(
    zip_path: Path,
    part_path: Path,
    source_file_date: date,
    selected: set[str],
    config: dict[str, Any],
) -> int:
    ensure_parent(part_path)
    temporary = part_path.with_name(part_path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    rows = 0
    wrote_header = False
    try:
        for raw in _read_zip_chunks(zip_path):
            chunk = _normalise_chunk(raw, selected, source_file_date, config)
            if chunk.empty:
                continue
            chunk.to_csv(temporary, mode="a", header=not wrote_header, index=False)
            wrote_header = True
            rows += len(chunk)
        if not wrote_header:
            pd.DataFrame(columns=EVENT_COLUMNS).to_csv(temporary, index=False)
        temporary.replace(part_path)
        return rows
    except (OSError, ValueError, zipfile.BadZipFile, pd.errors.ParserError):
        if temporary.exists():
            temporary.unlink()
        raise


def _write_events_incrementally(parts: list[Path], destination: Path) -> int:
    ensure_parent(destination)
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    rows = 0
    wrote_header = False
    for part in parts:
        for chunk in pd.read_csv(part, chunksize=CHUNK_SIZE, dtype=PART_DTYPES):
            chunk.to_csv(temporary, mode="a", header=not wrote_header, index=False)
            wrote_header = True
            rows += len(chunk)
    if not wrote_header:
        pd.DataFrame(columns=EVENT_COLUMNS).to_csv(temporary, index=False)
    temporary.replace(destination)
    return rows


def _stream_feature_arrays(
    parts: list[Path],
    start: date,
    end: date,
    countries: list[str],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aggregate partitions into bounded daily arrays, never all events."""
    dates = pd.date_range(start, end, freq="D", tz="UTC")
    date_index = {item.date(): index for index, item in enumerate(dates)}
    country_index = {country: index for index, country in enumerate(countries)}
    volume = np.zeros((len(dates), len(countries)), dtype=np.int64)
    volume_3 = np.zeros((len(dates), len(countries)), dtype=np.int64)
    negative = np.zeros((len(dates), len(countries)), dtype=np.int64)
    labor = np.zeros((len(dates), len(countries)), dtype=np.int64)
    source = nested(config, "sources", "gdelt", default={})
    threshold = float(source.get("negative_tone_threshold", -2.0))
    total_events = relevant_events = labor_events = 0
    first_event: pd.Timestamp | None = None
    last_event: pd.Timestamp | None = None
    invalid_countries = invalid_tones = duplicate_rows = 0
    for part in parts:
        for chunk in pd.read_csv(part, chunksize=CHUNK_SIZE, dtype=PART_DTYPES):
            total_events += len(chunk)
            relevant = chunk[chunk["is_supply_chain_relevant"].astype(int).eq(1)].copy()
            relevant_events += len(relevant)
            labor_events += int(relevant["is_labor_unrest"].astype(int).sum())
            invalid_countries += int(
                (~chunk["country"].astype(str).str.fullmatch(r"[A-Z]{3}", na=False)).sum()
            )
            tones = pd.to_numeric(chunk["avg_tone"], errors="coerce")
            invalid_tones += int((tones.notna() & ~tones.between(-100, 100)).sum())
            duplicate_rows += int(chunk.duplicated(subset=["event_id", "country"], keep=False).sum())
            if not relevant.empty:
                # DATEADDED is the first timestamp at which the event is
                # available in the news/event stream.  Rolling news features
                # use this observation clock; SQLDATE remains available as
                # the event/reference date in the event table.
                relevant["event_day"] = pd.to_datetime(
                    relevant["available_at"], utc=True, errors="coerce"
                ).dt.date
                relevant["available_day"] = pd.to_datetime(
                    relevant["available_at"], utc=True, errors="coerce"
                ).dt.date
                relevant["tone_negative"] = (
                    pd.to_numeric(relevant["avg_tone"], errors="coerce") <= threshold
                ).astype(int)
                grouped = (
                    relevant.dropna(subset=["event_day", "available_day"])
                    .groupby(["country", "event_day", "available_day"], as_index=False)
                    .agg(
                        count=("event_id", "size"),
                        negative=("tone_negative", "sum"),
                        labor=("is_labor_unrest", "max"),
                    )
                )
                for row in grouped.itertuples(index=False):
                    country = str(row.country)
                    event_day = date_index.get(row.event_day)
                    available_day = date_index.get(row.available_day)
                    if country not in country_index or event_day is None:
                        continue
                    if available_day is None or row.available_day > dates[-1].date():
                        continue
                    # An event contributes to trailing windows from its
                    # reference date through the end of that window, but only
                    # after GDELT has made it available.
                    first = max(event_day, available_day if available_day is not None else event_day, 0)
                    last = min(event_day + 6, len(dates) - 1)
                    if first > last:
                        continue
                    country_position = country_index[country]
                    volume[first : last + 1, country_position] += int(row.count)
                    labor[first : last + 1, country_position] = np.maximum(
                        labor[first : last + 1, country_position], int(row.labor)
                    )
                    negative_last = min(event_day + 2, len(dates) - 1)
                    negative_first = max(first, event_day)
                    if negative_first <= negative_last:
                        volume_3[negative_first : negative_last + 1, country_position] += int(
                            row.count
                        )
                        negative[negative_first : negative_last + 1, country_position] += int(
                            row.negative
                        )
            parsed_dates = pd.to_datetime(chunk["event_date"], errors="coerce", utc=True).dropna()
            if not parsed_dates.empty:
                chunk_first = parsed_dates.min()
                chunk_last = parsed_dates.max()
                first_event = chunk_first if first_event is None else min(first_event, chunk_first)
                last_event = chunk_last if last_event is None else max(last_event, chunk_last)
    records: list[dict[str, Any]] = []
    for date_position, current in enumerate(dates):
        for country_position, country in enumerate(countries):
            count = int(volume[date_position, country_position])
            negative_count = int(negative[date_position, country_position])
            count_3 = int(volume_3[date_position, country_position])
            records.append(
                {
                    "date": current,
                    "country": country,
                    "news_vol_7d": count,
                    "neg_tone_frac_3d": negative_count / count_3 if count_3 else 0.0,
                    "strike_flag_7d": int(labor[date_position, country_position]),
                }
            )
    result = pd.DataFrame(records, columns=FEATURE_COLUMNS)
    result["date"] = pd.to_datetime(result["date"], utc=True).dt.strftime("%Y-%m-%d")
    return result, {
        "event_count": total_events,
        "relevant_event_count": relevant_events,
        "labor_unrest_event_count": labor_events,
        "invalid_country_codes": invalid_countries,
        "invalid_tone_values": invalid_tones,
        "duplicate_event_country_rows": duplicate_rows,
        "first_event_date": first_event,
        "last_event_date": last_event,
    }


def _monthly_features(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame(columns=["month", "date", "country", *FEATURE_COLUMNS[2:]])
    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    # The graph snapshot is anchored at month end.  Daily rows are complete
    # for every configured country, so the month-end row is deterministic and
    # preserves the meaning of trailing 7/3-day features.
    month_end = frame["date"] + pd.offsets.MonthEnd(0)
    result = frame.loc[frame["date"].eq(month_end)].copy()
    result["month"] = result["date"].dt.strftime("%Y-%m")
    result["date"] = result["date"].dt.strftime("%Y-%m-%d")
    return result[["month", "date", "country", *FEATURE_COLUMNS[2:]]].reset_index(drop=True)


def _read_failures(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return payload if isinstance(payload, list) else []


def _write_failures(path: Path, failures: list[dict[str, Any]]) -> None:
    write_json(failures, path)


def _validation(
    stats: dict[str, Any],
    start: date,
    end: date,
    countries: list[str],
    failed_dates: set[str],
) -> dict[str, Any]:
    expected_dates = set(pd.date_range(start, end, freq="D").strftime("%Y-%m-%d"))
    covered_dates = {
        item for item in (stats.get("available_file_dates") or []) if item in expected_dates
    }
    missing_dates = sorted(expected_dates - covered_dates - failed_dates)
    invalid_countries = int(stats.get("invalid_country_codes", 0))
    invalid_tones = int(stats.get("invalid_tone_values", 0))
    duplicate_count = int(stats.get("duplicate_event_country_rows", 0))
    status = "PASS" if not invalid_tones and not invalid_countries and not missing_dates else "FAIL"
    return {
        "coverage": {"start_date": start.isoformat(), "end_date": end.isoformat()},
        "missing_dates_without_failed_file": missing_dates,
        "failed_dates": sorted(failed_dates),
        "duplicate_event_country_rows": duplicate_count,
        "invalid_country_codes": invalid_countries,
        "invalid_tone_values": invalid_tones,
        "required_columns_missing": sorted(REQUIRED_EVENT_COLUMNS - set(EVENT_COLUMNS)),
        "configured_country_count": len(countries),
        "status": status,
    }


def ingest(
    config: dict[str, Any],
    force: bool = False,
    mode: str | None = None,
    test_mode: bool = False,
    workers: int | None = None,
    limit: int | None = None,
) -> Path:
    """Download, process, validate, and write the GDELT dataset."""
    del mode  # The official raw distribution is the canonical implementation.
    start, end = _config_dates(config, test_mode=test_mode)
    paths = _redirect_test_outputs(_paths(config, test_mode=test_mode), test_mode)
    selected = _selected_iso3(config)
    source = nested(config, "sources", "gdelt", default={})
    workers = int(workers if workers is not None else source.get("download_workers", 8))
    if workers < 1 or workers > 16:
        raise ValueError("--workers must be between 1 and 16")
    if limit is not None and limit < 1:
        raise ValueError("--limit must be at least 1")
    paths["cache"].mkdir(parents=True, exist_ok=True)
    paths["parts"].mkdir(parents=True, exist_ok=True)
    LOGGER.info("[GDELT] Date range: %s → %s", start, end)
    LOGGER.info("[GDELT] Countries configured: %s", len(selected))

    failures = _read_failures(paths["failures"])
    failure_by_date = {
        entry.get("date"): entry
        for entry in failures
        if entry.get("date")
    }
    downloaded = reused = processed_rows = 0
    valid_parts: list[Path] = []

    dates = [item.date() for item in pd.date_range(start, end, freq="D")]
    if limit is not None:
        dates = dates[:limit]
        if dates:
            end = dates[-1]
    discovered = len(dates)
    LOGGER.info("[GDELT] Files discovered: %s", len(dates))
    ready: list[tuple[date, Path, Path]] = []
    pending: list[tuple[date, str, Path, Path]] = []
    for current in dates:
        stamp = current.strftime("%Y%m%d")
        year_dir = paths["cache"] / current.strftime("%Y")
        zip_path = year_dir / f"{stamp}.export.CSV.zip"
        part_path = paths["parts"] / current.strftime("%Y") / f"{stamp}.csv"
        if not force and _valid_part(part_path):
            valid_parts.append(part_path)
            reused += 1
            failure_by_date.pop(current.isoformat(), None)
            failure_by_date.pop(stamp, None)
            continue
        if not force and zip_path.exists() and _valid_zip(zip_path):
            reused += 1
            ready.append((current, zip_path, part_path))
        else:
            pending.append((current, _source_url(config, stamp), zip_path, part_path))

    total_pending = len(pending)
    if total_pending:
        LOGGER.info(
            "[GDELT] Downloading %s missing files with %s workers",
            total_pending,
            workers,
        )
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gdelt-download") as pool:
            futures = {
                pool.submit(_download_task, current, url, zip_path, config, force): (
                    position,
                    current,
                    url,
                    zip_path,
                    part_path,
                )
                for position, (current, url, zip_path, part_path) in enumerate(pending, start=1)
            }
            for future in as_completed(futures):
                position, current, url, zip_path, part_path = futures[future]
                try:
                    _, ok, attempts, error = future.result()
                except Exception as exc:  # one worker must not abort other dates
                    ok, attempts, error = False, 1, f"worker error: {exc}"
                if ok:
                    downloaded += 1
                    ready.append((current, zip_path, part_path))
                    failure_by_date.pop(current.isoformat(), None)
                    LOGGER.info(
                        "[GDELT] Completed %s/%s: %s",
                        position,
                        total_pending,
                        current.isoformat(),
                    )
                else:
                    failure_by_date[current.isoformat()] = {
                        "date": current.isoformat(),
                        "url": url,
                        "error": error or "unavailable",
                        "attempts": attempts,
                    }
                    LOGGER.warning(
                        "[GDELT] Failed %s/%s: %s (%s)",
                        position,
                        total_pending,
                        current.isoformat(),
                        error,
                    )

    for current, zip_path, part_path in sorted(ready, key=lambda item: item[0]):
        stamp = current.strftime("%Y%m%d")
        try:
            processed_rows += _process_day(zip_path, part_path, current, set(selected), config)
            valid_parts.append(part_path)
            failure_by_date.pop(current.isoformat(), None)
            failure_by_date.pop(stamp, None)
        except (OSError, ValueError, zipfile.BadZipFile, pd.errors.ParserError) as exc:
            failure_by_date[current.isoformat()] = {
                "date": current.isoformat(),
                "url": _source_url(config, stamp),
                "error": f"processing failed: {exc}",
                "attempts": 1,
            }
            LOGGER.warning("[GDELT] Processing failed: %s (%s)", stamp, exc)
    _write_failures(paths["failures"], list(failure_by_date.values()))

    events_rows = _write_events_incrementally(valid_parts, paths["events"])
    daily, stats = _stream_feature_arrays(valid_parts, start, end, selected, config)
    stats["available_file_dates"] = [
        pd.Timestamp(part.stem).date().isoformat() for part in valid_parts
    ]
    monthly = _monthly_features(daily)
    write_table(daily, paths["features"])
    write_table(monthly, paths["monthly"])

    failed_dates = {
        pd.Timestamp(entry["date"]).date().isoformat()
        for entry in failure_by_date.values()
        if entry.get("date")
    }
    validation = _validation(stats, start, end, selected, failed_dates)
    write_json(validation, paths["validation"])
    metadata = {
        "source": "GDELT Event Database",
        "official_url_template": source.get("url_template", f"{BASE_URL}/{{date}}.export.CSV.zip"),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "download_timestamp": vintage_now(),
        "countries": selected,
        "country_selection": "Comtrade country_universe.csv when available, otherwise configured ISO3 list",
        "event_filter": {
            "supply_chain_event_roots": source.get(
                "supply_chain_event_roots", ["14", "16", "17", "18", "19"]
            ),
            "labor_unrest_event_roots": source.get("labor_unrest_event_roots", ["14"]),
        },
        "negative_tone_threshold": float(source.get("negative_tone_threshold", -2.0)),
        "feature_definitions": {
            "news_vol_7d": "count of supply-chain-relevant event-country rows by DATEADDED in trailing 7 calendar days, inclusive",
            "neg_tone_frac_3d": "fraction of relevant event-country rows by DATEADDED in trailing 3 calendar days with AvgTone <= configured threshold",
            "strike_flag_7d": "1 when any configured labor-unrest root event by DATEADDED occurs in trailing 7 calendar days, otherwise 0",
            "monthly_alignment": "use the daily feature row at UTC month end; no future event/reference/availability timestamp is used",
        },
        "retained_event_columns": EVENT_COLUMNS,
        "file_count": len(valid_parts),
        "downloaded_file_count": downloaded,
        "reused_file_count": reused,
        "failed_file_count": len(failure_by_date),
        "download_workers": workers,
        "raw_event_country_row_count": int(events_rows),
        "event_count": int(stats["event_count"]),
        "relevant_event_count": int(stats["relevant_event_count"]),
        "labor_unrest_event_count": int(stats["labor_unrest_event_count"]),
        "country_count": int(
            daily["country"].nunique() if not daily.empty else 0
        ),
        "date_coverage": {
            "first_event_date": stats.get("first_event_date"),
            "last_event_date": stats.get("last_event_date"),
        },
        "validation_report": str(paths["validation"]),
        "test_mode": test_mode,
    }
    write_json(metadata, paths["metadata"])
    LOGGER.info("[GDELT] Processed: %s event-country rows", processed_rows)
    LOGGER.info("[GDELT] Relevant countries: %s", metadata["country_count"])
    LOGGER.info("[GDELT] Labor unrest events: %s", metadata["labor_unrest_event_count"])
    LOGGER.info("[GDELT] Failed downloads: %s", len(failure_by_date))
    LOGGER.info("[GDELT] Output rows: %s", events_rows)
    LOGGER.info(
        "[GDELT] Summary: discovered=%s downloaded=%s reused=%s failed=%s "
        "skipped=%s workers=%s raw_event_country_rows=%s relevant=%s "
        "labor_unrest=%s countries=%s",
        discovered,
        downloaded,
        reused,
        len(failure_by_date),
        0,
        workers,
        stats["event_count"],
        stats["relevant_event_count"],
        stats["labor_unrest_event_count"],
        metadata["country_count"],
    )
    max_failed = int(source.get("max_failed_files", 0))
    if len(failure_by_date) > max_failed:
        raise RuntimeError(
            f"GDELT dataset incomplete: {len(failure_by_date)} failed files exceeds "
            f"configured tolerance {max_failed}; see {paths['failures']}"
        )
    if validation["status"] != "PASS":
        raise RuntimeError(f"GDELT validation failed; see {paths['validation']}")
    return paths["events"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_cli(parser)
    parser.add_argument("--mode", choices=["auto", "bigquery", "raw"], default="raw")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Concurrent download workers (config/default 8; maximum 16)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process only the first N discovered dates for a bounded smoke test",
    )
    parser.add_argument(
        "--gdelt-test",
        action="store_true",
        help="Download only the configured test range (default: 2020-01-01 through 2020-01-03)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    output = ingest(
        load_config(args.config),
        force=args.force,
        mode=args.mode,
        test_mode=args.gdelt_test,
        workers=args.workers,
        limit=args.limit,
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
