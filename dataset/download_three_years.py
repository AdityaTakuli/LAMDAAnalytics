"""Download only the additional 2021-2023 source data.

The downloader is acquisition-only: it never fuses, processes, or trains.
Every output is written below ``data/three_year_2021_2023``.  Existing
2024/monthly/daily directories and caches are never read as write targets.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
import json
from pathlib import Path
import time
from typing import Any
import pandas as pd
import requests

from common import ensure_parent, env_value, load_config, nested, request_headers, stable_id, vintage_now
from ingest_comtrade_bilateral import _api_request
from ingest_gdelt import _download_task, _source_url, _valid_zip


BASE = Path(__file__).resolve().parent


def _path(config: dict[str, Any], section: str, key: str, default: str) -> Path:
    value = nested(config, section, key, default=default)
    path = Path(value)
    return path if path.is_absolute() else BASE / path


def _source_path(
    config: dict[str, Any], source_name: str, key: str, default: str
) -> Path:
    value = nested(config, "sources", source_name, key, default=default)
    path = Path(value)
    return path if path.is_absolute() else BASE / path


def _atomic_json(path: Path, payload: Any) -> None:
    ensure_parent(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _dates(config: dict[str, Any]) -> list[date]:
    start = pd.Timestamp(nested(config, "analysis", "start_date")).date()
    end = pd.Timestamp(nested(config, "analysis", "end_date")).date()
    return [stamp.date() for stamp in pd.date_range(start, end, freq="D")]


def _months(config: dict[str, Any]) -> list[str]:
    start = pd.Timestamp(nested(config, "analysis", "start_date"))
    end = pd.Timestamp(nested(config, "analysis", "end_date"))
    return [stamp.strftime("%Y%m") for stamp in pd.date_range(start, end, freq="MS")]


def _download_gdelt(
    config: dict[str, Any], dates: list[date], manifest: dict[str, Any]
) -> dict[str, Any]:
    source = nested(config, "sources", "gdelt", default={})
    cache = _source_path(
        config, "gdelt", "cache_dir", "data/three_year_2021_2023/cache/gdelt"
    )
    cache.mkdir(parents=True, exist_ok=True)
    pending = []
    reused = 0
    for current in dates:
        stamp = current.strftime("%Y%m%d")
        destination = cache / current.strftime("%Y") / f"{stamp}.export.CSV.zip"
        if destination.exists() and _valid_zip(destination):
            reused += 1
        else:
            pending.append((current, _source_url(config, stamp), destination))

    failures: list[dict[str, Any]] = []
    downloaded = 0
    workers = int(source.get("download_workers", 8))
    if pending:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gdelt-2021-2023") as pool:
            futures = {
                pool.submit(_download_task, current, url, destination, config, False): (
                    current,
                    url,
                    destination,
                )
                for current, url, destination in pending
            }
            for position, future in enumerate(as_completed(futures), start=1):
                current, url, destination = futures[future]
                try:
                    _, ok, attempts, error = future.result()
                except Exception as exc:
                    ok, attempts, error = False, 1, str(exc)
                if ok:
                    downloaded += 1
                else:
                    failures.append(
                        {
                            "date": current.isoformat(),
                            "url": url,
                            "attempts": attempts,
                            "error": error or "download failed",
                        }
                    )
                if position % 50 == 0 or position == len(pending):
                    print(f"[GDELT 2021-2023] completed {position}/{len(pending)}")
    result = {
        "date_start": dates[0].isoformat(),
        "date_end": dates[-1].isoformat(),
        "expected_files": len(dates),
        "downloaded_files": downloaded,
        "reused_files": reused,
        "failed_files": len(failures),
        "failures": failures,
        "cache_dir": str(cache),
        "processed": False,
    }
    manifest["sources"]["gdelt"] = result
    _atomic_json(_path(config, "outputs", "download_manifest", "data/three_year_2021_2023/download_manifest.json"), manifest)
    return result


def _request_json(
    url: str, params: dict[str, Any], config: dict[str, Any], destination: Path
) -> tuple[bool, int, str | None]:
    source = nested(config, "sources", "weather", default={})
    retries = int(source.get("max_retries", 4))
    timeout = float(source.get("timeout_seconds", 120))
    backoff = float(source.get("backoff_seconds", 2))
    if destination.exists():
        try:
            payload = json.loads(destination.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and nested(payload, "properties", "parameter", "T2M", default=None) is not None:
                return True, 0, None
        except (OSError, ValueError):
            destination.unlink()
    last_error: str | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(
                url, params=params, headers=request_headers(), timeout=timeout
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or nested(payload, "properties", "parameter", "T2M", default=None) is None:
                raise ValueError("NASA POWER response did not contain T2M data")
            ensure_parent(destination)
            temporary = destination.with_name(destination.name + ".tmp")
            temporary.write_text(
                json.dumps(payload, default=str), encoding="utf-8"
            )
            temporary.replace(destination)
            return True, attempt, None
        except (OSError, ValueError, requests.RequestException) as exc:
            last_error = str(exc)
            if attempt < retries:
                time.sleep(min(backoff * (2 ** (attempt - 1)), 300))
    return False, retries, last_error


def _download_weather(
    config: dict[str, Any], dates: list[date], manifest: dict[str, Any]
) -> dict[str, Any]:
    source = nested(config, "sources", "weather", default={})
    cache = _source_path(
        config, "weather", "cache_dir", "data/three_year_2021_2023/cache/weather"
    )
    start = dates[0]
    end = dates[-1]
    failures = []
    downloaded = 0
    reused = 0
    for location in source.get("locations", []):
        location_key = stable_id("location", location["name"])
        destination = cache / "nasa_power" / f"{location_key}_{start:%Y%m%d}_{end:%Y%m%d}.json"
        ok, attempts, error = _request_json(
            "https://power.larc.nasa.gov/api/temporal/daily/point",
            {
                "parameters": "T2M",
                "community": "AG",
                "longitude": location["longitude"],
                "latitude": location["latitude"],
                "start": start.strftime("%Y%m%d"),
                "end": end.strftime("%Y%m%d"),
                "format": "JSON",
            },
            config,
            destination,
        )
        if ok:
            if attempts == 0:
                reused += 1
            else:
                downloaded += 1
        else:
            failures.append(
                {
                    "location": location["name"],
                    "attempts": attempts,
                    "error": error or "download failed",
                }
            )
        print(f"[Weather 2021-2023] {location['name']}: {'ok' if ok else 'failed'}")
    result = {
        "date_start": start.isoformat(),
        "date_end": end.isoformat(),
        "expected_location_files": len(source.get("locations", [])),
        "downloaded_files": downloaded,
        "reused_files": reused,
        "failed_files": len(failures),
        "failures": failures,
        "cache_dir": str(cache),
        "processed": False,
    }
    manifest["sources"]["weather"] = result
    _atomic_json(_path(config, "outputs", "download_manifest", "data/three_year_2021_2023/download_manifest.json"), manifest)
    return result


def _download_comtrade(
    config: dict[str, Any], months: list[str], manifest: dict[str, Any]
) -> dict[str, Any]:
    source = nested(config, "sources", "comtrade", default={})
    cache = _source_path(
        config,
        "comtrade",
        "bilateral_cache_dir",
        "data/three_year_2021_2023/cache/comtrade/bilateral",
    )
    reporters = [str(code) for code in source["reporter_codes"]]
    partners = [str(code) for code in source["partner_codes"]]
    cache.mkdir(parents=True, exist_ok=True)
    requests_manifest = manifest["sources"].get("comtrade", {}).get("requests", {})
    failures = []
    downloaded = 0
    reused = 0
    total = len(reporters) * len(months)
    completed = 0
    for reporter in reporters:
        for period in months:
            key = f"{reporter}:{period}"
            destination = cache / f"reporter_{int(reporter):03d}_{period}.json"
            try:
                if destination.exists():
                    payload = json.loads(destination.read_text(encoding="utf-8"))
                    if not isinstance(payload, dict) or not isinstance(payload.get("response"), dict):
                        raise ValueError("invalid cached Comtrade response")
                    reused += 1
                    records = len(payload["response"].get("data", []))
                    cached = True
                else:
                    response = _api_request(config, reporter, partners, period)
                    payload = {
                        "request": {
                            "reporter_code": reporter,
                            "partner_codes": partners,
                            "period": period,
                            "flow_code": "M",
                            "commodity_codes": ["8541", "8542"],
                        },
                        "response": response,
                        "downloaded_at": vintage_now(),
                    }
                    _atomic_json(destination, payload)
                    downloaded += 1
                    records = len(response.get("data", []))
                    cached = False
                    delay = float(source.get("request_delay_seconds", 1.0))
                    if delay:
                        time.sleep(delay)
                requests_manifest[key] = {
                    "status": "success",
                    "cached": cached,
                    "records": records,
                    "cache_file": str(destination),
                }
            except Exception as exc:
                failures.append(
                    {"reporter_code": reporter, "period": period, "error": str(exc)}
                )
                requests_manifest[key] = {"status": "failed", "error": str(exc)}
            completed += 1
            if completed % 20 == 0 or completed == total:
                print(f"[Comtrade 2021-2023] completed {completed}/{total}")
                manifest["sources"]["comtrade"] = {
                    "requests": requests_manifest,
                    "completed": completed,
                    "expected_requests": total,
                    "failed_requests": len(failures),
                    "cache_dir": str(cache),
                    "processed": False,
                }
                _atomic_json(_path(config, "outputs", "download_manifest", "data/three_year_2021_2023/download_manifest.json"), manifest)
    result = {
        "period_start": months[0],
        "period_end": months[-1],
        "expected_requests": total,
        "downloaded_requests": downloaded,
        "reused_requests": reused,
        "failed_requests": len(failures),
        "failures": failures,
        "requests": requests_manifest,
        "cache_dir": str(cache),
        "processed": False,
    }
    manifest["sources"]["comtrade"] = result
    _atomic_json(_path(config, "outputs", "download_manifest", "data/three_year_2021_2023/download_manifest.json"), manifest)
    return result


def _download_gscpi(config: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    source = nested(config, "sources", "gscpi", default={})
    cache = _source_path(
        config, "gscpi", "cache_dir", "data/three_year_2021_2023/cache/gscpi"
    )
    destination = cache / "gscpi.xlsx"
    url = source["excel_url"]
    reused = destination.exists() and destination.stat().st_size > 0
    if not reused:
        retries = 4
        last_error = None
        for attempt in range(1, retries + 1):
            try:
                response = requests.get(url, headers=request_headers(), timeout=120)
                response.raise_for_status()
                ensure_parent(destination)
                temporary = destination.with_name(destination.name + ".tmp")
                temporary.write_bytes(response.content)
                temporary.replace(destination)
                last_error = None
                break
            except (OSError, requests.RequestException) as exc:
                last_error = str(exc)
                if attempt < retries:
                    time.sleep(min(2 ** attempt, 60))
        if last_error:
            raise RuntimeError(f"GSCPI download failed: {last_error}")
    result = {
        "url": url,
        "file": str(destination),
        "downloaded": not reused,
        "reused": reused,
        "note": "Raw workbook may contain a wider historical range; no filtering or normalization was performed.",
        "processed": False,
    }
    manifest["sources"]["gscpi"] = result
    _atomic_json(_path(config, "outputs", "download_manifest", "data/three_year_2021_2023/download_manifest.json"), manifest)
    return result


def run(config: dict[str, Any]) -> Path:
    root = _path(config, "outputs", "root", "data/three_year_2021_2023")
    root.mkdir(parents=True, exist_ok=True)
    (root / "raw").mkdir(exist_ok=True)
    (root / "cache").mkdir(exist_ok=True)
    manifest = {
        "status": "downloading",
        "date_range": {
            "start": nested(config, "analysis", "start_date"),
            "end": nested(config, "analysis", "end_date"),
        },
        "years": [2021, 2022, 2023],
        "download_timestamp": vintage_now(),
        "processed": False,
        "trained": False,
        "sources": {},
    }
    manifest_path = _path(
        config,
        "outputs",
        "download_manifest",
        "data/three_year_2021_2023/download_manifest.json",
    )
    _atomic_json(manifest_path, manifest)
    dates = _dates(config)
    months = _months(config)
    _download_gdelt(config, dates, manifest)
    _download_weather(config, dates, manifest)
    _download_comtrade(config, months, manifest)
    _download_gscpi(config, manifest)
    manifest["status"] = (
        "complete"
        if all(
            source.get("failed_files", source.get("failed_requests", 0)) == 0
            for source in manifest["sources"].values()
            if isinstance(source, dict)
        )
        else "completed_with_failures"
    )
    manifest["processed"] = False
    manifest["trained"] = False
    _atomic_json(manifest_path, manifest)
    (root / "raw" / "README.md").write_text(
        "Raw 2021-2023 downloads are stored in the source-specific cache directories.\n",
        encoding="utf-8",
    )
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_three_year_download.yaml")
    args = parser.parse_args()
    output = run(load_config(args.config))
    print(f"Download stage finished in {output}")
    print("No fusion, graph construction, or model training was run.")


if __name__ == "__main__":
    main()
