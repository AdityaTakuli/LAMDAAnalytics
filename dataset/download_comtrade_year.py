"""Download one additional Comtrade year at a time.

Only the 2021-2023 bilateral HS 8541/8542 JSON cache is touched. Successful
responses from the existing three-year acquisition are reused; this command
does not write processed tables, graphs, or model outputs.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any

import pandas as pd

from common import ensure_parent, load_config, nested, vintage_now
from ingest_comtrade_bilateral import _api_request


BASE = Path(__file__).resolve().parent


def _source_path(config: dict[str, Any]) -> Path:
    value = nested(
        config,
        "sources",
        "comtrade",
        "bilateral_cache_dir",
        default="data/three_year_2021_2023/cache/comtrade/bilateral",
    )
    path = Path(value)
    return path if path.is_absolute() else BASE / path


def _root(config: dict[str, Any]) -> Path:
    value = nested(
        config,
        "outputs",
        "root",
        default="data/three_year_2021_2023",
    )
    path = Path(value)
    return path if path.is_absolute() else BASE / path


def _atomic_json(path: Path, payload: Any) -> None:
    ensure_parent(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _cached_response(path: Path) -> tuple[bool, int]:
    if not path.exists():
        return False, 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        response = payload.get("response")
        if not isinstance(response, dict) or not isinstance(response.get("data"), list):
            return False, 0
        return True, len(response["data"])
    except (OSError, ValueError, TypeError):
        return False, 0


def run(config: dict[str, Any], year: int) -> Path:
    if year not in {2021, 2022, 2023}:
        raise ValueError("This command accepts only years 2021, 2022, or 2023")
    cache = _source_path(config)
    cache.mkdir(parents=True, exist_ok=True)
    months = [
        stamp.strftime("%Y%m")
        for stamp in pd.date_range(f"{year}-01-01", f"{year}-12-01", freq="MS")
    ]
    source = nested(config, "sources", "comtrade", default={})
    reporters = [str(value) for value in source["reporter_codes"]]
    partners = [str(value) for value in source["partner_codes"]]
    manifest_path = _root(config) / f"comtrade_{year}_manifest.json"
    manifest = {
        "status": "downloading",
        "year": year,
        "period_start": months[0],
        "period_end": months[-1],
        "reporters": reporters,
        "partners": partners,
        "hs_codes": ["8541", "8542"],
        "flow_code": "M",
        "download_timestamp": vintage_now(),
        "processed": False,
        "trained": False,
        "requests": {},
    }
    if manifest_path.exists():
        try:
            prior = json.loads(manifest_path.read_text(encoding="utf-8"))
            if prior.get("year") == year:
                manifest["requests"] = prior.get("requests", {})
        except (OSError, ValueError):
            pass
    _atomic_json(manifest_path, manifest)

    # A 403 is an authorization response, not a transient network failure.
    # One request per configured credential is enough to confirm it without
    # repeatedly spending retry time on a known denied request.
    request_config = deepcopy(config)
    request_config.setdefault("sources", {}).setdefault("comtrade", {})[
        "max_retries"
    ] = 0
    downloaded = reused = failed = 0
    total = len(reporters) * len(months)
    completed = 0
    for reporter in reporters:
        for period in months:
            key = f"{reporter}:{period}"
            destination = cache / f"reporter_{int(reporter):03d}_{period}.json"
            valid, records = _cached_response(destination)
            try:
                if valid:
                    reused += 1
                    cached = True
                else:
                    response = _api_request(request_config, reporter, partners, period)
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
                    records = len(response.get("data", []))
                    downloaded += 1
                    cached = False
                    delay = float(source.get("request_delay_seconds", 1.0))
                    if delay:
                        time.sleep(delay)
                manifest["requests"][key] = {
                    "status": "success",
                    "cached": cached,
                    "records": records,
                    "cache_file": str(destination),
                }
            except Exception as exc:
                failed += 1
                manifest["requests"][key] = {
                    "status": "failed",
                    "error": str(exc),
                    "cache_file": str(destination),
                }
            completed += 1
            if completed % 20 == 0 or completed == total:
                _atomic_json(manifest_path, manifest)
                print(f"[Comtrade {year}] completed {completed}/{total}")
    manifest.update(
        {
            "status": "complete" if failed == 0 else "completed_with_failures",
            "downloaded_requests": downloaded,
            "reused_requests": reused,
            "failed_requests": failed,
            "cache_dir": str(cache),
        }
    )
    _atomic_json(manifest_path, manifest)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_three_year_download.yaml")
    parser.add_argument(
        "--year",
        type=int,
        action="append",
        choices=(2021, 2022, 2023),
        required=True,
        help="Year to download; years run sequentially in the order supplied",
    )
    args = parser.parse_args()
    for year in args.year:
        output = run(load_config(args.config), year)
        print(f"Wrote {output}")
    print("Comtrade acquisition finished; no processing or training was run.")


if __name__ == "__main__":
    main()
