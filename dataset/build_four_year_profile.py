"""Build an isolated 2021-2024 processed profile without model training.

The command reuses the completed Comtrade caches, downloads the missing
2024 weather locations into the isolated profile, links existing GDELT raw
exports without copying them, normalizes GSCPI, and builds causal monthly
nodes, edges, and a graph. Existing one-year and three-year directories are
never overwritten.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import date, timedelta
import json
from pathlib import Path
import shutil
from typing import Any

import pandas as pd
import yaml

from build_graph import build as build_graph
from common import load_config, nested, write_table
from fuse_dataset import fuse
from ingest_comtrade_bilateral import _raw_frame, _standardize_bilateral, _read_cached
from ingest_gdelt import ingest as ingest_gdelt
from ingest_gscpi import ingest as ingest_gscpi
from ingest_weather import ingest as ingest_weather


BASE = Path(__file__).resolve().parent
THREE_YEAR = BASE / "data" / "three_year_2021_2023"
OUTPUT_ROOT = BASE / "data" / "four_year_2021_2024"


def _copy_config(config: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(config)
    result.pop("_config_path", None)
    result["analysis"] = {
        **result.get("analysis", {}),
        "start_date": "2021-01-01",
        "end_date": "2024-12-31",
        "monthly_step": "MS",
        "horizon_months": 1,
        "taus": [0.30, 0.35, 0.40],
        "default_tau": 0.35,
    }
    sources = result.setdefault("sources", {})
    gdelt = sources.setdefault("gdelt", {})
    gdelt.update(
        {
            "mode": "raw",
            "start_date": "2021-01-01",
            "end_date": "2024-12-31",
            "cache_dir": "data/four_year_2021_2024/cache/gdelt",
            "url_template": "https://data.gdeltproject.org/events/{date}.export.CSV.zip",
            "max_failed_files": 2,
            "negative_tone_threshold": -2.0,
            "labor_unrest_event_roots": ["14"],
            "supply_chain_event_roots": ["14", "16", "17", "18", "19"],
            "download_workers": 8,
        }
    )
    sources.setdefault("weather", {})["cache_dir"] = (
        "data/four_year_2021_2024/cache/weather"
    )
    sources.setdefault("gscpi", {})["cache_dir"] = (
        "data/four_year_2021_2024/cache/gscpi"
    )
    result["outputs"] = {
        "comtrade": "data/four_year_2021_2024/processed/comtrade.csv",
        "gdelt": "data/four_year_2021_2024/processed/gdelt_events.csv",
        "gdelt_features": "data/four_year_2021_2024/processed/gdelt_features.csv",
        "gdelt_monthly_features": "data/four_year_2021_2024/processed/gdelt_monthly_features.csv",
        "gdelt_metadata": "data/four_year_2021_2024/processed/gdelt_metadata.json",
        "gdelt_validation": "data/four_year_2021_2024/processed/gdelt_validation.json",
        "weather": "data/four_year_2021_2024/processed/weather_daily.csv",
        "gscpi": "data/four_year_2021_2024/processed/gscpi_monthly.csv",
        "country_universe": "data/four_year_2021_2024/processed/country_universe.csv",
        "nodes": "data/four_year_2021_2024/processed/nodes_monthly.csv",
        "edges": "data/four_year_2021_2024/processed/edges_monthly.csv",
        "graph": "data/four_year_2021_2024/processed/graph.json",
        "analysis_results": "data/four_year_2021_2024/results/data_analysis",
    }
    return result


def _link_gdelt_sources(destination: Path) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    candidates = [
        THREE_YEAR / "cache" / "gdelt",
        BASE / "cache" / "gdelt",
        BASE / "data" / "one_year_2024" / "cache" / "gdelt",
    ]
    start = date(2021, 1, 1)
    end = date(2024, 12, 31)
    linked = missing = 0
    missing_dates: list[str] = []
    for offset in range((end - start).days + 1):
        current = start + timedelta(days=offset)
        stamp = current.strftime("%Y%m%d")
        target = destination / current.strftime("%Y") / f"{stamp}.export.CSV.zip"
        if target.exists():
            linked += 1
            continue
        source = next(
            (
                root / current.strftime("%Y") / target.name
                for root in candidates
                if (root / current.strftime("%Y") / target.name).exists()
            ),
            None,
        )
        if source is None:
            missing += 1
            missing_dates.append(stamp)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source.resolve())
        linked += 1
    return {
        "linked_exports": linked,
        "missing_exports": missing,
        "missing_dates": missing_dates,
        "source_roots": [str(root) for root in candidates],
    }


def _build_comtrade(config: dict[str, Any]) -> Path:
    output = BASE / nested(config, "outputs", "comtrade")
    frames: list[pd.DataFrame] = []
    cache_roots = {
        2021: THREE_YEAR / "cache" / "comtrade" / "bilateral",
        2022: THREE_YEAR / "cache" / "comtrade" / "bilateral",
        2023: THREE_YEAR / "cache" / "comtrade" / "bilateral",
        2024: BASE / "data" / "one_year_2024" / "cache" / "comtrade" / "bilateral",
    }
    for year, root in cache_roots.items():
        for path in sorted(root.glob(f"reporter_*_{year}*.json")):
            payload = _read_cached(path)
            period = path.stem.rsplit("_", 1)[-1]
            frame = _standardize_bilateral(_raw_frame(payload), period, pd.Timestamp.now(tz="UTC").isoformat())
            if not frame.empty:
                frames.append(frame)
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    write_table(result, output)
    return output


def _copy_universe(config: dict[str, Any]) -> Path:
    destination = BASE / nested(config, "outputs", "country_universe")
    source = BASE / "data" / "one_year_2024" / "processed" / "country_universe.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def run(config_path: str = "config_three_year_download.yaml") -> Path:
    config = _copy_config(load_config(config_path))
    output_root = OUTPUT_ROOT
    (output_root / "processed").mkdir(parents=True, exist_ok=True)
    (output_root / "cache").mkdir(parents=True, exist_ok=True)
    (output_root / "results").mkdir(parents=True, exist_ok=True)
    (output_root / "config_used.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    _copy_universe(config)
    link_summary = _link_gdelt_sources(
        output_root / "cache" / "gdelt"
    )
    _build_comtrade(config)
    ingest_weather(config, force=False)
    workbook = THREE_YEAR / "cache" / "gscpi" / "gscpi.xlsx"
    ingest_gscpi(config, force=False, input_path=str(workbook))
    ingest_gdelt(config, force=False, mode="raw", workers=8)
    fuse(config, synthetic=False)
    build_graph(config)
    (output_root / "results" / "profile_build_summary.json").write_text(
        json.dumps(
            {
                "profile": "four_year_2021_2024",
                "processed": True,
                "trained": False,
                "gdelt_link_summary": link_summary,
                "weather_provider": nested(config, "sources", "weather", "provider"),
                "comtrade_source_years": [2021, 2022, 2023, 2024],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_three_year_download.yaml")
    args = parser.parse_args()
    output = run(args.config)
    print(f"Built four-year profile at {output}")
    print("Model training was not run.")


if __name__ == "__main__":
    main()
