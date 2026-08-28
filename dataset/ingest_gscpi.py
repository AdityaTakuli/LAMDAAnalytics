"""Download and normalize the NY Fed Global Supply Chain Pressure Index.

The NY Fed periodically changes the direct spreadsheet URL.  Configuration
may provide ``excel_url``; otherwise this script discovers an ``.xls`` or
``.xlsx`` link from the official overview page and caches both the page and
spreadsheet.  The output is one scalar per month with a retrieval vintage.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests

from common import (
    add_common_cli,
    canonical_column,
    ensure_parent,
    load_config,
    nested,
    request_headers,
    vintage_now,
    write_table,
)


def _find_excel(config: dict, cache_dir: Path, force: bool) -> Path:
    source = nested(config, "sources", "gscpi", default={})
    configured_url = source.get("excel_url")
    overview = source.get(
        "overview_url", "https://www.newyorkfed.org/research/policy/gscpi#/overview"
    )
    html_path = cache_dir / "overview.html"
    if force or not html_path.exists():
        response = requests.get(overview, headers=request_headers(), timeout=60)
        response.raise_for_status()
        ensure_parent(html_path)
        html_path.write_text(response.text, encoding="utf-8")
    html = html_path.read_text(encoding="utf-8")
    links = re.findall(r"""(?:href|src)\s*=\s*["']([^"']+\.(?:xlsx?|xls)(?:\?[^"']*)?)["']""", html, re.I)
    url = configured_url or (urljoin(overview, links[0]) if links else None)
    if not url:
        raise RuntimeError(
            "Could not discover the NY Fed Excel URL. Set sources.gscpi.excel_url "
            "in dataset/config.yaml or pass --input."
        )
    suffix = ".xlsx" if ".xlsx" in url.lower() else ".xls"
    target = cache_dir / f"gscpi{suffix}"
    if force or not target.exists():
        response = requests.get(url, headers=request_headers(), timeout=120)
        response.raise_for_status()
        ensure_parent(target)
        target.write_bytes(response.content)
    return target


def normalize(path: Path, vintage: str) -> pd.DataFrame:
    sheets = pd.read_excel(path, sheet_name=None)
    candidates: list[pd.DataFrame] = []
    for frame in sheets.values():
        if frame.empty:
            continue
        date_column = canonical_column(frame, ["date", "month", "observation_date", "time"])
        value_column = canonical_column(
            frame, ["gscpi", "global supply chain pressure index", "value", "index"]
        )
        if date_column and value_column:
            candidates.append(frame[[date_column, value_column]].rename(columns={date_column: "date", value_column: "global_risk"}))
            break
    if not candidates:
        raise ValueError(f"No date/GSCPI columns found in {path}")
    frame = candidates[0].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    frame["global_risk"] = pd.to_numeric(frame["global_risk"], errors="coerce")
    frame = frame.dropna(subset=["date", "global_risk"])
    result = (
        frame.assign(month=frame["date"].dt.strftime("%Y-%m"))
        .groupby("month", as_index=False)["global_risk"]
        .last()
    )
    result["published_at"] = vintage
    result["available_at"] = vintage
    result["vintage_date"] = vintage
    result["source"] = "nyfed_gscpi"
    return result[["month", "global_risk", "published_at", "available_at", "vintage_date", "source"]]


def ingest(config: dict, force: bool = False, input_path: str | None = None) -> Path:
    destination = Path(__file__).resolve().parent / nested(
        config, "outputs", "gscpi", default="processed/gscpi_monthly.csv"
    )
    if destination.exists() and not force:
        return destination
    cache_dir = Path(__file__).resolve().parent / nested(
        config, "sources", "gscpi", "cache_dir", default="cache/gscpi"
    )
    path = Path(input_path) if input_path else _find_excel(config, cache_dir, force)
    result = normalize(path, vintage_now())
    start = pd.Timestamp(nested(config, "analysis", "start_date"))
    end = pd.Timestamp(nested(config, "analysis", "end_date"))
    result = result[
        (pd.to_datetime(result["month"] + "-01") >= start.tz_localize(None))
        & (pd.to_datetime(result["month"] + "-01") <= end.tz_localize(None))
    ]
    return write_table(result, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_cli(parser)
    parser.add_argument("--input", help="Local NY Fed Excel file")
    args = parser.parse_args()
    output = ingest(load_config(args.config), force=args.force, input_path=args.input)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

