"""Download and normalize the static CSET semiconductor topology.

The GitHub repository is traversed through its public contents API.  Every
CSV is cached under ``cache/cset``.  Because the upstream files have evolved
over time, the normalizer uses column-name conventions instead of depending on
one particular filename.  Unrecognized files remain cached and are recorded
in the manifest for manual review.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from common import (
    add_common_cli,
    canonical_column,
    ensure_parent,
    load_config,
    nested,
    request_headers,
    stable_id,
    write_json,
    write_table,
)


def _contents(url: str) -> list[dict[str, Any]]:
    response = requests.get(url, headers=request_headers(), timeout=60)
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, list) else [payload]


def download_csvs(config: dict, force: bool = False) -> list[tuple[Path, str]]:
    source = nested(config, "sources", "cset", default={})
    api_url = source.get(
        "github_contents_url",
        "https://api.github.com/repos/georgetown-cset/eto-supply-chain/contents/data",
    )
    cache_dir = Path(__file__).resolve().parent / source.get("cache_dir", "cache/cset")
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []

    def visit(url: str, relative: str = "") -> list[tuple[Path, str]]:
        downloaded: list[tuple[Path, str]] = []
        for item in _contents(url):
            kind = item.get("type")
            name = item.get("name", "")
            item_path = f"{relative}/{name}".strip("/")
            if kind == "dir":
                downloaded.extend(visit(item["url"], item_path))
            elif kind == "file" and name.lower().endswith(".csv"):
                target = cache_dir / item_path
                ensure_parent(target)
                if force or not target.exists():
                    response = requests.get(
                        item.get("download_url") or item["url"],
                        headers=request_headers(),
                        timeout=60,
                    )
                    response.raise_for_status()
                    target.write_bytes(response.content)
                manifest.append({"path": item_path, "url": item.get("html_url", "")})
                downloaded.append((target, item_path))
        return downloaded

    result = visit(api_url)
    write_json(manifest, cache_dir / "manifest.json")
    return result


def _role(value: Any) -> str:
    text = str(value or "").strip().lower()
    for candidate in ("supplier", "manufacturer", "distributor"):
        if candidate in text:
            return candidate
    return text or "unknown"


def _looks_country(column: str) -> bool:
    return any(token in column.lower() for token in ("country", "nation", "iso", "location"))


def _looks_firm(column: str) -> bool:
    return any(
        token in column.lower()
        for token in ("firm", "company", "supplier", "manufacturer", "distributor", "organization", "entity")
    )


def normalize(files: list[tuple[Path, str]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    firm_country: list[dict[str, Any]] = []
    firm_firm: list[dict[str, Any]] = []
    recognized: set[str] = set()

    for path, relative in files:
        try:
            frame = pd.read_csv(path, dtype=str, low_memory=False)
        except Exception:
            continue
        if frame.empty:
            continue
        columns = [str(column) for column in frame.columns]
        role_column = canonical_column(frame, ["role", "entity_role", "type", "function"])
        country_columns = [column for column in columns if _looks_country(column)]
        firm_columns = [column for column in columns if _looks_firm(column) and column not in country_columns]

        # Common single-entity table: one firm/company and one country.
        firm_column = canonical_column(
            frame, ["firm", "firm_name", "company", "company_name", "organization", "entity_name"]
        )
        country_column = canonical_column(frame, ["country", "country_name", "nation", "iso3", "country_code"])
        if firm_column and country_column:
            recognized.add(relative)
            for _, row in frame.iterrows():
                firm_name = str(row.get(firm_column, "")).strip()
                country_name = str(row.get(country_column, "")).strip()
                if not firm_name or not country_name or firm_name.lower() == "nan":
                    continue
                firm_country.append(
                    {
                        "firm_id": stable_id("firm", firm_name),
                        "firm_name": firm_name,
                        "country_id": stable_id("country", country_name),
                        "country_name": country_name,
                        "role": _role(row.get(role_column, "")) if role_column else "unknown",
                        "source_file": relative,
                    }
                )

        # Relation tables: use explicit supplier/manufacturer/distributor
        # columns where possible, then fall back to the first two firm-like
        # columns. Country columns are deliberately excluded here.
        relation_columns = [column for column in firm_columns if column in columns]
        if len(relation_columns) < 2:
            relation_columns = [
                column for column in columns if not _looks_country(column) and _looks_firm(column)
            ]
        if len(relation_columns) >= 2:
            recognized.add(relative)
            left, right = relation_columns[:2]
            for _, row in frame.iterrows():
                left_name = str(row.get(left, "")).strip()
                right_name = str(row.get(right, "")).strip()
                if not left_name or not right_name or "nan" in {left_name.lower(), right_name.lower()}:
                    continue
                firm_firm.append(
                    {
                        "source_id": stable_id("firm", left_name),
                        "source_name": left_name,
                        "destination_id": stable_id("firm", right_name),
                        "destination_name": right_name,
                        "source_role": _role(left),
                        "destination_role": _role(right),
                        "relation": str(row.get(role_column, "topology")) if role_column else "topology",
                        "source_file": relative,
                    }
                )

    country_frame = pd.DataFrame(firm_country).drop_duplicates()
    firm_frame = pd.DataFrame(firm_firm).drop_duplicates()
    manifest_frame = pd.DataFrame({"source_file": sorted(recognized)})
    return country_frame, firm_frame, manifest_frame


def ingest(config: dict, force: bool = False) -> list[Path]:
    files = download_csvs(config, force=force)
    country, firm, manifest = normalize(files)
    output_dir = Path(__file__).resolve().parent / nested(
        config, "outputs", "cset_dir", default="processed/cset"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        write_table(country, output_dir / "firm_country.csv"),
        write_table(firm, output_dir / "firm_firm.csv"),
        write_table(manifest, output_dir / "recognized_files.csv"),
    ]
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_cli(parser)
    args = parser.parse_args()
    outputs = ingest(load_config(args.config), force=args.force)
    print("\n".join(f"Wrote {path}" for path in outputs))


if __name__ == "__main__":
    main()

