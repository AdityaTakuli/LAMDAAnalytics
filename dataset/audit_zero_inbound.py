"""Audit zero-inbound country-months before the regression pilot.

The audit is report-only. It never changes source data, drops rows, or fills
missing trade with zero. Human review of the generated CSV controls any later
exclusion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from common import load_config, nested, stable_id


BASE = Path(__file__).resolve().parent
FOUR_YEAR_ROOT = BASE / "data" / "four_year_2021_2024"
FOUR_YEAR_PROCESSED = FOUR_YEAR_ROOT / "processed"
BILATERAL_CACHE = FOUR_YEAR_ROOT / "cache" / "comtrade" / "bilateral"
OUTPUT = FOUR_YEAR_ROOT / "results" / "zero_value_audit.csv"


def _cache_file(reporter: str, period: str) -> Path:
    return BILATERAL_CACHE / f"reporter_{int(reporter):03d}_{period}.json"


def _response(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    response = payload.get("response")
    if not isinstance(response, dict):
        raise ValueError(f"Invalid Comtrade response wrapper: {path}")
    return response


def _classify(reporter: str, period: str) -> list[dict[str, str]]:
    path = _cache_file(reporter, period)
    base = {
        "reporter": reporter,
        "period": period,
    }
    if not path.exists():
        return [
            {
                **base,
                "partner": "ALL_SELECTED_PARTNERS",
                "raw_response_type": "response_file_missing",
                "classification": "artifact_suspected",
                "recommended_action": "exclude",
                "human_decision": "",
            }
        ]
    response = _response(path)
    data = response.get("data", [])
    if not isinstance(data, list) or not data:
        return [
            {
                **base,
                "partner": "ALL_SELECTED_PARTNERS",
                "raw_response_type": "successful_response_no_records",
                "classification": "artifact_suspected",
                "recommended_action": "exclude",
                "human_decision": "",
            }
        ]

    rows: list[dict[str, str]] = []
    grouped: dict[str, list[float]] = {}
    for item in data:
        partner = str(item.get("partnerCode", "UNKNOWN"))
        try:
            value = float(item.get("primaryValue", 0.0))
        except (TypeError, ValueError):
            value = 0.0
        grouped.setdefault(partner, []).append(value)
    for partner, values in sorted(grouped.items()):
        genuine = all(value == 0.0 for value in values)
        rows.append(
            {
                **base,
                "partner": partner,
                "raw_response_type": (
                    "successful_records_zero_trade_value"
                    if genuine
                    else "successful_records_positive_trade_value"
                ),
                "classification": "genuine_zero" if genuine else "artifact_suspected",
                "recommended_action": "keep" if genuine else "exclude",
                "human_decision": "",
            }
        )
    return rows


def run(
    nodes_path: str | Path = FOUR_YEAR_PROCESSED / "nodes_monthly.csv",
    output_path: str | Path = OUTPUT,
) -> Path:
    config = load_config("config.yaml")
    node_to_code = {
        str(location.get("node_id") or stable_id("country", location["name"])): str(
            location["comtrade_code"]
        )
        for location in nested(config, "sources", "weather", "locations", default=[])
    }
    nodes = pd.read_csv(nodes_path)
    required = {"host_country_id", "month", "inbound_flow_usd"}
    missing = required - set(nodes.columns)
    if missing:
        raise ValueError(f"Processed nodes are missing required columns: {sorted(missing)}")
    zero_rows = nodes[
        pd.to_numeric(nodes["inbound_flow_usd"], errors="coerce").fillna(0.0).eq(0.0)
    ][["host_country_id", "month"]].drop_duplicates()
    records: list[dict[str, str]] = []
    for _, row in zero_rows.sort_values(["host_country_id", "month"]).iterrows():
        host_country_id = str(row["host_country_id"])
        reporter = node_to_code.get(host_country_id, host_country_id)
        if reporter.startswith("country_"):
            reporter = reporter.rsplit("_", 1)[-1]
        period = str(row["month"]).replace("-", "")
        records.extend(_classify(reporter, period))
    result = pd.DataFrame(
        records,
        columns=[
            "reporter",
            "partner",
            "period",
            "raw_response_type",
            "classification",
            "recommended_action",
            "human_decision",
        ],
    )
    destination = Path(output_path)
    if not destination.is_absolute():
        destination = BASE / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    print(f"Audited {len(zero_rows)} zero-inbound country-months.")
    print(result["classification"].value_counts(dropna=False).to_string())
    print(f"Wrote {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", default=str(FOUR_YEAR_PROCESSED / "nodes_monthly.csv"))
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    run(args.nodes, args.output)


if __name__ == "__main__":
    main()
