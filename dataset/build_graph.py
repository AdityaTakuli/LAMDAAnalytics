"""Build a serialized country-level heterogeneous temporal graph.

The e-commerce study keeps the observable resolution honest: country-to-country
Comtrade edges carry the monthly trade signals and no firm-level edges are
invented from country data.  ``node_type`` remains explicit for schema
compatibility, while the current graph contains country nodes only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from common import add_common_cli, load_config, nested, read_table, write_json
from fuse_dataset import FEATURES


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def build_graph(nodes: pd.DataFrame, edges: pd.DataFrame, config: dict) -> dict[str, Any]:
    node_ids = sorted(nodes["node_id"].astype(str).unique().tolist()) if not nodes.empty else []
    node_types = {
        node_id: str(nodes.loc[nodes["node_id"].astype(str) == node_id, "node_type"].iloc[0])
        for node_id in node_ids
    }
    index_by_type: dict[str, dict[str, int]] = {"country": {}, "firm": {}}
    for node_type in ("country", "firm"):
        ids = [node_id for node_id in node_ids if node_types[node_id] == node_type]
        index_by_type[node_type] = {node_id: index for index, node_id in enumerate(ids)}

    node_records = []
    for _, row in nodes.sort_values(["month", "node_id"]).iterrows():
        record = {
            "timestamp": str(row["timestamp"]),
            "month": str(row["month"]),
            "node_id": str(row["node_id"]),
            "node_type": str(row["node_type"]),
            "node_index": index_by_type[str(row["node_type"])][str(row["node_id"])],
            "host_country_id": str(row["host_country_id"]),
            "features": {feature: float(row[feature]) for feature in FEATURES},
            "labels": {
                key: (None if pd.isna(row[key]) else int(row[key]))
                for key in row.index
                if key == "label" or key.startswith("label_tau_")
            },
            "inbound_flow_usd": float(row.get("inbound_flow_usd", 0.0)),
            "feature_provenance": str(row.get("feature_provenance", "{}")),
            "is_inherited": _as_bool(row.get("is_inherited", False)),
            "vintage_date": str(row.get("vintage_date", "")),
        }
        node_records.append(record)

    edge_records = []
    if not edges.empty:
        for _, row in edges.sort_values(["timestamp", "source", "destination"]).iterrows():
            source_id, destination_id = str(row["source"]), str(row["destination"])
            if source_id not in node_types or destination_id not in node_types:
                continue
            edge_records.append(
                {
                    "timestamp": str(row["timestamp"]),
                    "month": str(row["month"]),
                    "source": source_id,
                    "destination": destination_id,
                    "source_type": str(row["source_type"]),
                    "destination_type": str(row["destination_type"]),
                    "source_index": index_by_type[str(row["source_type"])][source_id],
                    "destination_index": index_by_type[str(row["destination_type"])][destination_id],
                    "edge_type": str(row["edge_type"]),
                    "trade_value_usd": float(row.get("trade_value_usd", 0.0)),
                    "flow_volume": float(row.get("flow_volume", 0.0)),
                    "provenance": str(row.get("provenance", "")),
                    "is_observed": _as_bool(row.get("is_observed", False)),
                    "is_inherited": _as_bool(row.get("is_inherited", False)),
                }
            )

    months = sorted(nodes["month"].astype(str).unique().tolist()) if not nodes.empty else []
    return {
        "format": "lamda-heterogeneous-temporal-graph-v1",
        "feature_names": FEATURES,
        "months": months,
        "node_types": {
            node_type: {
                "ids": list(mapping.keys()),
                "count": len(mapping),
            }
            for node_type, mapping in index_by_type.items()
        },
        "nodes": node_records,
        "edges": edge_records,
        "metadata": {
            "study": "India-centered e-commerce consumer goods basket",
            "trade_edges_are_country_level_only": True,
            "firm_trade_disaggregation": False,
            "cset_topology_used": False,
            "analysis_start": nested(config, "analysis", "start_date"),
            "analysis_end": nested(config, "analysis", "end_date"),
        },
    }


def build(config: dict) -> Path:
    base = Path(__file__).resolve().parent
    nodes = read_table(base / nested(config, "outputs", "nodes", default="processed/nodes_monthly.csv"))
    edges = read_table(base / nested(config, "outputs", "edges", default="processed/edges_monthly.csv"))
    graph = build_graph(nodes, edges, config)
    destination = base / nested(config, "outputs", "graph", default="processed/graph.json")
    write_json(graph, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_cli(parser)
    args = parser.parse_args()
    output = build(load_config(args.config))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

