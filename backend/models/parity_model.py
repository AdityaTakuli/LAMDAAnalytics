"""Serve the offline-evaluated models on the published-series features.

``models/serving/bundle.json`` and ``models/serving/published_features.csv``
are written by ``dataset/export_serving_model.py``. This module applies
the bundle with plain Python arithmetic (no scikit-learn at serving time):

    x_raw   = row[features], NaN -> 0, protective features negated
    z       = (x_raw - mean) / scale        # train-only statistics
    y_hat   = intercept + coef . z

which is exactly what the offline ridge computed. ``test_serving_parity.py``
checks the served predictions against the offline ones row by row.
"""
from __future__ import annotations

import csv
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

SERVING_DIR = Path(__file__).resolve().parent / "serving"

# ISO3 -> node id used in the fused panel (dataset/fuse_dataset.py naming).
ISO3_TO_NODE = {
    "CHN": "country_cn", "KOR": "country_kr", "HKG": "country_344", "SGP": "country_sg",
    "VNM": "country_vn", "MYS": "country_my", "JPN": "country_jp", "THA": "country_th",
    "IRL": "country_372", "USA": "country_us", "MEX": "country_mx", "PHL": "country_ph",
    "DEU": "country_de", "GBR": "country_gb", "FRA": "country_fr", "ITA": "country_380",
    "CZE": "country_203", "IND": "country_in",
}


class ForecastNotAvailable(LookupError):
    pass


@lru_cache(maxsize=1)
def _bundle() -> dict[str, Any]:
    return json.loads((SERVING_DIR / "bundle.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _features() -> dict[tuple[str, str], dict[str, float | None]]:
    table: dict[tuple[str, str], dict[str, float | None]] = {}
    with (SERVING_DIR / "published_features.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row.pop("node_id"), row.pop("month"))
            table[key] = {k: (float(v) if v not in ("", "nan", "NaN") else None) for k, v in row.items()}
    return table


def resolve_node(country: str) -> str:
    value = country.strip()
    if value.upper() in ISO3_TO_NODE:
        return ISO3_TO_NODE[value.upper()]
    if not value.startswith("country_"):
        value = f"country_{value.lower()}"
    return value


def predict_head(head: dict[str, Any], row: dict[str, float | None]) -> float:
    total = float(head["intercept"])
    for name, mean, scale, coef in zip(head["features"], head["mean"], head["scale"], head["coef"]):
        value = row.get(name)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            value = float(head["raw_nan_fill"])
        if name in head["sign_flipped"]:
            value = -value
        total += coef * (value - mean) / scale
    return total


def forecast(country: str, month: str) -> dict[str, Any]:
    node = resolve_node(country)
    row = _features().get((node, month))
    if row is None:
        raise ForecastNotAvailable(f"No published-series features for {node} at {month}")
    bundle = _bundle()
    heads = {}
    for name, head in bundle["heads"].items():
        heads[name] = {
            "prediction": predict_head(head, row),
            "target": head["target"],
            "target_definition": head["target_definition"],
            "offline_test_2024_r2": head["test_2024_r2"],
        }
    return {
        "node_id": node,
        "month": month,
        "forecast_for": "month T+1 inbound flow (HS 8541+8542, from panel partners)",
        "heads": heads,
        "features": {k: v for k, v in row.items()},
        "provenance": {
            "features": "published series fused offline (UN Comtrade, GDELT, NASA POWER, NY Fed GSCPI)",
            "model": bundle["source"],
            "train_months": bundle["train_months"],
        },
    }


def available_keys() -> list[tuple[str, str]]:
    return sorted(_features().keys())
