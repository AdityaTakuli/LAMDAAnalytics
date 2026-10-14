"""Served /forecast predictions must equal the offline evaluated predictions.

The offline side is results/paper21_revision/serving_parity_offline.csv,
written by dataset/rev21_export_serving_model.py with scikit-learn. The served
side is models/parity_model.py, both called directly and through the FastAPI
route. Run: backend/.venv/bin/python backend/test_serving_parity.py
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MAPPLS_API_KEY", "test")
os.environ.setdefault("SERP_API_KEY", "test")
os.environ.setdefault("WEATHER_API_KEY", "test")
os.environ.setdefault("GROQ_API_KEY", "test")

OFFLINE = Path(__file__).resolve().parents[1] / "results" / "paper21_revision" / "serving_parity_offline.csv"
TOLERANCE = 1e-9


def main() -> int:
    from fastapi.testclient import TestClient

    from main import app
    from models.parity_model import forecast

    rows = list(csv.DictReader(OFFLINE.open(newline="", encoding="utf-8")))
    worst = {}
    for row in rows:
        served = forecast(row["node_id"], row["month"])["heads"][row["head"]]["prediction"]
        diff = abs(served - float(row["offline_pred"]))
        worst[row["head"]] = max(worst.get(row["head"], 0.0), diff)
    for head, diff in worst.items():
        print(f"{head:20s} rows={sum(r['head'] == head for r in rows):3d}  max |served - offline| = {diff:.2e}")
        assert diff < TOLERANCE, f"{head} parity broken: {diff}"

    client = TestClient(app)
    sample = rows[0]
    response = client.get("/forecast", params={"country": sample["node_id"], "month": sample["month"]})
    assert response.status_code == 200, response.text
    body = response.json()
    api_diff = abs(body["heads"][sample["head"]]["prediction"] - float(sample["offline_pred"]))
    assert api_diff < TOLERANCE, api_diff
    assert client.get("/forecast", params={"country": "CHN", "month": "2024-06"}).status_code == 200
    assert client.get("/forecast", params={"country": "CHN", "month": "1999-01"}).status_code == 404
    print(f"HTTP /forecast parity ok (|diff| = {api_diff:.2e}); unknown month -> 404")
    print("PASS serving parity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
