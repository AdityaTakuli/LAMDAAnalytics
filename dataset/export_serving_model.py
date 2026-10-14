#!/usr/bin/env python3
"""Export the evaluated offline models for serving.

The reviewer's train-serve objection is that the paper evaluates one pipeline
(fused public series -> trained model) and deploys another (LLM features ->
fixed linear blend). This script writes the exact fitted models that the paper
scores, plus the published-series feature table they read, into
backend/models/serving/, where backend/models/parity_model.py serves them from
GET /forecast. The served arithmetic is the training arithmetic: same feature
list, same raw-NaN fill, same inventory sign flip, same train-only mean/scale,
same coefficients.

Heads
  contraction_ridge7  ridge(alpha=1) on the seven fused features, target T0
                      (benchmark headline model; test R^2 0.338 as first reported, -0.242 on rows with an observable target)
  logchange_ar1_12    ridge(alpha=1) on the target's own lags 1 and 12, target
                      T1 = log(V_{T+1}/V_T) (non-overlapping; test R^2 0.215)

Also writes results/benchmark/serving_parity_offline.csv: the offline
(sklearn) test-2024 predictions that backend/test_serving_parity.py must match.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tabular_analyses import ALL7, PROTECTIVE, TEST, TRAIN, load  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SERVE = ROOT / "backend" / "models" / "serving"
OUT = ROOT / "results" / "benchmark"

HEADS = {
    "contraction_ridge7": {"target": "T0", "features": ALL7,
                           "target_definition": "(V_{T+1} - median(V_{T-11..T})) / median(V_{T-11..T})"},
    "logchange_ar1_12": {"target": "T1", "features": ["T1_lag1", "T1_lag12"],
                         "target_definition": "log(V_{T+1} / V_T)"},
}


def main() -> int:
    SERVE.mkdir(parents=True, exist_ok=True)
    frame = load()
    # Unreported reporter-months are stored as zero inbound value; a row whose current or next
    # month is unreported has no observable target (round-2 correction, unreported_months.py).
    gap = frame["inbound_flow_usd"] <= 0
    gap_next = frame.groupby("host_country_id")["inbound_flow_usd"].shift(-1) <= 0
    frame.loc[gap | gap_next, ["T0", "T1"]] = float("nan")
    bundle = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "source": "dataset/export_serving_model.py",
              "panel": "dataset/data/four_year_2021_2024/processed/nodes_monthly.csv",
              "train_months": list(TRAIN), "heads": {}}
    offline = []
    for name, spec in HEADS.items():
        feats, target = spec["features"], spec["target"]
        ok = frame[target].notna()
        tr = frame["month"].between(*TRAIN) & ok
        te = frame["month"].between(*TEST) & ok
        raw = frame[feats].astype(float).fillna(0.0).to_numpy(copy=True)
        flip = [f for f in feats if f == PROTECTIVE]
        for f in flip:
            raw[:, feats.index(f)] *= -1.0
        mean = raw[tr.to_numpy()].mean(axis=0)
        scale = raw[tr.to_numpy()].std(axis=0)
        scale[scale < 1e-8] = 1.0
        x = (raw - mean) / scale
        model = Ridge(alpha=1.0, random_state=7).fit(x[tr.to_numpy()], frame.loc[tr, target].to_numpy(float))
        pred = model.predict(x[te.to_numpy()])
        r2 = r2_score(frame.loc[te, target].to_numpy(float), pred)
        bundle["heads"][name] = {
            "target": target, "target_definition": spec["target_definition"], "features": feats,
            "sign_flipped": flip, "raw_nan_fill": 0.0, "mean": mean.tolist(), "scale": scale.tolist(),
            "coef": model.coef_.tolist(), "intercept": float(model.intercept_), "alpha": 1.0,
            "test_2024_r2": float(r2), "test_rows": int(te.sum()),
        }
        offline.append(pd.DataFrame({"head": name, "node_id": frame.loc[te, "node_id"].to_numpy(),
                                     "month": frame.loc[te, "month"].to_numpy(), "offline_pred": pred}))
        print(f"{name:20s} test R2 {r2:.4f}  rows {int(te.sum())}")

    cols = ["node_id", "month", *ALL7, "T0_lag1", "T1_lag1", "T1_lag12"]
    table = frame[cols].copy()
    (SERVE / "bundle.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    table.to_csv(SERVE / "published_features.csv", index=False)
    pd.concat(offline).to_csv(OUT / "serving_parity_offline.csv", index=False)
    print("wrote", SERVE / "bundle.json", SERVE / "published_features.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
