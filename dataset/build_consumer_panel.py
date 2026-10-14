#!/usr/bin/env python3
"""Consumer basket: build the consumer-goods panel and combine it with the semiconductor panel.

Inputs
  cache/comtrade/consumer/*.json        (download_consumer_basket.py)
  processed/comtrade.csv                (semiconductor records, unchanged)
  processed/{gdelt_monthly_features,weather_daily,gscpi_monthly}.csv  (exogenous, unchanged)

Construction is fuse_dataset.build_fused_tables, unchanged, so the consumer
panel follows the same rules as the published one: flows kept only when both
partner and reporter are panel countries, inbound summed per reporter-month,
the same trade proxies, the same causal exogenous features, the same labels.

Check: rebuilding the semiconductor panel with the same call must reproduce
processed/nodes_monthly.csv; the script stops if it does not.

Outputs (processed/ is never written)
  processed_v2/comtrade_consumer.csv
  processed_v2/consumer/{nodes_monthly,edges_monthly}.csv     same schema as processed/
  processed_v2/semiconductor/{nodes_monthly,edges_monthly}.csv  copies of processed/
  processed_v2/combined_nodes_long.csv   country x basket x month, with a `basket` column
  processed_v2/build_summary.json
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fuse_dataset as fd  # noqa: E402
from common import load_config, read_table  # noqa: E402
from ingest_comtrade_bilateral import _standardize_bilateral  # noqa: E402

BASE = Path(__file__).resolve().parent
ROOT = BASE / "data/four_year_2021_2024"
PROC = ROOT / "processed"
OUT = ROOT / "processed_v2"
CACHE = ROOT / "cache/comtrade/consumer"
FEATURES = fd.FEATURES


def consumer_records() -> pd.DataFrame:
    frames = []
    files = sorted(CACHE.glob("reporter_*.json"))
    if not files:
        raise SystemExit("no consumer cache files; run download_consumer_basket.py first")
    for path in files:
        payload = json.loads(path.read_text())
        raw = pd.DataFrame(payload["response"]["data"])
        if raw.empty:
            continue
        for period, part in raw.groupby(raw["period"].astype(str)):
            frames.append(_standardize_bilateral(part.reset_index(drop=True), period, payload["downloaded_at"]))
    frame = pd.concat(frames, ignore_index=True)
    keys = ["month", "reporter_code", "partner_code", "commodity_code", "flow_code"]
    dupes = int(frame.duplicated(keys).sum())
    frame = frame.drop_duplicates(keys).reset_index(drop=True)
    return frame, len(files), dupes


def fused(config, trade: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    gdelt_monthly = fd._empty_or_read(PROC / "gdelt_monthly_features.csv", [])
    weather = fd._empty_or_read(PROC / "weather_daily.csv", [])
    gscpi = fd._empty_or_read(PROC / "gscpi_monthly.csv", [])
    return fd.build_fused_tables(config, trade, gdelt=pd.DataFrame(), gdelt_monthly=gdelt_monthly,
                                 weather=weather, gscpi=gscpi)


def same(a: pd.DataFrame, b: pd.DataFrame, cols: list[str]) -> float:
    key = ["host_country_id", "month"]
    m = a[key + cols].merge(b[key + cols], on=key, suffixes=("_a", "_b"))
    ok = np.ones(len(m), dtype=bool)
    for c in cols:
        x, y = pd.to_numeric(m[f"{c}_a"], errors="coerce"), pd.to_numeric(m[f"{c}_b"], errors="coerce")
        ok &= np.isclose(x.fillna(-9e9), y.fillna(-9e9), rtol=1e-9, atol=1e-6)
    return float(ok.mean()) if len(m) == len(a) == len(b) else 0.0


def main() -> int:
    config = load_config("config.yaml")
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. the build path must reproduce the published semiconductor panel
    semi_trade = read_table(PROC / "comtrade.csv")
    semi_nodes, semi_edges = fused(config, semi_trade)
    published = read_table(PROC / "nodes_monthly.csv")
    for f in (semi_nodes, published):
        f["month"] = f["month"].astype(str)
    check_cols = ["inbound_flow_usd", *FEATURES, "contraction"]
    match = same(semi_nodes, published, check_cols)
    print(f"semiconductor rebuild matches published panel on {match:.1%} of rows")
    if match < 1.0:
        raise SystemExit("rebuild does not reproduce processed/nodes_monthly.csv; stopping")

    # 2. consumer panel through the identical call
    cons_trade, n_files, dupes = consumer_records()
    cons_trade.to_csv(OUT / "comtrade_consumer.csv", index=False)
    cons_nodes, cons_edges = fused(config, cons_trade)
    for name, nodes, edges in (("consumer", cons_nodes, cons_edges), ("semiconductor", published, None)):
        d = OUT / name
        d.mkdir(exist_ok=True)
        if edges is None:
            shutil.copy2(PROC / "nodes_monthly.csv", d / "nodes_monthly.csv")
            shutil.copy2(PROC / "edges_monthly.csv", d / "edges_monthly.csv")
        else:
            nodes.to_csv(d / "nodes_monthly.csv", index=False)
            edges.to_csv(d / "edges_monthly.csv", index=False)

    # exogenous features must be identical across baskets (they are country-month)
    exo = ["news_vol_7d", "neg_tone_frac_3d", "strike_flag_7d", "weather_anomaly_7d", "global_risk"]
    exo_match = same(cons_nodes, published, exo)
    print(f"exogenous features identical across baskets on {exo_match:.1%} of rows")

    # 3. combined long panel
    long = pd.concat([published.assign(basket="semiconductor"), cons_nodes.assign(basket="consumer")],
                     ignore_index=True)
    long["series_id"] = long["host_country_id"] + "|" + long["basket"]
    long.to_csv(OUT / "combined_nodes_long.csv", index=False)

    cons_value = cons_trade["trade_value_usd"].sum()
    summary = {
        "cache_files": n_files, "consumer_records": int(len(cons_trade)), "duplicate_records_dropped": dupes,
        "consumer_hs_codes": sorted(cons_trade["commodity_code"].astype(str).unique().tolist()),
        "consumer_reporters": int(cons_trade["reporter_code"].nunique()),
        "consumer_months": int(cons_trade["month"].nunique()),
        "consumer_value_share_by_code": (cons_trade.groupby("commodity_code")["trade_value_usd"].sum() / cons_value).round(4).to_dict(),
        "semiconductor_rebuild_match": match, "exogenous_match": exo_match,
        "rows": {"semiconductor": int(len(published)), "consumer": int(len(cons_nodes)), "combined": int(len(long))},
        "valid_targets": {"semiconductor": int(published["target_valid"].astype(bool).sum()),
                          "consumer": int(cons_nodes["target_valid"].astype(bool).sum())},
        "zero_inbound_rows_consumer": int((cons_nodes["inbound_flow_usd"] <= 0).sum()),
    }
    (OUT / "build_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
