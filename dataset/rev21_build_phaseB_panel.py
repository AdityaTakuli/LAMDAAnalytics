#!/usr/bin/env python3
"""Phase B: build the 42-country panel for both baskets.

Trade
  semiconductor  processed/comtrade.csv (18 published reporters) + Phase B records, HS 8541/8542
  consumer       processed_v2/comtrade_consumer.csv (18 reporters) + Phase B records, 11 consumer codes
  Phase B reporters are importers only, with the same 20 partners, so the 18
  published series are unchanged.
Exogenous
  processed_v3/gdelt_monthly_features.csv (rev21_gdelt_features.py, verified against the published table)
  processed_v3/weather_daily.csv          (rev21_weather_phaseB.py, published rows copied unchanged)
  processed/gscpi_monthly.csv             (global)
Construction: fuse_dataset.build_fused_tables with config_phaseB.yaml (44 locations), unchanged.

Checks (the script stops if either fails)
  * every published country-month row (18 countries x 48 months) is reproduced
    exactly in each basket: inbound flow, seven features, contraction target;
  * every added country has news volume > 0 in most months (GDELT code match).

Outputs: processed_v3/{semiconductor,consumer}/{nodes,edges}_monthly.csv,
processed_v3/country_universe.csv, processed_v3/combined_nodes_long.csv,
processed_v3/build_summary.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fuse_dataset as fd  # noqa: E402
from common import load_config, read_table  # noqa: E402
from ingest_comtrade_bilateral import _standardize_bilateral  # noqa: E402
from rev21_build_consumer_panel import same  # noqa: E402
from rev21_download_phaseB import NEW_REPORTERS  # noqa: E402

BASE = Path(__file__).resolve().parent
ROOT = BASE / "data/four_year_2021_2024"
V2, V3 = ROOT / "processed_v2", ROOT / "processed_v3"
CACHE = ROOT / "cache/comtrade/phaseB"
SEMI = {"8541", "8542"}
# GDELT leaves the actor country code blank for Romanian actors (named "ROMANIA"
# with no code), so under the published news rule Romania would always show
# zero news. It is dropped rather than given a different rule.
EXCLUDED = {"642": "Romania: no GDELT actor country code"}
ADDED = {k: v for k, v in NEW_REPORTERS.items() if k not in EXCLUDED}


def phaseb_records() -> pd.DataFrame:
    frames = []
    for path in sorted(CACHE.glob("reporter_*.json")):
        payload = json.loads(path.read_text())
        raw = pd.DataFrame(payload["response"]["data"])
        if raw.empty:
            continue
        for period, part in raw.groupby(raw["period"].astype(str)):
            frames.append(_standardize_bilateral(part.reset_index(drop=True), period, payload["downloaded_at"]))
    frame = pd.concat(frames, ignore_index=True)
    frame = frame[~frame["reporter_code"].astype(str).isin(EXCLUDED)]
    keys = ["month", "reporter_code", "partner_code", "commodity_code", "flow_code"]
    return frame.drop_duplicates(keys).reset_index(drop=True)


def universe(config) -> pd.DataFrame:
    pub = read_table(ROOT / "processed/country_universe.csv")
    rank = pd.read_csv(ROOT / "cache/comtrade/india_partner_ranking/india_partner_ranking.csv")
    rank["partner"] = rank["partner"].astype(str)
    iso = {str(l["comtrade_code"]): l["iso3"] for l in config["sources"]["weather"]["locations"]}
    add = []
    for code, name in ADDED.items():
        value = float(rank.loc[rank["partner"] == code, "value"].sum())
        add.append({"country_name": name, "comtrade_code": int(code), "iso3": iso[code],
                    "trade_value_usd": value, "selection_reason": "phase_b_partner_with_monthly_reporter_data"})
    out = pd.concat([pub, pd.DataFrame(add)], ignore_index=True)
    out["rank"] = range(1, len(out) + 1)
    return out


def fused(config, trade):
    return fd.build_fused_tables(
        config, trade, gdelt=pd.DataFrame(),
        gdelt_monthly=read_table(V3 / "gdelt_monthly_features.csv"),
        weather=read_table(V3 / "weather_daily.csv"),
        gscpi=read_table(ROOT / "processed/gscpi_monthly.csv"))


def main() -> int:
    config = load_config("config_phaseB.yaml")
    V3.mkdir(parents=True, exist_ok=True)
    universe(config).to_csv(V3 / "country_universe.csv", index=False)

    new = phaseb_records()
    new["commodity_code"] = new["commodity_code"].astype(str)
    trades = {
        "semiconductor": pd.concat([read_table(ROOT / "processed/comtrade.csv"), new[new["commodity_code"].isin(SEMI)]], ignore_index=True),
        "consumer": pd.concat([read_table(V2 / "comtrade_consumer.csv"), new[~new["commodity_code"].isin(SEMI)]], ignore_index=True),
    }
    published = {"semiconductor": read_table(ROOT / "processed/nodes_monthly.csv"),
                 "consumer": read_table(V2 / "consumer/nodes_monthly.csv")}
    check_cols = ["inbound_flow_usd", *fd.FEATURES, "contraction"]
    summary, longs = {"phase_b_records": int(len(new)), "excluded": EXCLUDED}, []
    for basket, trade in trades.items():
        nodes, edges = fused(config, trade)
        nodes["month"] = nodes["month"].astype(str)
        pub = published[basket]
        pub["month"] = pub["month"].astype(str)
        old = nodes[nodes["host_country_id"].isin(set(pub["host_country_id"]))]
        match = same(old, pub, check_cols)
        print(f"{basket}: {nodes['host_country_id'].nunique()} countries, {len(nodes)} rows; "
              f"published 18-country rows reproduced: {match:.1%}")
        if match < 1.0:
            raise SystemExit(f"{basket}: published rows not reproduced; stopping")
        d = V3 / basket
        d.mkdir(exist_ok=True)
        nodes.to_csv(d / "nodes_monthly.csv", index=False)
        edges.to_csv(d / "edges_monthly.csv", index=False)
        added = nodes[~nodes["host_country_id"].isin(set(pub["host_country_id"]))]
        summary[basket] = {
            "countries": int(nodes["host_country_id"].nunique()), "rows": int(len(nodes)),
            "valid_targets": int(nodes["target_valid"].astype(bool).sum()),
            "valid_targets_added_countries": int(added["target_valid"].astype(bool).sum()),
            "edges": int(len(edges)), "published_rows_reproduced": match,
            "added_countries_zero_inbound_months": added.groupby("host_country_id")["inbound_flow_usd"].apply(lambda s: int((s <= 0).sum())).to_dict(),
        }
        longs.append(nodes.assign(basket=basket))
    news = longs[0].groupby("host_country_id")["news_vol_7d"].apply(lambda s: float((s > 0).mean()))
    summary["share_of_months_with_news"] = news.round(3).to_dict()
    weak = news[news < 0.5]
    if len(weak):
        print("WARNING: countries with news in fewer than half the months:", weak.to_dict())
    long = pd.concat(longs, ignore_index=True)
    long["series_id"] = long["host_country_id"] + "|" + long["basket"]
    long.to_csv(V3 / "combined_nodes_long.csv", index=False)
    (V3 / "build_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "share_of_months_with_news"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
