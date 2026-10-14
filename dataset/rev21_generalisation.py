#!/usr/bin/env python3
"""Paper-21 revision: does the verdict hold on a different product group?

The fused panel's inbound series is HS 8541 + HS 8542 imports from the 18
panel countries (fuse_dataset._standard_trade + selection filter). Here the
same construction is repeated per HS-4 code, so each sub-basket gets its own
inbound series, its own trade proxies (same formulas as the panel), and its
own T0 / T1 targets. The exogenous channels (GDELT, NASA POWER, GSCPI) are the
panel's, unchanged, because they are country-month quantities.

Check: summing the two sub-baskets reproduces the panel's inbound_flow_usd.

Writes results/paper21_revision/generalisation_by_hs.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fuse_dataset as fd  # noqa: E402
from common import load_config  # noqa: E402
from rev21_tabular import (ALL7, BLOCKS, EXO, TEST, TRADE, TRAIN, VALID,  # noqa: E402
                           fit_predict)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "paper21_revision"
PROC = Path(__file__).parent / "data/four_year_2021_2024/processed"


def inbound_long(config) -> pd.DataFrame:
    """Trade rows kept by the fusion filter, with their HS code."""
    raw = pd.read_csv(PROC / "comtrade.csv")
    trade = fd._standard_trade(raw, config)
    selected = fd._selected_country_ids(config)
    trade = trade[trade["source_country_id"].isin(selected) & trade["destination_country_id"].isin(selected)]
    trade["commodity_code"] = raw.loc[trade.index, "commodity_code"].astype(str)
    return trade


def inbound_series(trade: pd.DataFrame, code: str | None) -> pd.DataFrame:
    """fuse_dataset._inbound_monthly on one HS code (or both when code is None)."""
    sub = trade if code is None else trade[trade["commodity_code"] == code]
    return (sub.groupby(["destination_country_id", "month"], as_index=False)["trade_value_usd"].sum()
            .rename(columns={"destination_country_id": "host_country_id", "trade_value_usd": "inbound_value"}))


def exact_proxies(inbound: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    """Same loop as fuse_dataset.build_fused_tables: median of the last three
    months that have an inbound row, current value if none."""
    out = []
    for (host, month) in zip(nodes["host_country_id"], nodes["month"]):
        prior = inbound[inbound["host_country_id"] == host].sort_values("month")
        value = float(prior.loc[prior["month"] == month, "inbound_value"].sum())
        hist = prior.loc[prior["month"] < month, "inbound_value"].tail(3).tolist()
        base = float(np.median(hist)) if hist else value
        ratio = value / base if base > 0 else 1.0
        out.append((host, month, value, float(np.clip(30 * ratio, 5, 90)),
                    float(np.clip(30 * max(0.0, 1 - ratio), 0, 60))))
    return pd.DataFrame(out, columns=["host_country_id", "month", "V", "inventory_days_proxy", "trade_delay_proxy"])


def basket_frame(nodes: pd.DataFrame, inbound: pd.DataFrame) -> pd.DataFrame:
    f = nodes.merge(exact_proxies(inbound, nodes), on=["host_country_id", "month"], how="left")
    f = f.sort_values(["host_country_id", "month"]).reset_index(drop=True)
    g = f.groupby("host_country_id", sort=False)
    fut = g["V"].shift(-1)
    med = g["V"].transform(lambda s: s.rolling(12, min_periods=12).median())
    f["T0"] = ((fut - med) / med.where(med > 0)).where(fut.notna())
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.log(fut / f["V"].where(f["V"] > 0))
    t1[(fut <= 0) | fut.isna()] = np.nan
    f["T1"] = t1
    for t in ("T0", "T1"):
        f[f"{t}_lag1"] = g[t].shift(1)
        f[f"{t}_lag12"] = g[t].shift(12)
    return f


def ladder(f: pd.DataFrame, basket: str) -> list[dict]:
    rows = []
    for target in ("T1", "T0"):
        ok = f[target].notna() & np.isfinite(f[target])
        tr = f["month"].between(*TRAIN) & ok
        te = f["month"].between(*TEST) & ok
        y = f.loc[te, target].to_numpy(float)
        lag1, lag12 = f"{target}_lag1", f"{target}_lag12"
        sets = {"AR(1)": [lag1], "AR(1,12)": [lag1, lag12], "Trade proxies only": TRADE,
                "Exogenous only": EXO, "All seven features": ALL7, "All seven + own lags": [*ALL7, lag1, lag12]}
        rec = {"basket": basket, "target": target, "n_test": len(y),
               "lag1_autocorr": float(f.loc[ok & f[lag1].notna(), [target, lag1]].corr().iloc[0, 1]),
               "Persistence": r2_score(y, f.loc[te, lag1].fillna(0.0)),
               "Train median": r2_score(y, np.full_like(y, np.median(f.loc[tr, target])))}
        for name, feats in sets.items():
            rec[name] = r2_score(y, fit_predict(f, feats, target, tr, te))
        # pooled rolling-origin over 2023-2024
        ys, ps = [], {k: [] for k in sets}
        for start, end in BLOCKS:
            end_train = f"{int(start[:4]) - 1}-12" if start[5:] == "01" else f"{start[:4]}-06"
            trb = f["month"].between(TRAIN[0], end_train) & ok
            teb = f["month"].between(start, end) & ok
            ys.append(f.loc[teb, target].to_numpy(float))
            for name, feats in sets.items():
                ps[name].append(fit_predict(f, feats, target, trb, teb))
        yall = np.concatenate(ys)
        for name in sets:
            rec[f"pooled {name}"] = r2_score(yall, np.concatenate(ps[name]))
        rows.append(rec)
    return rows


def main() -> int:
    config = load_config("config.yaml")
    nodes = pd.read_csv(PROC / "nodes_monthly.csv")
    nodes["month"] = nodes["month"].astype(str)
    trade = inbound_long(config)
    base_nodes = nodes.drop(columns=["inventory_days_proxy", "trade_delay_proxy"])
    rows = []
    for code, name in ((None, "HS 8541+8542 (panel)"), ("8541", "HS 8541 only"), ("8542", "HS 8542 only")):
        f = basket_frame(base_nodes, inbound_series(trade, code))
        if code is None:
            chk = f.merge(nodes[["host_country_id", "month", "inbound_flow_usd", "inventory_days_proxy"]],
                          on=["host_country_id", "month"], suffixes=("", "_panel"))
            print("panel inbound reproduced:", np.isclose(chk["V"], chk["inbound_flow_usd"], atol=1).mean(),
                  " inventory proxy reproduced:", np.isclose(chk["inventory_days_proxy"], chk["inventory_days_proxy_panel"]).mean())
        else:
            print(name, "value share", f["V"].sum() / trade["trade_value_usd"].sum())
        rows += ladder(f, name)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "generalisation_by_hs.csv", index=False)
    print(out.round(3).T.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
