#!/usr/bin/env python3
"""Graph models on the larger panels: seed ensembles next to the tabular references.

Settings (runs made by run_graph_extended.sh / graph_basket_runs.py):
  41 countries, 2021-2024, paper split (test 2024): semiconductor, consumer, both
  41 countries, 2019-2025, train to 2022-12, validation 2023, test 2024-2025: semiconductor, consumer, both
For each: the five-seed ensemble (mean of the seeds' test scores) of GCN, TGN and
TGN without memory; persistence, the seven-feature ridge, trade proxies only and
own lags AR(1,12) from the tabular_analyses harness on the same test rows; and the
wild-cluster p-values of own lags against each graph model and of each graph
model against the seven-feature ridge, clustered by country and by month. "Both" pools the two baskets' test rows; the graph models are
trained per basket (the trainer takes one graph), the tabular models per basket too.

Only seeds whose run is complete are used; the number of seeds is reported.
Writes results/benchmark/extended_graph_summary.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tabular_analyses as tab  # noqa: E402
from dm_tests import wild_cluster_boot  # noqa: E402
from graph_basket_summary import ensemble  # noqa: E402

BASE = Path(__file__).resolve().parent
REV = BASE.parent / "results" / "benchmark"
GRAPH = ("gcn", "tgn", "tgn_no_memory")
SETTINGS = {
    "41 countries, 2021-24": ("processed_v3", "extended_2021_2024", ("2021-12", "2022-12"), ("2024-01", "2024-12")),
    "41 countries, 2019-25": ("processed_v4", "extended_2019_2025", ("2019-12", "2022-12"), ("2024-01", "2025-12")),
}


def done(folders: list[Path]) -> list[Path]:
    return [f for f in folders if (f / "predictions" / "tgn_no_memory.csv").exists()]


def table(panel: str, out: str, basket: str, target: str, train, test) -> tuple[pd.DataFrame, int]:
    """Test rows with y, graph ensembles and tabular predictions for one basket."""
    sub = "graph_t1" if target == "T1" else "benchmark_default"
    folders = done(sorted((REV / out / basket / sub).glob("seed_*" if target == "T1" else "seed_*/regression")))
    wide = ensemble(folders)
    if wide.empty:
        return wide, 0
    wide = wide.dropna(subset=["y", *GRAPH]).reset_index()
    tab.NODES = BASE / f"data/four_year_2021_2024/{panel}/{basket}/nodes_monthly.csv"
    f = tab.load()
    ok = f[target].notna() & np.isfinite(f[target])
    tr = f["month"].between(*train) & ok
    key = pd.MultiIndex.from_frame(wide[["node_id", "month"]])
    te = pd.MultiIndex.from_frame(f[["host_country_id", "month"]]).isin(key) & ok
    lag1, lag12 = f"{target}_lag1", f"{target}_lag12"
    t = f.loc[te, ["host_country_id", "month", target, lag1]].rename(
        columns={"host_country_id": "node_id", target: "y_tab", lag1: "persistence"})
    t["persistence"] = t["persistence"].fillna(0.0)
    for name, feats in (("all_seven", tab.ALL7), ("trade_only", tab.TRADE), ("ar1_12", [lag1, lag12])):
        t[name] = tab.fit_predict(f, feats, target, tr, te)
    m = wide.merge(t, on=["node_id", "month"], how="inner")
    gap = float(np.abs(m["y"] - m["y_tab"]).max())
    if gap > 1e-4:
        print(f"WARNING {out}/{basket} {target}: trainer and tabular targets differ by up to {gap:.4g}")
    m["basket"] = basket
    return m, len(folders)


def row(name: str, basket: str, target: str, m: pd.DataFrame, seeds) -> dict:
    y = m["y"].to_numpy(float)
    r = {"setting": name, "basket": basket, "target": target, "n": len(m), "countries": m["node_id"].nunique(), "seeds": seeds}
    for c in ("persistence", "all_seven", "trade_only", "ar1_12", "ridge_regression", *GRAPH):
        if c in m:
            r[c] = r2_score(y, m[c].to_numpy(float))
    for g in GRAPH:
        d = (y - m["ar1_12"].to_numpy(float)) ** 2 - (y - m[g].to_numpy(float)) ** 2
        r[f"p_country AR vs {g}"] = wild_cluster_boot(d, m["node_id"].to_numpy())[0]
        r[f"p_month AR vs {g}"] = wild_cluster_boot(d, m["month"].to_numpy())[0]
        d = (y - m[g].to_numpy(float)) ** 2 - (y - m["all_seven"].to_numpy(float)) ** 2
        r[f"p_country {g} vs all_seven"] = wild_cluster_boot(d, m["node_id"].to_numpy())[0]
        r[f"p_month {g} vs all_seven"] = wild_cluster_boot(d, m["month"].to_numpy())[0]
    return r


def classification(out: str, basket: str) -> dict | None:
    folders = done(sorted((REV / out / basket / "benchmark_default").glob("seed_*/classification")))
    w = ensemble(folders)
    if w.empty:
        return None
    r = {"setting": "41 countries, 2021-24", "basket": basket, "target": "T0 tau=0.20 PR-AUC", "seeds": len(folders),
         "n": int(w["y"].notna().sum()), "positives": int(w["y"].sum())}
    for c in ("logistic_regression", *GRAPH):
        if c in w:
            s = w[["y", c]].dropna()
            r[c] = average_precision_score(s["y"].astype(int), s[c])
    return r


def main() -> int:
    rows = []
    for name, (panel, out, train, test) in SETTINGS.items():
        for target in ("T1", "T0"):
            parts, seeds = [], []
            for basket in ("semiconductor", "consumer"):
                m, k = table(panel, out, basket, target, train, test)
                if m.empty:
                    continue
                rows.append(row(name, basket, target, m, k))
                parts.append(m)
                seeds.append(k)
            if len(parts) == 2:
                rows.append(row(name, "both", target, pd.concat(parts, ignore_index=True), min(seeds)))
    for basket in ("semiconductor", "consumer"):
        c = classification("extended_2021_2024", basket)
        if c:
            rows.append(c)
    out = pd.DataFrame(rows)
    out.to_csv(REV / "extended_graph_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.round(3).T.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
