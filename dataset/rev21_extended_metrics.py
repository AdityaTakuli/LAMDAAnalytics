#!/usr/bin/env python3
"""Extended panel: does the model ordering depend on the metric or on the fixed settings?

A. Other metrics. The extended tables report R^2 only. Here: RMSE, MAE and the share of
   months whose direction of change is called correctly, for
     * the ten half-year blocks of the seven-year panel (tabular models), and
     * the seven-year single split with the graph ensembles (test 2024-01..2025-11),
   plus wild-cluster tests on the absolute-error loss (own lags vs seven features; own lags
   vs each graph model), clustered by country and by month.
B. Settings. Ridge strength alpha in {0.01 .. 1000} and the choice of own lags
   ({1}, {1,12}, {1,2,3}, {1..12}), ten half-year blocks, both baskets.

Same harness as rev21_phaseD_analyses.py (train-only scaling, NaN -> 0, expanding origin).
Writes results/paper21_revision/phaseD/{other_metrics.csv, other_metrics_tests.csv, sensitivity_alpha_lags.csv}.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rev21_extended_graph_summary as gs  # noqa: E402
import rev21_phaseD_analyses as pda  # noqa: E402
import rev21_tabular as tab  # noqa: E402
from dm_tests import wild_cluster_boot  # noqa: E402

OUT = pda.OUT
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


def rolling(f: pd.DataFrame, feats, alpha: float = 1.0):
    """Ten expanding-origin blocks; returns y, prediction, country, month for the pooled test rows."""
    ok = f["T1"].notna() & np.isfinite(f["T1"])
    ys, ps, cl, mo = [], [], [], []
    for start, end in pda.BLOCKS:
        tr = f["month"].between(pda.FIRST, pda.prev_month(start)) & ok
        te = f["month"].between(start, end) & ok
        if te.sum() == 0:
            continue
        if feats == "persistence":
            p = f.loc[te, "T1_lag1"].fillna(0.0).to_numpy(float)
        else:
            x = tab.design(f, feats, tr)
            p = Ridge(alpha=alpha, random_state=7).fit(x[tr.to_numpy()], f.loc[tr, "T1"].to_numpy(float)).predict(x[te.to_numpy()])
        ys.append(f.loc[te, "T1"].to_numpy(float))
        ps.append(p)
        cl.append(f.loc[te, "host_country_id"].to_numpy())
        mo.append(f.loc[te, "month"].to_numpy())
    return tuple(np.concatenate(a) for a in (ys, ps, cl, mo))


def scores(y, p) -> dict:
    return {"r2": r2_score(y, p), "rmse": float(np.sqrt(np.mean((y - p) ** 2))), "mae": float(np.mean(np.abs(y - p))),
            "direction_acc": float(100 * np.mean(np.sign(p) == np.sign(y)))}


def abs_test(y, a, b, country, month) -> dict:
    d = np.abs(y - a) - np.abs(y - b)
    return {"mae_diff": float(d.mean()), "p_country": wild_cluster_boot(d, country)[0], "p_month": wild_cluster_boot(d, month)[0]}


def main() -> int:
    fr = pda.frames()
    for f in fr.values():
        g = f.groupby("host_country_id", sort=False)["T1"]
        for k in range(2, 12):
            f[f"T1_lag{k}"] = g.shift(k)
    data = {"semiconductor": fr["semiconductor"], "consumer": fr["consumer"], "both baskets": pd.concat(fr.values(), ignore_index=True)}
    rows, tests, sens = [], [], []

    # A1. ten blocks, tabular models
    for dname, f in data.items():
        extra = ["is_consumer"] if dname == "both baskets" else []
        sets = {"Persistence": "persistence", "Constant (zero change)": None, "Exogenous only": [*tab.EXO, *extra],
                "All seven features": [*tab.ALL7, *extra], "Trade proxies only": [*tab.TRADE, *extra],
                "Own lags 1, 12 (AR(1,12))": ["T1_lag1", "T1_lag12", *extra],
                "AR(1,12) + exogenous": ["T1_lag1", "T1_lag12", *tab.EXO, *extra]}
        preds = {}
        for name, feats in sets.items():
            if feats is None:
                y, _, cl, mo = rolling(f, "persistence")
                p = np.zeros_like(y)
            else:
                y, p, cl, mo = rolling(f, feats)
            preds[name] = p
            rows.append({"evaluation": "ten half-year blocks", "data": dname, "model": name, "n": len(y), **scores(y, p)})
        for a, b in (("Own lags 1, 12 (AR(1,12))", "All seven features"), ("AR(1,12) + exogenous", "Own lags 1, 12 (AR(1,12))")):
            tests.append({"evaluation": "ten half-year blocks", "data": dname, "A": a, "B": b, **abs_test(y, preds[a], preds[b], cl, mo)})
        print(f"metrics done: ten blocks / {dname}", flush=True)

    # A2. seven-year single split with the graph ensembles
    panel, out, train, test = gs.SETTINGS["41 countries, 2019-25"]
    parts = {b: gs.table(panel, out, b, "T1", train, test)[0] for b in ("semiconductor", "consumer")}
    parts["both baskets"] = pd.concat(parts.values(), ignore_index=True)
    names = {"persistence": "Persistence", "all_seven": "All seven features", "trade_only": "Trade proxies only",
             "ar1_12": "Own lags 1, 12 (AR(1,12))", "gcn": "GCN", "tgn": "TGN", "tgn_no_memory": "TGN, no memory"}
    for dname, m in parts.items():
        y = m["y"].to_numpy(float)
        for col, name in names.items():
            rows.append({"evaluation": "single split, test 2024-25", "data": dname, "model": name, "n": len(m), **scores(y, m[col].to_numpy(float))})
        for g in gs.GRAPH:
            tests.append({"evaluation": "single split, test 2024-25", "data": dname, "A": "Own lags 1, 12 (AR(1,12))", "B": names[g],
                          **abs_test(y, m["ar1_12"].to_numpy(float), m[g].to_numpy(float), m["node_id"].to_numpy(), m["month"].to_numpy())})
        print(f"metrics done: single split / {dname}", flush=True)

    # B. ridge strength and lag choice, ten blocks
    for dname, f in data.items():
        extra = ["is_consumer"] if dname == "both baskets" else []
        lagsets = {"Own lag 1": ["T1_lag1"], "Own lags 1, 12": ["T1_lag1", "T1_lag12"], "Own lags 1, 2, 3": ["T1_lag1", "T1_lag2", "T1_lag3"],
                   "Own lags 1 to 12": [f"T1_lag{k}" for k in range(1, 13)]}
        sets = {**{k: [*v, *extra] for k, v in lagsets.items()}, "All seven features": [*tab.ALL7, *extra],
                "Trade proxies only": [*tab.TRADE, *extra], "Exogenous only": [*tab.EXO, *extra],
                "AR(1,12) + exogenous": ["T1_lag1", "T1_lag12", *tab.EXO, *extra],
                "All seven + own lags 1, 12": [*tab.ALL7, "T1_lag1", "T1_lag12", *extra]}
        for name, feats in sets.items():
            for alpha in ALPHAS:
                y, p, _, _ = rolling(f, feats, alpha)
                sens.append({"data": dname, "model": name, "alpha": alpha, "r2": r2_score(y, p)})
        print(f"sensitivity done: {dname}", flush=True)

    res, tst, sn = pd.DataFrame(rows), pd.DataFrame(tests), pd.DataFrame(sens)
    res.to_csv(OUT / "other_metrics.csv", index=False)
    tst.to_csv(OUT / "other_metrics_tests.csv", index=False)
    sn.to_csv(OUT / "sensitivity_alpha_lags.csv", index=False)
    pd.set_option("display.width", 250, "display.max_rows", 300)
    print(res.round(4).to_string(index=False))
    print(tst.round(4).to_string(index=False))
    print(sn.pivot_table(index=["data", "model"], columns="alpha", values="r2").round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
