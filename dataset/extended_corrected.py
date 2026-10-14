#!/usr/bin/env python3
"""Correction for unreported months, extended panels: the overlapping-target (T0) numbers quoted in the
extended-panel subsection, recomputed without unreported-month artefacts (rows whose current
or next month Comtrade has not published). T1 numbers need no correction (T1 is undefined there).
Writes results/benchmark/corrected/corrected_extended_T0.csv and corrected_extended_graph_T0.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

import graph_extended_summary as gs  # noqa: E402
import extended_rolling_blocks as pda  # noqa: E402
import unreported_months as rm  # noqa: E402
import tabular_analyses as tab  # noqa: E402
from dm_tests import wild_cluster_boot  # noqa: E402

BASE = Path(__file__).resolve().parent
OUT = BASE.parent / "results" / "benchmark" / "corrected"
MYANMAR = "country_104"


def frame(panel: str, basket: str, clean: bool) -> pd.DataFrame:
    tab.NODES = BASE / f"data/four_year_2021_2024/{panel}/{basket}/nodes_monthly.csv"
    f = rm.flags(tab.load())
    if clean:
        f.loc[f["artefact"], ["T0", "label"]] = np.nan
    return f


def main() -> int:
    rows = []
    for panel, basket in (("processed", "semiconductor"), ("processed_v3", "semiconductor"), ("processed_v3", "consumer"),
                          ("processed_v4", "semiconductor"), ("processed_v4", "consumer")):
        for clean in (False, True):
            if panel == "processed" and basket != "semiconductor":
                continue
            f = frame(panel, basket, clean) if panel != "processed" else None
            if panel == "processed":
                tab.NODES = BASE / "data/four_year_2021_2024/processed/nodes_monthly.csv"
                f = rm.flags(tab.load())
                if clean:
                    f.loc[f["artefact"], ["T0", "label"]] = np.nan
            v = f["T0"].dropna()
            scope = "corrected" if clean else "as published"
            rows.append({"panel": panel, "basket": basket, "scope": scope, "evaluation": "all valid rows",
                         "n": len(v), "excess_kurtosis": kurtosis(v), "artefact_rows": int((f["artefact"] & tab.load()["T0"].notna().to_numpy()).sum()) if False else None})
            if panel == "processed_v3":
                for excl in (False, True):
                    g = f[f["host_country_id"] != MYANMAR].reset_index(drop=True) if excl else f
                    ok = g["T0"].notna()
                    tr, te = tab.months_between(g, tab.TRAIN) & ok, tab.months_between(g, tab.TEST) & ok
                    y = g.loc[te, "T0"]
                    rows.append({"panel": panel, "basket": basket, "scope": scope, "evaluation": "test 2024" + (" excl. Myanmar" if excl else ""),
                                 "n": int(te.sum()), "persistence": r2_score(y, g.loc[te, "T0_lag1"].fillna(0)),
                                 "all7": r2_score(y, tab.fit_predict(g, tab.ALL7, "T0", tr, te)),
                                 "exogenous": r2_score(y, tab.fit_predict(g, tab.EXO, "T0", tr, te)),
                                 "own_lag": r2_score(y, tab.fit_predict(g, ["T0_lag1"], "T0", tr, te))})
            if panel == "processed_v4":
                for excl in (False, True):
                    g = f[f["host_country_id"] != MYANMAR].reset_index(drop=True) if excl else f
                    rec = {"panel": panel, "basket": basket, "scope": scope, "evaluation": "ten blocks" + (" excl. Myanmar" if excl else "")}
                    for name, feats in (("persistence", "persistence"), ("all7", tab.ALL7), ("exogenous", tab.EXO), ("own_lag", ["T0_lag1"])):
                        y, p, _, _ = pda.rolling(g, feats, "T0")
                        rec[name] = r2_score(y, p)
                        rec["n"] = len(y)
                    rows.append(rec)
    out = pd.DataFrame(rows).drop(columns=["artefact_rows"])
    out.to_csv(OUT / "corrected_extended_T0.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.round(3).to_string(index=False))

    # graph models on the 41-country 2021-24 panel, T0 (regression), with and without artefacts
    panel, out_dir, train, test = gs.SETTINGS["41 countries, 2021-24"]
    grows = []
    for basket in ("semiconductor", "consumer"):
        m, k = gs.table(panel, out_dir, basket, "T0", train, test)
        m["artefact"] = rm.flags(frame(panel, basket, False)).set_index(["host_country_id", "month"])["artefact"].reindex(
            pd.MultiIndex.from_frame(m[["node_id", "month"]])).to_numpy()
        for scope, q in (("as published", m), ("corrected", m[~m["artefact"].astype(bool)])):
            y = q["y"].to_numpy(float)
            rec = {"basket": basket, "scope": scope, "n": len(q), "seeds": k, "ar1_12": r2_score(y, q["ar1_12"]),
                   "all_seven": r2_score(y, q["all_seven"]), "persistence": r2_score(y, q["persistence"])}
            for g in gs.GRAPH:
                rec[g] = r2_score(y, q[g])
                d = (y - q["ar1_12"].to_numpy(float)) ** 2 - (y - q[g].to_numpy(float)) ** 2
                rec[f"p_country AR vs {g}"] = wild_cluster_boot(d, q["node_id"].to_numpy())[0]
                rec[f"p_month AR vs {g}"] = wild_cluster_boot(d, q["month"].to_numpy())[0]
            grows.append(rec)
    gout = pd.DataFrame(grows)
    gout.to_csv(OUT / "corrected_extended_graph_T0.csv", index=False)
    print(gout.round(3).T.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
