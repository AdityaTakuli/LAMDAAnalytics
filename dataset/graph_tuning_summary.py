#!/usr/bin/env python3
"""Graph hyperparameter sensitivity on the seven-year panel (runs by run_graph_tuning.sh).

For each basket, architecture and variant (graph_basket_runs.TUNE; "base" is the paper's
setting, taken from the graph_t1 runs with the same three seeds): three-seed ensemble R^2 on
the validation year 2023 and on the test months 2024-01..2025-11. Two summaries per
architecture:
  selected  the variant with the best validation R^2 (what a tuned pipeline would report)
  oracle    the variant with the best TEST R^2 (not a valid estimate; the most favourable
            case for the graph models)
each compared with own lags AR(1,12) on the same test rows (wild-cluster p by country and by
month). "both" pools the two baskets, each with its own selected (or oracle) variant.
Only variants whose three seeds are complete are used.
Writes results/benchmark/extended_2019_2025/graph_tune_{variants,summary}.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tabular_analyses as tab  # noqa: E402
from dm_tests import wild_cluster_boot  # noqa: E402
from graph_basket_runs import TUNE  # noqa: E402

BASE = Path(__file__).resolve().parent
REV = BASE.parent / "results" / "benchmark" / "extended_2019_2025"
GRAPH = ("gcn", "tgn", "tgn_no_memory")
SEEDS = (7, 13, 21)
TRAIN, TEST = ("2019-12", "2022-12"), ("2024-01", "2025-12")


def folders(basket: str, variant: str) -> list[Path]:
    root = REV / basket / ("graph_t1" if variant == "base" else f"graph_tune/{variant}")
    out = [root / f"seed_{s}" for s in SEEDS]
    return out if all((f / "predictions" / "tgn_no_memory.csv").exists() for f in out) else []


def ensemble(fs: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(f / "predictions" / f"{g}.csv") for f in fs for g in GRAPH]
    allp = pd.concat(frames)
    allp = allp[allp["split"].isin(["validation", "test"])]
    allp["month"] = allp["month"].astype(str)
    ens = allp.groupby(["model", "split", "node_id", "month"], as_index=False).agg(y=("target", "first"), score=("score", "mean"))
    wide = ens.pivot_table(index=["split", "node_id", "month"], columns="model", values="score").reset_index()
    return wide.merge(ens.drop_duplicates(["split", "node_id", "month"])[["split", "node_id", "month", "y"]], on=["split", "node_id", "month"])


def ar_frame(basket: str) -> pd.DataFrame:
    tab.NODES = BASE / f"data/four_year_2021_2024/processed_v4/{basket}/nodes_monthly.csv"
    f = tab.load()
    ok = f["T1"].notna() & np.isfinite(f["T1"])
    tr, te = f["month"].between(*TRAIN) & ok, f["month"].between(*TEST) & ok
    out = f.loc[te, ["host_country_id", "month"]].rename(columns={"host_country_id": "node_id"})
    out["ar1_12"] = tab.fit_predict(f, ["T1_lag1", "T1_lag12"], "T1", tr, te)
    return out


def main() -> int:
    variants, tests, chosen = [], [], {}
    for basket in ("semiconductor", "consumer"):
        ar = ar_frame(basket)
        ens = {v: ensemble(fs) for v in TUNE if (fs := folders(basket, v))}
        for v, w in ens.items():
            for g in GRAPH:
                val, te = w[w["split"] == "validation"].dropna(subset=["y", g]), w[w["split"] == "test"].dropna(subset=["y", g])
                variants.append({"basket": basket, "model": g, "variant": v, "val_r2": r2_score(val["y"], val[g]),
                                 "test_r2": r2_score(te["y"], te[g]), "test_rows": len(te)})
        table = pd.DataFrame([r for r in variants if r["basket"] == basket])
        for g in GRAPH:
            t = table[table["model"] == g]
            for rule, pick in (("selected", t.loc[t["val_r2"].idxmax(), "variant"]), ("oracle", t.loc[t["test_r2"].idxmax(), "variant"])):
                w = ens[pick]
                m = w[w["split"] == "test"].dropna(subset=["y", g]).merge(ar, on=["node_id", "month"]).assign(basket=basket)
                chosen[(rule, g, basket)] = m[["node_id", "month", "basket", "y", "ar1_12"]].assign(pred=m[g].to_numpy())
                tests.append({"rule": rule, "basket": basket, "model": g, "variant": pick, "variants_available": len(t)})
    rows = []
    for rec in tests:
        m = chosen[(rec["rule"], rec["model"], rec["basket"])]
        rows.append({**rec, **compare(m)})
    for rule in ("selected", "oracle"):
        for g in GRAPH:
            if all((rule, g, b) in chosen for b in ("semiconductor", "consumer")):
                m = pd.concat([chosen[(rule, g, b)] for b in ("semiconductor", "consumer")], ignore_index=True)
                rows.append({"rule": rule, "basket": "both", "model": g, "variant": "per basket", **compare(m)})
    v, s = pd.DataFrame(variants), pd.DataFrame(rows)
    v.to_csv(REV / "graph_tune_variants.csv", index=False)
    s.to_csv(REV / "graph_tune_summary.csv", index=False)
    pd.set_option("display.width", 250, "display.max_rows", 300)
    print(v.pivot_table(index=["basket", "variant"], columns="model", values=["val_r2", "test_r2"]).round(3).to_string())
    print(s.round(4).to_string(index=False))
    return 0


def compare(m: pd.DataFrame) -> dict:
    y, a, p = (m[c].to_numpy(float) for c in ("y", "ar1_12", "pred"))
    d = (y - a) ** 2 - (y - p) ** 2
    return {"n": len(m), "graph_r2": r2_score(y, p), "ar_r2": r2_score(y, a),
            "p_country": wild_cluster_boot(d, m["node_id"].to_numpy())[0], "p_month": wild_cluster_boot(d, m["month"].to_numpy())[0]}


if __name__ == "__main__":
    raise SystemExit(main())
