#!/usr/bin/env python3
"""Graph models on the non-overlapping target, and paired tests.

Reads results/benchmark/graph_t1/seed_*/predictions (graph_logchange_rolling.py t1),
forms the five-seed ensemble, adds the tabular T1 models from tabular_analyses.py
on the same rows, and reports test-2024 R^2 plus wild-cluster-bootstrap
p-values for AR(1,12) against zero, AR(1), the seven-feature ridge, and each
graph ensemble. Also summarises the rolling-origin graph runs if present.

Writes results/benchmark/{t1_graph_ensemble.csv, t1_pairwise.csv,
rolling_graph.csv}.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dm_tests import dm_hln, wild_cluster_boot  # noqa: E402
from tabular_analyses import ALL7, BLOCKS, TEST, TRAIN, fit_predict, load  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "benchmark"
GRAPH = ("gcn", "tgn", "tgn_no_memory")


def ensemble(folders: list[Path]) -> pd.DataFrame:
    frames = []
    for folder in folders:
        for path in (folder / "predictions").glob("*.csv"):
            p = pd.read_csv(path)
            frames.append(p[p["split"] == "test"])
    allp = pd.concat(frames)
    allp["month"] = allp["month"].astype(str)
    ens = allp.groupby(["model", "node_id", "month"], as_index=False).agg(y=("target", "first"), score=("score", "mean"))
    wide = ens.pivot_table(index=["node_id", "month"], columns="model", values="score")
    wide["y"] = ens.groupby(["node_id", "month"])["y"].first()
    return wide


def main() -> int:
    f = load()
    wide = ensemble(sorted((OUT / "graph_t1").glob("seed_*")))
    ok = f["T1"].notna()
    tr = f["month"].between(*TRAIN) & ok
    te = f["month"].between(*TEST) & ok
    idx = pd.MultiIndex.from_frame(f.loc[te, ["node_id", "month"]])
    wide["ar1_12"] = pd.Series(fit_predict(f, ["T1_lag1", "T1_lag12"], "T1", tr, te), index=idx)
    wide["ar1"] = pd.Series(fit_predict(f, ["T1_lag1"], "T1", tr, te), index=idx)
    wide["ridge7"] = pd.Series(fit_predict(f, ALL7, "T1", tr, te), index=idx)
    wide["zero"] = 0.0
    wide = wide.dropna(subset=["y", "ar1_12", *GRAPH])
    y = wide["y"].to_numpy(float)
    clusters = np.array([i[0] for i in wide.index])
    rows = [{"model": m, "r2": r2_score(y, wide[m]), "n": len(y)} for m in [*GRAPH, "ridge_regression", "ar1", "ar1_12", "ridge7"] if m in wide]
    pd.DataFrame(rows).to_csv(OUT / "t1_graph_ensemble.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    tests = []
    for b in ["zero", "ar1", "ridge7", *GRAPH]:
        ea, eb = y - wide["ar1_12"].to_numpy(float), y - wide[b].to_numpy(float)
        wild_p, _ = wild_cluster_boot(ea ** 2 - eb ** 2, clusters)
        tests.append({"a": "ar1_12", "b": b, "r2_a": r2_score(y, wide["ar1_12"]), "r2_b": r2_score(y, wide[b]),
                      "dm_hln_p": dm_hln(ea, eb)[1], "wild_p": wild_p, "n": len(y), "clusters": len(set(clusters))})
    ea, eb = y - wide["ar1"].to_numpy(float), y
    tests.append({"a": "ar1", "b": "zero", "r2_a": r2_score(y, wide["ar1"]), "r2_b": r2_score(y, wide["zero"]),
                  "dm_hln_p": dm_hln(ea, eb)[1], "wild_p": wild_cluster_boot(ea ** 2 - eb ** 2, clusters)[0],
                  "n": len(y), "clusters": len(set(clusters))})
    pd.DataFrame(tests).to_csv(OUT / "t1_pairwise.csv", index=False)
    print(pd.DataFrame(tests).round(3).to_string(index=False))

    roll = OUT / "graph_rolling"
    if roll.exists():
        recs = []
        for target in ("T0", "T1"):
            ys, ps = [], {m: [] for m in GRAPH}
            for block in ("2023H1", "2023H2", "2024H1", "2024H2"):
                seeds = sorted((roll / target / block).glob("seed_*"))
                done = [s for s in seeds if (s / "predictions").exists() and any((s / "predictions").glob("tgn_no_memory.csv"))]
                if not done:
                    continue
                w = ensemble(done).dropna(subset=["y", *GRAPH])
                rec = {"target": target, "block": block, "seeds": len(done), "n": len(w)}
                for m in GRAPH:
                    rec[m] = r2_score(w["y"], w[m])
                    ps[m].append(w[m].to_numpy(float))
                ys.append(w["y"].to_numpy(float))
                recs.append(rec)
            if len(ys) == 4:
                yall = np.concatenate(ys)
                rec = {"target": target, "block": "pooled", "seeds": None, "n": len(yall)}
                for m in GRAPH:
                    rec[m] = r2_score(yall, np.concatenate(ps[m]))
                recs.append(rec)
        if recs:
            pd.DataFrame(recs).to_csv(OUT / "rolling_graph.csv", index=False)
            print(pd.DataFrame(recs).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
