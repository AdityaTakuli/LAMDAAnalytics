#!/usr/bin/env python3
"""Phase A: summarise the consumer-basket graph runs next to the semiconductor ones.

Seed ensembles (mean of the seeds' test scores, as in paper 21) for
  * the default benchmark (T0 regression R^2, T0 tau=0.20 classification PR-AUC),
  * the T1 target (regression R^2),
  * the rolling origins (block and pooled R^2),
for both baskets. Semiconductor numbers are read from the existing runs in
results/paper21_revision/, consumer numbers from results/paper21_revision/consumer/.

Writes results/paper21_revision/consumer/graph_summary.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, r2_score

REV = Path(__file__).resolve().parents[1] / "results" / "paper21_revision"
MODELS = ("ridge_regression", "logistic_regression", "hand_weighted_linear", "gcn", "tgn", "tgn_no_memory")
BLOCKS = ("2023H1", "2023H2", "2024H1", "2024H2")


def ensemble(folders: list[Path]) -> pd.DataFrame:
    frames = []
    for folder in folders:
        for path in (folder / "predictions").glob("*.csv"):
            p = pd.read_csv(path)
            frames.append(p[p["split"] == "test"])
    if not frames:
        return pd.DataFrame()
    allp = pd.concat(frames)
    allp["month"] = allp["month"].astype(str)
    ens = allp.groupby(["model", "node_id", "month"], as_index=False).agg(y=("target", "first"), score=("score", "mean"))
    wide = ens.pivot_table(index=["node_id", "month"], columns="model", values="score")
    wide["y"] = ens.groupby(["node_id", "month"])["y"].first()
    return wide


def score(wide: pd.DataFrame, metric) -> dict[str, float]:
    out = {}
    for m in MODELS:
        if m in wide:
            s = wide[["y", m]].dropna()
            if len(s) and s["y"].nunique() > 1:
                out[m] = metric(s["y"].astype(float) if metric is r2_score else s["y"].astype(int), s[m])
    return out


def main() -> int:
    rows = []
    for basket, root in (("semiconductor", REV), ("consumer", REV / "consumer")):
        for task, metric in (("regression", r2_score), ("classification", average_precision_score)):
            w = ensemble(sorted((root / "benchmark_default").glob(f"seed_*/{task}")))
            if not w.empty:
                rows.append({"basket": basket, "setting": f"T0 {task}", "n": int(w["y"].notna().sum()), **score(w, metric)})
        w = ensemble(sorted((root / "graph_t1").glob("seed_*")))
        if not w.empty:
            rows.append({"basket": basket, "setting": "T1 regression", "n": int(w["y"].notna().sum()), **score(w, r2_score)})
        for target in ("T0", "T1"):
            ys, ps = [], {}
            for block in BLOCKS:
                w = ensemble([s for s in sorted((root / "graph_rolling" / target / block).glob("seed_*")) if (s / "predictions" / "tgn_no_memory.csv").exists()])
                if w.empty:
                    continue
                w = w.dropna(subset=["y", "gcn", "tgn", "tgn_no_memory"])
                rows.append({"basket": basket, "setting": f"rolling {target} {block}", "n": len(w), **score(w, r2_score)})
                ys.append(w["y"].to_numpy(float))
                for m in ("gcn", "tgn", "tgn_no_memory"):
                    ps.setdefault(m, []).append(w[m].to_numpy(float))
            if len(ys) == len(BLOCKS):
                y = np.concatenate(ys)
                rows.append({"basket": basket, "setting": f"rolling {target} pooled", "n": len(y),
                             **{m: r2_score(y, np.concatenate(v)) for m, v in ps.items()}})
    out = pd.DataFrame(rows)
    (REV / "consumer").mkdir(parents=True, exist_ok=True)
    out.to_csv(REV / "consumer" / "graph_summary.csv", index=False)
    print(out.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
