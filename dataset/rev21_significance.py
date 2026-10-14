#!/usr/bin/env python3
"""Paper-21 revision: can 193 test rows order ridge, GCN, and TGN?

Reads the regenerated 5-seed predictions (rev21_rerun_benchmark.py), forms the
seed ensemble exactly as paper 21 does (mean of the five seeds' test scores),
and for every model pair reports, on the 2024 hold-out:

  * the difference in R^2 (regression) or PR-AUC (classification),
  * a 95% country-cluster bootstrap interval for that difference (18 clusters,
    5,000 resamples of whole countries, paired: both models on the same draw),
  * wild-cluster-bootstrap and DM-HLN p-values on the squared-error
    differential (dm_tests.py, unchanged), and Holm-adjusted wild p-values,
  * the minimum detectable difference at 80% power and two-sided 5% level,
    2.80 x the bootstrap standard error of the difference.

The own-lag ridge (paper-21 Table XIV row "c_{v,T-1} only") is added as a
comparator because it is the strongest model in the paper.

Writes results/paper21_revision/{ensemble_check.csv, pairwise_tests.csv,
seed_spread.csv}.
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dm_tests import dm_hln, wild_cluster_boot  # noqa: E402
from rev21_tabular import TEST, TRAIN, TRADE, fit_predict, load  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "paper21_revision"
RUNS = OUT / "benchmark_default"
SEEDS = (7, 13, 21, 42, 101)
GRAPH = ("gcn", "tgn", "tgn_no_memory")
REPS = 5000
RNG = np.random.default_rng(20261001)


def read_task(task: str) -> pd.DataFrame:
    frames = []
    for seed in SEEDS:
        folder = RUNS / f"seed_{seed}" / task / "predictions"
        for path in folder.glob("*.csv"):
            p = pd.read_csv(path)
            p = p[p["split"] == "test"].copy()
            p["seed"] = seed
            frames.append(p)
    allp = pd.concat(frames, ignore_index=True)
    allp["month"] = allp["month"].astype(str)
    return allp


def ensemble_table(allp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = ["node_id", "month"]
    ens = allp.groupby(["model", *key], as_index=False).agg(target=("target", "first"), score=("score", "mean"))
    wide = ens.pivot_table(index=key, columns="model", values="score")
    wide["y"] = ens.groupby(key)["target"].first()
    per_seed = allp.groupby(["model", "seed"]).apply(lambda g: g, include_groups=False)
    return wide, per_seed


def add_tabular(wide: pd.DataFrame) -> pd.DataFrame:
    f = load()
    ok = f["T0"].notna()
    tr = f["month"].between(*TRAIN) & ok
    te = f["month"].between(*TEST) & ok
    idx = pd.MultiIndex.from_frame(f.loc[te, ["node_id", "month"]])
    wide["own_lag_ridge"] = pd.Series(fit_predict(f, ["T0_lag1"], "T0", tr, te), index=idx)
    wide["trade_only_ridge"] = pd.Series(fit_predict(f, TRADE, "T0", tr, te), index=idx)
    return wide


def boot_diff(y, a, b, clusters, metric):
    uniq = np.unique(clusters)
    rows = {c: np.where(clusters == c)[0] for c in uniq}
    obs = metric(y, a) - metric(y, b)
    draws = np.empty(REPS)
    for r in range(REPS):
        pick = np.concatenate([rows[c] for c in RNG.choice(uniq, size=len(uniq), replace=True)])
        try:
            draws[r] = metric(y[pick], a[pick]) - metric(y[pick], b[pick])
        except ValueError:
            draws[r] = np.nan
    draws = draws[np.isfinite(draws)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    se = float(np.std(draws, ddof=1))
    return obs, lo, hi, se, float((draws > 0).mean())


def holm(p: pd.Series) -> pd.Series:
    order = p.sort_values().index
    m = len(p)
    adj, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * p[k]))
        adj[k] = running
    return pd.Series(adj)[p.index]


def regression(wide: pd.DataFrame) -> pd.DataFrame:
    models = ["ridge_regression", *GRAPH, "own_lag_ridge", "trade_only_ridge"]
    sub = wide.dropna(subset=["y", *models])
    y = sub["y"].to_numpy(float)
    clusters = np.array([i[0] for i in sub.index])
    rows = []
    for a, b in combinations(models, 2):
        pa, pb = sub[a].to_numpy(float), sub[b].to_numpy(float)
        obs, lo, hi, se, pgt = boot_diff(y, pa, pb, clusters, r2_score)
        ea, eb = y - pa, y - pb
        _, dm_p = dm_hln(ea, eb)
        wild_p, _ = wild_cluster_boot(ea ** 2 - eb ** 2, clusters)
        rows.append({"task": "regression", "metric": "R2", "model_a": a, "model_b": b,
                     "a": r2_score(y, pa), "b": r2_score(y, pb), "diff": obs, "ci_lo": lo, "ci_hi": hi,
                     "boot_se": se, "mdd_80": 2.80 * se, "p_boot_a_gt_b": pgt,
                     "dm_hln_p": dm_p, "wild_p": wild_p, "n": len(y), "clusters": len(set(clusters))})
    out = pd.DataFrame(rows)
    out["wild_p_holm"] = holm(out["wild_p"])
    return out


def classification(wide: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    models = ["logistic_regression", "hand_weighted_linear", *GRAPH]
    sub = wide.dropna(subset=models).copy()
    sub["lab"] = labels.reindex(sub.index)
    sub = sub.dropna(subset=["lab"])
    y = sub["lab"].to_numpy(int)
    clusters = np.array([i[0] for i in sub.index])
    rows = []
    for a, b in combinations(models, 2):
        pa, pb = sub[a].to_numpy(float), sub[b].to_numpy(float)
        obs, lo, hi, se, pgt = boot_diff(y, pa, pb, clusters, average_precision_score)
        rows.append({"task": "classification", "metric": "PR-AUC", "model_a": a, "model_b": b,
                     "a": average_precision_score(y, pa), "b": average_precision_score(y, pb),
                     "diff": obs, "ci_lo": lo, "ci_hi": hi, "boot_se": se, "mdd_80": 2.80 * se,
                     "p_boot_a_gt_b": pgt, "n": len(y), "positives": int(y.sum()),
                     "clusters": len(set(clusters))})
    return pd.DataFrame(rows)


def main() -> int:
    reg = read_task("regression")
    clf = read_task("classification")
    rwide, _ = ensemble_table(reg)
    cwide, _ = ensemble_table(clf)
    rwide = add_tabular(rwide)

    check = []
    for m in [c for c in rwide.columns if c != "y"]:
        s = rwide[["y", m]].dropna()
        check.append({"task": "regression", "model": m, "metric": "R2", "value": r2_score(s["y"], s[m]), "n": len(s)})
    for m in [c for c in cwide.columns if c != "y"]:
        s = cwide[["y", m]].dropna()
        check.append({"task": "classification", "model": m, "metric": "PR-AUC",
                      "value": average_precision_score(s["y"].astype(int), s[m]), "n": len(s)})
    check = pd.DataFrame(check)
    check.to_csv(OUT / "ensemble_check.csv", index=False)
    print(check.round(3).to_string(index=False))

    spread = []
    for task, allp in (("regression", reg), ("classification", clf)):
        for (model, seed), g in allp.groupby(["model", "seed"]):
            if model not in GRAPH:
                continue
            val = r2_score(g["target"], g["score"]) if task == "regression" else average_precision_score(g["target"].astype(int), g["score"])
            spread.append({"task": task, "model": model, "seed": seed, "value": val})
    spread = pd.DataFrame(spread)
    spread.to_csv(OUT / "seed_spread.csv", index=False)
    print(spread.groupby(["task", "model"])["value"].agg(["mean", "std", "min", "max"]).round(3))

    tests = pd.concat([regression(rwide), classification(cwide, cwide["y"])], ignore_index=True)
    tests.to_csv(OUT / "pairwise_tests.csv", index=False)
    print(tests.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
