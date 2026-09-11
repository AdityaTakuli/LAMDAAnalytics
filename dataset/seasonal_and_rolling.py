"""Phase 08 - two robustness checks requested in review.

(A) Seasonal lag. T1 increments have lag-12 autocorrelation +0.294 (paper F1), so
    the natural challenger to the univariate AR(1) is an AR(1,12) with the
    target's own lags 1 and 12. Same protocol as dm_tests.py: ridge with alpha
    tuned on 2023, scored once on 2024, DM-HLN / pairs / wild cluster bootstrap.

(B) Rolling origins. The paper's model verdict rests on one test year. Here the
    tabular models are refit at each expanding origin and scored on the same
    six-month blocks used for per-origin persistence. For each block the six
    preceding months are the validation window and everything from 2021-12
    before that is training. Blocks with fewer than eight training months are
    skipped. Graph models are not rerun (their cost is the reason for the
    matched-budget protocol).

Writes results/phase08/{seasonal_ar.csv, seasonal_vs_all.csv, rolling_origin_models.csv}.
Reads only the fused panel; retrains nothing outside this script.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from dm_tests import (IMPROVED, TEST, TRAIN, VALID, dm_hln, load_nodes, pairs_cluster_boot,
                      standardize, tune_ridge, wild_cluster_boot, win)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/phase08"
HORIZON, MIN_TRAIN_MONTHS = 6, 8


def add_lags(f: pd.DataFrame) -> pd.DataFrame:
    g = f.groupby("host_country_id", sort=False)
    for t in ("T0", "T1"):
        f[f"{t}_lag1"] = g[t].shift(1)
        f[f"{t}_lag12"] = g[t].shift(12)
    return f


def ridge_pred(f, target, feats, tr, va, te):
    z = standardize(f, feats, tr)
    return tune_ridge(z, target, f, tr, va).predict(z[te].to_numpy())


def rf_pred(f, target, feats, tr, va, te):
    z = standardize(f, feats, tr)
    ytr, yva = f.loc[tr, target].to_numpy(float), f.loc[va, target].to_numpy(float)
    best, bv = None, -np.inf
    for md in (2, 3, 4, None):
        for leaf in (1, 5, 10):
            m = RandomForestRegressor(n_estimators=300, max_depth=md, min_samples_leaf=leaf,
                                      random_state=0).fit(z[tr].to_numpy(), ytr)
            s = r2_score(yva, m.predict(z[va].to_numpy())) if np.std(yva) > 1e-12 else -np.inf
            if s > bv:
                bv, best = s, m
    return best.predict(z[te].to_numpy())


def test_pair(y, pa, pb, clusters):
    ea, eb = y - pa, y - pb
    d = ea ** 2 - eb ** 2
    _, dm_p = dm_hln(ea, eb)
    return dm_p, pairs_cluster_boot(d, clusters), wild_cluster_boot(d, clusters)[0]


def seasonal(f: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in ("T1", "T0"):
        valid = f[target].notna()
        tr = win(f["month"], TRAIN) & valid
        va = win(f["month"], VALID) & valid
        te = win(f["month"], TEST) & valid
        y = f.loc[te, target].to_numpy(float)
        clusters = f.loc[te, "host_country_id"].to_numpy()
        ar1 = ridge_pred(f, target, [f"{target}_lag1"], tr, va, te)
        ar112 = ridge_pred(f, target, [f"{target}_lag1", f"{target}_lag12"], tr, va, te)
        for name, pa, pb in (("AR(1,12) vs AR(1)", ar112, ar1),
                             ("AR(1,12) vs zero", ar112, np.zeros_like(y))):
            dm_p, pairs_p, wild_p = test_pair(y, pa, pb, clusters)
            rows.append({"target": target, "pair": name, "n": len(y),
                         "n_clusters": len(set(clusters)),
                         "r2_a": r2_score(y, pa), "r2_b": r2_score(y, pb),
                         "dm_p": dm_p, "pairs_p": pairs_p, "wild_p": wild_p})
            print(f"{target} {name:18s} n={len(y)} R2 {rows[-1]['r2_a']:+.4f} vs "
                  f"{rows[-1]['r2_b']:+.4f}  DM p={dm_p:.3f} pairs p={pairs_p:.3f} "
                  f"wild p={wild_p:.3f}", flush=True)
    return pd.DataFrame(rows)


def seasonal_vs_all(f: pd.DataFrame) -> pd.DataFrame:
    """T1 AR(1,12) against every model in dm_tests, graph entries as seed ensembles."""
    from dm_tests import add_graph_ensemble, tabular_preds
    t = "T1"
    table = add_graph_ensemble(tabular_preds(f, t), t)
    valid = f[t].notna()
    tr, va, te = (win(f["month"], b) & valid for b in (TRAIN, VALID, TEST))
    idx = pd.MultiIndex.from_frame(f.loc[te, ["host_country_id", "month"]])
    table["ar1_12"] = pd.Series(ridge_pred(f, t, ["T1_lag1", "T1_lag12"], tr, va, te), index=idx)
    rows = []
    for b in ("ar1_only", "gcn", "tgn_no_memory", "random_forest", "tgn", "ar_ridge"):
        sub = table[["y", "ar1_12", b]].dropna()
        y = sub["y"].to_numpy(float)
        clusters = np.array([ix[0] for ix in sub.index])
        dm_p, pairs_p, wild_p = test_pair(y, sub["ar1_12"].to_numpy(float),
                                          sub[b].to_numpy(float), clusters)
        rows.append({"pair": f"T1: AR(1,12) vs {b}", "n": len(y),
                     "n_clusters": len(set(clusters)), "r2_a": r2_score(y, sub["ar1_12"]),
                     "r2_b": r2_score(y, sub[b]), "dm_p": dm_p, "pairs_p": pairs_p,
                     "wild_p": wild_p})
        print(f"AR(1,12) {rows[-1]['r2_a']:.3f} vs {b:14s} {rows[-1]['r2_b']:.3f}  "
              f"DM p={dm_p:.3f} pairs p={pairs_p:.3f} wild p={wild_p:.3f}", flush=True)
    return pd.DataFrame(rows)


def rolling(f: pd.DataFrame) -> pd.DataFrame:
    months = sorted(f["month"].unique())
    first_train = months.index(TRAIN[0])
    rows = []
    for start in range(13, len(months), HORIZON):
        block = months[start:start + HORIZON]
        val = months[max(start - HORIZON, 0):start]
        train = months[first_train:max(start - HORIZON, 0)]
        if len(block) < 2 or len(train) < MIN_TRAIN_MONTHS:
            continue
        for target in ("T1", "T0"):
            valid = f[target].notna()
            tr = f["month"].isin(train) & valid
            va = f["month"].isin(val) & valid
            te = f["month"].isin(block) & valid
            y = f.loc[te, target].to_numpy(float)
            preds = {
                "zero": np.zeros_like(y),
                "persistence": f[f"{target}_lag1"][te].fillna(0.0).to_numpy(float),
                "ar1": ridge_pred(f, target, [f"{target}_lag1"], tr, va, te),
                "ar1_12": ridge_pred(f, target, [f"{target}_lag1", f"{target}_lag12"], tr, va, te),
                "ar_ridge_3feat": ridge_pred(f, target, IMPROVED, tr, va, te),
                "random_forest": rf_pred(f, target, IMPROVED, tr, va, te),
            }
            rec = {"target": target, "eval": f"{block[0]}..{block[-1]}", "n": len(y),
                   "train_months": len(train)}
            rec.update({k: r2_score(y, v) for k, v in preds.items()})
            rows.append(rec)
            print(f"{target} {rec['eval']:18s} n={len(y):3d} train={len(train):2d}  " +
                  "  ".join(f"{k}={rec[k]:+.3f}" for k in preds), flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    f = add_lags(load_nodes())
    OUT.mkdir(parents=True, exist_ok=True)
    print("== (A) seasonal lag")
    seasonal(f).to_csv(OUT / "seasonal_ar.csv", index=False)
    seasonal_vs_all(f).to_csv(OUT / "seasonal_vs_all.csv", index=False)
    print("== (B) rolling origins")
    roll = rolling(f)
    roll.to_csv(OUT / "rolling_origin_models.csv", index=False)
    cols = ["zero", "persistence", "ar1", "ar1_12", "ar_ridge_3feat", "random_forest"]
    print("\nmean over origins:")
    print(roll.groupby("target")[cols].mean().round(3).to_string())
    print("\nAR(1) best tabular model in each block:")
    for target, g in roll.groupby("target"):
        best = g[cols].idxmax(axis=1)
        print(f"  {target}: " + ", ".join(f"{e}:{b}" for e, b in zip(g['eval'], best)))


if __name__ == "__main__":
    main()
