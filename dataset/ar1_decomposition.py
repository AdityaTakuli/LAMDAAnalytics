"""Phase 0.8 C.1 (BLOCKING) - is clean-target R^2 just the target's own lag?

Fit, on the paper split (train 2021-12..2022-12, val 2023, test 2024):
  ar1_only     : Ridge on the target's own lag-1 (no flow_ratio, no exogenous)
  ar1_plus_flow: lag-1 + flow_ratio + flow_ratio_lag1
  ar_ridge     : the canonical 3-feature AR (flow_ratio, T0-lag, flow_ratio_lag1)
                 kept as a reference so 0.138 stays comparable.

Report test R^2 and wild-cluster / DM p for ar1_plus_flow vs ar1_only, on T0 and T1.

Interpretation: if ar1_only ~ ar1_plus_flow and the p is large, all detectable
skill on the clean target is the target's serial correlation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dm_tests import (  # noqa: E402
    RIDGE_ALPHAS, dm_hln, pairs_cluster_boot, wild_cluster_boot,
)
from phase0_t1_ladder import load, standardize, win  # noqa: E402

TRAIN, VALID, TEST = ("2021-12", "2022-12"), ("2023-01", "2023-12"), ("2024-01", "2024-11")
ROOT = Path(__file__).resolve().parent.parent


def tune_ridge(z, y, tr, va):
    ytr = y[tr].to_numpy(float)
    yva = y[va].to_numpy(float)
    best, bv = None, -np.inf
    for a in RIDGE_ALPHAS:
        mdl = Ridge(alpha=a).fit(z[tr].to_numpy(), ytr)
        s = r2_score(yva, mdl.predict(z[va].to_numpy())) if np.std(yva) > 1e-12 else -np.inf
        if s > bv:
            bv, best = s, mdl
    return best


def fit_pred(f, target, feats, tr, va, te):
    z = standardize(f, feats, tr)
    mdl = tune_ridge(z, f[target], tr, va)
    pred = mdl.predict(z[te].to_numpy())
    coef = dict(zip(feats, mdl.coef_.tolist()))
    coef["intercept"] = float(mdl.intercept_)
    return pred, coef


def compare(f, target):
    lagcol = f"{target}_lag1"
    grp = f.groupby("host_country_id", sort=False)
    f[lagcol] = grp[target].shift(1)
    valid = f[target].notna() & f[lagcol].notna()
    tr = win(f, TRAIN) & valid
    va = win(f, VALID) & valid
    te = win(f, TEST) & valid
    y = f.loc[te, target].to_numpy(float)
    idx = list(zip(f.loc[te, "host_country_id"], f.loc[te, "month"]))

    specs = {
        "ar1_only": [lagcol],
        "ar1_plus_flow": [lagcol, "flow_ratio", "flow_ratio_lag1"],
        "ar_ridge": ["flow_ratio", "contraction_lag1", "flow_ratio_lag1"],
        "persistence": None,
    }
    preds, coefs = {}, {}
    for name, feats in specs.items():
        if name == "persistence":
            preds[name] = f.loc[te, lagcol].to_numpy(float)
            coefs[name] = {lagcol: 1.0, "intercept": 0.0}
            continue
        preds[name], coefs[name] = fit_pred(f, target, feats, tr, va, te)

    print(f"\n==== {target}  test n={int(te.sum())}  "
          f"in-sample lag-1 AC (pooled) would imply R^2 ~ rho^2 ====")
    print(f"{'model':18s} {'R2':>7s}  coefs")
    for name in specs:
        r2 = r2_score(y, preds[name])
        print(f"{name:18s} {r2:7.3f}  {coefs[name]}")

    clusters = np.array([c for c, _ in idx])

    def trio(name_a, pred_a, name_b, pred_b):
        eA = y - pred_a
        eB = y - pred_b
        d = eA ** 2 - eB ** 2
        _, p_dm = dm_hln(eA, eB)
        p_pairs = pairs_cluster_boot(d, clusters)
        p_wild, _ = wild_cluster_boot(d, clusters)
        better = name_b if d.mean() > 0 else name_a
        print(f"  {name_a} vs {name_b}:  DM p={p_dm:.3f}  pairs p={p_pairs:.3f}  "
              f"wild p={p_wild:.3f}  better={better}  nC={len(set(clusters))}")
        return p_dm, p_pairs, p_wild

    p_dm, p_pairs, p_wild = trio("ar1_plus_flow", preds["ar1_plus_flow"],
                                 "ar1_only", preds["ar1_only"])
    z_dm, z_pairs, z_wild = trio("ar1_only", preds["ar1_only"],
                                 "zero", np.zeros_like(y))
    return {
        "target": target, "n_test": int(te.sum()), "n_clusters": len(set(clusters)),
        "ar1_only": r2_score(y, preds["ar1_only"]),
        "ar1_plus_flow": r2_score(y, preds["ar1_plus_flow"]),
        "ar_ridge": r2_score(y, preds["ar_ridge"]),
        "persistence": r2_score(y, preds["persistence"]),
        "wild_p_flow_vs_ar1": p_wild, "dm_p_flow_vs_ar1": p_dm,
        "pairs_p_flow_vs_ar1": p_pairs,
        "wild_p_ar1_vs_zero": z_wild, "dm_p_ar1_vs_zero": z_dm,
        "ar1_coef": coefs["ar1_only"].get(lagcol, np.nan),
    }


def main():
    f = load()
    rows = [compare(f.copy(), t) for t in ("T1", "T0")]
    out = pd.DataFrame(rows)
    (ROOT / "results/phase06").mkdir(parents=True, exist_ok=True)
    out.to_csv(ROOT / "results/phase06/ar1_decomposition.csv", index=False)
    print("\n-> results/phase06/ar1_decomposition.csv")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
