"""Phase 0.6 C.2 + Phase 0.7 C.3/C.4 - significance tests on squared-error loss, test 2024.

Three p-values per pair (unit of observation = country x month; 18 country clusters):
  - DM-HLN     : Diebold-Mariano, Harvey-Leybourne-Newbold small-sample correction (h=1).
                 Misspecified here (assumes a single series; ignores cross-country dep).
  - pairs boot : country-clustered pairs bootstrap (over-rejects with few clusters).
  - wild boot  : wild cluster bootstrap, Rademacher weights, null imposed, 9999 reps
                 (standard few-cluster remedy). Cluster count = 18.

Graph predictions are the CANONICAL converged runs (80 epochs, patience 15, 5 seeds),
read from results/phase07/graph_preds_{T0,T1}.csv as the seed-ensemble (mean over
seeds) predictor. No graph training happens here. Run graph_convergence.py first.

Pairs (T1): ar_ridge vs zero, ar_ridge vs random_forest, ar_ridge vs tgn(ensemble),
tgn vs tgn_no_memory.   Pairs (T0): persistence vs gcn(ensemble), persistence vs ar_ridge.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as tdist
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(__file__).parent / "data/four_year_2021_2024/processed"
TRAIN, VALID, TEST = ("2021-12", "2022-12"), ("2023-01", "2023-12"), ("2024-01", "2024-11")
PAPER = ["inventory_days_proxy", "trade_delay_proxy", "news_vol_7d", "neg_tone_frac_3d",
         "strike_flag_7d", "weather_anomaly_7d", "global_risk"]
PROTECTIVE = "inventory_days_proxy"
IMPROVED = ["flow_ratio", "contraction_lag1", "flow_ratio_lag1"]
RIDGE_ALPHAS = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
WILD_REPS, BOOT_REPS, SEED = 9999, 5000, 12345


def load_nodes():
    n = pd.read_csv(DATA / "nodes_monthly.csv")
    n["month"] = n["month"].astype(str)
    n = n.sort_values(["host_country_id", "month"]).reset_index(drop=True)
    v = n.groupby("host_country_id", sort=False)["inbound_flow_usd"]
    fut = v.shift(-1)
    med12 = v.transform(lambda s: s.rolling(12, min_periods=1).median())
    n["T0"] = (fut - med12) / med12.replace(0, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        pos = n["inbound_flow_usd"] > 0
        n["T1"] = np.log(fut / n["inbound_flow_usd"].where(pos))
        n.loc[fut <= 0, "T1"] = np.nan
    n["flow_ratio"] = n["inventory_days_proxy"] / 30.0
    g = n.groupby("host_country_id", sort=False)
    n["flow_ratio_lag1"] = g["flow_ratio"].shift(1)
    n["contraction_lag1"] = g["T0"].shift(1)
    n["T1_lag1"] = g["T1"].shift(1)
    n["T0_lag1"] = g["T0"].shift(1)
    return n


def win(s, b):
    return s.between(b[0], b[1])


def standardize(f, feats, tr):
    m = f[feats].astype(float).replace([np.inf, -np.inf], np.nan)
    if PROTECTIVE in feats:
        m[PROTECTIVE] = -m[PROTECTIVE]
    mean = m[tr].mean(); scale = m[tr].std().replace(0.0, 1.0).fillna(1.0)
    return ((m - mean) / scale).fillna(0.0)


def tune_ridge(z, target, f, tr, va):
    ytr = f.loc[tr, target].to_numpy(float); yva = f.loc[va, target].to_numpy(float)
    best, bv = None, -np.inf
    for a in RIDGE_ALPHAS:
        mdl = Ridge(alpha=a).fit(z[tr].to_numpy(), ytr)
        s = r2_score(yva, mdl.predict(z[va].to_numpy())) if np.std(yva) > 1e-12 else -np.inf
        if s > bv:
            bv, best = s, mdl
    return best


def tabular_preds(f, target):
    valid = f[target].notna()
    tr = win(f["month"], TRAIN) & valid
    va = win(f["month"], VALID) & valid
    te = win(f["month"], TEST) & valid
    idx = pd.MultiIndex.from_frame(f.loc[te, ["host_country_id", "month"]])
    out = pd.DataFrame(index=idx)
    out["y"] = f.loc[te, target].to_numpy(float)
    out["zero"] = 0.0
    out["persistence"] = f.groupby("host_country_id", sort=False)[target].shift(1)[te].to_numpy(float)
    z_ar = standardize(f, IMPROVED, tr)
    out["ar_ridge"] = tune_ridge(z_ar, target, f, tr, va).predict(z_ar[te].to_numpy())
    lagcol = f"{target}_lag1"
    if lagcol in f.columns:
        z1 = standardize(f, [lagcol], tr)
        out["ar1_only"] = tune_ridge(z1, target, f, tr, va).predict(z1[te].to_numpy())
    z7 = standardize(f, PAPER, tr)
    out["tuned_ridge_7feat"] = tune_ridge(z7, target, f, tr, va).predict(z7[te].to_numpy())
    ytr = f.loc[tr, target].to_numpy(float); yva = f.loc[va, target].to_numpy(float)
    best_rf, bv = None, -np.inf
    for md in (2, 3, 4, None):
        for leaf in (1, 5, 10):
            rf = RandomForestRegressor(n_estimators=300, max_depth=md, min_samples_leaf=leaf, random_state=0)
            rf.fit(z_ar[tr].to_numpy(), ytr)
            s = r2_score(yva, rf.predict(z_ar[va].to_numpy())) if np.std(yva) > 1e-12 else -np.inf
            if s > bv:
                bv, best_rf = s, rf
    out["random_forest"] = best_rf.predict(z_ar[te].to_numpy())
    return out.dropna(subset=["y"])


def add_graph_ensemble(table, target):
    """Merge seed-ensemble (mean over seeds) canonical graph predictions."""
    path = ROOT / f"results/phase07/graph_preds_{target}.csv"
    g = pd.read_csv(path)
    g["month"] = g["month"].astype(str)
    ens = (g.groupby(["model", "host_country_id", "month"])["pred"].mean().reset_index())
    for kind, col in (("gcn", "gcn"), ("tgn", "tgn"), ("tgn_no_mem", "tgn_no_memory")):
        sub = ens[ens["model"] == kind].set_index(["host_country_id", "month"])["pred"]
        table[col] = table.index.map(sub)
    return table


# ---------------- inference ----------------
def dm_hln(eA, eB, h=1):
    d = eA ** 2 - eB ** 2
    n = len(d); dbar = d.mean(); var = d.var(ddof=0) / n
    if var <= 0:
        return float("nan"), float("nan")
    dm = dbar / np.sqrt(var)
    stat = dm * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(stat), float(2 * (1 - tdist.cdf(abs(stat), df=n - 1)))


def pairs_cluster_boot(d, clusters, reps=BOOT_REPS):
    rng = np.random.default_rng(SEED)
    uniq = np.array(sorted(set(clusters)))
    groups = {c: d[clusters == c] for c in uniq}
    obs = d.mean(); boots = np.empty(reps)
    for r in range(reps):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        boots[r] = np.concatenate([groups[c] for c in pick]).mean()
    return float((np.abs(boots - boots.mean()) >= abs(obs)).mean())


def _cluster_t(d, clusters, uniq):
    n = len(d); beta = d.mean(); u = d - beta
    Sg = np.array([u[clusters == c].sum() for c in uniq])
    var = Sg @ Sg / n ** 2
    return beta / np.sqrt(var) if var > 0 else np.nan


def wild_cluster_boot(d, clusters, reps=WILD_REPS):
    """Wild cluster bootstrap, Rademacher weights, null imposed (mu=0)."""
    rng = np.random.default_rng(SEED)
    uniq = np.array(sorted(set(clusters)))
    idx_by_c = {c: np.where(clusters == c)[0] for c in uniq}
    t_obs = _cluster_t(d, clusters, uniq)
    # restricted residuals under H0: mu=0 -> residual = d
    count = 0
    for _ in range(reps):
        w = rng.choice([-1.0, 1.0], size=len(uniq))
        dstar = d.copy()
        for wj, c in zip(w, uniq):
            dstar[idx_by_c[c]] = wj * d[idx_by_c[c]]
        t_star = _cluster_t(dstar, clusters, uniq)
        if not np.isnan(t_star) and abs(t_star) >= abs(t_obs):
            count += 1
    return float((count + 1) / (reps + 1)), float(t_obs)


RESULTS = []


def run_pair(name, table, a, b, wild=True):
    sub = table[["y", a, b]].dropna()
    y = sub["y"].to_numpy(float)
    eA = y - sub[a].to_numpy(float); eB = y - sub[b].to_numpy(float)
    d = eA ** 2 - eB ** 2
    clusters = np.array([ix[0] for ix in sub.index])
    stat, p = dm_hln(eA, eB)
    pb = pairs_cluster_boot(d, clusters)
    pw, _ = wild_cluster_boot(d, clusters) if wild else (float("nan"), 0)
    better = b if d.mean() > 0 else a
    r2a, r2b = r2_score(y, sub[a]), r2_score(y, sub[b])
    print(f"{name:36s} n={len(y):3d} nC={len(set(clusters)):2d}  "
          f"R2({a})={r2a:+.3f} R2({b})={r2b:+.3f}  DM p={p:.3f}  pairs_p={pb:.3f}  "
          f"wild_p={pw:.3f}  better={better}", flush=True)
    RESULTS.append({
        "pair": name, "target": name.split(":")[0], "model_a": a, "model_b": b,
        "n": len(y), "n_clusters": len(set(clusters)),
        "r2_a": r2a, "r2_b": r2b, "dm_p": p, "pairs_p": pb, "wild_p": pw,
        "verdict": "tie" if min(p, pb, pw) >= 0.05 else better,
    })


def main():
    nodes = load_nodes()
    t0 = add_graph_ensemble(tabular_preds(nodes.copy(), "T0"), "T0")
    t1 = add_graph_ensemble(tabular_preds(nodes.copy(), "T1"), "T1")

    print("=== PRIMARY: T1 vs ar1_only (univariate own-lag, R2~0.171) ===")
    run_pair("T1: ar1_only vs zero", t1, "ar1_only", "zero")
    run_pair("T1: ar1_only vs tgn", t1, "ar1_only", "tgn")
    run_pair("T1: ar1_only vs tgn_no_memory", t1, "ar1_only", "tgn_no_memory")
    run_pair("T1: ar1_only vs gcn", t1, "ar1_only", "gcn")
    run_pair("T1: ar1_only vs random_forest", t1, "ar1_only", "random_forest")
    run_pair("T1: ar1_only vs ar_ridge_3feat", t1, "ar1_only", "ar_ridge")

    print("\n=== SECONDARY: T1 vs ar_ridge 3-feat (R2~0.138), kept for Table 8 ===")
    run_pair("T1: ar_ridge vs zero", t1, "ar_ridge", "zero")
    run_pair("T1: ar_ridge vs random_forest", t1, "ar_ridge", "random_forest")
    run_pair("T1: ar_ridge vs tgn (ensemble)", t1, "ar_ridge", "tgn")
    run_pair("T1: tgn vs tgn_no_memory", t1, "tgn", "tgn_no_memory")

    print("\n=== DM / bootstrap tests under T0 (contraction, test 2024) ===")
    run_pair("T0: persistence vs gcn (ensemble)", t0, "persistence", "gcn")
    run_pair("T0: persistence vs ar_ridge", t0, "persistence", "ar_ridge")

    print("\n(positive-difference model has larger squared error; 'better' = lower loss. "
          "Wild cluster bootstrap: Rademacher, null imposed, 9999 reps; "
          "18 clusters under T0, 17 under T1.)")

    out = ROOT / "results/phase06/dm_tests.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(RESULTS).to_csv(out, index=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
