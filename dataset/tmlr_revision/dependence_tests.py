"""TMLR revision, reviewer point 4: dependence between country-month observations.

The submitted tests cluster by country only. Rows of the same month share GSCPI
(one global series) and any common shock, so they are not independent across
countries. For every pair the paper tests (Tables sig and seasonal; the T0 pairs
on the corrected rows) plus the four exogenous contrasts, this script reports:

  country  wild cluster bootstrap by country (Rademacher, null imposed, 9,999 reps)
           -- the submitted procedure, recomputed here
  month    wild cluster bootstrap by month (11 test months, Webb six-point weights,
           null imposed, 9,999 reps)
  twoway   two-way (country and month) cluster-robust t (Cameron, Gelbach and
           Miller 2011), referred to t with min(G)-1 = 10 degrees of freedom
  dm_month Diebold-Mariano-HLN on the month-averaged loss differential (11 values):
           each month is one observation, so all cross-country dependence within
           a month is absorbed
and Holm-adjusted versions of each over the full family of tested pairs.

Retrains nothing except seconds-long ridge fits; graph predictions are the saved
five-seed ensembles. Output: out/dependence_tests.csv
Run from dataset/:  ../backend/.venv/bin/python tmlr_revision/dependence_tests.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as tdist
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import build_canonical as bc  # noqa: E402
import dm_tests as dmt  # noqa: E402
from missing_and_timing import frames, nodes_like  # noqa: E402
from phase0_t1_ladder import RIDGE, standardize, tune_reg, win  # noqa: E402
from preprocessing_exogenous import (add_gscpi_dynamics, add_news_normalized,  # noqa: E402
                                     add_weather_seasonal)

OUT = HERE / "out"
REPS, SEED = 9999, 12345
WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


# --------------------------------------------------------------------------- #
# predictions
# --------------------------------------------------------------------------- #
def ridge_preds(f, target, feats, frame=None):
    g = f if frame is None else frame
    valid = f[target].notna()
    tr, va, te = win(f, bc.TRAIN) & valid, win(f, bc.VALID) & valid, win(f, bc.TEST) & valid
    z = standardize(g, feats, tr)
    m = tune_reg(z[tr].to_numpy(), g.loc[tr, target].to_numpy(float),
                 z[va].to_numpy(), g.loc[va, target].to_numpy(float), Ridge, RIDGE)
    idx = pd.MultiIndex.from_frame(f.loc[te, ["host_country_id", "month"]])
    return pd.Series(m.predict(z[te].to_numpy()), index=idx)


def t1_table() -> pd.DataFrame:
    n = dmt.load_nodes()
    table = dmt.add_graph_ensemble(dmt.tabular_preds(n.copy(), "T1"), "T1")
    raw, _, _ = frames()
    raw["T1_lag12"] = raw.groupby("host_country_id", sort=False)["T1"].shift(12)
    table["ar1_12"] = ridge_preds(raw, "T1", ["T1_lag1", "T1_lag12"])
    tr = win(raw, bc.TRAIN) & raw["T1"].notna()
    exo = {"exo_news": (add_news_normalized, ["news_z", "news_surprise"]),
           "exo_weather": (add_weather_seasonal, ["weather_temp_z", "weather_temp_absz"]),
           "exo_gscpi": (add_gscpi_dynamics, ["gscpi_change", "gscpi_lag1"])}
    table["exo_base"] = ridge_preds(raw, "T1", bc.AR)
    for name, (fn, cols) in exo.items():
        table[name] = ridge_preds(raw, "T1", bc.AR + cols, fn(raw.copy(), tr))
    g = add_gscpi_dynamics(add_weather_seasonal(add_news_normalized(raw.copy(), tr), tr), tr)
    table["exo_all"] = ridge_preds(raw, "T1", bc.AR + sum((c for _, c in exo.values()), []), g)
    return table


def t0_table() -> pd.DataFrame:
    return dmt.add_graph_ensemble(dmt.tabular_preds(nodes_like(True), "T0"), "T0")


# --------------------------------------------------------------------------- #
# tests on the loss differential d = e_a^2 - e_b^2
# --------------------------------------------------------------------------- #
def cluster_var(u, groups):
    n = len(u)
    s = pd.Series(u).groupby(groups).sum().to_numpy()
    return float(s @ s) / n ** 2


def tstat(d, groups):
    v = cluster_var(d - d.mean(), groups)
    return d.mean() / np.sqrt(v) if v > 0 else np.nan


def wild(d, groups, weights, reps=REPS):
    rng = np.random.default_rng(SEED)
    codes, uniq = pd.factorize(groups)
    t_obs = tstat(d, codes)
    count = 0
    for _ in range(reps):
        w = rng.choice(weights, size=len(uniq))
        t_star = tstat(d * w[codes], codes)
        if not np.isnan(t_star) and abs(t_star) >= abs(t_obs):
            count += 1
    return (count + 1) / (reps + 1)


def twoway(d, c, m):
    u = d - d.mean()
    v = cluster_var(u, c) + cluster_var(u, m) - float(u @ u) / len(u) ** 2
    if v <= 0:
        v = max(cluster_var(u, c), cluster_var(u, m))
    df = min(len(set(c)), len(set(m))) - 1
    tt = d.mean() / np.sqrt(v)
    return float(2 * tdist.sf(abs(tt), df)), float(tt)


def dm_month(d, m):
    s = pd.Series(d).groupby(m).mean().to_numpy()
    n = len(s)
    var = s.var(ddof=0) / n
    stat = s.mean() / np.sqrt(var) * np.sqrt((n - 1) / n)  # HLN, h=1
    return float(2 * tdist.sf(abs(stat), n - 1))


def holm(p: pd.Series) -> pd.Series:
    order = p.sort_values().index
    m = len(p)
    adj, running = {}, 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (m - k) * p[i]))
        adj[i] = running
    return pd.Series(adj)[p.index]


PAIRS_T1 = [
    ("sig", "AR(1) vs zero", "ar1_only", "zero"),
    ("sig", "AR(1) vs TGN", "ar1_only", "tgn"),
    ("sig", "AR(1) vs TGN, no memory", "ar1_only", "tgn_no_memory"),
    ("sig", "AR(1) vs GCN", "ar1_only", "gcn"),
    ("sig", "AR(1) vs random forest", "ar1_only", "random_forest"),
    ("sig", "AR(1) vs 3-feat. ridge", "ar1_only", "ar_ridge"),
    ("sig", "3-feat. ridge vs zero", "ar_ridge", "zero"),
    ("sig", "TGN vs TGN, no memory", "tgn", "tgn_no_memory"),
    ("seasonal", "AR(1,12) vs AR(1)", "ar1_12", "ar1_only"),
    ("seasonal", "AR(1,12) vs GCN", "ar1_12", "gcn"),
    ("seasonal", "AR(1,12) vs TGN, no memory", "ar1_12", "tgn_no_memory"),
    ("seasonal", "AR(1,12) vs random forest", "ar1_12", "random_forest"),
    ("seasonal", "AR(1,12) vs TGN", "ar1_12", "tgn"),
    ("seasonal", "AR(1,12) vs 3-feat. ridge", "ar1_12", "ar_ridge"),
    ("exog", "AR ridge + news vs AR ridge", "exo_news", "exo_base"),
    ("exog", "AR ridge + weather vs AR ridge", "exo_weather", "exo_base"),
    ("exog", "AR ridge + GSCPI vs AR ridge", "exo_gscpi", "exo_base"),
    ("exog", "AR ridge + all vs AR ridge", "exo_all", "exo_base"),
]
PAIRS_T0 = [
    ("sig", "persistence vs GCN", "persistence", "gcn"),
    ("sig", "persistence vs 3-feat. ridge", "persistence", "ar_ridge"),
]


def run(table, target, pairs):
    rows = []
    for family, name, a, b in pairs:
        sub = table[["y", a, b]].dropna()
        y = sub["y"].to_numpy(float)
        d = (y - sub[a].to_numpy(float)) ** 2 - (y - sub[b].to_numpy(float)) ** 2
        c = np.array([ix[0] for ix in sub.index])
        m = np.array([ix[1] for ix in sub.index])
        p2, t2 = twoway(d, c, m)
        rec = {"target": target, "family": family, "pair": f"{target}: {name}", "n": len(y),
               "countries": len(set(c)), "months": len(set(m)),
               "r2_a": r2_score(y, sub[a]), "r2_b": r2_score(y, sub[b]),
               "better": a if d.mean() < 0 else b,
               "p_country": wild(d, c, np.array([-1.0, 1.0])),
               "p_month": wild(d, m, WEBB),
               "p_twoway": p2, "t_twoway": t2,
               "p_dm_month": dm_month(d, m)}
        rows.append(rec)
        print(f"{rec['pair']:42s} R2 {rec['r2_a']:+.3f} vs {rec['r2_b']:+.3f}  "
              f"country {rec['p_country']:.3f}  month {rec['p_month']:.3f}  "
              f"two-way {rec['p_twoway']:.3f}  DM(month) {rec['p_dm_month']:.3f}", flush=True)
    return rows


def rolling_table() -> pd.DataFrame:
    """Pooled predictions of the per-origin refits of seasonal_and_rolling.rolling
    (same blocks, same protocol), so the month-level tests have about 22 months."""
    import seasonal_and_rolling as sar
    f = sar.add_lags(dmt.load_nodes())
    months = sorted(f["month"].unique())
    first = months.index(bc.TRAIN[0])
    parts = []
    for start in range(13, len(months), sar.HORIZON):
        block = months[start:start + sar.HORIZON]
        val = months[max(start - sar.HORIZON, 0):start]
        train = months[first:max(start - sar.HORIZON, 0)]
        if len(block) < 2 or len(train) < sar.MIN_TRAIN_MONTHS:
            continue
        valid = f["T1"].notna()
        tr, va, te = (f["month"].isin(m) & valid for m in (train, val, block))
        idx = pd.MultiIndex.from_frame(f.loc[te, ["host_country_id", "month"]])
        parts.append(pd.DataFrame({
            "y": f.loc[te, "T1"].to_numpy(float), "zero": 0.0,
            "ar1_only": sar.ridge_pred(f, "T1", ["T1_lag1"], tr, va, te),
            "ar1_12": sar.ridge_pred(f, "T1", ["T1_lag1", "T1_lag12"], tr, va, te),
            "ar_ridge": sar.ridge_pred(f, "T1", dmt.IMPROVED, tr, va, te),
            "random_forest": sar.rf_pred(f, "T1", dmt.IMPROVED, tr, va, te)}, index=idx))
    return pd.concat(parts)


PAIRS_ROLL = [
    ("rolling", "AR(1) vs zero", "ar1_only", "zero"),
    ("rolling", "AR(1,12) vs zero", "ar1_12", "zero"),
    ("rolling", "AR(1,12) vs AR(1)", "ar1_12", "ar1_only"),
    ("rolling", "AR(1,12) vs 3-feat. ridge", "ar1_12", "ar_ridge"),
    ("rolling", "AR(1,12) vs random forest", "ar1_12", "random_forest"),
    ("rolling", "AR(1) vs 3-feat. ridge", "ar1_only", "ar_ridge"),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = run(t1_table(), "T1", PAIRS_T1) + run(t0_table(), "T0", PAIRS_T0)
    rows += run(rolling_table(), "T1 rolling 2023-02..2024-12", PAIRS_ROLL)
    out = pd.DataFrame(rows)
    # Holm over the 16 pairs the paper tests (Tables sig and seasonal, T0 corrected),
    # separately over the 4 exogenous contrasts and over the 6 rolling-origin pairs.
    for proc in ("p_country", "p_month", "p_twoway", "p_dm_month"):
        out[f"{proc}_holm"] = np.nan
        for fam_mask in (out["family"].isin(["sig", "seasonal"]), out["family"] == "exog",
                         out["family"] == "rolling"):
            out.loc[fam_mask, f"{proc}_holm"] = holm(out.loc[fam_mask, proc]).to_numpy()
    out["p_max"] = out[["p_country", "p_month", "p_twoway", "p_dm_month"]].max(axis=1)
    out.to_csv(OUT / "dependence_tests.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(out.round(3).drop(columns=["t_twoway"]).to_string(index=False))
    print(f"-> {OUT / 'dependence_tests.csv'}")


if __name__ == "__main__":
    main()
