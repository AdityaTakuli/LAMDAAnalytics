"""Phase 0.3 (BLOCKING) - full ladder under overlap-free targets + mechanism.

Answers the reviewer's blocking requests before Phase 1:
  (A) lag-1 autocorrelation of the T0, T1, T4 targets  -> the one-line mechanism.
  (B) full tabular/linear/tree ladder under T0, T1, T4 (regression + classification),
      to test whether the T0 rankings are preserved, inverted, or noise.
  (C) the four exogenous-channel ablations under T1.
  (D) per-origin persistence R^2 across expanding windows (regime vs skill).

Graph models (GCN/TGN/TGN-no-memory) under T1 are run separately in
graph_benchmark_t1.py because they are slow.

Phase 0.8: univariate AR(1) decomposition of residual T1 skill lives in
ar1_decomposition.py (own-lag vs own-lag+flow, wild cluster p).

Targets (per country, V = inbound_flow_usd, fut = V_{T+1}):
  T0 = (fut - med12)/med12         rolling-median contraction (shared denom 11/12)
  T1 = log(fut / V_T)              one-step log return          (overlap-free)
  T4 = log(fut / V_{T-11})         year-over-year log return    (overlap-free, smoother)

Classification labels:
  T0 -> y = 1 if c_T < -0.20                          (test prevalence 0.104)
  T1,T4 -> threshold = train 10.4th percentile of the target (prevalence-matched)
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, r2_score

DATA = Path(__file__).parent / "data/four_year_2021_2024/processed/nodes_monthly.csv"
TRAIN, VALID, TEST = ("2021-12", "2022-12"), ("2023-01", "2023-12"), ("2024-01", "2024-11")
TAU0 = 0.20
PREVALENCE = 0.104
PAPER = ["inventory_days_proxy", "trade_delay_proxy", "news_vol_7d", "neg_tone_frac_3d",
         "strike_flag_7d", "weather_anomaly_7d", "global_risk"]
PROTECTIVE = "inventory_days_proxy"


def load() -> pd.DataFrame:
    f = pd.read_csv(DATA)
    f["month"] = f["month"].astype(str)
    f = f.sort_values(["host_country_id", "month"]).reset_index(drop=True)
    v = f.groupby("host_country_id", sort=False)["inbound_flow_usd"]
    fut = v.shift(-1)
    med12 = v.transform(lambda s: s.rolling(12, min_periods=1).median())
    v_lag11 = v.shift(11)

    f["T0"] = (fut - med12) / med12.replace(0, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        pos = f["inbound_flow_usd"] > 0
        f["T1"] = np.log(fut / f["inbound_flow_usd"].where(pos))
        f.loc[fut <= 0, "T1"] = np.nan
        f["T4"] = np.log(fut / v_lag11.where(v_lag11 > 0))
        f.loc[fut <= 0, "T4"] = np.nan

    f["flow_ratio"] = f["inventory_days_proxy"] / 30.0
    grp = f.groupby("host_country_id", sort=False)
    f["flow_ratio_lag1"] = grp["flow_ratio"].shift(1)
    f["contraction_lag1"] = grp["T0"].shift(1)
    return f


def win(f, bounds):
    return f["month"].between(bounds[0], bounds[1])


# --------------------------------------------------------------------------- #
# (A) autocorrelation
# --------------------------------------------------------------------------- #
def autocorr(f: pd.DataFrame, target: str) -> float:
    lag = f.groupby("host_country_id", sort=False)[target].shift(1)
    ok = f[target].notna() & lag.notna()
    return float(np.corrcoef(f.loc[ok, target], lag[ok])[0, 1])


# --------------------------------------------------------------------------- #
# labels
# --------------------------------------------------------------------------- #
def label_series(f: pd.DataFrame, target: str) -> pd.Series:
    if target == "T0":
        return (f["T0"] < -TAU0).astype(float)
    thr = float(f.loc[win(f, TRAIN) & f[target].notna(), target].quantile(PREVALENCE))
    return (f[target] < thr).astype(float)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def score_row(y, pred, labels_eval) -> dict:
    r2 = r2_score(y, pred) if np.std(y) > 1e-12 else float("nan")
    dir_ = float(np.mean(np.sign(pred) == np.sign(y)))
    if labels_eval.min() == labels_eval.max():
        pr = float(labels_eval.mean())
    else:
        pr = float(average_precision_score(labels_eval, -pred))
    return {"R2": r2, "Dir": dir_, "PR": pr}


# --------------------------------------------------------------------------- #
# feature matrices
# --------------------------------------------------------------------------- #
def standardize(f, features, tr):
    m = f[features].astype(float).replace([np.inf, -np.inf], np.nan)
    if PROTECTIVE in features:
        m[PROTECTIVE] = -m[PROTECTIVE]
    mean = m[tr].mean()
    scale = m[tr].std().replace(0.0, 1.0).fillna(1.0)
    return ((m - mean) / scale).fillna(0.0)


def grid(space):
    keys = list(space)
    return [dict(zip(keys, c)) for c in product(*space.values())]


def tune_reg(x_tr, y_tr, x_val, y_val, est, cfgs):
    best, bv = None, -np.inf
    for c in cfgs:
        m = est(**c).fit(x_tr, y_tr)
        s = r2_score(y_val, m.predict(x_val)) if np.std(y_val) > 1e-12 else -np.inf
        if s > bv:
            bv, best = s, m
    return best


def tune_clf(x_tr, c_tr, x_val, c_val, est, cfgs):
    best, bv = None, -np.inf
    if c_tr.min() == c_tr.max():
        return None
    for c in cfgs:
        m = est(**c).fit(x_tr, c_tr)
        p = m.predict_proba(x_val)[:, 1]
        s = average_precision_score(c_val, p) if c_val.max() > 0 else -np.inf
        if s > bv:
            bv, best = s, m
    return best


RIDGE = grid({"alpha": [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]})
RF_R = grid({"n_estimators": [300], "max_depth": [2, 3, 4, None], "min_samples_leaf": [1, 5, 10], "random_state": [0]})
LGB_R = grid({"n_estimators": [100, 300], "num_leaves": [7, 15, 31], "learning_rate": [0.02, 0.05], "min_child_samples": [5, 20], "verbosity": [-1], "random_state": [0]})
LOGIT = grid({"C": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0], "max_iter": [2000], "class_weight": ["balanced"]})
RF_C = grid({"n_estimators": [300], "max_depth": [2, 3, 4, None], "min_samples_leaf": [1, 5, 10], "class_weight": ["balanced"], "random_state": [0]})
LGB_C = grid({"n_estimators": [100, 300], "num_leaves": [7, 15, 31], "learning_rate": [0.02, 0.05], "min_child_samples": [5, 20], "class_weight": ["balanced"], "verbosity": [-1], "random_state": [0]})


def ladder(f: pd.DataFrame, target: str) -> None:
    labels = label_series(f, target)
    valid = f[target].notna()
    tr = win(f, TRAIN) & valid
    va = win(f, VALID) & valid
    te = win(f, TEST) & valid

    y_te = f.loc[te, target].to_numpy(float)
    lab_te = labels[te].to_numpy(int)
    prev = float(lab_te.mean())
    print(f"\n#### Target {target}: test n={int(te.sum())}, positives={int(lab_te.sum())}, prevalence={prev:.3f}")
    head = f"{'model':22s} {'R2':>7s} {'Dir':>6s} {'PR-AUC':>7s}"
    print(head); print("-" * len(head))

    # naive predictors (continuous)
    grp = f.groupby("host_country_id", sort=False)[target]
    naive = {
        "Zero": np.zeros(int(te.sum())),
        "Persistence": grp.shift(1)[te].fillna(0.0).to_numpy(float),
        "Seasonal (T-11)": grp.shift(11)[te].fillna(0.0).to_numpy(float),
        "Per-country median": f.loc[te, "host_country_id"].map(
            f[tr].groupby("host_country_id")[target].median()).fillna(f.loc[tr, target].median()).to_numpy(float),
        "Global median": np.full(int(te.sum()), float(f.loc[tr, target].median())),
    }
    for name, pred in naive.items():
        s = score_row(y_te, pred, lab_te)
        print(f"{name:22s} {s['R2']:7.3f} {s['Dir']:6.3f} {s['PR']:7.3f}")

    # fitted regressors -> also give PR via -prediction
    for name, feats, est, cfgs in [
        ("Ridge (paper 7)", PAPER, Ridge, RIDGE),
        ("AR ridge (3)", ["flow_ratio", "contraction_lag1", "flow_ratio_lag1"], Ridge, RIDGE),
        ("Random forest", ["flow_ratio", "contraction_lag1", "flow_ratio_lag1"], RandomForestRegressor, RF_R),
        ("LightGBM", ["flow_ratio", "contraction_lag1", "flow_ratio_lag1"], LGBMRegressor, LGB_R),
    ]:
        z = standardize(f, feats, tr)
        model = tune_reg(z[tr].to_numpy(), f.loc[tr, target].to_numpy(float),
                         z[va].to_numpy(), f.loc[va, target].to_numpy(float), est, cfgs)
        pred = model.predict(z[te].to_numpy())
        s = score_row(y_te, pred, lab_te)
        print(f"{name:22s} {s['R2']:7.3f} {s['Dir']:6.3f} {s['PR']:7.3f}")

    # fitted classifiers -> PR-AUC from predict_proba (R2/Dir not meaningful)
    for name, feats, est, cfgs in [
        ("Logistic", ["flow_ratio", "contraction_lag1", "flow_ratio_lag1"], LogisticRegression, LOGIT),
        ("RF classifier", ["flow_ratio", "contraction_lag1", "flow_ratio_lag1"], RandomForestClassifier, RF_C),
        ("LightGBM classifier", ["flow_ratio", "contraction_lag1", "flow_ratio_lag1"], LGBMClassifier, LGB_C),
    ]:
        z = standardize(f, feats, tr)
        model = tune_clf(z[tr].to_numpy(), labels[tr].to_numpy(int),
                         z[va].to_numpy(), labels[va].to_numpy(int), est, cfgs)
        if model is None:
            print(f"{name:22s} {'--':>7s} {'--':>6s} {'--':>7s}")
            continue
        proba = model.predict_proba(z[te].to_numpy())[:, 1]
        pr = average_precision_score(lab_te, proba) if lab_te.max() > 0 else prev
        print(f"{name:22s} {'--':>7s} {'--':>6s} {pr:7.3f}")


# --------------------------------------------------------------------------- #
# (C) exogenous ablations under a target
# --------------------------------------------------------------------------- #
def exogenous(f: pd.DataFrame, target: str) -> None:
    from preprocessing_exogenous import (add_gscpi_dynamics, add_news_normalized, add_weather_seasonal)
    valid = f[target].notna()
    tr = win(f, TRAIN) & valid
    va = win(f, VALID) & valid
    te = win(f, TEST) & valid
    base = ["flow_ratio", "contraction_lag1", "flow_ratio_lag1"]
    labels = label_series(f, target)

    print(f"\n#### Exogenous ablation on top of AR, target {target} (test)")
    head = f"{'model':26s} {'R2':>7s} {'PR-AUC':>7s}"
    print(head); print("-" * len(head))
    specs = [("AR baseline", base, lambda g, m: (g, [])),
             ("+ improved news", base, lambda g, m: (add_news_normalized(g, m), ["news_z", "news_surprise"])),
             ("+ seasonal weather", base, lambda g, m: (add_weather_seasonal(g, m), ["weather_temp_z", "weather_temp_absz"])),
             ("+ GSCPI dynamics", base, lambda g, m: (add_gscpi_dynamics(g, m), ["gscpi_change", "gscpi_lag1"])),
             ("+ all improved exog", base, None)]
    for name, feats, fn in specs:
        g = f.copy()
        extra = []
        if name == "+ all improved exog":
            g = add_news_normalized(g, tr); g = add_weather_seasonal(g, tr); g = add_gscpi_dynamics(g, tr)
            extra = ["news_z", "news_surprise", "weather_temp_z", "weather_temp_absz", "gscpi_change", "gscpi_lag1"]
        elif fn is not None:
            g, extra = fn(g, tr)
        cols = feats + extra
        z = standardize(g, cols, tr)
        model = Ridge(alpha=1.0).fit(z[tr].to_numpy(), g.loc[tr, target].to_numpy(float))
        pred = model.predict(z[te].to_numpy())
        y_te = g.loc[te, target].to_numpy(float)
        lab_te = labels[te].to_numpy(int)
        r2 = r2_score(y_te, pred)
        pr = average_precision_score(lab_te, -pred) if lab_te.max() > 0 else float(lab_te.mean())
        print(f"{name:26s} {r2:7.3f} {pr:7.3f}")


# --------------------------------------------------------------------------- #
# (D) per-origin persistence
# --------------------------------------------------------------------------- #
def per_origin_persistence(f: pd.DataFrame) -> None:
    months = sorted(f["month"].unique())
    print("\n#### Per-origin persistence R^2 (expanding origin, 6-month eval blocks)")
    print(f"{'eval window':20s} {'n':>4s} {'T0 persist R2':>14s} {'T1 persist R2':>14s}")
    print("-" * 56)
    for target_pair in [None]:
        pass
    for end in range(13, len(months) - 6 + 1, 6):
        ev = months[end:end + 6]
        row = {}
        for target in ("T0", "T1"):
            lag = f.groupby("host_country_id", sort=False)[target].shift(1)
            mask = f["month"].isin(ev) & f[target].notna() & lag.notna()
            y = f.loc[mask, target].to_numpy(float)
            p = lag[mask].to_numpy(float)
            row[target] = r2_score(y, p) if len(y) > 2 and np.std(y) > 1e-12 else float("nan")
            row["n"] = int(mask.sum())
        print(f"{ev[0]+'..'+ev[-1]:20s} {row['n']:4d} {row['T0']:14.3f} {row['T1']:14.3f}")


def main() -> None:
    f = load()
    print("=== (A) lag-1 autocorrelation of the target (within country, pooled) ===")
    for t in ("T0", "T1", "T4"):
        print(f"  {t}: {autocorr(f, t):+.3f}")

    print("\n=== (B) full ladder by target (test 2024) ===")
    for t in ("T0", "T1", "T4"):
        ladder(f, t)

    print("\n=== (C) exogenous ablations under T1 ===")
    exogenous(f, "T1")

    print("\n=== (D) per-origin persistence ===")
    per_origin_persistence(f)


if __name__ == "__main__":
    main()
