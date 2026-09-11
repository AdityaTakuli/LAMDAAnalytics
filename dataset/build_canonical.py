"""Phase 0.7 C.2 - assemble results/canonical_numbers.csv, the single source of truth.

One row per (target, model, metric, split) with: value, sd, n_seeds, epochs, script,
run_date. Every table in the paper is generated from this file. Tabular/naive numbers
are recomputed with the exact phase0_t1_ladder logic; graph numbers are read from the
canonical converged runs (results/phase07/graph_convergence_{T0,T1}.csv). T2/T3 denominator
ablations use the target_definition_robustness definitions on the same tuned-ridge protocol.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from phase0_t1_ladder import (LGB_R, PAPER, RF_R, RIDGE, load, standardize, tune_reg, win)
from preprocessing_exogenous import (add_gscpi_dynamics, add_news_normalized, add_weather_seasonal)

ROOT = Path(__file__).resolve().parent.parent
TRAIN, VALID, TEST = ("2021-12", "2022-12"), ("2023-01", "2023-12"), ("2024-01", "2024-11")
TODAY = dt.date.today().isoformat()
AR = ["flow_ratio", "contraction_lag1", "flow_ratio_lag1"]


def ar1_rows(f, target):
    """Univariate AR(1) vs AR(1)+flow (strict valid = target & lag present), matches ar1_decomposition."""
    valid = f[target].notna() & f[f"{target}_lag1"].notna()
    tr, va, te = win(f, TRAIN) & valid, win(f, VALID) & valid, win(f, TEST) & valid
    y_te = f.loc[te, target].to_numpy(float)
    out = []
    for name, feats in [("ar1_only", [f"{target}_lag1"]),
                        ("ar1_plus_flow", [f"{target}_lag1", "flow_ratio", "flow_ratio_lag1"])]:
        z = standardize(f, feats, tr)
        m = tune_reg(z[tr].to_numpy(), f.loc[tr, target].to_numpy(float),
                     z[va].to_numpy(), f.loc[va, target].to_numpy(float), Ridge, RIDGE)
        out.append((name, r2_score(y_te, m.predict(z[te].to_numpy()))))
    return out


def exogenous_rows(f, target):
    """Canonical exogenous ablation on the tuned AR baseline (matches ar_ridge)."""
    valid = f[target].notna()
    tr, va, te = win(f, TRAIN) & valid, win(f, VALID) & valid, win(f, TEST) & valid
    y_te = f.loc[te, target].to_numpy(float)
    specs = [("exo_ar_baseline", None, []),
             ("exo_+news", add_news_normalized, ["news_z", "news_surprise"]),
             ("exo_+weather", add_weather_seasonal, ["weather_temp_z", "weather_temp_absz"]),
             ("exo_+gscpi", add_gscpi_dynamics, ["gscpi_change", "gscpi_lag1"]),
             ("exo_+all", "all", ["news_z", "news_surprise", "weather_temp_z",
                                  "weather_temp_absz", "gscpi_change", "gscpi_lag1"])]
    out = []
    for name, fn, extra in specs:
        g = f.copy()
        if fn == "all":
            g = add_news_normalized(g, tr); g = add_weather_seasonal(g, tr); g = add_gscpi_dynamics(g, tr)
        elif fn is not None:
            g = fn(g, tr)
        cols = AR + extra
        z = standardize(g, cols, tr)
        m = tune_reg(z[tr].to_numpy(), g.loc[tr, target].to_numpy(float),
                     z[va].to_numpy(), g.loc[va, target].to_numpy(float), Ridge, RIDGE)
        out.append((name, r2_score(y_te, m.predict(z[te].to_numpy()))))
    return out


def denominator_ablation_rows(f):
    """T2/T3 denominator ablations (definitions from target_definition_robustness) on the
    canonical protocol: persistence as in tabular_r2, AR ridge tuned on 2023 like ar_ridge."""
    import target_definition_robustness as tdr
    g = tdr.load()[["host_country_id", "month", "T2", "contraction_lag1_disjoint"]]
    f = f.merge(g, on=["host_country_id", "month"], how="left", validate="one_to_one")
    f["T3"] = f["T0"]
    out = []
    for target, lag in (("T2", "contraction_lag1"), ("T3", "contraction_lag1_disjoint")):
        valid = f[target].notna()
        tr, va, te = win(f, TRAIN) & valid, win(f, VALID) & valid, win(f, TEST) & valid
        y_te = f.loc[te, target].to_numpy(float)
        persist = f.groupby("host_country_id", sort=False)[target].shift(1)[te].fillna(0.0)
        out.append((target, "persistence", r2_score(y_te, persist.to_numpy(float)), int(te.sum())))
        feats = ["flow_ratio", lag, "flow_ratio_lag1"]
        z = standardize(f, feats, tr)
        m = tune_reg(z[tr].to_numpy(), f.loc[tr, target].to_numpy(float),
                     z[va].to_numpy(), f.loc[va, target].to_numpy(float), Ridge, RIDGE)
        out.append((target, "ar_ridge", r2_score(y_te, m.predict(z[te].to_numpy())), int(te.sum())))
    return out


def tabular_r2(f, target):
    valid = f[target].notna()
    tr = win(f, TRAIN) & valid
    va = win(f, VALID) & valid
    te = win(f, TEST) & valid
    y_te = f.loc[te, target].to_numpy(float)
    grp = f.groupby("host_country_id", sort=False)[target]

    def r2(p):
        return r2_score(y_te, p) if np.std(y_te) > 1e-12 else float("nan")

    rows = []
    rows.append(("zero", r2(np.zeros(int(te.sum())))))
    rows.append(("persistence", r2(grp.shift(1)[te].fillna(0.0).to_numpy(float))))
    rows.append(("seasonal_T-11", r2(grp.shift(11)[te].fillna(0.0).to_numpy(float))))
    pcm = f.loc[te, "host_country_id"].map(f[tr].groupby("host_country_id")[target].median())
    rows.append(("per_country_median", r2(pcm.fillna(f.loc[tr, target].median()).to_numpy(float))))
    rows.append(("global_median", r2(np.full(int(te.sum()), float(f.loc[tr, target].median())))))

    for name, feats, est, cfgs in [("ridge_paper7", PAPER, Ridge, RIDGE),
                                   ("ar_ridge", AR, Ridge, RIDGE),
                                   ("random_forest", AR, RandomForestRegressor, RF_R),
                                   ("lightgbm", AR, LGBMRegressor, LGB_R)]:
        z = standardize(f, feats, tr)
        m = tune_reg(z[tr].to_numpy(), f.loc[tr, target].to_numpy(float),
                     z[va].to_numpy(), f.loc[va, target].to_numpy(float), est, cfgs)
        rows.append((name, r2(m.predict(z[te].to_numpy()))))
    return rows


def main():
    f = load()
    for t in ("T0", "T1", "T4"):
        f[f"{t}_lag1"] = f.groupby("host_country_id", sort=False)[t].shift(1)
    records = []
    for target in ("T0", "T1", "T4"):
        for model, val in tabular_r2(f, target):
            records.append({"target": target, "model": model, "metric": "R2", "split": "test2024",
                            "value": round(val, 6), "sd": "", "n_seeds": 1, "epochs": "",
                            "script": "phase0_t1_ladder.py", "run_date": TODAY})
        for model, val in ar1_rows(f, target):
            records.append({"target": target, "model": model, "metric": "R2", "split": "test2024",
                            "value": round(val, 6), "sd": "", "n_seeds": 1, "epochs": "",
                            "script": "ar1_decomposition.py", "run_date": TODAY})
    for target in ("T0", "T1"):
        for model, val in exogenous_rows(f, target):
            records.append({"target": target, "model": model, "metric": "R2", "split": "test2024",
                            "value": round(val, 6), "sd": "", "n_seeds": 1, "epochs": "",
                            "script": "preprocessing_exogenous.py", "run_date": TODAY})
    for target, model, val, n in denominator_ablation_rows(f):
        print(f"denominator ablation {target} {model}: R2={val:.4f} n_test={n}")
        records.append({"target": target, "model": model, "metric": "R2", "split": "test2024",
                        "value": round(val, 6), "sd": "", "n_seeds": 1, "epochs": "",
                        "script": "target_definition_robustness.py", "run_date": TODAY})

    # canonical graph numbers from converged runs
    for target in ("T0", "T1"):
        p = ROOT / f"results/phase07/graph_convergence_{target}.csv"
        if not p.exists():
            print(f"WARNING missing {p}; run graph_convergence.py {target}")
            continue
        g = pd.read_csv(p)
        for _, r in g.iterrows():
            records.append({"target": target, "model": r["model"], "metric": "R2", "split": "test2024",
                            "value": round(float(r["mean"]), 6), "sd": round(float(r["sd"]), 6),
                            "n_seeds": int(r["n_seeds"]), "epochs": int(r["epochs"]),
                            "script": "graph_convergence.py", "run_date": TODAY})

    # horizon sweep (H_k) as canonical target-design evidence
    hs = ROOT / "results/phase06/horizon_sweep.csv"
    if hs.exists():
        h = pd.read_csv(hs)
        for _, r in h.iterrows():
            tgt = r["target"]
            for metric, col in (("lag1_autocorr", "lag1_ac"), ("R2", "persistence"),
                                ("R2", "ar_ridge"), ("R2_origin_mean", "persist_origin_mean")):
                if col not in h.columns or pd.isna(r[col]):
                    continue
                model = {"lag1_ac": "target", "persistence": "persistence",
                         "ar_ridge": "ar_ridge", "persist_origin_mean": "persistence"}[col]
                records.append({"target": tgt, "model": model, "metric": metric, "split": "test2024",
                                "value": round(float(r[col]), 6), "sd": "", "n_seeds": 1, "epochs": "",
                                "script": "horizon_sweep.py", "run_date": TODAY})

    # phase 08: lag-12 autoregression and rolling-origin refits (seasonal_and_rolling.py)
    p8 = ROOT / "results/phase08"
    if (p8 / "seasonal_ar.csv").exists():
        s = pd.read_csv(p8 / "seasonal_ar.csv")
        for _, r in s[s["pair"] == "AR(1,12) vs AR(1)"].iterrows():
            records.append({"target": r["target"], "model": "ar1_12", "metric": "R2", "split": "test2024",
                            "value": round(float(r["r2_a"]), 6), "sd": "", "n_seeds": 1, "epochs": "",
                            "script": "seasonal_and_rolling.py", "run_date": TODAY})
    if (p8 / "rolling_origin_models.csv").exists():
        roll = pd.read_csv(p8 / "rolling_origin_models.csv")
        for target, g in roll.groupby("target"):
            for model in ("persistence", "ar1", "ar1_12", "ar_ridge_3feat", "random_forest"):
                records.append({"target": target, "model": f"roll_{model}", "metric": "R2_origin_mean",
                                "split": "origins_2023-02..2024-12", "value": round(float(g[model].mean()), 6),
                                "sd": "", "n_seeds": 1, "epochs": "", "script": "seasonal_and_rolling.py",
                                "run_date": TODAY})

    out = pd.DataFrame(records)
    out.to_csv(ROOT / "results/canonical_numbers.csv", index=False)
    # consistency check: no (target, model, metric, split) with two different values
    dup = out.groupby(["target", "model", "metric", "split"])["value"].nunique()
    clashes = dup[dup > 1]
    print(f"wrote results/canonical_numbers.csv ({len(out)} rows)")
    if len(clashes):
        print("VALUE CLASHES (same key, different value):")
        print(clashes)
    else:
        print("OK: every (target, model, metric, split) key has a single value.")
    with pd.option_context("display.width", 160, "display.max_rows", 200):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
