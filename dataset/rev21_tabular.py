#!/usr/bin/env python3
"""Paper-21 revision: tabular analyses requested by the reviewer.

(A) Non-overlapping target. The paper-21 ablation (Table XIV) is re-run under
    exactly its harness (ridge alpha=1.0, train 2021-12..2022-12, train-only
    standardisation, inventory sign-flipped, raw NaN -> 0 as in
    training.baselines) on two targets:
      T0 = paper-21 contraction c_{v,T} = (V_{T+1} - med12_T) / med12_T
      T1 = log(V_{T+1} / V_T), one-step log change, zero window overlap.
    Row 1 on T0 must reproduce val -0.054 / test 0.338 (harness check).

(B) Longer evaluation. Every feature set is refit at four expanding origins and
    scored on the next six-month block (2023H1, 2023H2, 2024H1, 2024-07..11), so
    two years are out of sample instead of one. A pooled out-of-sample R^2 over
    all four blocks is also reported, plus the refit-on-train+validation score
    on 2024 that paper 21 says was not done.

(C) Overlap diagnostics: lag-1 autocorrelation of T0 and T1, and persistence
    (y_hat = y_{T-1}) on each target.

Writes results/paper21_revision/{ablation_T0_T1.csv, rolling_blocks.csv,
rolling_pooled.csv, overlap_diagnostics.csv}.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from training.data import build_targets  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "paper21_revision"
NODES = Path(__file__).parent / "data/four_year_2021_2024/processed/nodes_monthly.csv"

TRAIN = ("2021-12", "2022-12")
VALID = ("2023-01", "2023-12")
TEST = ("2024-01", "2024-12")
TAU = 0.20
PROTECTIVE = "inventory_days_proxy"
TRADE = ["inventory_days_proxy", "trade_delay_proxy"]
NEWS = ["news_vol_7d", "neg_tone_frac_3d", "strike_flag_7d"]
EXO = [*NEWS, "weather_anomaly_7d", "global_risk"]
ALL7 = [*TRADE, *EXO]


def load() -> pd.DataFrame:
    nodes = pd.read_csv(NODES)
    nodes["month"] = nodes["month"].astype(str)
    frame, _ = build_targets(nodes, horizon=1, baseline_window=12, baseline_min_periods=12, taus=(TAU,))
    frame = frame.sort_values(["host_country_id", "month"]).reset_index(drop=True)
    v = frame["inbound_flow_usd"].astype(float)
    fut = frame["future_inbound_flow_usd"].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.log(fut / v.where(v > 0))
    t1[(fut <= 0) | fut.isna()] = np.nan
    frame["T0"] = frame["contraction"].where(frame["target_valid"])
    frame["T1"] = t1
    g = frame.groupby("host_country_id", sort=False)
    for t in ("T0", "T1"):
        frame[f"{t}_lag1"] = g[t].shift(1)
        frame[f"{t}_lag12"] = g[t].shift(12)
    frame["label"] = np.where(frame["T0"].notna(), (frame["T0"] < -TAU).astype(float), np.nan)
    return frame


def months_between(frame: pd.DataFrame, bounds) -> pd.Series:
    return frame["month"].between(bounds[0], bounds[1])


def design(frame: pd.DataFrame, feats: list[str], train_mask: pd.Series) -> np.ndarray:
    """training.baselines semantics: raw NaN -> 0, sign flip, train-only z-score."""
    m = frame[feats].astype(float).fillna(0.0).to_numpy(copy=True)
    if PROTECTIVE in feats:
        m[:, feats.index(PROTECTIVE)] *= -1.0
    mu = m[train_mask.to_numpy()].mean(axis=0)
    sd = m[train_mask.to_numpy()].std(axis=0)
    sd[sd < 1e-8] = 1.0
    return (m - mu) / sd


def feature_sets(target: str) -> dict[str, list[str]]:
    lag1, lag12 = f"{target}_lag1", f"{target}_lag12"
    return {
        "All seven features": ALL7,
        "Trade proxies only": TRADE,
        "Exogenous only (news, weather, GSCPI)": EXO,
        "News channels only": NEWS,
        "Own lag only (AR(1))": [lag1],
        "Own lags 1 and 12 (AR(1,12))": [lag1, lag12],
        "Trade proxies + own lag": [*TRADE, lag1],
        "All seven + own lags 1, 12": [*ALL7, lag1, lag12],
    }


def fit_predict(frame, feats, target, tr, rows):
    x = design(frame, feats, tr)
    y = frame[target].to_numpy(float)
    model = Ridge(alpha=1.0, random_state=7).fit(x[tr.to_numpy()], y[tr.to_numpy()])
    return model.predict(x[rows.to_numpy()])


def pr_auc(frame, feats, tr, rows):
    x = design(frame, feats, tr)
    lab = frame["label"].to_numpy(float)
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=7)
    clf.fit(x[tr.to_numpy()], lab[tr.to_numpy()].astype(int))
    score = clf.predict_proba(x[rows.to_numpy()])[:, 1]
    return float(average_precision_score(lab[rows.to_numpy()].astype(int), score))


def part_a(frame: pd.DataFrame) -> pd.DataFrame:
    out = []
    for target in ("T0", "T1"):
        ok = frame[target].notna()
        tr, va, te = (months_between(frame, b) & ok for b in (TRAIN, VALID, TEST))
        y_va = frame.loc[va, target].to_numpy(float)
        y_te = frame.loc[te, target].to_numpy(float)
        base = {
            "target": target, "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum()),
        }
        med = float(np.median(frame.loc[tr, target]))
        out.append({**base, "features": "Train median (constant)", "k": 0,
                    "val_r2": r2_score(y_va, np.full_like(y_va, med)),
                    "test_r2": r2_score(y_te, np.full_like(y_te, med)), "test_pr_auc": np.nan})
        pers_va = frame.loc[va, f"{target}_lag1"].fillna(0.0).to_numpy(float)
        pers_te = frame.loc[te, f"{target}_lag1"].fillna(0.0).to_numpy(float)
        out.append({**base, "features": "Persistence (y_{T-1}, no fit)", "k": 0,
                    "val_r2": r2_score(y_va, pers_va), "test_r2": r2_score(y_te, pers_te),
                    "test_pr_auc": np.nan})
        for name, feats in feature_sets(target).items():
            row = {**base, "features": name, "k": len(feats),
                   "val_r2": r2_score(y_va, fit_predict(frame, feats, target, tr, va)),
                   "test_r2": r2_score(y_te, fit_predict(frame, feats, target, tr, te)),
                   "test_pr_auc": pr_auc(frame, feats, tr & frame["label"].notna(), te) if target == "T0" else np.nan}
            out.append(row)
    table = pd.DataFrame(out)
    return table


BLOCKS = [("2023-01", "2023-06"), ("2023-07", "2023-12"), ("2024-01", "2024-06"), ("2024-07", "2024-12")]


def part_b(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, pooled = [], []
    for target in ("T0", "T1"):
        ok = frame[target].notna()
        sets = {"Train median": None, "Persistence": "persistence", **feature_sets(target)}
        preds_all: dict[str, list[np.ndarray]] = {name: [] for name in sets}
        ys: list[np.ndarray] = []
        for start, end in BLOCKS:
            tr = frame["month"].between(TRAIN[0], f"{int(start[:4]) - (1 if start[5:] == '01' else 0)}-"
                                         f"{'12' if start[5:] == '01' else '06'}") & ok
            te = frame["month"].between(start, end) & ok
            y = frame.loc[te, target].to_numpy(float)
            ys.append(y)
            rec = {"target": target, "block": f"{start}..{end}", "n": len(y),
                   "train_rows": int(tr.sum()), "train_end": frame.loc[tr, "month"].max()}
            for name, feats in sets.items():
                if feats is None:
                    p = np.full_like(y, float(np.median(frame.loc[tr, target])))
                elif feats == "persistence":
                    p = frame.loc[te, f"{target}_lag1"].fillna(0.0).to_numpy(float)
                else:
                    p = fit_predict(frame, feats, target, tr, te)
                preds_all[name].append(p)
                rec[name] = r2_score(y, p)
            rows.append(rec)
        y_all = np.concatenate(ys)
        prec = {"target": target, "evaluation": "pooled 2023-01..2024-12 (4 expanding origins)",
                "n": len(y_all)}
        for name in sets:
            p_all = np.concatenate(preds_all[name])
            prec[name] = r2_score(y_all, p_all)
        prec["blocks_won_by_best"] = None
        pooled.append(prec)
    return pd.DataFrame(rows), pd.DataFrame(pooled)


def part_refit(frame: pd.DataFrame) -> pd.DataFrame:
    """Refit on train+validation (2021-12..2023-12), score 2024 once."""
    out = []
    for target in ("T0", "T1"):
        ok = frame[target].notna()
        tr = frame["month"].between(TRAIN[0], VALID[1]) & ok
        tr_only = months_between(frame, TRAIN) & ok
        te = months_between(frame, TEST) & ok
        y = frame.loc[te, target].to_numpy(float)
        for name, feats in feature_sets(target).items():
            out.append({"target": target, "features": name,
                        "test_r2_train_only": r2_score(y, fit_predict(frame, feats, target, tr_only, te)),
                        "test_r2_refit_train_val": r2_score(y, fit_predict(frame, feats, target, tr, te)),
                        "n_test": len(y), "n_fit_refit": int(tr.sum())})
    return pd.DataFrame(out)


def part_c(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in ("T0", "T1"):
        lag = frame[f"{target}_lag1"]
        ok = frame[target].notna() & lag.notna()
        ac_all = float(np.corrcoef(frame.loc[ok, target], lag[ok])[0, 1])
        te = ok & months_between(frame, TEST)
        ac_te = float(np.corrcoef(frame.loc[te, target], lag[te])[0, 1])
        rows.append({"target": target, "lag1_autocorr_all": ac_all, "lag1_autocorr_test2024": ac_te,
                     "window_overlap": "11/12 of the median window" if target == "T0" else "0"})
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = load()
    a = part_a(frame)
    a.to_csv(OUT / "ablation_T0_T1.csv", index=False)
    print(a.round(3).to_string(index=False))
    blocks, pooled = part_b(frame)
    blocks.to_csv(OUT / "rolling_blocks.csv", index=False)
    pooled.to_csv(OUT / "rolling_pooled.csv", index=False)
    print(blocks.round(3).T.to_string())
    print(pooled.round(3).T.to_string())
    refit = part_refit(frame)
    refit.to_csv(OUT / "refit_train_val.csv", index=False)
    print(refit.round(3).to_string(index=False))
    c = part_c(frame)
    c.to_csv(OUT / "overlap_diagnostics.csv", index=False)
    print(c.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
