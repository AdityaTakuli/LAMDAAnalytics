#!/usr/bin/env python3
"""Paper-21 revision: why does pooled bilateral supervision fail?

The reviewer asks whether country-month aggregation hides pair-specific
contagion or whether there is simply not enough data/supervision. Three
diagnostics, all on the same directed pair-month table, target, and split as
the paper's bilateral benchmark (training.pair_data.build_pair_targets,
baseline_min_periods=12, train <=2022-12, validation 2023, test 2024):

(1) Signal: dispersion, tail weight, and lag-1 autocorrelation of the pair
    target against the country target; a value-weighted R^2 that scores skill
    on the pairs that carry most of the trade.
(2) Contagion: a ladder that adds, in turn, the pair's own lag, the pair's
    trade-ratio proxy, destination-country features, source-country
    (exporter-side) features, and an exporter-wide export shock (the source
    country's exports to every *other* panel country at T relative to its
    3-month median). If pairwise contagion were detectable, the exporter-side
    rows should add skill over the pair's own history.
(3) Data: a learning curve that fits on 10/25/50/100% of training pairs
    (5 random draws each) and on train+validation (2021-12..2023-12), scored
    on the same 2024 pair-months. Flat curves mean more data of this kind is
    not the binding constraint for these model classes.

Writes results/paper21_revision/{bilateral_signal.csv, bilateral_ladder.csv,
bilateral_learning_curve.csv}.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import load_config  # noqa: E402
from training.data import FEATURES, load_tables, validate_tables  # noqa: E402
from training.pair_data import attach_node_features, build_pair_targets  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "paper21_revision"
TRAIN = ("2021-12", "2022-12")
TRAIN_VAL = ("2021-12", "2023-12")
TEST = ("2024-01", "2024-12")
RNG = np.random.default_rng(7)


def build() -> pd.DataFrame:
    config = load_config("config.yaml")
    nodes, edges = load_tables(config)
    validate_tables(nodes, edges)
    pairs, stats = build_pair_targets(edges, baseline_min_periods=12)
    pairs = attach_node_features(pairs, nodes)
    pairs = pairs.sort_values(["source", "destination", "month"]).reset_index(drop=True)
    g = pairs.groupby(["source", "destination"], sort=False)
    v = pairs["trade_value_usd"].astype(float)
    pairs["pair_lag1"] = g["contraction"].shift(1)
    prior = g["trade_value_usd"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).median())
    pairs["pair_ratio"] = (v / prior.where(prior > 0)).clip(0, 10)
    pairs["pair_log_value"] = np.log1p(v.clip(lower=0))
    fut = pairs["future_trade_value_usd"].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pairs["pair_T1"] = np.log(fut / v.where(v > 0)).where(fut > 0)
    pairs["pair_T1_lag1"] = g["pair_T1"].shift(1)

    # exporter-wide shock: source's exports to all other panel destinations at T
    exp = pairs.groupby(["source", "month"])["trade_value_usd"].sum().rename("src_total").reset_index()
    pairs = pairs.merge(exp, on=["source", "month"], how="left")
    pairs["src_other"] = pairs["src_total"] - v
    pairs = pairs.sort_values(["source", "destination", "month"]).reset_index(drop=True)
    tot = pairs.drop_duplicates(["source", "month"])[["source", "month", "src_total"]].sort_values(["source", "month"])
    tot["src_med3"] = tot.groupby("source")["src_total"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).median())
    pairs = pairs.merge(tot[["source", "month", "src_med3"]], on=["source", "month"], how="left")
    pairs["exporter_shock"] = (pairs["src_other"] / (pairs["src_med3"] - 0.0).where(pairs["src_med3"] > 0)).clip(0, 10)
    pairs["valid"] = pairs["target_valid"].fillna(False).astype(bool)
    print("pair rows", len(pairs), "valid", int(pairs["valid"].sum()), "unique pairs", stats["unique_directed_pairs"])
    return pairs


def design(f: pd.DataFrame, cols: list[str], tr: pd.Series) -> np.ndarray:
    m = f[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
    mu, sd = m[tr.to_numpy()].mean(0), m[tr.to_numpy()].std(0)
    sd[sd < 1e-8] = 1.0
    return (m - mu) / sd


def fit_score(f, cols, target, tr, te, weights=None, alpha=1.0):
    x = design(f, cols, tr)
    y = f[target].to_numpy(float)
    model = Ridge(alpha=alpha).fit(x[tr.to_numpy()], y[tr.to_numpy()])
    p = model.predict(x[te.to_numpy()])
    yt = y[te.to_numpy()]
    out = {"r2": r2_score(yt, p)}
    if weights is not None:
        out["r2_value_weighted"] = r2_score(yt, p, sample_weight=weights)
    return out


def signal(f: pd.DataFrame) -> pd.DataFrame:
    nodes = pd.read_csv(Path(__file__).parent / "data/four_year_2021_2024/processed/nodes_monthly.csv")
    nodes = nodes[nodes["target_valid"].fillna(False).astype(bool)].sort_values(["host_country_id", "month"])
    nodes["lag"] = nodes.groupby("host_country_id")["contraction"].shift(1)
    rows = []
    for name, frame, y, lag in (
        ("country-month (panel target)", nodes, "contraction", "lag"),
        ("pair-month (bilateral target)", f[f["valid"]], "contraction", "pair_lag1"),
        ("pair-month one-step log change", f[f["pair_T1"].notna()], "pair_T1", "pair_T1_lag1"),
    ):
        vals = frame[y].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        ok = frame[[y, lag]].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append({"unit": name, "rows": len(vals), "std": vals.std(), "iqr": vals.quantile(.75) - vals.quantile(.25),
                     "excess_kurtosis": kurtosis(vals.clip(-5, 50)), "share_abs_gt_1": float((vals.abs() > 1).mean()),
                     "lag1_autocorr": float(np.corrcoef(ok[y], ok[lag])[0, 1])})
    v = f.loc[f["valid"], "trade_value_usd"]
    small = (v < v.quantile(0.5)).mean()
    rows.append({"unit": "pair rows below median pair value (share of trade)",
                 "rows": int((v < v.median()).sum()), "std": float(v[v < v.median()].sum() / v.sum())})
    return pd.DataFrame(rows)


def ladder(f: pd.DataFrame) -> pd.DataFrame:
    ok = f["valid"] & np.isfinite(f["contraction"])
    tr = f["month"].between(*TRAIN) & ok
    te = f["month"].between(*TEST) & ok
    w = f.loc[te, "baseline_trade_value_usd"].to_numpy(float)
    src = [f"src_{c}" for c in FEATURES]
    dst = [f"dst_{c}" for c in FEATURES]
    steps = [
        ("Pair own lag (AR(1))", ["pair_lag1"]),
        ("+ pair trade ratio, log value", ["pair_lag1", "pair_ratio", "pair_log_value"]),
        ("+ destination-country features (7)", ["pair_lag1", "pair_ratio", "pair_log_value", *dst]),
        ("+ source-country features (7)", ["pair_lag1", "pair_ratio", "pair_log_value", *dst, *src]),
        ("+ exporter-wide export shock", ["pair_lag1", "pair_ratio", "pair_log_value", *dst, *src, "exporter_shock"]),
        ("Exporter-side only (source features + shock)", [*src, "exporter_shock"]),
        ("Node features only (paper's pooled inputs, no lags)", [*src, *dst, "pair_log_value"]),
    ]
    yte = f.loc[te, "contraction"].to_numpy(float)
    med = float(np.median(f.loc[tr, "contraction"]))
    rows = [{"model": "Train median", "r2": r2_score(yte, np.full_like(yte, med))},
            {"model": "Persistence (pair lag)", "r2": r2_score(yte, f.loc[te, "pair_lag1"].fillna(0))}]
    for name, cols in steps:
        rows.append({"model": name, **fit_score(f, cols, "contraction", tr, te, alpha=10.0)})
    # same ladder on the non-overlapping pair target
    ok1 = f["pair_T1"].notna() & np.isfinite(f["pair_T1"])
    tr1 = f["month"].between(*TRAIN) & ok1
    te1 = f["month"].between(*TEST) & ok1
    for name, cols in steps:
        cols1 = [c if c != "pair_lag1" else "pair_T1_lag1" for c in cols]
        rows.append({"model": f"[pair log change] {name}", **fit_score(f, cols1, "pair_T1", tr1, te1, alpha=10.0)})
    # large pairs: top quartile by mean training-period value (most of the trade)
    pv = f.loc[f["month"].between(*TRAIN)].groupby(["source", "destination"])["trade_value_usd"].mean()
    big = set(pv[pv >= pv.quantile(0.75)].index)
    share = pv[pv >= pv.quantile(0.75)].sum() / pv.sum()
    is_big = pd.Series([k in big for k in zip(f["source"], f["destination"])], index=f.index)
    trb, teb = tr1 & is_big, te1 & is_big
    for name, cols in steps:
        cols1 = [c if c != "pair_lag1" else "pair_T1_lag1" for c in cols]
        rows.append({"model": f"[pair log change, top-quartile pairs = {share:.0%} of value] {name}",
                     **fit_score(f, cols1, "pair_T1", trb, teb, alpha=10.0), "n_test_subset": int(teb.sum())})
    out = pd.DataFrame(rows)
    out["n_test"] = int(te.sum())
    return out


def learning_curve(f: pd.DataFrame) -> pd.DataFrame:
    ok = f["valid"] & np.isfinite(f["contraction"])
    te = f["month"].between(*TEST) & ok
    src = [f"src_{c}" for c in FEATURES]
    dst = [f"dst_{c}" for c in FEATURES]
    sets = {"own lag + ratio": ["pair_lag1", "pair_ratio", "pair_log_value"],
            "full contagion set": ["pair_lag1", "pair_ratio", "pair_log_value", *dst, *src, "exporter_shock"]}
    pair_ids = f.loc[ok & f["month"].between(*TRAIN), ["source", "destination"]].drop_duplicates()
    rows = []
    ok1 = f["pair_T1"].notna() & np.isfinite(f["pair_T1"])
    te1 = f["month"].between(*TEST) & ok1
    sets1 = {k: [c if c != "pair_lag1" else "pair_T1_lag1" for c in v] for k, v in sets.items()}
    for frac in (0.1, 0.25, 0.5, 1.0):
        for rep in range(5 if frac < 1 else 1):
            keep = pair_ids.sample(frac=frac, random_state=int(RNG.integers(1e9)))
            keep_key = set(map(tuple, keep.to_numpy()))
            in_keep = pd.Series([k in keep_key for k in zip(f["source"], f["destination"])], index=f.index)
            tr = f["month"].between(*TRAIN) & ok & in_keep
            tr1 = f["month"].between(*TRAIN) & ok1 & in_keep
            for name, cols in sets.items():
                rows.append({"target": "pair contraction", "train": f"{int(frac * 100)}% of train pairs", "rep": rep,
                             "features": name, "train_rows": int(tr.sum()),
                             **fit_score(f, cols, "contraction", tr, te, alpha=10.0)})
                rows.append({"target": "pair log change", "train": f"{int(frac * 100)}% of train pairs", "rep": rep,
                             "features": name, "train_rows": int(tr1.sum()),
                             **fit_score(f, sets1[name], "pair_T1", tr1, te1, alpha=10.0)})
    tr = f["month"].between(*TRAIN_VAL) & ok
    tr1 = f["month"].between(*TRAIN_VAL) & ok1
    for name, cols in sets.items():
        rows.append({"target": "pair contraction", "train": "train+validation (25 months)", "rep": 0, "features": name,
                     "train_rows": int(tr.sum()), **fit_score(f, cols, "contraction", tr, te, alpha=10.0)})
        rows.append({"target": "pair log change", "train": "train+validation (25 months)", "rep": 0, "features": name,
                     "train_rows": int(tr1.sum()), **fit_score(f, sets1[name], "pair_T1", tr1, te1, alpha=10.0)})
    return pd.DataFrame(rows)


def main() -> int:
    f = build()
    s = signal(f)
    s.to_csv(OUT / "bilateral_signal.csv", index=False)
    print(s.round(3).to_string(index=False))
    lad = ladder(f)
    lad.to_csv(OUT / "bilateral_ladder.csv", index=False)
    print(lad.round(3).to_string(index=False))
    lc = learning_curve(f)
    lc.to_csv(OUT / "bilateral_learning_curve.csv", index=False)
    print(lc.groupby(["target", "train", "features"])[["train_rows", "r2"]].agg(["mean", "std"]).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
