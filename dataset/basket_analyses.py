#!/usr/bin/env python3
"""Consumer-basket analyses: the benchmark tabular evidence on the consumer basket and on both baskets pooled.

Per basket (semiconductor = the published panel, consumer = the new one) this
reruns, with the identical harness of tabular_analyses.py:
  * the feature-group ablation on the overlapping target T0 and the
    non-overlapping target T1 (test 2024), with persistence and own lags;
  * the four-origin rolling evaluation (pooled 2023-2024 R^2) and the refit
    on train+validation;
  * lag-1 autocorrelation of both targets.
Pooled over both baskets (36 country-basket series), the same models are fit
on the stacked rows with a basket indicator, and the paired country-cluster
bootstrap gives the minimum detectable difference (MDD) for "all seven
features vs own lags", so the power gain from the extra data is measured, not
assumed.

Writes results/benchmark/consumer/{ablation_by_basket.csv,
rolling_by_basket.csv, refit_by_basket.csv, overlap_by_basket.csv,
pooled_models.csv, power_comparison.csv}.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tabular_analyses as tab  # noqa: E402
from dm_tests import wild_cluster_boot  # noqa: E402

import os

BASE = Path(__file__).resolve().parent
# LAMDA_PANEL=processed_v3 and LAMDA_OUT=extended_2021_2024 rerun everything on the 41-country panel
V2 = BASE / "data/four_year_2021_2024" / os.environ.get("LAMDA_PANEL", "processed_v2")
OUT = BASE.parent / "results" / "benchmark" / os.environ.get("LAMDA_OUT", "consumer")
BASKETS = {"semiconductor": V2 / "semiconductor" / "nodes_monthly.csv",
           "consumer": V2 / "consumer" / "nodes_monthly.csv"}
RNG = np.random.default_rng(20261002)


def load_basket(path: Path) -> pd.DataFrame:
    tab.NODES = path
    return tab.load()


def per_basket() -> dict[str, pd.DataFrame]:
    frames, abl, roll, refit, over = {}, [], [], [], []
    for name, path in BASKETS.items():
        f = load_basket(path)
        frames[name] = f
        abl.append(tab.part_a(f).assign(basket=name))
        blocks, pooled = tab.part_b(f)
        roll.append(blocks.assign(basket=name))
        roll.append(pooled.assign(basket=name, block="pooled"))
        refit.append(tab.part_refit(f).assign(basket=name))
        over.append(tab.part_c(f).assign(basket=name))
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(abl).to_csv(OUT / "ablation_by_basket.csv", index=False)
    pd.concat(roll).to_csv(OUT / "rolling_by_basket.csv", index=False)
    pd.concat(refit).to_csv(OUT / "refit_by_basket.csv", index=False)
    pd.concat(over).to_csv(OUT / "overlap_by_basket.csv", index=False)
    return frames


def stacked(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for name, f in frames.items():
        g = f.copy()
        g["basket"] = name
        g["is_consumer"] = float(name == "consumer")
        g["country"] = g["host_country_id"]
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def fit_rows(f: pd.DataFrame, feats: list[str], target: str, tr: pd.Series, te: pd.Series) -> np.ndarray:
    return tab.fit_predict(f, feats, target, tr, te)


def boot_mdd(y, a, b, clusters, reps=3000):
    uniq = np.unique(clusters)
    idx = {c: np.where(clusters == c)[0] for c in uniq}
    obs = r2_score(y, a) - r2_score(y, b)
    draws = []
    for _ in range(reps):
        pick = np.concatenate([idx[c] for c in RNG.choice(uniq, len(uniq), replace=True)])
        draws.append(r2_score(y[pick], a[pick]) - r2_score(y[pick], b[pick]))
    draws = np.array(draws)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return obs, lo, hi, 2.8 * draws.std(ddof=1)


def pooled(frames: dict[str, pd.DataFrame]) -> None:
    rows, power = [], []
    for target in ("T0", "T1"):
        lag1, lag12 = f"{target}_lag1", f"{target}_lag12"
        sets = {"All seven features": [*tab.ALL7, "is_consumer"],
                "Trade proxies only": [*tab.TRADE, "is_consumer"],
                "Exogenous only": [*tab.EXO, "is_consumer"],
                "Own lags 1 and 12 (AR(1,12))": [lag1, lag12, "is_consumer"],
                "All seven + own lags": [*tab.ALL7, lag1, lag12, "is_consumer"]}
        for scope, f in (("semiconductor only", stacked({"semiconductor": frames["semiconductor"]})),
                         ("consumer only", stacked({"consumer": frames["consumer"]})),
                         ("both baskets pooled", stacked(frames))):
            ok = f[target].notna() & np.isfinite(f[target])
            tr = f["month"].between(*tab.TRAIN) & ok
            te = f["month"].between(*tab.TEST) & ok
            y = f.loc[te, target].to_numpy(float)
            preds = {k: fit_rows(f, v, target, tr, te) for k, v in sets.items()}
            pers = f.loc[te, lag1].fillna(0.0).to_numpy(float)
            rec = {"target": target, "scope": scope, "n_test": len(y), "series": int(f.loc[te, "series_id"].nunique()) if "series_id" in f else int(f.loc[te, ["country", "basket"]].drop_duplicates().shape[0]),
                   "Persistence": r2_score(y, pers)}
            rec.update({k: r2_score(y, p) for k, p in preds.items()})
            rows.append(rec)
            clusters = f.loc[te, "country"].to_numpy()
            a, b = preds["Own lags 1 and 12 (AR(1,12))"], preds["All seven features"]
            obs, lo, hi, mdd = boot_mdd(y, a, b, clusters)
            ea, eb = y - a, y - b
            wild_p, _ = wild_cluster_boot(ea ** 2 - eb ** 2, clusters)
            power.append({"target": target, "scope": scope, "n_test": len(y), "clusters": len(set(clusters)),
                          "contrast": "own lags (AR(1,12)) - all seven", "diff_r2": obs, "ci_lo": lo, "ci_hi": hi,
                          "mdd_80": mdd, "wild_p": wild_p})
    pd.DataFrame(rows).to_csv(OUT / "pooled_models.csv", index=False)
    pd.DataFrame(power).to_csv(OUT / "power_comparison.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(pd.DataFrame(power).round(3).to_string(index=False))


def main() -> int:
    frames = per_basket()
    for name in BASKETS:
        a = pd.read_csv(OUT / "ablation_by_basket.csv")
        print(f"\n== {name}")
        print(a[a.basket == name][["target", "features", "val_r2", "test_r2", "test_pr_auc", "n_test"]].round(3).to_string(index=False))
    r = pd.read_csv(OUT / "rolling_by_basket.csv")
    print(r[r.block == "pooled"].drop(columns=["block"]).round(3).T.to_string())
    print(pd.read_csv(OUT / "overlap_by_basket.csv").round(3).to_string(index=False))
    pooled(frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
