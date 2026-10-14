#!/usr/bin/env python3
"""Extended panel analyses: seven years (2019-2025), 41 countries, two baskets.

Longer evaluation than the paper's single test year:
  * ten expanding-origin half-year blocks, 2021H1..2025H2. Each model is refit
    on every month from the first valid target (2019-12) up to the month
    before the block (same ridge harness as tabular_analyses.fit_predict);
  * per-block and pooled out-of-sample R^2 for persistence, own lags, trade
    proxies, exogenous channels, all seven features, all seven + own lags;
  * paired country-cluster bootstrap interval, minimum detectable difference
    (80% power) and wild-cluster p-value on the pooled out-of-sample rows for
      own lags AR(1,12) vs all seven features,
      all seven features vs trade proxies only (do the exogenous channels add anything?),
      AR(1,12) + exogenous vs AR(1,12);
  * the same on both baskets pooled (basket indicator added).
Targets: T1 (one-step log change, primary) and T0 (rolling-median contraction).
For T0 the result is reported with and without Myanmar, whose tiny
semiconductor flows produce contraction values above 6.

Writes results/benchmark/extended_2019_2025/{rolling_blocks.csv, rolling_pooled.csv, pooled_tests.csv}.
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

BASE = Path(__file__).resolve().parent
V4 = BASE / "data/four_year_2021_2024/processed_v4"
OUT = BASE.parent / "results" / "benchmark" / "extended_2019_2025"
FIRST = "2019-12"
BLOCKS = [(f"{y}-01", f"{y}-06") for y in range(2021, 2026)] + [(f"{y}-07", f"{y}-12") for y in range(2021, 2026)]
BLOCKS.sort()
RNG = np.random.default_rng(2026)
MYANMAR = "country_104"


def frames() -> dict[str, pd.DataFrame]:
    out = {}
    for basket in ("semiconductor", "consumer"):
        tab.NODES = V4 / basket / "nodes_monthly.csv"
        f = tab.load()
        f["basket"], f["is_consumer"] = basket, float(basket == "consumer")
        out[basket] = f
    return out


def prev_month(m: str) -> str:
    return str(pd.Period(m, freq="M") - 1)


def rolling(f: pd.DataFrame, feats: list[str] | str, target: str):
    ok = f[target].notna() & np.isfinite(f[target])
    ys, ps, cl, bl = [], [], [], []
    for start, end in BLOCKS:
        tr = f["month"].between(FIRST, prev_month(start)) & ok
        te = f["month"].between(start, end) & ok
        if te.sum() == 0:
            continue
        y = f.loc[te, target].to_numpy(float)
        p = f.loc[te, f"{target}_lag1"].fillna(0.0).to_numpy(float) if feats == "persistence" else tab.fit_predict(f, feats, target, tr, te)
        ys.append(y)
        ps.append(p)
        cl.append(f.loc[te, "host_country_id"].to_numpy())
        bl.append(np.full(len(y), f"{start[:4]}H{'1' if start[5:] == '01' else '2'}"))
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(cl), np.concatenate(bl)


def boot(y, a, b, cl, reps=3000):
    uniq = np.unique(cl)
    idx = {c: np.where(cl == c)[0] for c in uniq}
    d = np.array([r2_score(y[k], a[k]) - r2_score(y[k], b[k])
                  for k in (np.concatenate([idx[c] for c in RNG.choice(uniq, len(uniq))]) for _ in range(reps))])
    return np.percentile(d, 2.5), np.percentile(d, 97.5), 2.8 * d.std(ddof=1)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fr = frames()
    data = {"semiconductor": fr["semiconductor"], "consumer": fr["consumer"],
            "both baskets": pd.concat(fr.values(), ignore_index=True)}
    block_rows, pooled_rows, tests = [], [], []
    for dname, f0 in data.items():
        for target in ("T1", "T0"):
            scopes = [("all 41", f0)]
            if target == "T0":
                scopes.append(("excl. Myanmar", f0[f0["host_country_id"] != MYANMAR].reset_index(drop=True)))
            for scope, f in scopes:
                extra = ["is_consumer"] if dname == "both baskets" else []
                lag1, lag12 = f"{target}_lag1", f"{target}_lag12"
                sets = {"Persistence": "persistence", "Own lag (AR(1))": [lag1, *extra],
                        "Own lags 1, 12 (AR(1,12))": [lag1, lag12, *extra], "Trade proxies only": [*tab.TRADE, *extra],
                        "Exogenous only": [*tab.EXO, *extra], "All seven features": [*tab.ALL7, *extra],
                        "AR(1,12) + exogenous": [lag1, lag12, *tab.EXO, *extra],
                        "All seven + own lags": [*tab.ALL7, lag1, lag12, *extra]}
                preds = {}
                for name, feats in sets.items():
                    y, p, cl, bl = rolling(f, feats, target)
                    preds[name] = p
                    for b in pd.unique(bl):
                        m = bl == b
                        block_rows.append({"data": dname, "target": target, "scope": scope, "model": name, "block": b,
                                           "n": int(m.sum()), "r2": r2_score(y[m], p[m])})
                    per_block = [r2_score(y[bl == b], p[bl == b]) for b in pd.unique(bl)]
                    pooled_rows.append({"data": dname, "target": target, "scope": scope, "model": name, "n": len(y),
                                        "countries": len(set(cl)), "pooled_r2": r2_score(y, p),
                                        "blocks_positive": int(sum(v > 0 for v in per_block)), "blocks": len(per_block),
                                        "block_min": min(per_block), "block_max": max(per_block)})
                for a, b in (("Own lags 1, 12 (AR(1,12))", "All seven features"), ("All seven features", "Trade proxies only"),
                             ("AR(1,12) + exogenous", "Own lags 1, 12 (AR(1,12))")):
                    lo, hi, mdd = boot(y, preds[a], preds[b], cl)
                    wp, _ = wild_cluster_boot((y - preds[a]) ** 2 - (y - preds[b]) ** 2, cl)
                    tests.append({"data": dname, "target": target, "scope": scope, "A": a, "B": b, "n": len(y),
                                  "countries": len(set(cl)), "diff_r2": r2_score(y, preds[a]) - r2_score(y, preds[b]),
                                  "ci_lo": lo, "ci_hi": hi, "mdd_80": mdd, "wild_p": wp})
                print(f"done {dname} {target} {scope}", flush=True)
    pd.DataFrame(block_rows).to_csv(OUT / "rolling_blocks.csv", index=False)
    pooled = pd.DataFrame(pooled_rows)
    pooled.to_csv(OUT / "rolling_pooled.csv", index=False)
    t = pd.DataFrame(tests)
    t.to_csv(OUT / "pooled_tests.csv", index=False)
    pd.set_option("display.width", 250)
    print(pooled.round(3).to_string(index=False))
    print(t.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
