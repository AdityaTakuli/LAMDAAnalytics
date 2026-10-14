#!/usr/bin/env python3
"""Extended panel, 2021-2024: does adding countries buy statistical power?

On the 41-country panel (processed_v3), for each basket and both baskets
pooled, and for the original 18 countries versus all 41, report test-2024 and
pooled rolling (2023-2024) R^2 for the key models, plus the paired
country-cluster bootstrap interval and minimum detectable difference (MDD,
80% power) for two contrasts:
  * own lags AR(1,12) vs all seven features
  * all seven features vs trade proxies only (does adding the exogenous channels help?)
Models are refit on each scope separately (same harness as tabular_analyses).

Writes results/benchmark/extended_2021_2024/power_18_vs_41.csv
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
V3 = BASE / "data/four_year_2021_2024/processed_v3"
OUT = BASE.parent / "results" / "benchmark" / "extended_2021_2024"
ORIGINAL = set(pd.read_csv(BASE / "data/four_year_2021_2024/processed/nodes_monthly.csv")["host_country_id"])
RNG = np.random.default_rng(42)


def frames() -> dict[str, pd.DataFrame]:
    out = {}
    for basket in ("semiconductor", "consumer"):
        tab.NODES = V3 / basket / "nodes_monthly.csv"
        f = tab.load()
        f["basket"], f["is_consumer"] = basket, float(basket == "consumer")
        out[basket] = f
    return out


def boot(y, a, b, cl, reps=3000):
    uniq = np.unique(cl)
    idx = {c: np.where(cl == c)[0] for c in uniq}
    d = []
    for _ in range(reps):
        k = np.concatenate([idx[c] for c in RNG.choice(uniq, len(uniq))])
        d.append(r2_score(y[k], a[k]) - r2_score(y[k], b[k]))
    d = np.array(d)
    return np.percentile(d, 2.5), np.percentile(d, 97.5), 2.8 * d.std(ddof=1)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fr = frames()
    data = {"semiconductor": fr["semiconductor"], "consumer": fr["consumer"],
            "both baskets": pd.concat(fr.values(), ignore_index=True)}
    rows = []
    for dname, f0 in data.items():
        for scope in ("original 18", "all 41"):
            f = f0[f0["host_country_id"].isin(ORIGINAL)] if scope == "original 18" else f0
            f = f.reset_index(drop=True)
            for target in ("T0", "T1"):
                extra = ["is_consumer"] if dname == "both baskets" else []
                sets = {"ar": [f"{target}_lag1", f"{target}_lag12", *extra], "all7": [*tab.ALL7, *extra],
                        "trade": [*tab.TRADE, *extra]}
                ok = f[target].notna() & np.isfinite(f[target])
                tr = f["month"].between(*tab.TRAIN) & ok
                te = f["month"].between(*tab.TEST) & ok
                y = f.loc[te, target].to_numpy(float)
                cl = f.loc[te, "host_country_id"].to_numpy()
                p = {k: tab.fit_predict(f, v, target, tr, te) for k, v in sets.items()}
                rec = {"data": dname, "scope": scope, "target": target, "n_test": len(y),
                       "countries": len(set(cl)), "persistence": r2_score(y, f.loc[te, f"{target}_lag1"].fillna(0)),
                       "ar1_12": r2_score(y, p["ar"]), "all_seven": r2_score(y, p["all7"]),
                       "trade_only": r2_score(y, p["trade"])}
                for name, a, b in (("ar_minus_all7", "ar", "all7"), ("all7_minus_trade", "all7", "trade")):
                    lo, hi, mdd = boot(y, p[a], p[b], cl)
                    wp, _ = wild_cluster_boot((y - p[a]) ** 2 - (y - p[b]) ** 2, cl)
                    rec.update({f"{name}": r2_score(y, p[a]) - r2_score(y, p[b]), f"{name}_ci": f"[{lo:+.3f}, {hi:+.3f}]",
                                f"{name}_mdd": mdd, f"{name}_wild_p": wp})
                rows.append(rec)
                print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()}, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "power_18_vs_41.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
