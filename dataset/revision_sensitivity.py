#!/usr/bin/env python3
"""How much do the conclusions depend on evaluating on revised trade data?

Comtrade serves only the latest vintage, and every number in the paper uses the vintage downloaded
in August-October 2026. A forecaster working in real time would see the most recent month V_T as a
first release, which may later be revised. The revision error e enters the inputs at the forecast
origin: the own lag r_{T-1} = log(V_T / V_{T-1}) becomes r_{T-1} + e, and the trade-proxy ratio
V_T / median(V_{T-3..T-1}) is multiplied by exp(e) before the proxies are clipped. Older months
(the 12-month lag, the median window) and the outcome r_T, scored after the fact, are taken as
revised.

The ten half-year evaluation (seven-year panel, both baskets) is re-run with e ~ N(0, s^2) added
to every test row's inputs, s = 2%, 5%, 10% and 20% in log terms, training on revised data as in
the paper. Each level is repeated with 20 draws.

Writes results/benchmark/data_validation/revision_sensitivity.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

import extended_rolling_blocks as pda  # noqa: E402
import tabular_analyses as tab  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "benchmark" / "data_validation"
LEVELS = (0.0, 0.02, 0.05, 0.10, 0.20)
DRAWS = 20
MODELS = {"own lags": ["T1_lag1", "T1_lag12", "is_consumer"],
          "seven features": [*tab.ALL7, "is_consumer"],
          "own lags + exogenous": ["T1_lag1", "T1_lag12", *tab.EXO, "is_consumer"]}


def frame() -> pd.DataFrame:
    f = pd.concat(pda.frames().values(), ignore_index=True)
    f = f.sort_values(["basket", "host_country_id", "month"]).reset_index(drop=True)
    v = f["inbound_flow_usd"].astype(float)
    g = v.where(v > 0).groupby([f["basket"], f["host_country_id"]])
    base = g.transform(lambda s: s.shift(1).rolling(3, min_periods=1).median())
    f["ratio"] = (v / base).where(base > 0, 1.0)
    inv = np.clip(30.0 * f["ratio"], 5.0, 90.0)
    ok = (v > 0) & base.notna()
    match = np.isclose(inv[ok], f.loc[ok, "inventory_days_proxy"], atol=1e-6).mean()
    print(f"rebuilt inventory proxy matches the panel on {100 * match:.1f}% of rows with a full window", flush=True)
    f["proxy_rebuilt"] = ok & np.isclose(inv, f["inventory_days_proxy"], atol=1e-6)
    return f


def run(f: pd.DataFrame, s: float, rng: np.random.Generator) -> dict:
    g = f.copy()
    test = np.zeros(len(g), bool)
    for start, end in pda.BLOCKS:
        test |= g["month"].between(start, end).to_numpy()
    e = rng.normal(0.0, s, len(g)) * test
    g["T1_lag1"] = g["T1_lag1"] + np.where(g["T1_lag1"].notna(), e, 0.0)
    # perturb the proxies only where they could be rebuilt exactly from the import series
    r = g["ratio"] * np.exp(np.where(g["proxy_rebuilt"], e, 0.0))
    g["inventory_days_proxy"] = np.where(g["proxy_rebuilt"], np.clip(30.0 * r, 5.0, 90.0), g["inventory_days_proxy"])
    g["trade_delay_proxy"] = np.where(g["proxy_rebuilt"], np.clip(30.0 * np.maximum(0.0, 1.0 - r), 0.0, 60.0), g["trade_delay_proxy"])
    ok = g["T1"].notna() & np.isfinite(g["T1"])
    out = {}
    for name, feats in MODELS.items():
        ys, ps = [], []
        for start, end in pda.BLOCKS:
            tr = g["month"].between(pda.FIRST, pda.prev_month(start)) & ok
            te = g["month"].between(start, end) & ok
            # the training rows keep revised inputs: rebuild them from f, perturb only the test rows
            h = g.copy()
            h.loc[tr, feats] = f.loc[tr, feats]
            ys.append(h.loc[te, "T1"].to_numpy(float))
            ps.append(tab.fit_predict(h, feats, "T1", tr, te))
        out[name] = r2_score(np.concatenate(ys), np.concatenate(ps))
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    f = frame()
    rows = []
    for s in LEVELS:
        rng = np.random.default_rng(2026)
        draws = [run(f, s, rng) for _ in range(1 if s == 0 else DRAWS)]
        d = pd.DataFrame(draws)
        rec = {"revision_sd": s, "draws": len(d)}
        for m in MODELS:
            rec[f"r2 {m}"] = d[m].mean()
            rec[f"r2 {m} min"] = d[m].min()
            rec[f"r2 {m} max"] = d[m].max()
        rec["own lags - seven"] = (d["own lags"] - d["seven features"]).mean()
        rec["exogenous gain"] = (d["own lags + exogenous"] - d["own lags"]).mean()
        rows.append(rec)
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()}, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "revision_sensitivity.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
