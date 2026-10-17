#!/usr/bin/env python3
"""Two-way (country x month) cluster-robust tests for every model comparison the paper reports.

The one-way wild cluster bootstrap treats rows as independent across months (when clustering by
country) or across countries (when clustering by month). The two-way test allows both dependences
at once (Cameron, Gelbach and Miller), with reference t(G-1), G the smaller number of clusters.
Each comparison is a mean squared-error differential d = e_b^2 - e_a^2 (positive when model a is
better); its mean, scaled by n / sum((y - mean y)^2), is the R^2 difference, so the 95% interval
is reported in R^2 units.

A. Ten half-year blocks, seven-year panel, r_{v,T}: both baskets and each basket.
B. Two-year test (2024-2025) and 2024 test of the 41-country panels: own lags against each graph
   model and against the seven-feature model.
C. Original 18-country panel, test 2024, r_{v,T}: own lags against the other models.
D. The missing-data treatments and country groups of robustness_checks.py (parts A and B).

Writes results/benchmark/inference/two_way_tests.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

import extended_rolling_blocks as pda  # noqa: E402
import graph_extended_summary as ges  # noqa: E402
import graph_logchange_tests as glt  # noqa: E402
import robustness_checks as rc  # noqa: E402
import tabular_analyses as tab  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "benchmark" / "inference"


def two_way(y: np.ndarray, pa: np.ndarray, pb: np.ndarray, country: np.ndarray, month: np.ndarray) -> dict:
    """R^2(a) - R^2(b) with its two-way cluster-robust t, p and 95% interval."""
    d = (y - pb) ** 2 - (y - pa) ** 2
    t, p = rc.two_way_t(d, country, month)
    scale = len(y) / np.sum((y - y.mean()) ** 2)
    se = d.mean() / t if t != 0 else np.nan
    g = min(len(set(country)), len(set(month)))
    q = stats.t.ppf(0.975, g - 1)
    diff = r2_score(y, pa) - r2_score(y, pb)
    return {"r2_a": r2_score(y, pa), "r2_b": r2_score(y, pb), "diff_r2": diff, "t": t, "p_two_way": p,
            "ci_lo": diff - q * se * scale, "ci_hi": diff + q * se * scale, "df": g - 1,
            "n": len(y), "countries": len(set(country)), "months": len(set(month))}


def part_a() -> list[dict]:
    f = rc.frames()
    rows = []
    for basket, sub in (("both", f), ("semiconductor", f[f["is_consumer"] == 0]), ("consumer", f[f["is_consumer"] == 1])):
        p = rc.run(sub.reset_index(drop=True))
        y = p["y"].to_numpy(float)
        bench = []
        for start, end in pda.BLOCKS:
            ok = sub["T1"].notna() & np.isfinite(sub["T1"])
            tr = sub["month"].between(pda.FIRST, pda.prev_month(start)) & ok
            te = sub["month"].between(start, end) & ok
            bench.append(np.full(int(te.sum()), sub.loc[tr, "T1"].mean()))
        p["train_mean"] = np.concatenate(bench)
        for a, b, label in (("ar", "all7", "own lags vs seven features"),
                            ("ar", "trade", "own lags vs trade proxies"),
                            ("ar_exo", "ar", "exogenous channels added to own lags"),
                            ("ar", "train_mean", "own lags vs training-mean forecast"),
                            ("all7", "train_mean", "seven features vs training-mean forecast")):
            rows.append({"evaluation": "ten half-year blocks, 2021-2025, 41 countries", "basket": basket,
                         "comparison": label, "a": a, "b": b,
                         **two_way(y, p[a].to_numpy(float), p[b].to_numpy(float), p["country"].to_numpy(), p["month"].to_numpy())})
        for a, b, label in (("ar", "all7", "own lags vs seven features, absolute error"),
                            ("ar_exo", "ar", "exogenous channels added to own lags, absolute error")):
            d = np.abs(y - p[b].to_numpy(float)) - np.abs(y - p[a].to_numpy(float))
            t, pv = rc.two_way_t(d, p["country"].to_numpy(), p["month"].to_numpy())
            rows.append({"evaluation": "ten half-year blocks, 2021-2025, 41 countries", "basket": basket, "comparison": label,
                         "a": a, "b": b, "mae_diff": float(d.mean()), "t": t, "p_two_way": pv, "n": len(y)})
        print(f"done A {basket}", flush=True)
    return rows


def part_d() -> list[dict]:
    """The missing-data treatments and the country groups of the robustness checks, two-way."""
    f = rc.frames()
    base = rc.run(f)
    variants = {
        "drop the five countries with any unreported month": rc.run(f[~f["host_country_id"].isin(rc.GAP_COUNTRIES)].reset_index(drop=True)),
        "drop rows within 12 months after a gap": rc.run(f[~f["near_gap"]].reset_index(drop=True)),
        "drop June 2025 (interpolated news)": rc.run(f[f["month"] != "2025-06"].reset_index(drop=True)),
        "drop Hong Kong (sparse news coverage)": rc.run(f[f["host_country_id"] != rc.HONG_KONG].reset_index(drop=True)),
        "score 2021-2024 only": base[base["month"] < "2025-01"].reset_index(drop=True),
        "missing-lag indicators instead of zero fill": rc.run(f, extra=["T1_lag1_missing", "T1_lag12_missing"]),
    }
    size = f.groupby("host_country_id")["inbound_flow_usd"].median()
    terc = pd.qcut(size.rank(method="first"), 3, labels=["smallest third", "middle third", "largest third"])
    groups = {
        "original 18 countries": base["country"].isin(rc.ORIGINAL18),
        "23 added countries": ~base["country"].isin(rc.ORIGINAL18),
        "Asia-Pacific": base["country"].isin(rc.ASIA_PACIFIC),
        "Europe and Middle East": ~base["country"].isin(rc.ASIA_PACIFIC | rc.AMERICAS),
        **{f"importer size: {t}": base["country"].map(terc) == t for t in ("largest third", "middle third", "smallest third")},
    }
    for label, m in groups.items():
        variants[f"group: {label}"] = base[m.to_numpy()].reset_index(drop=True)
    rows = []
    for label, p in variants.items():
        y = p["y"].to_numpy(float)
        for a, b, comp in (("ar", "all7", "own lags vs seven features"), ("ar_exo", "ar", "exogenous channels added to own lags")):
            rows.append({"evaluation": "ten half-year blocks, robustness variant", "basket": "both", "variant": label,
                         "comparison": comp, "a": a, "b": b,
                         **two_way(y, p[a].to_numpy(float), p[b].to_numpy(float), p["country"].to_numpy(), p["month"].to_numpy())})
    print("done D", flush=True)
    return rows


def part_b() -> list[dict]:
    rows = []
    for setting, (panel, out, train, test) in ges.SETTINGS.items():
        parts = {}
        for basket in ("semiconductor", "consumer"):
            m, _ = ges.table(panel, out, basket, "T1", train, test)
            if not m.empty:
                parts[basket] = m
        if len(parts) == 2:
            parts["both"] = pd.concat(parts.values(), ignore_index=True)
        for basket, m in parts.items():
            y = m["y"].to_numpy(float)
            for b in ("all_seven", *ges.GRAPH):
                rows.append({"evaluation": f"{setting}, test {test[0][:4]}-{test[1][:4]}", "basket": basket,
                             "comparison": f"own lags vs {b}", "a": "ar1_12", "b": b,
                             **two_way(y, m["ar1_12"].to_numpy(float), m[b].to_numpy(float), m["node_id"].to_numpy(), m["month"].to_numpy())})
        print(f"done B {setting}", flush=True)
    return rows


def part_c() -> list[dict]:
    f = tab.load()
    wide = glt.ensemble(sorted((glt.OUT / "graph_t1").glob("seed_*")))
    ok = f["T1"].notna() & np.isfinite(f["T1"])
    tr = f["month"].between(*tab.TRAIN) & ok
    te = f["month"].between(*tab.TEST) & ok
    idx = pd.MultiIndex.from_frame(f.loc[te, ["node_id", "month"]])
    wide["ar1_12"] = pd.Series(tab.fit_predict(f, ["T1_lag1", "T1_lag12"], "T1", tr, te), index=idx)
    wide["ar1"] = pd.Series(tab.fit_predict(f, ["T1_lag1"], "T1", tr, te), index=idx)
    wide["ridge7"] = pd.Series(tab.fit_predict(f, tab.ALL7, "T1", tr, te), index=idx)
    wide["zero"] = 0.0
    wide = wide.dropna(subset=["y", "ar1_12", *glt.GRAPH]).reset_index()
    y = wide["y"].to_numpy(float)
    rows = []
    for b in ("zero", "ar1", "ridge7", *glt.GRAPH):
        rows.append({"evaluation": "original 18-country panel, test 2024", "basket": "semiconductor",
                     "comparison": f"own lags vs {b}", "a": "ar1_12", "b": b,
                     **two_way(y, wide["ar1_12"].to_numpy(float), wide[b].to_numpy(float), wide["node_id"].to_numpy(), wide["month"].to_numpy())})
    print("done C", flush=True)
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tab.NODES = Path(__file__).resolve().parent / "data/four_year_2021_2024/processed/nodes_monthly.csv"
    rows = part_c() + part_a() + part_b() + part_d()
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "two_way_tests.csv", index=False)
    pd.set_option("display.width", 250, "display.max_rows", 200)
    print(out[["evaluation", "basket", *(["variant"] if "variant" in out else []), "comparison", "diff_r2", "ci_lo", "ci_hi", "t", "p_two_way", "df", "n"]].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
