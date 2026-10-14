#!/usr/bin/env python3
"""Missing data, country/product selection, and the statistical assumptions of
the expanded (seven-year, ten half-year blocks, both baskets) evaluation on the log change r (T1).

A. Missing-data inventory (rows affected by each kind of gap) and sensitivity of the three main
   contrasts to six treatments of the gaps.
B. Selection: the same contrasts on subgroups of countries (original 18 vs added 23, importer
   size, region) and leave-one-country-out refits.
C. Statistical assumptions: two-way (country x month) cluster-robust tests, Holm adjustment over
   the family of main contrasts, a sign test over the ten blocks, out-of-sample R^2 against the
   training-mean forecast (Campbell-Thompson), serial correlation of the monthly loss
   differential and a moving-block bootstrap over months.

Wild-bootstrap draws: LAMDA_WILD_REPS (default 9,999; the sensitivity and subgroup tables use
1,999). Run parts separately with an argument, e.g. `python robustness_checks.py A`.

Main contrasts: own lags AR(1,12) minus all seven features; all seven minus trade proxies only;
AR(1,12)+exogenous minus AR(1,12). Writes results/benchmark/corrected/checks_*.csv
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
import tabular_analyses as tab  # noqa: E402
import dm_tests  # noqa: E402

REPS = int(__import__("os").environ.get("LAMDA_WILD_REPS", "9999"))


def wild_cluster_boot(d, clusters):
    return dm_tests.wild_cluster_boot(d, clusters, reps=REPS)
from extended_metrics import rolling  # noqa: E402

REV = Path(__file__).resolve().parents[1] / "results" / "benchmark"
OUT = REV / "corrected"
ORIGINAL18 = {"country_in", "country_cn", "country_kr", "country_344", "country_sg", "country_vn", "country_my", "country_jp",
              "country_th", "country_372", "country_us", "country_mx", "country_ph", "country_de", "country_gb", "country_fr",
              "country_380", "country_203"}
ASIA_PACIFIC = {"country_in", "country_cn", "country_kr", "country_344", "country_sg", "country_vn", "country_my", "country_jp",
                "country_th", "country_ph", "country_id", "country_116", "country_104", "country_au"}
AMERICAS = {"country_us", "country_mx", "country_124", "country_br"}
GAP_COUNTRIES = {"country_cn", "country_vn", "country_116", "country_233", "country_ph"}
HONG_KONG = "country_344"


def frames() -> pd.DataFrame:
    fr = pda.frames()
    for f in fr.values():
        g = f.groupby("host_country_id", sort=False)
        f["gap"] = f["inbound_flow_usd"] <= 0
        # rows whose lags or three-month proxy window touch an unreported month
        f["near_gap"] = g["gap"].transform(lambda s: s.astype(float).rolling(13, min_periods=1).max().astype(bool)) & ~f["gap"]
        for c in ("T1_lag1", "T1_lag12"):
            f[f"{c}_missing"] = f[c].isna().astype(float)
    return pd.concat(fr.values(), ignore_index=True)


def models(extra=None):
    e = ["is_consumer", *(extra or [])]
    return {"ar": ["T1_lag1", "T1_lag12", *e], "all7": [*tab.ALL7, *e], "trade": [*tab.TRADE, *e],
            "ar_exo": ["T1_lag1", "T1_lag12", *tab.EXO, *e]}


def run(f: pd.DataFrame, extra=None) -> pd.DataFrame:
    out = None
    for name, feats in models(extra).items():
        y, p, cl, mo = rolling(f, feats)
        d = pd.DataFrame({"y": y, name: p, "country": cl, "month": mo})
        out = d if out is None else out.assign(**{name: p})
    return out


CONTRASTS = (("AR(1,12) - all seven", "ar", "all7"), ("all seven - trade only", "all7", "trade"),
             ("(AR+exogenous) - AR(1,12)", "ar_exo", "ar"))


def contrast_rows(p: pd.DataFrame, label: str, tests: bool = True) -> list[dict]:
    y = p["y"].to_numpy(float)
    rec = {"scenario": label, "n": len(p), "countries": p["country"].nunique(),
           **{f"r2_{m}": r2_score(y, p[m]) for m in ("ar", "all7", "trade", "ar_exo")}}
    rows = []
    for cname, a, b in CONTRASTS:
        r = {**rec, "contrast": cname, "diff_r2": r2_score(y, p[a]) - r2_score(y, p[b])}
        if tests:
            d = (y - p[a]) ** 2 - (y - p[b]) ** 2
            d = -d  # positive when A has the lower squared error
            r["p_country"] = wild_cluster_boot(d, p["country"].to_numpy())[0]
            r["p_month"] = wild_cluster_boot(d, p["month"].to_numpy())[0]
        rows.append(r)
    return rows


def part_a(f: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    inv = []
    per_b = f.groupby("basket")
    for basket, g in per_b:
        nxt_gap = g.groupby("host_country_id")["gap"].shift(-1).fillna(False).astype(bool)
        inv += [
            {"basket": basket, "kind": "country-months in panel", "rows": len(g)},
            {"basket": basket, "kind": "unreported reporter-months (stored as zero)", "rows": int(g["gap"].sum()),
             "detail": g.loc[g["gap"]].groupby("host_country_id").size().to_dict()},
            {"basket": basket, "kind": "rows whose log change is undefined because of a gap", "rows": int((g["gap"] | nxt_gap).sum())},
            {"basket": basket, "kind": "rows with a valid log change", "rows": int(g["T1"].notna().sum())},
            {"basket": basket, "kind": "rows within 12 months after a gap (lags/proxies touched)", "rows": int(g["near_gap"].sum())},
            {"basket": basket, "kind": "rows with lag 12 missing (series start or gap)", "rows": int((g["T1"].notna() & g["T1_lag12"].isna()).sum())},
            {"basket": basket, "kind": "rows in June 2025 (news interpolated)", "rows": int((g["month"] == "2025-06").sum())},
            {"basket": basket, "kind": "Hong Kong rows with zero news volume", "rows": int(((g["host_country_id"] == HONG_KONG) & (g["news_vol_7d"] <= 0)).sum())},
        ]
    base = run(f)
    rows = contrast_rows(base, "baseline (as in the paper)")
    scen = {
        "drop the five countries with any unreported month": f[~f["host_country_id"].isin(GAP_COUNTRIES)],
        "drop rows within 12 months after a gap": f[~f["near_gap"]],
        "drop June 2025 (interpolated news)": f[f["month"] != "2025-06"],
        "drop Hong Kong (sparse news coverage)": f[f["host_country_id"] != HONG_KONG],
    }
    for label, g in scen.items():
        rows += contrast_rows(run(g.reset_index(drop=True)), label)
        print("done", label, flush=True)
    rows += contrast_rows(base[base["month"] < "2025-01"].reset_index(drop=True), "score 2021-2024 only (drop the 2025 blocks)")
    rows += contrast_rows(run(f, extra=["T1_lag1_missing", "T1_lag12_missing"]), "missing-lag indicators instead of zero fill")
    print("done part A", flush=True)
    return pd.DataFrame(inv), pd.DataFrame(rows)


def part_b(f: pd.DataFrame) -> pd.DataFrame:
    base = run(f)
    size = f.groupby("host_country_id")["inbound_flow_usd"].median()
    terc = pd.qcut(size.rank(method="first"), 3, labels=["smallest third", "middle third", "largest third"])
    groups = {
        "original 18 countries": base["country"].isin(ORIGINAL18),
        "23 added countries": ~base["country"].isin(ORIGINAL18),
        "Asia-Pacific": base["country"].isin(ASIA_PACIFIC),
        "Europe and Middle East": ~base["country"].isin(ASIA_PACIFIC | AMERICAS),
        "Americas": base["country"].isin(AMERICAS),
        **{f"importer size: {t}": base["country"].map(terc) == t for t in ("largest third", "middle third", "smallest third")},
    }
    rows = []
    for label, m in groups.items():
        rows += contrast_rows(base[m.to_numpy()].reset_index(drop=True), label, tests=(label != "Americas"))
    print("done subgroups", flush=True)
    loco = []
    for c in sorted(f["host_country_id"].unique()):
        p = run(f[f["host_country_id"] != c].reset_index(drop=True))
        y = p["y"].to_numpy(float)
        loco.append({"dropped": c, **{cn: r2_score(y, p[a]) - r2_score(y, p[b]) for cn, a, b in CONTRASTS},
                     "r2_ar": r2_score(y, p["ar"])})
    loco = pd.DataFrame(loco)
    loco.to_csv(OUT / "checks_loco.csv", index=False)
    for cn, _, _ in CONTRASTS:
        rows.append({"scenario": "leave one country out (41 refits)", "contrast": cn, "diff_r2": loco[cn].median(),
                     "loco_min": loco[cn].min(), "loco_max": loco[cn].max(),
                     "loco_argmin": loco.loc[loco[cn].idxmin(), "dropped"], "loco_argmax": loco.loc[loco[cn].idxmax(), "dropped"],
                     "r2_ar": loco["r2_ar"].median()})
    print("done LOCO", flush=True)
    return pd.DataFrame(rows)


def two_way_t(d: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Cameron-Gelbach-Miller two-way cluster-robust t for the mean of d; reference t(G-1), G = min clusters."""
    n, u = len(d), d - d.mean()

    def v(groups):
        s = pd.Series(u).groupby(groups).sum().to_numpy()
        return s @ s / n ** 2

    inter = pd.Series(a).astype(str) + "|" + pd.Series(b).astype(str)
    var = v(a) + v(b) - v(inter.to_numpy())
    if var <= 0:
        var = max(v(a), v(b))
    t = d.mean() / np.sqrt(var)
    g = min(len(set(a)), len(set(b)))
    return float(t), float(2 * stats.t.sf(abs(t), g - 1))


def block_boot_p(d: np.ndarray, month: np.ndarray, block: int = 6, reps: int = 9999, seed: int = 2026) -> float:
    """Moving-block bootstrap over consecutive months (circular), null imposed by centring."""
    ms = np.array(sorted(set(month)))
    tot = pd.Series(d).groupby(month).sum().reindex(ms).to_numpy()
    cnt = pd.Series(np.ones_like(d)).groupby(month).sum().reindex(ms).to_numpy()
    obs = tot.sum() / cnt.sum()
    cen = tot - obs * cnt
    rng = np.random.default_rng(seed)
    k = int(np.ceil(len(ms) / block))
    hits = 0
    for _ in range(reps):
        starts = rng.integers(0, len(ms), k)
        idx = np.concatenate([(s + np.arange(block)) % len(ms) for s in starts])[:len(ms)]
        if abs(cen[idx].sum() / cnt[idx].sum()) >= abs(obs):
            hits += 1
    return (hits + 1) / (reps + 1)


def part_c(f: pd.DataFrame) -> pd.DataFrame:
    p = run(f)
    y = p["y"].to_numpy(float)
    rows = []
    # out-of-sample R^2 against the training-mean forecast of each block
    bench = []
    for start, end in pda.BLOCKS:
        ok = f["T1"].notna() & np.isfinite(f["T1"])
        tr = f["month"].between(pda.FIRST, pda.prev_month(start)) & ok
        te = f["month"].between(start, end) & ok
        bench.append(np.full(int(te.sum()), f.loc[tr, "T1"].mean()))
    bench = np.concatenate(bench)
    for m in ("ar", "all7", "trade", "ar_exo"):
        sse = np.sum((y - p[m]) ** 2)
        rows.append({"check": "R2 against the training-mean forecast (Campbell-Thompson)", "model": m,
                     "value": 1 - sse / np.sum((y - bench) ** 2), "r2_test_mean": r2_score(y, p[m])})
    for cname, a, b in CONTRASTS:
        d = (y - p[b]) ** 2 - (y - p[a]) ** 2  # positive when A is better
        t, p2 = two_way_t(d, p["country"].to_numpy(), p["month"].to_numpy())
        monthly = pd.Series(d).groupby(p["month"].to_numpy()).mean().sort_index()
        ac1 = monthly.autocorr(1)
        lb = stats.chi2.sf(len(monthly) * (len(monthly) + 2) * sum(monthly.autocorr(k) ** 2 / (len(monthly) - k) for k in (1, 2, 3)), 3)
        rows.append({"check": "two-way clustered (country x month) t-test", "contrast": cname, "t": t, "p": p2})
        rows.append({"check": "moving-block bootstrap over months (blocks of 6)", "contrast": cname,
                     "p": block_boot_p(d, p["month"].to_numpy())})
        rows.append({"check": "lag-1 autocorrelation of the monthly loss differential", "contrast": cname, "value": ac1,
                     "ljung_box_p_3lags": lb, "months": len(monthly)})
        rows.append({"check": "kurtosis of the row loss differential", "contrast": cname, "value": stats.kurtosis(d)})
    print("done part C tests", flush=True)
    # blocks in which own lags beat all seven (both baskets), sign test
    blocks = pd.read_csv(REV / "extended_2019_2025" / "rolling_blocks.csv")
    b = blocks[(blocks["data"] == "both baskets") & (blocks["target"] == "T1")].pivot_table(index="block", columns="model", values="r2")
    wins = int((b["Own lags 1, 12 (AR(1,12))"] > b["All seven features"]).sum())
    rows.append({"check": "sign test: blocks where AR(1,12) beats all seven", "value": wins, "blocks": len(b),
                 "p": float(stats.binomtest(wins, len(b), 0.5).pvalue)})
    wins2 = int((b["AR(1,12) + exogenous"] > b["Own lags 1, 12 (AR(1,12))"]).sum())
    rows.append({"check": "sign test: blocks where adding exogenous channels helps", "value": wins2, "blocks": len(b),
                 "p": float(stats.binomtest(wins2, len(b), 0.5).pvalue)})
    # Holm over the family of main contrasts (month-clustered p, as reported)
    mc = pd.read_csv(REV / "extended_2019_2025" / "main_contrasts_two_clusterings.csv")
    ex = pd.read_csv(REV / "extended_2019_2025" / "exogenous_gain_two_clusterings.csv")
    fam = pd.concat([
        mc[mc["data"] == "both baskets"].rename(columns={"p month": "p_month"})[["contrast", "p_month"]],
        ex[ex["data"] == "both baskets"].rename(columns={"added to AR(1,12)": "contrast", "p (month-clustered)": "p_month"})[["contrast", "p_month"]],
    ], ignore_index=True).drop_duplicates("contrast")
    order = fam["p_month"].sort_values().index
    m = len(fam)
    adj, running = {}, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * fam.loc[i, "p_month"]))
        adj[i] = running
    fam["p_holm"] = pd.Series(adj)
    for _, r in fam.iterrows():
        rows.append({"check": "Holm adjustment over the main family (month-clustered p)", "contrast": r["contrast"],
                     "p": r["p_month"], "p_holm": r["p_holm"], "family_size": m})
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 260, "display.max_rows", 300, "display.max_colwidth", 70)
    f = frames()
    parts = sys.argv[1] if len(sys.argv) > 1 else "CAB"
    if "C" in parts:
        c = part_c(f)
        c.to_csv(OUT / "checks_statistics.csv", index=False)
        print(c.round(4).to_string(index=False), flush=True)
    if "A" in parts:
        inv, sens = part_a(f)
        inv.to_csv(OUT / "checks_missing_inventory.csv", index=False)
        sens.to_csv(OUT / "checks_missing_sensitivity.csv", index=False)
        print(inv.to_string(index=False))
        print(sens.round(4).to_string(index=False), flush=True)
    if "B" in parts:
        sel = part_b(f)
        sel.to_csv(OUT / "checks_selection.csv", index=False)
        print(sel.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
