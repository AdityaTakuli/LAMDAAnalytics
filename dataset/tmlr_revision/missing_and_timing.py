"""TMLR revision, reviewer points 1, 3 and 6: timing of features, missing data,
and the T0 numbers once unreported months are treated as missing.

Nothing is retrained except tabular ridge / RF / LightGBM fits that take seconds.
Graph models are NOT retrained: their saved five-seed test predictions
(results/phase07/graph_preds_T0.csv) are re-scored on the corrected rows.

What it does
  1. Reproduces the canonical T0/T1/T4 numbers from the unmodified panel (sanity
     check that this script uses the same code path as build_canonical.py).
  2. Treats Comtrade reporter-months stored as zero (unpublished; Vietnam 2024)
     as missing, so every target that needs them is undefined, and recomputes
     every T0 number in the paper. T1, T4 and H_k are recomputed as a check that
     they do not move.
  3. Counts excluded rows per split and reason, positives per split, and the
     imputations used for features.
  4. Checks timing: the two trade proxies are recomputed from flows at months
     <= T only; the GSCPI and temperature channels are rebuilt from values
     published by the end of month T, and the exogenous ablation is re-run.

Outputs: dataset/tmlr_revision/out/*.csv and out/figures/*.pdf
Run from dataset/:  ../backend/.venv/bin/python tmlr_revision/missing_and_timing.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

HERE = Path(__file__).resolve().parent
DATASET = HERE.parent
sys.path.insert(0, str(DATASET))

import build_canonical as bc  # noqa: E402
import dm_tests as dmt  # noqa: E402
import seasonal_and_rolling as sar  # noqa: E402
from phase0_t1_ladder import RIDGE, autocorr, load, standardize, tune_reg, win  # noqa: E402
from preprocessing_exogenous import add_news_normalized  # noqa: E402

ROOT = DATASET.parent
OUT = HERE / "out"
FIGS = OUT / "figures"
PROC = DATASET / "data/four_year_2021_2024/processed"
TRAIN, VALID, TEST = bc.TRAIN, bc.VALID, bc.TEST
SPLITS = {"train": TRAIN, "validation": VALID, "test": TEST}


# --------------------------------------------------------------------------- #
# frames
# --------------------------------------------------------------------------- #
def unreported_mask(f: pd.DataFrame) -> pd.Series:
    """Rows whose target needs an unpublished month (stored as zero inbound)."""
    v = f.groupby("host_country_id", sort=False)["inbound_flow_usd"]
    fut = v.shift(-1)
    return (fut == 0) | (f["inbound_flow_usd"] == 0)


def add_lags(f: pd.DataFrame) -> pd.DataFrame:
    g = f.groupby("host_country_id", sort=False)
    for t in ("T0", "T1", "T4"):
        f[f"{t}_lag1"] = g[t].shift(1)
    f["contraction_lag1"] = g["T0"].shift(1)
    return f


def frames():
    raw = add_lags(load())
    cor = load()
    m = unreported_mask(cor)
    cor.loc[m, "T0"] = np.nan
    cor = add_lags(cor)
    return raw, cor, m


# --------------------------------------------------------------------------- #
# 1-2. canonical numbers, raw and corrected
# --------------------------------------------------------------------------- #
def graph_scores(mask_rows: pd.DataFrame | None) -> list[dict]:
    g = pd.read_csv(ROOT / "results/phase07/graph_preds_T0.csv")
    g["month"] = g["month"].astype(str)
    if mask_rows is not None:
        keep = set(map(tuple, mask_rows[["host_country_id", "month"]].to_numpy()))
        g = g[[(c, m) in keep for c, m in zip(g["host_country_id"], g["month"])]]
    rows = []
    for model, sub in g.groupby("model"):
        per_seed = [r2_score(s["y"], s["pred"]) for _, s in sub.groupby("seed")]
        ens = sub.groupby(["host_country_id", "month"]).agg(y=("y", "first"), p=("pred", "mean"))
        rows.append({"model": model, "seed_mean": float(np.mean(per_seed)),
                     "seed_sd": float(np.std(per_seed, ddof=1)),
                     "ensemble": float(r2_score(ens["y"], ens["p"])), "n": len(ens)})
    return rows


def denominator_rows(f: pd.DataFrame, mask: pd.Series | None):
    import target_definition_robustness as tdr
    g = tdr.load()[["host_country_id", "month", "T2", "contraction_lag1_disjoint"]]
    f = f.merge(g, on=["host_country_id", "month"], how="left", validate="one_to_one")
    if mask is not None:
        f.loc[mask.to_numpy(), "T2"] = np.nan
    f["T3"] = f["T0"]
    out = []
    for target, lag in (("T2", "contraction_lag1"), ("T3", "contraction_lag1_disjoint")):
        valid = f[target].notna()
        tr, va, te = win(f, TRAIN) & valid, win(f, VALID) & valid, win(f, TEST) & valid
        y = f.loc[te, target].to_numpy(float)
        persist = f.groupby("host_country_id", sort=False)[target].shift(1)[te].fillna(0.0)
        out.append((target, "persistence", r2_score(y, persist.to_numpy(float)), int(te.sum())))
        z = standardize(f, ["flow_ratio", lag, "flow_ratio_lag1"], tr)
        m = tune_reg(z[tr].to_numpy(), f.loc[tr, target].to_numpy(float),
                     z[va].to_numpy(), f.loc[va, target].to_numpy(float), Ridge, RIDGE)
        out.append((target, "ar_ridge", r2_score(y, m.predict(z[te].to_numpy())), int(te.sum())))
    return out


def ar112(f: pd.DataFrame, target: str) -> float:
    g = f.groupby("host_country_id", sort=False)
    f = f.copy()
    f[f"{target}_lag12"] = g[target].shift(12)
    valid = f[target].notna()
    tr, va, te = win(f, TRAIN) & valid, win(f, VALID) & valid, win(f, TEST) & valid
    z = standardize(f, [f"{target}_lag1", f"{target}_lag12"], tr)
    m = tune_reg(z[tr].to_numpy(), f.loc[tr, target].to_numpy(float),
                 z[va].to_numpy(), f.loc[va, target].to_numpy(float), Ridge, RIDGE)
    return r2_score(f.loc[te, target], m.predict(z[te].to_numpy()))


def canonical_like(f: pd.DataFrame, mask: pd.Series | None, label: str) -> pd.DataFrame:
    recs = []
    for target in ("T0", "T1", "T4"):
        n = int((win(f, TEST) & f[target].notna()).sum())
        for model, val in bc.tabular_r2(f, target):
            recs.append((label, target, model, val, n))
        for model, val in bc.ar1_rows(f, target):
            recs.append((label, target, model, val, n))
        recs.append((label, target, "ar1_12", ar112(f, target), n))
        recs.append((label, target, "lag1_autocorr", autocorr(f, target), np.nan))
    for target in ("T0", "T1"):
        n = int((win(f, TEST) & f[target].notna()).sum())
        for model, val in bc.exogenous_rows(f, target):
            recs.append((label, target, model, val, n))
    for target, model, val, n in denominator_rows(f, mask):
        recs.append((label, target, model, val, n))
    te = win(f, TEST) & f["T0"].notna()
    for r in graph_scores(f.loc[te] if mask is not None else None):
        recs.append((label, "T0", f"{r['model']}_seed_mean", r["seed_mean"], r["n"]))
        recs.append((label, "T0", f"{r['model']}_seed_sd", r["seed_sd"], r["n"]))
        recs.append((label, "T0", f"{r['model']}_ensemble", r["ensemble"], r["n"]))
    return pd.DataFrame(recs, columns=["version", "target", "model", "value", "n_test"])


def check_reproduces_canonical(raw_tab: pd.DataFrame) -> None:
    canon = pd.read_csv(ROOT / "results/canonical_numbers.csv")
    canon = canon[(canon["metric"] == "R2") & (canon["split"] == "test2024")]
    merged = raw_tab.merge(canon[["target", "model", "value"]], on=["target", "model"],
                           suffixes=("", "_canon"))
    merged["absdiff"] = (merged["value"] - merged["value_canon"]).abs()
    merged.to_csv(OUT / "reproduce_check.csv", index=False)
    worst = merged["absdiff"].max()
    print(f"[check] {len(merged)} canonical R2 values recomputed; max |diff| = {worst:.2e}")
    if worst > 5e-6:
        print(merged.sort_values("absdiff", ascending=False).head(10).to_string())
        raise SystemExit("recomputation does not reproduce canonical_numbers.csv")


# --------------------------------------------------------------------------- #
# T0 inference, origins, rolling models (corrected)
# --------------------------------------------------------------------------- #
def nodes_like(mask_unreported: bool) -> pd.DataFrame:
    n = dmt.load_nodes()
    if mask_unreported:
        m = unreported_mask(n)
        n.loc[m, "T0"] = np.nan
        g = n.groupby("host_country_id", sort=False)
        n["contraction_lag1"] = g["T0"].shift(1)
        n["T0_lag1"] = g["T0"].shift(1)
    return n


def t0_tests(n: pd.DataFrame, label: str) -> pd.DataFrame:
    dmt.RESULTS.clear()
    t0 = dmt.add_graph_ensemble(dmt.tabular_preds(n.copy(), "T0"), "T0")
    dmt.run_pair("T0: persistence vs gcn (ensemble)", t0, "persistence", "gcn")
    dmt.run_pair("T0: persistence vs ar_ridge", t0, "persistence", "ar_ridge")
    dmt.run_pair("T0: ar1_only vs persistence", t0, "ar1_only", "persistence")
    out = pd.DataFrame(dmt.RESULTS)
    out.insert(0, "version", label)
    return out


def origins(f: pd.DataFrame, label: str) -> pd.DataFrame:
    months = sorted(f["month"].unique())
    rows = []
    for start in range(13, len(months), 6):
        ev = months[start:start + 6]
        if len(ev) < 2:
            continue
        rec = {"version": label, "eval": f"{ev[0]}..{ev[-1]}"}
        for t in ("T0", "T1", "T4"):
            lag = f.groupby("host_country_id", sort=False)[t].shift(1)
            m = f["month"].isin(ev) & f[t].notna() & lag.notna()
            rec[t] = r2_score(f.loc[m, t], lag[m]) if m.sum() > 2 else np.nan
            rec[f"n_{t}"] = int(m.sum())
        rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 3. missing-data inventory
# --------------------------------------------------------------------------- #
def inventory(raw: pd.DataFrame) -> pd.DataFrame:
    f = raw.copy()
    v = f.groupby("host_country_id", sort=False)["inbound_flow_usd"]
    fut, med12 = v.shift(-1), v.transform(lambda s: s.rolling(12, min_periods=1).median())
    rows = []

    def add(split, reason, n, note=""):
        rows.append({"split": split, "reason": reason, "rows": int(n), "note": note})

    add("panel", "country-months in the fused panel", len(f))
    add("panel", "2021-01..2021-11: lags of the 12-month median not yet observed (no split)",
        f["month"].between("2021-01", "2021-11").sum())
    add("panel", "2024-12: target month 2025-01 outside the panel (no split)", (f["month"] == "2024-12").sum())
    add("panel", "stored zero inbound = month not yet published by Comtrade",
        (f["inbound_flow_usd"] == 0).sum(), ", ".join(sorted(
            (f.loc[f["inbound_flow_usd"] == 0, "host_country_id"] + " " +
             f.loc[f["inbound_flow_usd"] == 0, "month"]).str.replace("country_", "").tolist())[:1]) + " ... (Vietnam 2024-01..12)")
    for split, b in SPLITS.items():
        w = win(f, b)
        add(split, "country-months in split", w.sum())
        nxt0 = w & (fut == 0) & (f["inbound_flow_usd"] > 0)
        cur0 = w & (f["inbound_flow_usd"] == 0)
        add(split, "T+1 unpublished, month T published (T1 undefined; T0 was -1 in the submission)", nxt0.sum())
        add(split, "month T unpublished (T1 undefined)", cur0.sum())
        add(split, "  of which T0 was -1 in the submission (median still > 0)",
            (cur0 & (fut == 0) & (med12 > 0)).sum())
        add(split, "  of which T0 undefined (12-month median = 0)", (cur0 & (med12 <= 0)).sum())
        add(split, "rows scored, T1", (w & f["T1"].notna()).sum())
        add(split, "rows scored, T0 as submitted", (w & f["T0"].notna()).sum())
        add(split, "rows scored, T0 corrected", (w & f["T0"].notna() & ~unreported_mask(f)).sum())
        t1lag12 = f.groupby("host_country_id", sort=False)["T1"].shift(12)
        add(split, "T1 rows whose lag-12 is unobserved (set to the training mean)",
            (w & f["T1"].notna() & t1lag12.isna()).sum())
        t1lag1 = f.groupby("host_country_id", sort=False)["T1"].shift(1)
        add(split, "T1 rows whose lag-1 is unobserved (set to the training mean)",
            (w & f["T1"].notna() & t1lag1.isna()).sum())
    # positives
    tau_thr = float(f.loc[win(f, TRAIN) & f["T1"].notna(), "T1"].quantile(0.104))
    for split, b in SPLITS.items():
        w = win(f, b)
        add(split, "T0 < -0.20 events, as submitted", (w & (f["T0"] < -0.20)).sum())
        add(split, "T0 < -0.20 events, corrected", (w & (f["T0"] < -0.20) & ~unreported_mask(f)).sum())
        add(split, "T1 below the training 10.4th percentile", (w & (f["T1"] < tau_thr)).sum(),
            f"threshold {tau_thr:.3f}")
    # exogenous-source gaps
    w = pd.read_csv(PROC / "weather_daily.csv", usecols=["node_id", "observed_date", "temperature_c"])
    add("sources", "NASA POWER daily temperatures missing or fill value (-999)",
        int(w["temperature_c"].isna().sum() + (w["temperature_c"] <= -900).sum()),
        f"of {len(w)} country-days")
    g = pd.read_csv(PROC / "gscpi_monthly.csv")
    add("sources", "GSCPI months missing in 2021-01..2024-12",
        48 - g["month"].astype(str).between("2021-01", "2024-12").sum())
    add("sources", "GDELT daily files not published (dates skipped)", 2, "2022-11-10, 2023-03-23")
    add("sources", "country-months with zero GDELT news volume", int((f["news_vol_7d"] == 0).sum()))
    add("sources", "feature values flagged as inherited/imputed in the panel",
        int(f["is_inherited"].astype(str).str.lower().eq("true").sum()))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 4. timing checks
# --------------------------------------------------------------------------- #
def proxy_check(raw: pd.DataFrame) -> dict:
    f = raw.copy()
    v = f.groupby("host_country_id", sort=False)["inbound_flow_usd"]
    base = v.transform(lambda s: s.shift(1).rolling(3, min_periods=1).median())
    q = f["inbound_flow_usd"] / base.replace(0, np.nan)
    inv = (30 * q.clip(lower=1 / 6, upper=3)).fillna(30.0)
    delay = (30 * (1 - q).clip(lower=0)).clip(upper=60).fillna(0.0)
    ok = base > 0
    return {"rows_checked": int(ok.sum()),
            "inventory_max_absdiff": float((inv[ok] - f.loc[ok, "inventory_days_proxy"]).abs().max()),
            "delay_max_absdiff": float((delay[ok] - f.loc[ok, "trade_delay_proxy"]).abs().max()),
            "future_flow_column_used_by_models": False}


def strict_weather(frame: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    """Monthly mean temperature over days already published at the end of month T
    (NASA POWER treated as available two days after observation)."""
    w = pd.read_csv(PROC / "weather_daily.csv")
    obs = pd.to_datetime(w["observed_date"], utc=True)
    avail = pd.to_datetime(w["available_at"], utc=True)
    month = obs.dt.to_period("M")
    month_end = month.dt.to_timestamp(how="end").dt.tz_localize("UTC")
    w = w[(avail <= month_end).to_numpy()].copy()
    w["month"] = month[(avail <= month_end).to_numpy()].astype(str).to_numpy()
    temp = (w.groupby(["node_id", "month"])["temperature_c"].mean().reset_index()
            .rename(columns={"node_id": "host_country_id", "temperature_c": "temp_mean"}))
    out = frame.merge(temp, on=["host_country_id", "month"], how="left")
    stats = out.loc[train_mask.values].groupby("host_country_id")["temp_mean"].agg(["mean", "std"])
    mu = out["host_country_id"].map(stats["mean"])
    sd = out["host_country_id"].map(stats["std"]).replace(0.0, np.nan)
    out["weather_temp_z"] = ((out["temp_mean"] - mu) / sd).fillna(0.0)
    out["weather_temp_absz"] = out["weather_temp_z"].abs()
    return out


def strict_gscpi(frame: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    """GSCPI as published by the end of month T: the month T-1 value and its change."""
    g = pd.read_csv(PROC / "gscpi_monthly.csv")[["month", "global_risk"]].copy()
    g["month"] = g["month"].astype(str)
    g = g.sort_values("month")
    g["gscpi_lag1"] = g["global_risk"].shift(1)                    # G_{T-1}
    g["gscpi_change"] = g["global_risk"].shift(1).diff()           # G_{T-1} - G_{T-2}
    g = g.drop(columns=["global_risk"]).fillna(0.0)
    return frame.merge(g, on="month", how="left")


def strict_exogenous(f: pd.DataFrame, target: str) -> list[tuple[str, float]]:
    valid = f[target].notna()
    tr, va, te = win(f, TRAIN) & valid, win(f, VALID) & valid, win(f, TEST) & valid
    y = f.loc[te, target].to_numpy(float)
    specs = [("exo_ar_baseline", []), ("exo_+news", ["news"]), ("exo_+weather", ["weather"]),
             ("exo_+gscpi", ["gscpi"]), ("exo_+all", ["news", "weather", "gscpi"])]
    cols = {"news": ["news_z", "news_surprise"], "weather": ["weather_temp_z", "weather_temp_absz"],
            "gscpi": ["gscpi_change", "gscpi_lag1"]}
    fns = {"news": add_news_normalized, "weather": strict_weather, "gscpi": strict_gscpi}
    out = []
    for name, groups in specs:
        g = f.copy()
        feats = list(bc.AR)
        for grp in groups:
            g = fns[grp](g, tr)
            feats += cols[grp]
        z = standardize(g, feats, tr)
        m = tune_reg(z[tr].to_numpy(), g.loc[tr, target].to_numpy(float),
                     z[va].to_numpy(), g.loc[va, target].to_numpy(float), Ridge, RIDGE)
        out.append((name, r2_score(y, m.predict(z[te].to_numpy()))))
    return out


# --------------------------------------------------------------------------- #
# figures (corrected T0)
# --------------------------------------------------------------------------- #
def figures(tab: pd.DataFrame, orig: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import make_paper_figures as mpf

    FIGS.mkdir(parents=True, exist_ok=True)
    cor = tab[tab["version"] == "corrected"]

    def val(t, m):
        return float(cor[(cor["target"] == t) & (cor["model"] == m)]["value"].iloc[0])

    rows = [(lab, val("T0", m), val("T1", m)) for m, lab in mpf.LADDER]
    lo, hi = -0.70, 0.78
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.axhline(0, color="gray", lw=0.6, zorder=0)
    y1s = [max(t1, lo) for _, _, t1 in rows]
    right = mpf.spread(y1s, gap=0.055, lo=lo + 0.02)
    left = mpf.spread([min(max(t0, lo), hi) for _, t0, _ in rows], gap=0.045, lo=lo + 0.02)
    for (lab, t0, t1), y1, ly1, ly0 in zip(rows, y1s, right, left):
        off = t1 < lo
        crosses = (t0 > 0.2) and (t1 < -0.5)
        color = "#d62728" if crosses else "#4c72b0"
        lw = 2.0 if crosses else 1.1
        y0 = max(t0, lo)
        ax.plot([0, 1], [y0, y1], "-", color=color, lw=lw, alpha=0.9, zorder=2)
        ax.plot([0], [y0], "v" if t0 < lo else "o", color=color, ms=5, zorder=3, clip_on=False)
        ax.plot([1], [y1], "v" if off else "o", color=color, ms=7 if off else 5, clip_on=False, zorder=3)
        ax.annotate(f"{lab}  {mpf.fmt3(t1)}" + (" (off scale)" if off else ""), xy=(1.0, y1),
                    xytext=(1.08, ly1), fontsize=6.5, color=color, va="center",
                    arrowprops=dict(arrowstyle="-", color="0.6", lw=0.5))
        ax.annotate(mpf.fmt3(t0), xy=(0.0, y0), xytext=(-0.08, ly0), fontsize=6.5, color=color,
                    va="center", ha="right", arrowprops=dict(arrowstyle="-", color="0.6", lw=0.5))
    ax.set_xlim(-0.42, 1.95)
    ax.set_ylim(lo, hi)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["T0\n(overlap $\\approx 11/12$)", "T1\n(overlap $0$)"])
    ax.set_ylabel("test-2024 $R^2$")
    ax.set_title("Removing window overlap inverts the ranking", fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "rank_inversion_corrected.pdf")
    plt.close(fig)

    o = orig[orig["version"] == "corrected"].reset_index(drop=True)
    x = np.arange(len(o))
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for t, c in (("T0", "#d62728"), ("T1", "#1f77b4"), ("T4", "#2ca02c")):
        ax.plot(x, o[t], "o-", color=c, label=t)
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([e.replace("..", "\n..") for e in o["eval"]], fontsize=7)
    ax.set_ylabel("persistence $R^2$")
    ax.set_title("Persistence skill is regime-dependent (zero parameters)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "per_origin_persistence_corrected.pdf")
    plt.close(fig)


def extra_figures(tab: pd.DataFrame) -> pd.DataFrame:
    """T0 origin-mean persistence (horizon_sweep method), corrected graph-seed figure,
    and the rank-inversion slope chart drawn on T4 (unaffected by the correction)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import horizon_sweep as hs
    import make_paper_figures as mpf

    rows = []
    for label, masked in (("submitted", False), ("corrected", True)):
        f = load()
        if masked:
            f.loc[unreported_mask(f), "T0"] = np.nan
        rows.append({"version": label, "T0_persist_origin_mean": hs.persistence_r2_by_origin(f, "T0")})

    cor = tab[tab["version"] == "corrected"]

    def val(t, m):
        return float(cor[(cor["target"] == t) & (cor["model"] == m)]["value"].iloc[0])

    # graph seeds, corrected T0 panel
    g = pd.read_csv(ROOT / "results/phase07/graph_preds_T0.csv")
    g["month"] = g["month"].astype(str)
    raw, corf, _ = frames()
    keep = corf.loc[win(corf, TEST) & corf["T0"].notna(), ["host_country_id", "month"]]
    g = g.merge(keep, on=["host_country_id", "month"])
    seeds = {m: [r2_score(s["y"], s["pred"]) for _, s in sub.groupby("seed")] for m, sub in g.groupby("model")}
    t1 = pd.read_csv(ROOT / "results/phase07/graph_convergence_T1.csv").set_index("model")
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 3.2))
    for ax, target, ylim in zip(axes, ("T0", "T1"), ((0.0, 0.55), (-0.05, 0.25))):
        for i, (m, lab) in enumerate(mpf.GRAPHS):
            vals = seeds[m] if target == "T0" else [float(v) for v in str(t1.loc[m, "per_seed"]).split()]
            ax.plot(np.full(len(vals), i), vals, "o", color="#4c72b0", ms=5, alpha=0.7, zorder=3)
            ax.plot([i - 0.22, i + 0.22], [np.mean(vals)] * 2, "-", color="#d62728", lw=1.8, zorder=4)
        for model, color, ls, lab in (("ar1_12", "#9467bd", "-.", "AR(1,12)"), ("ar1_only", "#2ca02c", "--", "AR(1)"),
                                      ("ar_ridge", "#8c564b", ":", "AR ridge")):
            v = val(target, model)
            ax.axhline(v, color=color, ls=ls, lw=1.1, label=f"{lab} = {mpf.fmt3(v)}")
        if target == "T1":
            ax.annotate(f"persistence {mpf.fmt3(val('T1', 'persistence'))}\n(off scale)", xy=(0.02, 0.03),
                        xycoords="axes fraction", fontsize=6.5, color="gray")
            ax.axhline(0, color="gray", lw=0.6, zorder=0)
        ax.set_xticks(range(len(mpf.GRAPHS)))
        ax.set_xticklabels([lab for _, lab in mpf.GRAPHS], fontsize=7.5)
        ax.set_xlim(-0.5, len(mpf.GRAPHS) - 0.5)
        ax.set_ylim(*ylim)
        ax.set_title(f"{target} (overlap {'$\\approx 11/12$' if target == 'T0' else '$0$'})", fontsize=9)
        ax.legend(loc="lower right", fontsize=6.5, frameon=False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("test-2024 $R^2$")
    fig.suptitle("Five seeds at matched budget; spread is small, level is not competitive", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIGS / "graph_seeds_corrected.pdf")
    plt.close(fig)

    # rank inversion on T4 -> T1
    rows_l = [(lab, val("T4", m), val("T1", m)) for m, lab in mpf.LADDER]
    lo, hi = -0.70, 0.62
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.axhline(0, color="gray", lw=0.6, zorder=0)
    y0s = [max(a, lo) for _, a, _ in rows_l]
    y1s = [max(b, lo) for _, _, b in rows_l]
    right = mpf.spread(y1s, gap=0.055, lo=lo + 0.02)
    left = mpf.spread(y0s, gap=0.05, lo=lo + 0.02)
    for (lab, a, b), y0, y1, ly0, ly1 in zip(rows_l, y0s, y1s, left, right):
        crosses = a > 0.4 and b < -0.5
        color = "#d62728" if crosses else "#4c72b0"
        ax.plot([0, 1], [y0, y1], "-", color=color, lw=2.0 if crosses else 1.1, alpha=0.9, zorder=2)
        ax.plot([0], [y0], "v" if a < lo else "o", color=color, ms=6 if a < lo else 5, clip_on=False, zorder=3)
        ax.plot([1], [y1], "v" if b < lo else "o", color=color, ms=6 if b < lo else 5, clip_on=False, zorder=3)
        ax.annotate(f"{lab}  {mpf.fmt3(b)}" + (" (off scale)" if b < lo else ""), xy=(1.0, y1), xytext=(1.08, ly1),
                    fontsize=6.5, color=color, va="center", arrowprops=dict(arrowstyle="-", color="0.6", lw=0.5))
        ax.annotate(mpf.fmt3(a) + (" (off)" if a < lo else ""), xy=(0.0, y0), xytext=(-0.08, ly0), fontsize=6.5,
                    color=color, va="center", ha="right", arrowprops=dict(arrowstyle="-", color="0.6", lw=0.5))
    ax.set_xlim(-0.42, 1.95)
    ax.set_ylim(lo, hi)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["T4 $=H_{12}$\n(overlap $11/12$)", "T1 $=H_1$\n(overlap $0$)"])
    ax.set_ylabel("test-2024 $R^2$")
    ax.set_title("Removing window overlap inverts the ranking", fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "rank_inversion_T4_T1.pdf")
    plt.close(fig)
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1:] == ["--extras"]:
        tab = pd.read_csv(OUT / "t0_corrected_numbers.csv")
        extra_figures(tab).to_csv(OUT / "t0_origin_mean.csv", index=False)
        print(pd.read_csv(OUT / "t0_origin_mean.csv"))
        return
    raw, cor, mask = frames()
    print(f"unreported-month rows (target needs a stored zero): {int(mask.sum())}")

    print("== 1. reproduce canonical numbers from the unmodified panel")
    raw_tab = canonical_like(raw, None, "submitted")
    check_reproduces_canonical(raw_tab)

    print("== 2. corrected (unreported months treated as missing)")
    cor_tab = canonical_like(cor, mask, "corrected")
    tab = pd.concat([raw_tab, cor_tab], ignore_index=True)
    tab.to_csv(OUT / "t0_corrected_numbers.csv", index=False)
    wide = tab.pivot_table(index=["target", "model"], columns="version", values="value").reset_index()
    wide["change"] = wide["corrected"] - wide["submitted"]
    wide.to_csv(OUT / "t0_corrected_wide.csv", index=False)
    with pd.option_context("display.width", 160, "display.max_rows", 200):
        print(wide.round(4).to_string(index=False))

    tests = pd.concat([t0_tests(nodes_like(False), "submitted"),
                       t0_tests(nodes_like(True), "corrected")], ignore_index=True)
    tests.to_csv(OUT / "t0_tests.csv", index=False)

    orig = pd.concat([origins(raw, "submitted"), origins(cor, "corrected")], ignore_index=True)
    orig.to_csv(OUT / "per_origin_persistence.csv", index=False)
    print(orig.round(3).to_string(index=False))

    roll = sar.rolling(sar.add_lags(nodes_like(True)))
    roll.to_csv(OUT / "rolling_origin_models_corrected.csv", index=False)

    print("== 3. missing-data inventory")
    inv = inventory(raw)
    inv.to_csv(OUT / "missing_inventory.csv", index=False)
    print(inv.to_string(index=False))

    print("== 4. timing")
    chk = proxy_check(raw)
    pd.DataFrame([chk]).to_csv(OUT / "timing_proxy_check.csv", index=False)
    print(chk)
    strict = []
    for target, frame in (("T1", raw), ("T0", cor)):
        base = dict(bc.exogenous_rows(frame, target))
        for name, val in strict_exogenous(frame, target):
            strict.append({"target": target, "model": name, "as_submitted": base[name],
                           "strictly_available": val})
    strict = pd.DataFrame(strict)
    strict.to_csv(OUT / "exogenous_strict_timing.csv", index=False)
    print(strict.round(4).to_string(index=False))

    figures(tab, orig)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
