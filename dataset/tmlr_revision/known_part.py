"""TMLR revision, extension: score only the part of a window target that is unknown.

A window target H_k(T) = log(V_{T+1}/V_{T-k+1}) splits into a part known at the
forecast date and one unknown month:

    H_k(T) = K_k(T) + r(T),   K_k(T) = log(V_T/V_{T-k+1}),   r(T) = log(V_{T+1}/V_T) = T1.

Any forecast h of H_k implies the forecast h - K_k of r, and the two make the same
error, so the scores are linked exactly:

    R2_window = 1 - (1 - R2_new) * Var(r)/Var(H_k).

Copying the known part (h = K_k, no parameters) therefore earns
F_k = 1 - Var(r)/Var(H_k) - mean(r)^2/Var(H_k) on the window: the free R2 a model gets
for reproducing data it already has. Persistence on H_k implies r_hat(T) = r(T-k),
the increment k months earlier (for k = 12, the same calendar month a year ago).

This script reports, on test 2024 (n = 187, unpublished Vietnamese months missing):
  known_part.csv          every model scored on the window and on the new month,
                          for k in {1,2,3,6,12}, two training designs:
                          direct (fit on H_k) and new-part (fit on T1, add K_k back)
  known_part_t0.csv       the same idea for the rolling-median target T0
  known_part_tests.csv    month-level and country-level tests for the key contrasts
  known_part_periods.csv  free R2 F_k in train, validation, test and 6-month blocks
  known_part_rolling.csv  training-design comparison over 22 months of rolling refits
  figures/known_part.pdf
Run from dataset/:  ../backend/.venv/bin/python tmlr_revision/known_part.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import horizon_sweep as hs  # noqa: E402
from dependence_tests import WEBB, holm, twoway, wild, dm_month  # noqa: E402
from missing_and_timing import frames  # noqa: E402
from phase0_t1_ladder import RIDGE, standardize, tune_reg  # noqa: E402

OUT = HERE / "out"
FIG = OUT / "figures"
KS = [1, 2, 3, 6, 12]
TRAIN, VALID, TEST = hs.TRAIN, hs.VALID, hs.TEST
ID = "host_country_id"


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load() -> pd.DataFrame:
    f = hs.load()                                   # H_k undefined when V_{T+1} or V_{T-k+1} is 0
    v = f["inbound_flow_usd"]
    f["lv"] = np.log(v.where(v > 0))
    g = f.groupby(ID, sort=False)
    f["r"] = f["H1"]
    for k in KS:
        f[f"K{k}"] = f["lv"] - g["lv"].shift(k - 1)
        f[f"r_lag{k}"] = g["r"].shift(k)            # increment k months earlier
    f["T1_lag1"], f["T1_lag12"] = f["r_lag1"], f["r_lag12"]
    _, cor, _ = frames()                            # corrected T0 (unpublished months missing)
    f = f.drop(columns=["T0", "T0_lag1"]).merge(
        cor[[ID, "month", "T0", "contraction_lag1"]], on=[ID, "month"], how="left",
        validate="one_to_one")
    f["med12"] = g["inbound_flow_usd"].transform(lambda s: s.rolling(12, min_periods=1).median())
    return f


def fit_predict(f, target, feats, tr, va, rows, paper_harness=False):
    """Ridge tuned on validation. paper_harness=True uses horizon_sweep's own helpers
    (reproduces the canonical H_k numbers); otherwise phase0_t1_ladder's (reproduces T1)."""
    if paper_harness:
        z = hs.standardize(f, feats, tr)
        m = hs.tune_ridge(z, target, f, tr, va)
    else:
        z = standardize(f, feats, tr)
        m = tune_reg(z[tr].to_numpy(), f.loc[tr, target].to_numpy(float),
                     z[va].to_numpy(), f.loc[va, target].to_numpy(float), Ridge, RIDGE)
    return m.predict(z[rows].to_numpy())


def masks(f, cols, bounds=(TRAIN, VALID, TEST)):
    ok = f[cols].notna().all(axis=1)
    return [hs.win(f, b) & ok for b in bounds]


# --------------------------------------------------------------------------- #
# 1. window vs new-month scores, test 2024
# --------------------------------------------------------------------------- #
def window_table(f):
    rows, preds = [], {}
    for k in KS:
        H, K = f"H{k}", f"K{k}"
        tr, va, te = masks(f, [H, "r", K])
        y, r, kk = (f.loc[te, c].to_numpy(float) for c in (H, "r", K))
        pers = f.groupby(ID, sort=False)[H].shift(1)[te].to_numpy(float)
        p = {
            ("zero-parameter", "zero"): np.zeros(len(y)),
            ("zero-parameter", "known part (no change)"): kk,
            ("zero-parameter", "persistence"): pers,
            ("direct", "3-feature ridge"): fit_predict(
                f, H, ["flow_ratio", f"{H}_lag1", "flow_ratio_lag1"], tr, va, te, True),
            ("direct", "3-feature ridge + known part as input"): fit_predict(
                f, H, ["flow_ratio", f"{H}_lag1", "flow_ratio_lag1", K], tr, va, te, True),
            ("direct", "7-feature ridge"): fit_predict(f, H, hs.PAPER, tr, va, te, True),
            ("new part", "known part + 3-feature ridge"): kk + fit_predict(
                f, "r", ["flow_ratio", "H1_lag1", "flow_ratio_lag1"], tr, va, te, True),
            ("new part", "known part + 7-feature ridge"): kk + fit_predict(
                f, "r", hs.PAPER, tr, va, te, True),
            ("new part", "known part + AR(1)"): kk + fit_predict(f, "r", ["T1_lag1"], tr, va, te),
            ("new part", "known part + AR(1,12)"): kk + fit_predict(
                f, "r", ["T1_lag1", "T1_lag12"], tr, va, te),
        }
        ratio = np.var(r) / np.var(y)
        for (design, model), h in p.items():
            r2w, r2n = r2_score(y, h), r2_score(r, h - kk)
            implied = 1 - (1 - r2n) * ratio
            assert abs(implied - r2w) < 1e-9, (k, model, implied, r2w)
            rows.append({"k": k, "design": design, "model": model, "n": len(y),
                         "r2_window": r2w, "r2_new_month": r2n,
                         "mse": float(np.mean((y - h) ** 2)), "var_ratio_r_over_H": ratio})
        # persistence on H_k implies the increment k months earlier
        lagk = f.loc[te, f"r_lag{k}"].to_numpy(float)
        ok = ~np.isnan(lagk)
        assert np.allclose((pers - kk)[ok], lagk[ok]), k
        idx = f.loc[te, [ID, "month"]]
        preds[k] = pd.DataFrame({"y": y, **{m: h for (_, m), h in p.items()}},
                                index=pd.MultiIndex.from_frame(idx))
    return pd.DataFrame(rows), preds


# --------------------------------------------------------------------------- #
# 2. the same idea on the rolling-median target T0
# --------------------------------------------------------------------------- #
def t0_table(f):
    """T0 = V_{T+1}/med12_T - 1. Its known part is the no-change level forecast
    V_T/med12_T - 1; a T1 forecast r_hat maps to T0 as V_T exp(r_hat)/med12_T - 1.
    A T0 forecast c implies r_hat = log((1+c) med12_T / V_T), defined when c > -1."""
    tr, va, te = masks(f, ["T0", "r"])
    y, r = f.loc[te, "T0"].to_numpy(float), f.loc[te, "r"].to_numpy(float)
    ratio = (f.loc[te, "inbound_flow_usd"] / f.loc[te, "med12"]).to_numpy(float)
    known = ratio - 1
    pers = f.groupby(ID, sort=False)["T0"].shift(1)[te].fillna(0.0).to_numpy(float)
    ar3 = fit_predict(f, "T0", ["flow_ratio", "contraction_lag1", "flow_ratio_lag1"], tr, va, te)
    r7 = fit_predict(f, "T0", hs.PAPER, tr, va, te)
    t1 = {"AR(1)": fit_predict(f, "r", ["T1_lag1"], tr, va, te),
          "AR(1,12)": fit_predict(f, "r", ["T1_lag1", "T1_lag12"], tr, va, te)}
    p = {("zero-parameter", "zero"): np.zeros(len(y)),
         ("zero-parameter", "known part (no change)"): known,
         ("zero-parameter", "persistence"): pers,
         ("direct", "3-feature ridge"): ar3, ("direct", "7-feature ridge"): r7}
    for name, rh in t1.items():
        p[("new part", f"known part + {name}")] = ratio * np.exp(rh) - 1
    rows = []
    c_id = f.loc[te, ID].to_numpy()
    m_id = f.loc[te, "month"].to_numpy()
    for a, b in (("known part + AR(1,12)", "3-feature ridge"),
                 ("known part (no change)", "persistence")):
        pa = p[("new part" if a.startswith("known part +") else "zero-parameter", a)]
        pb = p[("direct" if b == "3-feature ridge" else "zero-parameter", b)]
        d = (y - pa) ** 2 - (y - pb) ** 2
        rows.append({"design": "TEST", "model": f"{a} vs {b}", "n": len(y),
                     "p_country": wild(d, c_id, np.array([-1.0, 1.0])),
                     "p_month": wild(d, m_id, WEBB), "p_twoway": twoway(d, c_id, m_id)[0]})
    for (design, model), c in p.items():
        ok = c > -1
        implied = np.log((1 + c[ok]) / ratio[ok])
        rows.append({"design": design, "model": model, "n": len(y), "r2_T0": r2_score(y, c),
                     "r2_new_month": r2_score(r[ok], implied), "n_new_month": int(ok.sum())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 3. tests: the loss differential is the same under both scorings
# --------------------------------------------------------------------------- #
CONTRASTS = [  # (k, a, b)
    (12, "known part (no change)", "persistence"),
    (12, "known part (no change)", "3-feature ridge"),
    (12, "known part + 3-feature ridge", "3-feature ridge"),
    (12, "known part + 7-feature ridge", "7-feature ridge"),
    (12, "known part + AR(1,12)", "persistence"),
    (12, "known part + AR(1,12)", "3-feature ridge + known part as input"),
    (6, "known part + 3-feature ridge", "3-feature ridge"),
    (6, "known part + AR(1,12)", "persistence"),
    (3, "known part + 3-feature ridge", "3-feature ridge"),
    (3, "known part + AR(1,12)", "persistence"),
]


def tests(preds):
    rows = []
    for k, a, b in CONTRASTS:
        t = preds[k]
        y = t["y"].to_numpy()
        d = (y - t[a].to_numpy()) ** 2 - (y - t[b].to_numpy()) ** 2
        c = t.index.get_level_values(ID).to_numpy()
        m = t.index.get_level_values("month").to_numpy()
        rows.append({"k": k, "a": a, "b": b, "n": len(d), "mean_loss_diff": float(d.mean()),
                     "p_country": wild(d, c, np.array([-1.0, 1.0])),
                     "p_month": wild(d, m, WEBB), "p_twoway": twoway(d, c, m)[0],
                     "p_dm_month": dm_month(d, m)})
        print(f"k={k:2d} {a:38s} vs {b:38s} country {rows[-1]['p_country']:.4f} "
              f"month {rows[-1]['p_month']:.4f} two-way {rows[-1]['p_twoway']:.4f}", flush=True)
    out = pd.DataFrame(rows)
    for col in ("p_country", "p_month", "p_twoway", "p_dm_month"):
        out[col + "_holm"] = holm(out[col]).to_numpy()
    return out


# --------------------------------------------------------------------------- #
# 4. free R2 by period (no fitting)
# --------------------------------------------------------------------------- #
def periods(f):
    months = sorted(f["month"].unique())
    blocks = [("train 2021-12..2022-12", TRAIN), ("validation 2023", VALID), ("test 2024", TEST)]
    first = months.index("2021-12")
    for s in range(first, len(months), 6):
        blk = months[s:s + 6]
        if blk[-1] > TEST[1]:
            blk = [m for m in blk if m <= TEST[1]]
        if len(blk) >= 3:
            blocks.append((f"block {blk[0]}..{blk[-1]}", (blk[0], blk[-1])))
    rows = []
    for name, b in blocks:
        for k in KS:
            H, K = f"H{k}", f"K{k}"
            sel = hs.win(f, b) & f[[H, "r", K]].notna().all(axis=1)
            y, kk = f.loc[sel, H].to_numpy(float), f.loc[sel, K].to_numpy(float)
            pers = f.groupby(ID, sort=False)[H].shift(1)[sel].to_numpy(float)
            ok = ~np.isnan(pers)
            rows.append({"period": name, "k": k, "n": int(sel.sum()),
                         "free_r2_known_part": r2_score(y, kk),
                         "persistence_r2": r2_score(y[ok], pers[ok]),
                         "var_ratio_r_over_H": float(np.var(f.loc[sel, "r"]) / np.var(y))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 5. rolling refits (same blocks as seasonal_and_rolling: 6-month blocks,
#    previous 6 months for tuning, expanding training window)
# --------------------------------------------------------------------------- #
def rolling(f, k=12):
    import seasonal_and_rolling as sar
    H, K = f"H{k}", f"K{k}"
    months = sorted(f["month"].unique())
    first = months.index(TRAIN[0])
    parts = []
    for start in range(13, len(months), sar.HORIZON):
        block = months[start:start + sar.HORIZON]
        val = months[max(start - sar.HORIZON, 0):start]
        train = months[first:max(start - sar.HORIZON, 0)]
        if len(block) < 2 or len(train) < sar.MIN_TRAIN_MONTHS:
            continue
        ok = f[[H, "r", K]].notna().all(axis=1)
        tr, va, te = (f["month"].isin(m) & ok for m in (train, val, block))
        kk = f.loc[te, K].to_numpy(float)
        idx = pd.MultiIndex.from_frame(f.loc[te, [ID, "month"]])
        parts.append(pd.DataFrame({
            "y": f.loc[te, H].to_numpy(float), "r": f.loc[te, "r"].to_numpy(float), "K": kk,
            "known part (no change)": kk,
            "persistence": f.groupby(ID, sort=False)[H].shift(1)[te].to_numpy(float),
            "3-feature ridge": fit_predict(f, H, ["flow_ratio", f"{H}_lag1", "flow_ratio_lag1"],
                                           tr, va, te, True),
            "known part + 3-feature ridge": kk + fit_predict(
                f, "r", ["flow_ratio", "H1_lag1", "flow_ratio_lag1"], tr, va, te, True),
            "known part + AR(1,12)": kk + fit_predict(f, "r", ["T1_lag1", "T1_lag12"], tr, va, te),
        }, index=idx))
    t = pd.concat(parts)
    t = t[t["persistence"].notna()]
    models = ["known part (no change)", "persistence", "3-feature ridge",
              "known part + 3-feature ridge", "known part + AR(1,12)"]
    rows = [{"k": k, "model": m, "n": len(t), "months": t.index.get_level_values("month").nunique(),
             "r2_window": r2_score(t["y"], t[m]), "r2_new_month": r2_score(t["r"], t[m] - t["K"])}
            for m in models]
    c = t.index.get_level_values(ID).to_numpy()
    mo = t.index.get_level_values("month").to_numpy()
    for a, b in (("known part + 3-feature ridge", "3-feature ridge"),
                 ("known part + AR(1,12)", "persistence"),
                 ("known part (no change)", "3-feature ridge")):
        d = (t["y"] - t[a]).to_numpy() ** 2 - (t["y"] - t[b]).to_numpy() ** 2
        rows.append({"k": k, "model": f"TEST {a} vs {b}", "n": len(d), "months": len(set(mo)),
                     "p_country": wild(d, c, np.array([-1.0, 1.0])), "p_month": wild(d, mo, WEBB),
                     "p_twoway": twoway(d, c, mo)[0]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 6. figure
# --------------------------------------------------------------------------- #
def figure(tab):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = [("zero-parameter", "known part (no change)", "Known part, no model", "k", "-", "o"),
            ("zero-parameter", "persistence", "Persistence", "0.45", "--", "s"),
            ("direct", "3-feature ridge", "3-feature ridge, fit on window", "tab:red", "-", "^"),
            ("new part", "known part + 3-feature ridge", "3-feature ridge, fit on new month",
             "tab:blue", "-", "v"),
            ("new part", "known part + AR(1,12)", "AR(1,12), fit on new month", "tab:green", "-", "D")]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), sharex=True)
    for ax, col, title in ((axes[0], "r2_window", "Scored on the window $H_k$"),
                           (axes[1], "r2_new_month", "Scored on the new month only")):
        for design, model, label, color, ls, mk in show:
            s = tab[(tab["design"] == design) & (tab["model"] == model)].sort_values("k")
            ax.plot(s["k"], s[col], ls=ls, marker=mk, color=color, label=label, ms=4)
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xscale("log", base=2)
        ax.set_xticks(KS, [str(k) for k in KS])
        ax.set_xlabel("Window length $k$ (months)")
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("Test 2024 $R^2$")
    axes[1].set_ylim(-2.2, 0.5)
    axes[0].legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "known_part.pdf")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    f = load()
    tab, preds = window_table(f)
    tab.to_csv(OUT / "known_part.csv", index=False)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    for col in ("r2_window", "r2_new_month"):
        print(f"\n{col}")
        print(tab.pivot_table(index=["design", "model"], columns="k", values=col).round(3))
    t0 = t0_table(f)
    t0 = pd.concat([t0[t0["design"] != "TEST"], t0[t0["design"] == "TEST"]])
    t0.to_csv(OUT / "known_part_t0.csv", index=False)
    print("\nT0\n", t0.round(3).to_string(index=False))
    per = periods(f)
    per.to_csv(OUT / "known_part_periods.csv", index=False)
    print("\nfree R2 by period\n",
          per.pivot_table(index="period", columns="k", values="free_r2_known_part", sort=False)
          .round(3))
    figure(tab)
    tests(preds).to_csv(OUT / "known_part_tests.csv", index=False)
    roll = rolling(f)
    roll.to_csv(OUT / "known_part_rolling.csv", index=False)
    print("\nrolling\n", roll.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
