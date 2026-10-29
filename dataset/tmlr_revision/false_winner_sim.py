"""TMLR revision, extension: how often does a model comparison declare a false winner?

Monte Carlo calibrated to the panel. Each replication draws an 18-country monthly
panel of log imports whose one-step change has the panel's measured properties
(pre-2024, per-country standardised): lag-1 and lag-12 autocorrelation, the share of
variance common to all countries in a month, and the lag-1 and lag-12 autocorrelation
of the cross-country monthly mean. Per-country scales are the panel's own standard
deviations.

    x_{v,t} = w_{v,t} + n_{v,t} + s_{v,m(t)} + W_t + N_t + S_{m(t)}

w, W random walks; n, N white noise; s, S fixed month-of-year effects (lower case per
country, upper case shared). Imports are V = exp(scale_v x).

Two models are fitted with the paper's protocol (ridge, scalers fit on train 2021-12..
2022-12, penalty chosen on validation 2023 from the paper's grid, scored on 2024-01..11):
each uses the lag of its own target plus two inputs that carry no information about the
future. Model A's and model B's inputs are independent draws of the same kind, so
neither is better in population: every rejection of "equal expected loss" is a false
winner. Input kinds mirror the panel's exogenous channels:
    independent   white noise per country (weather: lag-1 AC about 0)
    country AR    AR(1), phi = 0.97, per country (news volume: 0.98)
    global AR     one AR(1), phi = 0.96, shared by all countries (GSCPI: 0.96)

Targets: H_k for k in {1,3,6,12} trained directly ("direct") or as the new month with
the known part added back ("new part", identical to k = 1 by construction), and T0.

Tests on the loss differential (two-sided, 5%): rows treated as independent; clustered
by country (CR1, t with 17 df); clustered by month (CR1, t with 10 df, the analytic
counterpart of the paper's month-level wild bootstrap and DM on month means);
two-way (Cameron, Gelbach and Miller 2011, t with 10 df). Optional --wild-check reruns
one cell with the paper's wild bootstrap by month (Webb weights).

Test lengths: 11 months (the paper) and 35 months (2024-01..2026-11) as a sensitivity.
Outputs (resumable, one file per scenario): out/false_winner/<scenario>.csv and
out/false_winner_summary.csv, out/false_winner_calibration.csv
Run from dataset/:  ../backend/.venv/bin/python tmlr_revision/false_winner_sim.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import t as tdist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

OUT = HERE / "out"
SIMDIR = OUT / "false_winner"
ALPHAS = np.array([0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])   # the paper's ridge grid
N_COUNTRIES = 18
FIRST = pd.Period("2020-01", "M")           # simulated history before the train window
TRAIN, VALID = ("2021-12", "2022-12"), ("2023-01", "2023-12")
KS = [1, 3, 6, 12]
REPS = 2000
FEATURES = {"independent": ("iid", 0.0), "country AR": ("country", 0.97),
            "global AR": ("global", 0.96)}
TEST_ENDS = {"11 months": "2024-11", "35 months": "2026-11"}


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
def panel_moments():
    from known_part import load
    f = load()
    pre = f[(f["month"] < "2024-01") & f["r"].notna()].copy()
    g = pre.groupby("host_country_id")["r"]
    pre["z"] = (pre["r"] - g.transform("mean")) / g.transform("std")
    full = f[["host_country_id", "month"]].merge(pre[["host_country_id", "month", "z"]],
                                                 how="left")
    zg = full.groupby("host_country_id")["z"]

    def ac(lag):
        ll = zg.shift(lag)
        ok = full["z"].notna() & ll.notna()
        return float(np.corrcoef(full["z"][ok], ll[ok])[0, 1])

    mm = pre.groupby("month")["z"].agg(["mean", "count"])
    n = mm["count"].mean()
    vb, vw = mm["mean"].var(), pre.groupby("month")["z"].var().mean()
    icc = (vb - vw / n) / (vb - vw / n + vw)
    return ({"ac1": ac(1), "ac12": ac(12), "icc": float(icc),
             "mm_ac1": float(mm["mean"].autocorr(1)), "mm_ac12": float(mm["mean"].autocorr(12))},
            g.std().to_numpy(float))


def implied(p, n=N_COUNTRIES):
    """Population moments of the standardised one-step change for variance components
    p = (w, n, s, W, N, S); month means include idiosyncratic noise / n."""
    w, nn, s, W, NN, S = p
    idio_v, idio_1, idio_12 = w + 2 * nn + 2 * s, -nn - s, 2 * s
    com_v, com_1, com_12 = W + 2 * NN + 2 * S, -NN - S, 2 * S
    v = idio_v + com_v
    mv = com_v + idio_v / n
    return {"ac1": (idio_1 + com_1) / v, "ac12": (idio_12 + com_12) / v, "icc": com_v / v,
            "mm_ac1": (com_1 + idio_1 / n) / mv, "mm_ac12": (com_12 + idio_12 / n) / mv}


def calibrate(target):
    keys = list(target)

    def loss(q):
        m = implied(np.exp(q))
        return sum((m[k] - target[k]) ** 2 for k in keys)

    best = None
    rng = np.random.default_rng(0)
    for _ in range(40):
        r = minimize(loss, rng.normal(size=6), method="Nelder-Mead",
                     options={"maxiter": 20000, "xatol": 1e-10, "fatol": 1e-14})
        if best is None or r.fun < best.fun:
            best = r
    p = np.exp(best.x)
    w, nn, s, W, NN, S = p
    p = p / (w + 2 * nn + 2 * s + W + 2 * NN + 2 * S)        # unit variance of the change
    return p, implied(p), float(best.fun)


# --------------------------------------------------------------------------- #
# simulation
# --------------------------------------------------------------------------- #
def months_until(end):
    return pd.period_range(FIRST, pd.Period(end, "M") + 1, freq="M")   # + target month


def simulate_levels(rng, p, scales, n_months):
    w, nn, s, W, NN, S = np.sqrt(p)
    N = N_COUNTRIES
    moy = np.arange(n_months) % 12
    x = (np.cumsum(rng.normal(scale=w, size=(N, n_months)), 1)
         + rng.normal(scale=nn, size=(N, n_months))
         + rng.normal(scale=s, size=(N, 12))[:, moy]
         + np.cumsum(rng.normal(scale=W, size=n_months))[None, :]
         + rng.normal(scale=NN, size=n_months)[None, :]
         + rng.normal(scale=S, size=12)[moy][None, :])
    sc = rng.choice(scales, size=N, replace=True)[:, None]
    return np.exp(sc * x)                                       # V > 0, log level scale_v * x


def features(rng, kind, n_months, nf=2):
    mode, phi = FEATURES[kind]
    N = N_COUNTRIES
    if mode == "iid":
        return rng.normal(size=(N, n_months, nf))
    shape = (N, n_months, nf) if mode == "country" else (1, n_months, nf)
    e = rng.normal(size=shape)
    a = np.zeros(shape)
    a[:, 0] = e[:, 0] / np.sqrt(1 - phi ** 2)
    for t in range(1, n_months):
        a[:, t] = phi * a[:, t - 1] + e[:, t]
    return np.broadcast_to(a, (N, n_months, nf)).copy()


def targets(V):
    """Row t (forecast date) holds what is known at t; targets use V_{t+1}."""
    lv = np.log(V)
    N, M = V.shape
    out = {}
    r = np.full((N, M), np.nan)
    r[:, :-1] = lv[:, 1:] - lv[:, :-1]
    out["r"] = r
    for k in KS:
        K = np.full((N, M), np.nan)
        K[:, k - 1:] = lv[:, k - 1:] - lv[:, :M - k + 1]
        out[f"K{k}"] = K
        out[f"H{k}"] = K + r
    med = np.full((N, M), np.nan)
    for t in range(M):
        med[:, t] = np.median(V[:, max(0, t - 11):t + 1], axis=1)
    t0 = np.full((N, M), np.nan)
    t0[:, :-1] = V[:, 1:] / med[:, :-1] - 1
    out["T0"] = t0
    return out


def lag1(a):
    b = np.full_like(a, np.nan)
    b[:, 1:] = a[:, :-1]
    return b


def ridge_fit_predict(X, y, tr, va, te):
    """X: (N, M, p). Standardise on train, pick alpha on validation R2, predict test."""
    Xtr, ytr = X[tr], y[tr]
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    Z = lambda m: (X[m] - mu) / sd  # noqa: E731
    Ztr, ym = Z(tr), ytr.mean()
    A = Ztr - Ztr.mean(0)
    G, b = A.T @ A, A.T @ (ytr - ym)
    Zva, yva = Z(va), y[va]
    best, bs = None, -np.inf
    for a in ALPHAS:
        beta = np.linalg.solve(G + a * np.eye(len(G)), b)
        icpt = ym - Ztr.mean(0) @ beta
        pv = Zva @ beta + icpt
        sc = 1 - np.mean((yva - pv) ** 2) / np.var(yva)
        if sc > bs:
            bs, best = sc, (beta, icpt)
    beta, icpt = best
    return Z(te) @ beta + icpt


def cr1_p(d, groups, df):
    codes, uniq = pd.factorize(groups)
    G = len(uniq)
    u = d - d.mean()
    s = np.bincount(codes, weights=u)
    v = (s @ s) / len(d) ** 2 * G / (G - 1)
    return float(2 * tdist.sf(abs(d.mean() / np.sqrt(v)), df)) if v > 0 else np.nan


def twoway_p(d, c, m):
    def cv(g):
        codes, _ = pd.factorize(g)
        s = np.bincount(codes, weights=u)
        return (s @ s) / len(u) ** 2
    u = d - d.mean()
    v = cv(c) + cv(m) - (u @ u) / len(u) ** 2
    if v <= 0:
        v = max(cv(c), cv(m))
    df = min(len(set(c)), len(set(m))) - 1
    return float(2 * tdist.sf(abs(d.mean() / np.sqrt(v)), df))


def one_rep(rng, p, scales, test_end, kinds):
    per = months_until(test_end)
    M = len(per)
    lab = np.array([str(q) for q in per])
    V = simulate_levels(rng, p, scales, M)
    T = targets(V)
    tr_m = (lab >= TRAIN[0]) & (lab <= TRAIN[1])
    va_m = (lab >= VALID[0]) & (lab <= VALID[1])
    te_m = (lab >= "2024-01") & (lab <= test_end)
    mask = lambda mm: np.broadcast_to(mm, (N_COUNTRIES, M))  # noqa: E731
    tr, va, te = mask(tr_m), mask(va_m), mask(te_m)
    cid = np.broadcast_to(np.arange(N_COUNTRIES)[:, None], (N_COUNTRIES, M))[te]
    mid = np.broadcast_to(np.arange(M)[None, :], (N_COUNTRIES, M))[te]
    rows = []
    for kind in kinds:
        FA, FB = features(rng, kind, M), features(rng, kind, M)
        cache = {}

        def pred(target, F, extra=None):
            key = (target, id(F), extra)
            if key not in cache:
                y = T[target]
                cols = [np.nan_to_num(lag1(y))[..., None]]
                if extra is not None:
                    cols.append(np.nan_to_num(T[extra])[..., None])
                X = np.concatenate(cols + [F], axis=2)
                cache[key] = ridge_fit_predict(X, y, tr, va, te)
            return cache[key]

        designs = [(f"H{k}", "direct", f"H{k}", None, None) for k in KS]
        designs += [(f"H{k}", "direct, known part as input", f"H{k}", None, f"K{k}")
                    for k in KS if k > 1]
        designs += [(f"H{k}", "new part", "r", f"K{k}", None) for k in KS if k > 1]
        designs += [("T0", "direct", "T0", None, None)]
        for tname, design, fit_on, kcol, extra in designs:
            y = T[tname][te]
            add = 0.0 if kcol is None else T[kcol][te]
            pa, pb = pred(fit_on, FA, extra) + add, pred(fit_on, FB, extra) + add
            d = (y - pa) ** 2 - (y - pb) ** 2
            rows.append({"features": kind, "target": tname, "design": design,
                         "r2_window_A": 1 - np.mean((y - pa) ** 2) / np.var(y),
                         "p_rows": float(2 * tdist.sf(abs(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))),
                                                      len(d) - 1)),
                         "p_country": cr1_p(d, cid, N_COUNTRIES - 1),
                         "p_month": cr1_p(d, mid, te_m.sum() - 1),
                         "p_twoway": twoway_p(d, cid, mid),
                         "d": d if WANT_D else None, "cid": cid, "mid": mid})
    return rows


WANT_D = False


def run_scenario(name, test_end, p, scales, reps):
    path = SIMDIR / f"{name}.csv"
    if path.exists():
        return pd.read_csv(path)
    rows = []
    for i in range(reps):
        rng = np.random.default_rng([20261029, list(TEST_ENDS).index(name), i])
        for rec in one_rep(rng, p, scales, test_end, list(FEATURES)):
            rec.pop("d"), rec.pop("cid"), rec.pop("mid")
            rec["rep"] = i
            rows.append(rec)
        if (i + 1) % 250 == 0:
            print(f"  {name}: {i + 1}/{reps}", flush=True)
    df = pd.DataFrame(rows)
    SIMDIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


WILD_CELLS = [("independent", "H12", "direct"), ("independent", "H12", "new part"),
              ("global AR", "H12", "direct"), ("global AR", "H1", "direct")]


def wild_check(p, scales, reps=1000, boot=399):
    """The paper's month-level wild bootstrap (Webb weights, null imposed) on four cells,
    next to the analytic month-clustered t, 11-month test."""
    from dependence_tests import WEBB
    global WANT_D
    WANT_D = True
    hits = {c: [0, 0] for c in WILD_CELLS}
    for i in range(reps):
        rng = np.random.default_rng([7, i])
        recs = one_rep(rng, p, scales, "2024-11", ["independent", "global AR"])
        br = np.random.default_rng([8, i])
        for rec in recs:
            cell = (rec["features"], rec["target"], rec["design"])
            if cell not in hits:
                continue
            d = rec["d"]
            codes, uniq = pd.factorize(rec["mid"])

            def tstat(x):
                u = x - x.mean()
                s = np.bincount(codes, weights=u)
                return x.mean() / np.sqrt((s @ s) / len(x) ** 2)
            t_obs = abs(tstat(d))
            cnt = sum(abs(tstat(d * br.choice(WEBB, size=len(uniq))[codes])) >= t_obs
                      for _ in range(boot))
            hits[cell][0] += rec["p_month"] < 0.05
            hits[cell][1] += (cnt + 1) / (boot + 1) < 0.05
    WANT_D = False
    return pd.DataFrame([{"features": c[0], "target": c[1], "design": c[2], "reps": reps,
                          "boot": boot, "reject_month_analytic": a / reps,
                          "reject_month_wild": w / reps} for c, (a, w) in hits.items()])


def summarise(frames):
    df = pd.concat(frames)
    g = df.groupby(["test_length", "features", "target", "design"], sort=False)
    out = g.agg(reps=("rep", "size"), mean_r2_window_A=("r2_window_A", "mean"),
                **{f"reject_{c}": (f"p_{c}", lambda s: float((s < 0.05).mean()))
                   for c in ("rows", "country", "month", "twoway")}).reset_index()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--wild-check", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    target, scales = panel_moments()
    p, fit, err = calibrate(target)
    cal = pd.DataFrame({"moment": list(target), "panel": list(target.values()),
                        "simulated_population": [fit[k] for k in target]})
    cal.to_csv(OUT / "false_winner_calibration.csv", index=False)
    print(cal.round(3).to_string(index=False), f"\nloss {err:.2e}")
    print("variance components (w, n, s, W, N, S):", np.round(p, 4))
    pd.Series(p, index=["w", "n", "s", "W", "N", "S"]).to_csv(OUT / "false_winner_components.csv")
    frames = []
    for name, end in TEST_ENDS.items():
        df = run_scenario(name, end, p, scales, args.reps)
        df["test_length"] = name
        frames.append(df)
    s = summarise(frames)
    s.to_csv(OUT / "false_winner_summary.csv", index=False)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(s.round(3).to_string(index=False))
    if args.wild_check:
        w = wild_check(p, scales)
        w.to_csv(OUT / "false_winner_wild_check.csv", index=False)
        print(w.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
