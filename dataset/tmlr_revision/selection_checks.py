"""TMLR revision, reviewer point 7: does the conclusion depend on the panel selection?

(A) Leave one country out. The original 18-country panel is refit 18 times with one
    country removed from training, validation and test; tabular models only.
(B) Different selection rules. The panel was chosen as India plus India's 17 largest
    reporting partners in HS 8541/8542. Here the same rule is applied with China, the
    United States, Germany, or Japan as the focal country. Candidates are limited to
    the 41 countries for which monthly bilateral data were already downloaded (the
    extended panel in data/four_year_2021_2024/processed_v3/semiconductor), so the
    universe the partners come from is still India-centred; the focal country and
    the ranking change. India itself is rebuilt the same way as a check against the
    published panel.

For each panel: lag-1 autocorrelation of T1, T4, T0 and persistence R2 on each
(target-design result), and T1 test-2024 R2 of persistence, AR(1), AR(1,12), the
three-feature AR ridge, the same ridge plus all exogenous channels, the
seven-feature ridge and a random forest (model result). Unpublished months stored
as zero are treated as missing. Graph models are not refit (cost).

Output: out/selection_loco.csv, out/selection_focal.csv, out/selection_focal_members.csv
Run from dataset/:  ../backend/.venv/bin/python tmlr_revision/selection_checks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

HERE = Path(__file__).resolve().parent
DATASET = HERE.parent
sys.path.insert(0, str(DATASET))

from phase0_t1_ladder import PAPER, RF_R, RIDGE, standardize, tune_reg, win  # noqa: E402
from preprocessing_exogenous import add_gscpi_dynamics, add_news_normalized  # noqa: E402

OUT = HERE / "out"
V3 = DATASET / "data/four_year_2021_2024/processed_v3"
PROC = DATASET / "data/four_year_2021_2024/processed"
TRAIN, VALID, TEST = ("2021-12", "2022-12"), ("2023-01", "2023-12"), ("2024-01", "2024-11")
AR3 = ["flow_ratio", "contraction_lag1", "flow_ratio_lag1"]
EXO = ["news_z", "news_surprise", "weather_temp_z", "weather_temp_absz", "gscpi_change", "gscpi_lag1"]
FOCALS = {"India": "country_in", "China": "country_cn", "United States": "country_us",
          "Germany": "country_de", "Japan": "country_jp"}


# --------------------------------------------------------------------------- #
# panel construction
# --------------------------------------------------------------------------- #
def targets(f: pd.DataFrame) -> pd.DataFrame:
    f = f.sort_values(["host_country_id", "month"]).reset_index(drop=True)
    f["inbound_flow_usd"] = f["inbound_flow_usd"].where(f["inbound_flow_usd"] > 0)  # zero = unpublished
    v = f.groupby("host_country_id", sort=False)["inbound_flow_usd"]
    fut = v.shift(-1)
    med12 = v.transform(lambda s: s.rolling(12, min_periods=1).median())
    f["T0"] = (fut - med12) / med12
    f.loc[f["inbound_flow_usd"].isna(), "T0"] = np.nan
    f["T1"] = np.log(fut / f["inbound_flow_usd"])
    f["T4"] = np.log(fut / v.shift(11))
    g = f.groupby("host_country_id", sort=False)
    for t in ("T0", "T1", "T4"):
        f[f"{t}_lag1"] = g[t].shift(1)
    f["T1_lag12"] = g["T1"].shift(12)
    f["flow_ratio"] = f["inventory_days_proxy"] / 30.0
    f["flow_ratio_lag1"] = g["flow_ratio"].shift(1)
    f["contraction_lag1"] = f["T0_lag1"]
    return f


def monthly_temp(path: Path) -> pd.DataFrame:
    w = pd.read_csv(path, usecols=["node_id", "observed_date", "temperature_c"])
    w["month"] = pd.to_datetime(w["observed_date"], utc=True).dt.strftime("%Y-%m")
    return (w.groupby(["node_id", "month"])["temperature_c"].mean().reset_index()
            .rename(columns={"node_id": "host_country_id", "temperature_c": "temp_mean"}))


def add_weather(frame: pd.DataFrame, tr: pd.Series, temp: pd.DataFrame) -> pd.DataFrame:
    out = frame.merge(temp, on=["host_country_id", "month"], how="left")
    stats = out.loc[tr.values].groupby("host_country_id")["temp_mean"].agg(["mean", "std"])
    mu = out["host_country_id"].map(stats["mean"])
    sd = out["host_country_id"].map(stats["std"]).replace(0.0, np.nan)
    out["weather_temp_z"] = ((out["temp_mean"] - mu) / sd).fillna(0.0)
    out["weather_temp_absz"] = out["weather_temp_z"].abs()
    return out


def proxies(f: pd.DataFrame) -> pd.DataFrame:
    """Trade proxies from inbound at months <= T, zero inbound treated as unpublished."""
    f = f.sort_values(["host_country_id", "month"]).reset_index(drop=True)
    v = f["inbound_flow_usd"].where(f["inbound_flow_usd"] > 0)
    base = v.groupby(f["host_country_id"]).transform(lambda s: s.shift(1).rolling(3, min_periods=1).median())
    q = v / base
    f["inventory_days_proxy"] = (30 * q.clip(lower=1 / 6, upper=3)).fillna(30.0)
    f["trade_delay_proxy"] = (30 * (1 - q).clip(lower=0)).clip(upper=60).fillna(0.0)
    return f


def focal_members(edges: pd.DataFrame, focal: str, k: int = 17) -> tuple[list[str], float]:
    """Focal country plus the k downloaded reporters that import the most from it
    (HS 8541/8542, 2021-2024): the focal country's largest export markets."""
    out = edges[(edges["source"] == focal) & (edges["destination"] != focal)]
    rank = out.groupby("destination")["trade_value_usd"].sum().sort_values(ascending=False)
    return [focal] + rank.index[:k].tolist(), float(rank.iloc[:k].sum() / rank.sum())


def panel_from(v3: pd.DataFrame, members: list[str]) -> pd.DataFrame:
    """Reporter set `members`; each reporter's inbound is its imports from the 18
    partner countries of the download (as in the extended panel)."""
    return targets(proxies(v3[v3["host_country_id"].isin(members)].copy()))


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def fit_predict(f, target, feats, est=Ridge, cfgs=RIDGE, frame=None):
    g = f if frame is None else frame
    valid = f[target].notna()
    tr, va, te = win(f, TRAIN) & valid, win(f, VALID) & valid, win(f, TEST) & valid
    z = standardize(g, feats, tr)
    m = tune_reg(z[tr].to_numpy(), g.loc[tr, target].to_numpy(float),
                 z[va].to_numpy(), g.loc[va, target].to_numpy(float), est, cfgs)
    return r2_score(f.loc[te, target], m.predict(z[te].to_numpy()))


def evaluate(f: pd.DataFrame, temp: pd.DataFrame) -> dict:
    f = f.reset_index(drop=True)
    te = win(f, TEST) & f["T1"].notna()
    tr = win(f, TRAIN) & f["T1"].notna()
    rec = {"n_test_T1": int(te.sum()), "countries_T1": int(f.loc[te, "host_country_id"].nunique())}
    for t in ("T1", "T4", "T0"):
        ok = f[t].notna() & f[f"{t}_lag1"].notna()
        rec[f"ac_{t}"] = float(np.corrcoef(f.loc[ok, t], f.loc[ok, f"{t}_lag1"])[0, 1])
        m = win(f, TEST) & f[t].notna()
        rec[f"persistence_{t}"] = r2_score(f.loc[m, t], f.loc[m, f"{t}_lag1"].fillna(0.0))
    rec["ar1"] = fit_predict(f, "T1", ["T1_lag1"])
    rec["ar1_12"] = fit_predict(f, "T1", ["T1_lag1", "T1_lag12"])
    rec["ridge3"] = fit_predict(f, "T1", AR3)
    g = add_gscpi_dynamics(add_weather(add_news_normalized(f.copy(), tr), tr, temp), tr)
    rec["ridge3_exog"] = fit_predict(f, "T1", AR3 + EXO, frame=g)
    rec["ridge7"] = fit_predict(f, "T1", PAPER)
    rec["random_forest"] = fit_predict(f, "T1", AR3, RandomForestRegressor, RF_R)
    rec["best_model"] = max(("ar1", "ar1_12", "ridge3", "ridge3_exog", "ridge7", "random_forest"),
                            key=lambda k: rec[k])
    return rec


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # (A) leave one country out on the published panel
    pub = pd.read_csv(PROC / "nodes_monthly.csv")
    pub["month"] = pub["month"].astype(str)
    temp_pub = monthly_temp(PROC / "weather_daily.csv")
    rows = [{"left_out": "(none)", **evaluate(targets(pub.copy()), temp_pub)}]
    for c in sorted(pub["host_country_id"].unique()):
        rows.append({"left_out": c, **evaluate(targets(pub[pub["host_country_id"] != c].copy()), temp_pub)})
        print(f"LOCO {c:12s} ar1={rows[-1]['ar1']:.3f} ar1_12={rows[-1]['ar1_12']:.3f} "
              f"ridge3={rows[-1]['ridge3']:.3f} +exog={rows[-1]['ridge3_exog']:.3f} "
              f"best={rows[-1]['best_model']}", flush=True)
    loco = pd.DataFrame(rows)
    loco.to_csv(OUT / "selection_loco.csv", index=False)

    # (B) different selection rules over the 41 downloaded reporters
    edges = pd.read_csv(V3 / "semiconductor/edges_monthly.csv",
                        usecols=["month", "source", "destination", "trade_value_usd"])
    v3 = pd.read_csv(V3 / "semiconductor/nodes_monthly.csv")
    v3["month"] = v3["month"].astype(str)
    v3 = v3[["host_country_id", "month", "inbound_flow_usd", "news_vol_7d", "neg_tone_frac_3d",
             "strike_flag_7d", "weather_anomaly_7d", "global_risk"]]
    temp_v3 = monthly_temp(V3 / "weather_daily.csv")
    published = sorted(pub["host_country_id"].unique())
    panels = {"India (published rule, rebuilt)": (published, np.nan)}
    for name, code in FOCALS.items():
        if name != "India":
            panels[f"{name}: + its 17 largest export markets"] = focal_members(edges, code)
    added = sorted(set(v3["host_country_id"]) - set(published))
    panels["The 23 added reporters only"] = (added, np.nan)
    panels["All 41 reporters"] = (sorted(v3["host_country_id"].unique()), np.nan)
    focal_rows, member_rows = [], []
    for name, (members, share) in panels.items():
        f = panel_from(v3, members)
        rec = {"panel": name, "members": len(members),
               "overlap_with_published_panel": len(set(members) & set(published)),
               "share_of_focal_exports_to_the_41": share, **evaluate(f, temp_v3)}
        if name.startswith("India"):
            chk = f.merge(targets(pub.copy())[["host_country_id", "month", "T1"]],
                          on=["host_country_id", "month"], suffixes=("", "_pub"))
            ok = chk["T1"].notna() & chk["T1_pub"].notna()
            rec["india_rebuild_max_T1_diff"] = float((chk.loc[ok, "T1"] - chk.loc[ok, "T1_pub"]).abs().max())
        focal_rows.append(rec)
        member_rows += [{"panel": name, "rank": i, "country": c} for i, c in enumerate(members)]
        print(f"{name:42s} n={len(members):2d} overlap={rec['overlap_with_published_panel']:2d} "
              f"ac T1/T4/T0={rec['ac_T1']:+.3f}/{rec['ac_T4']:+.3f}/{rec['ac_T0']:+.3f} "
              f"persist T1/T4/T0={rec['persistence_T1']:+.3f}/{rec['persistence_T4']:+.3f}/"
              f"{rec['persistence_T0']:+.3f}  ar1={rec['ar1']:.3f} ar1_12={rec['ar1_12']:.3f} "
              f"ridge3={rec['ridge3']:.3f} +exog={rec['ridge3_exog']:.3f} ridge7={rec['ridge7']:.3f} "
              f"rf={rec['random_forest']:.3f} best={rec['best_model']}", flush=True)
    pd.DataFrame(focal_rows).to_csv(OUT / "selection_focal.csv", index=False)
    pd.DataFrame(member_rows).to_csv(OUT / "selection_focal_members.csv", index=False)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
