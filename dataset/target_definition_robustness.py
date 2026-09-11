"""Phase 0.2 - Denominator-overlap stress test (BLOCKING).

The current target T0 and the current best AR feature (contraction_lag1) share 11
of 12 denominator months, and V_T appears in both. This script re-runs the AR
model and the naive ladder under four target definitions with decreasing
denominator overlap, to decide whether the trade-autocorrelation result is a
mechanical artifact of target construction.

Targets (per country, V = inbound_flow_usd, fut = V_{T+1}, med12 = median V_{T-11..T}):
  T0  (fut - med12) / med12                         shared denominator 11/12  (== stored contraction)
  T1  log(fut / V_T)                                no shared denominator
  T2  (fut - med12) / B_v,  B_v = train-window per-country mean V   denominator constant in T
  T3  (fut - med12) / med12, but contraction_lag1 rebuilt with a
      denominator over months <= T-12                              feature/target denominators disjoint

AR model = Ridge on (flow_ratio, contraction_lag1, flow_ratio_lag1).
Persistence = previous-month value of the SAME target.

Gate:
  * R^2 holds near 0.5 across T1,T2,T3 -> mechanically sound; keep T0 primary + robustness table.
  * R^2 collapses under T1 or T3      -> trade-autocorrelation is a construction artifact;
                                          promote that to Finding 1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

DATA = Path(__file__).parent / "data/four_year_2021_2024/processed/nodes_monthly.csv"
TRAIN, VALID, TEST = ("2021-12", "2022-12"), ("2023-01", "2023-12"), ("2024-01", "2024-11")
TAU = 0.20


def load() -> pd.DataFrame:
    frame = pd.read_csv(DATA)
    frame["month"] = frame["month"].astype(str)
    frame = frame.sort_values(["host_country_id", "month"]).reset_index(drop=True)

    v = frame.groupby("host_country_id", sort=False)["inbound_flow_usd"]
    fut = v.shift(-1)
    med12 = v.transform(lambda s: s.rolling(12, min_periods=1).median())
    base_disjoint = v.transform(lambda s: s.rolling(12, min_periods=1).median().shift(12))

    train_mask = frame["month"].between(*TRAIN)
    base_mean = (frame[train_mask].groupby("host_country_id")["inbound_flow_usd"].mean())
    b_v = frame["host_country_id"].map(base_mean)

    frame["T0"] = (fut - med12) / med12.replace(0, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        frame["T1"] = np.log(fut / frame["inbound_flow_usd"].where(frame["inbound_flow_usd"] > 0))
        frame.loc[fut <= 0, "T1"] = np.nan
    frame["T2"] = (fut - med12) / b_v.replace(0, np.nan)
    frame["T3"] = frame["T0"]

    # AR features
    frame["flow_ratio"] = frame["inventory_days_proxy"] / 30.0
    fr = frame.groupby("host_country_id", sort=False)["flow_ratio"]
    frame["flow_ratio_lag1"] = fr.shift(1)
    frame["contraction_lag1"] = frame.groupby("host_country_id", sort=False)["T0"].shift(1)
    frame["contraction_lag1_disjoint"] = (frame["inbound_flow_usd"] - base_disjoint) / base_disjoint.replace(0, np.nan)
    return frame


# feature set per target; T3 swaps in the disjoint-denominator lag
def ar_features(target: str) -> list[str]:
    lag = "contraction_lag1_disjoint" if target == "T3" else "contraction_lag1"
    return ["flow_ratio", lag, "flow_ratio_lag1"]


def r2(y, pred) -> float:
    ss_res = float(np.sum((pred - y) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def diracc(y, pred) -> float:
    return float(np.mean(np.sign(pred) == np.sign(y)))


def eval_target(frame: pd.DataFrame, target: str) -> dict:
    features = ar_features(target)
    needed = [target, *features]
    finite = frame[needed].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)

    tr = frame["month"].between(*TRAIN) & finite
    te = frame["month"].between(*TEST) & finite

    # AR model
    scaler_src = frame.loc[tr, features].astype(float)
    mean, scale = scaler_src.mean(), scaler_src.std().replace(0.0, 1.0).fillna(1.0)
    x_tr = ((frame.loc[tr, features].astype(float) - mean) / scale).to_numpy()
    x_te = ((frame.loc[te, features].astype(float) - mean) / scale).to_numpy()
    y_tr = frame.loc[tr, target].to_numpy(float)
    y_te = frame.loc[te, target].to_numpy(float)
    ridge = Ridge(alpha=1.0).fit(x_tr, y_tr)
    ar_pred = ridge.predict(x_te)

    # Persistence of the same target
    persist = frame.groupby("host_country_id", sort=False)[target].shift(1)
    pe = te & persist.notna()
    y_pe = frame.loc[pe, target].to_numpy(float)
    p_pred = persist[pe].to_numpy(float)

    return {
        "n_test": int(te.sum()),
        "ar_r2": r2(y_te, ar_pred), "ar_dir": diracc(y_te, ar_pred),
        "persist_n": int(pe.sum()),
        "persist_r2": r2(y_pe, p_pred), "persist_dir": diracc(y_pe, p_pred),
    }


def main() -> None:
    frame = load()
    overlap = {"T0": "11/12", "T1": "none", "T2": "constant in T", "T3": "disjoint"}
    print("Phase 0.2 - test 2024 R^2 (and directional accuracy) by target definition\n")
    head = (f"{'target':4s} {'denom overlap':14s} {'n':>4s} | "
            f"{'AR R2':>7s} {'AR dir':>7s} | {'persist R2':>11s} {'persist dir':>11s}")
    print(head)
    print("-" * len(head))
    for target in ("T0", "T1", "T2", "T3"):
        r = eval_target(frame, target)
        print(f"{target:4s} {overlap[target]:14s} {r['n_test']:4d} | "
              f"{r['ar_r2']:7.3f} {r['ar_dir']:7.3f} | "
              f"{r['persist_r2']:11.3f} {r['persist_dir']:11.3f}")
    print("-" * len(head))
    print("\nGate: does AR/persistence R^2 hold near 0.5 across T1, T2, T3, or collapse?")


if __name__ == "__main__":
    main()
