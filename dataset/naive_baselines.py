"""Phase 0.1 - Naive baseline ladder (BLOCKING).

Zero-parameter and single-statistic baselines on validation 2023 and test 2024,
for both regression and classification, on the paper's forward-chained split.
Anything fitted (per-country / global median) uses training months only.

Gate: compare persistence test R^2 against the fitted AR model's 0.578.
  * persistence R^2 well below ~0.35 -> the AR contribution is real.
  * persistence R^2 >= ~0.45      -> a zero-parameter rule matches fitted models;
                                     the headline must change.

Directional accuracy is reported for every model (the 55%-at-r=0.70 warning sign
must stay visible).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

DATA = Path(__file__).parent / "data/four_year_2021_2024/processed/nodes_monthly.csv"
TRAIN, VALID, TEST = ("2021-12", "2022-12"), ("2023-01", "2023-12"), ("2024-01", "2024-11")
TAU = 0.20


def load() -> pd.DataFrame:
    frame = pd.read_csv(DATA)
    frame["month"] = frame["month"].astype(str)
    frame = frame.sort_values(["host_country_id", "month"]).reset_index(drop=True)
    grouped = frame.groupby("host_country_id", sort=False)["contraction"]
    frame["persistence"] = grouped.shift(1)      # c_{v,T-1}
    frame["seasonal"] = grouped.shift(11)         # c_{v,T-11}
    return frame


def window(frame: pd.DataFrame, bounds) -> pd.Series:
    return frame["month"].between(bounds[0], bounds[1]) & frame["target_valid"].astype(bool)


def fitted_constants(frame: pd.DataFrame):
    train = frame[window(frame, TRAIN)]
    per_country = train.groupby("host_country_id")["contraction"].median()
    global_median = float(train["contraction"].median())
    return per_country, global_median


def predictions(frame: pd.DataFrame, mask: pd.Series, per_country, global_median) -> dict:
    sub = frame[mask]
    country_pred = sub["host_country_id"].map(per_country).fillna(global_median).to_numpy(float)
    return {
        "Zero": np.zeros(len(sub)),
        "Persistence c_(T-1)": sub["persistence"].fillna(0.0).to_numpy(float),
        "Seasonal c_(T-11)": sub["seasonal"].fillna(0.0).to_numpy(float),
        "Per-country median": country_pred,
        "Global median": np.full(len(sub), global_median),
    }


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    resid = pred - y
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    if np.std(pred) < 1e-12 or np.std(y) < 1e-12:
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(pred, y)[0, 1])
    diracc = float(np.mean(np.sign(pred) == np.sign(y)))

    labels = (y < -TAU).astype(int)
    score = -pred  # more-negative predicted contraction -> higher risk
    if labels.min() == labels.max():
        pr_auc = float(labels.mean())
    else:
        pr_auc = float(average_precision_score(labels, score))
    pred_pos = (pred < -TAU).astype(int)
    f1 = float(f1_score(labels, pred_pos, zero_division=0))
    return {"R2": r2, "RMSE": rmse, "MAE": mae, "r": pearson,
            "DirAcc": diracc, "PR-AUC": pr_auc, "F1@t": f1}


def report(frame: pd.DataFrame, label: str, bounds) -> None:
    per_country, global_median = fitted_constants(frame)
    mask = window(frame, bounds)
    sub = frame[mask]
    y = sub["contraction"].to_numpy(float)
    preds = predictions(frame, mask, per_country, global_median)

    prevalence = float((y < -TAU).mean())
    print(f"\n### {label}: n={len(y)}, positives={int((y < -TAU).sum())}, "
          f"prevalence={prevalence:.3f}")
    head = f"{'baseline':22s} {'R2':>7s} {'RMSE':>6s} {'MAE':>6s} {'r':>7s} {'DirAcc':>7s} {'PR-AUC':>7s} {'F1@t':>6s}"
    print(head)
    print("-" * len(head))
    for name, pred in preds.items():
        m = metrics(y, pred)
        r = f"{m['r']:7.3f}" if m["r"] == m["r"] else f"{'--':>7s}"
        print(f"{name:22s} {m['R2']:7.3f} {m['RMSE']:6.3f} {m['MAE']:6.3f} {r} "
              f"{m['DirAcc']:7.3f} {m['PR-AUC']:7.3f} {m['F1@t']:6.3f}")


def main() -> None:
    frame = load()
    filled = int(frame[window(frame, TEST)]["persistence"].isna().sum())
    print(f"Persistence NaNs filled with 0 on test window: {filled} (should be ~0)")
    report(frame, "VALIDATION 2023", VALID)
    report(frame, "TEST 2024", TEST)
    print("\nGate: compare Persistence TEST R2 against the fitted AR model R2 = 0.578.")


if __name__ == "__main__":
    main()
