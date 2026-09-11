"""Robustness checks for the preprocessing change proposed in preprocessing_ablation.py.

Two questions a reviewer will ask about a headline improvement on one test year:

1. Does it survive being re-run with a different train/test origin?
2. Is the gap wider than the sampling noise on 193 test rows?

This script answers both by rolling the origin forward month by month and by
bootstrapping the paired difference between the baseline and the candidate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, r2_score

from preprocessing_ablation import (
    BASE_FEATURES,
    PROTECTIVE,
    TAU,
    load,
    variant_baseline,
    variant_lags_only,
)

CANDIDATES = {"baseline": variant_baseline, "trade ratio + lags": variant_lags_only}
SEED = 0


def fit_predict(frame: pd.DataFrame, builder, train_months, eval_months):
    train_mask = frame["month"].isin(train_months)
    built, features = builder(frame, train_mask)

    tr = built["month"].isin(train_months)
    ev = built["month"].isin(eval_months)
    if not tr.any() or not ev.any():
        return None

    matrix = built[features].astype(float).copy()
    if PROTECTIVE in matrix.columns:
        matrix[PROTECTIVE] *= -1.0
    mean, scale = matrix[tr].mean(), matrix[tr].std().replace(0.0, 1.0).fillna(1.0)
    scaled = ((matrix - mean) / scale).fillna(0.0)

    y_tr = built.loc[tr, "contraction"].to_numpy(float)
    c_tr = (y_tr < -TAU).astype(int)
    y_ev = built.loc[ev, "contraction"].to_numpy(float)
    c_ev = (y_ev < -TAU).astype(int)

    ridge = Ridge(alpha=1.0).fit(scaled[tr].to_numpy(), y_tr)
    prediction = ridge.predict(scaled[ev].to_numpy())

    score = None
    if c_tr.min() != c_tr.max():
        logistic = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
            scaled[tr].to_numpy(), c_tr
        )
        score = logistic.predict_proba(scaled[ev].to_numpy())[:, 1]
    return y_ev, prediction, c_ev, score


def rolling_origin(frame: pd.DataFrame, months: list[str], min_train: int = 13, horizon: int = 6):
    """Expanding-window origins; each evaluates the next `horizon` months."""
    print(f"{'train through':>14s} {'eval':>17s} {'n':>4s} | "
          f"{'base R2':>8s} {'cand R2':>8s} | {'base PR':>8s} {'cand PR':>8s}")
    print("-" * 78)
    rows = []
    for end in range(min_train, len(months) - horizon + 1, horizon):
        train_months, eval_months = months[:end], months[end:end + horizon]
        record = {"eval": f"{eval_months[0]}..{eval_months[-1]}"}
        for name, builder in CANDIDATES.items():
            out = fit_predict(frame, builder, train_months, eval_months)
            if out is None:
                continue
            y, prediction, c, score = out
            record[f"{name}_r2"] = r2_score(y, prediction)
            record[f"{name}_pr"] = average_precision_score(c, score) if score is not None and c.max() else np.nan
            record["n"] = len(y)
        rows.append(record)
        print(f"{train_months[-1]:>14s} {record['eval']:>17s} {record['n']:4d} | "
              f"{record['baseline_r2']:8.3f} {record['trade ratio + lags_r2']:8.3f} | "
              f"{record['baseline_pr']:8.3f} {record['trade ratio + lags_pr']:8.3f}")
    table = pd.DataFrame(rows)
    print("-" * 78)
    print(f"{'MEAN':>32s} {'':4s} | {table['baseline_r2'].mean():8.3f} "
          f"{table['trade ratio + lags_r2'].mean():8.3f} | "
          f"{table['baseline_pr'].mean():8.3f} {table['trade ratio + lags_pr'].mean():8.3f}")
    wins = int((table["trade ratio + lags_r2"] > table["baseline_r2"]).sum())
    print(f"candidate beats baseline on R2 in {wins}/{len(table)} origins")
    return table


def bootstrap_paired(frame: pd.DataFrame, train_months, eval_months, resamples: int = 5000):
    """Paired bootstrap on the same evaluation rows, so the two models share resamples."""
    results = {name: fit_predict(frame, builder, train_months, eval_months)
               for name, builder in CANDIDATES.items()}
    y, base_pred, c, base_score = results["baseline"]
    _, cand_pred, _, cand_score = results["trade ratio + lags"]

    rng = np.random.default_rng(SEED)
    deltas_r2, deltas_pr = [], []
    n = len(y)
    for _ in range(resamples):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        deltas_r2.append(r2_score(y[idx], cand_pred[idx]) - r2_score(y[idx], base_pred[idx]))
        if c[idx].max() > 0:
            deltas_pr.append(
                average_precision_score(c[idx], cand_score[idx])
                - average_precision_score(c[idx], base_score[idx])
            )

    for label, deltas in (("R2", deltas_r2), ("PR-AUC", deltas_pr)):
        array = np.array(deltas)
        low, high = np.percentile(array, [2.5, 97.5])
        beats = float((array > 0).mean())
        print(f"  delta {label:7s}: mean {array.mean():+.3f}  95% CI [{low:+.3f}, {high:+.3f}]  "
              f"P(candidate better) = {beats:.3f}")


def main() -> None:
    frame = load()
    months = sorted(frame["month"].unique())
    print(f"{len(months)} supervised months: {months[0]}..{months[-1]}\n")

    print("=== Rolling-origin evaluation (expanding window, 6-month blocks) ===")
    rolling_origin(frame, months)

    print("\n=== Paired bootstrap on the paper's 2024 test window (5000 resamples) ===")
    train = [m for m in months if "2021-12" <= m <= "2022-12"]
    test = [m for m in months if "2024-01" <= m <= "2024-11"]
    bootstrap_paired(frame, train, test)


if __name__ == "__main__":
    main()
