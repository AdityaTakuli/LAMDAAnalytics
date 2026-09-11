"""Tuned, leakage-safe comparison against the numbers reported in paper.tex.

Discipline enforced here, which the headline single-alpha runs did not:

  * Every model is FIT on the training months (2021-12..2022-12).
  * Every hyperparameter is SELECTED on the 2023 validation months.
  * The 2024 test months are touched exactly once, to report the selected model.

Two feature sets are compared on identical rows and identical splits:

  * ``paper``    : the seven features of Table 'tab:features' (current pipeline).
  * ``improved`` : trade flow ratio + past-only causal lags (the change validated
                   in preprocessing_ablation.py / preprocessing_robustness.py).

The regression column is directly comparable to paper Table 'tab:reg' (test R^2)
and the classification column to paper Table 'tab:clf' (test PR-AUC).
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, r2_score

DATA = Path(__file__).parent / "data/four_year_2021_2024/processed"
TRAIN, VALID, TEST = ("2021-12", "2022-12"), ("2023-01", "2023-12"), ("2024-01", "2024-11")
TAU = 0.20

PAPER_FEATURES = [
    "inventory_days_proxy", "trade_delay_proxy", "news_vol_7d", "neg_tone_frac_3d",
    "strike_flag_7d", "weather_anomaly_7d", "global_risk",
]
PROTECTIVE = "inventory_days_proxy"

# Paper-reported test numbers, for a side-by-side sanity column.
PAPER_REG_R2 = {"Ridge": 0.349, "Random forest": 0.042, "LightGBM": -0.161, "LightGBM+lags": 0.110}
PAPER_CLF_PR = {"Logistic": 0.413, "Random forest": None, "LightGBM": 0.355, "Hand-weighted linear": 0.388}


def load() -> pd.DataFrame:
    frame = pd.read_csv(DATA / "nodes_monthly.csv")
    frame = frame[frame["target_valid"] == True].copy()  # noqa: E712
    frame["month"] = frame["month"].astype(str)
    return frame.sort_values(["host_country_id", "month"]).reset_index(drop=True)


def in_window(frame, bounds):
    return frame["month"].between(bounds[0], bounds[1])


def make_features(frame: pd.DataFrame, which: str) -> tuple[pd.DataFrame, list[str]]:
    out = frame.copy()
    if which == "paper":
        return out, list(PAPER_FEATURES)
    out["flow_ratio"] = out["inventory_days_proxy"] / 30.0
    grouped = out.groupby("host_country_id", sort=False)
    out["contraction_lag1"] = grouped["contraction"].shift(1).fillna(0.0)
    out["flow_ratio_lag1"] = grouped["flow_ratio"].shift(1).fillna(1.0)
    return out, ["flow_ratio", "contraction_lag1", "flow_ratio_lag1"]


def matrices(frame: pd.DataFrame, features: list[str]):
    """Standardise on train stats; return (X, y_reg, y_clf) per split."""
    tr = in_window(frame, TRAIN)
    matrix = frame[features].astype(float).replace([np.inf, -np.inf], np.nan)
    if PROTECTIVE in matrix.columns:
        matrix[PROTECTIVE] = -matrix[PROTECTIVE]
    mean = matrix[tr].mean()
    scale = matrix[tr].std().replace(0.0, 1.0).fillna(1.0)
    scaled = ((matrix - mean) / scale).fillna(0.0)

    out = {}
    for name, bounds in (("train", TRAIN), ("val", VALID), ("test", TEST)):
        mask = in_window(frame, bounds)
        y = frame.loc[mask, "contraction"].to_numpy(float)
        out[name] = (scaled[mask].to_numpy(), y, (y < -TAU).astype(int))
    return out


# --------------------------------------------------------------------------- #
# Hyperparameter grids (kept modest: 234 training rows punish complexity)
# --------------------------------------------------------------------------- #
def grid(space: dict) -> list[dict]:
    keys = list(space)
    return [dict(zip(keys, combo)) for combo in product(*space.values())]


REG_MODELS = {
    "Ridge": (Ridge, grid({"alpha": [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]})),
    "Random forest": (RandomForestRegressor, grid({
        "n_estimators": [300], "max_depth": [2, 3, 4, None],
        "min_samples_leaf": [1, 5, 10], "random_state": [0]})),
    "LightGBM": (LGBMRegressor, grid({
        "n_estimators": [100, 300], "num_leaves": [7, 15, 31],
        "learning_rate": [0.02, 0.05], "min_child_samples": [5, 20],
        "verbosity": [-1], "random_state": [0]})),
}
CLF_MODELS = {
    "Logistic": (LogisticRegression, grid({
        "C": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0], "max_iter": [2000],
        "class_weight": ["balanced"]})),
    "Random forest": (RandomForestClassifier, grid({
        "n_estimators": [300], "max_depth": [2, 3, 4, None],
        "min_samples_leaf": [1, 5, 10], "class_weight": ["balanced"], "random_state": [0]})),
    "LightGBM": (LGBMClassifier, grid({
        "n_estimators": [100, 300], "num_leaves": [7, 15, 31],
        "learning_rate": [0.02, 0.05], "min_child_samples": [5, 20],
        "class_weight": ["balanced"], "verbosity": [-1], "random_state": [0]})),
}


def tune_regression(splits, estimator, configs):
    x_tr, y_tr, _ = splits["train"]
    x_val, y_val, _ = splits["val"]
    x_te, y_te, _ = splits["test"]
    best, best_val = None, -np.inf
    for config in configs:
        model = estimator(**config).fit(x_tr, y_tr)
        score = r2_score(y_val, model.predict(x_val))
        if score > best_val:
            best_val, best = score, (config, model)
    config, model = best
    return best_val, r2_score(y_te, model.predict(x_te)), config


def tune_classification(splits, estimator, configs):
    x_tr, _, c_tr = splits["train"]
    x_val, _, c_val = splits["val"]
    x_te, _, c_te = splits["test"]
    best, best_val = None, -np.inf
    for config in configs:
        model = estimator(**config).fit(x_tr, c_tr)
        proba = model.predict_proba(x_val)[:, 1]
        score = average_precision_score(c_val, proba)
        if score > best_val:
            best_val, best = score, (config, model)
    config, model = best
    proba_te = model.predict_proba(x_te)[:, 1]
    return best_val, average_precision_score(c_te, proba_te), config


def run(task: str) -> None:
    frame = load()
    models = REG_MODELS if task == "regression" else CLF_MODELS
    tune = tune_regression if task == "regression" else tune_classification
    metric = "R2" if task == "regression" else "PR-AUC"
    paper_map = PAPER_REG_R2 if task == "regression" else PAPER_CLF_PR

    print(f"\n================ {task.upper()}  (test metric = {metric}) ================")
    header = (f"{'model':16s} {'features':9s} | {'val':>7s} {'test':>7s} "
              f"| {'paper':>7s} {'delta':>7s} | selected hyperparameters")
    print(header)
    print("-" * len(header))
    for feature_set in ("paper", "improved"):
        built, features = make_features(frame, feature_set)
        splits = matrices(built, features)
        for name, (estimator, configs) in models.items():
            val, test, config = tune(splits, estimator, configs)
            paper = paper_map.get(name) if feature_set == "paper" else None
            delta = f"{test - paper:+.3f}" if paper is not None else ""
            paper_str = f"{paper:.3f}" if paper is not None else "--"
            short = {k: v for k, v in config.items()
                     if k not in ("random_state", "verbosity", "max_iter", "class_weight", "n_estimators")}
            print(f"{name:16s} {feature_set:9s} | {val:7.3f} {test:7.3f} "
                  f"| {paper_str:>7s} {delta:>7s} | {short}")
        print("-" * len(header))


def main() -> None:
    run("regression")
    run("classification")
    print("\nval = 2023 (used for selection), test = 2024 (reported once).")


if __name__ == "__main__":
    main()
