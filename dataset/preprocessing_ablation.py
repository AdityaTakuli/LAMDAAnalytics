"""Compare preprocessing variants on the paper's forward-chained country-month split.

Every variant sees the same rows, the same target, and the same forward chain
(train 2021-12..2022-12, validation 2023, test 2024). Only the feature
transformation changes, so a difference in test score is attributable to
preprocessing rather than to model capacity or to a different sample.

All transforms are fitted on training months only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, r2_score

NODES = Path(__file__).parent / "data/four_year_2021_2024/processed/nodes_monthly.csv"

BASE_FEATURES = [
    "inventory_days_proxy",
    "trade_delay_proxy",
    "news_vol_7d",
    "neg_tone_frac_3d",
    "strike_flag_7d",
    "weather_anomaly_7d",
    "global_risk",
]
PROTECTIVE = "inventory_days_proxy"

TRAIN = ("2021-12", "2022-12")
VALID = ("2023-01", "2023-12")
TEST = ("2024-01", "2024-11")
TAU = 0.20


def load() -> pd.DataFrame:
    frame = pd.read_csv(NODES)
    frame = frame[frame["target_valid"] == True].copy()  # noqa: E712
    frame["month"] = frame["month"].astype(str)
    return frame.sort_values(["host_country_id", "month"]).reset_index(drop=True)


def window(frame: pd.DataFrame, bounds: tuple[str, str]) -> pd.Series:
    return frame["month"].between(bounds[0], bounds[1])


# --------------------------------------------------------------------------- #
# Feature builders. Each returns (frame_with_columns, feature_names).
# Anything fitted uses train_mask only.
# --------------------------------------------------------------------------- #
def variant_baseline(frame: pd.DataFrame, train_mask: pd.Series):
    return frame.copy(), list(BASE_FEATURES)


def variant_log_news(frame: pd.DataFrame, train_mask: pd.Series):
    out = frame.copy()
    out["news_vol_log"] = np.log1p(out["news_vol_7d"].clip(lower=0))
    features = [f if f != "news_vol_7d" else "news_vol_log" for f in BASE_FEATURES]
    return out, features


def variant_country_relative_news(frame: pd.DataFrame, train_mask: pd.Series):
    """News volume relative to that country's own training-period level.

    Absolute GDELT counts are a proxy for how much a country is written about,
    not for how disrupted it is. The ratio to the country's own baseline is the
    quantity the feature was supposed to represent.
    """
    out = frame.copy()
    logged = np.log1p(out["news_vol_7d"].clip(lower=0))
    ref = logged[train_mask].groupby(out.loc[train_mask, "host_country_id"]).median()
    out["news_vol_rel"] = logged - out["host_country_id"].map(ref).fillna(logged[train_mask].median())
    features = [f if f != "news_vol_7d" else "news_vol_rel" for f in BASE_FEATURES]
    return out, features


def variant_drop_redundant(frame: pd.DataFrame, train_mask: pd.Series):
    """Collapse the two collinear Comtrade proxies into the single ratio they encode."""
    out = frame.copy()
    out["flow_ratio"] = out["inventory_days_proxy"] / 30.0
    features = ["flow_ratio", "news_vol_7d", "neg_tone_frac_3d",
                "strike_flag_7d", "weather_anomaly_7d", "global_risk"]
    return out, features


def variant_trade_only(frame: pd.DataFrame, train_mask: pd.Series):
    """Only the channels that showed non-trivial marginal correlation."""
    out = frame.copy()
    out["flow_ratio"] = out["inventory_days_proxy"] / 30.0
    return out, ["flow_ratio", "global_risk"]


def variant_all_fixes(frame: pd.DataFrame, train_mask: pd.Series):
    out = frame.copy()
    out["flow_ratio"] = out["inventory_days_proxy"] / 30.0
    logged = np.log1p(out["news_vol_7d"].clip(lower=0))
    ref = logged[train_mask].groupby(out.loc[train_mask, "host_country_id"]).median()
    out["news_vol_rel"] = logged - out["host_country_id"].map(ref).fillna(logged[train_mask].median())
    features = ["flow_ratio", "news_vol_rel", "neg_tone_frac_3d",
                "strike_flag_7d", "weather_anomaly_7d", "global_risk"]
    return out, features


def variant_all_fixes_plus_lags(frame: pd.DataFrame, train_mask: pd.Series):
    out, features = variant_all_fixes(frame, train_mask)
    grouped = out.groupby("host_country_id", sort=False)
    out["contraction_lag1"] = grouped["contraction"].shift(1).fillna(0.0)
    out["flow_ratio_lag1"] = grouped["flow_ratio"].shift(1).fillna(1.0)
    return out, [*features, "contraction_lag1", "flow_ratio_lag1"]


def _add_lags(out: pd.DataFrame, features: list[str], ratio_column: str):
    grouped = out.groupby("host_country_id", sort=False)
    out["contraction_lag1"] = grouped["contraction"].shift(1).fillna(0.0)
    out[f"{ratio_column}_lag1"] = grouped[ratio_column].shift(1).fillna(out[ratio_column].median())
    return out, [*features, "contraction_lag1", f"{ratio_column}_lag1"]


def variant_baseline_plus_lags(frame: pd.DataFrame, train_mask: pd.Series):
    """Isolates the lag contribution: paper's features, nothing else changed."""
    out, features = variant_baseline(frame, train_mask)
    return _add_lags(out, features, PROTECTIVE)


def variant_lags_only(frame: pd.DataFrame, train_mask: pd.Series):
    """Are the exogenous channels contributing anything beyond trade autocorrelation?"""
    out = frame.copy()
    out["flow_ratio"] = out["inventory_days_proxy"] / 30.0
    return _add_lags(out, ["flow_ratio"], "flow_ratio")


VARIANTS = {
    "A. baseline (current paper)": variant_baseline,
    "B. + log1p(news_vol)": variant_log_news,
    "C. + country-relative news": variant_country_relative_news,
    "D. + drop collinear delay proxy": variant_drop_redundant,
    "E. trade ratio + GSCPI only": variant_trade_only,
    "F. all fixes (C + D)": variant_all_fixes,
    "G. all fixes + causal lags": variant_all_fixes_plus_lags,
    "H. baseline + causal lags only": variant_baseline_plus_lags,
    "I. trade ratio + lags (no exog)": variant_lags_only,
}


def evaluate(frame: pd.DataFrame, builder) -> dict[str, float]:
    """Score on 2023 validation and 2024 test from the same 2022 training fit.

    A preprocessing change is only worth adopting if it helps in both years. A
    gain confined to 2024 is indistinguishable from fitting the test year.
    """
    train_mask = window(frame, TRAIN)
    built, features = builder(frame, train_mask)

    tr = window(built, TRAIN)
    folds = {"val": window(built, VALID), "test": window(built, TEST)}

    matrix = built[features].astype(float).copy()
    if PROTECTIVE in matrix.columns:
        matrix[PROTECTIVE] *= -1.0

    mean = matrix[tr].mean()
    scale = matrix[tr].std().replace(0.0, 1.0).fillna(1.0)
    scaled = ((matrix - mean) / scale).fillna(0.0)

    x_tr = scaled[tr].to_numpy()
    y_tr = built.loc[tr, "contraction"].to_numpy(float)
    c_tr = (built.loc[tr, "contraction"] < -TAU).to_numpy(int)

    ridge = Ridge(alpha=1.0).fit(x_tr, y_tr)
    logistic = LogisticRegression(max_iter=2000, class_weight="balanced").fit(x_tr, c_tr)

    result: dict[str, float] = {"n_features": len(features)}
    for name, mask in folds.items():
        x, y = scaled[mask].to_numpy(), built.loc[mask, "contraction"].to_numpy(float)
        c = (built.loc[mask, "contraction"] < -TAU).to_numpy(int)
        result[f"{name}_R2"] = r2_score(y, ridge.predict(x))
        result[f"{name}_PR"] = average_precision_score(c, logistic.predict_proba(x)[:, 1])
    return result


def main() -> None:
    frame = load()
    for label, bounds in (("val 2023", VALID), ("test 2024", TEST)):
        mask = window(frame, bounds)
        positives = int((frame.loc[mask, "contraction"] < -TAU).sum())
        print(f"{label}: rows={int(mask.sum())} positives={positives} "
              f"prevalence={positives / int(mask.sum()):.3f}")
    print()

    header = f"{'variant':34s} {'k':>3s} | {'val R2':>7s} {'test R2':>8s} | {'val PR':>7s} {'test PR':>8s}"
    print(header)
    print("-" * len(header))
    for name, builder in VARIANTS.items():
        r = evaluate(frame, builder)
        print(f"{name:34s} {r['n_features']:3d} | {r['val_R2']:7.3f} {r['test_R2']:8.3f} "
              f"| {r['val_PR']:7.3f} {r['test_PR']:8.3f}")
    print("-" * len(header))
    for label, bounds in (("val 2023", VALID), ("test 2024", TEST)):
        mask = window(frame, bounds)
        prevalence = float((frame.loc[mask, "contraction"] < -TAU).mean())
        print(f"  prevalence baseline {label}: R2=0.000  PR-AUC={prevalence:.3f}")


if __name__ == "__main__":
    main()
