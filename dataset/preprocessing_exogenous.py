"""Can better-engineered exogenous channels beat the AR baseline?

The strong competitor is now variant I from preprocessing_ablation.py: the trade
flow ratio plus past-only causal lags, with no news/weather/GSCPI at all. This
script rebuilds each exogenous channel from its raw source in a more defensible
form and measures its *marginal* value on top of that AR baseline, on both the
2023 validation year and the 2024 test year.

Improved channels, all fitted on training months only:
  * news   -> per-country z-score (train stats) + surprise vs trailing median
  * weather-> continuous monthly-mean-temperature deviation, country-standardised,
              replacing the within-7-day binary flag
  * gscpi  -> level is constant within a month, so we add month-over-month change
              and a one-month lag, which do carry cross-time information
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, r2_score

DATA = Path(__file__).parent / "data/four_year_2021_2024/processed"
TRAIN = ("2021-12", "2022-12")
VALID = ("2023-01", "2023-12")
TEST = ("2024-01", "2024-11")
TAU = 0.20


def between(months: pd.Series, bounds) -> pd.Series:
    return months.between(bounds[0], bounds[1])


def load_nodes() -> pd.DataFrame:
    frame = pd.read_csv(DATA / "nodes_monthly.csv")
    frame = frame[frame["target_valid"] == True].copy()  # noqa: E712
    frame["month"] = frame["month"].astype(str)
    return frame.sort_values(["host_country_id", "month"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Improved exogenous features, built from raw sources
# --------------------------------------------------------------------------- #
def monthly_temperature() -> pd.DataFrame:
    weather = pd.read_csv(DATA / "weather_daily.csv")
    weather["month"] = pd.to_datetime(weather["observed_date"], utc=True).dt.to_period("M").astype(str)
    monthly = (
        weather.groupby(["node_id", "month"])["temperature_c"].mean().reset_index()
        .rename(columns={"node_id": "host_country_id", "temperature_c": "temp_mean"})
    )
    return monthly


def add_weather_seasonal(frame: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    temp = monthly_temperature()
    out = frame.merge(temp, on=["host_country_id", "month"], how="left")
    train_temp = out.loc[train_mask.values, ["host_country_id", "temp_mean"]]
    stats = train_temp.groupby("host_country_id")["temp_mean"].agg(["mean", "std"])
    mu = out["host_country_id"].map(stats["mean"])
    sd = out["host_country_id"].map(stats["std"]).replace(0.0, np.nan)
    out["weather_temp_z"] = ((out["temp_mean"] - mu) / sd).fillna(0.0)
    out["weather_temp_absz"] = out["weather_temp_z"].abs()
    return out


def add_news_normalized(frame: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    out = frame.copy()
    logged = np.log1p(out["news_vol_7d"].clip(lower=0))
    train_log = logged[train_mask.values]
    stats = train_log.groupby(out.loc[train_mask.values, "host_country_id"]).agg(["mean", "std"])
    mu = out["host_country_id"].map(stats["mean"]).fillna(train_log.mean())
    sd = out["host_country_id"].map(stats["std"]).replace(0.0, np.nan).fillna(train_log.std() or 1.0)
    out["news_z"] = ((logged - mu) / sd).fillna(0.0)
    grouped = out.groupby("host_country_id", sort=False)
    trailing = grouped.apply(
        lambda g: np.log1p(g["news_vol_7d"].clip(lower=0)).shift(1).rolling(3, min_periods=1).median(),
        include_groups=False,
    ).reset_index(level=0, drop=True)
    out["news_surprise"] = (logged - trailing).fillna(0.0)
    return out


def add_gscpi_dynamics(frame: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    gscpi = pd.read_csv(DATA / "gscpi_monthly.csv")[["month", "global_risk"]].copy()
    gscpi["month"] = gscpi["month"].astype(str)
    gscpi = gscpi.sort_values("month")
    gscpi["gscpi_change"] = gscpi["global_risk"].diff()
    gscpi["gscpi_lag1"] = gscpi["global_risk"].shift(1)
    gscpi = gscpi.drop(columns=["global_risk"]).fillna(0.0)
    return frame.merge(gscpi, on="month", how="left")


# --------------------------------------------------------------------------- #
# AR baseline + optional exogenous groups
# --------------------------------------------------------------------------- #
def build(frame: pd.DataFrame, train_mask: pd.Series, groups: tuple[str, ...]):
    out = frame.copy()
    out["flow_ratio"] = out["inventory_days_proxy"] / 30.0
    grouped = out.groupby("host_country_id", sort=False)
    out["contraction_lag1"] = grouped["contraction"].shift(1).fillna(0.0)
    out["flow_ratio_lag1"] = grouped["flow_ratio"].shift(1).fillna(1.0)
    features = ["flow_ratio", "contraction_lag1", "flow_ratio_lag1"]

    if "news" in groups:
        out = add_news_normalized(out, train_mask)
        features += ["news_z", "news_surprise"]
    if "weather" in groups:
        out = add_weather_seasonal(out, train_mask)
        features += ["weather_temp_z", "weather_temp_absz"]
    if "gscpi" in groups:
        out = add_gscpi_dynamics(out, train_mask)
        features += ["gscpi_change", "gscpi_lag1"]
    if "raw_exog" in groups:  # the paper's original exogenous channels, unchanged
        features += ["news_vol_7d", "neg_tone_frac_3d", "strike_flag_7d",
                     "weather_anomaly_7d", "global_risk"]
    return out, features


def evaluate(frame: pd.DataFrame, groups: tuple[str, ...]) -> dict:
    train_mask = between(frame["month"], TRAIN)
    built, features = build(frame, train_mask, groups)

    tr = between(built["month"], TRAIN)
    matrix = built[features].astype(float).replace([np.inf, -np.inf], np.nan)
    mean = matrix[tr].mean()
    scale = matrix[tr].std().replace(0.0, 1.0).fillna(1.0)
    scaled = ((matrix - mean) / scale).fillna(0.0)

    y_tr = built.loc[tr, "contraction"].to_numpy(float)
    c_tr = (y_tr < -TAU).astype(int)
    ridge = Ridge(alpha=1.0).fit(scaled[tr].to_numpy(), y_tr)
    logistic = LogisticRegression(max_iter=2000, class_weight="balanced").fit(scaled[tr].to_numpy(), c_tr)

    result = {"k": len(features)}
    for name, bounds in (("val", VALID), ("test", TEST)):
        mask = between(built["month"], bounds)
        x = scaled[mask].to_numpy()
        y = built.loc[mask, "contraction"].to_numpy(float)
        c = (y < -TAU).astype(int)
        result[f"{name}_R2"] = r2_score(y, ridge.predict(x))
        result[f"{name}_PR"] = average_precision_score(c, logistic.predict_proba(x)[:, 1])
    return result


MODELS = {
    "AR baseline (trade ratio + lags)": (),
    "AR + improved news": ("news",),
    "AR + seasonal weather": ("weather",),
    "AR + GSCPI dynamics": ("gscpi",),
    "AR + all improved exog": ("news", "weather", "gscpi"),
    "AR + original raw exog": ("raw_exog",),
}


def main() -> None:
    frame = load_nodes()
    header = f"{'model':38s} {'k':>3s} | {'val R2':>7s} {'test R2':>8s} | {'val PR':>7s} {'test PR':>8s}"
    print(header)
    print("-" * len(header))
    for name, groups in MODELS.items():
        r = evaluate(frame, groups)
        print(f"{name:38s} {r['k']:3d} | {r['val_R2']:7.3f} {r['test_R2']:8.3f} "
              f"| {r['val_PR']:7.3f} {r['test_PR']:8.3f}")
    print("-" * len(header))
    print("  (a channel earns its place only if it improves BOTH val and test)")


if __name__ == "__main__":
    main()
