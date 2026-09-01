"""Causal lag features for country-month tabular models."""

from __future__ import annotations

import pandas as pd

LAG_COLUMNS = (
    "contraction_lag1",
    "contraction_lag3_mean",
    "inbound_log_lag1",
    "inbound_roll3_mean",
)


def add_country_lags(frame: pd.DataFrame) -> pd.DataFrame:
    """Append past-only lags grouped by host country. No future leakage."""
    result = frame.sort_values(["host_country_id", "month"]).copy()
    grouped = result.groupby("host_country_id", sort=False)
    result["contraction_lag1"] = grouped["contraction"].shift(1)
    result["contraction_lag3_mean"] = grouped["contraction"].transform(
        lambda series: series.shift(1).rolling(3, min_periods=1).mean()
    )
    inbound = grouped["inbound_flow_usd"].shift(1)
    result["inbound_log_lag1"] = inbound.apply(lambda value: 0.0 if pd.isna(value) else float(__import__("numpy").log1p(max(float(value), 0.0))))
    result["inbound_roll3_mean"] = grouped["inbound_flow_usd"].transform(
        lambda series: series.shift(1).rolling(3, min_periods=1).mean()
    )
    for column in LAG_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)
    return result
