"""Fuse source tables into causal monthly node and edge records.

The fusion stage is the point where temporal semantics are enforced:

* features for month ``T`` use observations whose reference and availability
  timestamps are no later than the last day of ``T``;
* GDELT uses trailing seven-day and three-day windows anchored at that date;
* Comtrade inbound flow is kept at country level and is never allocated to
  firms;
* the e-commerce graph uses country nodes only; no firm topology is inferred;
* labels use only inbound flow and are created for every configured tau.

The module exposes pure functions so tests and training can operate on local
or synthetic DataFrames without any network access.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from common import (
    add_common_cli,
    load_config,
    nested,
    read_table,
    stable_id,
    write_table,
)

FEATURES = [
    "inventory_days_proxy",
    "trade_delay_proxy",
    "news_vol_7d",
    "neg_tone_frac_3d",
    "strike_flag_7d",
    "weather_anomaly_7d",
    "global_risk",
]


def _timestamp_series(frame: pd.DataFrame, column: str, fallback: Any = None) -> pd.Series:
    if column not in frame:
        if isinstance(fallback, pd.Series):
            return pd.to_datetime(fallback, errors="coerce", utc=True).reindex(frame.index)
        return pd.Series(pd.to_datetime(fallback, errors="coerce", utc=True), index=frame.index)
    values = pd.to_datetime(frame[column], errors="coerce", utc=True)
    if fallback is not None:
        if isinstance(fallback, pd.Series):
            fallback_values = pd.to_datetime(fallback, errors="coerce", utc=True).reindex(frame.index)
            values = values.fillna(fallback_values)
        else:
            values = values.fillna(pd.to_datetime(fallback, errors="coerce", utc=True))
    return values


def _country_id_from_code(code: Any, config: dict) -> str:
    value = str(code).strip()
    if value.lower() in {"", "nan", "none", "0", "0.0"}:
        return "country_0"
    if value.endswith(".0") and value[:-2].isdigit():
        value = value[:-2]
    for location in nested(config, "sources", "weather", "locations", default=[]):
        if str(location.get("comtrade_code", "")).strip() == value:
            return location.get("node_id") or stable_id("country", location["name"])
    return f"country_{value.lower() or 'unknown'}"


def _country_id_from_name(name: Any, config: dict) -> str:
    value = str(name).strip()
    normalized = value.casefold()
    for location in nested(config, "sources", "weather", "locations", default=[]):
        if location.get("name", "").casefold() == normalized:
            return location.get("node_id") or stable_id("country", location["name"])
        aliases = [str(alias).casefold() for alias in location.get("aliases", [])]
        if normalized in aliases:
            return location.get("node_id") or stable_id("country", location["name"])
    return stable_id("country", value)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return default if not np.isfinite(result) else result
    except (TypeError, ValueError):
        return default


def _standard_trade(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["month", "source_country_id", "destination_country_id", "trade_value_usd", "flow_volume"])
    result = frame.copy()
    result["month"] = pd.to_datetime(result["month"].astype(str) + "-01", errors="coerce", utc=True).dt.strftime("%Y-%m")
    partner_codes = result["partner_code"] if "partner_code" in result else pd.Series("0", index=result.index)
    reporter_codes = result["reporter_code"] if "reporter_code" in result else pd.Series("unknown", index=result.index)
    result["source_country_id"] = partner_codes.map(lambda value: _country_id_from_code(value, config))
    result["destination_country_id"] = reporter_codes.map(
        lambda value: _country_id_from_code(value, config)
    )
    result["trade_value_usd"] = pd.to_numeric(result.get("trade_value_usd", 0), errors="coerce").fillna(0.0)
    result["flow_volume"] = pd.to_numeric(result.get("flow_volume", 0), errors="coerce").fillna(0.0)
    result = result[result["source_country_id"] != "country_0"]
    result["available_at"] = _timestamp_series(result, "available_at", result["month"] + "-28")
    result["reference_date"] = _timestamp_series(result, "month", result["month"] + "-28")
    return result


def _inbound_monthly(trade: pd.DataFrame) -> pd.DataFrame:
    if trade.empty:
        return pd.DataFrame(columns=["month", "country_id", "inbound_value", "inbound_volume"])
    return (
        trade.groupby(["month", "destination_country_id"], as_index=False)
        .agg(inbound_value=("trade_value_usd", "sum"), inbound_volume=("flow_volume", "sum"))
        .rename(columns={"destination_country_id": "country_id"})
    )


def _causal_news(
    events: pd.DataFrame, node_id: str, country_code: str, anchor: pd.Timestamp
) -> tuple[int, float, int, list[str]]:
    if events.empty:
        return 0, 0.0, 0, []
    frame = events.copy()
    reference = _timestamp_series(frame, "reference_date", anchor)
    available = _timestamp_series(frame, "available_at", anchor)
    eligible = frame[(reference <= anchor) & (available <= anchor)]
    if eligible.empty:
        return 0, 0.0, 0, []
    if "is_supply_chain_relevant" in eligible:
        eligible = eligible[
            pd.to_numeric(eligible["is_supply_chain_relevant"], errors="coerce")
            .fillna(0)
            .gt(0)
        ]
    if eligible.empty:
        return 0, 0.0, 0, []
    countries = (
        eligible.get("actor1_country", pd.Series("", index=eligible.index)).astype(str).str.upper().eq(country_code.upper())
        | eligible.get("actor2_country", pd.Series("", index=eligible.index)).astype(str).str.upper().eq(country_code.upper())
    )
    # If a caller uses a textual country code, retain matching events. Events
    # without actor countries are not silently broadcast to every node.
    eligible = eligible[countries]
    if eligible.empty:
        return 0, 0.0, 0, []
    recent_7 = eligible[reference >= anchor - pd.Timedelta(days=6)]
    recent_3 = eligible[reference >= anchor - pd.Timedelta(days=2)]
    tone = recent_3["avg_tone"] if "avg_tone" in recent_3 else pd.Series(0.0, index=recent_3.index)
    unrest = (
        recent_7["labor_unrest_hint"]
        if "labor_unrest_hint" in recent_7
        else recent_7.get("is_labor_unrest", pd.Series(0, index=recent_7.index))
    )
    negative = pd.to_numeric(tone, errors="coerce").fillna(0).lt(0)
    hint = pd.to_numeric(unrest, errors="coerce").fillna(0).gt(0)
    source_values = recent_7["source_url"] if "source_url" in recent_7 else pd.Series(dtype=str)
    sources = source_values.dropna().astype(str).head(10).tolist()
    return int(len(recent_7)), float(negative.mean()) if len(recent_3) else 0.0, int(hint.any()), sources


def _monthly_news(
    monthly: pd.DataFrame, country_code: str, month: str
) -> tuple[int, float, int, list[str]] | None:
    """Read the precomputed month-end row without changing its semantics."""
    if monthly.empty or "country" not in monthly or "month" not in monthly:
        return None
    rows = monthly[
        monthly["country"].astype(str).str.upper().eq(country_code.upper())
        & monthly["month"].astype(str).eq(month)
    ]
    if rows.empty:
        return None
    row = rows.iloc[-1]
    return (
        int(pd.to_numeric(row.get("news_vol_7d", 0), errors="coerce") or 0),
        float(pd.to_numeric(row.get("neg_tone_frac_3d", 0), errors="coerce") or 0),
        int(pd.to_numeric(row.get("strike_flag_7d", 0), errors="coerce") or 0),
        [],
    )


def _causal_weather(weather: pd.DataFrame, node_id: str, anchor: pd.Timestamp) -> int:
    if weather.empty:
        return 0
    frame = weather[weather["node_id"].astype(str) == str(node_id)].copy()
    if frame.empty:
        return 0
    observed = _timestamp_series(frame, "observed_date", anchor)
    available = _timestamp_series(frame, "available_at", anchor)
    eligible = frame[(observed <= anchor) & (available <= anchor) & (observed >= anchor - pd.Timedelta(days=6))]
    values = pd.to_numeric(eligible.get("temperature_c", 0), errors="coerce").dropna()
    if len(values) < 2:
        return 0
    std = float(values.std(ddof=0))
    return int(std > 0 and float((values - values.mean()).abs().max()) > 1.5 * std)


def _causal_global(gscpi: pd.DataFrame, anchor: pd.Timestamp) -> float:
    if gscpi.empty:
        return 0.0
    frame = gscpi.copy()
    months = pd.to_datetime(frame["month"].astype(str) + "-01", errors="coerce", utc=True)
    available = _timestamp_series(frame, "available_at", anchor)
    eligible = frame[(months <= anchor) & (available <= anchor)].copy()
    if eligible.empty:
        return 0.0
    return _as_float(eligible.sort_values("month").iloc[-1].get("global_risk", 0.0))


def _country_code_for_node(node_id: str, config: dict) -> str:
    for location in nested(config, "sources", "weather", "locations", default=[]):
        expected = location.get("node_id") or stable_id("country", location["name"])
        if expected == node_id:
            return str(location.get("iso3", location.get("comtrade_code", location["name"])))
    return node_id.rsplit("_", 1)[-1]


def _selected_country_ids(config: dict) -> set[str]:
    """Load the Comtrade-selected universe, keeping the focal country fixed."""
    if config.get("_synthetic"):
        return {
            location.get("node_id") or stable_id("country", location["name"])
            for location in nested(config, "sources", "weather", "locations", default=[])
        }
    base = Path(__file__).resolve().parent
    path = base / nested(config, "outputs", "country_universe", default="processed/country_universe.csv")
    if path.exists() or path.with_suffix(".csv").exists():
        universe = read_table(path)
        if not universe.empty and "comtrade_code" in universe:
            return {
                _country_id_from_code(code, config)
                for code in universe["comtrade_code"].dropna().astype(str)
            }
    return {
        location.get("node_id") or stable_id("country", location["name"])
        for location in nested(config, "sources", "weather", "locations", default=[])
    }


def _labels(
    nodes: pd.DataFrame, inbound: pd.DataFrame, config: dict
) -> pd.DataFrame:
    taus = [float(value) for value in nested(config, "analysis", "taus", default=[0.30, 0.35, 0.40])]
    horizon = int(nested(config, "analysis", "horizon_months", default=1))
    flow = inbound.set_index(["country_id", "month"])["inbound_value"].to_dict()
    result = nodes.copy()
    result["inbound_flow_usd"] = [
        _as_float(flow.get((host, month), 0.0))
        for host, month in zip(result["host_country_id"], result["month"])
    ]
    result = result.sort_values(["host_country_id", "month"]).reset_index(drop=True)
    grouped = result.groupby("host_country_id", sort=False)
    future = grouped["inbound_flow_usd"].shift(-horizon)
    # Eq. 7 baseline is V[T+h-12:T+h-1]. For h=1 this is the
    # twelve-month window ending at the current step T.
    history = grouped["inbound_flow_usd"].transform(
        lambda series: series.shift(-(horizon - 1)).rolling(12, min_periods=1).median()
    )
    contraction = (future - history) / history.replace(0, np.nan)
    for tau in taus:
        name = f"label_tau_{tau:.2f}"
        result[name] = np.where(
            future.notna() & history.notna() & (history > 0),
            (contraction < -tau).astype("float"),
            np.nan,
        )
    default_tau = float(nested(config, "analysis", "default_tau", default=0.35))
    result["label"] = result[f"label_tau_{default_tau:.2f}"]
    return result


def build_fused_tables(
    config: dict,
    trade: pd.DataFrame,
    cset_firm_country: pd.DataFrame | None = None,
    cset_firm_firm: pd.DataFrame | None = None,
    gdelt: pd.DataFrame | None = None,
    gdelt_monthly: pd.DataFrame | None = None,
    weather: pd.DataFrame | None = None,
    gscpi: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build country-only monthly tables from already-loaded source frames.

    CSET arguments remain optional for backwards-compatible imports but are
    intentionally ignored: this study does not infer firm relationships from
    country-level trade data.
    """
    gdelt = gdelt if gdelt is not None else pd.DataFrame()
    gdelt_monthly = gdelt_monthly if gdelt_monthly is not None else pd.DataFrame()
    weather = weather if weather is not None else pd.DataFrame()
    gscpi = gscpi if gscpi is not None else pd.DataFrame()
    trade = _standard_trade(trade, config)
    selected = _selected_country_ids(config)
    if selected:
        trade = trade[
            trade["source_country_id"].isin(selected)
            & trade["destination_country_id"].isin(selected)
        ]
    inbound = _inbound_monthly(trade)
    steps = [step.strftime("%Y-%m") for step in pd.date_range(
        pd.Timestamp(nested(config, "analysis", "start_date")),
        pd.Timestamp(nested(config, "analysis", "end_date")),
        freq="MS",
    )]

    country_ids = selected or (set(inbound["country_id"]) if not inbound.empty else set())

    records: list[dict[str, Any]] = []
    for month in steps:
        anchor = pd.Timestamp(month + "-01", tz="UTC") + pd.offsets.MonthEnd(0)
        for country_id in sorted(country_ids):
            code = _country_code_for_node(country_id, config)
            monthly_news = _monthly_news(gdelt_monthly, code, month)
            news_vol, neg_tone, strike, sources = (
                monthly_news
                if monthly_news is not None
                else _causal_news(gdelt, country_id, code, anchor)
            )
            weather_flag = _causal_weather(weather, country_id, anchor)
            inbound_value = _as_float(
                inbound.loc[
                    (inbound["country_id"] == country_id) & (inbound["month"] == month), "inbound_value"
                ].sum()
                if not inbound.empty
                else 0
            )
            prior = inbound[inbound["country_id"] == country_id].sort_values("month") if not inbound.empty else pd.DataFrame()
            prior_values = prior.loc[prior["month"] < month, "inbound_value"].tail(3).tolist() if not prior.empty else []
            baseline = float(np.median(prior_values)) if prior_values else inbound_value
            ratio = inbound_value / baseline if baseline > 0 else 1.0
            inventory = float(np.clip(30.0 * ratio, 5.0, 90.0))
            trade_delay = float(np.clip(30.0 * max(0.0, 1.0 - ratio), 0.0, 60.0))
            features = {
                "inventory_days_proxy": inventory,
                "trade_delay_proxy": trade_delay,
                "news_vol_7d": float(news_vol),
                "neg_tone_frac_3d": float(neg_tone),
                "strike_flag_7d": float(strike),
                "weather_anomaly_7d": float(weather_flag),
                "global_risk": _causal_global(gscpi, anchor),
            }
            records.append(
                {
                    "timestamp": anchor.isoformat(),
                    "month": month,
                    "node_id": country_id,
                    "node_type": "country",
                    "host_country_id": country_id,
                    **features,
                    "feature_provenance": json.dumps({name: "observed" for name in FEATURES}),
                    "feature_sources": json.dumps({"news": sources}),
                    "is_inherited": False,
                    "vintage_date": anchor.isoformat(),
                }
            )

    nodes = _labels(pd.DataFrame(records), inbound, config)

    edges: list[dict[str, Any]] = []
    for _, row in trade.iterrows():
        if row["source_country_id"] == row["destination_country_id"]:
            continue
        edges.append(
            {
                "timestamp": f"{row['month']}-01T00:00:00+00:00",
                "month": row["month"],
                "source": row["source_country_id"],
                "destination": row["destination_country_id"],
                "source_type": "country",
                "destination_type": "country",
                "trade_value_usd": _as_float(row["trade_value_usd"]),
                "flow_volume": _as_float(row["flow_volume"]),
                "edge_type": "trade",
                "provenance": "observed_comtrade",
                "is_observed": True,
                "is_inherited": False,
            }
        )
    return nodes, pd.DataFrame(edges)


def _empty_or_read(path: Path, columns: Iterable[str]) -> pd.DataFrame:
    if path.exists() or path.with_suffix(".csv").exists():
        return read_table(path)
    return pd.DataFrame(columns=list(columns))


def _synthetic_inputs(config: dict) -> tuple[pd.DataFrame, ...]:
    config["_synthetic"] = True
    months = [step.strftime("%Y-%m") for step in pd.date_range(
        pd.Timestamp(nested(config, "analysis", "start_date")),
        pd.Timestamp(nested(config, "analysis", "end_date")),
        freq="MS",
    )]
    countries = ["SYN_A", "SYN_B", "SYN_C"]
    trade_rows = []
    for index, month in enumerate(months):
        for source, destination in [("SYN_A", "SYN_B"), ("SYN_B", "SYN_C"), ("SYN_C", "SYN_A")]:
            value = 1_000_000.0 * (1.0 + 0.03 * index)
            # Place contractions across all partitions so the offline
            # GCN-vs-TGN smoke comparison has positive validation/test labels.
            if destination == "SYN_B" and index in {8, 20, 32, 40, 48, 52}:
                value *= 0.5
            trade_rows.append(
                {
                    "month": month,
                    "reporter_code": destination,
                    "partner_code": source,
                    "trade_value_usd": value,
                    "flow_volume": value / 1000,
                    "available_at": f"{month}-28T00:00:00+00:00",
                }
            )
    config.setdefault("sources", {}).setdefault("weather", {})["locations"] = [
        {"name": country, "node_id": f"country_{country.lower()}", "comtrade_code": country}
        for country in countries
    ]
    return (
        pd.DataFrame(trade_rows),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    )


def fuse(config: dict, synthetic: bool = False) -> tuple[Path, Path]:
    if synthetic:
        inputs = _synthetic_inputs(config)
        nodes, edges = build_fused_tables(
            config, inputs[0], gdelt=inputs[3], weather=inputs[4], gscpi=inputs[5]
        )
    else:
        base = Path(__file__).resolve().parent
        trade = _empty_or_read(
            base / nested(config, "outputs", "comtrade", default="processed/comtrade.csv"), []
        )
        gdelt = _empty_or_read(
            base / nested(config, "outputs", "gdelt", default="processed/gdelt_events.csv"), []
        )
        gdelt_monthly = _empty_or_read(
            base
            / nested(
                config,
                "outputs",
                "gdelt_monthly_features",
                default="processed/gdelt_monthly_features.csv",
            ),
            [],
        )
        weather = _empty_or_read(
            base / nested(config, "outputs", "weather", default="processed/weather_daily.csv"), []
        )
        gscpi = _empty_or_read(
            base / nested(config, "outputs", "gscpi", default="processed/gscpi_monthly.csv"), []
        )
        nodes, edges = build_fused_tables(
            config,
            trade,
            gdelt=gdelt,
            gdelt_monthly=gdelt_monthly,
            weather=weather,
            gscpi=gscpi,
        )
    base = Path(__file__).resolve().parent
    node_path = base / nested(config, "outputs", "nodes", default="processed/nodes_monthly.csv")
    edge_path = base / nested(config, "outputs", "edges", default="processed/edges_monthly.csv")
    return write_table(nodes, node_path), write_table(edges, edge_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_cli(parser)
    parser.add_argument("--synthetic", action="store_true", help="Generate a deterministic offline smoke dataset")
    args = parser.parse_args()
    outputs = fuse(load_config(args.config), synthetic=args.synthetic)
    print("\n".join(f"Wrote {output}" for output in outputs))


if __name__ == "__main__":
    main()

