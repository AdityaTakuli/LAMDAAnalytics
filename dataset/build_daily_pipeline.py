"""Build the independent 2024 daily temporal-graph diagnostic pipeline.

This module deliberately does not download data and does not train models.  It
reuses the already downloaded 2024 daily GDELT/weather tables and the
already-processed monthly Comtrade/GSCPI tables.  Monthly structural inputs are
carried within their calendar month and remain explicitly marked as monthly.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from common import load_config, nested, read_table, stable_id, write_json, write_table


BASE = Path(__file__).resolve().parent
ISO_BY_CODE = {
    "50": "BGD",
    "156": "CHN",
    "203": "CZE",
    "251": "FRA",
    "276": "DEU",
    "344": "HKG",
    "372": "IRL",
    "380": "ITA",
    "392": "JPN",
    "410": "KOR",
    "458": "MYS",
    "484": "MEX",
    "490": "TWN",
    "608": "PHL",
    "699": "IND",
    "702": "SGP",
    "704": "VNM",
    "764": "THA",
    "826": "GBR",
    "842": "USA",
}
NODE_BY_CODE = {
    "156": "country_cn",
    "251": "country_fr",
    "276": "country_de",
    "392": "country_jp",
    "410": "country_kr",
    "458": "country_my",
    "484": "country_mx",
    "490": "country_tw",
    "608": "country_ph",
    "699": "country_in",
    "702": "country_sg",
    "704": "country_vn",
    "764": "country_th",
    "826": "country_gb",
    "842": "country_us",
}
DAILY_SOURCE_FEATURES = [
    "news_vol_7d",
    "neg_tone_frac_3d",
    "strike_flag_7d",
]
MONTHLY_STRUCTURAL_FEATURES = [
    "inventory_days_proxy",
    "trade_delay_proxy",
    "inbound_flow_usd",
]


def _path(config: dict[str, Any], section: str, key: str, default: str) -> Path:
    value = nested(config, section, key, default=default)
    path = Path(value)
    return path if path.is_absolute() else BASE / path


def _read(config: dict[str, Any], key: str, default: str) -> pd.DataFrame:
    return read_table(_path(config, "daily_pipeline", key, default))


def _output(config: dict[str, Any], key: str, default: str) -> Path:
    return _path(config, "outputs", key, default)


def _node_id(code: Any) -> str:
    value = str(code).strip()
    return NODE_BY_CODE.get(value, f"country_{value.lower()}")


def _selected_universe(config: dict[str, Any]) -> pd.DataFrame:
    universe = _read(config, "selected_universe", "data/one_year_2024/processed/country_universe.csv")
    required = {"country_name", "comtrade_code"}
    missing = required - set(universe.columns)
    if missing:
        raise ValueError(f"Selected universe is missing columns: {sorted(missing)}")
    result = universe.copy()
    result["comtrade_code"] = result["comtrade_code"].astype(str).str.replace(r"\.0$", "", regex=True)
    result["iso3"] = result.get("iso3", pd.Series(index=result.index, dtype="string"))
    result["iso3"] = result["iso3"].fillna(
        result["comtrade_code"].map(ISO_BY_CODE)
    ).astype(str).str.upper()
    result["node_id"] = result["comtrade_code"].map(_node_id)
    result["node_id"] = result["node_id"].where(
        result["node_id"].notna(),
        result["country_name"].map(lambda value: stable_id("country", value)),
    )
    result = result.drop_duplicates("node_id").reset_index(drop=True)
    if len(result) != 20:
        raise ValueError(f"Expected the existing 20-country universe, found {len(result)}")
    return result[["country_name", "comtrade_code", "iso3", "node_id"]]


def _date_range(config: dict[str, Any]) -> pd.DatetimeIndex:
    start = pd.Timestamp(nested(config, "analysis", "start_date"), tz="UTC")
    end = pd.Timestamp(nested(config, "analysis", "end_date"), tz="UTC")
    return pd.date_range(start, end, freq="D", tz="UTC")


def _daily_gdelt(config: dict[str, Any]) -> pd.DataFrame:
    frame = _read(config, "gdelt_features", "data/one_year_2024/processed/gdelt_features.csv")
    if frame.empty:
        return pd.DataFrame(columns=["date", "iso3", *DAILY_SOURCE_FEATURES])
    result = frame.rename(columns={"country": "iso3"}).copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce", utc=True).dt.strftime("%Y-%m-%d")
    result["iso3"] = result["iso3"].astype(str).str.upper()
    result = result.dropna(subset=["date"])
    for column in DAILY_SOURCE_FEATURES:
        result[column] = pd.to_numeric(result.get(column), errors="coerce")
    # The source is expected to have one row per country-day. Aggregation is a
    # defensive guard for reruns or compatible source versions.
    return (
        result.groupby(["date", "iso3"], as_index=False)
        .agg(
            news_vol_7d=("news_vol_7d", "sum"),
            neg_tone_frac_3d=("neg_tone_frac_3d", "mean"),
            strike_flag_7d=("strike_flag_7d", "max"),
        )
    )


def _weather(config: dict[str, Any]) -> pd.DataFrame:
    frame = _read(config, "weather_daily", "data/one_year_2024/processed/weather_daily.csv")
    if frame.empty:
        return pd.DataFrame(columns=["date", "node_id", "temperature_c"])
    result = frame.copy()
    result["date"] = pd.to_datetime(
        result["observed_date"], errors="coerce", utc=True
    ).dt.strftime("%Y-%m-%d")
    result["node_id"] = result["node_id"].astype(str)
    result["temperature_c"] = pd.to_numeric(result["temperature_c"], errors="coerce")
    result = result.dropna(subset=["date", "node_id", "temperature_c"])
    return result.groupby(["date", "node_id"], as_index=False)["temperature_c"].mean()


def _weather_anomaly(frame: pd.DataFrame) -> pd.Series:
    """Apply the existing seven-day temperature anomaly rule causally."""
    result = pd.Series(0.0, index=frame.index)
    for node_id, group in frame.groupby("node_id", sort=False):
        ordered = group.sort_values("date")
        dates = pd.to_datetime(ordered["date"], utc=True)
        values = ordered["temperature_c"]
        for index, (row_index, current_date) in enumerate(zip(ordered.index, dates)):
            recent = values.loc[
                (dates <= current_date)
                & (dates >= current_date - pd.Timedelta(days=6))
            ].dropna()
            if len(recent) < 2:
                continue
            spread = float(recent.std(ddof=0))
            if spread > 0 and float((recent - recent.mean()).abs().max()) > 1.5 * spread:
                result.loc[row_index] = 1.0
    return result


def _monthly_nodes(config: dict[str, Any]) -> pd.DataFrame:
    frame = _read(config, "monthly_nodes", "data/one_year_2024/processed/nodes_monthly.csv")
    columns = ["month", "node_id", *MONTHLY_STRUCTURAL_FEATURES]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    result = frame.copy()
    result["month"] = result["month"].astype(str)
    result["node_id"] = result["node_id"].astype(str)
    for column in MONTHLY_STRUCTURAL_FEATURES:
        result[column] = pd.to_numeric(result.get(column), errors="coerce")
    return result.sort_values(["month", "node_id"]).drop_duplicates(
        ["month", "node_id"], keep="last"
    )[columns]


def _monthly_edges(config: dict[str, Any]) -> pd.DataFrame:
    frame = _read(config, "monthly_edges", "data/one_year_2024/processed/edges_monthly.csv")
    if frame.empty:
        return pd.DataFrame()
    result = frame.copy()
    result["month"] = result["month"].astype(str)
    result["source"] = result["source"].astype(str)
    result["destination"] = result["destination"].astype(str)
    return result


def _gscpi(config: dict[str, Any]) -> pd.DataFrame:
    frame = _read(config, "gscpi_monthly", "data/one_year_2024/processed/gscpi_monthly.csv")
    if frame.empty:
        return pd.DataFrame(columns=["month", "global_risk"])
    result = frame.copy()
    result["month"] = result["month"].astype(str)
    result["global_risk"] = pd.to_numeric(result["global_risk"], errors="coerce")
    return result.sort_values("month").drop_duplicates("month", keep="last")[
        ["month", "global_risk"]
    ]


def _build_daily_nodes(
    config: dict[str, Any],
    universe: pd.DataFrame,
    dates: pd.DatetimeIndex,
    monthly_nodes: pd.DataFrame,
    monthly_edges: pd.DataFrame,
) -> pd.DataFrame:
    calendar = pd.DataFrame({"date": dates.strftime("%Y-%m-%d")})
    calendar["key"] = 1
    countries = universe.copy()
    countries["key"] = 1
    result = calendar.merge(countries, on="key", how="outer").drop(columns="key")
    result["month"] = result["date"].str[:7]

    gdelt = _daily_gdelt(config)
    result = result.merge(gdelt, on=["date", "iso3"], how="left")
    result["gdelt_observed"] = result["news_vol_7d"].notna()

    weather = _weather(config)
    result = result.merge(weather, on=["date", "node_id"], how="left")
    result["weather_observed"] = result["temperature_c"].notna()
    result["weather_anomaly_7d"] = _weather_anomaly(
        result[["date", "node_id", "temperature_c"]].dropna(subset=["temperature_c"])
    ).reindex(result.index, fill_value=0.0)

    structural = monthly_nodes.copy()
    if not monthly_edges.empty:
        observed = (
            monthly_edges.groupby(["month", "destination"], as_index=False)
            .size()
            .rename(columns={"destination": "node_id", "size": "trade_edge_count"})
        )
        structural = structural.merge(observed, on=["month", "node_id"], how="left")
    else:
        structural["trade_edge_count"] = 0
    structural["trade_edge_count"] = structural["trade_edge_count"].fillna(0).astype(int)
    structural["comtrade_observed"] = structural["trade_edge_count"].gt(0)
    # The monthly fusion has safe numeric defaults for missing inbound flow.
    # Daily construction must not reinterpret those defaults as observations.
    for column in MONTHLY_STRUCTURAL_FEATURES:
        structural.loc[~structural["comtrade_observed"], column] = np.nan
    result = result.merge(
        structural[
            ["month", "node_id", *MONTHLY_STRUCTURAL_FEATURES, "trade_edge_count", "comtrade_observed"]
        ],
        on=["month", "node_id"],
        how="left",
    )

    gscpi = _gscpi(config)
    result = result.merge(gscpi, on="month", how="left")
    result["gscpi_observed"] = result["global_risk"].notna()

    assumption = nested(
        config,
        "analysis",
        "monthly_availability_assumption",
        default="month_start",
    )
    result["trade_feature_source_period"] = result["month"]
    result["trade_feature_frequency"] = "monthly"
    result["trade_feature_availability_assumption"] = assumption
    result["gscpi_source_period"] = result["month"]
    result["gscpi_frequency"] = "monthly"
    result["gscpi_availability_assumption"] = assumption
    result["gdelt_frequency"] = "daily"
    result["weather_frequency"] = "daily"
    result["static_topology_frequency"] = "static"
    result["feature_provenance"] = result.apply(
        lambda row: json.dumps(
            {
                "gdelt": "observed_daily" if row["gdelt_observed"] else "missing",
                "weather": "observed_daily" if row["weather_observed"] else "missing",
                "comtrade": "observed_monthly_carried_within_month"
                if row["comtrade_observed"]
                else "missing",
                "gscpi": "observed_monthly_carried_within_month"
                if row["gscpi_observed"]
                else "missing",
                "cset": "not_used_by_existing_country_graph",
            },
            sort_keys=True,
        ),
        axis=1,
    )
    return result.sort_values(["date", "node_id"]).reset_index(drop=True)


def _edge_templates(
    monthly_edges: pd.DataFrame, universe: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, list[dict[str, Any]]]]:
    if monthly_edges.empty:
        empty = pd.DataFrame(
            columns=[
                "month",
                "source",
                "destination",
                "source_type",
                "destination_type",
                "trade_value_usd",
                "flow_volume",
                "edge_type",
                "provenance",
                "is_observed",
                "is_inherited",
                "source_period",
                "source_frequency",
            ]
        )
        return empty, {}
    selected = set(universe["node_id"].astype(str))
    result = monthly_edges[
        monthly_edges["source"].isin(selected)
        & monthly_edges["destination"].isin(selected)
    ].copy()
    result["source_period"] = result["month"]
    result["source_frequency"] = "monthly"
    result["is_inherited"] = True
    templates = {
        str(month): [
            {key: (None if pd.isna(value) else value) for key, value in row.items()}
            for row in group.to_dict(orient="records")
        ]
        for month, group in result.groupby("month", sort=True)
    }
    return result, templates


def _snapshot_manifest(
    dates: pd.DatetimeIndex, nodes: pd.DataFrame, templates: dict[str, list[dict[str, Any]]]
) -> pd.DataFrame:
    rows = []
    for timestamp in dates:
        date = timestamp.strftime("%Y-%m-%d")
        month = timestamp.strftime("%Y-%m")
        rows.append(
            {
                "date": date,
                "month": month,
                "node_count": int(nodes[nodes["date"].eq(date)]["node_id"].nunique()),
                "edge_template_month": month,
                "edge_count": len(templates.get(month, [])),
                "topology_source_frequency": "monthly",
                "topology_carried_forward": True,
            }
        )
    return pd.DataFrame(rows)


def _write_target_diagnostics(
    config: dict[str, Any],
    universe: pd.DataFrame,
    dates: pd.DatetimeIndex,
    daily_nodes: pd.DataFrame,
) -> dict[str, Any]:
    directory = _output(
        config,
        "target_diagnostics",
        "data/one_year_2024_daily/results/target_diagnostics",
    )
    directory.mkdir(parents=True, exist_ok=True)
    total = int(len(daily_nodes))
    summary = {
        "status": "not_constructed",
        "total_daily_observations": total,
        "valid_target_observations": 0,
        "positive_observations": 0,
        "negative_observations": 0,
        "prevalence": None,
        "horizon_days": int(
            nested(config, "analysis", "prediction_horizon_days", default=1)
        ),
        "reason": (
            "No independent daily ground-truth disruption variable is present. "
            "Monthly Comtrade cannot be converted into a daily label; future "
            "GDELT/weather values would be predictors or tautological proxies, "
            "not an independent target."
        ),
    }
    write_json(summary, directory / "target_distribution.json")
    pd.DataFrame(
        [
            {
                "scope": "overall",
                **summary,
            }
        ]
    ).to_csv(directory / "target_distribution.csv", index=False)
    by_country = pd.DataFrame(
        {
            "country": universe["country_name"],
            "node_id": universe["node_id"],
            "total_daily_observations": len(dates),
            "valid_target_observations": 0,
            "positive": 0,
            "negative": 0,
            "prevalence": np.nan,
            "status": "not_constructed",
        }
    )
    by_country.to_csv(directory / "target_by_country.csv", index=False)
    by_month = pd.DataFrame(
        {
            "month": sorted(daily_nodes["month"].unique()),
            "total_daily_observations": len(universe) * daily_nodes.groupby("month").date.nunique().to_numpy(),
            "valid_target_observations": 0,
            "positive": 0,
            "negative": 0,
            "prevalence": np.nan,
            "status": "not_constructed",
        }
    )
    by_month.to_csv(directory / "target_by_month.csv", index=False)
    (directory / "target_definition.md").write_text(
        f"""# Daily target definition

Status: **NOT CONSTRUCTED — daily target is not scientifically validated.**

The available independent observations are daily GDELT event-derived signals
and daily NASA POWER temperature observations. They are candidate predictors,
not an independent disruption ground truth. The available Comtrade and NY Fed
GSCPI observations are monthly. Expanding a monthly trade value over days,
dividing it by days, or using future GDELT mentions as the label would create
fabricated or tautological targets.

Accordingly, this diagnostic pipeline reports `{total}` country-day feature
rows but creates zero valid/positive/negative labels. The one-day horizon in
configuration is retained as a future design parameter only; it was not used
to manufacture labels. No daily model should be trained until an independent
daily target source or a pre-specified event-label protocol is supplied and
validated.
""",
        encoding="utf-8",
    )
    return summary


def _write_leakage_audit(
    config: dict[str, Any],
    daily_nodes: pd.DataFrame,
    summary: dict[str, Any],
    dates: pd.DatetimeIndex,
) -> None:
    path = _output(
        config,
        "diagnostics",
        "data/one_year_2024_daily/results/diagnostics",
    ) / "daily_leakage_audit.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    missing_gdelt = int((~daily_nodes["gdelt_observed"]).sum())
    missing_weather = int((~daily_nodes["weather_observed"]).sum())
    missing_trade = int((~daily_nodes["comtrade_observed"]).sum())
    missing_gscpi = int((~daily_nodes["gscpi_observed"]).sum())
    path.write_text(
        f"""# Daily leakage audit

Status: **PASS for temporal construction; target gate STOPPED training.**

## Clock and split

* Date range: `{dates[0].date()}` through `{dates[-1].date()}` inclusive.
* Daily rows are keyed by `(country, date)` and sorted chronologically.
* No random shuffle, oversampling, or future target construction is used.
* There is no model split or scaler fit because the target is not constructed.

## Source-by-source audit

* **GDELT:** original frequency is daily. Existing `gdelt_features.csv` is
  reused; its documented clock is DATEADDED/UTC day with trailing windows
  ending on that day. Missing daily country rows remain missing
  (`{missing_gdelt}` rows); they are not converted to zero.
* **Weather:** original frequency is daily. Existing NASA POWER observations
  are joined by observed UTC date. The existing seven-day anomaly rule is
  evaluated using observations at or before the current day. Missing rows
  remain missing (`{missing_weather}` rows).
* **Comtrade:** original frequency is monthly. Monthly structural features are
  carried only within the matching calendar month and marked
  `trade_feature_frequency=monthly`. Missing monthly inbound observations
  remain missing (`{missing_trade}` daily rows). No daily trade values are
  fabricated.
* **GSCPI:** original frequency is monthly. The matching monthly value is
  carried within the month and marked `gscpi_frequency=monthly`. The configured
  `month_start` availability assumption is explicit because the stored
  retrieval vintage is not a historical publication timestamp. Missing values
  remain missing (`{missing_gscpi}` rows).
* **CSET:** the existing graph builder does not use CSET relationships and no
  firm-level country mapping is inferred. Daily graph topology therefore uses
  only observed monthly country-level Comtrade edges, carried within month.

## Graph and scaling

* Every daily snapshot references the actual monthly edge template for that
  calendar month; no edge is created merely because a country exists.
* Daily node features are stored separately from edge templates.
* No standardization is performed in this diagnostic stage. Any future model
  must fit scaling statistics on training dates only.
* TGN memory is not initialized or replayed because training is prohibited
  until a valid daily target exists.

## Target gate

The target status is `{summary["status"]}`. There is no independent daily
ground truth in the reused 2024 sources, so training must stop.
""",
        encoding="utf-8",
    )


def run(config: dict[str, Any], force: bool = False) -> Path:
    root = _output(config, "root", "data/one_year_2024_daily")
    results = _output(config, "results", "data/one_year_2024_daily/results")
    if root.exists():
        if not force:
            raise FileExistsError(
                f"Daily output already exists; use --force to rebuild only {root}"
            )
        shutil.rmtree(root)
    for directory_key, default in (
        ("raw_directory", "data/one_year_2024_daily/raw"),
        ("cache_directory", "data/one_year_2024_daily/cache"),
        ("processed_directory", "data/one_year_2024_daily/processed"),
        ("results", "data/one_year_2024_daily/results"),
        ("diagnostics", "data/one_year_2024_daily/results/diagnostics"),
        ("target_diagnostics", "data/one_year_2024_daily/results/target_diagnostics"),
    ):
        _output(config, directory_key, default).mkdir(parents=True, exist_ok=True)

    for directory_key, message in (
        ("raw_directory", "No raw downloads are performed; source raw data remains in one_year_2024."),
        ("cache_directory", "No new source downloads are performed; existing 2024 caches are reused."),
    ):
        directory = _output(config, directory_key, f"data/one_year_2024_daily/{directory_key}")
        (directory / "README.md").write_text(message + "\n", encoding="utf-8")

    universe = _selected_universe(config)
    dates = _date_range(config)
    monthly_nodes = _monthly_nodes(config)
    monthly_edges = _monthly_edges(config)
    daily_nodes = _build_daily_nodes(
        config, universe, dates, monthly_nodes, monthly_edges
    )
    edge_frame, templates = _edge_templates(monthly_edges, universe)
    manifest = _snapshot_manifest(dates, daily_nodes, templates)

    processed = _output(
        config, "processed_directory", "data/one_year_2024_daily/processed"
    )
    write_table(daily_nodes, processed / "nodes_daily.csv")
    write_table(edge_frame, _output(
        config,
        "monthly_edge_templates",
        "data/one_year_2024_daily/processed/edge_templates_monthly.csv",
    ))
    write_table(
        manifest,
        _output(
            config,
            "snapshot_manifest",
            "data/one_year_2024_daily/processed/snapshot_manifest.csv",
        ),
    )
    graph = {
        "format": "lamda-country-daily-temporal-graph-v1",
        "date_range": {
            "start": dates[0].strftime("%Y-%m-%d"),
            "end": dates[-1].strftime("%Y-%m-%d"),
        },
        "feature_table": "processed/nodes_daily.csv",
        "edge_template_table": "processed/edge_templates_monthly.csv",
        "snapshot_manifest": "processed/snapshot_manifest.csv",
        "feature_names": [
            "news_vol_7d",
            "neg_tone_frac_3d",
            "strike_flag_7d",
            "temperature_c",
            "weather_anomaly_7d",
            "inventory_days_proxy",
            "trade_delay_proxy",
            "inbound_flow_usd",
            "global_risk",
        ],
        "snapshots": manifest.to_dict(orient="records"),
        "edge_templates": templates,
        "metadata": {
            "countries": int(universe["node_id"].nunique()),
            "daily_snapshots": int(len(dates)),
            "country_day_rows": int(len(daily_nodes)),
            "topology_source": "observed monthly country-level Comtrade edges",
            "topology_source_frequency": "monthly",
            "topology_carried_forward_within_month": True,
            "cset_relationships_used": False,
            "models_trained": False,
        },
    }
    write_json(
        graph,
        _output(config, "graph", "data/one_year_2024_daily/processed/graph_daily.json"),
    )

    target_summary = _write_target_diagnostics(config, universe, dates, daily_nodes)
    diagnostics_dir = _output(
        config, "diagnostics", "data/one_year_2024_daily/results/diagnostics"
    )
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    source_summary = {
        "date_range": {"start": dates[0].strftime("%Y-%m-%d"), "end": dates[-1].strftime("%Y-%m-%d")},
        "country_count": int(universe["node_id"].nunique()),
        "daily_snapshot_count": int(len(dates)),
        "country_day_rows": int(len(daily_nodes)),
        "gdelt_rows": int(daily_nodes["gdelt_observed"].sum()),
        "gdelt_countries": int(daily_nodes.loc[daily_nodes["gdelt_observed"], "node_id"].nunique()),
        "weather_rows": int(daily_nodes["weather_observed"].sum()),
        "weather_countries": int(daily_nodes.loc[daily_nodes["weather_observed"], "node_id"].nunique()),
        "comtrade_monthly_feature_rows": int(daily_nodes["comtrade_observed"].sum()),
        "gscpi_monthly_feature_rows": int(daily_nodes["gscpi_observed"].sum()),
        "monthly_edge_template_rows": int(len(edge_frame)),
        "monthly_edge_template_months": int(edge_frame["month"].nunique()) if not edge_frame.empty else 0,
        "snapshot_manifest_rows": int(len(manifest)),
        "target": target_summary,
        "models_trained": False,
    }
    write_json(source_summary, diagnostics_dir / "dataset_summary.json")
    write_json(
        {
            "source_frequency": {
                "GDELT": "daily",
                "Weather": "daily",
                "Comtrade": "monthly",
                "GSCPI": "monthly",
                "CSET": "static; not used by existing country graph",
            },
            "source_paths": {
                key: str(_path(config, "daily_pipeline", key, default))
                for key, default in {
                    "gdelt_features": "data/one_year_2024/processed/gdelt_features.csv",
                    "weather_daily": "data/one_year_2024/processed/weather_daily.csv",
                    "comtrade_monthly": "data/one_year_2024/processed/comtrade.csv",
                    "gscpi_monthly": "data/one_year_2024/processed/gscpi_monthly.csv",
                    "monthly_edges": "data/one_year_2024/processed/edges_monthly.csv",
                }.items()
            },
        },
        diagnostics_dir / "provenance.json",
    )
    _write_leakage_audit(config, daily_nodes, target_summary, dates)

    config_copy = dict(config)
    config_copy.pop("_config_path", None)
    (results / "config_used.yaml").write_text(
        yaml.safe_dump(config_copy, sort_keys=False), encoding="utf-8"
    )
    (results / "README.md").write_text(
        """# 2024 daily temporal graph diagnostic

This directory contains a diagnostic-only daily pipeline. It reuses existing
2024 GDELT/weather data and existing monthly Comtrade/GSCPI data without
downloading or modifying the monthly profile. Monthly sources are carried
within their calendar month and remain explicitly marked as monthly.

The target gate stopped before training because no independent daily
disruption ground truth is available. No GCN, TGN, TGN-no-memory, or other
model checkpoints, predictions, or metrics were created.
""",
        encoding="utf-8",
    )
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_daily.yaml")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild only the daily output directory; never modifies the monthly profile",
    )
    args = parser.parse_args()
    output = run(load_config(args.config), force=args.force)
    print(f"Wrote daily diagnostic pipeline to {output}")
    print("DAILY TARGET NOT SCIENTIFICALLY VALID — training stopped.")


if __name__ == "__main__":
    main()
