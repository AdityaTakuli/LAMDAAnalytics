"""Download and audit the selected 2024 bilateral Comtrade matrix.

The legacy Comtrade pull is preserved in
``processed/comtrade_coverage_fix_backup/``.  This command uses the existing
country universe, downloads one authenticated/public API response per
reporter-month, and writes only 2024 bilateral imports for HS 8541 and 8542.
Missing API rows remain missing; they are never materialized as zero trade.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import yaml

from build_graph import build as build_graph
from common import (
    add_common_cli,
    canonical_column,
    env_value,
    load_config,
    monthly_steps,
    nested,
    numeric,
    read_table,
    stable_id,
    vintage_now,
    write_json,
    write_table,
)
from fuse_dataset import fuse
from ingest_comtrade import standardize


COUNTRY_NAMES = {
    "36": "Australia",
    "50": "Bangladesh",
    "76": "Brazil",
    "156": "China",
    "203": "Czechia",
    "251": "France",
    "276": "Germany",
    "344": "Hong Kong",
    "360": "Indonesia",
    "372": "Ireland",
    "380": "Italy",
    "392": "Japan",
    "410": "South Korea",
    "458": "Malaysia",
    "484": "Mexico",
    "490": "Taiwan",
    "608": "Philippines",
    "699": "India",
    "702": "Singapore",
    "704": "Vietnam",
    "764": "Thailand",
    "784": "United Arab Emirates",
    "826": "United Kingdom",
    "842": "United States",
}


def _base() -> Path:
    return Path(__file__).resolve().parent


def _output_path(config: dict[str, Any], key: str, default: str) -> Path:
    value = nested(config, "outputs", key, default=default)
    path = Path(value)
    return path if path.is_absolute() else _base() / path


def _atomic_write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, default=str)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _selected_universe(config: dict[str, Any]) -> pd.DataFrame:
    universe_path = _output_path(
        config, "country_universe", "data/four_year_2021_2024/processed/country_universe.csv"
    )
    universe = read_table(universe_path)
    if universe.empty or "comtrade_code" not in universe:
        raise ValueError(f"Existing selected country universe is empty: {universe_path}")
    universe = universe.copy()
    universe["comtrade_code"] = (
        universe["comtrade_code"].astype(str).str.replace(r"\.0$", "", regex=True)
    )
    universe = universe.drop_duplicates("comtrade_code").reset_index(drop=True)
    if "699" not in set(universe["comtrade_code"]):
        raise ValueError("Existing country universe does not contain focal reporter 699")
    return universe


def _preserve_existing(config: dict[str, Any]) -> Path:
    processed = _output_path(config, "comtrade", "processed/comtrade.csv").parent
    backup = processed / "comtrade_coverage_fix_backup"
    backup.mkdir(parents=True, exist_ok=True)
    for filename in ("comtrade.csv", "country_universe.csv", "nodes_monthly.csv", "edges_monthly.csv", "graph.json"):
        source = processed / filename
        destination = backup / filename
        if source.exists() and not destination.exists():
            shutil.copy2(source, destination)
    return backup


def _api_request(
    config: dict[str, Any],
    reporter: str,
    partner_codes: list[str],
    period: str,
) -> dict[str, Any]:
    source = nested(config, "sources", "comtrade", default={})
    key = env_value(config, source.get("subscription_key_env"), "")
    secondary_key = env_value(config, source.get("secondary_key_env"), "")
    flow_code = str(source.get("flow_code", "M"))
    hs_codes = [str(code) for code in source.get("hs_codes", [8541, 8542])]
    if flow_code != "M":
        raise ValueError(f"Bilateral coverage fix requires imports (flow_code=M), got {flow_code}")
    if hs_codes != ["8541", "8542"]:
        raise ValueError(f"Bilateral coverage fix requires HS 8541 and 8542, got {hs_codes}")
    params = {
        "reportercode": reporter,
        "flowCode": flow_code,
        "period": period,
        "cmdCode": ",".join(hs_codes),
        "partnerCode": ",".join(partner_codes),
        "partner2Code": None,
        "maxRecords": int(source.get("max_records", 250000 if key else 500)),
        "format": "JSON",
        "breakdownMode": "classic",
        "includeDesc": "true",
    }
    retries = int(source.get("max_retries", 4))
    timeout = float(source.get("timeout_seconds", 120))
    backoff = float(source.get("backoff_seconds", 2.0))
    last_error: Exception | None = None
    credentials = [value for value in (key, secondary_key) if value]
    for credential in credentials or [None]:
        endpoint = source.get(
            "api_url" if credential else "public_api_url",
            "https://comtradeapi.un.org/data/v1/get/C/M/HS"
            if credential
            else "https://comtradeapi.un.org/public/v1/preview/C/M/HS",
        )
        request_params = dict(params)
        if credential:
            request_params["subscription-key"] = credential
        for attempt in range(retries + 1):
            try:
                response = requests.get(endpoint, params=request_params, timeout=timeout)
                if response.status_code == 200:
                    payload = response.json()
                    if not isinstance(payload, dict) or "data" not in payload:
                        raise RuntimeError("Comtrade response did not contain a data field")
                    return payload
                message = f"HTTP {response.status_code}"
                last_error = RuntimeError(message)
                if response.status_code in {401, 403}:
                    break
                if response.status_code == 404:
                    raise RuntimeError(message)
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                last_error = exc
            if attempt < retries:
                time.sleep(backoff * (2**attempt))
    raise RuntimeError(f"Comtrade request failed after {retries + 1} attempts: {last_error}")


def _cache_path(cache_dir: Path, reporter: str, period: str) -> Path:
    return cache_dir / f"reporter_{int(reporter):03d}_{period}.json"


def _read_cached(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("response"), dict):
        raise ValueError(f"Invalid bilateral cache response: {path}")
    return payload


def _raw_frame(payload: dict[str, Any]) -> pd.DataFrame:
    data = payload.get("response", {}).get("data", [])
    return pd.DataFrame(data if isinstance(data, list) else [])


def _standardize_bilateral(
    raw: pd.DataFrame, period: str, vintage: str
) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(
            columns=[
                "month",
                "period",
                "reporter",
                "partner",
                "commodity",
                "flow",
                "trade_value",
                "quantity",
                "weight",
                "reporter_code",
                "partner_code",
                "commodity_code",
                "flow_code",
                "trade_value_usd",
                "flow_volume",
                "published_at",
                "available_at",
                "source",
                "vintage_date",
            ]
        )
    result = standardize(raw, period, vintage)
    reporter = canonical_column(raw, ["reporterCode", "reporterCodeISO", "reporter"])
    partner = canonical_column(raw, ["partnerCode", "partnerCodeISO", "partner"])
    commodity = canonical_column(raw, ["cmdCode", "cmdCodeISO", "commodityCode", "commodity"])
    flow = canonical_column(raw, ["flowCode", "flow"])
    trade_value = canonical_column(
        raw, ["primaryValue", "primaryValueUSD", "tradeValue", "tradeValueUSD", "cifvalue", "fobvalue"]
    )
    quantity = canonical_column(raw, ["qty", "quantity"])
    net_weight = canonical_column(raw, ["netWgt", "netWeight"])
    gross_weight = canonical_column(raw, ["grossWgt", "grossWeight"])
    result["period"] = period
    result["reporter"] = raw[reporter].astype(str) if reporter else result["reporter_code"]
    result["partner"] = raw[partner].astype(str) if partner else result["partner_code"]
    result["commodity"] = raw[commodity].astype(str) if commodity else result["commodity_code"]
    result["flow"] = raw[flow].astype(str) if flow else result["flow_code"]
    result["trade_value"] = numeric(raw, trade_value)
    result["quantity"] = numeric(raw, quantity)
    result["weight"] = numeric(raw, net_weight)
    result["net_weight"] = numeric(raw, net_weight)
    result["gross_weight"] = numeric(raw, gross_weight)
    result["source"] = "un_comtrade_bilateral"
    return result


def _coverage_grid(
    frame: pd.DataFrame,
    reporters: list[str],
    partners: list[str],
    months: list[str],
) -> pd.DataFrame:
    frame = frame.copy()
    for column in ("reporter_code", "partner_code", "month"):
        if column in frame:
            frame[column] = frame[column].astype(str)
    reporters = [str(value) for value in reporters]
    partners = [str(value) for value in partners]
    months = [str(value) for value in months]
    if frame.empty:
        observed = pd.DataFrame(
            columns=["reporter_code", "partner_code", "month", "observed_records"]
        )
    else:
        observed = (
            frame.groupby(["reporter_code", "partner_code", "month"], as_index=False)
            .size()
            .rename(columns={"size": "observed_records"})
        )
    grid = pd.MultiIndex.from_product(
        [reporters, partners, months],
        names=["reporter_code", "partner_code", "month"],
    ).to_frame(index=False)
    result = grid.merge(
        observed,
        on=["reporter_code", "partner_code", "month"],
        how="left",
    )
    result["observed_records"] = result["observed_records"].fillna(0).astype(int)
    result["observed"] = result["observed_records"].gt(0)
    return result


def _country_label(code: str) -> str:
    return COUNTRY_NAMES.get(str(code), f"Comtrade code {code}")


def _node_label(node_id: str, config: dict[str, Any]) -> str:
    for location in nested(config, "sources", "weather", "locations", default=[]):
        expected = location.get("node_id") or stable_id("country", location["name"])
        if str(expected) == str(node_id):
            return str(location["name"])
    return _country_label(str(node_id).rsplit("_", 1)[-1])


def _target_statistics(
    nodes: pd.DataFrame, config: dict[str, Any], destination: Path
) -> dict[str, Any]:
    frame = nodes.copy()
    frame["month"] = frame["month"].astype(str)
    frame["inbound_flow_usd"] = pd.to_numeric(
        frame["inbound_flow_usd"], errors="coerce"
    )
    frame = frame.sort_values(["host_country_id", "month"]).reset_index(drop=True)
    grouped = frame.groupby("host_country_id", sort=False)
    horizon = int(nested(config, "analysis", "horizon_months", default=1))
    future = grouped["inbound_flow_usd"].shift(-horizon)
    history = grouped["inbound_flow_usd"].transform(
        lambda values: values.shift(-(horizon - 1)).rolling(12, min_periods=1).median()
    )
    contraction = (future - history) / history.replace(0, np.nan)
    valid = future.notna() & history.notna() & history.gt(0)
    tau = float(nested(config, "analysis", "default_tau", default=0.35))
    label = contraction.lt(-tau)
    frame["future_inbound_flow_usd"] = future
    frame["historical_median_inbound_usd"] = history
    frame["contraction"] = contraction
    frame["target_valid"] = valid
    frame["label"] = np.where(valid, label.astype(int), np.nan)

    target_rows = frame[
        ["host_country_id", "month", "target_valid", "contraction", "label"]
    ].copy()
    target_rows = target_rows.rename(columns={"host_country_id": "country_id"})
    target_rows["country"] = target_rows["country_id"].map(
        lambda value: _node_label(str(value), config)
    )
    target_rows.to_csv(destination / "target_coverage.csv", index=False)

    valid_frame = frame[valid].copy()
    by_country_observed = (
        valid_frame.groupby("host_country_id", as_index=False)
        .agg(
            valid_targets=("target_valid", "size"),
            positive_targets=("label", "sum"),
            mean_contraction=("contraction", "mean"),
        )
        .rename(columns={"host_country_id": "country_id"})
    )
    by_country = pd.DataFrame(
        {"country_id": sorted(frame["host_country_id"].astype(str).unique())}
    ).merge(by_country_observed, on="country_id", how="left")
    by_country["valid_targets"] = by_country["valid_targets"].fillna(0).astype(int)
    by_country["positive_targets"] = by_country["positive_targets"].fillna(0).astype(int)
    by_country["negative_targets"] = (
        by_country["valid_targets"] - by_country["positive_targets"]
    )
    by_country["prevalence"] = (
        by_country["positive_targets"]
        / by_country["valid_targets"].replace(0, np.nan)
    )
    by_country["country"] = by_country["country_id"].map(
        lambda value: _node_label(str(value), config)
    )
    by_country = by_country[
        [
            "country",
            "country_id",
            "valid_targets",
            "positive_targets",
            "negative_targets",
            "prevalence",
            "mean_contraction",
        ]
    ].sort_values("country")
    by_country.to_csv(destination / "coverage_by_country.csv", index=False)

    valid_contractions = valid_frame["contraction"].astype(float)
    quantiles = valid_contractions.quantile(
        [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    ).to_dict()

    def stats(values: pd.Series) -> dict[str, Any]:
        values = values.dropna().astype(float)
        q = values.quantile([0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).to_dict()
        return {
            "count": int(len(values)),
            "min": float(values.min()) if len(values) else None,
            "max": float(values.max()) if len(values) else None,
            "mean": float(values.mean()) if len(values) else None,
            "median": float(values.median()) if len(values) else None,
            "std": float(values.std(ddof=0)) if len(values) else None,
            **{f"p{int(key * 100):02d}": float(value) for key, value in q.items()},
        }

    by_country_stats = {
        str(country): stats(group["contraction"])
        for country, group in valid_frame.groupby("host_country_id")
    }
    by_month_stats = {
        str(month): stats(group["contraction"])
        for month, group in valid_frame.groupby("month")
    }
    tau_counts = {}
    for configured_tau in nested(config, "analysis", "taus", default=[0.30, 0.35, 0.40]):
        configured_tau = float(configured_tau)
        positives = int(valid_frame["contraction"].lt(-configured_tau).sum())
        tau_counts[f"{configured_tau:.2f}"] = {
            "positive": positives,
            "negative": int(len(valid_frame) - positives),
            "prevalence": float(positives / len(valid_frame))
            if len(valid_frame)
            else None,
        }
    target_summary = {
        "total_country_month_rows": int(len(frame)),
        "valid_target_rows": int(valid.sum()),
        "missing_target_rows": int((~valid).sum()),
        "positive": int(label[valid].sum()),
        "negative": int(valid.sum() - label[valid].sum()),
        "prevalence": float(label[valid].mean()) if valid.any() else None,
        "horizon_months": horizon,
        "tau": tau,
        "counts_by_tau": tau_counts,
        "continuous_target_overall": stats(valid_frame["contraction"]),
        "continuous_target_by_country": by_country_stats,
        "continuous_target_by_month": by_month_stats,
    }
    write_json(target_summary, destination / "target_coverage.json")
    write_json(
        {
            "overall": stats(valid_frame["contraction"]),
            "by_country": by_country_stats,
            "by_month": by_month_stats,
        },
        destination / "target_distribution.json",
    )
    distribution_rows = [{"scope": "overall", "key": "all", **stats(valid_frame["contraction"])}]
    distribution_rows.extend(
        {"scope": "country", "key": key, **value}
        for key, value in by_country_stats.items()
    )
    distribution_rows.extend(
        {"scope": "month", "key": key, **value} for key, value in by_month_stats.items()
    )
    pd.DataFrame(distribution_rows).to_csv(
        destination / "target_distribution.csv", index=False
    )
    return target_summary


def _write_readme(
    destination: Path,
    reporters: list[str],
    partners: list[str],
    months: list[str],
    coverage: dict[str, Any],
    failures: list[dict[str, Any]],
    backup: Path,
) -> None:
    (destination / "README.md").write_text(
        f"""# 2024 Comtrade coverage fix

This artifact replaces the India-only 2024 Comtrade pull with requested
bilateral import responses for the existing selected country universe.

* Reporters requested: {len(reporters)}
* Partners requested: {len(partners)}
* HS codes: 8541, 8542
* Flow: M (imports)
* Months: {months[0]} through {months[-1]}
* Cache: `data/four_year_2021_2024/cache/comtrade/bilateral/`
* Previous Comtrade/graph artifacts: `{backup}`
* API failures: {len(failures)}
* Missing API cells are preserved as missing, not converted to zero.

The processed table and downstream graph were regenerated only after the
bilateral requests completed. No model training is performed by this command.
December has no one-month-ahead target because January 2025 is outside the
2024-only corpus.
""",
        encoding="utf-8",
    )


def run(config: dict[str, Any], force: bool = False) -> Path:
    destination = _output_path(
        config,
        "comtrade_coverage_fix",
        "data/four_year_2021_2024/results/comtrade_coverage_fix",
    )
    summary_marker = destination / "coverage_summary.json"
    if destination.exists() and not force and summary_marker.exists():
        try:
            previous_summary = json.loads(summary_marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous_summary = {}
        if not previous_summary.get("api_failures"):
            raise FileExistsError(
                f"Coverage-fix directory is complete; use --force to rebuild: {destination}"
            )
    destination.mkdir(parents=True, exist_ok=True)
    backup = _preserve_existing(config)
    config_copy = dict(config)
    config_copy.pop("_config_path", None)
    (destination / "config_used.yaml").write_text(
        yaml.safe_dump(config_copy, sort_keys=False), encoding="utf-8"
    )
    universe = _selected_universe(config)
    reporters = universe["comtrade_code"].astype(str).tolist()
    partners = reporters.copy()
    months = [step.strftime("%Y%m") for step in monthly_steps(config)]
    month_labels = [f"{month[:4]}-{month[4:]}" for month in months]
    source = nested(config, "sources", "comtrade", default={})
    cache_dir_value = source.get(
        "bilateral_cache_dir", "data/four_year_2021_2024/cache/comtrade/bilateral"
    )
    cache_dir = Path(cache_dir_value)
    cache_dir = cache_dir if cache_dir.is_absolute() else _base() / cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    manifest = {"version": 1, "requests": {}}
    if manifest_path.exists() and not force:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)

    frames: list[pd.DataFrame] = []
    failures: list[dict[str, Any]] = []
    delay = float(source.get("request_delay_seconds", 1.0))
    total = len(reporters) * len(months)
    completed = 0
    for reporter in reporters:
        for period, month in zip(months, month_labels):
            key = f"{reporter}:{period}"
            cached_path = _cache_path(cache_dir, reporter, period)
            try:
                if cached_path.exists() and not force:
                    payload = _read_cached(cached_path)
                    cached = True
                else:
                    payload = {
                        "request": {
                            "reporter_code": reporter,
                            "partner_codes": partners,
                            "period": period,
                            "flow_code": "M",
                            "commodity_codes": ["8541", "8542"],
                        },
                        "response": _api_request(config, reporter, partners, period),
                    }
                    _atomic_write_json(payload, cached_path)
                    cached = False
                    if delay:
                        time.sleep(delay)
                raw = _raw_frame(payload)
                standardized = _standardize_bilateral(raw, period, vintage_now())
                frames.append(standardized)
                manifest["requests"][key] = {
                    "status": "success",
                    "cached": cached,
                    "records": int(len(standardized)),
                    "cache_file": str(cached_path),
                }
            except Exception as exc:
                failures.append(
                    {"reporter_code": reporter, "period": period, "error": str(exc)}
                )
                manifest["requests"][key] = {"status": "failed", "error": str(exc)}
            completed += 1
            if completed % 10 == 0 or completed == total:
                print(f"[Comtrade bilateral] completed {completed}/{total}")
            _atomic_write_json(manifest, manifest_path)

    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    coverage = _coverage_grid(frame, reporters, partners, month_labels)
    coverage.to_csv(destination / "reporter_partner_matrix.csv", index=False)
    monthly = (
        coverage.groupby("month", as_index=False)
        .agg(
            expected_cells=("observed", "size"),
            observed_cells=("observed", "sum"),
            observed_records=("observed_records", "sum"),
        )
    )
    monthly["missing_cells"] = monthly["expected_cells"] - monthly["observed_cells"]
    monthly.to_csv(destination / "monthly_coverage.csv", index=False)
    pair_coverage = coverage.groupby(
        ["reporter_code", "partner_code"], as_index=False
    )["observed"].any()
    reporter_month_coverage = coverage.groupby(
        ["reporter_code", "month"], as_index=False
    )["observed"].any()
    partner_month_coverage = coverage.groupby(
        ["partner_code", "month"], as_index=False
    )["observed"].any()

    raw_records = int(len(frame))
    summary = {
        "period": {"start": month_labels[0], "end": month_labels[-1]},
        "requested_reporters": reporters,
        "requested_partners": partners,
        "requested_hs_codes": ["8541", "8542"],
        "requested_flow": "M",
        "raw_records": raw_records,
        "unique_reporters": sorted(frame["reporter_code"].astype(str).unique().tolist())
        if not frame.empty
        else [],
        "unique_partners": sorted(frame["partner_code"].astype(str).unique().tolist())
        if not frame.empty
        else [],
        "unique_reporter_count": int(frame["reporter_code"].nunique()) if not frame.empty else 0,
        "unique_partner_count": int(frame["partner_code"].nunique()) if not frame.empty else 0,
        "missing_requested_reporters": sorted(
            set(reporters)
            - set(frame["reporter_code"].astype(str).unique().tolist())
        )
        if not frame.empty
        else reporters,
        "missing_requested_partners": sorted(
            set(partners)
            - set(frame["partner_code"].astype(str).unique().tolist())
        )
        if not frame.empty
        else partners,
        "missing_reporter_partner_pairs": int((~pair_coverage["observed"]).sum()),
        "missing_reporter_months": int((~reporter_month_coverage["observed"]).sum()),
        "missing_partner_months": int((~partner_month_coverage["observed"]).sum()),
        "reporter_partner_pairs": int(
            frame[["reporter_code", "partner_code"]].drop_duplicates().shape[0]
        )
        if not frame.empty
        else 0,
        "months": sorted(frame["month"].astype(str).unique().tolist())
        if not frame.empty
        else [],
        "month_count": int(frame["month"].nunique()) if not frame.empty else 0,
        "hs_codes": sorted(frame["commodity_code"].astype(str).unique().tolist())
        if not frame.empty
        else [],
        "hs_code_count": int(frame["commodity_code"].nunique()) if not frame.empty else 0,
        "import_records": int((frame["flow_code"].astype(str) == "M").sum())
        if not frame.empty
        else 0,
        "duplicate_business_keys": int(
            frame.duplicated(
                ["month", "reporter_code", "partner_code", "commodity_code", "flow_code"]
            ).sum()
        )
        if not frame.empty
        else 0,
        "api_failures": failures,
        "missing_cells": int((~coverage["observed"]).sum()),
        "missing_trade_converted_to_zero": False,
        "cache_dir": str(cache_dir),
        "backup_dir": str(backup),
    }
    write_json(summary, destination / "coverage_summary.json")
    if failures:
        _write_readme(destination, reporters, partners, month_labels, summary, failures, backup)
        raise RuntimeError(
            f"{len(failures)} bilateral requests failed; processed outputs were not replaced"
        )

    universe["country_name"] = universe.apply(
        lambda row: COUNTRY_NAMES.get(str(row["comtrade_code"]), row["country_name"]),
        axis=1,
    )
    write_table(
        universe,
        _output_path(config, "country_universe", "processed/country_universe.csv"),
    )
    processed_path = _output_path(config, "comtrade", "processed/comtrade.csv")
    write_table(frame, processed_path)
    summary["processed_records"] = int(len(frame))
    write_json(summary, destination / "coverage_summary.json")

    node_path, edge_path = fuse(config)
    graph_path = build_graph(config)
    nodes = read_table(node_path)
    target_summary = _target_statistics(nodes, config, destination)
    summary["processed_nodes"] = str(node_path)
    summary["processed_edges"] = str(edge_path)
    summary["graph"] = str(graph_path)
    summary["target_coverage"] = target_summary
    summary["country_code_mapping"] = {
        code: COUNTRY_NAMES.get(code, "unresolved") for code in reporters
    }
    write_json(summary, destination / "coverage_summary.json")
    _write_readme(destination, reporters, partners, month_labels, summary, failures, backup)
    write_json(
        {
            "selected_universe": [
                {
                    "code": str(row["comtrade_code"]),
                    "name": str(row["country_name"]),
                }
                for _, row in universe.iterrows()
            ],
            "unresolved_codes": [
                code for code in reporters if code not in COUNTRY_NAMES
            ],
        },
        destination / "country_code_mapping.json",
    )
    print(f"Wrote bilateral Comtrade data to {processed_path}")
    print(f"Wrote coverage diagnostics to {destination}")
    print("Models were not trained.")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_cli(parser)
    args = parser.parse_args()
    output = run(load_config(args.config), force=args.force)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
