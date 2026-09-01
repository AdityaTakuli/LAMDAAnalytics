"""Profile the acquired 2021-2024 data before any model training.

This is an analysis-only command. It reads raw caches and the existing 2024
processed source tables, then writes a coverage, quality, target, and training
readiness report. It does not fuse data, build graphs, or train models.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from common import load_config, nested


BASE = Path(__file__).resolve().parent
YEARS = (2021, 2022, 2023, 2024)
MONTHS = [f"{year}-{month:02d}" for year in YEARS for month in range(1, 13)]
DATE_PATTERN = re.compile(r"(20\d{6})")


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else BASE / path


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _codes(config: dict[str, Any]) -> list[str]:
    return [
        str(value)
        for value in nested(
            config, "sources", "comtrade", "reporter_codes", default=[]
        )
    ]


FOUR_YEAR_ROOT = BASE / "data" / "four_year_2021_2024"


def _comtrade_manifest_paths() -> dict[int, Path]:
    root = FOUR_YEAR_ROOT
    paths = {year: root / f"comtrade_{year}_manifest.json" for year in (2021, 2022, 2023)}
    paths[2024] = root / "cache" / "comtrade" / "bilateral" / "manifest.json"
    return paths


def _comtrade_cache_paths() -> dict[int, Path]:
    root = FOUR_YEAR_ROOT / "cache" / "comtrade" / "bilateral"
    return {year: root for year in YEARS}


def _manifest_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for year, path in _comtrade_manifest_paths().items():
        manifest = _json(path)
        requests = manifest.get("requests", {})
        statuses = pd.Series(
            [str(value.get("status", "unknown")) for value in requests.values()]
        ).value_counts()
        summary[str(year)] = {
            "manifest": str(path),
            "requests": int(len(requests)),
            "status_counts": {str(key): int(value) for key, value in statuses.items()},
            "failed_requests": int((statuses.get("failed", 0))),
        }
    return summary


def _comtrade_records() -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    response_summary: dict[str, Any] = {}
    for year, directory in _comtrade_cache_paths().items():
        files = sorted(directory.glob(f"reporter_*_{year}*.json"))
        response_summary[str(year)] = {
            "cache_dir": str(directory),
            "response_files": len(files),
            "empty_successful_responses": 0,
        }
        for path in files:
            payload = _json(path)
            response = payload.get("response", {})
            data = response.get("data", [])
            if not data:
                response_summary[str(year)]["empty_successful_responses"] += 1
            for item in data:
                rows.append(
                    {
                        "year": year,
                        "period": str(item.get("period", ""))[:6],
                        "reporter_code": str(item.get("reporterCode", "")),
                        "partner_code": str(item.get("partnerCode", "")),
                        "flow_code": str(item.get("flowCode", "")),
                        "cmd_code": str(item.get("cmdCode", "")),
                        "primary_value": item.get("primaryValue"),
                        "net_weight": item.get("netWgt"),
                        "quantity": item.get("qty"),
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, response_summary
    for column in ("primary_value", "net_weight", "quantity"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["month"] = pd.to_datetime(frame["period"], format="%Y%m", errors="coerce").dt.strftime(
        "%Y-%m"
    )
    return frame, response_summary


def _comtrade_analysis(
    frame: pd.DataFrame, codes: list[str], manifests: dict[str, Any], responses: dict[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame]:
    expected_requests = len(codes) * 12 * 4
    expected_pair_months = expected_requests * len(codes)
    if frame.empty:
        return (
            {
                "request_manifest": manifests,
                "response_files": responses,
                "expected_requests": expected_requests,
                "expected_reporter_partner_month_cells": expected_pair_months,
                "records": 0,
                "observed_reporter_partner_month_cells": 0,
                "duplicate_observation_keys": 0,
                "negative_primary_value_rows": 0,
                "missing_primary_value_rows": 0,
                "zero_primary_value_rows": 0,
                "all_requests_succeeded": False,
            },
            pd.DataFrame(),
        )
    key_columns = ["period", "reporter_code", "partner_code", "cmd_code"]
    pair_months = frame[key_columns[:3]].drop_duplicates()
    duplicate_count = int(frame.duplicated(key_columns).sum())
    inbound = (
        frame[frame["partner_code"].isin(codes)]
        .groupby(["month", "reporter_code"], as_index=False)["primary_value"]
        .sum(min_count=1)
        .rename(columns={"reporter_code": "country_code", "primary_value": "inbound_value_usd"})
    )
    summary = {
        "request_manifest": manifests,
        "response_files": responses,
        "expected_requests": expected_requests,
        "expected_reporter_partner_month_cells": expected_pair_months,
        "records": int(len(frame)),
        "observed_reporter_partner_month_cells": int(len(pair_months)),
        "reporter_months_with_records": int(
            frame[["month", "reporter_code"]].drop_duplicates().shape[0]
        ),
        "observed_trade_cell_coverage_fraction": float(
            len(pair_months) / expected_pair_months
        ),
        "empty_successful_responses_total": int(
            sum(item["empty_successful_responses"] for item in responses.values())
        ),
        "unique_reporters_in_records": sorted(frame["reporter_code"].dropna().unique().tolist()),
        "unique_partners_in_records": sorted(frame["partner_code"].dropna().unique().tolist()),
        "months_in_records": sorted(frame["month"].dropna().unique().tolist()),
        "hs_codes_in_records": sorted(frame["cmd_code"].dropna().unique().tolist()),
        "flow_codes_in_records": sorted(frame["flow_code"].dropna().unique().tolist()),
        "duplicate_observation_keys": duplicate_count,
        "negative_primary_value_rows": int(frame["primary_value"].lt(0).sum()),
        "missing_primary_value_rows": int(frame["primary_value"].isna().sum()),
        "zero_primary_value_rows": int(frame["primary_value"].fillna(0).eq(0).sum()),
        "all_requests_succeeded": all(
            value["failed_requests"] == 0
            and value["status_counts"].get("success", 0) == value["requests"]
            for value in manifests.values()
        ),
    }
    return summary, inbound


def _target_analysis(inbound: pd.DataFrame, codes: list[str]) -> dict[str, Any]:
    index = pd.MultiIndex.from_product([codes, MONTHS], names=["country_code", "month"])
    values = inbound.set_index(["country_code", "month"])["inbound_value_usd"].reindex(index)
    values = values.fillna(0.0).rename("inbound_value_usd").reset_index()
    values["month_date"] = pd.to_datetime(values["month"] + "-01", utc=True)
    values = values.sort_values(["country_code", "month_date"])
    values["baseline"] = values.groupby("country_code")["inbound_value_usd"].transform(
        # Eq. 7 for h=1 uses V[T-11] ... V[T], including the current
        # feature month and excluding the future target month T+1.
        lambda series: series.rolling(12, min_periods=12).median()
    )
    values["future_value"] = values.groupby("country_code")["inbound_value_usd"].shift(-1)
    values["contraction"] = (values["future_value"] - values["baseline"]) / values["baseline"]
    values["target_valid"] = (
        values["baseline"].gt(0)
        & values["future_value"].notna()
        & values["contraction"].notna()
    )
    valid = values[values["target_valid"]].copy()
    by_tau: dict[str, Any] = {}
    for tau in (0.30, 0.35, 0.40):
        positives = int(valid["contraction"].lt(-tau).sum())
        by_tau[f"{tau:.2f}"] = {
            "positive": positives,
            "negative": int(len(valid) - positives),
            "prevalence": float(positives / len(valid)) if len(valid) else None,
        }
    splits = {
        "train_2021_2022": values[values["month"].str[:4].isin(["2021", "2022"])],
        "validation_2023": values[values["month"].str[:4].eq("2023")],
        "test_2024": values[values["month"].str[:4].eq("2024")],
    }
    split_summary: dict[str, Any] = {}
    for name, split in splits.items():
        split_valid = split[split["target_valid"]]
        positives = int(split_valid["contraction"].lt(-0.35).sum())
        split_summary[name] = {
            "rows": int(len(split)),
            "target_valid": int(len(split_valid)),
            "positive_tau_0.35": positives,
            "negative_tau_0.35": int(len(split_valid) - positives),
            "prevalence_tau_0.35": float(positives / len(split_valid))
            if len(split_valid)
            else None,
        }
    return {
        "country_month_rows": int(len(values)),
        "expected_country_month_rows": len(codes) * len(MONTHS),
        "valid_target_rows": int(len(valid)),
        "invalid_target_rows": int(len(values) - len(valid)),
        "target_month_range": [
            str(valid["month"].min()) if len(valid) else None,
            str(valid["month"].max()) if len(valid) else None,
        ],
        "zero_inbound_country_months": int(values["inbound_value_usd"].eq(0).sum()),
        "contraction_statistics": {
            "min": float(valid["contraction"].min()) if len(valid) else None,
            "median": float(valid["contraction"].median()) if len(valid) else None,
            "mean": float(valid["contraction"].mean()) if len(valid) else None,
            "std": float(valid["contraction"].std()) if len(valid) > 1 else 0.0,
            "max": float(valid["contraction"].max()) if len(valid) else None,
        },
        "counts_by_tau": by_tau,
        "chronological_split": split_summary,
        "has_both_classes_in_every_split_tau_0.35": all(
            item["positive_tau_0.35"] > 0 and item["negative_tau_0.35"] > 0
            for item in split_summary.values()
            if item["target_valid"]
        ),
    }


def _gdelt_analysis() -> dict[str, Any]:
    roots = [
        FOUR_YEAR_ROOT / "cache" / "gdelt",
    ]
    dates: set[str] = set()
    files = 0
    for root in roots:
        for path in root.rglob("*.zip"):
            files += 1
            match = DATE_PATTERN.search(path.name)
            if match:
                dates.add(match.group(1))
    start = date(2021, 1, 1)
    end = date(2024, 12, 31)
    expected = {
        (start + timedelta(days=offset)).strftime("%Y%m%d")
        for offset in range((end - start).days + 1)
    }
    dates &= expected
    return {
        "cache_roots": [str(root) for root in roots],
        "zip_files_seen": files,
        "unique_daily_exports": len(dates),
        "expected_daily_exports": len(expected),
        "missing_dates": sorted(expected - dates),
        "coverage_fraction": float(len(dates) / len(expected)),
    }


def _weather_analysis(expected_locations: list[str]) -> dict[str, Any]:
    roots = [
        FOUR_YEAR_ROOT / "cache" / "weather",
    ]
    json_files = [path for root in roots for path in root.rglob("*.json")]
    daily_rows = 0
    locations: dict[str, int] = {}
    for path in json_files:
        payload = _json(path)
        parameters = payload.get("properties", {}).get("parameter", {})
        dates = set()
        for values in parameters.values():
            if isinstance(values, dict):
                dates.update(str(key) for key in values)
        locations[path.stem] = len(dates)
        daily_rows += len(dates)
    processed = FOUR_YEAR_ROOT / "processed" / "weather_daily.csv"
    processed_summary: dict[str, Any] = {"path": str(processed), "exists": processed.exists()}
    if processed.exists():
        frame = pd.read_csv(processed)
        processed_locations = (
            sorted(frame["location_name"].dropna().astype(str).unique().tolist())
            if "location_name" in frame
            else []
        )
        processed_summary.update(
            {
                "rows": int(len(frame)),
                "countries": len(processed_locations),
                "locations": processed_locations,
                "missing_expected_locations": sorted(
                    set(expected_locations) - set(processed_locations)
                ),
                "min_date": str(frame["observed_date"].min())
                if "observed_date" in frame
                else None,
                "max_date": str(frame["observed_date"].max())
                if "observed_date" in frame
                else None,
            }
        )
    return {
        "raw_json_files": len(json_files),
        "raw_location_day_counts": locations,
        "raw_location_days_total": daily_rows,
        "processed_profile": processed_summary,
    }


def _gscpi_analysis() -> dict[str, Any]:
    workbook_root = FOUR_YEAR_ROOT / "cache" / "gscpi"
    workbooks = sorted(workbook_root.rglob("*.xlsx"))
    workbook = workbooks[0] if workbooks else workbook_root / "gscpi.xlsx"
    processed = FOUR_YEAR_ROOT / "processed" / "gscpi_monthly.csv"
    result: dict[str, Any] = {
        "raw_workbook": str(workbook),
        "raw_workbook_exists": workbook.exists(),
        "processed_profile": {"path": str(processed), "exists": processed.exists()},
    }
    if workbook.exists():
        try:
            sheets = pd.ExcelFile(workbook).sheet_names
            result["raw_sheet_names"] = sheets
        except (OSError, ValueError):
            result["raw_sheet_names"] = []
    if processed.exists():
        frame = pd.read_csv(processed)
        result["processed_profile"].update(
            {
                "rows": int(len(frame)),
                "columns": list(frame.columns),
                "min_date": str(frame.iloc[:, 0].min()) if len(frame) else None,
                "max_date": str(frame.iloc[:, 0].max()) if len(frame) else None,
            }
        )
    return result


def _write_report(destination: Path, report: dict[str, Any]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "analysis_summary.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    readiness = report["training_readiness"]
    lines = [
        "# 2021–2024 data analysis",
        "",
        f"Overall training readiness: **{readiness['decision']}**",
        "",
        readiness["reason"],
        "",
        "## Gate summary",
        "",
        f"- Acquisition gate: **{readiness['acquisition_gate']}** "
        "(all Comtrade requests returned successfully).",
        f"- Feature gate: **{readiness['feature_gate']}** "
        "(four-year processed artifacts and complete 2024 weather coverage required).",
        f"- Label gate: **{readiness['label_gate']}** "
        "(both classes in each split and at least 10 training positives required).",
        f"- Continuous-target gate: **{readiness['continuous_target_gate']}** "
        f"({report['target']['valid_target_rows']} valid non-constant contraction "
        "targets support regression).",
        "",
        "## Acquisition findings",
        "",
        f"- Comtrade requests: {report['comtrade']['total_successful_requests']} successful of "
        f"{report['comtrade']['total_expected_requests']}.",
        f"- Comtrade records: {report['comtrade']['records']}.",
        f"- Reporter-partner-month cells with records: "
        f"{report['comtrade']['observed_reporter_partner_month_cells']} of "
        f"{report['comtrade']['expected_reporter_partner_month_cells']} "
        f"({report['comtrade']['observed_trade_cell_coverage_fraction']:.1%}); "
        "unobserved cells are not failed API requests.",
        f"- Successful Comtrade responses with no records: "
        f"{report['comtrade']['empty_successful_responses_total']}.",
        f"- GDELT daily export coverage: {report['gdelt']['unique_daily_exports']} of "
        f"{report['gdelt']['expected_daily_exports']}; missing dates: "
        f"{', '.join(report['gdelt']['missing_dates']) or 'none'}.",
        f"- Processed weather profile: {report['weather']['processed_profile'].get('countries', 0)} "
        f"locations; missing configured locations: "
        f"{', '.join(report['weather']['processed_profile'].get('missing_expected_locations', [])) or 'none'}.",
        "",
        "## Target and split findings",
        "",
        f"- Country-month rows: {report['target']['country_month_rows']}.",
        f"- Valid one-month-ahead targets: {report['target']['valid_target_rows']}.",
        f"- Country-months with zero inbound value: "
        f"{report['target']['zero_inbound_country_months']}.",
        f"- Tau 0.35 counts: {json.dumps(report['target']['counts_by_tau']['0.35'])}.",
    ]
    for split, values in report["target"]["chronological_split"].items():
        lines.append(
            f"- {split}: {values['target_valid']} valid targets, "
            f"{values['positive_tau_0.35']} positive, {values['negative_tau_0.35']} negative."
        )
    lines.extend(
        [
            "",
            "## Required next step",
            "",
            "The binary classification gate remains blocked by class scarcity. "
            "Use the processed profile for a continuous contraction-regression pilot; "
            "do not train from raw JSON, ZIP, or Excel files.",
        ]
    )
    (destination / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config_path: str = "config.yaml") -> Path:
    config = load_config(config_path)
    codes = _codes(config)
    manifests = _manifest_summary()
    frame, responses = _comtrade_records()
    comtrade, inbound = _comtrade_analysis(frame, codes, manifests, responses)
    target = _target_analysis(inbound, codes)
    gdelt = _gdelt_analysis()
    expected_locations = [
        str(location.get("name"))
        for location in nested(config, "sources", "weather", "locations", default=[])
        if location.get("name")
    ]
    weather = _weather_analysis(expected_locations)
    gscpi = _gscpi_analysis()

    total_expected = sum(item["requests"] for item in manifests.values())
    total_successful = sum(item["status_counts"].get("success", 0) for item in manifests.values())
    processed_root = FOUR_YEAR_ROOT / "processed"
    processed_paths = [
        processed_root / "nodes_monthly.csv",
        processed_root / "edges_monthly.csv",
        processed_root / "graph.json",
    ]
    processed_complete = all(path.exists() for path in processed_paths)
    both_classes = target["has_both_classes_in_every_split_tau_0.35"]
    missing_weather = weather["processed_profile"].get("missing_expected_locations", [])
    train_positives = target["chronological_split"]["train_2021_2022"][
        "positive_tau_0.35"
    ]
    acquisition_gate = "PASS" if comtrade["all_requests_succeeded"] else "FAIL"
    feature_gate = (
        "PASS" if processed_complete and not missing_weather else "FAIL"
    )
    label_gate = (
        "PASS" if both_classes and train_positives >= 10 else "FAIL"
    )
    continuous_target_gate = (
        "PASS"
        if target["valid_target_rows"] >= 100
        and target["contraction_statistics"]["std"] > 0
        else "FAIL"
    )
    readiness_reasons: list[str] = []
    if not comtrade["all_requests_succeeded"]:
        readiness_reasons.append("Comtrade request coverage is incomplete")
    if not both_classes:
        readiness_reasons.append("at least one chronological split has a one-class target")
    if missing_weather:
        readiness_reasons.append(
            f"2024 processed weather is missing {len(missing_weather)} of "
            f"{len(expected_locations)} configured locations"
        )
    if train_positives < 10:
        readiness_reasons.append(
            f"the training split has only {train_positives} positive tau=0.35 targets"
        )
    if not processed_complete:
        readiness_reasons.append("four-year processed tables and leakage audits do not exist")
    if readiness_reasons:
        if feature_gate == "PASS" and continuous_target_gate == "PASS":
            decision = "READY FOR CONTINUOUS-TARGET PILOT; BINARY CLASSIFICATION BLOCKED"
            reason = (
                "; ".join(readiness_reasons)
                + "; use continuous contraction regression instead of a binary benchmark."
            )
        else:
            decision = "NOT READY FOR FINAL TRAINING"
            reason = "; ".join(readiness_reasons) + "."
    else:
        decision = "READY FOR TRAINING REVIEW"
        reason = "Source coverage, target classes, and processed artifacts passed this profile."

    report = {
        "analysis_scope": {
            "years": list(YEARS),
            "country_count": len(codes),
            "country_codes": codes,
            "monthly_snapshots": len(MONTHS),
            "analysis_timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        },
        "comtrade": {
            **comtrade,
            "total_expected_requests": total_expected,
            "total_successful_requests": total_successful,
        },
        "target": target,
        "gdelt": gdelt,
        "weather": weather,
        "gscpi": gscpi,
        "training_readiness": {
            "decision": decision,
            "reason": reason,
            "acquisition_gate": acquisition_gate,
            "feature_gate": feature_gate,
            "label_gate": label_gate,
            "continuous_target_gate": continuous_target_gate,
            "recommended_target": "continuous contraction regression",
            "four_year_processed_root_exists": processed_root.exists(),
            "four_year_processed_artifacts_complete": processed_complete,
            "gdelt_has_known_missing_dates": bool(gdelt["missing_dates"]),
            "target_has_both_classes_in_every_split_tau_0.35": both_classes,
            "do_not_train_yet": True,
            "models_trained": False,
        },
    }
    output_value = nested(
        config, "outputs", "analysis_results", default="data/four_year_2021_2024/results/data_analysis"
    )
    output = _path(output_value)
    _write_report(output, report)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    output = run(args.config)
    print(f"Wrote four-year analysis to {output}")


if __name__ == "__main__":
    main()
