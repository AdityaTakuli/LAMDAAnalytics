"""Ingest monthly bilateral UN Comtrade data for the configured ECGB basket.

The official ``comtradeapicall`` package is used when installed.  The package
returns a DataFrame from ``getFinalData`` (or the no-key
``previewFinalData``).  Each monthly response is cached independently, which
is important because Comtrade can revise historical observations.

The resulting table is intentionally still source-shaped.  ``fuse_dataset.py``
performs country identifiers, inbound aggregation, and feature derivation.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import pandas as pd

from common import (
    add_cache_metadata,
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
    write_table,
)


def _call_comtrade(config: dict, period: str) -> pd.DataFrame:
    try:
        package = importlib.import_module("comtradeapicall")
    except ImportError as exc:
        raise RuntimeError(
            "comtradeapicall is required for live Comtrade ingestion. "
            "Install dataset/requirements.txt or provide --input."
        ) from exc

    source = nested(config, "sources", "comtrade", default={})
    key = env_value(config, source.get("subscription_key_env"), "")
    secondary_key = env_value(config, source.get("secondary_key_env"), "")
    function_name = "getFinalData" if key else "previewFinalData"
    function = getattr(package, function_name)
    kwargs = {
        "typeCode": "C",
        "freqCode": "M",
        "clCode": source.get("classification", "HS"),
        "period": period,
        "reporterCode": source.get("reporter_code", "all"),
        "cmdCode": ",".join(
            str(code)
            for code in source.get(
                "hs_codes", [8517, 8471, 8528, 8516, 8507, 9403, 6109, 6203, 6204, 9503, 3304]
            )
        ),
        "flowCode": source.get("flow_code", "M"),
        "partnerCode": None,
        "partner2Code": None,
        "customsCode": None,
        "motCode": None,
        "maxRecords": int(source.get("max_records", 250000 if key else 500)),
        "format_output": "JSON",
        "aggregateBy": None,
        "breakdownMode": "classic",
        "countOnly": None,
        "includeDesc": True,
    }
    if key:
        try:
            return pd.DataFrame(function(key, **kwargs))
        except Exception as primary_error:
            if not secondary_key:
                raise
            print(
                f"Primary Comtrade key failed for {period}; trying secondary key.",
                file=sys.stderr,
            )
            try:
                return pd.DataFrame(function(secondary_key, **kwargs))
            except Exception:
                raise primary_error
    return pd.DataFrame(function(**kwargs))


def standardize(frame: pd.DataFrame, period: str, vintage: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "month",
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

    reporter = canonical_column(frame, ["reporterCode", "reporterCodeISO", "reporter"])
    partner = canonical_column(frame, ["partnerCode", "partnerCodeISO", "partner"])
    commodity = canonical_column(frame, ["cmdCode", "cmdCodeISO", "commodityCode", "commodity"])
    flow = canonical_column(frame, ["flowCode", "flow"])
    trade_value = canonical_column(
        frame, ["primaryValue", "primaryValueUSD", "tradeValue", "tradeValueUSD", "cifvalue", "fobvalue"]
    )
    volume = canonical_column(frame, ["netWgt", "netWeight", "qty", "quantity", "grossWgt"])
    published = canonical_column(frame, ["publicationDate", "publishedAt", "dateAdded", "date"])

    result = pd.DataFrame(index=frame.index)
    result["month"] = period[:4] + "-" + period[4:6]
    result["reporter_code"] = frame[reporter].astype(str) if reporter else "all"
    result["partner_code"] = frame[partner].astype(str) if partner else "0"
    result["commodity_code"] = frame[commodity].astype(str) if commodity else "ECGB"
    result["flow_code"] = frame[flow].astype(str) if flow else "M"
    result["trade_value_usd"] = numeric(frame, trade_value)
    result["flow_volume"] = numeric(frame, volume)
    result["published_at"] = (
        frame[published].map(lambda value: str(value)) if published else vintage
    )
    result["available_at"] = result["published_at"]
    result["vintage_date"] = vintage
    result["source"] = "un_comtrade"
    return result.reset_index(drop=True)


def _norm_comtrade_code(value: object) -> str:
    text = str(value).strip()
    if text.lower() in {"", "nan", "none"}:
        return ""
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def select_country_universe(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Keep India fixed and select countries with reporter-level inbound data."""
    focal = nested(config, "analysis", "focal_country", default={})
    country_count = int(nested(config, "analysis", "country_count", default=20))
    focal_code = _norm_comtrade_code(focal.get("comtrade_code", "699"))

    if frame.empty or "partner_code" not in frame or "reporter_code" not in frame:
        reporters = pd.DataFrame(columns=["comtrade_code", "trade_value_usd"])
        partner_rank = pd.DataFrame(columns=["comtrade_code", "trade_value_usd"])
    else:
        working = frame.copy()
        working["reporter_code"] = working["reporter_code"].map(_norm_comtrade_code)
        working["partner_code"] = working["partner_code"].map(_norm_comtrade_code)
        reporter_codes = {
            code for code in working["reporter_code"].dropna().astype(str) if code
        }
        reporters = (
            working.groupby("reporter_code", as_index=False)["trade_value_usd"]
            .sum()
            .rename(columns={"reporter_code": "comtrade_code"})
            .sort_values("trade_value_usd", ascending=False)
        )
        focal_imports = working[working["reporter_code"] == focal_code]
        partner_rank = (
            focal_imports[~focal_imports["partner_code"].isin({"0", ""})]
            .groupby("partner_code", as_index=False)["trade_value_usd"]
            .sum()
            .rename(columns={"partner_code": "comtrade_code"})
            .sort_values("trade_value_usd", ascending=False)
        )
        partner_rank = partner_rank[
            partner_rank["comtrade_code"].isin(reporter_codes)
            & (partner_rank["comtrade_code"] != focal_code)
        ]

    records = [
        {
            "rank": 1,
            "country_name": focal.get("name", "India"),
            "comtrade_code": focal_code,
            "iso3": focal.get("iso3", "IND"),
            "trade_value_usd": float(
                reporters.loc[reporters["comtrade_code"] == focal_code, "trade_value_usd"].sum()
            )
            if not reporters.empty
            else float(frame["trade_value_usd"].sum()) if not frame.empty else 0.0,
            "selection_reason": "fixed_focal_country",
        }
    ]
    selected_codes = {focal_code}
    for _, row in partner_rank.iterrows():
        if len(records) >= country_count:
            break
        code = _norm_comtrade_code(row["comtrade_code"])
        if not code or code in selected_codes:
            continue
        location = next(
            (
                item
                for item in nested(config, "sources", "weather", "locations", default=[])
                if _norm_comtrade_code(item.get("comtrade_code", "")) == code
            ),
            {},
        )
        records.append(
            {
                "rank": len(records) + 1,
                "country_name": location.get("name", code),
                "comtrade_code": code,
                "iso3": location.get("iso3", ""),
                "trade_value_usd": float(row["trade_value_usd"]),
                "selection_reason": "top_partner_with_reporter_data",
            }
        )
        selected_codes.add(code)

    if len(records) < country_count and not reporters.empty:
        for _, row in reporters.iterrows():
            if len(records) >= country_count:
                break
            code = _norm_comtrade_code(row["comtrade_code"])
            if not code or code in selected_codes:
                continue
            location = next(
                (
                    item
                    for item in nested(config, "sources", "weather", "locations", default=[])
                    if _norm_comtrade_code(item.get("comtrade_code", "")) == code
                ),
                {},
            )
            records.append(
                {
                    "rank": len(records) + 1,
                    "country_name": location.get("name", code),
                    "comtrade_code": code,
                    "iso3": location.get("iso3", ""),
                    "trade_value_usd": float(row["trade_value_usd"]),
                    "selection_reason": "reporter_backfill",
                }
            )
            selected_codes.add(code)

    return pd.DataFrame(records)


def ingest(config: dict, force: bool = False, input_path: str | None = None) -> Path:
    destination = Path(nested(config, "outputs", "comtrade", default="processed/comtrade.csv"))
    if not destination.is_absolute():
        destination = Path(__file__).resolve().parent / destination
    if input_path:
        frame = read_table(Path(input_path))
        universe_path = Path(__file__).resolve().parent / nested(
            config, "outputs", "country_universe", default="processed/country_universe.csv"
        )
        write_table(select_country_universe(frame, config), universe_path)
        return write_table(frame, destination)
    if destination.exists() and not force:
        universe_path = Path(__file__).resolve().parent / nested(
            config, "outputs", "country_universe", default="processed/country_universe.csv"
        )
        if not (universe_path.exists() or universe_path.with_suffix(".csv").exists()):
            write_table(select_country_universe(read_table(destination), config), universe_path)
        return destination

    rows: list[pd.DataFrame] = []
    source = nested(config, "sources", "comtrade", default={})
    raw_dir = Path(__file__).resolve().parent / source.get("cache_dir", "cache/comtrade")
    raw_dir.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz="UTC")

    for step in monthly_steps(config):
        period = step.strftime("%Y%m")
        raw_path = raw_dir / f"comtrade_{period}.csv"
        if raw_path.exists() and not force:
            raw = pd.read_csv(raw_path)
        elif step > now:
            print(f"Skipping future Comtrade period {period}", file=sys.stderr)
            continue
        else:
            try:
                raw = _call_comtrade(config, period)
            except Exception as exc:
                raise RuntimeError(
                    f"Comtrade request failed for {period}. Cached files can be "
                    "used with --force omitted, or pass --input to ingest a local pull."
                ) from exc
            raw.to_csv(raw_path, index=False)
        rows.append(standardize(raw, period, vintage_now()))

    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    universe_path = Path(__file__).resolve().parent / nested(
        config, "outputs", "country_universe", default="processed/country_universe.csv"
    )
    write_table(select_country_universe(result, config), universe_path)
    return write_table(result, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_cli(parser)
    parser.add_argument("--input", help="Use a local Comtrade CSV/parquet instead of the API")
    args = parser.parse_args()
    output = ingest(load_config(args.config), force=args.force, input_path=args.input)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

