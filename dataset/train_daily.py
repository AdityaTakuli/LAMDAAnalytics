"""Validate the daily target gate before any 2024 model training.

The current 2024 daily sources do not contain an independent daily target.
This command records the isolated training experiment, validates the proposed
chronological split, and stops without importing or executing model training.
It never writes to ``one_year_2024`` or ``one_year_2024_daily``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from common import load_config, nested, read_table, write_json


BASE = Path(__file__).resolve().parent
RESULT_SUBDIRECTORIES = (
    "diagnostics",
    "target_diagnostics",
    "metrics",
    "predictions",
    "checkpoints",
    "graphs",
)


def _path(config: dict[str, Any], section: str, key: str, default: str) -> Path:
    value = nested(config, section, key, default=default)
    path = Path(value)
    return path if path.is_absolute() else BASE / path


def _date_range(config: dict[str, Any]) -> pd.DatetimeIndex:
    return pd.date_range(
        pd.Timestamp(nested(config, "analysis", "start_date"), tz="UTC"),
        pd.Timestamp(nested(config, "analysis", "end_date"), tz="UTC"),
        freq="D",
    )


def _split(config: dict[str, Any]) -> dict[str, list[str]]:
    settings = nested(config, "daily_training", default={})
    split = {}
    for name in ("train", "validation", "test"):
        start = pd.Timestamp(settings[f"{name}_start"]).strftime("%Y-%m-%d")
        end = pd.Timestamp(settings[f"{name}_end"]).strftime("%Y-%m-%d")
        split[name] = [day.strftime("%Y-%m-%d") for day in pd.date_range(start, end, freq="D")]
    return split


def _validate_split(
    config: dict[str, Any], dates: pd.DatetimeIndex, nodes: pd.DataFrame
) -> dict[str, Any]:
    split = _split(config)
    all_days = set(dates.strftime("%Y-%m-%d"))
    memberships = [day for values in split.values() for day in values]
    if len(memberships) != len(set(memberships)):
        raise ValueError("Daily train/validation/test ranges overlap")
    if not set(memberships).issubset(all_days):
        raise ValueError("Daily split contains dates outside the configured 2024 range")
    excluded = nested(config, "daily_training", "excluded_dates", default=[])
    excluded = [pd.Timestamp(day).strftime("%Y-%m-%d") for day in excluded]
    if set(excluded) & set(memberships):
        raise ValueError("An excluded date is present in a training split")
    node_days = nodes["date"].astype(str).str[:10]
    return {
        "split": split,
        "excluded_dates": excluded,
        "split_country_day_rows": {
            name: int(node_days.isin(values).sum()) for name, values in split.items()
        },
        "split_date_counts": {name: len(values) for name, values in split.items()},
        "training_executed": False,
    }


def run(config: dict[str, Any], force: bool = False) -> Path:
    input_root = _path(
        config,
        "daily_training",
        "input_root",
        "data/one_year_2024_daily",
    )
    output_root = _path(
        config,
        "outputs",
        "root",
        "data/one_year_2024_daily_training",
    )
    results = _path(
        config,
        "outputs",
        "results",
        "data/one_year_2024_daily_training/results",
    )
    if not input_root.exists():
        raise FileNotFoundError(f"Daily diagnostic input does not exist: {input_root}")
    if output_root.exists():
        if not force:
            raise FileExistsError(
                f"Daily training output exists; use --force to rebuild only {output_root}"
            )
        shutil.rmtree(output_root)
    for name in RESULT_SUBDIRECTORIES:
        (results / name).mkdir(parents=True, exist_ok=True)

    nodes_path = input_root / "processed" / "nodes_daily.csv"
    summary_path = input_root / "results" / "diagnostics" / "dataset_summary.json"
    target_path = input_root / "results" / "target_diagnostics" / "target_distribution.json"
    nodes = read_table(nodes_path)
    dataset_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    target_summary = json.loads(target_path.read_text(encoding="utf-8"))
    dates = _date_range(config)
    split_statistics = _validate_split(config, dates, nodes)

    if target_summary.get("status") != "ready":
        gate = {
            "status": "blocked",
            "training_executed": False,
            "target_status": target_summary.get("status"),
            "reason": target_summary.get("reason"),
            "required_before_training": (
                "An independently observed daily outcome with a pre-specified "
                "horizon and timestamp availability rule."
            ),
            "models": ["gcn", "tgn", "tgn_no_memory"],
        }
    else:
        raise RuntimeError(
            "A daily target is marked ready, but this target-gate command does not "
            "train models without an explicit approved training implementation."
        )

    config_copy = dict(config)
    config_copy.pop("_config_path", None)
    (results / "config_used.yaml").write_text(
        yaml.safe_dump(config_copy, sort_keys=False), encoding="utf-8"
    )
    diagnostics = results / "diagnostics"
    target_diagnostics = results / "target_diagnostics"
    for source_name, destination_name in (
        ("dataset_summary.json", "dataset_summary.json"),
        ("provenance.json", "provenance.json"),
    ):
        source = input_root / "results" / "diagnostics" / source_name
        if source.exists():
            shutil.copy2(source, diagnostics / destination_name)
    source_target_dir = input_root / "results" / "target_diagnostics"
    for source in source_target_dir.iterdir():
        if source.is_file():
            shutil.copy2(source, target_diagnostics / source.name)
    leakage_source = input_root / "results" / "diagnostics" / "daily_leakage_audit.md"
    leakage_text = leakage_source.read_text(encoding="utf-8") if leakage_source.exists() else ""
    (diagnostics / "daily_leakage_audit.md").write_text(
        leakage_text
        + "\n\n## Daily training gate\n\n"
        + "Status: **BLOCKED before training.** The target diagnostics report "
        + f"`{target_summary.get('status')}`. No scaler was fitted, no graph was "
        + "replayed, no TGN memory was initialized, and no model artifacts were created.\n",
        encoding="utf-8",
    )
    write_json(dataset_summary, diagnostics / "dataset_summary.json")
    write_json(split_statistics, diagnostics / "split_statistics.json")
    write_json(gate, diagnostics / "training_gate.json")
    (results / "README.md").write_text(
        """# 2024 daily training experiment

This isolated experiment reads `one_year_2024_daily` and never modifies that
diagnostic dataset or the monthly `one_year_2024` profile.

Training is intentionally blocked. The 2024 source tables provide daily
GDELT/weather predictors and monthly Comtrade/GSCPI structural inputs, but no
independent daily disruption outcome. A monthly target repeated across days,
future GDELT/weather observations, or fabricated events would be invalid.

The proposed chronological split is recorded in
`diagnostics/split_statistics.json`. GCN, TGN, and TGN-no-memory checkpoints,
predictions, metrics, and training plots were not created because the target
gate failed.
""",
        encoding="utf-8",
    )
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_daily_training.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = run(load_config(args.config), force=args.force)
    print(f"Wrote isolated daily training diagnostics to {output}")
    print("DAILY TARGET NOT SCIENTIFICALLY VALID — training stopped.")


if __name__ == "__main__":
    main()
