#!/usr/bin/env python3
"""Run all models on country-month and bilateral setups; plot comparisons."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import load_config  # noqa: E402
from training import runtime  # noqa: E402
from training.benchmark_report import build_master_table, write_benchmark_plots, write_benchmark_report  # noqa: E402
from training.pair_pipeline import resolve_pair_options, run as run_pair  # noqa: E402
from training.pipeline import resolve_options, run as run_country  # noqa: E402

BASE = Path(__file__).resolve().parent
BENCHMARK_ROOT = BASE / "data/four_year_2021_2024/results/model_training/benchmark_comparison"


def main() -> int:
    runtime.configure_logging()
    config = load_config("config.yaml")
    graph_models = ["gcn", "tgn", "tgn_no_memory"]
    epochs = int((config.get("model_training") or {}).get("epochs", 40))

    runs: dict[str, dict[str, Path]] = {}

    for setup, runner, resolve, subpath in (
        ("country_month", run_country, resolve_options, "country_month"),
        ("pair_pooled", run_pair, resolve_pair_options, "pair_pooled"),
    ):
        runs[setup] = {}
        for task in ("regression", "classification"):
            out = BENCHMARK_ROOT / subpath / task
            overrides = {
                "task": task,
                "overwrite": True,
                "output_dir": out,
                "models": graph_models,
                "epochs": epochs,
            }
            if setup == "country_month":
                overrides["baselines"] = None  # all baselines including lightgbm
            options = resolve(config, overrides)
            print(f"\n=== {setup} / {task} -> {out} ===", flush=True)
            summary = runner(config, options)
            runs[setup][task] = Path(summary["output_dir"])

    master = build_master_table(runs)
    plots = write_benchmark_plots(master, BENCHMARK_ROOT / "plots")
    report = write_benchmark_report(master, BENCHMARK_ROOT, plots)
    payload = {
        "runs": {setup: {task: str(path) for task, path in tasks.items()} for setup, tasks in runs.items()},
        "report": str(report),
        "plots": plots,
    }
    (BENCHMARK_ROOT / "benchmark_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
