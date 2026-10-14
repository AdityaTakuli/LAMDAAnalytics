#!/usr/bin/env python3
"""Consumer basket: the benchmark trainer (unchanged) on the consumer-basket panel.

Same protocol as benchmark_rerun.py / graph_logchange_rolling.py: 40 epochs,
patience 10, CPU, seeds 7/13/21/42/101 (3 seeds for rolling origins). Only the
input tables change, to processed_v2/consumer/{nodes,edges}_monthly.csv.

Modes
  default   T0 target, regression + classification, 5 seeds
  t1        T1 target (log change), regression, 5 seeds
  rolling   T0 and T1, four six-month blocks, 3 seeds
  tune      T1, one hyperparameter variant named by LAMDA_TUNE (see TUNE), seeds from LAMDA_SEEDS
Completed runs are skipped, so the script can be rerun after an interruption.
Outputs under results/benchmark/consumer/{benchmark_default,graph_t1,graph_rolling}/.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import load_config  # noqa: E402
from training import data as data_module  # noqa: E402
from training import runtime  # noqa: E402
from training.pipeline import resolve_options, run  # noqa: E402
import graph_logchange_rolling as gx  # noqa: E402

import os

# defaults: consumer-basket panel. Extended panel, 2021-2024: LAMDA_PANEL=processed_v3 LAMDA_BASKET=semiconductor|consumer
# LAMDA_OUT=extended_2021_2024/<basket>
_PANEL = os.environ.get("LAMDA_PANEL", "processed_v2")
_BASKET = os.environ.get("LAMDA_BASKET", "consumer")
ROOT = Path(__file__).resolve().parents[1] / "results" / "benchmark" / os.environ.get("LAMDA_OUT", "consumer")
V2 = f"data/four_year_2021_2024/{_PANEL}/{_BASKET}"
# LAMDA_SEEDS=7,13 restricts the seeds of this process (one process per seed runs them in parallel).
# LAMDA_SPLIT=2019-01:2022-12,2023-01:2023-12,2024-01:2025-12 replaces the config split (seven-year panel).
SEEDS = tuple(int(s) for s in os.environ.get("LAMDA_SEEDS", "7,13,21,42,101").split(","))
SPLIT = [part.split(":") for part in os.environ["LAMDA_SPLIT"].split(",")] if os.environ.get("LAMDA_SPLIT") else None


def consumer_config() -> dict:
    config = copy.deepcopy(load_config("config.yaml"))
    config["outputs"]["nodes"] = f"{V2}/nodes_monthly.csv"
    config["outputs"]["edges"] = f"{V2}/edges_monthly.csv"
    return config


# Hyperparameter variants for the sensitivity check. "base" is the paper's setting.
# keys under "model" replace config["model"] entries; the rest are trainer overrides.
TUNE = {
    "base": {},
    "lr_low": {"learning_rate": 0.001},
    "lr_high": {"learning_rate": 0.01},
    "weight_decay": {"weight_decay": 0.001},
    "small": {"model": {"embedding_dim": 16, "memory_dim": 16, "message_dim": 32}},
    "large": {"model": {"embedding_dim": 64, "memory_dim": 64, "message_dim": 128}},
    "mse_loss": {"regression_loss": "mse"},
    "long": {"epochs": 100, "patience": 20},
}


def one(config, out: Path, seed: int, task: str = "regression", split=None, baselines=None, extra=None) -> None:
    if (out / "predictions" / "tgn_no_memory.csv").exists():
        print(f"=== skip (done) {out}", flush=True)
        return
    overrides = {"task": task, "overwrite": True, "output_dir": out, "models": ["gcn", "tgn", "tgn_no_memory"],
                 "epochs": 40, "seed": seed, "baselines": baselines, "device": "cpu"}
    if split:
        overrides["split_overrides"] = {"train": split[0], "validation": split[1], "test": split[2]}
    overrides.update(extra or {})
    print(f"=== {out}", flush=True)
    run(config, resolve_options(config, overrides))


def main() -> int:
    runtime.configure_logging()
    config = consumer_config()
    mode = sys.argv[1] if len(sys.argv) > 1 else "default"
    if mode == "default":
        for seed in SEEDS:
            for task in ("regression", "classification"):
                one(config, ROOT / "benchmark_default" / f"seed_{seed}" / task, seed, task, split=SPLIT)
    elif mode == "t1":
        data_module.build_targets = gx.t1_targets
        for seed in SEEDS:
            one(config, ROOT / "graph_t1" / f"seed_{seed}", seed, split=SPLIT, baselines=["train_median", "ridge_regression"])
    elif mode == "tune":
        name = os.environ["LAMDA_TUNE"]
        extra = dict(TUNE[name])
        config["model"] = {**config.get("model", {}), **extra.pop("model", {})}
        data_module.build_targets = gx.t1_targets
        for seed in SEEDS:
            one(config, ROOT / "graph_tune" / name / f"seed_{seed}", seed, split=SPLIT,
                baselines=["train_median", "ridge_regression"], extra=extra)
    elif mode in ("rolling_T0", "rolling_T1"):
        target = mode[-2:]
        data_module.build_targets = gx.t1_targets if target == "T1" else gx._ORIGINAL
        for block, split in gx.BLOCKS.items():
            for seed in (7, 13, 21):
                one(config, ROOT / "graph_rolling" / target / block / f"seed_{seed}", seed, split=split,
                    baselines=["train_median", "ridge_regression"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
