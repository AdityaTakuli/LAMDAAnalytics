#!/usr/bin/env python3
"""Paper-21 revision: graph models on the non-overlapping target and on rolling origins.

Uses the paper-21 trainer unchanged (training.pipeline.run, 40 epochs,
patience 10, CPU) and swaps only the regression target or the split:

  t1        target = log(V_{T+1}/V_T) instead of the rolling-median
            contraction; default split (train 2021-12..2022-12, validation
            2023, test 2024); seeds 7, 13, 21, 42, 101.
  rolling   default target and T1; four six-month test blocks
            (2023H1, 2023H2, 2024H1, 2024H2). For each block the three
            months before it are the early-stopping window and everything
            from 2021-12 before that is training; seeds 7, 13, 21.

The target swap is a wrapper around training.data.build_targets that
overwrites the ``contraction`` column and ``target_valid`` mask after the
documented construction, so every model (baselines included) sees the same
rows. Writes under results/paper21_revision/graph_{t1,rolling}/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import load_config  # noqa: E402
from training import data as data_module  # noqa: E402
from training import runtime  # noqa: E402
from training.pipeline import resolve_options, run  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "results" / "paper21_revision"
_ORIGINAL = data_module.build_targets


def t1_targets(nodes, horizon=1, baseline_window=12, baseline_min_periods=1, taus=(0.3,)):
    frame, stats = _ORIGINAL(nodes, horizon, baseline_window, baseline_min_periods, taus)
    v = frame["inbound_flow_usd"].astype(float)
    fut = frame["future_inbound_flow_usd"].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.log(fut / v.where(v > 0))
    t1[(fut <= 0) | fut.isna()] = np.nan
    frame["contraction"] = t1
    # same supervised months as the paper (12-month history), finite log change
    frame["target_valid"] = frame["target_valid"] & np.isfinite(t1)
    stats["target_override"] = "T1 = log(V_{T+1}/V_T)"
    return frame, stats


BLOCKS = {
    "2023H1": (["2021-01", "2022-09"], ["2022-10", "2022-12"], ["2023-01", "2023-06"]),
    "2023H2": (["2021-01", "2023-03"], ["2023-04", "2023-06"], ["2023-07", "2023-12"]),
    "2024H1": (["2021-01", "2023-09"], ["2023-10", "2023-12"], ["2024-01", "2024-06"]),
    "2024H2": (["2021-01", "2024-03"], ["2024-04", "2024-06"], ["2024-07", "2024-12"]),
}


def one(config, out, seed, split=None):
    if (out / "predictions" / "tgn_no_memory.csv").exists():
        print(f"=== skip (done) {out}", flush=True)
        return
    overrides = {"task": "regression", "overwrite": True, "output_dir": out,
                 "models": ["gcn", "tgn", "tgn_no_memory"], "epochs": 40, "seed": seed,
                 "baselines": ["train_median", "ridge_regression"], "device": "cpu"}
    if split:
        overrides["split_overrides"] = {"train": split[0], "validation": split[1], "test": split[2]}
    print(f"=== {out}", flush=True)
    run(config, resolve_options(config, overrides))


def main() -> int:
    runtime.configure_logging()
    config = load_config("config.yaml")
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("t1", "all"):
        data_module.build_targets = t1_targets
        for seed in (7, 13, 21, 42, 101):
            one(config, ROOT / "graph_t1" / f"seed_{seed}", seed)
        data_module.build_targets = _ORIGINAL
    if mode in ("rolling", "all", "rolling_T0", "rolling_T1"):
        targets = {"rolling_T0": ("T0",), "rolling_T1": ("T1",)}.get(mode, ("T0", "T1"))
        for target in targets:
            data_module.build_targets = t1_targets if target == "T1" else _ORIGINAL
            for block, split in BLOCKS.items():
                for seed in (7, 13, 21):
                    one(config, ROOT / "graph_rolling" / target / block / f"seed_{seed}", seed, split)
        data_module.build_targets = _ORIGINAL
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
