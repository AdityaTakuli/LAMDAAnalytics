#!/usr/bin/env python3
"""Paper-21 revision: regenerate the 5-seed country-month benchmark predictions.

The per-seed prediction CSVs behind Tables XI-XIII of paper2.tex were not kept
on disk (only checkpoints and logs). This re-runs exactly the benchmark_rerun2
protocol (config.yaml, 40 epochs, patience 10, seeds 7/13/21/42/101, both
tasks) so that paired significance tests can be computed on the same rows.

Optional argv[1]: "refit" trains on train+validation (2021-12..2023-12) with a
2-month carved validation tail used only for early stopping. That is the
refit-on-train+val protocol a reviewer asked for.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import load_config  # noqa: E402
from training import runtime  # noqa: E402
from training.pipeline import resolve_options, run  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "results" / "paper21_revision"
SEEDS = (7, 13, 21, 42, 101)


def main() -> int:
    runtime.configure_logging()
    mode = sys.argv[1] if len(sys.argv) > 1 else "default"
    config = load_config("config.yaml")
    for seed in SEEDS:
        for task in ("regression", "classification"):
            out = ROOT / f"benchmark_{mode}" / f"seed_{seed}" / task
            overrides = {
                "task": task,
                "overwrite": True,
                "output_dir": out,
                "models": ["gcn", "tgn", "tgn_no_memory"],
                "epochs": 40,
                "seed": seed,
                "baselines": None,
                "device": "cpu",
            }
            if mode == "refit":
                overrides["split_overrides"] = {
                    "train": ["2021-01", "2023-10"],
                    "validation": ["2023-11", "2023-12"],
                    "test": ["2024-01", "2024-12"],
                }
            options = resolve_options(config, overrides)
            print(f"=== {mode} seed {seed} {task} -> {out}", flush=True)
            run(config, options)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
