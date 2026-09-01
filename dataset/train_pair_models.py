#!/usr/bin/env python3
"""Train pooled bilateral pair-month supply-chain models.

Examples
--------
Classification on all observed directed trade links::

    python train_pair_models.py --config config.yaml --task classification

Regression on bilateral flow contraction::

    python train_pair_models.py --config config.yaml --task regression

Both tasks::

    python train_pair_models.py --config config.yaml --task both
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import load_config  # noqa: E402
from training import runtime  # noqa: E402
from training.pipeline import TrainingGateError  # noqa: E402
from training.models import REGRESSION, TASKS  # noqa: E402
from training.pair_pipeline import PairOptions, resolve_pair_options, run  # noqa: E402

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_DATA = 2
EXIT_DEVICE = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--task", choices=[*TASKS, "both"], default="both")
    parser.add_argument("--tau", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Graph models to train (default: gcn tgn tgn_no_memory)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    tasks = list(TASKS) if args.task == "both" else [args.task]
    summaries = []

    for task in tasks:
        overrides = {
            "task": task,
            "tau": args.tau,
            "epochs": args.epochs,
            "device": args.device,
            "overwrite": args.overwrite,
            "dry_run": args.dry_run,
            "models": args.models,
        }
        if args.output_dir:
            overrides["output_dir"] = args.output_dir / task if args.task == "both" else args.output_dir
        options = resolve_pair_options(config, overrides)
        try:
            runtime.configure_logging()
            summary = run(config, options)
            summaries.append(summary)
        except TrainingGateError as exc:
            print(f"DATA GATE: {exc}", file=sys.stderr)
            return EXIT_DATA
        except FileNotFoundError as exc:
            print(exc, file=sys.stderr)
            return EXIT_ERROR
        except runtime.EnvironmentError_ as exc:
            print(exc, file=sys.stderr)
            return EXIT_DEVICE
        except Exception:
            traceback.print_exc()
            return EXIT_ERROR

    if not args.dry_run and summaries:
        print(json.dumps({"tasks": tasks, "runs": [item.get("output_dir") for item in summaries]}, indent=2))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
