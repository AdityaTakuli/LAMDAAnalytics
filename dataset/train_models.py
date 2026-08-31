#!/usr/bin/env python3
"""Train and compare the country-month supply-chain risk models.

This is the entry point for model training. It reads the fused tables produced
by ``fuse_dataset.py`` / ``build_graph.py``, builds a chronological
train/validation/test split, trains the GCN, TGN, and TGN-no-memory models
alongside non-graph baselines, and writes a self-describing run directory.

It runs identically on CPU and on CUDA (Linux or Windows); see
``dataset/TRAINING.md`` for the setup commands.

Examples
--------
Verify the installation, including the GPU, without touching any data::

    python train_models.py --self-test --device cuda

Show exactly what a run would do, and write nothing::

    python train_models.py --config config.yaml --dry-run

Continuous contraction regression on the 2024 profile::

    python train_models.py --config config.yaml --task regression --device auto

Binary classification on the four-year profile::

    python train_models.py --config config_4year.yaml --task classification --device cuda

Exit codes
----------
0  the run completed
1  an unexpected error (the traceback is printed)
2  the data or the split cannot support the requested run
3  the requested compute device is unavailable
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

# Make the flat pipeline modules importable no matter where this was launched.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from training import runtime  # noqa: E402
from training.data import DataValidationError  # noqa: E402
from training.models import GRAPH_MODELS, TASKS  # noqa: E402
from training.pipeline import (  # noqa: E402
    Options,
    TrainingGateError,
    default_output_dir,
    resolve_options,
    run,
)

from common import load_config  # noqa: E402

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_DATA = 2
EXIT_DEVICE = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_models.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    group = parser.add_argument_group("inputs")
    group.add_argument("--config", default="config.yaml", help="YAML profile to train from (default: config.yaml)")
    group.add_argument(
        "--task",
        choices=[*TASKS, "both"],
        help="classification (binary contraction label), regression (continuous contraction), or both",
    )
    group.add_argument("--tau", type=float, help="Contraction threshold for the binary label (default: from config)")
    group.add_argument(
        "--baseline-min-periods",
        type=int,
        help="Minimum months required for the rolling baseline median (default: from config, usually 1)",
    )

    group = parser.add_argument_group("split")
    group.add_argument("--split-mode", choices=["date_ranges", "counts"], help="Override the configured split mode")
    group.add_argument("--train-range", help="Training months as START:END, e.g. 2021-01:2022-12")
    group.add_argument("--validation-range", help="Validation months as START:END, e.g. 2023-01:2023-12")
    group.add_argument("--test-range", help="Test months as START:END, e.g. 2024-01:2024-12")
    group.add_argument("--train-months", type=int, help="Number of training months (counts mode)")
    group.add_argument("--validation-months", type=int, help="Number of validation months (counts mode)")
    group.add_argument("--test-months", type=int, help="Number of test months (counts mode)")

    group = parser.add_argument_group("models and optimisation")
    group.add_argument(
        "--models",
        help=f"Comma-separated graph models to train (default: all). Choices: {', '.join(GRAPH_MODELS)}",
    )
    group.add_argument("--baselines", help="Comma-separated baselines to run (default: all applicable)")
    group.add_argument("--epochs", type=int, help="Training epochs per model")
    group.add_argument("--learning-rate", type=float, help="Adam learning rate")
    group.add_argument("--weight-decay", type=float, help="Adam weight decay (default 0.0)")
    group.add_argument("--grad-clip", type=float, help="Gradient-norm clip (default 5.0; 0 disables)")
    group.add_argument("--patience", type=int, help="Early-stopping patience in epochs (default: disabled)")
    group.add_argument("--threshold", type=float, help="Decision threshold for classification metrics (default 0.5)")
    group.add_argument("--regression-loss", choices=["huber", "mse"], help="Regression loss (default huber)")
    group.add_argument("--huber-beta", type=float, help="Huber transition point (default 0.1)")

    group = parser.add_argument_group("runtime")
    group.add_argument(
        "--device",
        default=None,
        help="auto (default), cpu, cuda, or cuda:<index>. 'cuda' fails loudly if no GPU is usable.",
    )
    group.add_argument("--seed", type=int, help="Random seed (default: from config, usually 7)")
    group.add_argument(
        "--deterministic",
        action="store_true",
        help="Request deterministic kernels. Bitwise reproducibility is guaranteed on CPU only.",
    )
    group.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    group = parser.add_argument_group("outputs and gates")
    group.add_argument("--output-dir", help="Run directory (default: <results_dir>/model_training/<task>)")
    group.add_argument("--overwrite", action="store_true", help="Replace an existing run directory")
    group.add_argument(
        "--allow-degenerate",
        action="store_true",
        help="Continue even when the training split has a single class. Plumbing checks only; "
        "the resulting metrics are not a benchmark.",
    )
    group.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the data, resolve the split, print the plan, and write nothing",
    )
    group.add_argument("--json", action="store_true", help="Print the run summary as JSON on stdout")

    group = parser.add_argument_group("self-test")
    group.add_argument(
        "--self-test",
        action="store_true",
        help="Run the whole stack on synthetic data to verify the install and the device",
    )
    group.add_argument("--self-test-epochs", type=int, default=5, help="Epochs for the self-test (default 5)")

    return parser


def _range(value: str | None) -> list[str] | None:
    """Parse ``START:END`` into the two month bounds the split resolver wants."""
    if not value:
        return None
    if ":" not in value:
        raise ValueError(f"Expected a range as START:END (for example 2021-01:2022-12), received {value!r}")
    start, end = value.split(":", 1)
    return [start.strip(), end.strip()]


def _csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _overrides(args: argparse.Namespace) -> dict[str, object]:
    split_overrides: dict[str, object] = {}
    if args.split_mode:
        split_overrides["mode"] = args.split_mode
    for name, value in (
        ("train", _range(args.train_range)),
        ("validation", _range(args.validation_range)),
        ("test", _range(args.test_range)),
    ):
        if value:
            split_overrides[name] = value
            split_overrides.setdefault("mode", "date_ranges")
    for name, value in (
        ("train_months", args.train_months),
        ("validation_months", args.validation_months),
        ("test_months", args.test_months),
    ):
        if value is not None:
            split_overrides[name] = value
            split_overrides.setdefault("mode", "counts")

    return {
        "task": args.task if args.task != "both" else None,
        "tau": args.tau,
        "baseline_min_periods": args.baseline_min_periods,
        "models": _csv(args.models),
        "baselines": _csv(args.baselines),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "patience": args.patience,
        "threshold": args.threshold,
        "regression_loss": args.regression_loss,
        "huber_beta": args.huber_beta,
        "seed": args.seed,
        "device": args.device,
        "deterministic": args.deterministic or None,
        "allow_degenerate": args.allow_degenerate or None,
        "overwrite": args.overwrite or None,
        "output_dir": args.output_dir,
        "log_level": args.log_level,
        "split_overrides": split_overrides or None,
    }


def _run_self_test(args: argparse.Namespace) -> int:
    from training.selftest import run_self_test

    logger = runtime.configure_logging(args.log_level)
    logger.info("Running the training self-test on synthetic data")
    summary = run_self_test(
        device=args.device or "auto",
        output_dir=Path(args.output_dir) if args.output_dir else None,
        epochs=args.self_test_epochs,
        task=args.task if args.task in TASKS else "classification",
        seed=args.seed or 7,
    )
    passed = bool(summary.get("self_test_passed"))
    failed = [check for check in summary.get("self_test_checks", []) if not check["passed"]]
    if args.json:
        print(json.dumps({"passed": passed, "checks": summary.get("self_test_checks", [])}, indent=2))
    logger.info("-" * 78)
    if passed:
        logger.info("SELF-TEST PASSED on device %s", summary.get("device"))
        return EXIT_OK
    logger.error("SELF-TEST FAILED (%d check(s)): %s", len(failed), [check["check"] for check in failed])
    return EXIT_ERROR


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.self_test:
        return _run_self_test(args)

    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Config not found: {args.config}", file=sys.stderr)
        return EXIT_DATA

    tasks = ["classification", "regression"] if args.task == "both" else None

    try:
        overrides = _overrides(args)
        summaries = []
        for task in tasks or [None]:
            task_overrides = dict(overrides)
            if task is not None:
                task_overrides["task"] = task
            options: Options = resolve_options(config, task_overrides)
            if options.output_dir is None:
                options.output_dir = default_output_dir(config, options.task)
            elif tasks:
                options.output_dir = Path(options.output_dir) / options.task
            summaries.append(run(config, options, dry_run=args.dry_run))
        if args.json:
            print(json.dumps(summaries if len(summaries) > 1 else summaries[0], indent=2, default=str))
        return EXIT_OK
    except runtime.EnvironmentError_ as error:
        print(f"\nDevice error: {error}", file=sys.stderr)
        return EXIT_DEVICE
    except (DataValidationError, TrainingGateError) as error:
        print(f"\nCannot run this training job:\n\n{error}\n", file=sys.stderr)
        return EXIT_DATA
    except ValueError as error:
        print(f"\nInvalid option: {error}\n", file=sys.stderr)
        return EXIT_DATA
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return EXIT_ERROR
    except Exception:  # pragma: no cover - unexpected failures must stay visible
        traceback.print_exc()
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
