"""End-to-end training pipeline for the country-month models.

``train_models.py`` is a thin CLI over :func:`run`. Keeping the orchestration
here means the same code path is exercised by the CLI, by the self-test, and
by anything that later wants to import it.
"""

from __future__ import annotations

import logging
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from training import baselines as baseline_module
from training import data as data_module
from training import engine
from training import models as model_factory
from training import paths, report, runtime

from common import nested  # noqa: E402  (requires the paths bootstrap)

LOGGER = logging.getLogger("training.pipeline")

DEFAULT_MODELS = ("gcn", "tgn", "tgn_no_memory")


class TrainingGateError(RuntimeError):
    """Raised when the data cannot support the requested task."""


@dataclass
class Options:
    """Every setting a run needs, already resolved from config plus CLI."""

    task: str = model_factory.CLASSIFICATION
    tau: float = 0.35
    horizon: int = 1
    baseline_window: int = 12
    baseline_min_periods: int = 1
    taus: list[float] = field(default_factory=lambda: [0.30, 0.35, 0.40])
    models: list[str] = field(default_factory=lambda: list(DEFAULT_MODELS))
    baselines: list[str] | None = None
    split_settings: dict[str, Any] = field(default_factory=dict)
    split_overrides: dict[str, Any] = field(default_factory=dict)
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 5.0
    patience: int | None = None
    threshold: float = 0.5
    regression_loss: str = "huber"
    huber_beta: float = 0.1
    seed: int = 7
    device: str = "auto"
    deterministic: bool = False
    allow_degenerate: bool = False
    overwrite: bool = False
    output_dir: Path | None = None
    model_config: dict[str, Any] = field(default_factory=dict)
    log_level: str = "INFO"

    def to_manifest(self) -> dict[str, Any]:
        payload = {key: value for key, value in self.__dict__.items()}
        payload["output_dir"] = str(self.output_dir) if self.output_dir else None
        return payload


def _block(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    return dict(value) if isinstance(value, Mapping) else {}


def resolve_options(config: Mapping[str, Any], overrides: Mapping[str, Any] | None = None) -> Options:
    """Merge config defaults with CLI overrides. CLI always wins."""
    overrides = {key: value for key, value in (overrides or {}).items() if value is not None}
    settings = _block(config, "model_training")
    training = _block(config, "training")
    analysis = _block(config, "analysis")

    taus = [float(value) for value in analysis.get("taus", [0.20, 0.30, 0.35, 0.40])]
    task = str(overrides.get("task", settings.get("task", model_factory.CLASSIFICATION))).lower()
    default_tau = float(
        analysis.get("classification_tau", analysis.get("default_tau", 0.35))
        if task == model_factory.CLASSIFICATION
        else analysis.get("default_tau", 0.35)
    )
    tau = float(overrides.get("tau", settings.get("tau", default_tau)))
    if tau not in taus:
        taus = sorted({*taus, tau})

    split_settings = dict(settings.get("split") or {})
    if not split_settings:
        experiment = _block(config, "country_month_experiment")
        if experiment:
            split_settings = {
                "mode": "counts",
                "train_months": experiment.get("train_months", 7),
                "validation_months": experiment.get("validation_months", 2),
                "test_months": experiment.get("test_months", 2),
            }
        else:
            split_settings = {
                "mode": "counts",
                "train_months": training.get("train_months", 6),
                "validation_months": training.get("validation_months", 3),
                "test_months": training.get("test_months", 3),
            }

    requested_models = overrides.get("models") or settings.get("models") or list(DEFAULT_MODELS)
    unknown = [name for name in requested_models if name not in model_factory.GRAPH_MODELS]
    if unknown:
        raise ValueError(
            f"Unknown model name(s) {unknown}; valid graph models are {list(model_factory.GRAPH_MODELS)}"
        )

    if task not in model_factory.TASKS:
        raise ValueError(f"Unknown task {task!r}; expected one of {model_factory.TASKS}")

    options = Options(
        task=task,
        tau=tau,
        taus=taus,
        horizon=int(overrides.get("horizon", analysis.get("horizon_months", 1))),
        baseline_window=int(settings.get("baseline_window", 12)),
        baseline_min_periods=int(
            overrides.get("baseline_min_periods", settings.get("baseline_min_periods", 1))
        ),
        models=list(requested_models),
        baselines=overrides.get("baselines") or settings.get("baselines"),
        split_settings=split_settings,
        split_overrides=dict(overrides.get("split_overrides") or {}),
        epochs=int(overrides.get("epochs", settings.get("epochs", training.get("epochs", 10)))),
        learning_rate=float(
            overrides.get("learning_rate", settings.get("learning_rate", training.get("learning_rate", 1e-3)))
        ),
        weight_decay=float(overrides.get("weight_decay", settings.get("weight_decay", 0.0))),
        grad_clip=float(overrides.get("grad_clip", settings.get("grad_clip", 5.0))),
        patience=overrides.get("patience", settings.get("patience")),
        threshold=float(overrides.get("threshold", settings.get("decision_threshold", 0.5))),
        regression_loss=str(overrides.get("regression_loss", settings.get("regression_loss", "huber"))).lower(),
        huber_beta=float(overrides.get("huber_beta", settings.get("huber_beta", 0.1))),
        seed=int(overrides.get("seed", settings.get("seed", training.get("seed", 7)))),
        device=str(overrides.get("device", settings.get("device", "auto"))),
        deterministic=bool(overrides.get("deterministic", settings.get("deterministic", False))),
        allow_degenerate=bool(overrides.get("allow_degenerate", False)),
        overwrite=bool(overrides.get("overwrite", False)),
        output_dir=Path(overrides["output_dir"]) if overrides.get("output_dir") else None,
        model_config=_block(config, "model"),
        log_level=str(overrides.get("log_level", "INFO")),
    )
    if options.patience is not None:
        options.patience = int(options.patience)
        if options.patience < 1:
            raise ValueError(f"--patience must be >= 1, received {options.patience}")
    if options.epochs < 1:
        raise ValueError(f"--epochs must be >= 1, received {options.epochs}")
    if options.regression_loss not in {"huber", "mse"}:
        raise ValueError(f"--regression-loss must be 'huber' or 'mse', received {options.regression_loss!r}")
    return options


def default_output_dir(config: Mapping[str, Any], task: str) -> Path:
    configured = nested(config, "outputs", "model_training_dir")
    if configured:
        return paths.resolve(configured) / task
    results = nested(config, "outputs", "results_dir")
    if results:
        return paths.resolve(results) / "model_training" / task
    metrics = nested(config, "outputs", "training_metrics", default="processed/training_metrics.json")
    return paths.resolve(metrics).parent / "model_training" / task


def _prepare_output_dir(directory: Path, overwrite: bool) -> Path:
    if directory.exists() and any(directory.iterdir()):
        if not overwrite:
            raise TrainingGateError(
                f"Output directory already contains artifacts: {directory}\n"
                "Re-run with --overwrite to replace it, or pass --output-dir <other path>. "
                "Refusing to silently mix results from two runs."
            )
        import shutil

        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _gate_classification(class_balance: Mapping[str, Any], tau: float, options: Options) -> list[str]:
    """Refuse to report a benchmark that a single-class split cannot support."""
    warnings: list[str] = []
    train = class_balance["train"]
    if train["valid_rows"] == 0:
        raise TrainingGateError(
            "The training partition has no country-month with an observable target. "
            "Check diagnostics/target_summary.json and widen the training range."
        )
    if train["single_class"]:
        message = (
            f"The training partition contains a single class at tau={tau:.2f}: "
            f"{train['positive']} positive / {train['negative']} negative over {train['valid_rows']} valid rows.\n"
            "A binary classifier fitted on one class learns nothing, and every discrimination metric "
            "would be undefined. This is a property of the data, not a bug.\n"
            "Choose one of:\n"
            "  1. Run the continuous target instead (recommended by dataset/README.md):\n"
            "       python train_models.py --config <config> --task regression\n"
            "  2. Train on a wider window that contains positives (the four-year profile).\n"
            "  3. Proceed anyway, for a plumbing check only, with --allow-degenerate.\n"
            "Do not lower tau after seeing model results, and do not fabricate or rebalance labels."
        )
        if not options.allow_degenerate:
            raise TrainingGateError(message)
        warnings.append("DEGENERATE RUN: " + message.splitlines()[0])
    for name in ("validation", "test"):
        partition = class_balance[name]
        if partition["valid_rows"] == 0:
            warnings.append(f"The {name} partition has no valid target rows; its metrics will be N/A.")
        elif partition["single_class"]:
            warnings.append(
                f"The {name} partition contains a single class "
                f"({partition['positive']} positive / {partition['negative']} negative); "
                "ROC-AUC, PR-AUC, precision, recall, and F1 are undefined there."
            )
    return warnings


def run(
    config: Mapping[str, Any],
    options: Options,
    tables: tuple[pd.DataFrame, pd.DataFrame] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute one task end to end and return the run summary."""
    started = time.perf_counter()
    output_dir = options.output_dir or default_output_dir(config, options.task)
    output_dir = Path(output_dir)

    choice = runtime.resolve_device(options.device)
    device = choice.device
    determinism = runtime.set_determinism(options.seed, strict=options.deterministic)

    if dry_run:
        runtime.configure_logging(options.log_level)
    else:
        _prepare_output_dir(output_dir, options.overwrite)
        runtime.configure_logging(options.log_level, output_dir / "training.log")
    LOGGER.info("=" * 78)
    LOGGER.info("Task            : %s", options.task)
    LOGGER.info("Config          : %s", config.get("_config_path", "<in-memory>"))
    LOGGER.info("Output          : %s", output_dir if not dry_run else "<dry run: nothing is written>")
    environment = runtime.environment_report(choice)
    for line in runtime.describe_environment(environment).splitlines():
        LOGGER.info(line)
    LOGGER.info("=" * 78)

    # ---------------------------------------------------------------- data --
    nodes, edges = tables if tables is not None else data_module.load_tables(config)
    nodes = nodes.copy()
    edges = edges.copy() if edges is not None else pd.DataFrame()
    validation_report = data_module.validate_tables(nodes, edges)
    LOGGER.info(
        "Loaded %d country-month rows, %d edge rows, %d months, %d countries",
        validation_report["node_rows"],
        validation_report["edge_rows"],
        validation_report["month_count"],
        validation_report["node_count"],
    )

    frame, target_stats = data_module.build_targets(
        nodes,
        horizon=options.horizon,
        baseline_window=options.baseline_window,
        baseline_min_periods=options.baseline_min_periods,
        taus=options.taus,
    )
    label_check = data_module.cross_check_labels(frame, options.taus)
    LOGGER.info(
        "Targets: %d valid of %d country-months (horizon=%d month)",
        target_stats["valid_targets"], target_stats["country_month_rows"], options.horizon,
    )
    if target_stats.get("baseline_caveat"):
        LOGGER.warning(target_stats["baseline_caveat"])

    months = validation_report["months"]
    supervised = data_module.supervised_months(frame, months)
    if len(supervised) < 3:
        raise TrainingGateError(
            f"Only {len(supervised)} month(s) have an observable target; a chronological "
            "train/validation/test split needs at least three. Extend the data window."
        )
    split = data_module.resolve_split(months, supervised, options.split_settings, options.split_overrides)
    for note in split.notes:
        LOGGER.warning(note)
    LOGGER.info(
        "Split: train %s..%s (%d) | validation %s..%s (%d) | test %s..%s (%d)",
        split.train[0], split.train[-1], len(split.train),
        split.validation[0], split.validation[-1], len(split.validation),
        split.test[0], split.test[-1], len(split.test),
    )

    if options.task == model_factory.CLASSIFICATION:
        target_column = f"derived_label_tau_{options.tau:.2f}"
        class_balance = data_module.class_summary(frame, split, target_column)
        gate_warnings = _gate_classification(class_balance, options.tau, options)
        target_summary: dict[str, Any] = {"class_balance": class_balance}
        weight = data_module.positive_weight(frame, split.train, target_column)
    else:
        target_column = "contraction"
        gate_warnings = []
        target_summary = {"contraction_by_split": data_module.regression_summary(frame, split)}
        weight = 1.0
        for name, stats in target_summary["contraction_by_split"].items():
            if stats["valid_rows"] == 0:
                raise TrainingGateError(
                    f"The {name} partition has no valid contraction target; cannot run a regression."
                )
    for warning in gate_warnings:
        LOGGER.warning(warning)

    node_ids = sorted(frame["node_id"].astype(str).unique().tolist())
    train_rows = frame[frame["month"].isin(split.train)]
    standardizer = data_module.FeatureStandardizer().fit(train_rows)
    edge_scale = data_module.edge_scales(edges, split.train)
    LOGGER.info(
        "Fitted on training months only: standardizer (%d features), edge scales (%.4f, %.4f), positive weight %.3f",
        len(data_module.FEATURES), edge_scale[0], edge_scale[1], weight,
    )

    plan = {
        "task": options.task,
        "target_column": target_column,
        "tau": options.tau if options.task == model_factory.CLASSIFICATION else None,
        "models": list(options.models),
        "baselines": list(
            options.baselines
            or (
                baseline_module.CLASSIFICATION_BASELINES
                if options.task == "classification"
                else baseline_module.REGRESSION_BASELINES
            )
        ),
        "epochs": options.epochs,
        "device": str(device),
        "split": split.to_manifest(),
        "output_dir": str(output_dir),
    }
    if dry_run:
        LOGGER.info("Dry run complete; no model was trained and nothing was written.")
        return {
            "status": "dry_run",
            "plan": plan,
            "environment": environment,
            "data_validation": validation_report,
            "target_summary": {**target_stats, **target_summary},
            "label_cross_check": label_check,
        }

    ordered_months = split.all_months()
    batches = data_module.build_month_batches(
        frame, edges, ordered_months, node_ids, standardizer, edge_scale, target_column, device
    )
    LOGGER.info(
        "Materialised %d monthly snapshots on %s (%d nodes, %d supervised rows)",
        len(batches), device, len(node_ids), sum(batch.valid_count for batch in batches.values()),
    )

    # ----------------------------------------------------------- baselines --
    predictions: dict[str, pd.DataFrame] = {}
    metrics: dict[str, dict[str, Any]] = {}
    histories: dict[str, list[dict[str, Any]]] = {}
    model_manifests: dict[str, Any] = {}

    baseline_predictions, baseline_notes = baseline_module.run_baselines(
        options.task, frame, split, standardizer, target_column, seed=options.seed, selected=options.baselines
    )
    for name, prediction_frame in baseline_predictions.items():
        predictions[name] = prediction_frame
        metrics[name] = report.metrics_by_split(options.task, prediction_frame, options.threshold)
        report.write_predictions(prediction_frame, output_dir / "predictions" / f"{name}.csv")
        LOGGER.info("Baseline %-22s scored %d row(s)", name, len(prediction_frame))

    # -------------------------------------------------------- graph models --
    for model_name in options.models:
        LOGGER.info("-" * 78)
        LOGGER.info("Training %s (%s) on %s", model_name, options.task, device)
        model, model_kwargs = model_factory.build_model(
            model_name, options.task, options.model_config, len(data_module.FEATURES), device
        )
        result = engine.train_model(
            model,
            model_name,
            options.task,
            batches,
            split.train,
            split.validation,
            node_count=len(node_ids),
            device=device,
            epochs=options.epochs,
            learning_rate=options.learning_rate,
            weight_decay=options.weight_decay,
            grad_clip=options.grad_clip,
            patience=options.patience,
            positive_weight=weight,
            regression_loss=options.regression_loss,
            huber_beta=options.huber_beta,
            threshold=options.threshold,
        )
        scores = engine.score_all_months(
            model, model_name, batches, ordered_months, len(node_ids), device
        )
        prediction_frame = report.graph_scores_to_frame(batches, split, scores, node_ids, model_name)
        predictions[model_name] = prediction_frame
        metrics[model_name] = report.metrics_by_split(options.task, prediction_frame, options.threshold)
        histories[model_name] = result.history()
        model_manifests[model_name] = result.to_manifest()
        report.write_predictions(prediction_frame, output_dir / "predictions" / f"{model_name}.csv")
        report.save_checkpoint(
            model,
            output_dir / "checkpoints" / f"{model_name}.pt",
            model_name=model_name,
            task=options.task,
            model_kwargs=model_kwargs,
            features=data_module.FEATURES,
            normalizer=standardizer.to_dict(),
            edge_scale=edge_scale,
            node_order=node_ids,
            split=split,
            target_column=target_column,
            tau=options.tau if options.task == model_factory.CLASSIFICATION else None,
            training_manifest=result.to_manifest(),
        )
        LOGGER.info(
            "%s finished in %.1fs; best epoch %s; test rows %d",
            model_name, result.seconds, result.best_epoch, int(metrics[model_name]["test"].get("n", 0)),
        )

    # ------------------------------------------------------------- reports --
    report.write_comparison(options.task, metrics, output_dir / "metrics" / "comparison.csv")
    for name, split_metrics in metrics.items():
        report.write_json(split_metrics, output_dir / "metrics" / f"{name}.json")
    figures = report.write_plots(
        options.task, frame, split, histories, predictions, metrics, output_dir / "plots"
    )

    diagnostics = output_dir / "diagnostics"
    report.write_json(validation_report, diagnostics / "data_validation.json")
    report.write_json({**target_stats, **target_summary}, diagnostics / "target_summary.json")
    report.write_json(label_check, diagnostics / "label_cross_check.json")
    report.write_json(split.to_manifest(), diagnostics / "split.json")
    if options.task == model_factory.CLASSIFICATION:
        report.write_json(target_summary["class_balance"], diagnostics / "class_balance.json")
    report.write_json(baseline_notes, diagnostics / "baseline_notes.json")
    report.write_leakage_audit(
        diagnostics / "leakage_audit.md",
        task=options.task,
        split=split,
        horizon=options.horizon,
        target_stats=target_stats,
        target_column=target_column,
        node_count=len(node_ids),
        features=data_module.FEATURES,
    )

    config_copy = {key: value for key, value in config.items() if key != "_config_path"}
    (output_dir / "config_used.yaml").write_text(
        yaml.safe_dump(config_copy, sort_keys=False), encoding="utf-8"
    )

    summary = {
        "status": "ok",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task": options.task,
        "target_column": target_column,
        "tau": options.tau if options.task == model_factory.CLASSIFICATION else None,
        "config_path": str(config.get("_config_path", "<in-memory>")),
        "input_tables": {key: str(value) for key, value in data_module.table_paths(config).items()}
        if tables is None
        else {"nodes": "<in-memory>", "edges": "<in-memory>"},
        "output_dir": str(output_dir),
        "device": str(device),
        "device_reason": choice.reason,
        "determinism": determinism,
        "environment": environment,
        "cuda_memory": runtime.cuda_memory_summary(device),
        "options": options.to_manifest(),
        "split": split.to_manifest(),
        "positive_weight": weight,
        "edge_scale": {"trade_value_log1p": edge_scale[0], "flow_volume_log1p": edge_scale[1]},
        "normalizer": standardizer.to_dict(),
        "countries": len(node_ids),
        "supervised_rows": int(sum(batch.valid_count for batch in batches.values())),
        "models": model_manifests,
        "metrics": metrics,
        "warnings": gate_warnings + list(split.notes),
        "figures": figures,
        "wall_seconds": round(time.perf_counter() - started, 3),
        "host": platform.node(),
        "synthetic": tables is not None,
    }
    report.write_json(summary, output_dir / "run_summary.json")
    _write_readme(output_dir, summary)
    LOGGER.info("=" * 78)
    LOGGER.info("Run complete in %.1fs -> %s", summary["wall_seconds"], output_dir)
    _log_comparison(options.task, metrics)
    return summary


def _log_comparison(task: str, metrics: Mapping[str, Mapping[str, Any]]) -> None:
    headline = ("average_precision", "roc_auc") if task == "classification" else ("rmse", "mae")
    LOGGER.info("%-22s %-12s %6s  %s", "model", "split", "n", " ".join(f"{name:>18}" for name in headline))
    for model_name, split_metrics in metrics.items():
        for split_name in ("validation", "test"):
            values = split_metrics.get(split_name, {})
            rendered = " ".join(
                f"{values.get(name):>18.4f}" if isinstance(values.get(name), (int, float)) else f"{'N/A':>18}"
                for name in headline
            )
            LOGGER.info("%-22s %-12s %6s  %s", model_name, split_name, values.get("n", 0), rendered)


def _write_readme(output_dir: Path, summary: Mapping[str, Any]) -> None:
    task = summary["task"]
    headline = "average_precision (PR-AUC)" if task == "classification" else "rmse"
    (output_dir / "README.md").write_text(
        f"""# Run artifacts — {task}

Generated {summary['generated_at']} on `{summary['host']}` using device
`{summary['device']}` in {summary['wall_seconds']}s.

| Path | Contents |
| --- | --- |
| `run_summary.json` | Everything about this run: options, environment, split, per-model metrics. Start here. |
| `metrics/comparison.csv` | One row per (model, split). The headline column is `{headline}`. |
| `metrics/<model>.json` | Full metric block per model, including `note` for undefined metrics. |
| `predictions/<model>.csv` | Per country-month scores: `model, split, month, node_id, target, score`. |
| `checkpoints/<model>.pt` | Weights plus the kwargs, feature order, normaliser, and split needed to reload them. |
| `diagnostics/leakage_audit.md` | What was fitted on what, and why no future information reaches a feature. |
| `diagnostics/target_summary.json` | Target validity, contraction distribution, and class balance. |
| `diagnostics/label_cross_check.json` | Re-derived labels compared against the labels stored in the fused table. |
| `diagnostics/data_validation.json` | Row counts, grid completeness, per-feature ranges, orphan edges. |
| `diagnostics/split.json` | The exact months in each partition, and any month that was dropped. |
| `plots/` | Target distribution, loss curves, model comparison, prediction distributions. |
| `config_used.yaml` | The configuration this run actually saw. |
| `training.log` | The full console log. |

## Reading the numbers honestly

* A metric printed as `null` with a `note` is undefined for that partition, not zero.
* `n` is the number of country-months with an **observable** target. Rows whose
  future value or baseline is missing are excluded, never counted as negatives.
* Test metrics come from a single scoring pass with the validation-selected
  weights. They are not a tuning target.

## Warnings raised by this run

{chr(10).join(f'* {warning}' for warning in summary.get('warnings', [])) or '* none'}
""",
        encoding="utf-8",
    )
