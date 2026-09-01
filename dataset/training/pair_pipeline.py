"""End-to-end pooled bilateral pair-month training."""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import yaml

from common import nested
from training import runtime
from training.data import (
    FEATURES,
    FeatureStandardizer,
    build_targets,
    load_tables,
    resolve_split,
    validate_tables,
    edge_scales,
)
from training.pipeline import TrainingGateError
from training.models import CLASSIFICATION, REGRESSION
from training.pair_baselines import run_pair_baselines
from training.pair_data import (
    PairFeatureStandardizer,
    assign_splits,
    attach_node_features,
    build_pair_month_batches,
    build_pair_targets,
    enrich_pair_features,
    pair_class_summary,
    pair_positive_weight,
    pair_regression_summary,
    pair_supervised_months,
)
from training.pair_engine import replay_pair, train_mlp_tabular, train_pair_model
from training.pair_models import GRAPH_PAIR_MODELS, build_pair_model, parameter_count
from training.pair_report import (
    metrics_by_split,
    pair_scores_to_frame,
    write_comparison,
    write_json,
    write_pair_plots,
    write_readme,
)

LOGGER = logging.getLogger("training.pair_pipeline")

DEFAULT_GRAPH_MODELS = list(GRAPH_PAIR_MODELS)


@dataclass
class PairOptions:
    task: str = CLASSIFICATION
    tau: float = 0.20
    taus: list[float] = field(default_factory=lambda: [0.20, 0.30, 0.35, 0.40])
    horizon: int = 1
    baseline_window: int = 12
    baseline_min_periods: int = 12
    models: list[str] = field(default_factory=lambda: list(DEFAULT_GRAPH_MODELS))
    split_settings: dict[str, Any] = field(default_factory=dict)
    split_overrides: dict[str, Any] = field(default_factory=dict)
    epochs: int = 40
    learning_rate: float = 3e-3
    weight_decay: float = 0.0
    grad_clip: float = 5.0
    patience: int = 10
    threshold: float = 0.5
    regression_loss: str = "huber"
    huber_beta: float = 0.1
    seed: int = 7
    device: str = "auto"
    allow_degenerate: bool = False
    output_dir: Path | None = None
    overwrite: bool = False
    dry_run: bool = False


def resolve_pair_options(config: Mapping[str, Any], overrides: Mapping[str, Any] | None = None) -> PairOptions:
    overrides = {key: value for key, value in (overrides or {}).items() if value is not None}
    settings = dict(config.get("model_training") or {})
    pair_settings = dict(settings.get("pair") or {})
    analysis = dict(config.get("analysis") or {})
    taus = [float(value) for value in analysis.get("taus", [0.20, 0.30, 0.35, 0.40])]
    task = str(overrides.get("task", pair_settings.get("task", settings.get("task", CLASSIFICATION)))).lower()
    default_tau = float(
        analysis.get("classification_tau", analysis.get("default_tau", 0.35))
        if task == CLASSIFICATION
        else analysis.get("default_tau", 0.35)
    )
    tau = float(overrides.get("tau", pair_settings.get("tau", settings.get("tau", default_tau))))
    if tau not in taus:
        taus = sorted({*taus, tau})
    split_settings = dict(settings.get("split") or {})
    models = overrides.get("models") or pair_settings.get("models") or list(DEFAULT_GRAPH_MODELS)
    output = overrides.get("output_dir") or pair_settings.get("output_dir")
    return PairOptions(
        task=task,
        tau=tau,
        taus=taus,
        horizon=int(analysis.get("horizon_months", 1)),
        baseline_window=int(settings.get("baseline_window", 12)),
        baseline_min_periods=int(settings.get("baseline_min_periods", 12)),
        models=list(models),
        split_settings=split_settings,
        split_overrides=dict(overrides.get("split_overrides") or {}),
        epochs=int(overrides.get("epochs", settings.get("epochs", 40))),
        learning_rate=float(overrides.get("learning_rate", settings.get("learning_rate", 3e-3))),
        weight_decay=float(overrides.get("weight_decay", settings.get("weight_decay", 0.0))),
        grad_clip=float(overrides.get("grad_clip", settings.get("grad_clip", 5.0))),
        patience=int(overrides.get("patience", settings.get("patience", 10))),
        threshold=float(overrides.get("threshold", settings.get("decision_threshold", 0.5))),
        regression_loss=str(overrides.get("regression_loss", settings.get("regression_loss", "huber"))),
        huber_beta=float(overrides.get("huber_beta", settings.get("huber_beta", 0.1))),
        seed=int(overrides.get("seed", settings.get("seed", 7))),
        device=str(overrides.get("device", settings.get("device", "auto"))),
        allow_degenerate=bool(overrides.get("allow_degenerate", False)),
        output_dir=Path(output) if output else None,
        overwrite=bool(overrides.get("overwrite", False)),
        dry_run=bool(overrides.get("dry_run", False)),
    )


def default_pair_output_dir(config: Mapping[str, Any], task: str) -> Path:
    base = Path(__file__).resolve().parent.parent
    root = nested(config, "outputs", "model_training_dir", default="data/four_year_2021_2024/results/model_training")
    subdir = nested(config, "model_training", "pair", "run_name", default="pair_pooled")
    return base / root / subdir / task


def _prepare_output_dir(directory: Path, overwrite: bool) -> Path:
    if directory.exists() and any(directory.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory {directory} already exists. Re-run with --overwrite or choose another path."
            )
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def run(config: Mapping[str, Any], options: PairOptions) -> dict[str, Any]:
    started = time.perf_counter()
    choice = runtime.resolve_device(options.device)
    device = choice.device
    runtime.set_determinism(options.seed)
    runtime.configure_logging()

    nodes, edges = load_tables(config)
    validation = validate_tables(nodes, edges)
    node_frame, node_target_stats = build_targets(
        nodes,
        horizon=options.horizon,
        baseline_window=options.baseline_window,
        baseline_min_periods=options.baseline_min_periods,
        taus=options.taus,
    )
    pair_frame, pair_target_stats = build_pair_targets(
        edges,
        horizon=options.horizon,
        baseline_window=options.baseline_window,
        baseline_min_periods=options.baseline_min_periods,
        taus=options.taus,
    )
    pair_frame = attach_node_features(pair_frame, node_frame)

    months = sorted(pair_frame["month"].astype(str).unique().tolist())
    supervised = pair_supervised_months(pair_frame, months)
    if len(supervised) < 3:
        raise TrainingGateError("Fewer than three supervised months are available for pair training.")
    split = resolve_split(months, supervised, options.split_settings, options.split_overrides)

    target_column = "contraction" if options.task == REGRESSION else f"derived_label_tau_{options.tau:.2f}"
    if options.task == CLASSIFICATION:
        balance = pair_class_summary(pair_frame, split, target_column)
        train = balance["train"]
        if train["single_class"] and not options.allow_degenerate:
            raise TrainingGateError(
                f"Training partition is single-class at tau={options.tau:.2f}: "
                f"{train['positive']} positive / {train['negative']} negative."
            )
        positive_weight = pair_positive_weight(pair_frame, split.train, target_column)
        target_summary = {"class_balance": balance}
    else:
        positive_weight = 1.0
        target_summary = {"contraction_by_split": pair_regression_summary(pair_frame, split)}

    node_ids = sorted(node_frame["node_id"].astype(str).unique().tolist())
    node_standardizer = FeatureStandardizer().fit(node_frame[node_frame["month"].isin(split.train)])
    edge_scale = edge_scales(edges, split.train)
    pair_frame = enrich_pair_features(pair_frame, edge_scale)
    pair_standardizer = PairFeatureStandardizer().fit(
        pair_frame[pair_frame["month"].isin(split.train) & pair_frame["target_valid"].fillna(False)]
    )
    pair_frame = assign_splits(pair_frame, split)

    output_dir = options.output_dir or default_pair_output_dir(config, options.task)
    if options.dry_run:
        LOGGER.info("Dry run complete; pair output would be written to %s", output_dir)
        return {"dry_run": True, "output_dir": str(output_dir), "split": split.to_manifest()}

    output_dir = _prepare_output_dir(output_dir, options.overwrite)
    (output_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    diagnostics = output_dir / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    write_json(split.to_manifest(), diagnostics / "split.json")
    write_json(pair_target_stats, diagnostics / "target_summary.json")
    write_json(target_summary, diagnostics / ("class_balance.json" if options.task == CLASSIFICATION else "regression_summary.json"))
    (output_dir / "config_used.yaml").write_text(yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8")

    ordered_months = split.all_months()
    batches = build_pair_month_batches(
        pair_frame,
        node_frame,
        edges,
        ordered_months,
        node_ids,
        node_standardizer,
        edge_scale,
        target_column,
        device,
    )

    metrics: dict[str, dict[str, Any]] = {}
    predictions: dict[str, pd.DataFrame] = {}
    histories: dict[str, list[dict[str, Any]]] = {}

    baseline_preds, baseline_notes = run_pair_baselines(
        options.task,
        pair_frame,
        split,
        pair_standardizer,
        target_column,
        seed=options.seed,
    )
    for name, frame in baseline_preds.items():
        predictions[name] = frame
        metrics[name] = metrics_by_split(options.task, frame, options.threshold)
        frame.to_csv(output_dir / "predictions" / f"{name}.csv", index=False)

    model_config = dict(config.get("model") or {})
    pair_feature_dim = len(pair_standardizer.features)

    # Tabular MLP on concatenated pair features.
    mlp_model, mlp_kwargs = build_pair_model("mlp", options.task, model_config, len(FEATURES), pair_feature_dim, device)
    mlp_result = train_mlp_tabular(
        mlp_model,
        pair_frame,
        split,
        pair_standardizer,
        target_column,
        options.task,
        device,
        epochs=options.epochs,
        learning_rate=options.learning_rate,
        positive_weight=positive_weight,
        regression_loss=options.regression_loss,
        huber_beta=options.huber_beta,
        threshold=options.threshold,
        seed=options.seed,
    )
    predictions["mlp"] = mlp_result["predictions"]
    metrics["mlp"] = metrics_by_split(options.task, predictions["mlp"], options.threshold)
    predictions["mlp"].to_csv(output_dir / "predictions" / "mlp.csv", index=False)
    histories["mlp"] = mlp_result["history"]
    torch.save(
        {
            "format": "lamda-pair-checkpoint-v1",
            "model_name": "mlp",
            "task": options.task,
            "model_state_dict": mlp_model.state_dict(),
            "model_kwargs": mlp_kwargs,
            "pair_normalizer": pair_standardizer.to_dict(),
        },
        output_dir / "checkpoints" / "mlp.pt",
    )

    for model_name in options.models:
        if model_name not in GRAPH_PAIR_MODELS:
            LOGGER.warning("Skipping unknown graph model %s", model_name)
            continue
        LOGGER.info("Training %s (%s, pair supervision) on %s", model_name, options.task, device)
        model, model_kwargs = build_pair_model(
            model_name, options.task, model_config, len(FEATURES), pair_feature_dim, device
        )
        result = train_pair_model(
            model,
            model_name,
            options.task,
            batches,
            split.train,
            split.validation,
            len(node_ids),
            device,
            epochs=options.epochs,
            learning_rate=options.learning_rate,
            weight_decay=options.weight_decay,
            grad_clip=options.grad_clip,
            patience=options.patience,
            positive_weight=positive_weight,
            regression_loss=options.regression_loss,
            huber_beta=options.huber_beta,
            threshold=options.threshold,
        )
        histories[model_name] = result.history()
        torch.save(
            {
                "format": "lamda-pair-checkpoint-v1",
                "model_name": model_name,
                "task": options.task,
                "model_state_dict": model.state_dict(),
                "model_kwargs": model_kwargs,
                "node_normalizer": node_standardizer.to_dict(),
                "edge_scale": {"trade_value_log1p": edge_scale[0], "flow_volume_log1p": edge_scale[1]},
                "node_order": node_ids,
                "training": result.to_manifest(),
            },
            output_dir / "checkpoints" / f"{model_name}.pt",
        )
        train_scores = replay_pair(model, model_name, batches, split.all_months(), len(node_ids), device)
        pred_frame = pair_scores_to_frame(
            batches, split, train_scores, model_name, target_column, pair_frame
        )
        predictions[model_name] = pred_frame
        metrics[model_name] = metrics_by_split(options.task, predictions[model_name], options.threshold)
        predictions[model_name].to_csv(output_dir / "predictions" / f"{model_name}.csv", index=False)

    write_comparison(options.task, metrics, output_dir / "metrics" / "comparison.csv")
    for name, split_metrics in metrics.items():
        write_json(split_metrics, output_dir / "metrics" / f"{name}.json")

    plots = write_pair_plots(options.task, pair_frame, histories, predictions, metrics, output_dir / "plots")

    summary = {
        "supervision": "pooled_directed_pair_month",
        "task": options.task,
        "tau": options.tau if options.task == CLASSIFICATION else None,
        "target_column": target_column,
        "seconds": round(time.perf_counter() - started, 3),
        "output_dir": str(output_dir),
        "environment": runtime.environment_report(choice),
        "split": split.to_manifest(),
        "pair_target_stats": pair_target_stats,
        "node_target_stats": node_target_stats,
        "validation": validation,
        "baseline_notes": baseline_notes,
        "metrics": metrics,
        "plots": plots,
        "models": list(predictions.keys()),
    }
    write_json(summary, output_dir / "run_summary.json")
    write_readme(output_dir, summary)
    LOGGER.info("Pair run complete in %.1fs -> %s", summary["seconds"], output_dir)
    return summary
