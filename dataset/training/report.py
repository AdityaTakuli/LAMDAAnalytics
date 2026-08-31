"""Artifact writing: predictions, metrics, checkpoints, plots, and audits.

Every run writes a self-describing directory. Someone who has only that
directory should be able to say what data was used, how it was split, what was
fitted on what, which weights were selected, and what the numbers mean —
without reading the source.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from training.baselines import PREDICTION_COLUMNS
from training.data import MonthBatch, Split
from training.metrics import compute as compute_metrics
from training.metrics import metric_names

LOGGER = logging.getLogger("training.report")

SPLIT_NAMES = ("train", "validation", "test")


# --------------------------------------------------------------------------- #
# Predictions
# --------------------------------------------------------------------------- #
def graph_scores_to_frame(
    batches: Mapping[str, MonthBatch],
    split: Split,
    scores: Mapping[str, np.ndarray],
    node_ids: Sequence[str],
    model_name: str,
) -> pd.DataFrame:
    """Long prediction table restricted to country-months with a valid target."""
    rows: list[dict[str, Any]] = []
    for split_name, months in split.as_dict().items():
        for month in months:
            batch = batches.get(month)
            values = scores.get(month)
            if batch is None or values is None:
                continue
            mask = batch.mask.detach().cpu().numpy()
            target = batch.target.detach().cpu().numpy()
            for position, keep in enumerate(mask):
                if not keep:
                    continue
                rows.append(
                    {
                        "model": model_name,
                        "split": split_name,
                        "month": month,
                        "node_id": str(node_ids[position]),
                        "target": float(target[position]),
                        "score": float(values[position]),
                    }
                )
    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS)


def write_predictions(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def metrics_by_split(task: str, frame: pd.DataFrame, threshold: float = 0.5) -> dict[str, Any]:
    """Compute metrics for each partition present in a prediction frame."""
    result: dict[str, Any] = {}
    for split_name in SPLIT_NAMES:
        subset = frame[frame["split"].eq(split_name)] if not frame.empty else frame
        result[split_name] = compute_metrics(
            task,
            subset["target"].to_numpy() if not subset.empty else np.empty(0),
            subset["score"].to_numpy() if not subset.empty else np.empty(0),
            threshold=threshold,
        )
    return result


def write_comparison(task: str, metrics: Mapping[str, Mapping[str, Any]], path: Path) -> Path:
    """One flat CSV row per (model, split), ordered for direct reading."""
    names = metric_names(task)
    rows: list[dict[str, Any]] = []
    for model_name, split_metrics in metrics.items():
        for split_name in SPLIT_NAMES:
            values = split_metrics.get(split_name) or {}
            row: dict[str, Any] = {"model": model_name, "split": split_name, "n": values.get("n", 0)}
            if task == "classification":
                row.update(
                    {
                        "positive": values.get("positive"),
                        "negative": values.get("negative"),
                        "prevalence": values.get("prevalence"),
                    }
                )
            row.update({name: values.get(name) for name in names})
            row["note"] = values.get("note", "")
            rows.append(row)
    frame = pd.DataFrame(rows)
    order = {name: index for index, name in enumerate(SPLIT_NAMES)}
    frame = frame.sort_values(["split", "model"], key=lambda column: column.map(order).fillna(column))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def write_json(payload: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return path


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
def save_checkpoint(
    model: torch.nn.Module,
    path: Path,
    *,
    model_name: str,
    task: str,
    model_kwargs: Mapping[str, Any],
    features: Sequence[str],
    normalizer: Mapping[str, Any],
    edge_scale: tuple[float, float],
    node_order: Sequence[str],
    split: Split,
    target_column: str,
    tau: float | None,
    training_manifest: Mapping[str, Any],
) -> Path:
    """Write a checkpoint that can be reloaded without the original config."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "lamda-country-month-checkpoint-v1",
            "model_name": model_name,
            "task": task,
            "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "model_kwargs": dict(model_kwargs),
            "features": list(features),
            "normalizer": dict(normalizer),
            "edge_scale": {"trade_value_log1p": edge_scale[0], "flow_volume_log1p": edge_scale[1]},
            "node_order": list(node_order),
            "split": split.to_manifest(),
            "target_column": target_column,
            "tau": tau,
            "training": dict(training_manifest),
            "torch_version": torch.__version__,
            "synthetic": False,
        },
        path,
    )
    return path


# --------------------------------------------------------------------------- #
# Plots (never fatal)
# --------------------------------------------------------------------------- #
def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def write_plots(
    task: str,
    target_frame: pd.DataFrame,
    split: Split,
    histories: Mapping[str, list[dict[str, Any]]],
    predictions: Mapping[str, pd.DataFrame],
    metrics: Mapping[str, Mapping[str, Any]],
    directory: Path,
) -> list[str]:
    """Render diagnostic figures. A plotting failure is logged, never raised."""
    written: list[str] = []
    directory.mkdir(parents=True, exist_ok=True)
    try:
        plt = _pyplot()
    except Exception as exc:  # pragma: no cover - matplotlib is optional at runtime
        LOGGER.warning("Plots skipped: matplotlib is unavailable (%s)", exc)
        return written

    def _save(figure, name: str) -> None:
        destination = directory / name
        figure.tight_layout()
        figure.savefig(destination, dpi=140)
        plt.close(figure)
        written.append(str(destination))

    try:
        valid = target_frame[target_frame["target_valid"].fillna(False).astype(bool)]
        contraction = pd.to_numeric(valid["contraction"], errors="coerce").dropna()
        if len(contraction):
            figure, axis = plt.subplots(figsize=(7, 4))
            axis.hist(contraction, bins=min(40, max(10, len(contraction) // 5)), color="#4C72B0")
            axis.axvline(0.0, color="#333333", linewidth=1)
            axis.set_title("Distribution of the one-month-ahead contraction target")
            axis.set_xlabel("contraction = (future - baseline) / baseline")
            axis.set_ylabel("country-months")
            _save(figure, "target_distribution.png")
    except Exception as exc:
        LOGGER.warning("target_distribution plot failed: %s", exc)

    try:
        curves = {name: history for name, history in histories.items() if history}
        if curves:
            figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
            for name, history in curves.items():
                epochs = [record["epoch"] for record in history]
                axes[0].plot(epochs, [record["train_loss"] for record in history], marker="o", label=name)
                axes[1].plot(epochs, [record["validation_loss"] for record in history], marker="o", label=name)
            axes[0].set_title("Training loss")
            axes[1].set_title("Validation loss")
            for axis in axes:
                axis.set_xlabel("epoch")
                axis.set_ylabel("loss")
                axis.legend(fontsize=8)
            _save(figure, "loss_curves.png")
    except Exception as exc:
        LOGGER.warning("loss_curves plot failed: %s", exc)

    try:
        headline = "average_precision" if task == "classification" else "rmse"
        labels, validation_values, test_values = [], [], []
        for name, split_metrics in metrics.items():
            labels.append(name)
            validation_values.append(split_metrics.get("validation", {}).get(headline))
            test_values.append(split_metrics.get("test", {}).get(headline))
        if any(value is not None for value in validation_values + test_values):
            positions = np.arange(len(labels))
            figure, axis = plt.subplots(figsize=(max(7, len(labels) * 1.4), 4))
            axis.bar(positions - 0.2, [value or 0.0 for value in validation_values], width=0.4, label="validation")
            axis.bar(positions + 0.2, [value or 0.0 for value in test_values], width=0.4, label="test")
            axis.set_xticks(positions)
            axis.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
            axis.set_ylabel(headline)
            axis.set_title(f"Model comparison: {headline} (missing bars are undefined metrics)")
            axis.legend()
            _save(figure, "model_comparison.png")
    except Exception as exc:
        LOGGER.warning("model_comparison plot failed: %s", exc)

    try:
        frames = {name: frame for name, frame in predictions.items() if not frame.empty}
        if frames:
            columns = min(3, len(frames))
            rows = int(np.ceil(len(frames) / columns))
            figure, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.4 * rows), squeeze=False)
            for index, (name, frame) in enumerate(frames.items()):
                axis = axes[index // columns][index % columns]
                subset = frame[frame["split"].isin(["validation", "test"])]
                if subset.empty:
                    subset = frame
                if task == "classification":
                    for value, colour in ((0, "#4C72B0"), (1, "#C44E52")):
                        scores = subset.loc[subset["target"].eq(value), "score"]
                        if len(scores):
                            axis.hist(scores, bins=15, alpha=0.6, color=colour, label=f"label={value}")
                    axis.set_xlabel("risk score")
                    axis.legend(fontsize=7)
                else:
                    axis.scatter(subset["target"], subset["score"], s=12, alpha=0.7, color="#4C72B0")
                    limits = [
                        float(min(subset["target"].min(), subset["score"].min())),
                        float(max(subset["target"].max(), subset["score"].max())),
                    ]
                    axis.plot(limits, limits, color="#333333", linewidth=1)
                    axis.set_xlabel("actual contraction")
                    axis.set_ylabel("predicted")
                axis.set_title(name, fontsize=9)
            for index in range(len(frames), rows * columns):
                axes[index // columns][index % columns].axis("off")
            _save(figure, "prediction_distributions.png")
    except Exception as exc:
        LOGGER.warning("prediction_distributions plot failed: %s", exc)

    return written


# --------------------------------------------------------------------------- #
# Audits
# --------------------------------------------------------------------------- #
def write_leakage_audit(
    path: Path,
    *,
    task: str,
    split: Split,
    horizon: int,
    target_stats: Mapping[str, Any],
    target_column: str,
    node_count: int,
    features: Sequence[str],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    excluded = ", ".join(f"`{month}`" for month in split.excluded_months) or "none"
    notes = "\n".join(f"* {note}" for note in split.notes) or "* none"
    path.write_text(
        f"""# Leakage audit

Task: `{task}` on target column `{target_column}`.

## Temporal construction

* Prediction horizon: `{horizon}` month(s). The target for month `T` is derived
  from inbound Comtrade flow at `T+{horizon}` and a rolling baseline that ends
  at `T+{horizon}-1`.
* The model inputs are the {len(features)} fused features of month `T` only:
  {", ".join(f"`{name}`" for name in features)}.
* `inbound_flow_usd`, `future_inbound_flow_usd`, `baseline_inbound_flow_usd`,
  `contraction`, and every `label_tau_*` column are targets or target
  ingredients. None of them is a model input.
* GCN edges come from the same month's snapshot only.
* TGN events are replayed once, in chronological order, from
  `{split.train[0]}` to `{split.test[-1]}`. Memory is reset before the first
  training month and then carried forward across partition boundaries; it is
  never reset before a validation or test month, and never sees a later month
  before an earlier one.

## Partitions

* Train      : `{split.train[0]}` .. `{split.train[-1]}` ({len(split.train)} months)
* Validation : `{split.validation[0]}` .. `{split.validation[-1]}` ({len(split.validation)} months)
* Test       : `{split.test[0]}` .. `{split.test[-1]}` ({len(split.test)} months)
* Partitions are disjoint and strictly forward-chained. No shuffling, no
  cross-validation folds that mix time, no resampling.
* Months excluded for having no observable target: {excluded}.
* Split notes:
{notes}

## Fitted on training months only

* Feature standardisation (mean and scale).
* Edge log1p scaling for `trade_value_usd` and `flow_volume`.
* The class weight used by the weighted BCE loss.
* The logistic-regression, ridge, and constant baselines.
* Model selection uses the validation partition only. The test partition is
  scored exactly once, with the selected weights.

## Target validity

* Country-month rows: {target_stats.get('country_month_rows')}
* Valid targets: {target_stats.get('valid_targets')}
* Invalid targets: {target_stats.get('invalid_targets')}
  (future value unobserved: {target_stats.get('invalid_reasons', {}).get('future_value_unobserved')},
  baseline unavailable: {target_stats.get('invalid_reasons', {}).get('baseline_unavailable')},
  baseline not positive: {target_stats.get('invalid_reasons', {}).get('baseline_not_positive')})
* Invalid rows are excluded from training and evaluation. They are never
  converted into negative examples, and no label is synthesised, oversampled,
  or rebalanced.
* All {node_count} country nodes stay in every graph snapshot so the topology
  is unchanged; only the supervised rows are restricted.

## What this audit does and does not establish

It establishes that the temporal construction and the fitting boundaries are
correct. It does not establish that the resulting sample size or class balance
supports a conclusive performance claim; read `metrics/comparison.csv`
together with `diagnostics/class_balance.json` before quoting any number.
""",
        encoding="utf-8",
    )
    return path
