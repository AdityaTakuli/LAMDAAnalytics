"""Metric computation with explicit, honest handling of degenerate splits.

A partition that contains a single class has no defined ROC-AUC, precision,
recall, or F1. Those metrics are reported as ``null`` with a note rather than
being silently replaced by 0.0 or 1.0, because a fabricated number here would
travel straight into a results table.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

CLASSIFICATION_METRICS: tuple[str, ...] = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
    "brier_score",
)

REGRESSION_METRICS: tuple[str, ...] = (
    "mae",
    "rmse",
    "r2",
    "pearson_r",
    "spearman_r",
    "directional_accuracy",
)

EMPTY_NOTE = "N/A - no valid target observations in this partition"
SINGLE_CLASS_NOTE = (
    "N/A - only one class is present, so precision, recall, F1, ROC-AUC and PR-AUC are undefined; "
    "accuracy and balanced accuracy are descriptive only"
)


def _empty(names: Sequence[str]) -> dict[str, Any]:
    return {
        "n": 0,
        "positive": 0,
        "negative": 0,
        "prevalence": None,
        **{name: None for name in names},
        "note": EMPTY_NOTE,
    }


def classification_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5
) -> dict[str, Any]:
    """Binary metrics from probability scores in ``[0, 1]``."""
    labels = np.asarray(labels, dtype=float).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    keep = ~np.isnan(labels) & ~np.isnan(scores)
    labels, scores = labels[keep].astype(int), scores[keep]
    if labels.size == 0:
        return _empty(CLASSIFICATION_METRICS)

    predicted = (scores >= threshold).astype(int)
    positives = int(labels.sum())
    result: dict[str, Any] = {
        "n": int(labels.size),
        "positive": positives,
        "negative": int(labels.size - positives),
        "prevalence": float(labels.mean()),
        "decision_threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predicted)),
        "brier_score": float(np.mean((scores - labels) ** 2)),
    }
    if np.unique(labels).size < 2:
        result.update(
            {
                "balanced_accuracy": float(accuracy_score(labels, predicted)),
                "precision": None,
                "recall": None,
                "f1": None,
                "roc_auc": None,
                "average_precision": None,
                "note": SINGLE_CLASS_NOTE,
            }
        )
        return result
    result.update(
        {
            "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
            "precision": float(precision_score(labels, predicted, zero_division=0)),
            "recall": float(recall_score(labels, predicted, zero_division=0)),
            "f1": float(f1_score(labels, predicted, zero_division=0)),
            "roc_auc": float(roc_auc_score(labels, scores)),
            "average_precision": float(average_precision_score(labels, scores)),
            "note": "",
        }
    )
    return result


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Error and rank-agreement metrics for the continuous contraction target."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    keep = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true, y_pred = y_true[keep], y_pred[keep]
    if y_true.size == 0:
        return {
            "n": 0,
            **{name: None for name in REGRESSION_METRICS},
            "note": EMPTY_NOTE,
        }

    residual = y_pred - y_true
    variance = float(np.var(y_true))
    result: dict[str, Any] = {
        "n": int(y_true.size),
        "target_mean": float(y_true.mean()),
        "target_std": float(np.std(y_true)),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "r2": float(1.0 - np.sum(residual ** 2) / np.sum((y_true - y_true.mean()) ** 2))
        if variance > 0
        else None,
        "directional_accuracy": float(np.mean(np.sign(y_pred) == np.sign(y_true))),
        "note": "",
    }
    if y_true.size >= 2 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        result["pearson_r"] = float(np.corrcoef(y_true, y_pred)[0, 1])
        result["spearman_r"] = float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"))
    else:
        result["pearson_r"] = None
        result["spearman_r"] = None
        result["note"] = "N/A - correlation is undefined for a constant series"
    if variance <= 0:
        result["note"] = (result["note"] + " " if result["note"] else "") + (
            "N/A - R^2 is undefined because the target has zero variance"
        ).strip()
    return result


def compute(task: str, y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    if task == "classification":
        return classification_metrics(y_true, y_pred, threshold=threshold)
    return regression_metrics(y_true, y_pred)


def metric_names(task: str) -> tuple[str, ...]:
    return CLASSIFICATION_METRICS if task == "classification" else REGRESSION_METRICS


def selection_score(task: str, metrics: dict[str, Any]) -> float | None:
    """Higher-is-better model-selection value, or ``None`` when undefined.

    Classification prefers PR-AUC because the positive class is rare; ROC-AUC
    is the fallback. Regression uses negative RMSE. When neither is defined
    (for example a single-class validation split) the caller falls back to the
    validation loss.
    """
    if not metrics:
        return None
    if task == "classification":
        for name in ("average_precision", "roc_auc"):
            value = metrics.get(name)
            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                return float(value)
        return None
    value = metrics.get("rmse")
    return None if value is None else -float(value)
