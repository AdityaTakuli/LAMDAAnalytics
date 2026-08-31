"""Non-graph reference models.

A graph model is only interesting relative to something simpler. These
baselines are fitted under exactly the same chronological split and the same
train-only standardisation as the GCN and TGN, so the comparison is fair:

classification
  ``constant_prevalence``   predicts the training prevalence for every row
  ``hand_weighted_linear``  the fixed, unfitted deployment heuristic
  ``logistic_regression``   a balanced linear classifier on the same features

regression
  ``train_median``          predicts the median training contraction
  ``ridge_regression``      a linear model on the same features
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

from training import paths  # noqa: F401  (sys.path bootstrap)
from training.data import FEATURES, FeatureStandardizer, Split

from model_tgn import DEFAULT_LINEAR_WEIGHTS, linear_composition  # noqa: E402

LOGGER = logging.getLogger("training.baselines")

PREDICTION_COLUMNS = ["model", "split", "month", "node_id", "target", "score"]

CLASSIFICATION_BASELINES = ("constant_prevalence", "hand_weighted_linear", "logistic_regression")
REGRESSION_BASELINES = ("train_median", "ridge_regression")


def _partition(frame: pd.DataFrame, months: Sequence[str], target_column: str) -> pd.DataFrame:
    subset = frame[
        frame["month"].isin(list(months))
        & frame["target_valid"].fillna(False).astype(bool)
        & frame[target_column].notna()
    ]
    return subset.copy()


def _rows(subset: pd.DataFrame, model: str, split_name: str, target_column: str, scores: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": model,
            "split": split_name,
            "month": subset["month"].astype(str).to_numpy(),
            "node_id": subset["node_id"].astype(str).to_numpy(),
            "target": pd.to_numeric(subset[target_column], errors="coerce").to_numpy(dtype=float),
            "score": np.asarray(scores, dtype=float),
        },
        columns=PREDICTION_COLUMNS,
    )


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=PREDICTION_COLUMNS)


def run_baselines(
    task: str,
    frame: pd.DataFrame,
    split: Split,
    standardizer: FeatureStandardizer,
    target_column: str,
    seed: int = 7,
    selected: Sequence[str] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Fit and score every applicable baseline; return frames and notes."""
    available = CLASSIFICATION_BASELINES if task == "classification" else REGRESSION_BASELINES
    wanted = [name for name in (selected or available) if name in available]
    predictions: dict[str, pd.DataFrame] = {}
    notes: dict[str, Any] = {}

    train = _partition(frame, split.train, target_column)
    if train.empty:
        for name in wanted:
            predictions[name] = _empty_frame()
            notes[name] = "N/A - the training partition has no valid target rows"
        return predictions, notes

    train_target = pd.to_numeric(train[target_column], errors="coerce").to_numpy(dtype=float)

    for name in wanted:
        parts: list[pd.DataFrame] = []
        note: str | None = None

        if name == "constant_prevalence":
            constant = float(np.mean(train_target))
            note = f"constant score {constant:.6f} (training prevalence)"
            for split_name, months in split.as_dict().items():
                subset = _partition(frame, months, target_column)
                if subset.empty:
                    continue
                parts.append(_rows(subset, name, split_name, target_column, np.full(len(subset), constant)))

        elif name == "train_median":
            constant = float(np.median(train_target))
            note = f"constant prediction {constant:.6f} (median training contraction)"
            for split_name, months in split.as_dict().items():
                subset = _partition(frame, months, target_column)
                if subset.empty:
                    continue
                parts.append(_rows(subset, name, split_name, target_column, np.full(len(subset), constant)))

        elif name == "hand_weighted_linear":
            note = "fixed deployment weights; not fitted on any partition"
            for split_name, months in split.as_dict().items():
                subset = _partition(frame, months, target_column)
                if subset.empty:
                    continue
                scores = np.asarray(
                    [
                        linear_composition({feature: float(row[feature]) for feature in FEATURES})[0]
                        for _, row in subset.iterrows()
                    ],
                    dtype=float,
                )
                parts.append(_rows(subset, name, split_name, target_column, scores))

        elif name == "logistic_regression":
            if np.unique(train_target).size < 2:
                predictions[name] = _empty_frame()
                notes[name] = (
                    "N/A - the training partition contains a single class, so a logistic "
                    "classifier cannot be fitted"
                )
                LOGGER.warning("logistic_regression skipped: single-class training labels")
                continue
            classifier = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
            classifier.fit(standardizer.transform(train), train_target.astype(int))
            note = "balanced logistic regression on train-standardised features"
            for split_name, months in split.as_dict().items():
                subset = _partition(frame, months, target_column)
                if subset.empty:
                    continue
                scores = classifier.predict_proba(standardizer.transform(subset))[:, 1]
                parts.append(_rows(subset, name, split_name, target_column, scores))

        elif name == "ridge_regression":
            model = Ridge(alpha=1.0, random_state=seed)
            model.fit(standardizer.transform(train), train_target)
            note = "ridge regression (alpha=1.0) on train-standardised features"
            for split_name, months in split.as_dict().items():
                subset = _partition(frame, months, target_column)
                if subset.empty:
                    continue
                parts.append(
                    _rows(subset, name, split_name, target_column, model.predict(standardizer.transform(subset)))
                )

        predictions[name] = pd.concat(parts, ignore_index=True) if parts else _empty_frame()
        if note:
            notes[name] = note

    notes["hand_weighted_linear_weights"] = dict(DEFAULT_LINEAR_WEIGHTS) if "hand_weighted_linear" in wanted else None
    return predictions, {key: value for key, value in notes.items() if value is not None}
