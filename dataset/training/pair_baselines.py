"""Tabular baselines for pooled pair-month supervision."""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from training.pair_data import PAIR_PREDICTION_COLUMNS, PairFeatureStandardizer

LOGGER = logging.getLogger("training.pair_baselines")

CLASSIFICATION_BASELINES = ("constant_prevalence", "logistic_regression", "random_forest", "lightgbm", "mlp")
REGRESSION_BASELINES = ("train_median", "ridge_regression", "random_forest", "lightgbm", "mlp")


def _partition(frame: pd.DataFrame, months: Sequence[str], target_column: str) -> pd.DataFrame:
    return frame[
        frame["month"].isin(list(months))
        & frame["target_valid"].fillna(False).astype(bool)
        & frame[target_column].notna()
    ].copy()


def _rows(
    subset: pd.DataFrame,
    model: str,
    split_name: str,
    target_column: str,
    scores: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": model,
            "split": split_name,
            "month": subset["month"].astype(str).to_numpy(),
            "source": subset["source"].astype(str).to_numpy(),
            "destination": subset["destination"].astype(str).to_numpy(),
            "target": pd.to_numeric(subset[target_column], errors="coerce").to_numpy(dtype=float),
            "score": np.asarray(scores, dtype=float),
        },
        columns=PAIR_PREDICTION_COLUMNS,
    )


def run_pair_baselines(
    task: str,
    frame: pd.DataFrame,
    split,
    standardizer: PairFeatureStandardizer,
    target_column: str,
    seed: int = 7,
    selected: Sequence[str] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    available = CLASSIFICATION_BASELINES if task == "classification" else REGRESSION_BASELINES
    wanted = [name for name in (selected or available) if name in available and name != "mlp"]
    predictions: dict[str, pd.DataFrame] = {}
    notes: dict[str, Any] = {}

    train = _partition(frame, split.train, target_column)
    if train.empty:
        for name in wanted:
            predictions[name] = pd.DataFrame(columns=PAIR_PREDICTION_COLUMNS)
            notes[name] = "N/A - the training partition has no valid pair targets"
        return predictions, notes

    train_target = pd.to_numeric(train[target_column], errors="coerce").to_numpy(dtype=float)
    train_features = standardizer.transform(train)

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

        elif name == "logistic_regression":
            if np.unique(train_target).size < 2:
                predictions[name] = pd.DataFrame(columns=PAIR_PREDICTION_COLUMNS)
                notes[name] = "N/A - single-class training labels"
                continue
            classifier = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
            classifier.fit(train_features, train_target.astype(int))
            note = "balanced logistic regression on concatenated source, destination, and edge features"
            for split_name, months in split.as_dict().items():
                subset = _partition(frame, months, target_column)
                if subset.empty:
                    continue
                scores = classifier.predict_proba(standardizer.transform(subset))[:, 1]
                parts.append(_rows(subset, name, split_name, target_column, scores))

        elif name == "ridge_regression":
            model = Ridge(alpha=1.0, random_state=seed)
            model.fit(train_features, train_target)
            note = "ridge regression on concatenated source, destination, and edge features"
            for split_name, months in split.as_dict().items():
                subset = _partition(frame, months, target_column)
                if subset.empty:
                    continue
                scores = model.predict(standardizer.transform(subset))
                parts.append(_rows(subset, name, split_name, target_column, scores))

        elif name == "random_forest":
            if task == "classification":
                if np.unique(train_target).size < 2:
                    predictions[name] = pd.DataFrame(columns=PAIR_PREDICTION_COLUMNS)
                    notes[name] = "N/A - single-class training labels"
                    continue
                model = RandomForestClassifier(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=-1,
                )
                model.fit(train_features, train_target.astype(int))
                note = "balanced random forest on pair features"
                predict = lambda subset: model.predict_proba(standardizer.transform(subset))[:, 1]
            else:
                model = RandomForestRegressor(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_leaf=2,
                    random_state=seed,
                    n_jobs=-1,
                )
                model.fit(train_features, train_target)
                note = "random forest on pair features"
                predict = lambda subset: model.predict(standardizer.transform(subset))
            for split_name, months in split.as_dict().items():
                subset = _partition(frame, months, target_column)
                if subset.empty:
                    continue
                parts.append(_rows(subset, name, split_name, target_column, predict(subset)))

        elif name == "lightgbm":
            try:
                import lightgbm as lgb
            except ImportError as exc:
                predictions[name] = pd.DataFrame(columns=PAIR_PREDICTION_COLUMNS)
                notes[name] = f"N/A - lightgbm unavailable ({exc})"
                continue
            if task == "classification":
                if np.unique(train_target).size < 2:
                    predictions[name] = pd.DataFrame(columns=PAIR_PREDICTION_COLUMNS)
                    notes[name] = "N/A - single-class training labels"
                    continue
                pos = max(float(train_target.sum()), 1.0)
                neg = max(float(len(train_target) - train_target.sum()), 1.0)
                model = lgb.LGBMClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    num_leaves=31,
                    scale_pos_weight=neg / pos,
                    random_state=seed,
                    verbosity=-1,
                ).fit(train_features, train_target.astype(int))
                note = "LightGBM on pair features"
                predict = lambda subset: model.predict_proba(standardizer.transform(subset))[:, 1]
            else:
                model = lgb.LGBMRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    num_leaves=31,
                    random_state=seed,
                    verbosity=-1,
                ).fit(train_features, train_target)
                note = "LightGBM on pair features"
                predict = lambda subset: model.predict(standardizer.transform(subset))
            for split_name, months in split.as_dict().items():
                subset = _partition(frame, months, target_column)
                if subset.empty:
                    continue
                parts.append(_rows(subset, name, split_name, target_column, predict(subset)))

        predictions[name] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=PAIR_PREDICTION_COLUMNS)
        if note:
            notes[name] = note

    return predictions, notes
