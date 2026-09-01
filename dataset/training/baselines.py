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
  ``random_forest``         ensemble on the same features
  ``lightgbm``              gradient boosting on the same features
  ``lightgbm_lags``         gradient boosting with causal lag features
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from training import paths  # noqa: F401  (sys.path bootstrap)
from training.data import FEATURES, FeatureStandardizer, Split
from training.lag_features import LAG_COLUMNS, add_country_lags

from model_tgn import DEFAULT_LINEAR_WEIGHTS, linear_composition  # noqa: E402

LOGGER = logging.getLogger("training.baselines")

PREDICTION_COLUMNS = ["model", "split", "month", "node_id", "target", "score"]

CLASSIFICATION_BASELINES = (
    "constant_prevalence",
    "hand_weighted_linear",
    "logistic_regression",
    "random_forest",
    "lightgbm",
    "lightgbm_lags",
)
REGRESSION_BASELINES = ("train_median", "ridge_regression", "random_forest", "lightgbm", "lightgbm_lags")

TABULAR_EXTRA_FEATURES = list(LAG_COLUMNS)


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


def _feature_matrix(frame: pd.DataFrame, standardizer: FeatureStandardizer) -> np.ndarray:
    return standardizer.transform(frame)


def _fit_lightgbm(task: str, train_x: np.ndarray, train_y: np.ndarray, seed: int):
    import lightgbm as lgb

    if task == "classification":
        pos = max(float(train_y.sum()), 1.0)
        neg = max(float(len(train_y) - train_y.sum()), 1.0)
        return lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=5,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=neg / pos,
            random_state=seed,
            verbosity=-1,
        ).fit(train_x, train_y.astype(int))
    return lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=5,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        verbosity=-1,
    ).fit(train_x, train_y)


def _predict_lightgbm(model, task: str, features: np.ndarray) -> np.ndarray:
    if task == "classification":
        return model.predict_proba(features)[:, 1]
    return model.predict(features)


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
    frame = add_country_lags(frame)
    available = CLASSIFICATION_BASELINES if task == "classification" else REGRESSION_BASELINES
    wanted = [name for name in (selected or available) if name in available]
    predictions: dict[str, pd.DataFrame] = {}
    notes: dict[str, Any] = {}
    lag_standardizer = FeatureStandardizer()
    lag_standardizer.features = [*FEATURES, *TABULAR_EXTRA_FEATURES]

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

        elif name == "random_forest":
            if task == "classification":
                if np.unique(train_target).size < 2:
                    predictions[name] = _empty_frame()
                    notes[name] = "N/A - single-class training labels"
                    continue
                model = RandomForestClassifier(
                    n_estimators=300,
                    max_depth=6,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=-1,
                )
                model.fit(standardizer.transform(train), train_target.astype(int))
                note = "balanced random forest on train-standardised features"
                predict = lambda subset: model.predict_proba(standardizer.transform(subset))[:, 1]
            else:
                model = RandomForestRegressor(
                    n_estimators=300,
                    max_depth=6,
                    min_samples_leaf=2,
                    random_state=seed,
                    n_jobs=-1,
                )
                model.fit(standardizer.transform(train), train_target)
                note = "random forest on train-standardised features"
                predict = lambda subset: model.predict(standardizer.transform(subset))
            for split_name, months in split.as_dict().items():
                subset = _partition(frame, months, target_column)
                if subset.empty:
                    continue
                parts.append(_rows(subset, name, split_name, target_column, predict(subset)))

        elif name in {"lightgbm", "lightgbm_lags"}:
            use_lags = name == "lightgbm_lags"
            active = lag_standardizer if use_lags else standardizer
            train_subset = _partition(frame, split.train, target_column)
            if use_lags:
                active.fit(train_subset)
            train_x = active.transform(train_subset)
            train_y = pd.to_numeric(train_subset[target_column], errors="coerce").to_numpy(dtype=float)
            if task == "classification" and np.unique(train_y).size < 2:
                predictions[name] = _empty_frame()
                notes[name] = "N/A - single-class training labels"
                continue
            try:
                model = _fit_lightgbm(task, train_x, train_y, seed)
            except Exception as exc:
                predictions[name] = _empty_frame()
                notes[name] = f"N/A - lightgbm unavailable ({exc})"
                continue
            note = "LightGBM on standardised features" + (" with causal lags" if use_lags else "")
            for split_name, months in split.as_dict().items():
                subset = _partition(frame, months, target_column)
                if subset.empty:
                    continue
                scores = _predict_lightgbm(model, task, active.transform(subset))
                parts.append(_rows(subset, name, split_name, target_column, scores))

        predictions[name] = pd.concat(parts, ignore_index=True) if parts else _empty_frame()
        if note:
            notes[name] = note

    notes["hand_weighted_linear_weights"] = dict(DEFAULT_LINEAR_WEIGHTS) if "hand_weighted_linear" in wanted else None
    return predictions, {key: value for key, value in notes.items() if value is not None}
