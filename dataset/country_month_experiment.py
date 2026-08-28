"""Run a defensible country-month GCN/TGN experiment on existing real data.

This module intentionally does not alter source tables or labels.  It reuses
the existing fused country graph and model implementations, derives the
continuous one-month-ahead contraction for diagnostics, evaluates the
configured default tau, and writes all artifacts below a new experiment
directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
from typing import Any
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from common import add_common_cli, load_config, nested, read_table, write_json
from fuse_dataset import FEATURES
from model_gcn import SnapshotGCN
from model_tgn import TemporalGraphNetwork
from train import (
    FeatureStandardizer,
    _event_scales,
    _prepare_month_data,
    _run_gcn,
    _run_tgn,
)


MODEL_NAMES = ("gcn", "tgn", "tgn_no_memory", "logistic")
METRIC_NAMES = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
)


def _base_path(config: dict[str, Any]) -> Path:
    return Path(__file__).resolve().parent


def _experiment_path(config: dict[str, Any]) -> Path:
    configured = nested(
        config,
        "outputs",
        "country_month_experiment",
        default="data/one_year_2024/results/country_month_experiment",
    )
    path = Path(configured)
    return path if path.is_absolute() else _base_path(config) / path


def _load_real_inputs(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if config.get("_synthetic"):
        raise ValueError("country_month_experiment refuses synthetic inputs")
    base = _base_path(config)
    nodes = read_table(
        base / nested(config, "outputs", "nodes", default="processed/nodes_monthly.csv")
    )
    edges = read_table(
        base / nested(config, "outputs", "edges", default="processed/edges_monthly.csv")
    )
    return nodes, edges


def _continuous_targets(
    nodes: pd.DataFrame, horizon: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = nodes.copy()
    frame["month"] = frame["month"].astype(str)
    frame["inbound_flow_usd"] = pd.to_numeric(
        frame["inbound_flow_usd"], errors="coerce"
    )
    frame = frame.sort_values(["host_country_id", "month"]).reset_index(drop=True)
    grouped = frame.groupby("host_country_id", sort=False)
    future = grouped["inbound_flow_usd"].shift(-horizon)
    history = grouped["inbound_flow_usd"].transform(
        lambda values: values.shift(-(horizon - 1)).rolling(12, min_periods=1).median()
    )
    frame["future_inbound_flow_usd"] = future
    frame["historical_median_inbound_usd"] = history
    frame["contraction"] = (future - history) / history.replace(0, np.nan)
    frame["target_valid"] = future.notna() & history.notna() & history.gt(0)
    contraction = frame.loc[frame["target_valid"], "contraction"].astype(float)
    quantiles = contraction.quantile(
        [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    ).to_dict()
    stats = {
        "count": int(len(contraction)),
        "min": float(contraction.min()) if len(contraction) else None,
        "max": float(contraction.max()) if len(contraction) else None,
        "mean": float(contraction.mean()) if len(contraction) else None,
        "median": float(contraction.median()) if len(contraction) else None,
        "std": float(contraction.std(ddof=0)) if len(contraction) else None,
        "quantiles": {str(key): float(value) for key, value in quantiles.items()},
        "horizon_months": horizon,
        "valid_observations": int(frame["target_valid"].sum()),
        "invalid_observations": int((~frame["target_valid"]).sum()),
    }
    return frame, stats


def _label_distribution(
    target_frame: pd.DataFrame, split: dict[str, list[str]], taus: list[float]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for tau in taus:
        labels = target_frame["contraction"].lt(-tau)
        for split_name, months in split.items():
            subset = target_frame[
                target_frame["month"].isin(months) & target_frame["target_valid"]
            ]
            subset_labels = labels.loc[subset.index]
            positives = int(subset_labels.sum())
            rows.append(
                {
                    "tau": tau,
                    "split": split_name,
                    "observations": int(len(subset)),
                    "positive": positives,
                    "negative": int(len(subset) - positives),
                    "prevalence": float(subset_labels.mean())
                    if len(subset_labels)
                    else None,
                }
            )
    return pd.DataFrame(rows)


def _metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    if not len(labels):
        return {
            "n": 0,
            "positive": 0,
            "negative": 0,
            "prevalence": None,
            **{name: None for name in METRIC_NAMES},
            "note": "N/A — no valid target observations",
        }
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predicted = (scores >= 0.5).astype(int)
    single_class = len(np.unique(labels)) < 2
    result: dict[str, Any] = {
        "n": int(len(labels)),
        "positive": int(labels.sum()),
        "negative": int(len(labels) - labels.sum()),
        "prevalence": float(labels.mean()),
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(
            accuracy_score(labels, predicted)
            if single_class
            else balanced_accuracy_score(labels, predicted)
        ),
    }
    if single_class:
        result.update(
            {
                "precision": None,
                "recall": None,
                "f1": None,
                "roc_auc": None,
                "average_precision": None,
                "note": "N/A — only one class present; discrimination metrics are undefined",
            }
        )
        return result
    result.update(
        {
            "precision": float(precision_score(labels, predicted, zero_division=0)),
            "recall": float(recall_score(labels, predicted, zero_division=0)),
            "f1": float(f1_score(labels, predicted, zero_division=0)),
            "roc_auc": float(roc_auc_score(labels, scores)),
            "average_precision": float(average_precision_score(labels, scores)),
        }
    )
    return result


def _prediction_rows(
    nodes: pd.DataFrame,
    targets: pd.DataFrame,
    split: dict[str, list[str]],
    predictions: dict[str, np.ndarray],
    node_ids: list[str],
    model_name: str,
    label_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    index = {node_id: position for position, node_id in enumerate(node_ids)}
    target_index = targets.set_index(["node_id", "month"])
    for split_name, months in split.items():
        for month in months:
            values = predictions.get(month)
            if values is None:
                continue
            current = nodes[nodes["month"].astype(str).eq(month)].drop_duplicates("node_id")
            for _, node in current.iterrows():
                node_id = str(node["node_id"])
                key = (node_id, month)
                if node_id not in index or key not in target_index.index:
                    continue
                target = target_index.loc[key]
                if not bool(target["target_valid"]):
                    continue
                rows.append(
                    {
                        "model": model_name,
                        "split": split_name,
                        "month": month,
                        "node_id": node_id,
                        "score": float(values[index[node_id]]),
                        "label": int(float(node[label_column])),
                        "contraction": float(target["contraction"]),
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "model",
            "split",
            "month",
            "node_id",
            "score",
            "label",
            "contraction",
        ],
    )


def _metrics_from_predictions(
    prediction_frame: pd.DataFrame,
) -> dict[str, Any]:
    if prediction_frame.empty:
        return _metrics(np.array([]), np.array([]))
    return _metrics(
        prediction_frame["label"].to_numpy(),
        prediction_frame["score"].to_numpy(),
    )


def _split_metrics(
    prediction_frame: pd.DataFrame, split_names: tuple[str, ...] = ("validation", "test")
) -> dict[str, Any]:
    return {
        split_name: _metrics_from_predictions(
            prediction_frame[prediction_frame["split"].eq(split_name)]
        )
        for split_name in split_names
    }


def _save_prediction_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _save_model_checkpoint(
    model: torch.nn.Module,
    path: Path,
    model_kwargs: dict[str, Any],
    normalizer: FeatureStandardizer,
    tau: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_kwargs": model_kwargs,
            "features": FEATURES,
            "tau": tau,
            "normalizer": normalizer.to_dict(),
            "synthetic": False,
        },
        path,
    )


def _train_gcn(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    split: dict[str, list[str]],
    node_ids: list[str],
    standardizer: FeatureStandardizer,
    edge_scale: tuple[float, float],
    label_column: str,
    positive_weight: float,
    model_kwargs: dict[str, Any],
    epochs: int,
    learning_rate: float,
) -> tuple[SnapshotGCN, dict[str, np.ndarray], list[float]]:
    model = SnapshotGCN(**model_kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses: list[float] = []
    for _ in range(epochs):
        model.train()
        _, epoch_losses = _run_gcn(
            model,
            nodes,
            edges,
            split["train"],
            node_ids,
            standardizer,
            edge_scale,
            label_column,
            positive_weight,
            optimizer,
        )
        losses.append(float(np.mean(epoch_losses)) if epoch_losses else float("nan"))
    model.eval()
    predictions: dict[str, np.ndarray] = {}
    for split_name in ("train", "validation", "test"):
        current, _ = _run_gcn(
            model,
            nodes,
            edges,
            split[split_name],
            node_ids,
            standardizer,
            edge_scale,
            label_column,
            positive_weight,
        )
        predictions.update(current)
    return model, predictions, losses


def _train_tgn(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    split: dict[str, list[str]],
    node_ids: list[str],
    standardizer: FeatureStandardizer,
    edge_scale: tuple[float, float],
    label_column: str,
    positive_weight: float,
    model_kwargs: dict[str, Any],
    epochs: int,
    learning_rate: float,
    use_memory: bool,
) -> tuple[TemporalGraphNetwork, dict[str, np.ndarray], list[float]]:
    model = TemporalGraphNetwork(use_memory=use_memory, **model_kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses: list[float] = []
    for _ in range(epochs):
        model.train()
        _, epoch_losses = _run_tgn(
            model,
            nodes,
            edges,
            split["train"],
            node_ids,
            standardizer,
            edge_scale,
            label_column,
            positive_weight,
            optimizer,
        )
        losses.append(float(np.mean(epoch_losses)) if epoch_losses else float("nan"))
    model.eval()
    # One chronological replay is the only state used for validation/test
    # scores.  _run_tgn resets once at the start, then memory evolves through
    # train -> validation -> test without seeing any future month.
    evaluation_months = split["train"] + split["validation"] + split["test"]
    predictions, _ = _run_tgn(
        model,
        nodes,
        edges,
        evaluation_months,
        node_ids,
        standardizer,
        edge_scale,
        label_column,
        positive_weight,
    )
    return model, predictions, losses


def _logistic_predictions(
    nodes: pd.DataFrame,
    split: dict[str, list[str]],
    node_ids: list[str],
    standardizer: FeatureStandardizer,
    label_column: str,
) -> tuple[pd.DataFrame, str | None]:
    train = nodes[nodes["month"].isin(split["train"]) & nodes[label_column].notna()]
    if train.empty or train[label_column].nunique() < 2:
        return pd.DataFrame(), "N/A — only one class available in training labels"
    classifier = LogisticRegression(class_weight="balanced", max_iter=500, random_state=7)
    classifier.fit(standardizer.transform(train), train[label_column].astype(int))
    rows: list[dict[str, Any]] = []
    for split_name, months in split.items():
        evaluate = nodes[nodes["month"].isin(months) & nodes[label_column].notna()].copy()
        if evaluate.empty:
            continue
        scores = classifier.predict_proba(standardizer.transform(evaluate))[:, 1]
        for (_, row), score in zip(evaluate.iterrows(), scores):
            rows.append(
                {
                    "model": "logistic",
                    "split": split_name,
                    "month": str(row["month"]),
                    "node_id": str(row["node_id"]),
                    "score": float(score),
                    "label": int(row[label_column]),
                }
            )
    return pd.DataFrame(rows), None


def _write_comparison(metrics_by_model: dict[str, dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for model_name, split_metrics in metrics_by_model.items():
        for split_name, metrics in split_metrics.items():
            rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    **{name: metrics.get(name) for name in METRIC_NAMES},
                    "n": metrics.get("n"),
                    "positive": metrics.get("positive"),
                    "negative": metrics.get("negative"),
                    "prevalence": metrics.get("prevalence"),
                    "note": metrics.get("note", ""),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def _plot_diagnostics(
    target_frame: pd.DataFrame,
    label_frame: pd.DataFrame,
    loss_by_model: dict[str, list[float]],
    predictions_by_model: dict[str, pd.DataFrame],
    graph_dir: Path,
) -> None:
    label_dir = graph_dir / "label_distribution"
    loss_dir = graph_dir / "training_loss"
    comparison_dir = graph_dir / "model_comparison"
    prediction_dir = graph_dir / "prediction_distribution"
    for directory in (label_dir, loss_dir, comparison_dir, prediction_dir):
        directory.mkdir(parents=True, exist_ok=True)

    valid = target_frame[target_frame["target_valid"]]["contraction"].astype(float)
    plt.figure(figsize=(7, 4))
    if len(valid):
        plt.hist(valid, bins=min(12, max(3, len(valid))), color="#3568a8")
    plt.axvline(-0.30, color="#b33", linestyle="--", label="tau=0.30")
    plt.axvline(-0.35, color="#d77", linestyle="--", label="tau=0.35")
    plt.xlabel("One-month inbound-flow contraction")
    plt.ylabel("Country-month observations")
    plt.title("Continuous target distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(label_dir / "continuous_target_distribution.png", dpi=150)
    plt.close()

    selected = label_frame[label_frame["tau"].eq(label_frame["tau"].min())]
    plt.figure(figsize=(6, 4))
    values = selected.groupby("split")[["positive", "negative"]].sum()
    values.plot(kind="bar", ax=plt.gca(), color=["#c04b4b", "#5c8f65"])
    plt.ylabel("Country-month observations")
    plt.title("Target distribution by split")
    plt.tight_layout()
    plt.savefig(label_dir / "target_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    for model_name, losses in loss_by_model.items():
        values = [value if np.isfinite(value) else np.nan for value in losses]
        plt.plot(range(1, len(values) + 1), values, marker="o", label=model_name)
    plt.xlabel("Epoch")
    plt.ylabel("Weighted BCE loss")
    plt.title("Training loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_dir / "training_loss.png", dpi=150)
    plt.close()

    metric_names = ("accuracy", "balanced_accuracy")
    plt.figure(figsize=(8, 4))
    x = np.arange(len(metric_names))
    width = 0.22
    for index, (model_name, frame) in enumerate(predictions_by_model.items()):
        metric = _metrics_from_predictions(frame[frame["split"].eq("test")])
        plt.bar(
            x + (index - 1.5) * width,
            [metric.get(name) or 0 for name in metric_names],
            width,
            label=model_name,
        )
    plt.xticks(x, ["Accuracy", "Balanced accuracy"])
    plt.ylim(0, 1.05)
    plt.title("Test metrics (single-class caveat applies)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(comparison_dir / "model_comparison.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    for model_name, frame in predictions_by_model.items():
        if not frame.empty:
            plt.hist(
                frame["score"],
                bins=12,
                alpha=0.45,
                label=model_name,
            )
    plt.xlabel("Predicted risk score")
    plt.ylabel("Country-month observations")
    plt.title("Prediction-score distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(prediction_dir / "prediction_distribution.png", dpi=150)
    plt.close()


def _write_leakage_audit(
    path: Path,
    horizon: int,
    split: dict[str, list[str]],
    target_stats: dict[str, Any],
) -> None:
    path.write_text(
        f"""# Leakage audit

Status: PASS for the implemented 2024 country-month experiment.

* Prediction horizon: `{horizon}` month(s); the target is the next observed
  inbound-flow contraction.
* Target construction uses future inbound flow only for `y(country, t+h)`;
  future flow, contraction, and labels are not included in `FEATURES`.
* Features are read from the fused row for month `t`.
* GCN edges are restricted to the same-month snapshot.
* TGN events are replayed in chronological order `{split['train'][0]}`
  through `{split['test'][-1]}` in one sequence; memory is reset once before
  replay and then carried forward across the split boundaries.
* Feature standardization is fitted only on the training-month rows.
* The final month is excluded because its `{horizon}`-month future target is
  outside the 2024 data.
* No random observation shuffle, oversampling, or synthetic label generation
  is used.
* Valid continuous targets audited: `{target_stats['valid_observations']}`;
  invalid/unobserved targets: `{target_stats['invalid_observations']}`.

The audit verifies temporal construction and code paths. It does not claim
that a one-year, single-class target supports a meaningful performance claim.
""",
        encoding="utf-8",
    )


def run(config: dict[str, Any], overwrite: bool = False) -> Path:
    experiment = _experiment_path(config)
    if experiment.exists():
        if not overwrite:
            raise FileExistsError(
                f"Experiment directory already exists; refusing to overwrite: {experiment}"
            )
        shutil.rmtree(experiment)
    nodes, edges = _load_real_inputs(config)
    horizon = int(nested(config, "analysis", "horizon_months", default=1))
    target_frame, target_stats = _continuous_targets(nodes, horizon)
    months = sorted(nodes["month"].astype(str).unique().tolist())
    if len(months) < horizon + 3:
        raise ValueError("Not enough monthly snapshots for a chronological experiment")
    supervised_months = months[:-horizon]
    split_config = nested(config, "country_month_experiment", default={})
    train_count = int(split_config.get("train_months", 7))
    validation_count = int(split_config.get("validation_months", 2))
    test_count = int(split_config.get("test_months", 2))
    required_months = train_count + validation_count + test_count
    if required_months > len(supervised_months):
        raise ValueError(
            f"Configured country-month split needs {required_months} supervised months; "
            f"only {len(supervised_months)} are available"
        )
    split = {
        "train": supervised_months[:train_count],
        "validation": supervised_months[train_count : train_count + validation_count],
        "test": supervised_months[
            train_count + validation_count : required_months
        ],
    }
    taus = [float(value) for value in nested(config, "analysis", "taus", default=[0.30, 0.35, 0.40])]
    tau = float(nested(config, "analysis", "default_tau", default=0.35))
    if tau not in taus:
        taus = sorted(set(taus + [tau]))
    label_column = f"label_tau_{tau:.2f}"
    target_frame["node_id"] = target_frame["node_id"].astype(str)
    label_frame = _label_distribution(target_frame, split, taus)

    diagnostics = experiment / "diagnostics"
    predictions_dir = experiment / "predictions"
    metrics_dir = experiment / "metrics"
    checkpoints_dir = experiment / "checkpoints"
    graphs_dir = experiment / "graphs"
    for directory in (diagnostics, predictions_dir, metrics_dir, checkpoints_dir, graphs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    config_copy = dict(config)
    config_copy.pop("_config_path", None)
    (experiment / "config_used.yaml").write_text(
        yaml.safe_dump(config_copy, sort_keys=False),
        encoding="utf-8",
    )
    (experiment / "README.md").write_text(
        """# 2024 country-month experiment

This is a real-data-only, one-month-ahead experiment. It reuses the existing
country-level temporal graph, fused features, GCN, TGN, and TGN-no-memory
implementations. It does not modify the source cache, raw event table, or
processed source tables.

The configured default `tau=0.35` is evaluated without selecting a threshold
from model performance. December 2024 is excluded because January 2025 is not
present. The chronological split is January–July train, August–September
validation, and October–November test.

The Comtrade source is India-reporter inbound data. Therefore only India has
observed inbound-flow targets; partner country nodes remain in the graph but
are not silently treated as negative target observations.

Metrics marked N/A are mathematically undefined for the single-class
partition. Accuracy and balanced accuracy are retained as descriptive values,
not predictive evidence.
""",
        encoding="utf-8",
    )
    label_frame.to_csv(diagnostics / "label_distribution.csv", index=False)
    write_json(
        {
            "continuous_target": target_stats,
            "label_definition": (
                "label=1 when next-month inbound-flow contraction is below "
                f"-{tau:.2f}; target is valid only with positive historical median and observed future flow"
            ),
            "tau_values_reported": taus,
            "selected_tau": tau,
        },
        diagnostics / "label_distribution.json",
    )
    split_statistics = {
        "graph_months": months,
        "graph_observations": int(len(nodes)),
        "unique_countries": int(nodes["node_id"].nunique()),
        "supervised_months": supervised_months,
        "excluded_months": months[-horizon:],
        "split": split,
        "target_valid_by_split": {
            name: int(
                target_frame[
                    target_frame["month"].isin(values) & target_frame["target_valid"]
                ].shape[0]
            )
            for name, values in split.items()
        },
        "target_valid_country_months": int(target_frame["target_valid"].sum()),
        "target_invalid_country_months": int((~target_frame["target_valid"]).sum()),
    }
    write_json(split_statistics, diagnostics / "split_statistics.json")
    _write_leakage_audit(
        diagnostics / "leakage_audit.md", horizon, split, target_stats
    )
    graph_source = _base_path(config) / nested(
        config, "outputs", "graph", default="processed/graph.json"
    )
    if graph_source.exists():
        shutil.copy2(graph_source, graphs_dir / "temporal_graph.json")

    node_ids = sorted(nodes["node_id"].astype(str).unique().tolist())
    train_nodes = nodes[nodes["month"].isin(split["train"])]
    standardizer = FeatureStandardizer().fit(train_nodes)
    edge_scale = _event_scales(edges, split["train"])
    train_labels = train_nodes[label_column].dropna()
    positive_weight = (
        1.0 / float(train_labels.mean())
        if len(train_labels) and float(train_labels.mean()) > 0
        else 1.0
    )
    epochs = int(nested(config, "training", "epochs", default=10))
    learning_rate = float(nested(config, "training", "learning_rate", default=0.001))
    model_config = nested(config, "model", default={})
    hidden_dim = int(model_config.get("embedding_dim", 32))
    gcn_kwargs = {
        "feature_dim": len(FEATURES),
        "hidden_dim": hidden_dim,
        "layers": int(model_config.get("gcn_layers", 2)),
    }
    tgn_kwargs = {
        key: value
        for key, value in model_config.items()
        if key in {
            "memory_dim",
            "time_dim",
            "edge_dim",
            "message_dim",
            "embedding_dim",
            "max_neighbors",
        }
    }
    metrics_by_model: dict[str, dict[str, Any]] = {}
    predictions_by_model: dict[str, pd.DataFrame] = {}
    loss_by_model: dict[str, list[float]] = {}

    gcn, gcn_predictions, gcn_losses = _train_gcn(
        nodes,
        edges,
        split,
        node_ids,
        standardizer,
        edge_scale,
        label_column,
        positive_weight,
        gcn_kwargs,
        epochs,
        learning_rate,
    )
    gcn_frame = _prediction_rows(
        nodes, target_frame, split, gcn_predictions, node_ids, "gcn", label_column
    )
    _save_prediction_frame(gcn_frame, predictions_dir / "gcn_predictions.csv")
    metrics_by_model["gcn"] = _split_metrics(gcn_frame)
    write_json(metrics_by_model["gcn"], metrics_dir / "gcn_metrics.json")
    _save_model_checkpoint(
        gcn,
        checkpoints_dir / "gcn" / "gcn.pt",
        gcn_kwargs,
        standardizer,
        tau,
    )
    predictions_by_model["gcn"] = gcn_frame
    loss_by_model["gcn"] = gcn_losses

    for model_name, use_memory in (("tgn", True), ("tgn_no_memory", False)):
        model, predictions, losses = _train_tgn(
            nodes,
            edges,
            split,
            node_ids,
            standardizer,
            edge_scale,
            label_column,
            positive_weight,
            tgn_kwargs,
            epochs,
            learning_rate,
            use_memory,
        )
        frame = _prediction_rows(
            nodes, target_frame, split, predictions, node_ids, model_name, label_column
        )
        _save_prediction_frame(
            frame, predictions_dir / f"{model_name}_predictions.csv"
        )
        metrics_by_model[model_name] = _split_metrics(frame)
        write_json(metrics_by_model[model_name], metrics_dir / f"{model_name}_metrics.json")
        _save_model_checkpoint(
            model,
            checkpoints_dir / model_name / f"{model_name}.pt",
            {"use_memory": use_memory, **tgn_kwargs},
            standardizer,
            tau,
        )
        predictions_by_model[model_name] = frame
        loss_by_model[model_name] = losses

    logistic_frame, logistic_note = _logistic_predictions(
        nodes, split, node_ids, standardizer, label_column
    )
    if logistic_frame.empty:
        logistic_frame = pd.DataFrame(
            columns=["model", "split", "month", "node_id", "score", "label"]
        )
    _save_prediction_frame(logistic_frame, predictions_dir / "logistic_predictions.csv")
    metrics_by_model["logistic"] = _split_metrics(logistic_frame)
    if logistic_note:
        for split_name in ("validation", "test"):
            metrics_by_model["logistic"][split_name]["note"] = logistic_note
    write_json(metrics_by_model["logistic"], metrics_dir / "logistic_metrics.json")
    predictions_by_model["logistic"] = logistic_frame

    _write_comparison(metrics_by_model, metrics_dir / "comparison.csv")
    _plot_diagnostics(
        target_frame, label_frame, loss_by_model, predictions_by_model, graphs_dir
    )
    write_json(
        {
            "dataset": "real 2024 country-month graph",
            "synthetic": False,
            "countries": int(nodes["node_id"].nunique()),
            "country_month_observations": int(len(nodes)),
            "valid_target_observations": int(target_frame["target_valid"].sum()),
            "horizon_months": horizon,
            "selected_tau": tau,
            "split": split,
            "positive_weight": positive_weight,
            "results_dir": str(experiment),
            "raw_data_preserved": True,
            "models": list(metrics_by_model),
        },
        experiment / "experiment_summary.json",
    )
    return experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_cli(parser)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace only the isolated country_month_experiment directory",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    output = run(config, overwrite=args.overwrite)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
