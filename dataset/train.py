"""Train and compare chronological supply-chain risk models.

The default split is forward chained: the first 36 months for training, the
next 12 for validation, and the final 12 for testing. Events are processed
strictly by month, model memory is reset at every fold boundary, and all
feature normalization statistics are fit on the training partition only.

This script reports directional results, not a conclusive benchmark. With
only sixty monthly steps, a single event or label can materially move a
metric. It also logs the train/serve distribution caveat: the historical
GDELT source and the deployed SERP/scrape news path are not guaranteed to
share a feature distribution.

The comparison includes a snapshot GCN baseline. GCN receives the same
monthly node features and same-month bidirectional trade adjacency, but has no
cross-month memory, directly testing whether TGN temporal memory earns
improvement over graph convolution alone.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, roc_auc_score

from common import add_common_cli, load_config, nested, read_table, write_json
from fuse_dataset import FEATURES, build_fused_tables, _synthetic_inputs
from model_gcn import SnapshotGCN, undirected_edge_index
from model_tgn import DEFAULT_LINEAR_WEIGHTS, TemporalGraphNetwork, linear_composition


@dataclass
class FeatureStandardizer:
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame) -> "FeatureStandardizer":
        values = frame[FEATURES].astype(float).to_numpy(copy=True)
        values[:, FEATURES.index("inventory_days_proxy")] *= -1
        self.mean = values.mean(axis=0)
        self.scale = values.std(axis=0)
        self.scale[self.scale < 1e-8] = 1.0
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.mean is None or self.scale is None:
            raise RuntimeError("FeatureStandardizer must be fit on training data first")
        values = frame[FEATURES].astype(float).to_numpy(copy=True)
        values[:, FEATURES.index("inventory_days_proxy")] *= -1
        return (values - self.mean) / self.scale

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()} if self.mean is not None else {}


def _load_inputs(config: dict, synthetic: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = Path(__file__).resolve().parent
    if synthetic:
        inputs = _synthetic_inputs(config)
        nodes, edges = build_fused_tables(
            config, inputs[0], gdelt=inputs[3], weather=inputs[4], gscpi=inputs[5]
        )
        return nodes, edges
    nodes = read_table(base / nested(config, "outputs", "nodes", default="processed/nodes_monthly.csv"))
    edges = read_table(base / nested(config, "outputs", "edges", default="processed/edges_monthly.csv"))
    return nodes, edges


def _results_dir(config: dict) -> Path:
    base = Path(__file__).resolve().parent
    configured = nested(config, "outputs", "results_dir")
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else base / path
    metrics = Path(
        nested(config, "outputs", "training_metrics", default="processed/training_metrics.json")
    )
    return (metrics if metrics.is_absolute() else base / metrics).parent


def _prediction_scores(
    predictions: dict[str, np.ndarray], node_ids: list[str]
) -> dict[str, dict[str, float]]:
    return {
        month: {
            node_id: float(score)
            for node_id, score in zip(node_ids, values.tolist())
        }
        for month, values in predictions.items()
    }


def _splits(months: list[str], config: dict) -> dict[str, list[str]]:
    train_end = int(nested(config, "training", "train_months", default=8))
    validation_end = train_end + int(nested(config, "training", "validation_months", default=12))
    test_months = int(
        nested(config, "training", "test_months", default=len(months) - validation_end)
    )
    return {
        "train": months[:train_end],
        "validation": months[train_end:validation_end],
        "test": months[validation_end:validation_end + test_months],
    }


def _prevalence(nodes: pd.DataFrame, months: Iterable[str], label_col: str) -> dict[str, Any]:
    subset = nodes[nodes["month"].isin(list(months))][label_col].dropna()
    positives = int(subset.sum()) if len(subset) else 0
    return {
        "n": int(len(subset)),
        "positive": positives,
        "negative": int(len(subset) - positives),
        "prevalence": float(subset.mean()) if len(subset) else None,
    }


def _prepare_month_data(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    month: str,
    node_ids: list[str],
    standardizer: FeatureStandardizer,
    edge_scale: tuple[float, float],
    label_col: str,
) -> tuple[torch.Tensor, list[dict[str, Any]], np.ndarray]:
    current = nodes[nodes["month"] == month].copy()
    current = current.drop_duplicates("node_id").set_index("node_id")
    ordered = pd.DataFrame(index=node_ids)
    for feature in FEATURES:
        ordered[feature] = current[feature] if feature in current else 0.0
    ordered = ordered.fillna(0.0)
    features = torch.tensor(standardizer.transform(ordered.reset_index()), dtype=torch.float32)
    index = {node_id: position for position, node_id in enumerate(node_ids)}
    events: list[dict[str, Any]] = []
    month_edges = edges[edges["month"] == month] if not edges.empty else pd.DataFrame()
    value_scale, volume_scale = edge_scale
    for _, row in month_edges.iterrows():
        source, destination = str(row["source"]), str(row["destination"])
        if source not in index or destination not in index:
            continue
        events.append(
            {
                "source_index": index[source],
                "destination_index": index[destination],
                "time": float(pd.Period(month, freq="M").ordinal),
                "time_delta": 1.0,
                "edge_features": [
                    math.log1p(max(0.0, float(row.get("trade_value_usd", 0.0)))) / value_scale,
                    math.log1p(max(0.0, float(row.get("flow_volume", 0.0)))) / volume_scale,
                ],
            }
        )
    labels = np.full(len(node_ids), np.nan, dtype=float)
    if not current.empty and label_col in current:
        for node_id, value in current[label_col].items():
            if node_id in index and pd.notna(value):
                labels[index[node_id]] = float(value)
    return features, events, labels


def _event_scales(edges: pd.DataFrame, train_months: Iterable[str]) -> tuple[float, float]:
    subset = edges[edges["month"].isin(list(train_months))] if not edges.empty else edges
    if subset.empty:
        return 1.0, 1.0
    values = np.log1p(pd.to_numeric(subset.get("trade_value_usd", 0), errors="coerce").fillna(0).clip(lower=0))
    volumes = np.log1p(pd.to_numeric(subset.get("flow_volume", 0), errors="coerce").fillna(0).clip(lower=0))
    return max(float(values.max()), 1.0), max(float(volumes.max()), 1.0)


def _class_weight(nodes: pd.DataFrame, months: Iterable[str], label_col: str) -> float:
    prevalence = _prevalence(nodes, months, label_col)["prevalence"]
    return float(1.0 / float(prevalence)) if prevalence and prevalence > 0 else 1.0


def _weighted_bce(probabilities: torch.Tensor, labels: torch.Tensor, positive_weight: float) -> Tensor:
    probabilities = probabilities.clamp(1e-6, 1.0 - 1e-6)
    weights = torch.where(labels > 0.5, torch.tensor(positive_weight, device=labels.device), torch.tensor(1.0, device=labels.device))
    return (-(weights * (labels * torch.log(probabilities) + (1.0 - labels) * torch.log(1.0 - probabilities)))).mean()


def _run_tgn(
    model: TemporalGraphNetwork,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    months: list[str],
    node_ids: list[str],
    standardizer: FeatureStandardizer,
    edge_scale: tuple[float, float],
    label_col: str,
    positive_weight: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[dict[str, np.ndarray], list[float]]:
    model.reset_memory(len(node_ids))
    predictions: dict[str, np.ndarray] = {}
    losses: list[float] = []
    for month in months:
        feature_tensor, events, labels = _prepare_month_data(
            nodes, edges, month, node_ids, standardizer, edge_scale, label_col
        )
        probabilities = model(feature_tensor, events=events, current_time=float(pd.Period(month, freq="M").ordinal))
        predictions[month] = probabilities.detach().cpu().numpy()
        mask = ~np.isnan(labels)
        if optimizer is not None and mask.any():
            target = torch.tensor(labels[mask], dtype=torch.float32)
            loss = _weighted_bce(probabilities[torch.tensor(mask)], target, positive_weight)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
    return predictions, losses


def _run_gcn(
    model: SnapshotGCN,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    months: list[str],
    node_ids: list[str],
    standardizer: FeatureStandardizer,
    edge_scale: tuple[float, float],
    label_col: str,
    positive_weight: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[dict[str, np.ndarray], list[float]]:
    predictions: dict[str, np.ndarray] = {}
    losses: list[float] = []
    for month in months:
        feature_tensor, events, labels = _prepare_month_data(
            nodes, edges, month, node_ids, standardizer, edge_scale, label_col
        )
        edge_index = undirected_edge_index(events, feature_tensor.device)
        probabilities = model(feature_tensor, edge_index)
        predictions[month] = probabilities.detach().cpu().numpy()
        mask = ~np.isnan(labels)
        if optimizer is not None and mask.any():
            target = torch.tensor(labels[mask], dtype=torch.float32)
            loss = _weighted_bce(probabilities[torch.tensor(mask)], target, positive_weight)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
    return predictions, losses


def _evaluate_predictions(
    nodes: pd.DataFrame, months: list[str], node_ids: list[str], predictions: dict[str, np.ndarray], label_col: str
) -> dict[str, Any]:
    labels, scores = [], []
    index = {node_id: position for position, node_id in enumerate(node_ids)}
    for month in months:
        current = nodes[nodes["month"] == month].drop_duplicates("node_id").set_index("node_id")
        for node_id, row in current.iterrows():
            value = row.get(label_col)
            if pd.notna(value) and month in predictions and node_id in index:
                labels.append(float(value))
                scores.append(float(predictions[month][index[node_id]]))
    if not labels:
        return {"n": 0, "positive": 0, "prevalence": None}
    y_true = np.asarray(labels, dtype=int)
    y_score = np.asarray(scores)
    y_hat = (y_score >= 0.5).astype(int)
    result: dict[str, Any] = {
        "n": int(len(y_true)),
        "positive": int(y_true.sum()),
        "prevalence": float(y_true.mean()),
        "accuracy": float(accuracy_score(y_true, y_hat)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(y_true, y_hat))
            if len(np.unique(y_true)) > 1
            else float(accuracy_score(y_true, y_hat))
        ),
    }
    if len(np.unique(y_true)) > 1:
        result["roc_auc"] = float(roc_auc_score(y_true, y_score))
        result["average_precision"] = float(average_precision_score(y_true, y_score))
    return result


def _baseline_metrics(nodes: pd.DataFrame, split: list[str], train_prevalence: float, label_col: str) -> dict[str, Any]:
    values = nodes[nodes["month"].isin(split)][label_col].dropna().astype(int).to_numpy()
    if not len(values):
        return {"n": 0, "positive": 0, "prevalence": None}
    score = np.full(len(values), train_prevalence)
    prediction = (score >= 0.5).astype(int)
    return {
        "n": int(len(values)),
        "positive": int(values.sum()),
        "prevalence": float(values.mean()),
        "accuracy": float(accuracy_score(values, prediction)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(values, prediction))
            if len(np.unique(values)) > 1
            else float(accuracy_score(values, prediction))
        ),
    }


def _logistic_metrics(
    nodes: pd.DataFrame,
    train_months: list[str],
    eval_months: list[str],
    standardizer: FeatureStandardizer,
    label_col: str,
) -> dict[str, Any]:
    train = nodes[nodes["month"].isin(train_months) & nodes[label_col].notna()]
    evaluate = nodes[nodes["month"].isin(eval_months) & nodes[label_col].notna()]
    if train.empty or evaluate.empty or train[label_col].nunique() < 2:
        return {"n": int(len(evaluate)), "note": "logistic regression unavailable: training labels have one class"}
    classifier = LogisticRegression(class_weight="balanced", max_iter=500, random_state=7)
    classifier.fit(standardizer.transform(train), train[label_col].astype(int))
    score = classifier.predict_proba(standardizer.transform(evaluate))[:, 1]
    result = _evaluate_arrays(evaluate[label_col].astype(int).to_numpy(), score)
    return result


def _evaluate_arrays(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    prediction = (scores >= 0.5).astype(int)
    result: dict[str, Any] = {
        "n": int(len(labels)),
        "positive": int(labels.sum()),
        "prevalence": float(labels.mean()) if len(labels) else None,
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(labels, prediction))
            if len(np.unique(labels)) > 1
            else float(accuracy_score(labels, prediction))
        ),
    }
    if len(np.unique(labels)) > 1:
        result["roc_auc"] = float(roc_auc_score(labels, scores))
        result["average_precision"] = float(average_precision_score(labels, scores))
    return result


def train(config: dict, synthetic: bool = False) -> dict[str, Any]:
    seed = int(nested(config, "training", "seed", default=7))
    np.random.seed(seed)
    torch.manual_seed(seed)
    results_dir = _results_dir(config)
    results_dir.mkdir(parents=True, exist_ok=True)
    nodes, edges = _load_inputs(config, synthetic)
    months = sorted(nodes["month"].astype(str).unique().tolist())
    split = _splits(months, config)
    expected_months = (
        int(nested(config, "training", "train_months", default=36))
        + int(nested(config, "training", "validation_months", default=12))
        + int(nested(config, "training", "test_months", default=12))
    )
    if len(months) < expected_months:
        print(
            f"WARNING: {len(months)} monthly steps are available but "
            f"{expected_months} are configured; results are directional."
        )
    node_ids = sorted(nodes["node_id"].astype(str).unique().tolist())
    train_nodes = nodes[nodes["month"].isin(split["train"])]
    standardizer = FeatureStandardizer().fit(train_nodes)
    edge_scale = _event_scales(edges, split["train"])
    epochs = int(nested(config, "training", "epochs", default=10))
    learning_rate = float(nested(config, "training", "learning_rate", default=0.001))
    model_kwargs = nested(config, "model", default={})
    tau_values = [float(value) for value in nested(config, "analysis", "taus", default=[0.30, 0.35, 0.40])]
    results: dict[str, Any] = {
        "caveat": (
            f"{len(months)} monthly steps are available; forward-chained metrics "
            "are directional, not conclusive."
        ),
        "train_serve_caveat": "GDELT training features and live SERP/API news features may have different distributions; compare them on a common held-out period before deployment.",
        "months": months,
        "split": split,
        "normalizer_fit_partition": "train",
        "results_dir": str(results_dir),
        "input_tables": {
            "nodes": str(
                Path(__file__).resolve().parent
                / nested(config, "outputs", "nodes", default="processed/nodes_monthly.csv")
            ),
            "edges": str(
                Path(__file__).resolve().parent
                / nested(config, "outputs", "edges", default="processed/edges_monthly.csv")
            ),
        },
        "synthetic": synthetic,
        "results": [],
    }
    score_artifacts: dict[str, Any] = {}

    for tau in tau_values:
        label_col = f"label_tau_{tau:.2f}"
        prevalence = {name: _prevalence(nodes, values, label_col) for name, values in split.items()}
        positive_weight = _class_weight(nodes, split["train"], label_col)
        print(f"\nTau={tau:.2f} label prevalence: {json.dumps(prevalence)}")
        print(f"Class-weighted BCE positive weight: {positive_weight:.3f}")
        tau_result: dict[str, Any] = {
            "tau": tau,
            "label_column": label_col,
            "label_prevalence": prevalence,
            "positive_weight": positive_weight,
            "models": {},
        }
        tau_scores: dict[str, Any] = {}
        train_prevalence = float(prevalence["train"]["prevalence"] or 0.0)
        for name in ("constant_prevalence", "hand_weighted_linear"):
            tau_result["models"][name] = {
                "validation": _baseline_metrics(nodes, split["validation"], train_prevalence, label_col),
                "test": _baseline_metrics(nodes, split["test"], train_prevalence, label_col),
            }
        tau_result["models"]["logistic_regression"] = {
            "validation": _logistic_metrics(nodes, split["train"], split["validation"], standardizer, label_col),
            "test": _logistic_metrics(nodes, split["train"], split["test"], standardizer, label_col),
        }

        # Use the explicit linear composition as the hand-weighted baseline.
        def linear_eval(eval_months: list[str]) -> dict[str, Any]:
            subset = nodes[nodes["month"].isin(eval_months) & nodes[label_col].notna()]
            scores = np.asarray([linear_composition(row.to_dict())[0] for _, row in subset.iterrows()])
            return _evaluate_arrays(subset[label_col].astype(int).to_numpy(), scores) if len(subset) else {"n": 0}

        tau_result["models"]["hand_weighted_linear"] = {
            "validation": linear_eval(split["validation"]),
            "test": linear_eval(split["test"]),
            "weights": DEFAULT_LINEAR_WEIGHTS,
        }

        gcn = SnapshotGCN(
            feature_dim=len(FEATURES),
            hidden_dim=int(model_kwargs.get("embedding_dim", 32)),
            layers=int(model_kwargs.get("gcn_layers", 2)),
        )
        gcn_optimizer = torch.optim.Adam(gcn.parameters(), lr=learning_rate)
        gcn_epoch_losses = []
        for _ in range(epochs):
            gcn.train()
            _, losses = _run_gcn(
                gcn, nodes, edges, split["train"], node_ids, standardizer, edge_scale,
                label_col, positive_weight, gcn_optimizer
            )
            gcn_epoch_losses.append(float(np.mean(losses)) if losses else None)
        gcn.eval()
        validation_predictions, _ = _run_gcn(
            gcn, nodes, edges, split["validation"], node_ids, standardizer, edge_scale,
            label_col, positive_weight
        )
        test_predictions, _ = _run_gcn(
            gcn, nodes, edges, split["test"], node_ids, standardizer, edge_scale,
            label_col, positive_weight
        )
        tau_result["models"]["gcn"] = {
            "validation": _evaluate_predictions(
                nodes, split["validation"], node_ids, validation_predictions, label_col
            ),
            "test": _evaluate_predictions(
                nodes, split["test"], node_ids, test_predictions, label_col
            ),
            "epoch_losses": gcn_epoch_losses,
            "note": "Snapshot GCN; no temporal memory and no cross-month state.",
        }
        tau_scores["gcn"] = {
            "validation": _prediction_scores(validation_predictions, node_ids),
            "test": _prediction_scores(test_predictions, node_ids),
        }
        gcn_checkpoint = Path(__file__).resolve().parent / nested(
            config, "outputs", "checkpoints_dir", default="checkpoints"
        ) / f"gcn_tau_{tau:.2f}.pt"
        gcn_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": gcn.state_dict(),
                "model_kwargs": {
                    "feature_dim": len(FEATURES),
                    "hidden_dim": int(model_kwargs.get("embedding_dim", 32)),
                    "layers": int(model_kwargs.get("gcn_layers", 2)),
                },
                "features": FEATURES,
                "tau": tau,
                "normalizer": standardizer.to_dict(),
            },
            gcn_checkpoint,
        )
        tau_result["models"]["gcn"]["checkpoint"] = str(gcn_checkpoint)

        for model_name, use_memory in (("tgn", True), ("tgn_no_memory", False)):
            model = TemporalGraphNetwork(
                feature_dim=len(FEATURES),
                use_memory=use_memory,
                **{key: value for key, value in model_kwargs.items() if key in {
                    "memory_dim", "time_dim", "edge_dim", "message_dim", "embedding_dim", "max_neighbors"
                }},
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            epoch_losses = []
            for _ in range(epochs):
                model.train()
                _, losses = _run_tgn(
                    model, nodes, edges, split["train"], node_ids, standardizer, edge_scale,
                    label_col, positive_weight, optimizer
                )
                epoch_losses.append(float(np.mean(losses)) if losses else None)
            model.eval()
            # Reset memory at the fold boundary, then replay train → validation
            # → test chronologically to preserve deployment-time state semantics.
            model.reset_memory(len(node_ids))
            train_predictions, _ = _run_tgn(
                model, nodes, edges, split["train"], node_ids, standardizer, edge_scale,
                label_col, positive_weight
            )
            validation_predictions, _ = _run_tgn(
                model, nodes, edges, split["validation"], node_ids, standardizer, edge_scale,
                label_col, positive_weight
            )
            test_predictions, _ = _run_tgn(
                model, nodes, edges, split["test"], node_ids, standardizer, edge_scale,
                label_col, positive_weight
            )
            # The separate calls above intentionally retain memory inside the
            # model, but collect predictions from each chronological segment.
            tau_result["models"][model_name] = {
                "validation": _evaluate_predictions(nodes, split["validation"], node_ids, validation_predictions, label_col),
                "test": _evaluate_predictions(nodes, split["test"], node_ids, test_predictions, label_col),
                "epoch_losses": epoch_losses,
            }
            tau_scores[model_name] = {
                "validation": _prediction_scores(validation_predictions, node_ids),
                "test": _prediction_scores(test_predictions, node_ids),
            }
            checkpoint = Path(__file__).resolve().parent / nested(
                config, "outputs", "checkpoints_dir", default="checkpoints"
            ) / f"{model_name}_tau_{tau:.2f}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_kwargs": {
                        "feature_dim": len(FEATURES),
                        "use_memory": use_memory,
                        **{key: value for key, value in model_kwargs.items() if key in {
                            "memory_dim", "time_dim", "edge_dim", "message_dim", "embedding_dim", "max_neighbors"
                        }},
                    },
                    "features": FEATURES,
                    "tau": tau,
                    "normalizer": standardizer.to_dict(),
                },
                checkpoint,
            )
            tau_result["models"][model_name]["checkpoint"] = str(checkpoint)
        score_artifacts[f"{tau:.2f}"] = tau_scores
        results["results"].append(tau_result)

    output = results_dir / "training_metrics.json"
    write_json(results, output)
    write_json(
        {
            "node_order": node_ids,
            "scores": score_artifacts,
            "score_semantics": "sigmoid risk scores; validation/test keys retain chronological month and node identity",
        },
        results_dir / "prediction_scores.json",
    )
    graph_source = Path(__file__).resolve().parent / nested(
        config, "outputs", "graph", default="processed/graph.json"
    )
    if graph_source.exists():
        shutil.copy2(graph_source, results_dir / "graph.json")
    print(f"\nWrote {output}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_cli(parser)
    parser.add_argument("--synthetic", action="store_true", help="Run on deterministic offline fused data")
    args = parser.parse_args()
    train(load_config(args.config), synthetic=args.synthetic)


if __name__ == "__main__":
    main()

