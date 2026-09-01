"""Bilateral pair-month supervision for pooled directed trade links.

Each row is one observed importer ← exporter link at month ``T``. The target is
the one-month-ahead contraction of that bilateral flow, using the same baseline
rule as country-month training (Eq. 7). Node features from both endpoints and
the edge's own trade history enter the model inputs; targets never do.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from training.data import (
    FEATURES,
    FeatureStandardizer,
    Split,
    build_month_batches,
    edge_scales,
    resolve_split,
    supervised_months,
    validate_tables,
)
from training import paths  # noqa: F401

LOGGER = logging.getLogger("training.pair_data")

PAIR_PREDICTION_COLUMNS = [
    "model",
    "split",
    "month",
    "source",
    "destination",
    "target",
    "score",
]

EDGE_FEATURE_NAMES = ("edge_log_trade", "edge_log_volume")


@dataclass
class PairMonthBatch:
    """One monthly snapshot with graph structure and supervised edge rows."""

    month: str
    time: float
    features: torch.Tensor
    edge_index: torch.Tensor
    events: list[dict[str, Any]]
    sup_src: torch.Tensor
    sup_dst: torch.Tensor
    sup_edge_feat: torch.Tensor
    sup_target: torch.Tensor
    sup_mask: torch.Tensor
    pair_keys: list[tuple[str, str]]

    @property
    def valid_count(self) -> int:
        return int(self.sup_mask.sum().item())


def aggregate_edges(edges: pd.DataFrame) -> pd.DataFrame:
    """Sum commodity rows into one source-destination-month trade cell."""
    if edges.empty:
        return pd.DataFrame(
            columns=["month", "source", "destination", "trade_value_usd", "flow_volume"]
        )
    frame = edges.copy()
    frame["month"] = frame["month"].astype(str)
    frame["source"] = frame["source"].astype(str)
    frame["destination"] = frame["destination"].astype(str)
    frame["trade_value_usd"] = pd.to_numeric(frame["trade_value_usd"], errors="coerce").fillna(0.0)
    frame["flow_volume"] = pd.to_numeric(frame.get("flow_volume", 0), errors="coerce").fillna(0.0)
    return (
        frame.groupby(["month", "source", "destination"], as_index=False)
        .agg(trade_value_usd=("trade_value_usd", "sum"), flow_volume=("flow_volume", "sum"))
        .sort_values(["source", "destination", "month"])
        .reset_index(drop=True)
    )


def build_pair_targets(
    edges: pd.DataFrame,
    *,
    horizon: int = 1,
    baseline_window: int = 12,
    baseline_min_periods: int = 12,
    taus: Sequence[float] = (0.20, 0.30, 0.35, 0.40),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Derive bilateral contraction targets on aggregated directed trade flows."""
    frame = aggregate_edges(edges)
    if frame.empty:
        raise ValueError("Cannot build pair targets from an empty edge table.")

    grouped = frame.groupby(["source", "destination"], sort=False)
    future = grouped["trade_value_usd"].shift(-horizon)
    baseline = grouped["trade_value_usd"].transform(
        lambda series: series.shift(-(horizon - 1))
        .rolling(baseline_window, min_periods=baseline_min_periods)
        .median()
    )
    contraction = (future - baseline) / baseline.replace(0, np.nan)
    frame["future_trade_value_usd"] = future
    frame["baseline_trade_value_usd"] = baseline
    frame["contraction"] = contraction
    frame["target_valid"] = (
        future.notna() & baseline.notna() & baseline.gt(0) & contraction.notna()
    )
    for tau in taus:
        key = f"{float(tau):.2f}"
        frame[f"derived_label_tau_{key}"] = np.where(
            frame["target_valid"],
            (contraction < -float(tau)).astype(float),
            np.nan,
        )

    valid = frame.loc[frame["target_valid"], "contraction"].astype(float)
    stats = {
        "supervision_unit": "directed_pair_month",
        "horizon_months": int(horizon),
        "baseline_window_months": int(baseline_window),
        "baseline_min_periods": int(baseline_min_periods),
        "pair_month_rows": int(len(frame)),
        "valid_targets": int(frame["target_valid"].sum()),
        "unique_directed_pairs": int(frame[["source", "destination"]].drop_duplicates().shape[0]),
        "contraction": {
            "count": int(len(valid)),
            "min": float(valid.min()) if len(valid) else None,
            "max": float(valid.max()) if len(valid) else None,
            "mean": float(valid.mean()) if len(valid) else None,
            "median": float(valid.median()) if len(valid) else None,
            "std": float(valid.std(ddof=0)) if len(valid) else None,
        },
        "positives_by_tau": {
            f"{float(tau):.2f}": int(np.nansum(frame[f"derived_label_tau_{float(tau):.2f}"].to_numpy()))
            for tau in taus
        },
    }
    return frame, stats


def attach_node_features(pair_frame: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    """Join source and destination country-month features onto each pair row."""
    nodes = nodes.copy()
    nodes["month"] = nodes["month"].astype(str)
    nodes["node_id"] = nodes["node_id"].astype(str)
    node_cols = ["month", "node_id", *FEATURES]
    lookup = nodes[node_cols].drop_duplicates(["month", "node_id"])
    src = lookup.rename(columns={"node_id": "source", **{feature: f"src_{feature}" for feature in FEATURES}})
    dst = lookup.rename(columns={"node_id": "destination", **{feature: f"dst_{feature}" for feature in FEATURES}})
    result = pair_frame.merge(src, on=["month", "source"], how="left")
    result = result.merge(dst, on=["month", "destination"], how="left")
    missing = result[[f"src_{feature}" for feature in FEATURES]].isna().any(axis=1).sum()
    if missing:
        LOGGER.warning("%d pair-month row(s) are missing node features on the source side.", int(missing))
    return result


def pair_feature_columns() -> list[str]:
    return [f"src_{name}" for name in FEATURES] + [f"dst_{name}" for name in FEATURES] + list(EDGE_FEATURE_NAMES)


class PairFeatureStandardizer:
    """Train-only scaler for the concatenated pair feature vector."""

    def __init__(self, features: Sequence[str] | None = None):
        self.features = list(features or pair_feature_columns())
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame) -> PairFeatureStandardizer:
        values = frame[self.features].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        self.mean = values.mean(axis=0)
        self.scale = values.std(axis=0, ddof=0)
        self.scale[self.scale < 1e-8] = 1.0
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.mean is None or self.scale is None:
            raise RuntimeError("PairFeatureStandardizer.fit must be called before transform.")
        values = frame[self.features].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        return ((values - self.mean) / self.scale).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        if self.mean is None or self.scale is None:
            return {}
        return {
            "features": self.features,
            "mean": [float(value) for value in self.mean],
            "scale": [float(value) for value in self.scale],
        }


def enrich_pair_features(
    pair_frame: pd.DataFrame,
    edge_scale: tuple[float, float],
) -> pd.DataFrame:
    """Add normalised log edge trade features used by graph link heads."""
    result = pair_frame.copy()
    trade = np.log1p(
        pd.to_numeric(result["trade_value_usd"], errors="coerce").fillna(0).clip(lower=0).to_numpy()
    )
    volume = np.log1p(
        pd.to_numeric(result["flow_volume"], errors="coerce").fillna(0).clip(lower=0).to_numpy()
    )
    value_scale, volume_scale = edge_scale
    result["edge_log_trade"] = trade / max(value_scale, 1.0)
    result["edge_log_volume"] = volume / max(volume_scale, 1.0)
    return result


def build_pair_month_batches(
    pair_frame: pd.DataFrame,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    months: Sequence[str],
    node_ids: Sequence[str],
    node_standardizer: FeatureStandardizer,
    edge_scale: tuple[float, float],
    target_column: str,
    device: torch.device,
) -> dict[str, PairMonthBatch]:
    """Materialise monthly graph tensors plus supervised edge rows."""
    node_batches = build_month_batches(
        nodes,
        edges,
        months,
        node_ids,
        node_standardizer,
        edge_scale,
        target_column="contraction",
        device=device,
    )
    index = {node_id: position for position, node_id in enumerate(node_ids)}
    pair_frame = pair_frame.copy()
    pair_frame["month"] = pair_frame["month"].astype(str)
    by_month = {month: group for month, group in pair_frame.groupby("month", sort=False)}
    value_scale, volume_scale = edge_scale
    batches: dict[str, PairMonthBatch] = {}

    for month in months:
        node_batch = node_batches[month]
        month_pairs = by_month.get(month)
        if month_pairs is None or month_pairs.empty:
            empty = torch.empty(0, dtype=torch.long, device=device)
            batches[month] = PairMonthBatch(
                month=month,
                time=node_batch.time,
                features=node_batch.features,
                edge_index=node_batch.edge_index,
                events=node_batch.events,
                sup_src=empty,
                sup_dst=empty,
                sup_edge_feat=torch.empty((0, 2), dtype=torch.float32, device=device),
                sup_target=torch.empty(0, dtype=torch.float32, device=device),
                sup_mask=torch.empty(0, dtype=torch.bool, device=device),
                pair_keys=[],
            )
            continue

        supervised = month_pairs[
            month_pairs["target_valid"].fillna(False).astype(bool)
            & month_pairs[target_column].notna()
        ]
        src_idx: list[int] = []
        dst_idx: list[int] = []
        edge_feats: list[list[float]] = []
        targets: list[float] = []
        keys: list[tuple[str, str]] = []
        for _, row in supervised.iterrows():
            source = str(row["source"])
            destination = str(row["destination"])
            source_index = index.get(source)
            destination_index = index.get(destination)
            if source_index is None or destination_index is None:
                continue
            trade = float(
                np.log1p(max(float(row.get("trade_value_usd", 0.0) or 0.0), 0.0)) / max(value_scale, 1.0)
            )
            volume = float(
                np.log1p(max(float(row.get("flow_volume", 0.0) or 0.0), 0.0)) / max(volume_scale, 1.0)
            )
            src_idx.append(source_index)
            dst_idx.append(destination_index)
            edge_feats.append([trade, volume])
            targets.append(float(row[target_column]))
            keys.append((source, destination))

        if src_idx:
            sup_src = torch.tensor(src_idx, dtype=torch.long, device=device)
            sup_dst = torch.tensor(dst_idx, dtype=torch.long, device=device)
            sup_edge_feat = torch.tensor(edge_feats, dtype=torch.float32, device=device)
            sup_target = torch.tensor(targets, dtype=torch.float32, device=device)
            sup_mask = torch.ones(len(targets), dtype=torch.bool, device=device)
        else:
            empty_long = torch.empty(0, dtype=torch.long, device=device)
            sup_src = sup_dst = empty_long
            sup_edge_feat = torch.empty((0, 2), dtype=torch.float32, device=device)
            sup_target = torch.empty(0, dtype=torch.float32, device=device)
            sup_mask = torch.empty(0, dtype=torch.bool, device=device)

        batches[month] = PairMonthBatch(
            month=month,
            time=node_batch.time,
            features=node_batch.features,
            edge_index=node_batch.edge_index,
            events=node_batch.events,
            sup_src=sup_src,
            sup_dst=sup_dst,
            sup_edge_feat=sup_edge_feat,
            sup_target=sup_target,
            sup_mask=sup_mask,
            pair_keys=keys,
        )
    return batches


def pair_supervised_months(pair_frame: pd.DataFrame, months: Sequence[str]) -> list[str]:
    valid = pair_frame[pair_frame["target_valid"].fillna(False).astype(bool)]
    present = set(valid["month"].astype(str).unique().tolist())
    return [month for month in months if month in present]


def pair_positive_weight(pair_frame: pd.DataFrame, months: Sequence[str], label_column: str) -> float:
    subset = pair_frame[
        pair_frame["month"].isin(list(months))
        & pair_frame["target_valid"].fillna(False).astype(bool)
        & pair_frame[label_column].notna()
    ]
    labels = pd.to_numeric(subset[label_column], errors="coerce").dropna()
    if labels.empty:
        return 1.0
    prevalence = float(labels.mean())
    if prevalence <= 0.0:
        return 1.0
    return float(min(1.0 / prevalence, 1e4))


def pair_class_summary(pair_frame: pd.DataFrame, split: Split, label_column: str) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name, months in split.as_dict().items():
        subset = pair_frame[
            pair_frame["month"].isin(months)
            & pair_frame["target_valid"].fillna(False).astype(bool)
            & pair_frame[label_column].notna()
        ]
        labels = pd.to_numeric(subset[label_column], errors="coerce").dropna()
        positives = int(labels.sum())
        summary[name] = {
            "months": months,
            "valid_rows": int(len(labels)),
            "positive": positives,
            "negative": int(len(labels) - positives),
            "prevalence": float(labels.mean()) if len(labels) else None,
            "single_class": bool(len(labels) and labels.nunique() < 2),
        }
    return summary


def pair_regression_summary(pair_frame: pd.DataFrame, split: Split) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name, months in split.as_dict().items():
        subset = pair_frame[
            pair_frame["month"].isin(months) & pair_frame["target_valid"].fillna(False).astype(bool)
        ]
        values = pd.to_numeric(subset["contraction"], errors="coerce").dropna()
        summary[name] = {
            "months": months,
            "valid_rows": int(len(values)),
            "mean": float(values.mean()) if len(values) else None,
            "std": float(values.std(ddof=0)) if len(values) else None,
            "min": float(values.min()) if len(values) else None,
            "max": float(values.max()) if len(values) else None,
        }
    return summary


def assign_splits(pair_frame: pd.DataFrame, split: Split) -> pd.DataFrame:
    """Attach a partition name to every pair-month row."""
    result = pair_frame.copy()
    mapping: dict[str, str] = {}
    for name, months in split.as_dict().items():
        for month in months:
            mapping[str(month)] = name
    result["split"] = result["month"].astype(str).map(mapping)
    return result
