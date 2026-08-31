"""Load, validate, and tensorise the fused country-month tables.

Nothing in this module invents data. It reads the tables produced by
``fuse_dataset.py`` / ``build_graph.py``, re-derives the documented target from
``inbound_flow_usd``, cross-checks the re-derived labels against the labels
already stored in the table, and refuses to continue when the inputs do not
support the requested experiment.

Two rules are enforced structurally rather than by convention:

1. Anything fitted (feature standardisation, edge scaling, class weights,
   baseline statistics) is fitted on training months only.
2. A month with no observable future value is never turned into a negative
   example. It is dropped from the supervised pool and reported.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from training import paths

from common import nested, read_table  # noqa: E402  (requires the paths bootstrap)

try:  # Keep one definition of the feature contract for the whole repository.
    from fuse_dataset import FEATURES as _PIPELINE_FEATURES
except Exception:  # pragma: no cover - only if the flat module is unavailable
    _PIPELINE_FEATURES = None

FEATURES: list[str] = list(_PIPELINE_FEATURES) if _PIPELINE_FEATURES else [
    "inventory_days_proxy",
    "trade_delay_proxy",
    "news_vol_7d",
    "neg_tone_frac_3d",
    "strike_flag_7d",
    "weather_anomaly_7d",
    "global_risk",
]

# Higher inventory cover is protective, so it is sign-flipped before scaling.
# This mirrors ``model_tgn.orient_features`` and ``train.FeatureStandardizer``.
PROTECTIVE_FEATURE = "inventory_days_proxy"

REQUIRED_NODE_COLUMNS = ["month", "node_id", "host_country_id", "inbound_flow_usd", *FEATURES]
REQUIRED_EDGE_COLUMNS = ["month", "source", "destination"]

LOGGER = logging.getLogger("training.data")


class DataValidationError(RuntimeError):
    """Raised when the fused tables cannot support the requested training run."""


# --------------------------------------------------------------------------- #
# Loading and validation
# --------------------------------------------------------------------------- #
def table_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    return {
        "nodes": paths.resolve(nested(config, "outputs", "nodes", default="processed/nodes_monthly.csv")),
        "edges": paths.resolve(nested(config, "outputs", "edges", default="processed/edges_monthly.csv")),
        "graph": paths.resolve(nested(config, "outputs", "graph", default="processed/graph.json")),
    }


def load_tables(config: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the fused node and edge tables named by the config."""
    locations = table_paths(config)
    for name in ("nodes", "edges"):
        target = locations[name]
        if not target.exists() and not target.with_suffix(".csv").exists():
            raise DataValidationError(
                f"Missing {name} table: {target}\n"
                "Build it first with:\n"
                f"  python fuse_dataset.py --config {Path(str(config.get('_config_path', 'config.yaml'))).name}\n"
                f"  python build_graph.py  --config {Path(str(config.get('_config_path', 'config.yaml'))).name}"
            )
    nodes = read_table(locations["nodes"])
    edges = read_table(locations["edges"])
    return nodes, edges


def _month_series(frame: pd.DataFrame) -> pd.Series:
    months = frame["month"].astype(str).str.strip()
    bad = months[~months.str.fullmatch(r"\d{4}-\d{2}")]
    if len(bad):
        raise DataValidationError(
            f"{len(bad)} row(s) have a month that is not formatted YYYY-MM, "
            f"for example {sorted(bad.unique())[:5]}"
        )
    return months


def validate_tables(nodes: pd.DataFrame, edges: pd.DataFrame) -> dict[str, Any]:
    """Fail fast on anything that would silently corrupt a training run."""
    if nodes.empty:
        raise DataValidationError("The node table is empty; there is nothing to train on.")

    missing = [column for column in REQUIRED_NODE_COLUMNS if column not in nodes.columns]
    if missing:
        raise DataValidationError(
            f"Node table is missing required column(s): {missing}. "
            f"Expected at least: {REQUIRED_NODE_COLUMNS}"
        )
    if not edges.empty:
        missing_edges = [column for column in REQUIRED_EDGE_COLUMNS if column not in edges.columns]
        if missing_edges:
            raise DataValidationError(
                f"Edge table is missing required column(s): {missing_edges}. "
                f"Expected at least: {REQUIRED_EDGE_COLUMNS}"
            )

    nodes["month"] = _month_series(nodes)
    nodes["node_id"] = nodes["node_id"].astype(str)
    nodes["host_country_id"] = nodes["host_country_id"].astype(str)

    duplicates = nodes.duplicated(subset=["node_id", "month"]).sum()
    if duplicates:
        raise DataValidationError(
            f"{duplicates} duplicate (node_id, month) row(s) found. The supervised unit is one "
            "country-month; de-duplicate the fused table before training."
        )

    feature_report: dict[str, Any] = {}
    for feature in FEATURES:
        values = pd.to_numeric(nodes[feature], errors="coerce")
        non_numeric = int(values.isna().sum() - nodes[feature].isna().sum())
        if non_numeric > 0:
            raise DataValidationError(
                f"Feature {feature!r} contains {non_numeric} non-numeric value(s); refusing to coerce them."
            )
        nodes[feature] = values
        feature_report[feature] = {
            "missing": int(values.isna().sum()),
            "min": None if values.dropna().empty else float(values.min()),
            "max": None if values.dropna().empty else float(values.max()),
            "mean": None if values.dropna().empty else float(values.mean()),
            "std": None if values.dropna().empty else float(values.std(ddof=0)),
            "constant": bool(values.dropna().nunique() <= 1),
        }
    nodes["inbound_flow_usd"] = pd.to_numeric(nodes["inbound_flow_usd"], errors="coerce")

    if not edges.empty:
        edges["month"] = _month_series(edges)
        edges["source"] = edges["source"].astype(str)
        edges["destination"] = edges["destination"].astype(str)
        for column in ("trade_value_usd", "flow_volume"):
            if column in edges.columns:
                edges[column] = pd.to_numeric(edges[column], errors="coerce").fillna(0.0)
            else:
                edges[column] = 0.0

    months = sorted(nodes["month"].unique().tolist())
    node_ids = sorted(nodes["node_id"].unique().tolist())
    known = set(node_ids)
    orphan_edges = 0
    if not edges.empty:
        orphan_edges = int((~edges["source"].isin(known) | ~edges["destination"].isin(known)).sum())

    grid = len(months) * len(node_ids)
    report = {
        "node_rows": int(len(nodes)),
        "edge_rows": int(len(edges)),
        "months": months,
        "month_count": len(months),
        "node_count": len(node_ids),
        "complete_country_month_grid": bool(len(nodes) == grid),
        "expected_grid_rows": int(grid),
        "orphan_edge_rows": orphan_edges,
        "edge_months_missing_from_nodes": sorted(
            set(edges["month"].unique()) - set(months)
        ) if not edges.empty else [],
        "features": feature_report,
        "inbound_flow_missing": int(nodes["inbound_flow_usd"].isna().sum()),
        "inbound_flow_nonpositive": int((nodes["inbound_flow_usd"].fillna(0) <= 0).sum()),
    }
    if orphan_edges:
        LOGGER.warning(
            "%d edge row(s) reference a node that is absent from the node table; "
            "they are ignored during training.", orphan_edges
        )
    constant = [name for name, stats in feature_report.items() if stats["constant"]]
    if constant:
        LOGGER.warning(
            "Feature(s) %s are constant across the whole table; they carry no signal "
            "and their standardised value is fixed at 0.", constant
        )
    return report


# --------------------------------------------------------------------------- #
# Target construction
# --------------------------------------------------------------------------- #
def build_targets(
    nodes: pd.DataFrame,
    horizon: int = 1,
    baseline_window: int = 12,
    baseline_min_periods: int = 1,
    taus: Sequence[float] = (0.30, 0.35, 0.40),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Re-derive the documented one-month-ahead contraction target.

    ``future``   = inbound_flow_usd(country, T + horizon)
    ``baseline`` = rolling median of the ``baseline_window`` months ending at
                   ``T + horizon - 1``
    ``contraction`` = (future - baseline) / baseline
    ``label(tau)``  = 1 when contraction < -tau, and only where the target is valid.

    A row is valid only when the future value is observed and the baseline is
    positive. Everything else is ``target_valid = False`` and is excluded from
    both training and evaluation; it is never coerced into a negative example.
    """
    if horizon < 1:
        raise DataValidationError(f"horizon_months must be >= 1, received {horizon}")

    frame = nodes.copy()
    frame["month"] = frame["month"].astype(str)
    frame = frame.sort_values(["host_country_id", "month"]).reset_index(drop=True)
    grouped = frame.groupby("host_country_id", sort=False)["inbound_flow_usd"]

    future = grouped.shift(-horizon)
    baseline = grouped.transform(
        lambda series: series.shift(-(horizon - 1))
        .rolling(baseline_window, min_periods=baseline_min_periods)
        .median()
    )
    contraction = (future - baseline) / baseline.replace(0, np.nan)

    frame["future_inbound_flow_usd"] = future
    frame["baseline_inbound_flow_usd"] = baseline
    frame["contraction"] = contraction
    frame["target_valid"] = (
        future.notna() & baseline.notna() & baseline.gt(0) & contraction.notna()
    )

    for tau in taus:
        frame[f"derived_label_tau_{float(tau):.2f}"] = np.where(
            frame["target_valid"], (contraction < -float(tau)).astype(float), np.nan
        )

    valid = frame.loc[frame["target_valid"], "contraction"].astype(float)
    stats = {
        "horizon_months": int(horizon),
        "baseline_window_months": int(baseline_window),
        "baseline_min_periods": int(baseline_min_periods),
        "country_month_rows": int(len(frame)),
        "valid_targets": int(frame["target_valid"].sum()),
        "invalid_targets": int((~frame["target_valid"]).sum()),
        "invalid_reasons": {
            "future_value_unobserved": int(future.isna().sum()),
            "baseline_unavailable": int(baseline.isna().sum()),
            "baseline_not_positive": int((baseline.notna() & baseline.le(0)).sum()),
        },
        "contraction": {
            "count": int(len(valid)),
            "min": float(valid.min()) if len(valid) else None,
            "max": float(valid.max()) if len(valid) else None,
            "mean": float(valid.mean()) if len(valid) else None,
            "median": float(valid.median()) if len(valid) else None,
            "std": float(valid.std(ddof=0)) if len(valid) else None,
            "quantiles": {
                str(q): float(valid.quantile(q)) for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
            } if len(valid) else {},
        },
        "positives_by_tau": {
            f"{float(tau):.2f}": int(np.nansum(frame[f"derived_label_tau_{float(tau):.2f}"].to_numpy()))
            for tau in taus
        },
    }
    if baseline_min_periods < baseline_window:
        stats["baseline_caveat"] = (
            f"baseline_min_periods={baseline_min_periods} < baseline_window={baseline_window}: early "
            "months use a shorter-than-12-month median. This reproduces the labels stored in the "
            "existing 2024 table. Set baseline_min_periods to 12 once a full 12-month history exists."
        )
    return frame, stats


def cross_check_labels(frame: pd.DataFrame, taus: Sequence[float]) -> dict[str, Any]:
    """Compare re-derived labels with the labels stored in the fused table."""
    checks: dict[str, Any] = {}
    for tau in taus:
        key = f"{float(tau):.2f}"
        stored_column = f"label_tau_{key}"
        derived_column = f"derived_label_tau_{key}"
        if stored_column not in frame.columns:
            checks[key] = {"status": "absent", "detail": f"{stored_column} is not in the node table"}
            continue
        stored = pd.to_numeric(frame[stored_column], errors="coerce")
        derived = pd.to_numeric(frame[derived_column], errors="coerce")
        both = stored.notna() & derived.notna()
        agree = int((stored[both] == derived[both]).sum())
        comparable = int(both.sum())
        checks[key] = {
            "status": "match" if comparable and agree == comparable else ("no_overlap" if not comparable else "mismatch"),
            "comparable_rows": comparable,
            "agreeing_rows": agree,
            "stored_labelled_rows": int(stored.notna().sum()),
            "derived_labelled_rows": int(derived.notna().sum()),
            "stored_positive": int(stored.fillna(0).sum()),
            "derived_positive": int(derived.fillna(0).sum()),
        }
        if checks[key]["status"] == "mismatch":
            LOGGER.warning(
                "Stored and re-derived labels disagree on %d of %d comparable rows at tau=%s. "
                "Training uses the re-derived label so the target definition is explicit.",
                comparable - agree, comparable, key,
            )
    return checks


# --------------------------------------------------------------------------- #
# Chronological split
# --------------------------------------------------------------------------- #
@dataclass
class Split:
    train: list[str]
    validation: list[str]
    test: list[str]
    mode: str
    supervised_months: list[str] = field(default_factory=list)
    excluded_months: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, list[str]]:
        return {"train": self.train, "validation": self.validation, "test": self.test}

    def all_months(self) -> list[str]:
        return [*self.train, *self.validation, *self.test]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
            "month_counts": {name: len(months) for name, months in self.as_dict().items()},
            "supervised_months": self.supervised_months,
            "excluded_months": self.excluded_months,
            "notes": self.notes,
        }


def _month_range(bounds: Sequence[str]) -> list[str]:
    if len(bounds) != 2:
        raise DataValidationError(
            f"A date-range split needs exactly two entries [start, end]; received {list(bounds)!r}"
        )
    start, end = (str(bounds[0])[:7], str(bounds[1])[:7])
    if start > end:
        raise DataValidationError(f"Split range start {start} is after its end {end}")
    periods = pd.period_range(start=start, end=end, freq="M")
    return [str(period) for period in periods]


def resolve_split(
    months: Sequence[str],
    supervised_months: Sequence[str],
    settings: Mapping[str, Any],
    overrides: Mapping[str, Any] | None = None,
) -> Split:
    """Build a chronological, disjoint train/validation/test split.

    Two modes are supported:

    ``date_ranges`` (preferred, explicit)::

        split:
          mode: date_ranges
          train:      ["2021-01", "2022-12"]
          validation: ["2023-01", "2023-12"]
          test:       ["2024-01", "2024-12"]

    ``counts`` (take the first N supervised months, then the next M, ...)::

        split:
          mode: counts
          train_months: 7
          validation_months: 2
          test_months: 2

    Months whose target cannot be observed are removed from every partition
    and reported in ``notes``; they are never relabelled as negatives.
    """
    overrides = dict(overrides or {})
    settings = dict(settings or {})
    supervised = list(supervised_months)
    notes: list[str] = []

    excluded = sorted(set(months) - set(supervised))
    if excluded:
        notes.append(
            f"{len(excluded)} month(s) have no observable target at this horizon and are excluded "
            f"from every partition: {excluded}"
        )

    mode = str(overrides.get("mode") or settings.get("mode") or "date_ranges").lower()

    if mode == "date_ranges":
        requested: dict[str, list[str]] = {}
        for name in ("train", "validation", "test"):
            bounds = overrides.get(name) or settings.get(name)
            if not bounds:
                raise DataValidationError(
                    f"Split mode 'date_ranges' requires a {name!r} range such as "
                    f'["2024-01", "2024-07"]. Set model_training.split.{name} in the config '
                    f"or pass --{name.replace('validation', 'validation')}-range START:END."
                )
            requested[name] = _month_range(bounds)
        resolved: dict[str, list[str]] = {}
        for name, wanted in requested.items():
            kept = [month for month in wanted if month in supervised]
            dropped = [month for month in wanted if month not in supervised]
            if dropped:
                notes.append(
                    f"{name}: dropped {len(dropped)} requested month(s) that are absent or have no "
                    f"observable target: {dropped}"
                )
            resolved[name] = kept
    elif mode == "counts":
        counts = {}
        for name, key, default in (
            ("train", "train_months", 7),
            ("validation", "validation_months", 2),
            ("test", "test_months", 2),
        ):
            counts[name] = int(overrides.get(key, settings.get(key, default)))
            if counts[name] < 1:
                raise DataValidationError(f"{key} must be >= 1, received {counts[name]}")
        required = sum(counts.values())
        if required > len(supervised):
            raise DataValidationError(
                f"The configured split needs {required} supervised months "
                f"({counts['train']} train + {counts['validation']} validation + {counts['test']} test) "
                f"but only {len(supervised)} are available "
                f"({supervised[0]}..{supervised[-1]}). "
                "Lower the counts, switch to mode: date_ranges, or extend the data window."
            )
        resolved = {
            "train": supervised[: counts["train"]],
            "validation": supervised[counts["train"] : counts["train"] + counts["validation"]],
            "test": supervised[counts["train"] + counts["validation"] : required],
        }
        notes.append(
            f"counts mode consumed the first {required} of {len(supervised)} supervised months"
        )
    else:
        raise DataValidationError(
            f"Unknown split mode {mode!r}; valid values are 'date_ranges' and 'counts'."
        )

    split = Split(
        train=resolved["train"],
        validation=resolved["validation"],
        test=resolved["test"],
        mode=mode,
        supervised_months=supervised,
        excluded_months=excluded,
        notes=notes,
    )
    _validate_split(split)
    return split


def _validate_split(split: Split) -> None:
    for name, months in split.as_dict().items():
        if not months:
            raise DataValidationError(
                f"The {name} partition is empty after intersecting the requested months with the "
                f"months that actually have an observable target. Available supervised months: "
                f"{split.supervised_months}"
            )
    seen: set[str] = set()
    for name, months in split.as_dict().items():
        overlap = seen & set(months)
        if overlap:
            raise DataValidationError(
                f"The {name} partition overlaps an earlier partition on {sorted(overlap)}; "
                "train, validation, and test months must be disjoint."
            )
        seen |= set(months)
        if months != sorted(months):
            raise DataValidationError(f"The {name} months are not in chronological order: {months}")
    if not (max(split.train) < min(split.validation) < max(split.validation) < min(split.test)):
        raise DataValidationError(
            "Partitions must be strictly forward-chained: every training month must precede every "
            f"validation month, which must precede every test month. Received "
            f"train={split.train[0]}..{split.train[-1]}, "
            f"validation={split.validation[0]}..{split.validation[-1]}, "
            f"test={split.test[0]}..{split.test[-1]}."
        )


# --------------------------------------------------------------------------- #
# Fitted transforms (training months only)
# --------------------------------------------------------------------------- #
@dataclass
class FeatureStandardizer:
    """Zero-mean/unit-variance scaling fitted on the training partition only."""

    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    features: list[str] = field(default_factory=lambda: list(FEATURES))

    def _oriented(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame.reindex(columns=self.features).astype(float).fillna(0.0).to_numpy(copy=True)
        values[:, self.features.index(PROTECTIVE_FEATURE)] *= -1.0
        return values

    def fit(self, frame: pd.DataFrame) -> "FeatureStandardizer":
        if frame.empty:
            raise DataValidationError("Cannot fit the feature standardizer on an empty training partition.")
        values = self._oriented(frame)
        self.mean = values.mean(axis=0)
        self.scale = values.std(axis=0)
        # A constant feature has zero variance; dividing by 1 keeps it at 0.
        self.scale[self.scale < 1e-8] = 1.0
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.mean is None or self.scale is None:
            raise DataValidationError("FeatureStandardizer.transform called before fit.")
        return (self._oriented(frame) - self.mean) / self.scale

    def to_dict(self) -> dict[str, Any]:
        if self.mean is None or self.scale is None:
            return {}
        return {
            "features": self.features,
            "sign_flipped": [PROTECTIVE_FEATURE],
            "mean": [float(value) for value in self.mean],
            "scale": [float(value) for value in self.scale],
        }


def edge_scales(edges: pd.DataFrame, train_months: Iterable[str]) -> tuple[float, float]:
    """log1p edge normalisers fitted on the training months only."""
    if edges.empty:
        return 1.0, 1.0
    subset = edges[edges["month"].isin(list(train_months))]
    if subset.empty:
        return 1.0, 1.0
    values = np.log1p(pd.to_numeric(subset["trade_value_usd"], errors="coerce").fillna(0).clip(lower=0))
    volumes = np.log1p(pd.to_numeric(subset["flow_volume"], errors="coerce").fillna(0).clip(lower=0))
    return max(float(values.max()), 1.0), max(float(volumes.max()), 1.0)


# --------------------------------------------------------------------------- #
# Tensorisation
# --------------------------------------------------------------------------- #
@dataclass
class MonthBatch:
    """Everything one monthly step needs, already on the target device."""

    month: str
    time: float
    features: torch.Tensor          # [num_nodes, num_features]
    edge_index: torch.Tensor        # [2, 2 * num_edges], bidirectional, for the GCN
    events: list[dict[str, Any]]    # chronological event dicts, for the TGN
    target: torch.Tensor            # [num_nodes], 0 where invalid
    mask: torch.Tensor              # [num_nodes] bool, True where the target is valid

    @property
    def valid_count(self) -> int:
        return int(self.mask.sum().item())


def build_month_batches(
    frame: pd.DataFrame,
    edges: pd.DataFrame,
    months: Sequence[str],
    node_ids: Sequence[str],
    standardizer: FeatureStandardizer,
    edge_scale: tuple[float, float],
    target_column: str,
    device: torch.device,
) -> dict[str, MonthBatch]:
    """Materialise every monthly snapshot once, on ``device``.

    Building these tensors once instead of per epoch is what makes repeated
    epochs cheap, and it guarantees that every model in the comparison sees
    byte-identical inputs.
    """
    index = {node_id: position for position, node_id in enumerate(node_ids)}
    value_scale, volume_scale = edge_scale
    node_count = len(node_ids)
    batches: dict[str, MonthBatch] = {}

    frame = frame.copy()
    frame["month"] = frame["month"].astype(str)
    frame["node_id"] = frame["node_id"].astype(str)
    by_month = {month: group for month, group in frame.groupby("month", sort=False)}
    edges_by_month: dict[str, pd.DataFrame] = {}
    if not edges.empty:
        edges_by_month = {month: group for month, group in edges.groupby("month", sort=False)}

    for month in months:
        current = by_month.get(month, frame.iloc[0:0])
        current = current.drop_duplicates("node_id").set_index("node_id")
        ordered = current.reindex(list(node_ids))

        feature_values = standardizer.transform(ordered)
        features = torch.as_tensor(feature_values, dtype=torch.float32, device=device)

        target_values = np.zeros(node_count, dtype=np.float32)
        mask_values = np.zeros(node_count, dtype=bool)
        if target_column in ordered.columns:
            raw = pd.to_numeric(ordered[target_column], errors="coerce").to_numpy(dtype=float)
            valid = ~np.isnan(raw)
            if "target_valid" in ordered.columns:
                valid &= ordered["target_valid"].fillna(False).to_numpy(dtype=bool)
            target_values[valid] = raw[valid].astype(np.float32)
            mask_values = valid
        target = torch.as_tensor(target_values, dtype=torch.float32, device=device)
        mask = torch.as_tensor(mask_values, dtype=torch.bool, device=device)

        month_time = float(pd.Period(month, freq="M").ordinal)
        sources: list[int] = []
        destinations: list[int] = []
        events: list[dict[str, Any]] = []
        month_edges = edges_by_month.get(month)
        if month_edges is not None and not month_edges.empty:
            trade = np.log1p(
                pd.to_numeric(month_edges["trade_value_usd"], errors="coerce").fillna(0).clip(lower=0).to_numpy()
            ) / value_scale
            volume = np.log1p(
                pd.to_numeric(month_edges["flow_volume"], errors="coerce").fillna(0).clip(lower=0).to_numpy()
            ) / volume_scale
            for position, (source, destination) in enumerate(
                zip(month_edges["source"].to_numpy(), month_edges["destination"].to_numpy())
            ):
                source_index = index.get(str(source))
                destination_index = index.get(str(destination))
                if source_index is None or destination_index is None:
                    continue
                sources.append(source_index)
                destinations.append(destination_index)
                events.append(
                    {
                        "source_index": source_index,
                        "destination_index": destination_index,
                        "time": month_time,
                        # One month between consecutive snapshots; the harmonic
                        # time encoder consumes this delta directly.
                        "time_delta": 1.0,
                        "edge_features": torch.tensor(
                            [float(trade[position]), float(volume[position])],
                            dtype=torch.float32,
                            device=device,
                        ),
                    }
                )
        if sources:
            forward = torch.tensor([sources, destinations], dtype=torch.long, device=device)
            edge_index = torch.cat([forward, forward.flip(0)], dim=1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

        batches[month] = MonthBatch(
            month=month,
            time=month_time,
            features=features,
            edge_index=edge_index,
            events=events,
            target=target,
            mask=mask,
        )
    return batches


def supervised_months(frame: pd.DataFrame, months: Sequence[str]) -> list[str]:
    """Months that contain at least one country with an observable target."""
    valid = frame[frame["target_valid"].fillna(False).astype(bool)]
    present = set(valid["month"].astype(str).unique().tolist())
    return [month for month in months if month in present]


def positive_weight(frame: pd.DataFrame, months: Sequence[str], label_column: str) -> float:
    """Inverse training prevalence, used as the weighted-BCE positive weight."""
    subset = frame[frame["month"].isin(list(months)) & frame["target_valid"].fillna(False).astype(bool)]
    labels = pd.to_numeric(subset[label_column], errors="coerce").dropna()
    if labels.empty:
        return 1.0
    prevalence = float(labels.mean())
    if prevalence <= 0.0:
        return 1.0
    return float(min(1.0 / prevalence, 1e4))


def class_summary(frame: pd.DataFrame, split: Split, label_column: str) -> dict[str, Any]:
    """Per-partition label counts, including the degenerate single-class case."""
    summary: dict[str, Any] = {}
    for name, months in split.as_dict().items():
        subset = frame[frame["month"].isin(months) & frame["target_valid"].fillna(False).astype(bool)]
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


def regression_summary(frame: pd.DataFrame, split: Split) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name, months in split.as_dict().items():
        subset = frame[frame["month"].isin(months) & frame["target_valid"].fillna(False).astype(bool)]
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
