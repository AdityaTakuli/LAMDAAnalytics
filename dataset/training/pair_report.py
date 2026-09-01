"""Reporting helpers for pooled pair-month runs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from training.metrics import compute as compute_metrics
from training.metrics import metric_names
from training.pair_data import PAIR_PREDICTION_COLUMNS, PairMonthBatch
from training.report import SPLIT_NAMES, _pyplot

LOGGER = logging.getLogger("training.pair_report")


def pair_scores_to_frame(
    batches: Mapping[str, PairMonthBatch],
    split,
    scores: Mapping[str, tuple[np.ndarray, list[tuple[str, str]]]],
    model_name: str,
    target_column: str,
    pair_frame: pd.DataFrame,
) -> pd.DataFrame:
    lookup = pair_frame.set_index(["month", "source", "destination"])
    rows: list[dict[str, Any]] = []
    for split_name, months in split.as_dict().items():
        for month in months:
            batch = batches.get(month)
            payload = scores.get(month)
            if batch is None or payload is None or batch.valid_count == 0:
                continue
            values, keys = payload
            for index, (source, destination) in enumerate(keys):
                try:
                    target = float(lookup.loc[(month, source, destination), target_column])
                except KeyError:
                    continue
                rows.append(
                    {
                        "model": model_name,
                        "split": split_name,
                        "month": month,
                        "source": source,
                        "destination": destination,
                        "target": target,
                        "score": float(values[index]),
                    }
                )
    return pd.DataFrame(rows, columns=PAIR_PREDICTION_COLUMNS)


def metrics_by_split(task: str, frame: pd.DataFrame, threshold: float = 0.5) -> dict[str, Any]:
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


def write_pair_plots(
    task: str,
    pair_frame: pd.DataFrame,
    histories: Mapping[str, list[dict[str, Any]]],
    predictions: Mapping[str, pd.DataFrame],
    metrics: Mapping[str, Mapping[str, Any]],
    directory: Path,
) -> list[str]:
    written: list[str] = []
    directory.mkdir(parents=True, exist_ok=True)
    try:
        plt = _pyplot()
    except Exception as exc:
        LOGGER.warning("Plots skipped: %s", exc)
        return written

    def _save(figure, name: str) -> None:
        destination = directory / name
        figure.tight_layout()
        figure.savefig(destination, dpi=140)
        plt.close(figure)
        written.append(str(destination))

    try:
        valid = pair_frame[pair_frame["target_valid"].fillna(False).astype(bool)]
        contraction = pd.to_numeric(valid["contraction"], errors="coerce").dropna()
        if len(contraction):
            figure, axis = plt.subplots(figsize=(7, 4))
            axis.hist(contraction, bins=min(40, max(10, len(contraction) // 50)), color="#4C72B0")
            axis.axvline(0.0, color="#333333", linewidth=1)
            axis.set_title("Distribution of bilateral pair-month contraction targets")
            axis.set_xlabel("contraction = (future - baseline) / baseline")
            axis.set_ylabel("pair-months")
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
            axis.set_title(f"Pooled pair model comparison: {headline}")
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
                    axis.scatter(subset["target"], subset["score"], s=8, alpha=0.5, color="#4C72B0")
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

    try:
        test_frames = []
        for name, frame in predictions.items():
            subset = frame[frame["split"].eq("test")].copy()
            if subset.empty:
                continue
            subset["model"] = name
            test_frames.append(subset)
        if test_frames:
            combined = pd.concat(test_frames, ignore_index=True)
            if task == "classification":
                pivot = combined.groupby(["source", "destination"])["target"].mean().reset_index(name="prevalence")
                pivot = pivot.sort_values("prevalence", ascending=False).head(20)
                figure, axis = plt.subplots(figsize=(8, 5))
                labels = [f"{row.source}->{row.destination}" for row in pivot.itertuples()]
                axis.barh(labels[::-1], pivot["prevalence"].to_numpy()[::-1], color="#55A868")
                axis.set_title("Top 20 directed pairs by test-set positive prevalence")
                axis.set_xlabel("positive prevalence")
                _save(figure, "top_pair_prevalence.png")
    except Exception as exc:
        LOGGER.warning("top_pair_prevalence plot failed: %s", exc)

    return written


def write_readme(output_dir: Path, summary: Mapping[str, Any]) -> None:
    task = summary.get("task", "classification")
    headline = "average_precision" if task == "classification" else "rmse"
    lines = [
        "# Pooled bilateral pair-month training run",
        "",
        f"Supervision unit: directed importer ← exporter links pooled across the panel.",
        f"Task: `{task}`; headline metric: `{headline}`.",
        "",
        "## Layout",
        "",
        "| Path | Description |",
        "| --- | --- |",
        "| `run_summary.json` | Options, split, metrics, environment |",
        "| `metrics/comparison.csv` | One row per (model, split) |",
        "| `predictions/<model>.csv` | Per pair-month scores |",
        "| `diagnostics/` | Class balance, split, target summary |",
        "| `plots/` | Loss curves, model comparison, target distribution |",
        "",
        f"Archived country-month runs live under `../country_month/`.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
