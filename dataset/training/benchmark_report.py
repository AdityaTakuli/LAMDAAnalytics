"""Cross-setup benchmark plots for country-month vs bilateral pair runs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("training.benchmark_report")

MODEL_ORDER = [
    "train_median",
    "constant_prevalence",
    "ridge_regression",
    "logistic_regression",
    "hand_weighted_linear",
    "random_forest",
    "lightgbm",
    "lightgbm_lags",
    "mlp",
    "gcn",
    "tgn_no_memory",
    "tgn",
]


def _load_comparison(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def build_master_table(runs: Mapping[str, Mapping[str, Path]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for setup, tasks in runs.items():
        for task, root in tasks.items():
            comparison = _load_comparison(root / "metrics" / "comparison.csv")
            if comparison.empty:
                continue
            test = comparison[comparison["split"].astype(str).eq("test")].copy()
            for _, row in test.iterrows():
                rows.append(
                    {
                        "setup": setup,
                        "task": task,
                        "model": str(row["model"]),
                        "n": row.get("n"),
                        "rmse": row.get("rmse"),
                        "r2": row.get("r2"),
                        "mae": row.get("mae"),
                        "average_precision": row.get("average_precision"),
                        "roc_auc": row.get("roc_auc"),
                        "f1": row.get("f1"),
                    }
                )
    return pd.DataFrame(rows)


def write_benchmark_plots(master: pd.DataFrame, output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        LOGGER.warning("matplotlib unavailable: %s", exc)
        return []

    written: list[str] = []

    def save(fig, name: str) -> None:
        path = output_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(str(path))

    def bar_compare(task: str, metric: str, title: str, ascending: bool) -> None:
        subset = master[master["task"].eq(task)].dropna(subset=[metric])
        if subset.empty:
            return
        pivot = subset.pivot_table(index="model", columns="setup", values=metric, aggfunc="first")
        pivot = pivot.reindex([m for m in MODEL_ORDER if m in pivot.index]).dropna(how="all")
        if pivot.empty:
            return
        fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(pivot))))
        y = np.arange(len(pivot.index))
        width = 0.35
        setups = list(pivot.columns)
        for index, setup in enumerate(setups):
            offset = (index - (len(setups) - 1) / 2) * width
            ax.barh(y + offset, pivot[setup].to_numpy(), height=width, label=setup.replace("_", " "))
        ax.set_yticks(y)
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel(metric)
        ax.set_title(title)
        ax.legend()
        save(fig, f"test_{task}_{metric}.png")

    bar_compare("regression", "rmse", "Test regression RMSE (lower is better)", True)
    bar_compare("regression", "r2", "Test regression R² (higher is better)", False)
    bar_compare("classification", "average_precision", "Test classification PR-AUC (higher is better)", False)
    bar_compare("classification", "roc_auc", "Test classification ROC-AUC (higher is better)", False)

    # Overview grouped bars
    for task, metric, ylabel in (
        ("regression", "r2", "R²"),
        ("classification", "average_precision", "PR-AUC"),
    ):
        subset = master[master["task"].eq(task)].dropna(subset=[metric])
        if subset.empty:
            continue
        models = [m for m in MODEL_ORDER if m in set(subset["model"])]
        setups = sorted(subset["setup"].unique())
        fig, ax = plt.subplots(figsize=(max(10, len(models) * 0.75), 5))
        x = np.arange(len(models))
        width = 0.35
        for index, setup in enumerate(setups):
            sub = subset[subset["setup"].eq(setup)].set_index("model").reindex(models)[metric]
            ax.bar(x + (index - 0.5) * width, sub.to_numpy(), width=width, label=setup.replace("_", " "))
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Test {ylabel}: country-month vs bilateral pairs")
        ax.legend()
        save(fig, f"overview_{task}_{metric}.png")

    return written


def write_benchmark_report(master: pd.DataFrame, output_dir: Path, plot_paths: list[str]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    master = master.sort_values(["task", "setup", "model"])
    master.to_csv(output_dir / "master_comparison.csv", index=False)

    lines = [
        "# Full model benchmark",
        "",
        "All models on **country-month (India-framed)** vs **pooled bilateral pairs**.",
        "",
    ]
    for task in ("regression", "classification"):
        lines.append(f"## {task.title()} (test split)")
        lines.append("")
        sub = master[master["task"].eq(task)]
        for setup in sorted(sub["setup"].unique()):
            lines.append(f"### {setup}")
            lines.append("")
            part = sub[sub["setup"].eq(setup)]
            if task == "regression":
                part = part.sort_values("r2", ascending=False)
                for _, row in part.iterrows():
                    lines.append(
                        f"- **{row['model']}**: R²={row['r2']}, RMSE={row['rmse']}, MAE={row['mae']}, n={row['n']}"
                    )
            else:
                part = part.sort_values("average_precision", ascending=False)
                for _, row in part.iterrows():
                    lines.append(
                        f"- **{row['model']}**: PR-AUC={row['average_precision']}, "
                        f"ROC-AUC={row['roc_auc']}, F1={row['f1']}, n={row['n']}"
                    )
            lines.append("")
    lines.append("## Plots")
    for path in plot_paths:
        lines.append(f"- `{Path(path).name}`")
    report = output_dir / "BENCHMARK_REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
