#!/usr/bin/env python3
"""Bootstrap 95% CIs for country-month test metrics (2024 hold-out)."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "dataset/data/four_year_2021_2024/results/model_training/benchmark_comparison/country_month"
OUT = Path(__file__).resolve().parent
N_BOOT = 5000
SEED = 42


def load_test(path: Path) -> tuple[np.ndarray, np.ndarray]:
    y, p = [], []
    with path.open() as f:
        for row in csv.DictReader(f):
            if row["split"] != "test":
                continue
            y.append(float(row["target"]))
            p.append(float(row["score"]))
    return np.asarray(y), np.asarray(p)


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def pr_auc(y: np.ndarray, scores: np.ndarray) -> float:
    try:
        from sklearn.metrics import average_precision_score

        return float(average_precision_score(y, scores))
    except Exception:
        order = np.argsort(-scores)
        y_sorted = y[order]
        tp = np.cumsum(y_sorted)
        fp = np.cumsum(1 - y_sorted)
        prec = tp / np.maximum(tp + fp, 1)
        rec = tp / max(y.sum(), 1)
        return float(np.trapezoid(prec, rec))


def bootstrap_ci(y: np.ndarray, pred: np.ndarray, metric_fn, n: int = N_BOOT) -> dict:
    rng = np.random.default_rng(SEED)
    n_rows = len(y)
    stats = []
    for _ in range(n):
        idx = rng.integers(0, n_rows, n_rows)
        stats.append(metric_fn(y[idx], pred[idx]))
    arr = np.asarray(stats)
    return {
        "point": metric_fn(y, pred),
        "ci_low": float(np.percentile(arr, 2.5)),
        "ci_high": float(np.percentile(arr, 97.5)),
        "n": int(n_rows),
        "n_boot": n,
    }


def main() -> None:
    results = {"regression": {}, "classification": {}}

    for model in ("ridge_regression", "tgn", "gcn"):
        path = BENCH / "regression/predictions" / f"{model}.csv"
        if path.exists():
            y, pred = load_test(path)
            results["regression"][model] = bootstrap_ci(y, pred, r2)

    for model in ("tgn_no_memory", "logistic_regression", "tgn", "gcn"):
        path = BENCH / "classification/predictions" / f"{model}.csv"
        if path.exists():
            y, pred = load_test(path)
            y_bin = (y >= 0.5).astype(float)  # targets are 0/1
            results["classification"][model] = bootstrap_ci(y_bin, pred, pr_auc)

    out_json = OUT / "bootstrap_ci.json"
    out_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    lines = ["# Bootstrap 95% CIs (test 2024, country-month)", "", f"Resamples: {N_BOOT}, seed: {SEED}", ""]
    lines.append("## Regression ($R^2$)")
    lines.append("| Model | Point | 95% CI | n |")
    lines.append("| --- | ---: | --- | ---: |")
    for m, v in results["regression"].items():
        lines.append(f"| {m} | {v['point']:.3f} | [{v['ci_low']:.3f}, {v['ci_high']:.3f}] | {v['n']} |")
    lines.append("")
    lines.append("## Classification (PR-AUC, $\\tau=0.20$)")
    lines.append("| Model | Point | 95% CI | n |")
    lines.append("| --- | ---: | --- | ---: |")
    for m, v in results["classification"].items():
        lines.append(f"| {m} | {v['point']:.3f} | [{v['ci_low']:.3f}, {v['ci_high']:.3f}] | {v['n']} |")
    (OUT / "bootstrap_ci.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_json)
    print((OUT / "bootstrap_ci.md").read_text())


if __name__ == "__main__":
    main()
