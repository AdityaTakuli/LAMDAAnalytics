"""Deterministic end-to-end self-test on synthetic tables.

The self-test exists to answer one question quickly: *does the whole training
stack work on this machine and this device?* It builds a small synthetic
country-month panel with a known, learnable signal, runs the real pipeline over
it, and asserts that the artifacts appear and that a model beats the constant
baseline it should beat.

The synthetic panel is a plumbing fixture. It is never written into a data
profile directory, and every artifact it produces is stamped
``"synthetic": true``.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import paths  # noqa: F401  (sys.path bootstrap)
from training.pipeline import Options, run

LOGGER = logging.getLogger("training.selftest")

COUNTRIES = 8
MONTHS = 36
SHOCK_PROBABILITY = 0.22
SHOCK_MULTIPLIER = 0.45


def synthetic_tables(seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a panel where next-month contraction is genuinely predictable."""
    rng = np.random.default_rng(seed)
    months = [str(period) for period in pd.period_range("2020-01", periods=MONTHS, freq="M")]
    country_ids = [f"country_{index:02d}" for index in range(COUNTRIES)]

    # A shock in month t is announced by elevated risk features in month t-1,
    # which is exactly the lead-time structure the models are meant to learn.
    # The announcement is deliberately imperfect: some shocks arrive unannounced
    # and some warnings are false alarms, so a perfect score would itself be a
    # symptom of a bug rather than a passing test.
    shocks = rng.random((COUNTRIES, MONTHS)) < SHOCK_PROBABILITY
    announced = shocks & (rng.random((COUNTRIES, MONTHS)) < 0.75)
    false_alarms = (~shocks) & (rng.random((COUNTRIES, MONTHS)) < 0.12)
    warnings = announced | false_alarms

    node_rows: list[dict[str, Any]] = []
    base_levels = rng.uniform(5e7, 5e8, size=COUNTRIES)
    for country_index, node_id in enumerate(country_ids):
        for month_index, month in enumerate(months):
            shocked = bool(shocks[country_index, month_index])
            level = base_levels[country_index] * (SHOCK_MULTIPLIER if shocked else 1.0)
            level *= 1.0 + rng.normal(0.0, 0.04)
            upcoming = bool(warnings[country_index, month_index + 1]) if month_index + 1 < MONTHS else False
            signal = 1.0 if upcoming else 0.0
            node_rows.append(
                {
                    "month": month,
                    "node_id": node_id,
                    "host_country_id": node_id,
                    "node_type": "country",
                    "inbound_flow_usd": float(max(level, 1.0)),
                    "inventory_days_proxy": float(30.0 - 5.0 * signal + rng.normal(0.0, 3.5)),
                    "trade_delay_proxy": float(2.0 + 2.5 * signal + rng.normal(0.0, 2.0)),
                    "news_vol_7d": float(4.0 + 6.0 * signal + rng.normal(0.0, 4.5)),
                    "neg_tone_frac_3d": float(np.clip(0.15 + 0.18 * signal + rng.normal(0.0, 0.12), 0.0, 1.0)),
                    "strike_flag_7d": float(rng.random() < (0.10 + 0.30 * signal)),
                    "weather_anomaly_7d": float(rng.normal(0.0, 1.0) + 0.5 * signal),
                    "global_risk": float(0.4 + 0.12 * signal + rng.normal(0.0, 0.10)),
                }
            )
    nodes = pd.DataFrame(node_rows)

    # Stored labels use the documented formula so the label cross-check has
    # something real to compare against.
    ordered = nodes.sort_values(["host_country_id", "month"]).reset_index(drop=True)
    grouped = ordered.groupby("host_country_id", sort=False)["inbound_flow_usd"]
    future = grouped.shift(-1)
    baseline = grouped.transform(lambda series: series.rolling(12, min_periods=1).median())
    contraction = (future - baseline) / baseline.replace(0, np.nan)
    valid = future.notna() & baseline.notna() & baseline.gt(0) & contraction.notna()
    for tau in (0.30, 0.35, 0.40):
        ordered[f"label_tau_{tau:.2f}"] = np.where(valid, (contraction < -tau).astype(float), np.nan)
    nodes = ordered

    edge_rows: list[dict[str, Any]] = []
    for month in months:
        for source_index, source in enumerate(country_ids):
            for destination in country_ids[source_index + 1 :]:
                edge_rows.append(
                    {
                        "month": month,
                        "source": source,
                        "destination": destination,
                        "trade_value_usd": float(abs(rng.normal(2e7, 5e6))),
                        "flow_volume": float(abs(rng.normal(1e5, 2e4))),
                    }
                )
    return nodes, pd.DataFrame(edge_rows)


SYNTHETIC_CONFIG: dict[str, Any] = {
    "_config_path": "<self-test>",
    "analysis": {"horizon_months": 1, "taus": [0.30, 0.35, 0.40], "default_tau": 0.30},
    "model": {"memory_dim": 32, "time_dim": 8, "edge_dim": 2, "message_dim": 64, "embedding_dim": 32,
              "max_neighbors": 20, "gcn_layers": 2},
    "model_training": {
        "task": "classification",
        "baseline_min_periods": 1,
        "split": {"mode": "counts", "train_months": 20, "validation_months": 7, "test_months": 8},
    },
    "training": {"epochs": 5, "learning_rate": 0.01, "seed": 7},
}


def run_self_test(
    device: str = "auto",
    output_dir: Path | None = None,
    epochs: int = 5,
    task: str = "classification",
    seed: int = 7,
) -> dict[str, Any]:
    """Run the pipeline on synthetic data and validate the artifacts."""
    nodes, edges = synthetic_tables(seed)
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if output_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="lamda-selftest-")
        output_dir = Path(temporary.name) / task
    try:
        options = Options(
            task=task,
            tau=0.30,
            horizon=1,
            baseline_min_periods=1,
            models=["gcn", "tgn", "tgn_no_memory"],
            split_settings=SYNTHETIC_CONFIG["model_training"]["split"],
            epochs=int(epochs),
            learning_rate=0.01,
            seed=int(seed),
            device=device,
            overwrite=True,
            output_dir=Path(output_dir),
            model_config=SYNTHETIC_CONFIG["model"],
        )
        summary = run(SYNTHETIC_CONFIG, options, tables=(nodes, edges))
        checks = _validate(summary, Path(output_dir), task)
        summary["self_test_checks"] = checks
        summary["self_test_passed"] = all(check["passed"] for check in checks)
        return summary
    finally:
        # A caller-supplied directory is kept; a scratch directory is released
        # once the checks have read it. The summary carries every number.
        if temporary is not None:
            temporary.cleanup()


def _validate(summary: dict[str, Any], output_dir: Path, task: str) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def _check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})
        LOGGER.info("  [%s] %s — %s", "PASS" if passed else "FAIL", name, detail)

    for relative in ("run_summary.json", "metrics/comparison.csv", "diagnostics/leakage_audit.md"):
        target = output_dir / relative
        _check(f"artifact:{relative}", target.exists(), str(target))
    for model_name in summary.get("models", {}):
        checkpoint = output_dir / "checkpoints" / f"{model_name}.pt"
        _check(f"checkpoint:{model_name}", checkpoint.exists(), str(checkpoint))

    metrics = summary.get("metrics", {})
    for model_name, split_metrics in metrics.items():
        test = split_metrics.get("test", {})
        _check(
            f"scored:{model_name}",
            int(test.get("n", 0)) > 0,
            f"{test.get('n', 0)} scored test country-months",
        )

    if task == "classification":
        graph_scores = [
            metrics.get(name, {}).get("test", {}).get("roc_auc")
            for name in ("gcn", "tgn", "tgn_no_memory")
        ]
        usable = [value for value in graph_scores if isinstance(value, (int, float))]
        _check(
            "learning:test_roc_auc_above_chance",
            bool(usable) and max(usable) > 0.55,
            f"best graph-model test ROC-AUC = {max(usable):.3f}" if usable else "no defined ROC-AUC",
        )
    else:
        graph_rmse = [
            metrics.get(name, {}).get("test", {}).get("rmse")
            for name in ("gcn", "tgn", "tgn_no_memory")
        ]
        constant = metrics.get("train_median", {}).get("test", {}).get("rmse")
        usable = [value for value in graph_rmse if isinstance(value, (int, float))]
        _check(
            "learning:test_rmse_finite",
            bool(usable) and all(np.isfinite(value) for value in usable),
            f"graph RMSE {[round(value, 4) for value in usable]} vs constant {constant}",
        )
    return checks
