#!/usr/bin/env python3
"""Tests for the training scripts.

Runs with pytest::

    python -m pytest tests/test_training.py -v

and without it, so a machine with no test runner installed can still verify
the install::

    python tests/test_training.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

DATASET_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATASET_DIR))

from training import data as data_module  # noqa: E402
from training import metrics as metrics_module  # noqa: E402
from training import models as model_factory  # noqa: E402
from training import runtime  # noqa: E402
from training.data import DataValidationError  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _panel(countries: int = 3, months: int = 18, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    period = [str(value) for value in pd.period_range("2022-01", periods=months, freq="M")]
    rows = []
    for index in range(countries):
        node_id = f"country_{index}"
        for month_index, month in enumerate(period):
            rows.append(
                {
                    "month": month,
                    "node_id": node_id,
                    "host_country_id": node_id,
                    "inbound_flow_usd": float(1e8 * (1.0 + 0.1 * rng.standard_normal())),
                    **{feature: float(rng.random()) for feature in data_module.FEATURES},
                }
            )
    return pd.DataFrame(rows)


def _edges(frame: pd.DataFrame) -> pd.DataFrame:
    ids = sorted(frame["node_id"].unique())
    rows = []
    for month in sorted(frame["month"].unique()):
        for source in ids:
            for destination in ids:
                if source != destination:
                    rows.append(
                        {
                            "month": month,
                            "source": source,
                            "destination": destination,
                            "trade_value_usd": 1e6,
                            "flow_volume": 1e3,
                        }
                    )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Target construction
# --------------------------------------------------------------------------- #
def test_contraction_matches_the_documented_formula():
    """future vs the rolling median baseline, computed by hand."""
    values = [100.0, 100.0, 100.0, 50.0]
    frame = pd.DataFrame(
        {
            "month": ["2024-01", "2024-02", "2024-03", "2024-04"],
            "node_id": ["c"] * 4,
            "host_country_id": ["c"] * 4,
            "inbound_flow_usd": values,
            **{feature: [0.0] * 4 for feature in data_module.FEATURES},
        }
    )
    result, stats = data_module.build_targets(frame, horizon=1, baseline_min_periods=1, taus=[0.30])
    # March: future = 50, baseline = median(100, 100, 100) = 100 -> -0.5
    march = result[result["month"].eq("2024-03")].iloc[0]
    assert abs(float(march["contraction"]) + 0.5) < 1e-9
    assert bool(march["target_valid"]) is True
    assert float(march["derived_label_tau_0.30"]) == 1.0
    # April has no observable future value and must not become a negative.
    april = result[result["month"].eq("2024-04")].iloc[0]
    assert bool(april["target_valid"]) is False
    assert pd.isna(april["derived_label_tau_0.30"])
    assert stats["valid_targets"] == 3


def test_invalid_targets_are_never_relabelled_as_negatives():
    frame = _panel()
    frame.loc[frame["month"].eq("2022-05"), "inbound_flow_usd"] = 0.0
    result, _ = data_module.build_targets(frame, horizon=1, baseline_min_periods=1, taus=[0.35])
    labels = result.loc[~result["target_valid"], "derived_label_tau_0.35"]
    assert labels.isna().all()


def test_strict_baseline_requires_a_full_window():
    frame = _panel(countries=1, months=14)
    result, _ = data_module.build_targets(frame, horizon=1, baseline_window=12, baseline_min_periods=12)
    valid_months = sorted(result.loc[result["target_valid"], "month"].tolist())
    # The first eleven months cannot form a twelve-month baseline.
    assert valid_months[0] == "2022-12"


def test_stored_labels_reproduce_when_the_real_table_is_present():
    nodes_path = DATASET_DIR / "data/one_year_2024/processed/nodes_monthly.csv"
    if not nodes_path.exists():
        return  # the profile has not been built on this machine
    nodes = pd.read_csv(nodes_path)
    frame, _ = data_module.build_targets(nodes, horizon=1, baseline_min_periods=1, taus=[0.30, 0.35, 0.40])
    checks = data_module.cross_check_labels(frame, [0.30, 0.35, 0.40])
    for tau, check in checks.items():
        assert check["status"] == "match", f"tau={tau}: {check}"


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def _supervised(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    result, _ = data_module.build_targets(frame, horizon=1, baseline_min_periods=1)
    months = sorted(frame["month"].unique().tolist())
    return months, data_module.supervised_months(result, months)


def test_date_range_split_is_forward_chained():
    months, supervised = _supervised(_panel(months=18))
    split = data_module.resolve_split(
        months,
        supervised,
        {
            "mode": "date_ranges",
            "train": ["2022-01", "2022-10"],
            "validation": ["2022-11", "2023-02"],
            "test": ["2023-03", "2023-05"],
        },
    )
    assert max(split.train) < min(split.validation)
    assert max(split.validation) < min(split.test)
    assert not set(split.train) & set(split.test)


def test_split_rejects_overlapping_partitions():
    months, supervised = _supervised(_panel(months=18))
    try:
        data_module.resolve_split(
            months,
            supervised,
            {
                "mode": "date_ranges",
                "train": ["2022-01", "2022-10"],
                "validation": ["2022-09", "2023-02"],
                "test": ["2023-03", "2023-05"],
            },
        )
    except DataValidationError as error:
        assert "overlap" in str(error).lower()
        return
    raise AssertionError("overlapping partitions were accepted")


def test_split_rejects_counts_that_exceed_the_data():
    months, supervised = _supervised(_panel(months=10))
    try:
        data_module.resolve_split(
            months, supervised, {"mode": "counts", "train_months": 20, "validation_months": 5, "test_months": 5}
        )
    except DataValidationError as error:
        assert "supervised months" in str(error)
        return
    raise AssertionError("an impossible split was accepted")


def test_unsupervised_months_are_excluded_from_every_partition():
    frame = _panel(months=18)
    months, supervised = _supervised(frame)
    assert months[-1] not in supervised  # no next-month target for the last month
    split = data_module.resolve_split(
        months,
        supervised,
        {
            "mode": "date_ranges",
            "train": ["2022-01", "2022-10"],
            "validation": ["2022-11", "2023-02"],
            "test": ["2023-03", "2023-06"],
        },
    )
    assert months[-1] not in split.all_months()
    assert months[-1] in split.excluded_months


# --------------------------------------------------------------------------- #
# Fitted transforms
# --------------------------------------------------------------------------- #
def test_standardizer_is_fitted_on_training_rows_only():
    frame = _panel(months=18)
    train = frame[frame["month"] <= "2022-10"]
    standardizer = data_module.FeatureStandardizer().fit(train)
    transformed = standardizer.transform(train)
    assert np.allclose(transformed.mean(axis=0), 0.0, atol=1e-8)
    # Refitting on everything must move the statistics; if it did not, the
    # training-only fit would be meaningless.
    everything = data_module.FeatureStandardizer().fit(frame)
    assert not np.allclose(standardizer.mean, everything.mean)


def test_protective_feature_is_sign_flipped():
    frame = _panel(months=18)
    standardizer = data_module.FeatureStandardizer().fit(frame)
    index = data_module.FEATURES.index(data_module.PROTECTIVE_FEATURE)
    high = frame.iloc[[0]].copy()
    high[data_module.PROTECTIVE_FEATURE] = 1000.0
    low = high.copy()
    low[data_module.PROTECTIVE_FEATURE] = -1000.0
    assert standardizer.transform(high)[0, index] < standardizer.transform(low)[0, index]


def test_edge_scales_use_training_months_only():
    frame = _panel(months=18)
    edges = _edges(frame)
    edges.loc[edges["month"].eq("2023-05"), "trade_value_usd"] = 1e15
    train_months = sorted(frame["month"].unique())[:10]
    value_scale, _ = data_module.edge_scales(edges, train_months)
    assert value_scale < np.log1p(1e15)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_duplicate_country_months_are_rejected():
    frame = _panel(months=6)
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    try:
        data_module.validate_tables(frame, pd.DataFrame())
    except DataValidationError as error:
        assert "duplicate" in str(error).lower()
        return
    raise AssertionError("duplicate country-months were accepted")


def test_missing_feature_column_is_rejected():
    frame = _panel(months=6).drop(columns=["global_risk"])
    try:
        data_module.validate_tables(frame, pd.DataFrame())
    except DataValidationError as error:
        assert "global_risk" in str(error)
        return
    raise AssertionError("a missing feature column was accepted")


def test_bad_month_format_is_rejected():
    frame = _panel(months=6)
    frame.loc[0, "month"] = "March 2022"
    try:
        data_module.validate_tables(frame, pd.DataFrame())
    except DataValidationError as error:
        assert "YYYY-MM" in str(error)
        return
    raise AssertionError("a malformed month was accepted")


# --------------------------------------------------------------------------- #
# Tensorisation and leakage
# --------------------------------------------------------------------------- #
def test_month_batches_mask_only_valid_targets():
    frame = _panel(months=18)
    edges = _edges(frame)
    data_module.validate_tables(frame, edges)
    result, _ = data_module.build_targets(frame, horizon=1, baseline_min_periods=1)
    node_ids = sorted(frame["node_id"].unique().tolist())
    months = sorted(frame["month"].unique().tolist())
    standardizer = data_module.FeatureStandardizer().fit(result[result["month"].isin(months[:10])])
    batches = data_module.build_month_batches(
        result, edges, months, node_ids, standardizer, (1.0, 1.0), "contraction", torch.device("cpu")
    )
    assert batches[months[0]].features.shape == (len(node_ids), len(data_module.FEATURES))
    # The final month has no observable target anywhere.
    assert batches[months[-1]].valid_count == 0
    assert batches[months[0]].valid_count == len(node_ids)
    assert batches[months[0]].edge_index.shape[0] == 2


def test_no_target_column_reaches_the_feature_matrix():
    forbidden = {
        "inbound_flow_usd",
        "future_inbound_flow_usd",
        "baseline_inbound_flow_usd",
        "contraction",
        "target_valid",
    }
    assert not forbidden & set(data_module.FEATURES)
    assert not any(name.startswith("label") for name in data_module.FEATURES)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_single_class_metrics_are_null_not_zero():
    result = metrics_module.classification_metrics(np.zeros(20), np.random.rand(20))
    for name in ("precision", "recall", "f1", "roc_auc", "average_precision"):
        assert result[name] is None
    assert "only one class" in result["note"]


def test_empty_partition_metrics_are_null():
    result = metrics_module.classification_metrics(np.array([]), np.array([]))
    assert result["n"] == 0
    assert result["roc_auc"] is None


def test_regression_metrics_are_exact_for_a_known_case():
    result = metrics_module.regression_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0]))
    assert abs(result["mae"] - 1 / 3) < 1e-9
    assert abs(result["rmse"] - np.sqrt(1 / 3)) < 1e-9


# --------------------------------------------------------------------------- #
# Models and runtime
# --------------------------------------------------------------------------- #
def test_every_model_builds_for_every_task():
    device = torch.device("cpu")
    for task in model_factory.TASKS:
        for name in model_factory.GRAPH_MODELS:
            model, kwargs = model_factory.build_model(name, task, {"embedding_dim": 16}, 7, device)
            assert model_factory.parameter_count(model) > 0
            assert kwargs["feature_dim"] == 7


def test_classification_head_is_bounded_and_regression_head_is_not():
    """The contraction target is a signed ratio, so its head must not saturate."""
    device = torch.device("cpu")
    features = torch.full((4, 7), 3.0)
    edge_index = torch.empty((2, 0), dtype=torch.long)

    classifier, _ = model_factory.build_model("gcn", "classification", {"embedding_dim": 16}, 7, device)
    with torch.no_grad():
        probabilities = classifier(features, edge_index)
    assert torch.all((probabilities >= 0.0) & (probabilities <= 1.0))

    regressor, _ = model_factory.build_model("gcn", "regression", {"embedding_dim": 16}, 7, device)
    with torch.no_grad():
        regressor.head[-1].bias.fill_(7.5)
        values = regressor(features, edge_index)
    assert torch.isfinite(values).all()
    assert float(values.max()) > 1.0, "the regression head is saturating like a sigmoid"


def test_cuda_request_fails_loudly_when_unavailable():
    if torch.cuda.is_available():
        choice = runtime.resolve_device("cuda")
        assert choice.device.type == "cuda"
        return
    try:
        runtime.resolve_device("cuda")
    except runtime.EnvironmentError_ as error:
        assert "cuda" in str(error).lower()
        return
    raise AssertionError("a CUDA request silently fell back to the CPU")


def test_auto_device_always_resolves():
    assert runtime.resolve_device("auto").device.type in {"cpu", "cuda"}


def test_unknown_device_string_is_rejected():
    try:
        runtime.resolve_device("gpu0")
    except runtime.EnvironmentError_:
        return
    raise AssertionError("an unknown device string was accepted")


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #
def test_self_test_passes_on_this_machine():
    from training.selftest import run_self_test

    summary = run_self_test(device="cpu", epochs=3)
    failed = [check["check"] for check in summary["self_test_checks"] if not check["passed"]]
    assert summary["self_test_passed"], f"failed checks: {failed}"


def test_checkpoint_round_trips():
    import tempfile

    from training.models import load_checkpoint
    from training.report import save_checkpoint

    device = torch.device("cpu")
    model, kwargs = model_factory.build_model("tgn", "classification", {"embedding_dim": 16}, 7, device)
    split = data_module.Split(
        train=["2024-01"], validation=["2024-02"], test=["2024-03"], mode="date_ranges"
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "tgn.pt"
        save_checkpoint(
            model,
            path,
            model_name="tgn",
            task="classification",
            model_kwargs=kwargs,
            features=data_module.FEATURES,
            normalizer={"mean": [0.0] * 7, "scale": [1.0] * 7},
            edge_scale=(1.0, 1.0),
            node_order=["a", "b"],
            split=split,
            target_column="derived_label_tau_0.35",
            tau=0.35,
            training_manifest={"epochs_run": 1},
        )
        reloaded, metadata = load_checkpoint(path)
    assert metadata["task"] == "classification"
    assert metadata["features"] == data_module.FEATURES
    original = torch.nn.utils.parameters_to_vector(model.parameters())
    restored = torch.nn.utils.parameters_to_vector(reloaded.parameters())
    assert torch.allclose(original, restored)


def _main() -> int:
    tests = [
        (name, value)
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    failures: list[str] = []
    for name, test in tests:
        try:
            test()
            print(f"  PASS  {name}")
        except Exception as error:  # noqa: BLE001 - the runner reports every failure
            failures.append(f"{name}: {error}")
            print(f"  FAIL  {name}: {error}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
