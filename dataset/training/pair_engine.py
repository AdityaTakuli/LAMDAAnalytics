"""Training loop for pooled bilateral pair-month models."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from training import models as model_factory
from training.engine import EpochRecord, TrainingResult, make_loss
from training.metrics import compute as compute_metrics
from training.metrics import selection_score
from training.pair_data import PairMonthBatch
from training.pair_models import is_graph, is_temporal

LOGGER = logging.getLogger("training.pair_engine")


def _advance_temporal(model: nn.Module, batch: PairMonthBatch, model_name: str) -> None:
    if is_temporal(model_name):
        model.tgn.process_events(batch.events, batch.features)


def forward_pair(model: nn.Module, batch: PairMonthBatch, model_name: str) -> Tensor:
    if model_name == "mlp":
        raise RuntimeError("MLP pair models use tabular batches, not graph pair batches.")
    if is_temporal(model_name):
        return model(
            batch.features,
            batch.events,
            batch.time,
            batch.sup_src,
            batch.sup_dst,
            batch.sup_edge_feat,
        )
    return model(
        batch.features,
        batch.edge_index,
        batch.sup_src,
        batch.sup_dst,
        batch.sup_edge_feat,
    )


@torch.no_grad()
def replay_pair(
    model: nn.Module,
    model_name: str,
    batches: Mapping[str, PairMonthBatch],
    months: Sequence[str],
    node_count: int,
    device: torch.device,
) -> dict[str, tuple[np.ndarray, list[tuple[str, str]]]]:
    model.eval()
    if is_temporal(model_name):
        model.reset_memory(node_count, device)
    scores: dict[str, tuple[np.ndarray, list[tuple[str, str]]]] = {}
    for month in months:
        batch = batches[month]
        if batch.valid_count == 0:
            _advance_temporal(model, batch, model_name)
            scores[month] = (np.empty(0), [])
            continue
        if model_name == "mlp":
            continue
        prediction = forward_pair(model, batch, model_name)
        scores[month] = (
            prediction.detach().cpu().numpy().astype(float),
            list(batch.pair_keys),
        )
    return scores


@torch.no_grad()
def replay_pair_loss(
    model: nn.Module,
    model_name: str,
    batches: Mapping[str, PairMonthBatch],
    warmup_months: Sequence[str],
    scored_months: Sequence[str],
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    node_count: int,
    device: torch.device,
) -> tuple[float | None, dict[str, tuple[np.ndarray, list[tuple[str, str]]]]]:
    model.eval()
    if is_temporal(model_name):
        model.reset_memory(node_count, device)
        for month in warmup_months:
            batch = batches[month]
            if batch.valid_count:
                forward_pair(model, batch, model_name)
            else:
                _advance_temporal(model, batch, model_name)
    losses: list[float] = []
    scores = replay_pair(model, model_name, batches, scored_months, node_count, device)
    for month in scored_months:
        batch = batches[month]
        if batch.valid_count == 0:
            continue
        prediction = forward_pair(model, batch, model_name)
        losses.append(float(loss_fn(prediction, batch.sup_target).detach().cpu()))
    return (float(np.mean(losses)) if losses else None), scores


def _gather_pair(
    batches: Mapping[str, PairMonthBatch],
    months: Sequence[str],
    scores: Mapping[str, tuple[np.ndarray, list[tuple[str, str]]]],
) -> tuple[np.ndarray, np.ndarray]:
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for month in months:
        batch = batches.get(month)
        payload = scores.get(month)
        if batch is None or payload is None or batch.valid_count == 0:
            continue
        targets.append(batch.sup_target.detach().cpu().numpy())
        predictions.append(np.asarray(payload[0]))
    if not targets:
        return np.empty(0), np.empty(0)
    return np.concatenate(targets), np.concatenate(predictions)


def train_pair_model(
    model: nn.Module,
    model_name: str,
    task: str,
    batches: Mapping[str, PairMonthBatch],
    train_months: Sequence[str],
    validation_months: Sequence[str],
    node_count: int,
    device: torch.device,
    epochs: int = 10,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    grad_clip: float = 5.0,
    patience: int | None = None,
    positive_weight: float = 1.0,
    regression_loss: str = "huber",
    huber_beta: float = 0.1,
    threshold: float = 0.5,
) -> TrainingResult:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = make_loss(task, positive_weight, regression_loss, huber_beta)
    temporal = is_temporal(model_name)

    result = TrainingResult(
        model_name=model_name,
        task=task,
        parameter_count=sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    )
    best_state: dict[str, Tensor] | None = None
    best_selection: float | None = None
    best_loss: float | None = None
    epochs_without_improvement = 0
    started = time.perf_counter()

    trainable_months = [month for month in train_months if batches[month].valid_count > 0]
    if not trainable_months:
        raise RuntimeError(f"No training month contains a supervised pair target for {model_name}.")

    for epoch in range(1, int(epochs) + 1):
        epoch_started = time.perf_counter()
        model.train()
        if temporal:
            model.reset_memory(node_count, device)
        epoch_losses: list[float] = []
        for month in train_months:
            batch = batches[month]
            if batch.valid_count == 0:
                _advance_temporal(model, batch, model_name)
                continue
            prediction = forward_pair(model, batch, model_name)
            loss = loss_fn(prediction, batch.sup_target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        validation_loss, validation_scores = replay_pair_loss(
            model,
            model_name,
            batches,
            train_months,
            validation_months,
            loss_fn,
            node_count,
            device,
        )
        y_true, y_pred = _gather_pair(batches, validation_months, validation_scores)
        validation_metrics = compute_metrics(task, y_true, y_pred, threshold=threshold)
        selection = selection_score(task, validation_metrics)

        record = EpochRecord(
            epoch=epoch,
            train_loss=float(np.mean(epoch_losses)) if epoch_losses else None,
            validation_loss=validation_loss,
            validation_selection=selection,
            seconds=time.perf_counter() - epoch_started,
        )
        result.epochs.append(record)
        LOGGER.info(
            "  %-14s epoch %2d/%-2d  train_loss=%s  val_loss=%s  val_selection=%s  (%.2fs)",
            model_name,
            epoch,
            epochs,
            f"{record.train_loss:.5f}" if record.train_loss is not None else "n/a",
            f"{validation_loss:.5f}" if validation_loss is not None else "n/a",
            f"{selection:.5f}" if selection is not None else "n/a",
            record.seconds,
        )

        improved = False
        if selection is not None:
            if best_selection is None or selection > best_selection:
                best_selection = selection
                improved = True
        elif validation_loss is not None:
            if best_loss is None or validation_loss < best_loss:
                best_loss = validation_loss
                improved = True

        if improved:
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            result.best_epoch = epoch
            result.best_selection = best_selection
            result.best_validation_loss = validation_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if patience is not None and epochs_without_improvement >= patience:
            result.stopped_early = True
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    result.seconds = time.perf_counter() - started
    result.selection_rule = (
        "best validation average_precision, else lowest validation loss"
        if task == model_factory.CLASSIFICATION
        else "lowest validation RMSE, else lowest validation loss"
    )
    return result


def train_mlp_tabular(
    model: nn.Module,
    pair_frame: pd.DataFrame,
    split,
    standardizer,
    target_column: str,
    task: str,
    device: torch.device,
    *,
    epochs: int = 40,
    learning_rate: float = 3e-3,
    positive_weight: float = 1.0,
    regression_loss: str = "huber",
    huber_beta: float = 0.1,
    threshold: float = 0.5,
    seed: int = 7,
) -> dict[str, Any]:
    """Fit the tabular MLP on all supervised training pair-month rows."""
    import pandas as pd
    from training.pair_baselines import _partition, _rows
    from training.pair_data import PAIR_PREDICTION_COLUMNS
    from training.engine import make_loss

    torch.manual_seed(seed)
    train = _partition(pair_frame, split.train, target_column)
    val = _partition(pair_frame, split.validation, target_column)
    if train.empty:
        raise RuntimeError("No training rows available for the tabular MLP.")

    train_x = torch.as_tensor(standardizer.transform(train), dtype=torch.float32, device=device)
    train_y = torch.as_tensor(
        pd.to_numeric(train[target_column], errors="coerce").to_numpy(dtype=float),
        dtype=torch.float32,
        device=device,
    )
    val_x = torch.as_tensor(standardizer.transform(val), dtype=torch.float32, device=device) if not val.empty else None
    val_y = (
        torch.as_tensor(
            pd.to_numeric(val[target_column], errors="coerce").to_numpy(dtype=float),
            dtype=torch.float32,
            device=device,
        )
        if not val.empty
        else None
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = make_loss(task, positive_weight, regression_loss, huber_beta)
    history: list[dict[str, Any]] = []
    best_state = None
    best_selection = None

    for epoch in range(1, int(epochs) + 1):
        model.train()
        prediction = model(train_x)
        loss = loss_fn(prediction, train_y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        val_loss = None
        val_selection = None
        if val_x is not None and val_y is not None and len(val_y):
            model.eval()
            with torch.no_grad():
                val_pred = model(val_x)
                val_loss = float(loss_fn(val_pred, val_y).detach().cpu())
                metrics = compute_metrics(
                    task,
                    val_y.detach().cpu().numpy(),
                    val_pred.detach().cpu().numpy(),
                    threshold=threshold,
                )
                val_selection = selection_score(task, metrics)
            model.train()

        record = {
            "epoch": epoch,
            "train_loss": float(loss.detach().cpu()),
            "validation_loss": val_loss,
            "validation_selection": val_selection,
        }
        history.append(record)
        if val_selection is not None and (best_selection is None or val_selection > best_selection):
            best_selection = val_selection
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    parts: list[pd.DataFrame] = []
    with torch.no_grad():
        for split_name, months in split.as_dict().items():
            subset = _partition(pair_frame, months, target_column)
            if subset.empty:
                continue
            features = torch.as_tensor(standardizer.transform(subset), dtype=torch.float32, device=device)
            scores = model(features).detach().cpu().numpy()
            parts.append(_rows(subset, "mlp", split_name, target_column, scores))
    predictions = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=PAIR_PREDICTION_COLUMNS)
    return {"predictions": predictions, "history": history}
