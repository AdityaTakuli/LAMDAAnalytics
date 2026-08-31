"""Chronological training and evaluation loops for the graph models.

Both loops obey the same three rules:

* one optimisation step per monthly snapshot, in calendar order, never shuffled;
* the loss is computed only over country-months whose target is observable;
* temporal memory is reset once at the start of a replay and then carried
  forward, so validation and test months inherit exactly the state a deployed
  model would have, and never see a future month.

Model selection uses the validation partition only. The test partition is
scored once, at the end, with the selected weights.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from training import models as model_factory
from training.data import MonthBatch
from training.metrics import compute as compute_metrics
from training.metrics import selection_score

LOGGER = logging.getLogger("training.engine")


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float | None
    validation_loss: float | None
    validation_selection: float | None
    seconds: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "validation_loss": self.validation_loss,
            "validation_selection": self.validation_selection,
            "seconds": round(self.seconds, 3),
        }


@dataclass
class TrainingResult:
    model_name: str
    task: str
    epochs: list[EpochRecord] = field(default_factory=list)
    best_epoch: int | None = None
    best_selection: float | None = None
    best_validation_loss: float | None = None
    selection_rule: str = ""
    predictions: dict[str, np.ndarray] = field(default_factory=dict)
    seconds: float = 0.0
    stopped_early: bool = False
    parameter_count: int = 0

    def history(self) -> list[dict[str, Any]]:
        return [record.as_dict() for record in self.epochs]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "task": self.task,
            "parameters": self.parameter_count,
            "epochs_run": len(self.epochs),
            "best_epoch": self.best_epoch,
            "best_validation_selection": self.best_selection,
            "best_validation_loss": self.best_validation_loss,
            "selection_rule": self.selection_rule,
            "stopped_early": self.stopped_early,
            "train_seconds": round(self.seconds, 3),
            "epoch_history": self.history(),
        }


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def weighted_bce(probabilities: Tensor, target: Tensor, positive_weight: float) -> Tensor:
    """Class-weighted binary cross-entropy on probabilities in ``[0, 1]``.

    The models emit a sigmoid probability rather than a logit, so the
    probability is clamped before the log to keep the gradient finite.
    """
    probabilities = probabilities.clamp(1e-6, 1.0 - 1e-6)
    weights = torch.where(
        target > 0.5,
        torch.as_tensor(positive_weight, dtype=probabilities.dtype, device=probabilities.device),
        torch.as_tensor(1.0, dtype=probabilities.dtype, device=probabilities.device),
    )
    losses = -(weights * (target * torch.log(probabilities) + (1.0 - target) * torch.log(1.0 - probabilities)))
    return losses.mean()


def huber(prediction: Tensor, target: Tensor, beta: float = 0.1) -> Tensor:
    return torch.nn.functional.smooth_l1_loss(prediction, target, beta=float(beta))


def mse(prediction: Tensor, target: Tensor) -> Tensor:
    return torch.nn.functional.mse_loss(prediction, target)


def make_loss(task: str, positive_weight: float, regression_loss: str = "huber", huber_beta: float = 0.1):
    """Return ``loss(prediction, target) -> Tensor`` for the requested task."""
    if task == model_factory.CLASSIFICATION:
        return lambda prediction, target: weighted_bce(prediction, target, positive_weight)
    if regression_loss == "mse":
        return mse
    return lambda prediction, target: huber(prediction, target, huber_beta)


# --------------------------------------------------------------------------- #
# Forward helpers
# --------------------------------------------------------------------------- #
def forward(model: nn.Module, batch: MonthBatch, model_name: str) -> Tensor:
    if model_factory.is_temporal(model_name):
        return model(batch.features, events=batch.events, current_time=batch.time)
    return model(batch.features, batch.edge_index)


def _masked(prediction: Tensor, batch: MonthBatch) -> tuple[Tensor, Tensor]:
    return prediction[batch.mask], batch.target[batch.mask]


def _synchronise(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# --------------------------------------------------------------------------- #
# Replay
# --------------------------------------------------------------------------- #
@torch.no_grad()
def replay(
    model: nn.Module,
    model_name: str,
    batches: Mapping[str, MonthBatch],
    months: Sequence[str],
    node_count: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Score months in chronological order without updating any weights.

    For the temporal models the memory is reset once, before the first month,
    and then evolves forward. The replay must therefore always start at the
    first training month, even when only the test scores are wanted.
    """
    model.eval()
    if model_factory.is_temporal(model_name):
        model.reset_memory(node_count, device)
    scores: dict[str, np.ndarray] = {}
    for month in months:
        batch = batches[month]
        prediction = forward(model, batch, model_name)
        scores[month] = prediction.detach().cpu().numpy().astype(float)
    return scores


@torch.no_grad()
def replay_loss(
    model: nn.Module,
    model_name: str,
    batches: Mapping[str, MonthBatch],
    warmup_months: Sequence[str],
    scored_months: Sequence[str],
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    node_count: int,
    device: torch.device,
) -> tuple[float | None, dict[str, np.ndarray]]:
    """Mean loss and scores over ``scored_months`` after replaying ``warmup_months``."""
    model.eval()
    if model_factory.is_temporal(model_name):
        model.reset_memory(node_count, device)
        for month in warmup_months:
            forward(model, batches[month], model_name)
    losses: list[float] = []
    scores: dict[str, np.ndarray] = {}
    for month in scored_months:
        batch = batches[month]
        prediction = forward(model, batch, model_name)
        scores[month] = prediction.detach().cpu().numpy().astype(float)
        if batch.valid_count:
            predicted, target = _masked(prediction, batch)
            losses.append(float(loss_fn(predicted, target).detach().cpu()))
    return (float(np.mean(losses)) if losses else None), scores


def _gather(
    batches: Mapping[str, MonthBatch],
    months: Sequence[str],
    scores: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten masked (target, score) pairs across ``months``."""
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for month in months:
        batch = batches.get(month)
        if batch is None or month not in scores or not batch.valid_count:
            continue
        mask = batch.mask.detach().cpu().numpy()
        targets.append(batch.target.detach().cpu().numpy()[mask])
        predictions.append(np.asarray(scores[month])[mask])
    if not targets:
        return np.empty(0), np.empty(0)
    return np.concatenate(targets), np.concatenate(predictions)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_model(
    model: nn.Module,
    model_name: str,
    task: str,
    batches: Mapping[str, MonthBatch],
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
    """Fit one model and keep the weights that scored best on validation.

    ``patience`` enables early stopping. Selection prefers the validation
    metric returned by :func:`training.metrics.selection_score`; when that
    metric is undefined (a single-class validation split, for instance) the
    lowest validation loss is used instead, and the rule actually applied is
    recorded in the result.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = make_loss(task, positive_weight, regression_loss, huber_beta)
    temporal = model_factory.is_temporal(model_name)

    result = TrainingResult(
        model_name=model_name,
        task=task,
        parameter_count=model_factory.parameter_count(model),
    )
    best_state: dict[str, Tensor] | None = None
    best_selection: float | None = None
    best_loss: float | None = None
    epochs_without_improvement = 0
    started = time.perf_counter()

    trainable_months = [month for month in train_months if batches[month].valid_count > 0]
    if not trainable_months:
        raise RuntimeError(
            f"No training month contains an observable target; {model_name} cannot be fitted."
        )

    for epoch in range(1, int(epochs) + 1):
        epoch_started = time.perf_counter()
        model.train()
        if temporal:
            model.reset_memory(node_count, device)
        epoch_losses: list[float] = []
        for month in train_months:
            batch = batches[month]
            prediction = forward(model, batch, model_name)
            if not batch.valid_count:
                # Still replayed so the temporal state stays continuous.
                continue
            predicted, target = _masked(prediction, batch)
            loss = loss_fn(predicted, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        validation_loss, validation_scores = replay_loss(
            model, model_name, batches, train_months, validation_months, loss_fn, node_count, device
        )
        y_true, y_pred = _gather(batches, validation_months, validation_scores)
        validation_metrics = compute_metrics(task, y_true, y_pred, threshold=threshold)
        selection = selection_score(task, validation_metrics)
        _synchronise(device)

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
                improved = True
                best_selection = selection
            rule = "validation metric (higher is better)"
        elif validation_loss is not None:
            if best_loss is None or validation_loss < best_loss:
                improved = True
            rule = "validation loss (lower is better); the validation metric is undefined"
        else:
            improved = True
            rule = "last epoch; validation provides neither a metric nor a loss"
        result.selection_rule = rule

        if validation_loss is not None and (best_loss is None or validation_loss < best_loss):
            best_loss = validation_loss

        if improved:
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            result.best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if patience is not None and epochs_without_improvement >= patience:
                result.stopped_early = True
                LOGGER.info(
                    "  %-14s early stop after epoch %d (no validation improvement for %d epochs)",
                    model_name, epoch, patience,
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    result.best_selection = best_selection
    result.best_validation_loss = best_loss
    result.seconds = time.perf_counter() - started
    return result


def score_all_months(
    model: nn.Module,
    model_name: str,
    batches: Mapping[str, MonthBatch],
    ordered_months: Sequence[str],
    node_count: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Final chronological pass over train -> validation -> test."""
    return replay(model, model_name, batches, ordered_months, node_count, device)

