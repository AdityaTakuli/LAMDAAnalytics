"""Model construction for the classification and regression tasks.

The classification models are the repository's existing ``SnapshotGCN`` and
``TemporalGraphNetwork`` used unchanged, so their checkpoints stay compatible
with the rest of the project.

The regression models reuse the same layers but replace the bounded sigmoid
risk decoder with a linear head, because the contraction target is a signed
ratio and is not confined to ``[0, 1]``. Nothing else about the architecture
changes, which keeps the two tasks comparable.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import torch
from torch import Tensor, nn

from training import paths  # noqa: F401  (sys.path bootstrap)

from model_gcn import GraphConvolution, SnapshotGCN  # noqa: E402
from model_tgn import TemporalGraphNetwork  # noqa: E402

CLASSIFICATION = "classification"
REGRESSION = "regression"
TASKS = (CLASSIFICATION, REGRESSION)

GRAPH_MODELS = ("gcn", "tgn", "tgn_no_memory")
TGN_KWARG_KEYS = ("memory_dim", "time_dim", "edge_dim", "message_dim", "embedding_dim", "max_neighbors")


class GCNRegressor(nn.Module):
    """``SnapshotGCN`` layers with an unbounded linear output head."""

    def __init__(self, feature_dim: int = 7, hidden_dim: int = 32, layers: int = 2):
        super().__init__()
        if layers < 1:
            raise ValueError("GCN requires at least one graph-convolution layer")
        modules = [GraphConvolution(feature_dim, hidden_dim)]
        modules.extend(GraphConvolution(hidden_dim, hidden_dim) for _ in range(layers - 1))
        self.convolutions = nn.ModuleList(modules)
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, features: Tensor, edge_index: Tensor) -> Tensor:
        hidden = features
        for layer in self.convolutions:
            hidden = torch.relu(layer(hidden, edge_index))
        return self.head(hidden).squeeze(-1)


class TGNRegressor(nn.Module):
    """``TemporalGraphNetwork`` embeddings with an unbounded linear head.

    Memory, messages, time encoding, and temporal attention are the wrapped
    module's own; only the decoder differs.
    """

    def __init__(self, feature_dim: int = 7, use_memory: bool = True, **tgn_kwargs: Any):
        super().__init__()
        self.tgn = TemporalGraphNetwork(feature_dim=feature_dim, use_memory=use_memory, **tgn_kwargs)
        embedding_dim = self.tgn.embedding_dim
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )

    @property
    def use_memory(self) -> bool:
        return self.tgn.use_memory

    def reset_memory(self, num_nodes: int, device: torch.device | None = None) -> None:
        self.tgn.reset_memory(num_nodes, device)

    def forward(
        self,
        node_features: Tensor,
        events: Iterable[Mapping[str, Any]] = (),
        current_time: float = 0.0,
    ) -> Tensor:
        self.tgn.process_events(events, node_features)
        embeddings = self.tgn.node_embeddings(node_features, current_time)
        return self.head(embeddings).squeeze(-1)


def gcn_kwargs(model_config: Mapping[str, Any], feature_dim: int) -> dict[str, Any]:
    return {
        "feature_dim": int(feature_dim),
        "hidden_dim": int(model_config.get("embedding_dim", 32)),
        "layers": int(model_config.get("gcn_layers", 2)),
    }


def tgn_kwargs(model_config: Mapping[str, Any], feature_dim: int, use_memory: bool) -> dict[str, Any]:
    kwargs = {key: int(model_config[key]) for key in TGN_KWARG_KEYS if key in model_config}
    kwargs["feature_dim"] = int(feature_dim)
    kwargs["use_memory"] = bool(use_memory)
    return kwargs


def build_model(
    name: str,
    task: str,
    model_config: Mapping[str, Any],
    feature_dim: int,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    """Instantiate one model on ``device`` and return it with its kwargs.

    The kwargs are stored in the checkpoint so a saved model can be rebuilt
    without the config file that produced it.
    """
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}; expected one of {TASKS}")
    if name not in GRAPH_MODELS:
        raise ValueError(f"Unknown graph model {name!r}; expected one of {GRAPH_MODELS}")

    if name == "gcn":
        kwargs = gcn_kwargs(model_config, feature_dim)
        model: nn.Module = SnapshotGCN(**kwargs) if task == CLASSIFICATION else GCNRegressor(**kwargs)
    else:
        kwargs = tgn_kwargs(model_config, feature_dim, use_memory=(name == "tgn"))
        model = TemporalGraphNetwork(**kwargs) if task == CLASSIFICATION else TGNRegressor(**kwargs)

    model = model.to(device)
    return model, kwargs


def is_temporal(name: str) -> bool:
    return name in {"tgn", "tgn_no_memory"}


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def load_checkpoint(path: Any, device: torch.device | str = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild a trained model from a checkpoint written by ``train_models.py``.

    The checkpoint carries its own ``model_name``, ``task``, and ``model_kwargs``,
    so no config file is needed::

        from training.models import load_checkpoint
        model, meta = load_checkpoint("…/checkpoints/tgn.pt", device="cuda")
        print(meta["features"], meta["normalizer"]["mean"])
    """
    device = torch.device(device)
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"{path} is not a train_models.py checkpoint")

    name = payload.get("model_name", "tgn")
    task = payload.get("task", CLASSIFICATION)
    kwargs = dict(payload.get("model_kwargs", {}))

    if name == "gcn":
        model: nn.Module = SnapshotGCN(**kwargs) if task == CLASSIFICATION else GCNRegressor(**kwargs)
    else:
        model = TemporalGraphNetwork(**kwargs) if task == CLASSIFICATION else TGNRegressor(**kwargs)

    model.load_state_dict(payload["model_state_dict"])
    model = model.to(device)
    model.eval()
    metadata = {key: value for key, value in payload.items() if key != "model_state_dict"}
    return model, metadata
