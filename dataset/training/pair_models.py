"""Graph and tabular models for bilateral pair-month supervision."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor, nn

from model_gcn import GraphConvolution  # noqa: E402
from model_tgn import TemporalGraphNetwork  # noqa: E402
from training.models import CLASSIFICATION, REGRESSION, TGN_KWARG_KEYS, gcn_kwargs, tgn_kwargs

GRAPH_PAIR_MODELS = ("gcn", "tgn", "tgn_no_memory")
TABULAR_PAIR_MODELS = ("mlp",)


class LinkHead(nn.Module):
    """Score a directed trade link from endpoint embeddings and edge features."""

    def __init__(self, embedding_dim: int, edge_dim: int = 2, task: str = CLASSIFICATION):
        super().__init__()
        input_dim = embedding_dim * 2 + edge_dim
        hidden = max(embedding_dim, 16)
        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        ]
        self.layers = nn.Sequential(*layers)
        self.task = task

    def forward(self, source: Tensor, destination: Tensor, edge_features: Tensor) -> Tensor:
        combined = torch.cat([source, destination, edge_features], dim=-1)
        output = self.layers(combined).squeeze(-1)
        if self.task == CLASSIFICATION:
            return torch.sigmoid(output)
        return output


class GCNPairModel(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 32, layers: int = 2, task: str = CLASSIFICATION):
        super().__init__()
        modules = [GraphConvolution(feature_dim, hidden_dim)]
        modules.extend(GraphConvolution(hidden_dim, hidden_dim) for _ in range(layers - 1))
        self.convolutions = nn.ModuleList(modules)
        self.link_head = LinkHead(hidden_dim, edge_dim=2, task=task)
        self.task = task

    def node_embeddings(self, features: Tensor, edge_index: Tensor) -> Tensor:
        hidden = features
        for layer in self.convolutions:
            hidden = torch.relu(layer(hidden, edge_index))
        return hidden

    def forward(
        self,
        features: Tensor,
        edge_index: Tensor,
        source_index: Tensor,
        destination_index: Tensor,
        edge_features: Tensor,
    ) -> Tensor:
        embeddings = self.node_embeddings(features, edge_index)
        return self.link_head(embeddings[source_index], embeddings[destination_index], edge_features)


class TGNPairModel(nn.Module):
    def __init__(self, feature_dim: int, use_memory: bool = True, task: str = CLASSIFICATION, **tgn_kwargs: Any):
        super().__init__()
        self.tgn = TemporalGraphNetwork(feature_dim=feature_dim, use_memory=use_memory, **tgn_kwargs)
        self.link_head = LinkHead(self.tgn.embedding_dim, edge_dim=2, task=task)
        self.task = task

    @property
    def use_memory(self) -> bool:
        return self.tgn.use_memory

    def reset_memory(self, num_nodes: int, device: torch.device | None = None) -> None:
        self.tgn.reset_memory(num_nodes, device)

    def forward(
        self,
        node_features: Tensor,
        events: list[Mapping[str, Any]],
        current_time: float,
        source_index: Tensor,
        destination_index: Tensor,
        edge_features: Tensor,
    ) -> Tensor:
        self.tgn.process_events(events, node_features)
        embeddings = self.tgn.node_embeddings(node_features, current_time)
        return self.link_head(embeddings[source_index], embeddings[destination_index], edge_features)


class MLPPairModel(nn.Module):
    """Tabular baseline on concatenated source, destination, and edge features."""

    def __init__(self, input_dim: int, task: str = CLASSIFICATION, hidden_dim: int = 64):
        super().__init__()
        self.task = task
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        output = self.network(features).squeeze(-1)
        if self.task == CLASSIFICATION:
            return torch.sigmoid(output)
        return output


def build_pair_model(
    name: str,
    task: str,
    model_config: Mapping[str, Any],
    feature_dim: int,
    pair_feature_dim: int,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    if name == "gcn":
        kwargs = gcn_kwargs(model_config, feature_dim)
        model: nn.Module = GCNPairModel(task=task, **kwargs)
    elif name in {"tgn", "tgn_no_memory"}:
        kwargs = tgn_kwargs(model_config, feature_dim, use_memory=(name == "tgn"))
        model = TGNPairModel(task=task, **kwargs)
    elif name == "mlp":
        kwargs = {"input_dim": int(pair_feature_dim), "hidden_dim": int(model_config.get("embedding_dim", 64))}
        model = MLPPairModel(task=task, **kwargs)
    else:
        raise ValueError(f"Unknown pair model {name!r}")
    return model.to(device), kwargs


def is_temporal(name: str) -> bool:
    return name in {"tgn", "tgn_no_memory"}


def is_graph(name: str) -> bool:
    return name in GRAPH_PAIR_MODELS


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
