"""Snapshot Graph Convolutional Network baseline.

This is intentionally a simple, dependency-light GCN baseline for the
ablation comparison. It performs mean neighbor aggregation independently for
each monthly graph snapshot and has no temporal memory. TGN remains the model
that carries state across chronologically ordered events.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class GraphConvolution(nn.Module):
    """Directed mean-neighbor graph convolution with explicit self features."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.self_linear = nn.Linear(in_features, out_features)
        self.neighbor_linear = nn.Linear(in_features, out_features)

    def forward(self, features: Tensor, edge_index: Tensor) -> Tensor:
        if edge_index.numel() == 0:
            neighbors = torch.zeros_like(features)
        else:
            source, destination = edge_index
            neighbors = torch.zeros_like(features)
            neighbors.index_add_(0, destination, features[source])
            degree = torch.zeros(features.shape[0], device=features.device)
            degree.index_add_(
                0,
                destination,
                torch.ones(destination.shape[0], device=features.device),
            )
            neighbors = neighbors / degree.clamp_min(1.0).unsqueeze(1)
        return self.self_linear(features) + self.neighbor_linear(neighbors)


class SnapshotGCN(nn.Module):
    """Two-layer GCN with binary or continuous node decoder."""

    def __init__(
        self,
        feature_dim: int = 7,
        hidden_dim: int = 32,
        layers: int = 2,
        output_activation: str = "sigmoid",
    ):
        super().__init__()
        if layers < 1:
            raise ValueError("GCN requires at least one graph-convolution layer")
        if output_activation not in {"sigmoid", "linear"}:
            raise ValueError("output_activation must be 'sigmoid' or 'linear'")
        self.output_activation = output_activation
        modules = [GraphConvolution(feature_dim, hidden_dim)]
        modules.extend(GraphConvolution(hidden_dim, hidden_dim) for _ in range(layers - 1))
        self.convolutions = nn.ModuleList(modules)
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: Tensor, edge_index: Tensor) -> Tensor:
        hidden = features
        for layer in self.convolutions:
            hidden = torch.relu(layer(hidden, edge_index))
        output = self.decoder(hidden).squeeze(-1)
        return torch.sigmoid(output) if self.output_activation == "sigmoid" else output


def undirected_edge_index(
    events: list[dict], device: torch.device | None = None
) -> Tensor:
    """Convert event dictionaries to a bidirectional snapshot edge index."""
    device = device or torch.device("cpu")
    if not events:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    forward = torch.tensor(
        [
            [int(event["source_index"]) for event in events],
            [int(event["destination_index"]) for event in events],
        ],
        dtype=torch.long,
        device=device,
    )
    return torch.cat([forward, forward.flip(0)], dim=1)

