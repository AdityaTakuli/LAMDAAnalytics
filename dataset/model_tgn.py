"""Temporal Graph Network model used by ``train.py``.

This is a compact PyTorch implementation of the requested TGN components:

* learned harmonic time encoding;
* endpoint message MLP;
* mean aggregation of messages touching each node in a chronological batch;
* GRU memory updater;
* attention over recent temporal neighbors;
* MLP/sigmoid node-risk decoder.

The module also provides a deterministic linear composition fallback for
serving when a learned checkpoint cannot be loaded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import Tensor, nn

FEATURES = [
    "inventory_days_proxy",
    "trade_delay_proxy",
    "news_vol_7d",
    "neg_tone_frac_3d",
    "strike_flag_7d",
    "weather_anomaly_7d",
    "global_risk",
]

# Table IV-style deployment fallback weights.  They are configurable because
# the paper's exact learned coefficients are not shipped with this repository.
DEFAULT_LINEAR_WEIGHTS = {
    "inventory_days_proxy": 0.20,
    "trade_delay_proxy": 0.20,
    "news_vol_7d": 0.15,
    "neg_tone_frac_3d": 0.15,
    "strike_flag_7d": 0.15,
    "weather_anomaly_7d": 0.10,
    "global_risk": 0.05,
}


def orient_features(values: Tensor | np.ndarray | Mapping[str, float]) -> Tensor | dict[str, float]:
    """Sign-invert protective features before standardization."""
    if isinstance(values, Mapping):
        result = {name: float(values.get(name, 0.0)) for name in FEATURES}
        result["inventory_days_proxy"] = -result["inventory_days_proxy"]
        return result
    result = values.clone() if isinstance(values, Tensor) else np.asarray(values).copy()
    inventory_index = FEATURES.index("inventory_days_proxy")
    result[..., inventory_index] *= -1
    return result


def linear_composition(
    features: Mapping[str, float],
    weights: Mapping[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    """Return the fault-tolerant linear risk score and contributions."""
    weights = weights or DEFAULT_LINEAR_WEIGHTS
    oriented = orient_features(features)
    # Raw features are converted to bounded signals for this deployment
    # fallback.  A feature already in [0, 1] is retained as-is.
    signals: dict[str, float] = {}
    for name in FEATURES:
        value = float(oriented[name])
        if name == "inventory_days_proxy":
            value = 1.0 - np.clip(abs(value) / 90.0, 0.0, 1.0)
        elif name == "trade_delay_proxy":
            value = np.clip(value / 60.0, 0.0, 1.0)
        elif name == "news_vol_7d":
            value = np.clip(value / 20.0, 0.0, 1.0)
        signals[name] = float(np.clip(value, 0.0, 1.0))
    total = float(sum(weights.values())) or 1.0
    contribution = {name: float(weights.get(name, 0.0) * signals[name] / total) for name in FEATURES}
    return float(np.clip(sum(contribution.values()), 0.0, 1.0)), contribution


class HarmonicTimeEncoder(nn.Module):
    """Learned phi(delta_t) = cos(w * delta_t + theta)."""

    def __init__(self, dimension: int):
        super().__init__()
        self.frequency = nn.Parameter(torch.randn(dimension) * 0.1)
        self.phase = nn.Parameter(torch.zeros(dimension))

    def forward(self, delta_t: Tensor) -> Tensor:
        delta_t = delta_t.reshape(-1, 1).float()
        return torch.cos(delta_t * self.frequency.reshape(1, -1) + self.phase.reshape(1, -1))


class TemporalGraphNetwork(nn.Module):
    """A stateful, event-driven TGN for node-level risk."""

    def __init__(
        self,
        feature_dim: int = 7,
        memory_dim: int = 32,
        time_dim: int = 8,
        edge_dim: int = 2,
        message_dim: int = 64,
        embedding_dim: int = 32,
        max_neighbors: int = 20,
        use_memory: bool = True,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.memory_dim = memory_dim
        self.time_dim = time_dim
        self.edge_dim = edge_dim
        self.embedding_dim = embedding_dim
        self.max_neighbors = max_neighbors
        self.use_memory = use_memory

        self.time_encoder = HarmonicTimeEncoder(time_dim)
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * memory_dim + time_dim + edge_dim, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, memory_dim),
        )
        self.memory_updater = nn.GRUCell(memory_dim, memory_dim)
        attention_input = memory_dim + feature_dim + time_dim
        self.key_projection = nn.Linear(attention_input, embedding_dim)
        self.value_projection = nn.Linear(attention_input, embedding_dim)
        self.query_projection = nn.Linear(memory_dim, embedding_dim)
        self.embedding_projection = nn.Sequential(
            nn.Linear(memory_dim + feature_dim + embedding_dim, embedding_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )
        self.memory: Tensor | None = None
        self.neighbor_history: list[list[tuple[int, float]]] = []

    def reset_memory(self, num_nodes: int, device: torch.device | None = None) -> None:
        device = device or next(self.parameters()).device
        self.memory = torch.zeros(num_nodes, self.memory_dim, device=device)
        self.neighbor_history = [[] for _ in range(num_nodes)]

    def _require_memory(self, num_nodes: int, device: torch.device) -> Tensor:
        if self.memory is None or self.memory.shape[0] != num_nodes or len(self.neighbor_history) != num_nodes:
            self.reset_memory(num_nodes, device)
        if not self.use_memory:
            return torch.zeros(num_nodes, self.memory_dim, device=device)
        return self.memory

    def process_events(
        self,
        events: Iterable[Mapping[str, Any]],
        node_features: Tensor,
    ) -> None:
        """Process one chronological batch and update memory once per node.

        All endpoint messages are computed from the same pre-batch memory.
        Accumulating then averaging them gives the order-invariant mean
        aggregator requested for a batch.
        """
        events = list(events)
        if not events:
            return
        num_nodes = node_features.shape[0]
        device = node_features.device
        old_memory = self._require_memory(num_nodes, device).detach().clone()
        messages: dict[int, list[Tensor]] = {}
        for event in events:
            source = int(event["source_index"])
            destination = int(event["destination_index"])
            delta = torch.as_tensor([float(event.get("time_delta", 0.0))], dtype=torch.float32, device=device)
            edge = torch.as_tensor(
                event.get("edge_features", [event.get("trade_value_usd", 0.0), event.get("flow_volume", 0.0)]),
                dtype=torch.float32,
                device=device,
            ).reshape(1, -1)
            if edge.shape[1] != self.edge_dim:
                edge = torch.nn.functional.pad(edge, (0, max(0, self.edge_dim - edge.shape[1])))[:, : self.edge_dim]
            time_encoding = self.time_encoder(delta)
            message_input = torch.cat([old_memory[source:source + 1], old_memory[destination:destination + 1], time_encoding, edge], dim=1)
            message = self.message_mlp(message_input).squeeze(0)
            messages.setdefault(source, []).append(message)
            messages.setdefault(destination, []).append(message)
            event_time = float(event.get("time", 0.0))
            self.neighbor_history[source].append((destination, event_time))
            self.neighbor_history[destination].append((source, event_time))

        if self.use_memory:
            new_memory = old_memory.clone()
            for node, node_messages in messages.items():
                aggregate = torch.stack(node_messages, dim=0).mean(dim=0)
                new_memory[node] = self.memory_updater(aggregate.unsqueeze(0), old_memory[node].unsqueeze(0)).squeeze(0)
            self.memory = new_memory
        for node in range(num_nodes):
            if len(self.neighbor_history[node]) > self.max_neighbors * 2:
                self.neighbor_history[node] = self.neighbor_history[node][-self.max_neighbors * 2 :]

    def node_embeddings(self, node_features: Tensor, current_time: float) -> Tensor:
        num_nodes = node_features.shape[0]
        memory = self._require_memory(num_nodes, node_features.device)
        embeddings: list[Tensor] = []
        for node in range(num_nodes):
            neighbors = self.neighbor_history[node][-self.max_neighbors :]
            if neighbors:
                items: list[Tensor] = []
                for neighbor, event_time in neighbors:
                    delta = torch.tensor([max(0.0, current_time - event_time)], device=node_features.device)
                    time_encoding = self.time_encoder(delta).squeeze(0)
                    item = torch.cat([memory[neighbor], node_features[neighbor], time_encoding], dim=0)
                    items.append(item)
                stacked = torch.stack(items)
                keys = self.key_projection(stacked)
                values = self.value_projection(stacked)
                query = self.query_projection(memory[node]).reshape(1, -1)
                scores = (keys @ query.T).squeeze(-1) / np.sqrt(self.embedding_dim)
                context = torch.softmax(scores, dim=0) @ values
            else:
                context = torch.zeros(self.embedding_dim, device=node_features.device)
            embeddings.append(
                self.embedding_projection(torch.cat([memory[node], node_features[node], context], dim=0))
            )
        return torch.stack(embeddings)

    def forward(
        self,
        node_features: Tensor,
        events: Iterable[Mapping[str, Any]] = (),
        current_time: float = 0.0,
    ) -> Tensor:
        self.process_events(events, node_features)
        embeddings = self.node_embeddings(node_features, current_time)
        return torch.sigmoid(self.decoder(embeddings)).squeeze(-1)


def load_model_or_fallback(
    checkpoint: str | Path,
    **model_kwargs: Any,
) -> tuple[TemporalGraphNetwork | None, dict[str, Any]]:
    """Load a TGN state dict; return metadata indicating fallback on failure."""
    path = Path(checkpoint)
    if not path.exists():
        return None, {"mode": "linear_fallback", "reason": f"missing checkpoint: {path}"}
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = TemporalGraphNetwork(**model_kwargs)
        state_dict = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
        model.load_state_dict(state_dict)
        return model, {"mode": "tgn", "checkpoint": str(path)}
    except Exception as exc:
        return None, {"mode": "linear_fallback", "reason": str(exc), "checkpoint": str(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="tgn_model.pth")
    args = parser.parse_args()
    model, metadata = load_model_or_fallback(args.checkpoint)
    print(json.dumps(metadata, indent=2))
    if model is not None:
        print("Loaded TGN checkpoint.")


if __name__ == "__main__":
    main()

