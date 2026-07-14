"""Factorized generator: graph-level reaction-type head + atom-level site head.

Replaces the monolithic per-(substrate, rule) generator scorer with two heads sharing one
GraphEncoder: `type_logits` predicts *which reaction type* applies to the substrate
(graph-level, radius-0 reaction-type vocabulary), and `site_logits` predicts *where* it
applies (per-atom). Factoring type and site lets the model reuse the SoM-style localization
signal instead of re-encoding every candidate rule graph per substrate.

See docs/superpowers/specs (factorized dense-MLE generator redesign) for the full design.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Sequence, Union

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_mean_pool

from ..utils.transform import EDGE_DIM, SINGLE_NODE_DIM
from ._graph import GraphEncoder


class FactorizedGenerator(nn.Module):
    """Shared GNN encoder with a graph-level type head and a node-level site head."""

    def __init__(
        self,
        num_types: int,
        in_channels: int = SINGLE_NODE_DIM,
        edge_dim: int = EDGE_DIM,
        hidden_dims: Sequence[int] = (192, 256),
        out_dim: int = 128,
        conv_kind: str = "gatv2",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.arch = {
            "num_types": int(num_types),
            "in_channels": int(in_channels),
            "edge_dim": int(edge_dim),
            "hidden_dims": list(hidden_dims),
            "out_dim": int(out_dim),
            "conv_kind": conv_kind,
            "dropout": float(dropout),
        }
        self.encoder = GraphEncoder(in_channels, edge_dim, list(hidden_dims), out_dim, conv_kind=conv_kind, dropout=dropout)
        self.type_head = nn.Linear(out_dim, num_types)
        self.site_head = nn.Linear(out_dim, 1)

    def type_logits(self, data: Data) -> torch.Tensor:
        return self.type_head(self.encoder.forward(data))

    def site_logits(self, data: Data) -> torch.Tensor:
        return self.site_head(self.encoder.forward_nodes(data)).squeeze(-1)

    def fit(
        self,
        dataset: Sequence[Data],
        epochs: int = 15,
        lr: float = 1e-4,
        batch_size: int = 32,
    ) -> List[float]:
        """MLE training: BCEWithLogits on both heads (multi-label, NOT softmax CE — a
        substrate's `.y_type` is multi-hot and `.y_site` is per-atom multi-label).

        Returns the per-epoch mean loss (sum of the two BCE terms) so callers can assert it
        decreases (see `test_factorized_fit_reduces_loss`).
        """
        num_types = self.arch["num_types"]
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        history: List[float] = []
        indices = list(range(len(dataset)))
        self.train()
        for _ in range(max(epochs, 0)):
            random.shuffle(indices)
            total_loss = 0.0
            n_batches = 0
            for start in range(0, len(indices), max(batch_size, 1)):
                chunk = [dataset[i] for i in indices[start:start + batch_size]]
                if not chunk:
                    continue
                batch = Batch.from_data_list(chunk)
                opt.zero_grad()
                node_emb = self.encoder.forward_nodes(batch)
                graph_emb = global_mean_pool(node_emb, batch.batch)
                type_logits = self.type_head(graph_emb)
                site_logits = self.site_head(node_emb).squeeze(-1)

                y_type = batch.y_type.to(type_logits.dtype).view(-1, num_types)
                y_site = batch.y_site.to(site_logits.dtype).view(-1)

                loss = F.binary_cross_entropy_with_logits(type_logits, y_type) + \
                    F.binary_cross_entropy_with_logits(site_logits, y_site)
                loss.backward()
                opt.step()
                total_loss += float(loss)
                n_batches += 1
            history.append(total_loss / max(n_batches, 1))
        return history

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"arch": self.arch, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: Union[str, Path], map_location: str = "cpu") -> "FactorizedGenerator":
        state = torch.load(path, map_location=map_location, weights_only=False)
        arch: Optional[dict] = state.get("arch") if isinstance(state, dict) else None
        state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
        model = cls(**arch) if arch else cls(num_types=1)
        model.load_state_dict(state_dict)
        return model
