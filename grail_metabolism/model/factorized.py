"""Factorized generator: graph-level reaction-type head + atom-level site head.

Replaces the monolithic per-(substrate, rule) generator scorer with two heads sharing one
GraphEncoder: `type_logits` predicts *which reaction type* applies to the substrate
(graph-level, radius-0 reaction-type vocabulary), and `site_logits` predicts *where* it
applies (per-atom). Factoring type and site lets the model reuse the SoM-style localization
signal instead of re-encoding every candidate rule graph per substrate.

See docs/superpowers/specs (factorized dense-MLE generator redesign) for the full design.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch_geometric.data import Data

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
