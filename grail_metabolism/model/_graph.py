from __future__ import annotations

from typing import Iterable, List, Literal, Sequence

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, GCNConv, GINConv, global_mean_pool


class GraphEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        edge_dim: int,
        hidden_dims: Sequence[int],
        out_dim: int,
        conv_kind: Literal["gatv2", "gcn", "gin"] = "gatv2",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must not be empty")
        self.conv_kind = conv_kind
        self.dropout = nn.Dropout(dropout)
        dims = [in_channels, *hidden_dims, out_dim]
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for left, right in zip(dims[:-1], dims[1:]):
            self.convs.append(self._build_conv(left, right, edge_dim=edge_dim, conv_kind=conv_kind))
            self.norms.append(nn.BatchNorm1d(right))

    def _build_conv(self, in_channels: int, out_channels: int, edge_dim: int, conv_kind: str) -> nn.Module:
        if conv_kind == "gatv2":
            return GATv2Conv(in_channels, out_channels, edge_dim=edge_dim, dropout=float(self.dropout.p))
        if conv_kind == "gcn":
            return GCNConv(in_channels, out_channels)
        if conv_kind == "gin":
            mlp = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.ReLU(inplace=True),
                nn.Linear(out_channels, out_channels),
            )
            return GINConv(mlp)
        raise ValueError(f"Unsupported conv kind: {conv_kind}")

    def _apply_conv(self, conv: nn.Module, x, edge_index, edge_attr):
        if self.conv_kind == "gatv2":
            return conv(x, edge_index, edge_attr)
        return conv(x, edge_index)

    def forward_nodes(self, data: Data) -> torch.Tensor:
        x = data.x.float()
        edge_index = data.edge_index.long()
        edge_attr = data.edge_attr.float() if getattr(data, "edge_attr", None) is not None else None
        for conv, norm in zip(self.convs, self.norms):
            x = self._apply_conv(conv, x, edge_index, edge_attr)
            x = norm(x)
            x = torch.relu(x)
            x = self.dropout(x)
        return x

    def forward(self, data: Data) -> torch.Tensor:
        x = self.forward_nodes(data)
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return global_mean_pool(x, batch)
