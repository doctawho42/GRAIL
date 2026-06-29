"""Stage 2a: minimal listwise-trained product-level reranker (the GO/DEAD gate).

A cross-encoder over the 18-dim substrate-product pair graph (``utils.transform.from_pair``)
plus a learned per-rule embedding -- the cross-rule signal. It scores one logit per
candidate, trained with a listwise InfoNCE objective per substrate (see
``workflows.reranker.listwise_infonce``). NO site channel, NO regio-sibling loss, NO RNS:
this is the minimal gate that asks whether a listwise objective on this architecture
beats the generator's own ranking. Headroom is ~96% cross-rule (which rule fires), so the
rule-context embedding is the lever and the pair graph is reused unchanged.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch_geometric.data import Batch

from ._graph import GraphEncoder
from ..utils.transform import EDGE_DIM, PAIR_NODE_DIM


class MinimalReranker(nn.Module):
    def __init__(
        self,
        in_channels: int = PAIR_NODE_DIM,
        edge_dim: int = EDGE_DIM,
        hidden_dims: Sequence[int] = (64, 128),
        out_dim: int = 128,
        n_rules: int = 0,
        rule_embed_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_rules = int(n_rules)
        self.rule_embed_dim = int(rule_embed_dim)
        # Shared GNN backbone over the merged MCS-aware pair graph (substrate+product).
        self.encoder = GraphEncoder(
            in_channels=in_channels,
            edge_dim=edge_dim,
            hidden_dims=list(hidden_dims),
            out_dim=out_dim,
            conv_kind="gatv2",
            dropout=dropout,
        )
        # Rule-context signal: the cross-rule lever. max(.,1) keeps Embedding valid when
        # n_rules == 0 (e.g. a degenerate config); rule_id is always clamped < n_rules.
        self.rule_embedding = nn.Embedding(max(self.n_rules, 1), self.rule_embed_dim)
        self.head = nn.Sequential(
            nn.Linear(out_dim + self.rule_embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, pair_batch: Batch, rule_id: torch.Tensor) -> torch.Tensor:
        """One scalar logit per candidate.

        ``pair_batch`` is a ``Batch`` of ``from_pair`` graphs (N candidates);
        ``rule_id`` is a ``LongTensor[N]`` of the firing rule per candidate. Concatenate
        the pooled graph embedding with the rule embedding, push through the MLP head,
        and squeeze to ``(N,)``.
        """
        graph_embed = self.encoder(pair_batch)  # (N, out_dim)
        rule_id = rule_id.to(graph_embed.device).long()
        rule_embed = self.rule_embedding(rule_id)  # (N, rule_embed_dim)
        features = torch.cat([graph_embed, rule_embed], dim=-1)
        return self.head(features).squeeze(-1)
