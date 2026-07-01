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
from torch_geometric.data import Batch, Data

from ._graph import GraphEncoder
from ..utils.transform import EDGE_DIM, PAIR_NODE_DIM, SINGLE_NODE_DIM


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


class BiEncoderReranker(nn.Module):
    """Stage 2a (fair): a NO-MCS siamese bi-encoder reranker.

    The MCS pair-graph reranker had two confounds: (1) it ran ``rdFMCS`` per candidate
    (slow -> couldn't train at scale) and (2) it carried a learned per-rule
    ``nn.Embedding`` over 7581 rules that can't train on 120 substrates. This model
    removes both:

    * A SHARED ``GraphEncoder`` over the 16-dim SINGLE graph (``utils.transform.from_rdmol``)
      encodes the substrate and each product independently -- no merged MCS pair graph,
      no ``from_pair``, no ``rdFMCS`` (so the example build is ~seconds, not 23 s/example).
    * The rule signal is the empirical per-rule **prior log-odds scalar** feature
      (``generator.rule_prior_logits[rule_id]``), passed in as ``rule_prior``. There is NO
      ``nn.Embedding`` over rules anywhere in this module.

    ``forward`` encodes the substrate ONCE and broadcasts it across the N candidates, then
    builds the interaction features ``[sub, prod, sub*prod, |sub-prod|]``, concatenates the
    two scalar features ``rule_prior`` and ``gen_score``, and an MLP maps to one logit per
    candidate. Trained with the same per-substrate listwise InfoNCE objective as the pair
    reranker (``workflows.reranker.listwise_infonce``).
    """

    def __init__(
        self,
        in_channels: int = SINGLE_NODE_DIM,
        edge_dim: int = EDGE_DIM,
        hidden_dims: Sequence[int] = (64, 128),
        out_dim: int = 128,
        dropout: float = 0.1,
        use_rule_prior: bool = True,
        use_gen_score: bool = True,
    ) -> None:
        super().__init__()
        self.out_dim = int(out_dim)
        # Public alias for the pooled-embedding dim (the Set-GFlowNet StopHead / frontier
        # pooling in ``model.set_gflownet`` reads ``reranker.embed_dim``).
        self.embed_dim = int(out_dim)
        self.use_rule_prior = bool(use_rule_prior)
        self.use_gen_score = bool(use_gen_score)
        # Siamese: ONE encoder shared between substrate and products (single graphs).
        self.encoder = GraphEncoder(
            in_channels=in_channels,
            edge_dim=edge_dim,
            hidden_dims=list(hidden_dims),
            out_dim=out_dim,
            conv_kind="gatv2",
            dropout=dropout,
        )
        # Interaction features [sub, prod, sub*prod, |sub-prod|] = 4*out_dim, plus up to two
        # scalar features (rule_prior, gen_score). NO rule embedding.
        # When an ablation flag is False the corresponding scalar is zeroed out but the head
        # input dimension is unchanged so checkpoints remain compatible.
        head_in = 4 * self.out_dim + 2
        self.head = nn.Sequential(
            nn.Linear(head_in, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def encode_substrate(self, graphs) -> torch.Tensor:
        """Pooled single-graph embedding(s).

        ``graphs`` is a ``from_rdmol`` ``Data`` (one graph) or a ``Batch`` of them; returns
        ``(M, out_dim)`` pooled embeddings (M == number of graphs). This is the substrate
        encoding path factored out of ``forward`` so the Set-GFlowNet frontier pooling can
        reuse the trained encoder.

        BatchNorm1d in ``GraphEncoder`` raises on a 1-row (single-graph) batch in train mode.
        ``forward`` sidesteps this by encoding the substrate inside the joint product batch;
        callers that legitimately encode a lone graph (the STOP-head frontier pooling starts
        from ``[root]``) hit that edge. When there is a single graph we run the encoder's norm
        layers in eval for this call only (a coarse pooled rep -- not updating its running
        BatchNorm stats here is fine) and restore the prior mode. Gradients still flow.
        """
        device = next(self.parameters()).device
        batch = graphs
        if not isinstance(batch, Batch):
            batch = Batch.from_data_list([batch])
        batch = batch.to(device)
        n_graphs = int(batch.num_graphs) if hasattr(batch, "num_graphs") else 1
        if n_graphs <= 1:
            # Toggle only the encoder's BatchNorm layers to eval for this single-row encode;
            # keep them differentiable (no torch.no_grad) so gradients propagate.
            norm_layers = [m for m in self.encoder.modules() if isinstance(m, nn.BatchNorm1d)]
            prev = [m.training for m in norm_layers]
            for m in norm_layers:
                m.eval()
            try:
                out = self.encoder(batch)
            finally:
                for m, was_training in zip(norm_layers, prev):
                    m.train(was_training)
            return out
        return self.encoder(batch)

    def forward(
        self,
        sub_graph: Data,
        prod_batch: Batch,
        rule_prior: torch.Tensor,
        gen_score: torch.Tensor,
    ) -> torch.Tensor:
        """One scalar logit per candidate.

        ``sub_graph`` is a single ``from_rdmol`` graph for the (shared) substrate;
        ``prod_batch`` is a ``Batch`` of N ``from_rdmol`` product graphs; ``rule_prior`` and
        ``gen_score`` are length-N scalar feature tensors. The substrate is encoded once and
        broadcast over the N products.

        To keep ``BatchNorm1d`` valid (it errors on a 1-row batch in train mode), the
        substrate is encoded inside the SAME batch as the products and split back out.
        """
        device = next(self.parameters()).device
        # Encode substrate + products together so BatchNorm sees >1 graph even when the
        # caller scores a single product. The substrate is graph 0 of the joint batch.
        joint = Batch.from_data_list([sub_graph] + prod_batch.to_data_list()).to(device)
        embed = self.encoder(joint)  # (N+1, out_dim)
        sub_emb = embed[0:1]  # (1, out_dim)
        prod_emb = embed[1:]  # (N, out_dim)
        n = prod_emb.size(0)
        sub_b = sub_emb.expand(n, -1)  # broadcast the shared substrate over candidates

        interaction = torch.cat(
            [sub_b, prod_emb, sub_b * prod_emb, (sub_b - prod_emb).abs()], dim=-1
        )  # (N, 4*out_dim)
        rule_prior = rule_prior.to(device).float().view(n, 1)
        gen_score = gen_score.to(device).float().view(n, 1)
        # Ablation: zero out a scalar feature when its flag is disabled.
        # The head input dimension is kept the same so checkpoints stay compatible.
        if not self.use_rule_prior:
            rule_prior = torch.zeros_like(rule_prior)
        if not self.use_gen_score:
            gen_score = torch.zeros_like(gen_score)
        features = torch.cat([interaction, rule_prior, gen_score], dim=-1)
        return self.head(features).squeeze(-1)
