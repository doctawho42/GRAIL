import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter_add, scatter_mean, scatter_softmax

from torch.nn import Sequential, ReLU
from torch_geometric.nn import (
    global_add_pool,
    global_mean_pool,
)

from typing import Callable, Union

from torch import Tensor

from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import OptPairTensor


class ContinuousAtomEncoder(nn.Module):
    """Encoder for continuous atom features instead of discrete"""

    def __init__(self, hidden_dim, input_dim=18):
        super(ContinuousAtomEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        return self.linear(x)


class ContinuousBondEncoder(nn.Module):
    """Encoder for continuous bond features instead of discrete"""

    def __init__(self, hidden_dim, input_dim=18):
        super(ContinuousBondEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(input_dim, hidden_dim)

    def forward(self, edge_attr):
        return self.linear(edge_attr)


class AttentionModule(nn.Module):
    def __init__(self, input_dim):
        super(AttentionModule, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, query, keys):
        atten_scores_list = []
        for key in keys:
            atten_scores = torch.sum(query * key, dim=0)
            max_scores = atten_scores.max()
            min_scores = atten_scores.min()
            norm_scores = (atten_scores - min_scores) / (max_scores - min_scores + 1e-6)
            atten_scores = self.linear(norm_scores)
            atten_scores_list.append(atten_scores)

        stacked_attention_scores = torch.stack(atten_scores_list, dim=-1)
        atten_weights = F.softmax(stacked_attention_scores, dim=-1)

        atten_rep = torch.sum(torch.stack(keys, dim=-1) * atten_weights, dim=-1)
        fina_atten_rep = atten_rep + query

        return fina_atten_rep


class EdgePathNN(nn.Module):
    def __init__(
            self,
            hidden_dim,
            cutoff,
            y,
            n_classes,
            device,
            residuals=False,
            encode_distances=False,
            use_edge_attr=False,
            node_feat_dim=18,  # Your node feature dimension
            edge_feat_dim=18,  # Your edge feature dimension
            readout="sum",
            path_agg="sum",
            dropout=0,
            use_fingerprint=False,  # Whether to use fingerprint features
            fingerprint_dim=1024,  # Morgan fingerprint dimension
    ):
        """
        Modified EdgePathNN compatible with continuous features
        """
        super(EdgePathNN, self).__init__()
        self.cutoff = cutoff
        self.device = device
        self.residuals = residuals
        self.dropout = dropout
        self.encode_distances = encode_distances
        self.attention_module = AttentionModule(hidden_dim)
        self.y = y
        self.use_fingerprint = use_fingerprint

        # Use continuous encoders instead of discrete embeddings
        self.feature_encoder = ContinuousAtomEncoder(hidden_dim, node_feat_dim)
        self.bond_encoder = ContinuousBondEncoder(hidden_dim, edge_feat_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)

        # Distance Encoding + LSTM parameters for DE + Edge
        if encode_distances:
            self.distance_encoder = nn.Embedding(cutoff, hidden_dim)
            self.lstm = nn.LSTM(
                input_size=hidden_dim * 3,
                hidden_size=hidden_dim,
                batch_first=True,
                bidirectional=False,
                num_layers=1,
                bias=True,
            )
        else:
            self.lstm = nn.LSTM(
                input_size=hidden_dim * 2,
                hidden_size=hidden_dim,
                batch_first=True,
                bidirectional=False,
                num_layers=1,
                bias=True,
            )

        # 1 MLP per conv layer
        self.convs = nn.ModuleList([])
        for i in range(self.cutoff - 1):
            mlp = Sequential(
                Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                ReLU(),
            )
            if path_agg == "sum":
                bn = nn.BatchNorm1d(hidden_dim)
            else:
                bn = None

            self.convs.append(
                EdgePathConv(
                    hidden_dim,
                    self.lstm,
                    mlp,
                    bn,
                    residuals=self.residuals,
                    path_agg=path_agg,
                    dropout=self.dropout,
                )
            )

        self.use_edge_attr = use_edge_attr
        self.hidden_dim = hidden_dim

        # Handle fingerprint features if used
        if self.use_fingerprint:
            self.fp_encoder = nn.Linear(fingerprint_dim, hidden_dim)
            self.linear1 = Linear(hidden_dim * 2, hidden_dim)  # Combine graph + fingerprint features
        else:
            self.linear1 = Linear(hidden_dim, hidden_dim)

        self.linear2 = Linear(hidden_dim, n_classes)

        if readout == "sum":
            self.pooling = global_add_pool
        elif readout == "mean":
            self.pooling = global_mean_pool
        self.reset_parameters()

    def reset_parameters(self):
        # Reset continuous encoders
        nn.init.xavier_uniform_(self.feature_encoder.linear.weight.data)
        nn.init.xavier_uniform_(self.bond_encoder.linear.weight.data)

        for conv in self.convs:
            conv.reset_parameters()
        self.lstm.reset_parameters()

        if hasattr(self, "distance_encoder"):
            nn.init.xavier_uniform_(self.distance_encoder.weight.data)
        if hasattr(self, "fp_encoder"):
            nn.init.xavier_uniform_(self.fp_encoder.weight.data)

        self.linear1.reset_parameters()
        self.linear2.reset_parameters()

    def forward(self, data):
        # Handle fingerprint if present
        if self.use_fingerprint and hasattr(data, 'fp'):
            fp_encoded = self.fp_encoder(data.fp)

        # Save the result of each convolution at every step using a list
        conv_results = []

        # Map node initial features to d-dim space
        # [n_nodes, hidden_size]
        W_0 = self.feature_encoder(data.x)

        # Map initial edge features to d-dim space
        # [n_edges, hidden_size]
        edge_attr = self.bond_encoder(data.edge_attr)

        for i in range(self.cutoff - 1):
            # Distance encoding - check if paths exist in data
            if self.encode_distances and hasattr(data, f"sp_dists_{i + 2}"):
                dist_emb = self.distance_encoder(getattr(data, f"sp_dists_{i + 2}"))
            else:
                dist_emb = None

            # Get edge feature with respect to paths
            if hasattr(data, f"edge_indices_{i + 2}") and hasattr(data, f"path_{i + 2}"):
                edge_indices = getattr(data, f"edge_indices_{i + 2}")
                edge_attr_in = edge_attr[edge_indices]
                paths = getattr(data, f"path_{i + 2}")
            else:
                # If paths don't exist, skip this convolution
                continue

            # Update node representations
            if i == 0:
                W_1 = self.convs[i](
                    W_0, paths, edge_attr_in, dist_emb=dist_emb
                )
            else:
                W = (1 - self.y) * W_0 - self.y * torch.sum(torch.stack(conv_results, dim=-1), dim=-1)
                W_1 = self.convs[i](
                    W, paths, edge_attr_in, dist_emb=dist_emb
                )

            conv_results.append(W_1)

        # If no convolutions were applied, use initial features
        if len(conv_results) == 0:
            out = W_0
        else:
            # get attention result
            out = self.attention_module(W_0, conv_results)

        # Readout and predict
        # [n_graphs, hidden_size]
        out = self.pooling(out, data.batch)

        # Combine with fingerprint features if available
        if self.use_fingerprint and hasattr(data, 'fp'):
            # Assuming data.fp is already batched appropriately
            if fp_encoded.size(0) == out.size(0):  # Same batch size
                out = torch.cat([out, fp_encoded], dim=1)
        out = self.linear1(out)
        out = F.relu(out)
        out = self.linear2(out)
        return out


class EdgePathConv(torch.nn.Module):
    def __init__(
            self,
            hidden_dim,
            rnn: Callable,
            mlp: Callable,
            batch_norm: Callable,
            residuals=True,
            path_agg="sum",
            dropout=0.5,
    ):
        super(EdgePathConv, self).__init__()
        self.nn = mlp
        self.rnn = rnn
        self.bn = batch_norm
        if self.bn is None:
            self.bn = nn.Identity()

        self.hidden_dim = hidden_dim
        self.residuals = residuals
        self.dropout = dropout
        self.path_agg = path_agg

        if path_agg == "sum":
            self.path_pooling = scatter_add
        elif path_agg == "mean":
            self.path_pooling = scatter_mean

        self.reset_parameters()

    def reset_parameters(self):
        if self.nn is not None:
            for c in self.nn.children():
                if hasattr(c, "reset_parameters"):
                    c.reset_parameters()
        if hasattr(self.bn, "reset_parameters"):
            self.bn.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], paths, edge_attr, dist_emb=None):
        # Adding a vector of zeros to first edge features entry as there are no relevant edge at first sequence input
        # [n_paths, path_length, hidden_size]
        edge_attr = torch.cat(
            [
                torch.zeros(paths.size(0), 1, self.hidden_dim, device=x.device),
                edge_attr,
            ],
            dim=1,
        )
        # [n_paths, path_length, hidden_size]
        x_cat = x[paths]

        # Distance Encoding
        if dist_emb is not None:
            # [n_paths, path_length, hidden_size * 3]
            x_cat = torch.cat([x_cat, dist_emb, edge_attr], dim=-1)
        else:
            # [n_paths, path_length, hidden_size * 2]
            x_cat = torch.cat([x_cat, edge_attr], dim=-1)

        # Applying dropout to lstm input
        x_cat = F.dropout(x_cat, training=self.training, p=self.dropout)

        # Path representations
        # [1, n_paths, hidden_size]
        _, (h, _) = self.rnn(x_cat)

        # Summing paths to get intermediate node representations
        # [n_nodes, hidden_size]
        h = self.path_pooling(
            h.squeeze(0),
            paths[:, -1],
            dim=0,
            out=torch.zeros(x.size(0), self.hidden_dim, device=x.device),
        )

        # Applying residuals connections + BN
        if self.residuals:
            h = self.bn(x + h)
        else:
            h = self.bn(h)

        # Dropout before MLP
        h = F.dropout(h, training=self.training, p=self.dropout)
        # 2-layers MLP for phi function
        # [n_nodes, hidden_size]
        h = self.nn(h)

        return h