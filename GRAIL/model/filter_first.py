import torch
import torch_geometric.data
from torch import nn
from torch_geometric.nn import GATv2Conv, global_mean_pool

class Filter(nn.Module):
    def __init__(self) -> None:
        super(Filter, self).__init__()

        self.conv1_sub = GATv2Conv(10, 300, dropout=0.25, edge_dim=6)
        self.conv2_sub = GATv2Conv(300, 60, dropout=0.25, edge_dim=6)
        self.conv3_sub = GATv2Conv(60, 730, dropout=0.25, edge_dim=6)
        self.conv4_sub = GATv2Conv(730, 370, dropout=0.25, edge_dim=6)

        self.conv1 = GATv2Conv(10, 300, dropout=0.25, edge_dim=6)
        self.conv2 = GATv2Conv(300, 60, dropout=0.25, edge_dim=6)
        self.conv3 = GATv2Conv(60, 730, dropout=0.25, edge_dim=6)
        self.conv4 = GATv2Conv(730, 370, dropout=0.25, edge_dim=6)

        self.FCNN = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(1252, 680),
                nn.ReLU(),
                nn.BatchNorm1d(680),
                nn.Linear(680, 520),
                nn.ReLU(),
                nn.BatchNorm1d(520),
                nn.Linear(520, 260),
                nn.ReLU(),
                nn.BatchNorm1d(260),
                nn.Linear(260, 1),
                nn.Sigmoid()
                )

    def forward(self, sub: torch_geometric.data.Data, met: torch_geometric.data.Data) -> torch.Tensor:
        # 1. Metabolite
        met.x = met.x.to(torch.float32)
        met.edge_attr = met.edge_attr.to(torch.float32)
        node = self.conv1(met.x, met.edge_index, edge_attr=met.edge_attr)
        node = node.relu()
        node = self.conv2(node, met.edge_index, edge_attr=met.edge_attr)
        node = node.relu()
        node = self.conv3(node, met.edge_index, edge_attr=met.edge_attr)
        node = node.relu()
        node = self.conv4(node, met.edge_index, edge_attr=met.edge_attr)
        node = node.relu()

        node = global_mean_pool(node, met.batch)

        # 2. Substrate
        sub.x = sub.x.to(torch.float32)
        sub.edge_attr = sub.edge_attr.to(torch.float32)
        node_sub = self.conv1_sub(sub.x, sub.edge_index, edge_attr=sub.edge_attr)
        node_sub = node_sub.relu()
        node_sub = self.conv2_sub(node_sub, sub.edge_index, edge_attr=sub.edge_attr)
        node_sub = node_sub.relu()
        node_sub = self.conv3_sub(node_sub, sub.edge_index, edge_attr=sub.edge_attr)
        node_sub = node_sub.relu()
        node_sub = self.conv4_sub(node_sub, sub.edge_index, edge_attr=sub.edge_attr)
        node_sub = node_sub.relu()

        node_sub = global_mean_pool(node_sub, sub.batch)

        # 3. Apply a final classifier
        fp_sub = sub.fp.to(torch.float32)
        fp_met = met.fp.to(torch.float32)
        x = torch.cat((node_sub, fp_sub, node, fp_met), dim=1)
        x = self.FCNN(x)
        return x
