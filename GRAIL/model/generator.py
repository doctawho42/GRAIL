import torch
from torch.nn import Module, Sequential, ReLU, Linear, Bilinear
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Batch, Data
from torch_geometric import nn

class RuleParse(Module):

    def __init__(self, rule_dict: dict[str, Batch]) -> None:
        super(RuleParse, self).__init__()
        self.rule_dict = rule_dict
        self.module = nn.Sequential('x, edge_index, edge_attr', [
            (GATv2Conv(16, 100, edge_dim=18), 'x, edge_index, edge_attr -> x'),
            ReLU(inplace=True),
            (GATv2Conv(100, 200, edge_dim=18), 'x, edge_index, edge_attr -> x'),
            ReLU(inplace=True),
            Linear(200, 400),
        ])
        self.ffn = Sequential(
            Linear(400, 200),
            ReLU(inplace=True),
            Linear(200, 100),
            ReLU(inplace=True),
            Linear(100, 100)
        )

    def forward(self) -> torch.Tensor:
        batch = Batch.from_data_list(list(self.rule_dict.values()))
        batch.x = batch.x.to(torch.float32)
        batch.edge_attr = batch.edge_attr.to(torch.float32)
        batch.edge_index = batch.edge_index.to(torch.int64)
        data = self.module(batch.x, batch.edge_index, batch.edge_attr)
        x = global_mean_pool(data, batch.batch)
        x = self.ffn(x)
        return x

class Generator(Module):
    def __init__(self, rule_dict: dict[str, Batch]) -> None:
        super(Generator, self).__init__()
        self.parser = RuleParse(rule_dict)
        self.rules = rule_dict
        self.bilinear = Bilinear(100, 100, 1)
        self.module = nn.Sequential('x, edge_index, edge_attr', [
            (GATv2Conv(16, 100, edge_dim=18), 'x, edge_index, edge_attr -> x'),
        ])
        self.linear = nn.Linear(100, 100)

    def forward(self, data: Data) -> torch.Tensor:
        y = self.parser()
        data.x = data.x.to(torch.float32)
        data.edge_attr = data.edge_attr.to(torch.float32)
        data.edge_index = data.edge_index.to(torch.int64)
        x = self.module(data.x, data.edge_index, edge_attr=data.edge_attr)
        x = global_mean_pool(x, data.batch)
        x = x.repeat(len(self.rules), 1)
        x = self.bilinear(x, y)
        x = x.T.squeeze()
        return x