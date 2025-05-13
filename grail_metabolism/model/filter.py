import torch
import numpy as np
import typing as tp
from torch.nn import Module, Sequential, ReLU, Linear, BatchNorm1d, Dropout, Sigmoid
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric import nn
import torch_geometric
from grail_metabolism.model.wrapper import GFilter
from grail_metabolism.utils.preparation import MolFrame, cpunum
from grail_metabolism.utils.transform import from_rule, from_rdmol, from_pair
from rdkit import Chem
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm
from grail_metabolism.model.train_model import PULoss
from multipledispatch import dispatch

class Filter(GFilter):
    def __init__(self, in_channels, edge_dim, arg_vec: tp.List[int], mode: tp.Literal['single', 'pair']):
        super(Filter, self).__init__()
        if mode == 'pair':
            self.module = nn.Sequential('x, edge_index, edge_attr', [
                (GATv2Conv(in_channels, arg_vec[0], edge_dim=edge_dim, dropout=0.25), 'x, edge_index, edge_attr -> x'),
                ReLU(inplace=True),
                (GATv2Conv(arg_vec[0], arg_vec[1], edge_dim=edge_dim, dropout=0.25), 'x, edge_index, edge_attr -> x'),
                ReLU(inplace=True),
                (GATv2Conv(arg_vec[1], arg_vec[2], edge_dim=edge_dim, dropout=0.25), 'x, edge_index, edge_attr -> x'),
                ReLU(inplace=True)
            ])
            self.dropout = Dropout(0.1)
            self.lin1 = Linear((arg_vec[2] + 256 * 2), arg_vec[3])
            self.bn1 = BatchNorm1d(arg_vec[3])
            self.lin2 = Linear(arg_vec[3], arg_vec[4])
            self.bn2 = BatchNorm1d(arg_vec[4])
            self.lin3 = Linear(arg_vec[4], arg_vec[5])
            self.bn3 = BatchNorm1d(arg_vec[5])
            self.lin4 = Linear(arg_vec[5], 1)
        elif mode == 'single':
            self.conv1_sub = GATv2Conv(in_channels, arg_vec[0], dropout=0.25, edge_dim=edge_dim)
            self.conv2_sub = GATv2Conv(arg_vec[0], arg_vec[1], dropout=0.25, edge_dim=edge_dim)
            self.conv3_sub = GATv2Conv(arg_vec[1], arg_vec[2], dropout=0.25, edge_dim=edge_dim)
            self.conv4_sub = GATv2Conv(arg_vec[2], arg_vec[3], dropout=0.25, edge_dim=edge_dim)

            self.conv1 = GATv2Conv(in_channels, arg_vec[0], dropout=0.25, edge_dim=edge_dim)
            self.conv2 = GATv2Conv(arg_vec[0], arg_vec[1], dropout=0.25, edge_dim=edge_dim)
            self.conv3 = GATv2Conv(arg_vec[1], arg_vec[2], dropout=0.25, edge_dim=edge_dim)
            self.conv4 = GATv2Conv(arg_vec[2], arg_vec[3], dropout=0.25, edge_dim=edge_dim)

            self.FCNN = Sequential(
                Linear(arg_vec[3] * 2 + 256 * 2, arg_vec[4]),
                ReLU(inplace=True),
                BatchNorm1d(arg_vec[4]),
                Linear(arg_vec[4], arg_vec[5]),
                ReLU(inplace=True),
                BatchNorm1d(arg_vec[5]),
                Linear(arg_vec[5], 50),
                ReLU(inplace=True),
                BatchNorm1d(50),
                Linear(50, 1),
                Sigmoid()
            )
        else:
            raise TypeError('Unsupported mode')
        self.mode = mode

    @dispatch()
    def forward(self, data: Data) -> torch.Tensor:
        data.x = data.x.to(torch.float32)
        data.edge_attr = data.edge_attr.to(torch.float32)
        data.edge_index = data.edge_index.to(torch.int64)
        data.fp = data.fp.to(torch.float32)

        x = self.module(data.x, data.edge_index, data.edge_attr)
        x = global_mean_pool(x, data.batch)
        x = torch.cat([x, data.fp], dim=1)

        x = self.lin1(x).relu()
        x = self.lin2(x).relu()
        x = self.lin3(x).relu()
        x = self.lin4(x).sigmoid()

        return x

    @dispatch(Data)
    def forward(self, sub: Data, met: Data) -> torch.Tensor:
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

    def fit(self, data: MolFrame, lr: float = 1e-5, eps: int = 100, verbose: bool = False,
            prior: float = 0.75) -> 'Filter':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device)

        criterion = PULoss(prior)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=1e-8)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.8)

        if verbose:
            print('Starting DataLoaders generation')

        if self.mode =='pair':
            train_loader = []
            for pairs in data.graphs.values():
                for pair in pairs:
                    if pair is not None:
                        train_loader.append(pair)
            train_loader = DataLoader(train_loader, batch_size=128, shuffle=True)

            history = []
            for _ in tqdm(range(100)):
                self.train()
                for batch in train_loader:
                    out = self(batch)
                    loss = criterion(out, batch.y.unsqueeze(1).float())
                    history.append(loss.item())
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                scheduler.step()
            return self

        elif self.mode == 'single':
            def collate(data_list: tp.List) -> tp.Tuple[Batch, Batch]:
                batchA = Batch.from_data_list([data[0] for data in data_list])
                batchB = Batch.from_data_list([data[1] for data in data_list])
                return batchA, batchB

            train_loader = []
            for mol in self.card.keys():
                sub = self.single[mol]
                if sub is not None:
                    for met in self.card[mol]:
                        if self.single[met] is not None:
                            train_loader.append((sub, self.single[met]))
            for mol in self.negs.keys():
                sub = self.single[mol]
                if sub is not None:
                    for met in self.negs[mol]:
                        if self.single[met] is not None:
                            train_loader.append((sub, self.single[met]))
            train_loader = DataLoader(train_loader, batch_size=128, shuffle=True, collate_fn=collate)

            history = []
            for _ in tqdm(range(100)):
                self.train()
                for batch in train_loader:
                    met_batch = batch[1].to(device)
                    sub_batch = batch[0].to(device)
                    out = self(sub_batch, met_batch)
                    loss = criterion(out, met_batch.y.unsqueeze(1).float())
                    history.append(loss.item())
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                scheduler.step()
            return self

        else:
            raise TypeError('Unsupported mode')

    def predict(self, sub: str, prod: str) -> int:
        sub_mol = Chem.MolFromSmiles(sub)
        prod_mol = Chem.MolFromSmiles(prod)
        if self.mode == 'pair':
            graph = from_pair(sub_mol, prod_mol)
            return int(cpunum(self(graph)).item())
        elif self.mode == 'single':
            graph_sub, graph_prod = from_rdmol(sub_mol), from_rdmol(prod_mol)
            return int(cpunum(self(graph_sub, graph_prod)).item())
        else:
            raise TypeError('Unsupported mode')

def create_filter_pairs(arg_vec: tp.List[int]) -> Module:
    class Filter(GFilter):
        def __init__(self) -> None:
            super(Filter, self).__init__()
            self.module = nn.Sequential('x, edge_index, edge_attr', [
                (GATv2Conv(12, arg_vec[0], edge_dim=6, dropout=0.25), 'x, edge_index, edge_attr -> x'),
                ReLU(inplace=True),
                (GATv2Conv(arg_vec[0], arg_vec[1], edge_dim=6, dropout=0.25), 'x, edge_index, edge_attr -> x'),
                ReLU(inplace=True),
                (GATv2Conv(arg_vec[1], arg_vec[2], edge_dim=6, dropout=0.25), 'x, edge_index, edge_attr -> x'),
                ReLU(inplace=True)
            ])
            self.dropout = Dropout(0.1)
            self.lin1 = Linear((arg_vec[2] + 256*2), arg_vec[3])
            self.bn1 = BatchNorm1d(arg_vec[3])
            self.lin2 = Linear(arg_vec[3], arg_vec[4])
            self.bn2 = BatchNorm1d(arg_vec[4])
            self.lin3 = Linear(arg_vec[4], arg_vec[5])
            self.bn3 = BatchNorm1d(arg_vec[5])
            self.lin4 = Linear(arg_vec[5], 1)

        def forward(self, data: Data) -> torch.Tensor:
            data.x = data.x.to(torch.float32)
            data.edge_attr = data.edge_attr.to(torch.float32)
            data.edge_index = data.edge_index.to(torch.int64)
            data.fp = data.fp.to(torch.float32)

            x = self.module(data.x, data.edge_index, data.edge_attr)
            x = global_mean_pool(x, data.batch)
            x = torch.cat([x, data.fp], dim=1)

            x = self.dropout(x)
            x = self.lin1(x).relu()
            x = self.bn1(x)
            x = self.lin2(x).relu()
            x = self.bn2(x)
            x = self.lin3(x).relu()
            x = self.bn3(x)
            x = self.lin4(x).sigmoid()
            return x

        def fit(self, data: MolFrame, lr: float = 1e-5, eps: int = 100, verbose: bool = False, prior: float = 0.75) -> 'Filter':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.to(device)

            criterion = PULoss(prior)
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=1e-8)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.8)

            if verbose:
                print('Starting DataLoaders generation')

            train_loader = []
            for pairs in data.graphs.values():
                for pair in pairs:
                    if pair is not None:
                        train_loader.append(pair)
            train_loader = DataLoader(train_loader, batch_size=128, shuffle=True)

            history = []
            for _ in tqdm(range(100)):
                self.train()
                for batch in train_loader:
                    out = self(batch)
                    loss = criterion(out, batch.y.unsqueeze(1).float())
                    history.append(loss.item())
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                scheduler.step()
            return self

    return Filter()

def create_filter_singles(arg_vec: tp.List[int]) -> Module:
    class Filter(Module):
        def __init__(self) -> None:
            super(Filter, self).__init__()
            self.conv1_sub = GATv2Conv(10, arg_vec[0], dropout=0.25, edge_dim=6)
            self.conv2_sub = GATv2Conv(arg_vec[0], arg_vec[1], dropout=0.25, edge_dim=6)
            self.conv3_sub = GATv2Conv(arg_vec[1], arg_vec[2], dropout=0.25, edge_dim=6)
            self.conv4_sub = GATv2Conv(arg_vec[2], arg_vec[3], dropout=0.25, edge_dim=6)

            self.conv1 = GATv2Conv(10, arg_vec[0], dropout=0.25, edge_dim=6)
            self.conv2 = GATv2Conv(arg_vec[0], arg_vec[1], dropout=0.25, edge_dim=6)
            self.conv3 = GATv2Conv(arg_vec[1], arg_vec[2], dropout=0.25, edge_dim=6)
            self.conv4 = GATv2Conv(arg_vec[2], arg_vec[3], dropout=0.25, edge_dim=6)

            self.FCNN = Sequential(
                Linear(arg_vec[3]*2 + 256*2, arg_vec[4]),
                ReLU(inplace=True),
                BatchNorm1d(arg_vec[4]),
                Linear(arg_vec[4], arg_vec[5]),
                ReLU(inplace=True),
                BatchNorm1d(arg_vec[5]),
                Linear(arg_vec[5], 50),
                ReLU(inplace=True),
                BatchNorm1d(50),
                Linear(50, 1),
                Sigmoid()
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

        def fit(self, data: MolFrame, lr: float = 1e-5, eps: int = 100, verbose: bool = False,
                prior: float = 0.75) -> 'Filter':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.to(device)

            criterion = PULoss(prior)
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=1e-8)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.8)

            def collate(data_list: tp.List) -> tp.Tuple[Batch, Batch]:
                batchA = Batch.from_data_list([data[0] for data in data_list])
                batchB = Batch.from_data_list([data[1] for data in data_list])
                return batchA, batchB

            if verbose:
                print('Starting DataLoaders generation')

            train_loader = []
            for mol in self.card.keys():
                sub = self.single[mol]
                if sub is not None:
                    for met in self.card[mol]:
                        if self.single[met] is not None:
                            train_loader.append((sub, self.single[met]))
            for mol in self.negs.keys():
                sub = self.single[mol]
                if sub is not None:
                    for met in self.negs[mol]:
                        if self.single[met] is not None:
                            train_loader.append((sub, self.single[met]))
            train_loader = DataLoader(train_loader, batch_size=128, shuffle=True, collate_fn=collate)

            history = []
            for _ in tqdm(range(100)):
                self.train()
                for batch in train_loader:
                    met_batch = batch[1].to(device)
                    sub_batch = batch[0].to(device)
                    out = self(sub_batch, met_batch)
                    loss = criterion(out, met_batch.y.unsqueeze(1).float())
                    history.append(loss.item())
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                scheduler.step()
            return self

    return Filter()