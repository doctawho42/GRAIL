import torch
from pathlib import Path
import pickle as pkl
from sklearn.impute import SimpleImputer
import numpy as np
from torch.nn import Module, Sequential, ReLU, Linear, Bilinear, MultiheadAttention, BatchNorm1d, Dropout, init, Sigmoid
from torch.nn.functional import dropout, threshold
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_geometric import nn
from .wrapper import GGenerator
from ..utils.preparation import MolFrame, cpunum, iscorrect, standardize_mol
from ..utils.transform import from_rdmol, from_rule
from tqdm.auto import tqdm
from rdkit import Chem
from rdkit.Chem.AllChem import ReactionFromSmarts
from rdkit.Chem import MACCSkeys
import typing as tp
from .train_model import AsymmetricBCELoss
import matplotlib.pyplot as plt
import random
from itertools import chain
from ..nn.molpath import EdgePathNN

class RuleParse(Module):
    def __init__(self,
                 rule_dict: dict[str, Batch],
                 arg_vec: tp.List[int] = [100, 200, 400, 200, 100],
                 use_molpath: bool = False,
                 molpath_hidden: tp.Optional[int] = None,
                 molpath_cutoff: tp.Optional[int] = None,
                 molpath_y: tp.Optional[float] = None
                 ) -> None:
        super(RuleParse, self).__init__()
        self.use_molpath = use_molpath
        self.rule_dict = rule_dict
        if not use_molpath:
            self.rule_encoder = nn.Sequential('x, edge_index, edge_attr', [
                (GATv2Conv(16, arg_vec[0], edge_dim=18, dropout=0.25), 'x, edge_index, edge_attr -> x'),
                ReLU(inplace=True),
                BatchNorm1d(arg_vec[0]),
                (GATv2Conv(arg_vec[0], arg_vec[1], edge_dim=18, dropout=0.25), 'x, edge_index, edge_attr -> x'),
                ReLU(inplace=True),
                BatchNorm1d(arg_vec[1]),
                Linear(arg_vec[1], arg_vec[2]),
                BatchNorm1d(arg_vec[2])
            ])
            self.ffn = Sequential(
                Linear(arg_vec[2], arg_vec[3]),
                ReLU(inplace=True),
                Linear(arg_vec[3], arg_vec[4]),
                ReLU(inplace=True),
                BatchNorm1d(arg_vec[4]),
                Linear(arg_vec[4], 100)
            )
            self.batch_norm = BatchNorm1d(arg_vec[2])
        else:
            self.rule_encoder = EdgePathNN(
                                            hidden_dim=molpath_hidden,
                                            cutoff=molpath_cutoff,  # Maximum path length
                                            y=molpath_y,  # Attention parameter
                                            n_classes=100,
                                            device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
                                            node_feat_dim=16,  # Your node feature dimension
                                            edge_feat_dim=18,  # Your edge feature dimension
                                            use_fingerprint=False,  # Whether to use fingerprints
                                            readout="sum",
                                            path_agg="sum",
                                            dropout=0.1
                                        )
            self.ffn = torch.nn.Identity()
            self.batch_norm = torch.nn.Identity()

    def forward(self) -> torch.Tensor:
        # Create the batch of rules
        device = next(self.parameters()).device
        batch = Batch.from_data_list(list(self.rule_dict.values()))

        # Перемещаем весь батч на устройство
        batch = batch.to(device)

        batch.x = batch.x.to(torch.float32)
        batch.edge_attr = batch.edge_attr.to(torch.float32)
        batch.edge_index = batch.edge_index.to(torch.int64)

        # Apply Module to the batch
        if self.use_molpath:
            x = self.rule_encoder(batch)
        else:
            data = self.rule_encoder(batch.x, batch.edge_index, batch.edge_attr)
            x = global_mean_pool(data, data.batch)
            x = self.batch_norm(x)
            # Apply the feed-forward network
            x = self.ffn(x)
        return x

class MoleculeAugmentor:
    @staticmethod
    def atom_masking(graph, mask_ratio=0.15):
        """Маскирование случайных атомов"""
        num_nodes = graph.x.size(0)
        if num_nodes == 0:
            return graph

        num_mask = max(1, int(mask_ratio * num_nodes))
        num_mask = min(num_mask, num_nodes - 1)  # Оставляем хотя бы 1 узел

        mask_indices = torch.randperm(num_nodes)[:num_mask]
        mask_token = torch.zeros_like(graph.x[0:1]).squeeze(0)

        graph.x[mask_indices] = mask_token
        return graph

    @staticmethod
    def bond_deletion(graph, delete_ratio=0.15):
        """Удаление случайных связей"""
        num_edges = graph.edge_index.size(1)
        num_delete = int(delete_ratio * num_edges)

        if num_edges > 0 and num_delete > 0:
            keep_indices = torch.randperm(num_edges)[num_delete:]
            graph.edge_index = graph.edge_index[:, keep_indices]
            graph.edge_attr = graph.edge_attr[keep_indices]
        return graph

    @staticmethod
    def subgraph_removal(graph, remove_ratio=0.2):
        """Удаление случайного подграфа"""
        num_nodes = graph.x.size(0)
        if num_nodes <= 1:
            return graph

        num_remove = max(1, int(remove_ratio * num_nodes))

        # Выбираем случайный стартовый узел
        start_node = torch.randint(0, num_nodes, (1,))

        # BFS для выбора связного подграфа
        removed_nodes = set()
        queue = [start_node.item()]

        while len(removed_nodes) < num_remove and queue:
            node = queue.pop(0)
            if node not in removed_nodes:
                removed_nodes.add(node)
                # Добавляем соседей
                neighbors = graph.edge_index[1, graph.edge_index[0] == node].tolist()
                queue.extend([n for n in neighbors if n not in removed_nodes])

        # Маскируем удаленные узлы
        removed_nodes = list(removed_nodes)[:num_remove]
        mask_token = torch.zeros_like(graph.x[0])
        graph.x[removed_nodes] = mask_token

        return graph

    @classmethod
    def augment_batch(cls, graph_batch):
        """Применяем случайную аугментацию к батчу"""
        aug_type = random.choice(['atom_masking', 'bond_deletion', 'subgraph_removal'])

        try:
            if aug_type == 'atom_masking':
                return cls.atom_masking(graph_batch)
            elif aug_type == 'bond_deletion':
                return cls.bond_deletion(graph_batch)
            else:
                return cls.subgraph_removal(graph_batch)
        except Exception as e:
            return graph_batch

def get_maccs_smarts():
    """Получение SMARTS-паттернов для MACCS-ключей"""
    maccs_smarts = []
    for i in range(1, 167):  # MACCS keys 1-166
        try:
            # Получаем SMARTS для каждого ключа
            smarts = MACCSkeys.smartsPatts[i]
            maccs_smarts.append(smarts)
        except (KeyError, AttributeError):
            # Если ключ не существует, используем пустой паттерн
            maccs_smarts.append('')

    return maccs_smarts

class MACCSRuleParse(Module):
    def __init__(self,
                 maccs_smarts,
                 arg_vec: tp.List[int] = [100, 200, 400, 200, 100],
                 use_molpath: bool = False,
                 molpath_hidden: tp.Optional[int] = None,
                 molpath_cutoff: tp.Optional[int] = None,
                 molpath_y: tp.Optional[float] = None
                 ):
        super().__init__()
        self.use_molpath = use_molpath
        self.maccs_smarts = maccs_smarts
        self.maccs_graphs = self._create_maccs_graphs()

        # RuleParse для MACCS-паттернов
        if not use_molpath:
            self.rule_encoder = nn.Sequential('x, edge_index, edge_attr', [
                (GATv2Conv(16, arg_vec[0], edge_dim=18, dropout=0.25), 'x, edge_index, edge_attr -> x'),
                ReLU(inplace=True),
                BatchNorm1d(arg_vec[0]),
                (GATv2Conv(arg_vec[0], arg_vec[1], edge_dim=18, dropout=0.25), 'x, edge_index, edge_attr -> x'),
                ReLU(inplace=True),
                BatchNorm1d(arg_vec[1]),
                Linear(arg_vec[1], arg_vec[2]),
                BatchNorm1d(arg_vec[2])
            ])
            self.ffn = Sequential(
                Linear(arg_vec[2], arg_vec[3]),
                ReLU(inplace=True),
                Linear(arg_vec[3], arg_vec[4]),
                ReLU(inplace=True),
                BatchNorm1d(arg_vec[4]),
                Linear(arg_vec[4], 100)
            )
            self.batch_norm = BatchNorm1d(arg_vec[2])
        else:
            self.rule_encoder = EdgePathNN(
                hidden_dim=molpath_hidden,
                cutoff=molpath_cutoff,  # Maximum path length
                y=molpath_y,  # Attention parameter
                n_classes=100,
                device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
                node_feat_dim=16,  # Your node feature dimension
                edge_feat_dim=18,  # Your edge feature dimension
                use_fingerprint=False,  # Whether to use fingerprints
                readout="sum",
                path_agg="sum",
                dropout=0.1
            )
            self.ffn = torch.nn.Identity()
            self.batch_norm = torch.nn.Identity()

        self.batch_norm = BatchNorm1d(arg_vec[2])

    def _create_maccs_graphs(self):
        """Создание графов для MACCS-паттернов"""
        maccs_graphs = {}
        for i, smarts in enumerate(self.maccs_smarts):
            if smarts and smarts[0] != '':  # Если SMARTS не пустой
                try:
                    graph = from_rule(smarts[0]+'>>'+'')
                    maccs_graphs[i] = graph
                except Exception as e:
                    print(f"Failed to create graph for MACCS key {i}: {e}")
                    maccs_graphs[i] = None
            else:
                maccs_graphs[i] = None
        return maccs_graphs

    def forward(self):
        """Получение эмбеддингов для всех MACCS-паттернов"""
        device = next(self.parameters()).device

        valid_graphs = [graph for graph in self.maccs_graphs.values() if graph is not None]
        if not valid_graphs:
            return torch.zeros(166, 100, device=device)

        batch = Batch.from_data_list(valid_graphs)

        # Перемещаем ВСЕ компоненты батча на устройство
        batch = batch.to(device)

        # Явно преобразуем типы данных после перемещения на устройство
        batch.x = batch.x.to(torch.float32)
        batch.edge_attr = batch.edge_attr.to(torch.float32)
        batch.edge_index = batch.edge_index.to(torch.int64)

        # Применяем RuleParse
        if self.use_molpath:
            x = self.rule_encoder(batch)
        else:
            data = self.rule_encoder(batch.x, batch.edge_index, batch.edge_attr)
            x = global_mean_pool(data, data.batch)  # Теперь batch.batch тоже на device
            x = self.batch_norm(x)
            x = self.ffn(x)

        # Создаем полную матрицу на правильном устройстве
        full_embeddings = torch.zeros(166, 100, device=device)
        valid_indices = [i for i, graph in enumerate(self.maccs_graphs.values()) if graph is not None]
        if valid_indices:
            full_embeddings[valid_indices] = x

        return full_embeddings

class MACCSPredictor(Module):
    def __init__(self,
                 molecular_encoder,
                 maccs_smarts,
                 arg_vec: tp.List[int] = [100, 200, 400, 200, 100],
                 use_molpath: bool = False,
                 molpath_hidden: tp.Optional[int] = None,
                 molpath_cutoff: tp.Optional[int] = None,
                 molpath_y: tp.Optional[float] = None
                 ):
        super().__init__()
        self.use_molpath = use_molpath
        self.maccs_smarts = maccs_smarts
        self.molpath_hidden = molpath_hidden
        self.molpath_cutoff = molpath_cutoff
        self.molpath_y = molpath_y
        self.molecular_encoder = molecular_encoder
        self.maccs_ruleparse = MACCSRuleParse(maccs_smarts,
                                              arg_vec=arg_vec,
                                              use_molpath=use_molpath,
                                              molpath_hidden=molpath_hidden,
                                              molpath_cutoff=molpath_cutoff,
                                              molpath_y=molpath_y
        )

        # Замораживаем молекулярный энкодер
        for param in self.molecular_encoder.parameters():
            param.requires_grad = False

        # Предиктор MACCS-ключей
        self.predictor = Sequential(
            Linear(166, 256),
            ReLU(inplace=True),
            Dropout(0.2),
            Linear(256, 166),
        )

        self.sigmoid = Sigmoid()

    def forward(self, molecular_graph):
        device = next(self.parameters()).device
        molecular_graph = molecular_graph.to(device)

        with torch.no_grad():
            if self.use_molpath:
                mol_embedding = self.molecular_encoder(molecular_graph)
            else:
                node_embeddings = self.molecular_encoder(
                    molecular_graph.x,
                    molecular_graph.edge_index,
                    molecular_graph.edge_attr
                )
                mol_embedding = global_mean_pool(node_embeddings, molecular_graph.batch)
            #print(f"mol_embedding range: [{mol_embedding.min().item():.6f}, {mol_embedding.max().item():.6f}]")

        maccs_embeddings = self.maccs_ruleparse()
        #print(f"maccs_embeddings range: [{maccs_embeddings.min().item():.6f}, {maccs_embeddings.max().item():.6f}]")

        similarities = torch.mm(mol_embedding, maccs_embeddings.T)
        #print(f"similarities range: [{similarities.min().item():.6f}, {similarities.max().item():.6f}]")

        logits = self.predictor(similarities)
        #print(f"logits range: [{logits.min().item():.6f}, {logits.max().item():.6f}]")

        predictions = self.sigmoid(logits)
        #print(f"predictions after sigmoid range: [{predictions.min().item():.6f}, {predictions.max().item():.6f}]")

        return predictions

class Generator(GGenerator):
    def __init__(self, rule_dict: dict[str, Batch], in_channels: int, edge_dim: int,
                 arg_vec: tp.Optional[tp.List[int]] = None,
                 rp_arg_vec: tp.Optional[tp.List[int]] = None,
                 projection_dim: int = 256,
                 use_maccs_pretraining: bool = False,
                 use_molpath: bool = False,
                 molpath_hidden: tp.Optional[int] = None,
                 molpath_cutoff: tp.Optional[int] = None,
                 molpath_y: tp.Optional[float] = None
                 ):
        super(Generator, self).__init__()
        if rp_arg_vec is None:
            rp_arg_vec = [200] * 5
        self.rp_arg_vec = rp_arg_vec
        self.use_molpath = use_molpath
        self.molpath_hidden = molpath_hidden
        self.molpath_cutoff = molpath_cutoff
        self.molpath_y = molpath_y
        self.parser = RuleParse(rule_dict,
                                arg_vec=rp_arg_vec,
                                use_molpath=use_molpath,
                                molpath_hidden=molpath_hidden,
                                molpath_cutoff=molpath_cutoff,
                                molpath_y=molpath_y)
        self.rules = rule_dict
        self.num_rules = len(rule_dict)
        self.use_molpath = use_molpath

        if arg_vec is None:
            arg_vec = [500] * 2

        # Основной GNN энкодер
        if not use_molpath:
            self.gnn_encoder = nn.Sequential('x, edge_index, edge_attr', [
                (GATv2Conv(in_channels, arg_vec[0], edge_dim=edge_dim, dropout=0.25), 'x, edge_index, edge_attr -> x'),
                ReLU(inplace=True),
                BatchNorm1d(arg_vec[0]),
                (GATv2Conv(arg_vec[0], arg_vec[1], edge_dim=edge_dim, dropout=0.25), 'x, edge_index, edge_attr -> x'),
                ReLU(inplace=True),
                BatchNorm1d(arg_vec[1]),
                (GATv2Conv(arg_vec[1], 100, edge_dim=edge_dim, dropout=0.25), 'x, edge_index, edge_attr -> x'),
            ])
        else:
            self.gnn_encoder = EdgePathNN(
                                            hidden_dim=molpath_hidden,
                                            cutoff=molpath_cutoff,  # Maximum path length
                                            y=molpath_y,  # Attention parameter
                                            n_classes=100,
                                            device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
                                            node_feat_dim=16,  # Your node feature dimension
                                            edge_feat_dim=18,  # Your edge feature dimension
                                            use_fingerprint=True,  # Whether to use fingerprints
                                            fingerprint_dim=1024,  # Your fingerprint dimension
                                            readout="sum",
                                            path_agg="sum",
                                            dropout=0.1
                                        )

        # Проекционная головка для контрастивного обучения
        self.projection_head = Sequential(
            Linear(100, projection_dim),
            BatchNorm1d(projection_dim),
            ReLU(inplace=True),
            Linear(projection_dim, projection_dim)
        )

        # Предикторы для маскированного моделирования
        self.atom_predictor = Linear(100, in_channels)
        self.bond_predictor = Linear(200, edge_dim)

        # Основные компоненты генератора
        self.embed_dim = 100
        self.attention = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=4,
            batch_first=True
        )
        self.final_proj = Linear(self.embed_dim, self.num_rules)
        self.sigmoid = Sigmoid()
        self.dropout = Dropout(0.2)

        # Флаги предобучения
        self.pretrained = False
        self.use_maccs_pretraining = use_maccs_pretraining
        if use_maccs_pretraining:
            self.maccs_smarts = get_maccs_smarts()

        # Инициализация весов
        self._initialize_weights()

    def _initialize_weights(self):
        """Инициализация весов для стабильности"""
        for module in [self.projection_head, self.atom_predictor, self.bond_predictor]:
            if isinstance(module, Sequential):
                for layer in module:
                    if isinstance(layer, Linear):
                        # Используем Xavier инициализацию для лучшей стабильности
                        init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            init.constant_(layer.bias, 0)
            elif isinstance(module, Linear):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

        # Добавьте инициализацию для предиктора MACCS
        if hasattr(self, 'maccs_predictor'):
            for module in self.maccs_predictor.modules():
                if isinstance(module, Linear):
                    init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        init.constant_(module.bias, 0)

    def forward(self, data: Data, mode: str = 'generation') -> torch.Tensor:
        """
        Унифицированный forward с поддержкой разных режимов
        """
        data.x = data.x.to(torch.float32)
        data.edge_attr = data.edge_attr.to(torch.float32)
        data.edge_index = data.edge_index.to(torch.int64)

        # Получаем базовые эмбеддинги от GNN энкодера
        if self.use_molpath:
            x = self.gnn_encoder(data)
        else:
            x = self.gnn_encoder(data.x, data.edge_index, data.edge_attr)

        if mode == 'contrastive':
            # Режим контрастивного обучения
            graph_emb = global_mean_pool(x, data.batch)
            z = self.projection_head(graph_emb)
            return torch.nn.functional.normalize(z, dim=1)

        elif mode == 'masked_modeling':
            # Режим маскированного моделирования
            return x

        elif mode == 'generation':
            # Основной режим генерации
            mol_emb = global_mean_pool(x, data.batch).to(torch.float32)
            mol_emb = self.dropout(mol_emb)

            # Получаем эмбеддинги правил
            rule_emb = self.parser().to(torch.float32)

            # Вычисляем логиты
            attn_output, _ = self.attention(
                query=mol_emb.unsqueeze(1),
                key=rule_emb.unsqueeze(0).repeat(mol_emb.size(0), 1, 1),
                value=rule_emb.unsqueeze(0).repeat(mol_emb.size(0), 1, 1)
            )
            logits = self.final_proj(attn_output.squeeze(1))
            return self.sigmoid(logits)

        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _contrastive_loss(self, z1, z2, temperature=0.1):
        """Контрастивная потеря"""
        batch_size = z1.size(0)

        # Нормализация
        z1 = torch.nn.functional.normalize(z1, p=2, dim=1)
        z2 = torch.nn.functional.normalize(z2, p=2, dim=1)

        # Вычисляем сходства
        similarities = torch.mm(z1, z2.T) / max(temperature, 1e-8)

        # Метки: диагональные элементы - положительные пары
        labels = torch.arange(batch_size, device=z1.device)

        return torch.nn.functional.cross_entropy(similarities, labels)

    def _mask_batch(self, batch, mask_ratio=0.15):
        """Маскирование батча"""
        try:
            masked_batch = batch.clone()
            num_atoms = batch.x.size(0)

            if num_atoms <= 1:
                return batch, None, None, None

            num_mask_atoms = max(1, min(num_atoms - 1, int(mask_ratio * num_atoms)))
            mask_indices = torch.randperm(num_atoms)[:num_mask_atoms]

            mask_indices = mask_indices[mask_indices < num_atoms]
            if len(mask_indices) == 0:
                return batch, None, None, None

            atom_targets = batch.x[mask_indices].clone()
            mask_token = torch.zeros_like(batch.x[0])
            masked_batch.x[mask_indices] = mask_token

            bond_targets = None
            if (hasattr(batch, 'edge_attr') and batch.edge_attr is not None and
                batch.edge_attr.size(0) > 0):

                num_edges = batch.edge_attr.size(0)
                num_mask_bonds = max(1, min(num_edges - 1, int(mask_ratio * num_edges)))
                mask_bond_indices = torch.randperm(num_edges)[:num_mask_bonds]

                mask_bond_indices = mask_bond_indices[mask_bond_indices < num_edges]
                if len(mask_bond_indices) > 0:
                    bond_targets = batch.edge_attr[mask_bond_indices].clone()
                    bond_mask_token = torch.zeros_like(batch.edge_attr[0])
                    masked_batch.edge_attr[mask_bond_indices] = bond_mask_token

            return masked_batch, atom_targets, bond_targets, mask_indices

        except Exception as e:
            return batch, None, None, None

    def _compute_masked_loss(self, node_embeddings, atom_targets, bond_targets, atom_mask_indices):
        """Вычисление потерь маскированного моделирования"""
        try:
            total_loss = 0
            loss_components = 0

            # Loss для атомов
            if atom_targets is not None and atom_mask_indices is not None:
                valid_indices = atom_mask_indices[atom_mask_indices < node_embeddings.size(0)]
                if len(valid_indices) > 0:
                    masked_embeddings = node_embeddings[valid_indices]

                    if masked_embeddings.size(0) == atom_targets.size(0):
                        atom_pred = self.atom_predictor(masked_embeddings)
                        atom_loss = torch.nn.functional.mse_loss(atom_pred, atom_targets)
                        total_loss += atom_loss
                        loss_components += 1

            # Loss для связей
            if bond_targets is not None and len(bond_targets) > 0:
                # Используем среднее по всем узлам для предсказания связей
                mean_embedding = node_embeddings.mean(dim=0, keepdim=True)
                bond_pred = self.bond_predictor(mean_embedding)

                # Среднее по целевым связям
                mean_bond_targets = bond_targets.mean(dim=0, keepdim=True)
                bond_loss = torch.nn.functional.mse_loss(bond_pred, mean_bond_targets)
                total_loss += bond_loss
                loss_components += 1

            if loss_components == 0:
                return torch.tensor(0.0, device=node_embeddings.device)

            return total_loss / loss_components

        except Exception as e:
            return torch.tensor(0.0, device=node_embeddings.device)

    def _create_pretraining_graphs(self, smiles_list):
        """Создание графов для предобучения"""
        graphs = []
        for smiles in tqdm(smiles_list, desc="Creating pretraining graphs"):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    graph = from_rdmol(mol)
                    if graph is not None:
                        graph.x = graph.x.to(torch.float32)
                        graph.edge_attr = graph.edge_attr.to(torch.float32)
                        graph.edge_index = graph.edge_index.to(torch.int64)
                        graphs.append(graph)
            except Exception as e:
                continue

        print(f"Created {len(graphs)} graphs for pretraining")
        return graphs

    def _contrastive_pretrain_phase(self, graphs, epochs, batch_size, lr):
        """Контрастивное обучение с обновлением весов Generator"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        optimizer = torch.optim.AdamW([
            {'params': self.gnn_encoder.parameters(), 'lr': lr},
            {'params': self.projection_head.parameters(), 'lr': lr}
        ], weight_decay=1e-6)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        augmentor = MoleculeAugmentor()

        dataloader = DataLoader(graphs, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.train()
            total_loss = 0
            num_batches = 0

            for batch in tqdm(dataloader, desc=f"Contrastive Epoch {epoch+1}"):
                try:
                    batch = batch.to(device)

                    # Создаем аугментированные версии
                    aug1 = augmentor.augment_batch(batch)
                    aug2 = augmentor.augment_batch(batch)

                    # Получаем проекции через ОДИН И ТОТ ЖЕ gnn_encoder
                    z1 = self(aug1, mode='contrastive')
                    z2 = self(aug2, mode='contrastive')

                    # Контрастивная потеря
                    loss = self._contrastive_loss(z1, z2)

                    if torch.isnan(loss):
                        continue

                    # ОБНОВЛЯЕМ ВЕСА gnn_encoder И projection_head
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.gnn_encoder.parameters()) +
                        list(self.projection_head.parameters()),
                        max_norm=1.0
                    )
                    optimizer.step()

                    total_loss += loss.item()
                    num_batches += 1

                except Exception as e:
                    continue

            if num_batches > 0:
                avg_loss = total_loss / num_batches
                current_lr = scheduler.get_last_lr()[0]
                print(f"Contrastive Epoch {epoch+1}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")

            scheduler.step()

    def _masked_modeling_pretrain_phase(self, graphs, epochs, batch_size, lr):
        """Маскированное моделирование с обновлением весов Generator"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        optimizer = torch.optim.AdamW([
            {'params': self.gnn_encoder.parameters(), 'lr': lr},
            {'params': self.atom_predictor.parameters(), 'lr': lr},
            {'params': self.bond_predictor.parameters(), 'lr': lr}
        ], weight_decay=1e-6)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        dataloader = DataLoader(graphs, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.train()
            total_loss = 0
            num_batches = 0

            for batch in tqdm(dataloader, desc=f"Masked Epoch {epoch+1}"):
                try:
                    batch = batch.to(device)

                    # Маскируем батч
                    masked_batch, atom_targets, bond_targets, atom_mask_indices = \
                        self._mask_batch(batch)

                    if atom_targets is None:
                        continue

                    # Получаем эмбеддинги через ОДИН И ТОТ ЖЕ gnn_encoder
                    node_embeddings = self(masked_batch, mode='masked_modeling')

                    # Вычисляем потери для атомов и связей
                    loss = self._compute_masked_loss(
                        node_embeddings, atom_targets, bond_targets, atom_mask_indices
                    )

                    if torch.isnan(loss) or loss.item() == 0:
                        continue

                    # ОБНОВЛЯЕМ ВЕСА gnn_encoder, atom_predictor, bond_predictor
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.gnn_encoder.parameters()) +
                        list(self.atom_predictor.parameters()) +
                        list(self.bond_predictor.parameters()),
                        max_norm=1.0
                    )
                    optimizer.step()

                    total_loss += loss.item()
                    num_batches += 1

                except Exception as e:
                    continue

            if num_batches > 0:
                avg_loss = total_loss / num_batches
                current_lr = scheduler.get_last_lr()[0]
                print(f"Masked Modeling Epoch {epoch+1}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")

            scheduler.step()

    def _create_maccs_dataset(self, smiles_list):
        """Создание датасета с MACCS-ключами"""
        graphs = []
        for smiles in tqdm(smiles_list, desc="Creating MACCS dataset"):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    # Создаем граф молекулы
                    graph = from_rdmol(mol)
                    if graph is not None:
                        graph.x = graph.x.to(torch.float32)
                        graph.edge_attr = graph.edge_attr.to(torch.float32)
                        graph.edge_index = graph.edge_index.to(torch.int64)

                        # Вычисляем MACCS-ключи и создаем тензор с правильной формой
                        maccs_fp = MACCSkeys.GenMACCSKeys(mol)
                        maccs_bits = np.array([maccs_fp.GetBit(i) for i in range(166)])

                        # Явно создаем тензор с формой [166] (не [166, 1] и не [1, 166])
                        graph.maccs = torch.tensor(maccs_bits, dtype=torch.float32)

                        graphs.append(graph)
            except Exception as e:
                print(f"Error creating graph for {smiles}: {e}")
                continue

        print(f"Created MACCS dataset with {len(graphs)} molecules")
        return graphs

    def _transfer_maccs_weights(self, maccs_ruleparse):
        """Перенос обученных весов из MACCS RuleParse в основной RuleParse"""
        # Копируем веса rule_encoder
        self.parser.rule_encoder.load_state_dict(maccs_ruleparse.rule_encoder.state_dict())

        # Копируем веса FFN
        self.parser.ffn.load_state_dict(maccs_ruleparse.ffn.state_dict())

        # Копируем веса batch_norm
        try:
            self.parser.batch_norm.load_state_dict(maccs_ruleparse.batch_norm.state_dict())
        except Exception as e:
            print(f'{e}, maybe its torch.nn.Identity()')

    def pretrain_maccs(self, smiles_list, epochs=50, batch_size=64, lr=1e-4):
        """Предобучение RuleParse на задаче предсказания MACCS-ключей"""
        if not self.use_maccs_pretraining:
            print("MACCS pretraining is disabled. Set use_maccs_pretraining=True in constructor.")
            return

        print("Starting MACCS-based RuleParse pretraining...")

        # Используем CPU для отладки
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Создаем модель предсказания MACCS
        maccs_predictor = MACCSPredictor(self.gnn_encoder,
                                         self.maccs_smarts,
                                         arg_vec=self.rp_arg_vec,
                                         use_molpath=self.use_molpath,
                                         molpath_hidden=self.molpath_hidden,
                                         molpath_cutoff=self.molpath_cutoff,
                                         molpath_y=self.molpath_y
                                         )
        maccs_predictor.to(device)

        # Создаем датасет молекул с MACCS-ключами
        dataset = self._create_maccs_dataset(smiles_list)

        if len(dataset) < batch_size:
            print(f"Not enough molecules: {len(dataset)} < {batch_size}")
            return

        optimizer = torch.optim.AdamW(
            list(maccs_predictor.maccs_ruleparse.parameters()) +
            list(maccs_predictor.predictor.parameters()),
            lr=lr, weight_decay=1e-6
        )

        criterion = torch.nn.BCELoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            maccs_predictor.train()
            total_loss = 0
            num_batches = 0

            for batch in tqdm(dataloader, desc=f"MACCS Epoch {epoch+1}"):
                try:
                    # Перемещаем ВЕСЬ батч на устройство
                    batch = batch.to(device)

                    # Получаем предсказания
                    predictions = maccs_predictor(batch)

                    # Проверяем, что все предсказания в диапазоне [0, 1]
                    if predictions.min() < 0 or predictions.max() > 1:
                        print(f"ERROR: Predictions out of bounds! min={predictions.min().item():.6f}, max={predictions.max().item():.6f}")
                        # Принудительно ограничиваем диапазон
                        predictions = torch.clamp(predictions, min=0.0, max=1.0)
                        print(f"After clamping: [{predictions.min().item():.6f}, {predictions.max().item():.6f}]")

                    # Проверяем на NaN и бесконечности
                    if torch.isnan(predictions).any():
                        print("ERROR: NaN in predictions!")
                        continue

                    if torch.isinf(predictions).any():
                        print("ERROR: Inf in predictions!")
                        continue

                    # Получаем истинные MACCS-ключи
                    true_maccs = batch.maccs

                    # Проверяем и исправляем форму целевых меток
                    if true_maccs.dim() == 1:
                        true_maccs = true_maccs.view(predictions.shape)

                    # Проверяем целевые метки
                    #print(f"True MACCS range: [{true_maccs.min().item():.0f}, {true_maccs.max().item():.0f}]")

                    # Убедимся, что целевые метки содержат только 0 и 1
                    unique_vals = torch.unique(true_maccs)
                    #print(f"Unique values in true_maccs: {unique_vals}")

                    if not torch.all((true_maccs == 0) | (true_maccs == 1)):
                        print("WARNING: true_maccs contains values other than 0 and 1!")
                        # Принудительно преобразуем к 0 и 1
                        true_maccs = torch.where(true_maccs > 0.5, 1.0, 0.0)

                    # Убедимся, что формы совпадают
                    assert predictions.shape == true_maccs.shape, \
                        f"Shape mismatch: predictions {predictions.shape} vs true_maccs {true_maccs.shape}"

                    # Вычисляем потерю
                    loss = criterion(predictions, true_maccs)

                    #print(f"Loss: {loss.item():.6f}")

                    if torch.isnan(loss):
                        print("WARNING: NaN loss!")
                        continue

                    # Оптимизация
                    optimizer.zero_grad()
                    loss.backward()

                    # Проверяем градиенты
                    total_norm = 0.0
                    for p in maccs_predictor.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5
                    #print(f"Gradient norm: {total_norm:.6f}")

                    torch.nn.utils.clip_grad_norm_(
                        list(maccs_predictor.maccs_ruleparse.parameters()) +
                        list(maccs_predictor.predictor.parameters()),
                        max_norm=1.0
                    )
                    optimizer.step()

                    total_loss += loss.item()
                    num_batches += 1

                except Exception as e:
                    print(f"Error in MACCS training: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            if num_batches > 0:
                avg_loss = total_loss / num_batches
                current_lr = scheduler.get_last_lr()[0]
                print(f"MACCS Epoch {epoch+1}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")

            scheduler.step()

        # Переносим обученные веса RuleParse в основной парсер
        self._transfer_maccs_weights(maccs_predictor.maccs_ruleparse)
        print("MACCS pretraining completed and weights transferred to RuleParse!")

    def pretrain(self, smiles_list, epochs=50, batch_size=64, lr=1e-4,
                 contrastive_ratio=0.6, save_path=None):
        """
        Интегрированное предобучение с переносом весов
        """
        # Создаем графы для предобучения
        graphs = self._create_pretraining_graphs(smiles_list)

        if len(graphs) < batch_size:
            print(f"Not enough graphs: {len(graphs)} < {batch_size}")
            return

        # Вычисляем количество эпох для каждого этапа
        contrastive_epochs = int(epochs * contrastive_ratio)
        masked_epochs = epochs - contrastive_epochs

        print(f"Starting integrated pretraining:")
        print(f"  - Contrastive learning: {contrastive_epochs} epochs")
        print(f"  - Masked modeling: {masked_epochs} epochs")

        # Этап 1: Контрастивное обучение
        if contrastive_epochs > 0:
            print("Phase 1: Contrastive Learning")
            self._contrastive_pretrain_phase(graphs, contrastive_epochs, batch_size, lr)

        # Этап 2: Маскированное моделирование
        if masked_epochs > 0:
            print("Phase 2: Masked Modeling")
            self._masked_modeling_pretrain_phase(graphs, masked_epochs, batch_size, lr)

        # Сохраняем веса если указан путь
        if save_path:
            self.save_pretrained_weights(save_path)

        self.pretrained = True
        print("Integrated pretraining completed successfully!")

    def comprehensive_pretrain(self, smiles_list, epochs=100, batch_size=64, lr=1e-4,
                              contrastive_ratio=0.4, maccs_ratio=0.3, masked_ratio=0.3,
                              save_path=None):
        """
        Комплексное предобучение с тремя этапами:
        1. Контрастивное обучение (молекулярный энкодер)
        2. MACCS предобучение (RuleParse)
        3. Маскированное моделирование (уточнение)
        """

        # Проверка пропорций
        total_ratio = contrastive_ratio + maccs_ratio + masked_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            print(f"Warning: Ratios sum to {total_ratio}, normalizing to 1.0")
            contrastive_ratio /= total_ratio
            maccs_ratio /= total_ratio
            masked_ratio /= total_ratio

        # Создаем графы
        graphs = self._create_pretraining_graphs(smiles_list)

        if len(graphs) < batch_size:
            print(f"Not enough graphs: {len(graphs)} < {batch_size}")
            return

        # Вычисляем количество эпох для каждого этапа
        contrastive_epochs = int(epochs * contrastive_ratio)
        maccs_epochs = int(epochs * maccs_ratio)
        masked_epochs = epochs - contrastive_epochs - maccs_epochs

        print("Starting comprehensive pretraining:")
        print(f"  - Contrastive learning: {contrastive_epochs} epochs")
        print(f"  - MACCS pretraining: {maccs_epochs} epochs")
        print(f"  - Masked modeling: {masked_epochs} epochs")

        # Этап 1: Контрастивное обучение (молекулярный энкодер)
        if contrastive_epochs > 0:
            print("\n=== Phase 1: Contrastive Learning ===")
            self._contrastive_pretrain_phase(graphs, contrastive_epochs, batch_size, lr)

        # Этап 2: MACCS предобучение (RuleParse)
        if maccs_epochs > 0 and self.use_maccs_pretraining:
            print("\n=== Phase 2: MACCS-based RuleParse Pretraining ===")
            self.pretrain_maccs(smiles_list, maccs_epochs, batch_size, lr)

        # Этап 3: Маскированное моделирование (уточнение)
        if masked_epochs > 0:
            print("\n=== Phase 3: Masked Modeling ===")
            self._masked_modeling_pretrain_phase(graphs, masked_epochs, batch_size, lr)

        # Сохраняем веса если указан путь
        if save_path:
            self.save_pretrained_weights(save_path)

        self.pretrained = True
        print("\nComprehensive pretraining completed successfully!")

    def save_pretrained_weights(self, path):
        """Сохранение предобученных весов"""
        weights = {
            'gnn_encoder': self.gnn_encoder.state_dict(),
            'projection_head': self.projection_head.state_dict(),
            'atom_predictor': self.atom_predictor.state_dict(),
            'bond_predictor': self.bond_predictor.state_dict(),
            'parser': self.parser.state_dict(),
        }
        torch.save(weights, path)
        print(f"Pretrained weights saved to {path}")

    def load_pretrained_weights(self, path):
        """Загрузка предобученных весов"""
        weights = torch.load(path)
        self.gnn_encoder.load_state_dict(weights['gnn_encoder'])
        self.projection_head.load_state_dict(weights['projection_head'])
        self.atom_predictor.load_state_dict(weights['atom_predictor'])
        self.bond_predictor.load_state_dict(weights['bond_predictor'])
        self.parser.load_state_dict(weights['parser'])
        self.pretrained = True
        print(f"Pretrained weights loaded from {path}")

    def fit(self, data: MolFrame, lr: float = 1e-5, verbose: bool = True,
            gamma: int = 2, freeze_pretrained: bool = False) -> 'Generator':
        """
        Обучение с поддержкой заморозки предобученных слоев
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device)

        # Настраиваем параметры для оптимизации
        if self.pretrained and freeze_pretrained:
            # Замораживаем предобученные компоненты
            for param in self.gnn_encoder.parameters():
                param.requires_grad = False

            # Обучаем только специфичные для генерации компоненты
            optimizer_params = [
                {'params': self.attention.parameters(), 'lr': lr},
                {'params': self.final_proj.parameters(), 'lr': lr},
                {'params': self.parser.parameters(), 'lr': lr * 0.1}  # Меньший LR для парсера
            ]
            print("Training with frozen pretrained weights")
        else:
            # Обучаем все параметры
            optimizer_params = self.parameters()
            print("Training all parameters")

        criterion = AsymmetricBCELoss(gamma=gamma)
        optimizer = torch.optim.AdamW(optimizer_params, lr=lr, betas=(0.9, 0.99), weight_decay=1e-10)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )

        if verbose:
            print('Starting DataLoaders generation')

        train_loader = []
        vecs = data.reaction_labels

        for substrate in data.map:
            datum = data.single[substrate].clone()
            # Преобразуем целевые метки в форму [num_rules]
            datum.y = torch.tensor(vecs[substrate], dtype=torch.float32)
            train_loader.append(datum)

        train_loader = DataLoader(train_loader, batch_size=128, shuffle=True)

        history = []
        best_loss = float('inf')
        for _ in tqdm(range(100)):
            self.train()
            epoch_loss = 0
            for batch in train_loader:
                batch = batch.to(device)
                out = self(batch)

                # Правильное преобразование формы целевых меток
                target = batch.y.view(out.shape[0], -1)  # [batch_size, num_rules]

                # Проверка размерностей
                assert out.shape == target.shape, \
                    f"Shape mismatch: out {out.shape} vs target {target.shape}"

                loss = criterion(out, target)
                if verbose:
                    history.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            if verbose:
                print(f'Loss {epoch_loss}')
            scheduler.step(epoch_loss)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(self.state_dict(), 'best_generator.pth')

        return self

    @torch.no_grad()
    def generate(self, sub: str, pca: bool = True) -> list[str]:
        self.eval()
        mol = Chem.MolFromSmiles(sub)
        sub_mol = from_rdmol(mol)
        if pca:
            ats = Path(__file__).parent / '..' / 'data' / 'pca_ats_single.pkl'
            bonds = Path(__file__).parent / '..' / 'data' / 'pca_bonds_single.pkl'
            with open(ats, 'rb') as file:
                pca_x = pkl.load(file)
            with open(bonds, 'rb') as file:
                pca_b = pkl.load(file)
            for i in range(len(sub_mol.x)):
                for j in range(len(sub_mol.x[i])):
                    if sub_mol.x[i][j] == float('inf'):
                        sub_mol.x[i][j] = 0
            sub_mol.x = torch.tensor(SimpleImputer(missing_values=np.nan,
                                          strategy='constant',
                                          fill_value=0).fit_transform(sub_mol.x))
            sub_mol.x = torch.tensor(pca_x.transform(sub_mol.x))
            try:
                sub_mol.edge_attr = torch.tensor(pca_b.transform(sub_mol.edge_attr))
            except ValueError:
                print('Some issue happened with this molecule:')
                print(sub)
        vector = cpunum(self(sub_mol).squeeze())

        # Adaptive thresholding
        top_k = max(1, int(len(vector) * 0.1))
        threshold = np.partition(vector, -top_k)[-top_k]

        active_rules = np.where(vector >= threshold)[0]
        out = []
        for rule in active_rules:
            rxn = ReactionFromSmarts(list(self.rules.keys())[rule])
            try:
                mols_prebuild = chain.from_iterable(rxn.RunReactants((mol,)))
            except ValueError:
                continue
            if not mols_prebuild:
                continue
            else:
                mols_splitted = []
                for preb in mols_prebuild:
                    mols_splitted += Chem.MolToSmiles(preb).split('.')
                mols_splitted = [x for x in mols_splitted if iscorrect(x)]
                mols_splitted = list(map(Chem.MolFromSmiles, mols_splitted))
                mols_splitted = [x for x in mols_splitted if x is not None]
                if not mols_splitted:
                    continue
                try:
                    mols_standart = list(map(standardize_mol, mols_splitted))
                except Chem.KekulizeException:
                    continue
                except RuntimeError:
                    continue
                except Chem.AtomValenceException:
                    continue
                for stand in mols_standart:
                    out.append(stand)
        return out

    @torch.no_grad()
    def jaccard(self, test_frame: MolFrame) -> tp.List[float]:
        self.eval()
        jaccards = []
        for sub in test_frame.map:
            mols = set([Chem.MolToSmiles(x) for x in self.generate(sub)])
            reals = test_frame.map[sub]
            jaccards.append(len(reals & mols) / len(reals | mols))
        return jaccards

class SimpleGenerator(GGenerator):
    def __init__(self, rules: tp.List[str]):
        self.rules = rules

    def fit(self, data: MolFrame):
        pass

    def generate(self, sub: str) -> tp.List[str]:
        mol = Chem.MolFromSmiles(sub)
        out = []
        for i, rule in enumerate(tqdm(self.rules)):
            rxn = ReactionFromSmarts(rule)
            try:
                mols_prebuild = chain.from_iterable(rxn.RunReactants((mol,)))
            except ValueError:
                continue
            if not mols_prebuild:
                continue
            else:
                mols_splitted = []
                for preb in mols_prebuild:
                    mols_splitted += Chem.MolToSmiles(preb).split('.')
                mols_splitted = [x for x in mols_splitted if iscorrect(x)]
                mols_splitted = list(map(Chem.MolFromSmiles, mols_splitted))
                mols_splitted = [x for x in mols_splitted if x is not None]
                if not mols_splitted:
                    continue
                try:
                    mols_standart = list(map(standardize_mol, mols_splitted))
                except Chem.KekulizeException:
                    continue
                except RuntimeError:
                    continue
                except Chem.AtomValenceException:
                    continue
                for stand in mols_standart:
                    out.append(stand)
        return out
