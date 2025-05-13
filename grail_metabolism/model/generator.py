import torch
from pathlib import Path
import pickle as pkl
from sklearn.impute import SimpleImputer
import numpy as np
from torch.nn import Module, Sequential, ReLU, Linear, Bilinear
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_geometric import nn
from .wrapper import GGenerator
from ..utils.preparation import MolFrame, cpunum, iscorrect, standardize_mol
from ..utils.transform import from_rdmol
from tqdm.auto import tqdm
from rdkit import Chem
from rdkit.Chem.AllChem import ReactionFromSmarts
from itertools import chain
import typing as tp

def generate_vectors(reaction_dict, real_products_dict, len_of_rules):
    vectors = {}
    for substrate in reaction_dict:
        # Initialize a vector of 474 zeros
        vector = [0] * len_of_rules
        # Get the real products for this substrate, default to empty set if not present
        real_products = real_products_dict.get(substrate, set())
        # Iterate over each product and its indexes in the reaction_dict
        for product, indexes in reaction_dict[substrate].items():
            if product in real_products:
                for idx in indexes:
                    # Ensure the index is within the valid range
                    if 0 <= idx < len_of_rules:
                        vector[idx] = 1
        vectors[substrate] = vector
    return vectors

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
        # Create the batch of rules
        batch = Batch.from_data_list(list(self.rule_dict.values()))
        batch.x = batch.x.to(torch.float32)
        batch.edge_attr = batch.edge_attr.to(torch.float32)
        batch.edge_index = batch.edge_index.to(torch.int64)
        # Apply Module to the batch
        data = self.module(batch.x, batch.edge_index, batch.edge_attr)
        x = global_mean_pool(data, batch.batch)
        # Apply the feed-forward network
        x = self.ffn(x)
        return x

class Generator(GGenerator):
    def __init__(self, rule_dict: dict[str, Batch], in_channels: int, edge_dim: int) -> None:
        super(Generator, self).__init__()
        self.parser = RuleParse(rule_dict)
        self.rules = rule_dict
        self.bilinear = Bilinear(100, 100, 1)
        self.module = nn.Sequential('x, edge_index, edge_attr', [
            (GATv2Conv(in_channels, 100, edge_dim=edge_dim), 'x, edge_index, edge_attr -> x'),
        ])
        self.linear = nn.Linear(100, 100)

    '''def forward(self, data: Data) -> torch.Tensor:
        y = self.parser()  # [num_rules, 100]
        data.x = data.x.to(torch.float32)
        data.edge_attr = data.edge_attr.to(torch.float32)
        data.edge_index = data.edge_index.to(torch.int64)

        # Get embeddings for the input molecules
        x = self.module(data.x, data.edge_index, edge_attr=data.edge_attr)
        x = global_mean_pool(x, data.batch)  # [batch_size, 100]

        # Create a matrix of interactions between all rules and all molecules in the batches
        x = x.unsqueeze(0).repeat(len(self.rules), 1, 1)  # [num_rules, batch_size, 100]
        y = y.unsqueeze(1).expand(-1, x.size(1), -1)  # [num_rules, batch_size, 100]

        # Apply bilinear to get logits
        logits = self.bilinear(x, y).squeeze(-1)  # [num_rules, batch_size]

        # Transpose
        return logits.T  # [batch_size, num_rules]'''

    def forward(self, data: Data) -> torch.Tensor:
        # Получаем эмбеддинги правил с явным указанием типа
        rule_embeddings = self.parser().to(torch.float32)  # [num_rules, 100]

        # Обрабатываем входные данные
        data.x = data.x.to(torch.float32)
        data.edge_attr = data.edge_attr.to(torch.float32)
        data.edge_index = data.edge_index.to(torch.int64)

        # Получаем эмбеддинги молекул
        x = self.module(data.x, data.edge_index, edge_attr=data.edge_attr)
        x = global_mean_pool(x, data.batch).to(torch.float32)  # [batch_size, 100]

        # Вычисляем логиты
        logits = torch.matmul(x, rule_embeddings.T)  # [batch_size, num_rules]

        return logits

    def fit(self, data: MolFrame, lr: float = 1e-5, verbose: bool = True) -> 'Generator':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device)

        criterion = torch.nn.HuberLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=1e-10)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.8)

        if verbose:
            print('Starting DataLoaders generation')

        train_loader = []
        mapping_sample = data.metabolic_mapping(list(self.rules.keys()))
        num_rules = len(self.rules)  # Получаем актуальное количество правил
        vecs = generate_vectors(mapping_sample, data.map, num_rules)

        for substrate in data.map:
            datum = data.single[substrate].clone()
            # Преобразуем целевые метки в форму [num_rules]
            datum.y = torch.tensor(vecs[substrate], dtype=torch.float32)
            train_loader.append(datum)

        train_loader = DataLoader(train_loader, batch_size=128, shuffle=True)

        history = []
        for _ in tqdm(range(100)):
            self.train()
            for batch in train_loader:
                batch = batch.to(device)
                out = self(batch)

                # Правильное преобразование формы целевых меток
                target = batch.y.view(out.shape[0], -1)  # [batch_size, num_rules]

                # Проверка размерностей
                assert out.shape == target.shape, \
                    f"Shape mismatch: out {out.shape} vs target {target.shape}"

                loss = criterion(out, target)
                history.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

        return self

    def generate(self, sub: str, pca: bool = True) -> list[str]:
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
                print(sub, sub_mol.edge_attr, sub_mol.x)
        vector = cpunum(self(sub_mol).squeeze())
        out = []
        for i, rule in enumerate(tqdm(self.rules)):
            if vector[i] == 1:
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