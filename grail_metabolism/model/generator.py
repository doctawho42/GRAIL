import torch
from pathlib import Path
import pickle as pkl
from sklearn.impute import SimpleImputer
import numpy as np
from torch.nn import Module, Sequential, ReLU, Linear, Bilinear, MultiheadAttention, BatchNorm1d, Dropout, init, Sigmoid
from torch.nn.functional import dropout
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
from .train_model import AsymmetricBCELoss
import matplotlib.pyplot as plt

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
            (GATv2Conv(16, 100, edge_dim=18, dropout=0.25), 'x, edge_index, edge_attr -> x'),
            ReLU(inplace=True),
            BatchNorm1d(100),
            (GATv2Conv(100, 200, edge_dim=18, dropout=0.25), 'x, edge_index, edge_attr -> x'),
            ReLU(inplace=True),
            BatchNorm1d(200),
            Linear(200, 400),
            BatchNorm1d(400)
        ])
        self.ffn = Sequential(
            Linear(400, 200),
            ReLU(inplace=True),
            Linear(200, 100),
            ReLU(inplace=True),
            BatchNorm1d(100),
            Linear(100, 100)
        )
        self.batch_norm = BatchNorm1d(400)

    def forward(self) -> torch.Tensor:
        # Create the batch of rules
        batch = Batch.from_data_list(list(self.rule_dict.values()))
        batch.x = batch.x.to(torch.float32)
        batch.edge_attr = batch.edge_attr.to(torch.float32)
        batch.edge_index = batch.edge_index.to(torch.int64)
        # Apply Module to the batch
        data = self.module(batch.x, batch.edge_index, batch.edge_attr)
        x = global_mean_pool(data, batch.batch)
        x = self.batch_norm(x)
        # Apply the feed-forward network
        x = self.ffn(x)
        return x

class Generator(GGenerator):
    def __init__(self, rule_dict: dict[str, Batch], in_channels: int, edge_dim: int, arg_vec: tp.Optional[tp.List[int]] = None) -> None:
        super(Generator, self).__init__()
        self.parser = RuleParse(rule_dict)
        self.rules = rule_dict
        self.num_rules = len(rule_dict)
        if arg_vec is None:
            arg_vec = [100] * 2
        self.module = nn.Sequential('x, edge_index, edge_attr', [
            (GATv2Conv(in_channels, arg_vec[0], edge_dim=edge_dim, dropout=0.25), 'x, edge_index, edge_attr -> x'),
            ReLU(inplace=True),
            BatchNorm1d(arg_vec[0]),
            (GATv2Conv(arg_vec[0], arg_vec[1], edge_dim=edge_dim, dropout=0.25), 'x, edge_index, edge_attr -> x'),
            ReLU(inplace=True),
            BatchNorm1d(arg_vec[1]),
            (GATv2Conv(arg_vec[1], 100, edge_dim=edge_dim, dropout=0.25), 'x, edge_index, edge_attr -> x'),
        ])
        self.embed_dim = 100
        self.attention = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=4,
            batch_first=True
        )
        self.final_proj = nn.Linear(self.embed_dim, self.num_rules)
        self.sigmoid = Sigmoid()
        self.dropout = Dropout(0.2)

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
        return logits.T  # [batch_size, num_rules]
    '''

    def forward(self, data: Data) -> torch.Tensor:
        # Получаем эмбеддинги правил с явным указанием типа
        rule_emb = self.parser().to(torch.float32)  # [num_rules, 100]

        # Обрабатываем входные данные
        data.x = data.x.to(torch.float32)
        data.edge_attr = data.edge_attr.to(torch.float32)
        data.edge_index = data.edge_index.to(torch.int64)

        # Получаем эмбеддинги молекул
        x = self.module(data.x, data.edge_index, edge_attr=data.edge_attr)
        mol_emb = global_mean_pool(x, data.batch).to(torch.float32) # [batch_size, 100]
        mol_emb = self.dropout(mol_emb)

        # Вычисляем логиты
        attn_output, _ = self.attention(
            query=mol_emb.unsqueeze(1),  # [batch_size, 1, embed_dim]
            key=rule_emb.unsqueeze(0).repeat(mol_emb.size(0), 1, 1),  # [batch_size, num_rules, embed_dim]
            value=rule_emb.unsqueeze(0).repeat(mol_emb.size(0), 1, 1)  # [batch_size, num_rules, embed_dim]
        )
        logits = self.final_proj(attn_output.squeeze(1))

        return self.sigmoid(logits)

    def fit(self, data: MolFrame, lr: float = 1e-5, verbose: bool = True, gamma: int = 2) -> 'Generator':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device)

        criterion = AsymmetricBCELoss(gamma=gamma)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=1e-10)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(  # Адаптивный шедулер
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
                if verbose: plt.plot(loss.item())
                history.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            if verbose: print(f'Loss{epoch_loss}')
            scheduler.step(epoch_loss)

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
        top_k = max(1, int(len(vector) * 0.25))  # Выбираем топ 25% правил
        threshold = np.partition(vector, -top_k)[-top_k]

        active_rules = np.where(vector >= threshold)[0]
        '''threshold = 0.3 if len(vector[vector > 0.4]) == 0 else 0.4
        active_rules = torch.where(torch.tensor(vector >= threshold))[0]'''
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