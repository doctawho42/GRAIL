import re
from rdkit import Chem
from rdkit.Chem import AllChem

from torch_geometric.data import Data
from rdkit.Chem import rdEHTTools
from rdkit.Chem import rdFMCS
import torch
from typing import List, Dict, Any, Optional
from torch_geometric.transforms import AddLaplacianEigenvectorPE
from numpy.linalg import eig
from torch_geometric.utils import (
    get_laplacian,
    to_scipy_sparse_matrix
)

import numpy as np

bond_types = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                       [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                       [1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                       [1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                       [1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
                       [1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
                       [1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
                       [1., 0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                       [1., 1., 0.5, 0., 0., 0., 0., 0., 0., 0., 0.],
                       [1., 1., 1., 0.5, 0., 0., 0., 0., 0., 0., 0.],
                       [1., 1., 1., 1., 0.5, 0., 0., 0., 0., 0., 0.],
                       [1., 1., 1., 1., 1., 0.5, 0., 0., 0., 0., 0.],
                       [1., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                       [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                       [1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
                       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])

x_map: Dict[str, List[Any]] = {
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
        'CHI_TETRAHEDRAL',
        'CHI_ALLENE',
        'CHI_SQUAREPLANAR',
        'CHI_TRIGONALBIPYRAMIDAL',
        'CHI_OCTAHEDRAL',
    ],
    'hybridization': list(np.eye(8)),
}

e_map: Dict[str, List[Any]] = {
    'bond_type': list(bond_types),
    'stereo': list(np.eye(6)),
}

oh_matrix = np.eye(16)
one_hot = {
    '6': oh_matrix[0],
    '7': oh_matrix[1],
    '8': oh_matrix[2],
    '9': oh_matrix[3],
    '15': oh_matrix[4],
    '16': oh_matrix[5],
    '17': oh_matrix[6],
    '35': oh_matrix[7],
    '53': oh_matrix[8],
    'P': oh_matrix[4],
    'S': oh_matrix[5],
    'C': oh_matrix[0],
    'c': oh_matrix[9],
    'n': oh_matrix[10],
    'N': oh_matrix[1],
    'O': oh_matrix[2],
    'F': oh_matrix[3],
    'Cl': oh_matrix[6],
    'Br': oh_matrix[7],
    'I': oh_matrix[8],
    '=': oh_matrix[12] + oh_matrix[13],
    '-': oh_matrix[13],
    ':': oh_matrix[13] + 0.5*oh_matrix[12],
    'H': oh_matrix[14],
    'X': oh_matrix[15],
    'R': oh_matrix[15],
    '*': np.ones(16)
}

def from_rdmol(mol: Any) -> Optional[Data]:
    r"""Converts a :class:`rdkit.Chem.Mol` instance to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        mol (rdkit.Chem.Mol): The :class:`rdkit` molecule.
    """

    assert isinstance(mol, Chem.Mol)
    mol.ComputeGasteigerCharges()

    xs: List[List[float]] = []
    for atom in mol.GetAtoms():  # type: ignore
        row: List[float] = []
        row.append(atom.GetAtomicNum() / 83)
        row.append(atom.GetTotalDegree() / 11)
        row.append(atom.GetFormalCharge())
        row.append(atom.GetTotalNumHs() / 9)
        row.append(atom.GetNumRadicalElectrons() / 5)
        row.extend(x_map['hybridization'][int(atom.GetHybridization())])
        row.append(int(atom.GetIsAromatic()))
        row.append(int(atom.IsInRing()))
        row.append(float(atom.GetProp('_GasteigerCharge')))
        xs.append(row)
        if len(row) == 15:
            return None

    x = torch.tensor(xs, dtype=torch.double).view(-1, 16)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():  # type: ignore
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        e = []
        e.extend(e_map['bond_type'][int(bond.GetBondType())])
        e.extend(e_map['stereo'][int(bond.GetStereo())])
        e.append(int(bond.GetIsConjugated()))

        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.double).view(-1, 18)

    if edge_index.numel() > 0:  # Sort indices.
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def from_pair(mol1: Any, mol2: Any) -> Optional[Data]:
    r"""Converts a pair of :class:`rdkit.Chem.Mol` objects to a
    :class:`torch_geometric.data.Data` graph with connected
    largest common substructures.

    Args:
        mol1 (rdkit.Chem.Mol): The :class:`rdkit` molecule.
        mol2 (rdkit.Chem.Mol): The :class:`rdkit` molecule.
    """

    assert isinstance(mol1, Chem.Mol) & isinstance(mol2, Chem.Mol)
    mol1.ComputeGasteigerCharges()
    mol2.ComputeGasteigerCharges()

    xs: List[List[float]] = []
    for atom in mol1.GetAtoms():  # type: ignore
        row: List[float] = []
        row.append(atom.GetAtomicNum() / 83)
        row.append(atom.GetTotalDegree() / 11)
        row.append(atom.GetFormalCharge())
        row.append(atom.GetTotalNumHs() / 9)
        row.append(atom.GetNumRadicalElectrons() / 5)
        row.extend(x_map['hybridization'][int(atom.GetHybridization())])
        row.append(int(atom.GetIsAromatic()))
        row.append(int(atom.IsInRing()))
        row.append(float(atom.GetProp('_GasteigerCharge')))
        row.append(0)
        xs.append(row)

    for atom in mol2.GetAtoms():
        row: List[float] = []
        row.append(atom.GetAtomicNum() / 83)
        row.append(atom.GetTotalDegree() / 11)
        row.append(atom.GetFormalCharge())
        row.append(atom.GetTotalNumHs() / 9)
        row.append(atom.GetNumRadicalElectrons() / 5)
        row.extend(x_map['hybridization'][int(atom.GetHybridization())])
        row.append(int(atom.GetIsAromatic()))
        row.append(int(atom.IsInRing()))
        row.append(float(atom.GetProp('_GasteigerCharge')))
        row.append(1)
        xs.append(row)

    xs.append(list(np.zeros(17)))
    xs.append(list(np.zeros(17)))

    x = torch.tensor(xs, dtype=torch.double).view(-1, 17)

    edge_indices, edge_attrs = [], []
    idxs = set()
    count_1 = 0
    for bond in mol1.GetBonds():  # type: ignore
        i = bond.GetBeginAtomIdx()
        idxs.add(i)
        j = bond.GetEndAtomIdx()
        idxs.add(j)

        e = []
        e.extend(e_map['bond_type'][int(bond.GetBondType())])
        e.extend(e_map['stereo'][int(bond.GetStereo())])
        e.append(int(bond.GetIsConjugated()))

        count_1 += 2
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    idx_1 = max(list(idxs))
    count_2 = 0
    second_indices = []
    for bond in mol2.GetBonds():  # type: ignore
        i = bond.GetBeginAtomIdx() + idx_1 + 1
        j = bond.GetEndAtomIdx() + idx_1 + 1

        e = []
        e.extend(e_map['bond_type'][int(bond.GetBondType())])
        e.extend(e_map['stereo'][int(bond.GetStereo())])
        e.append(int(bond.GetIsConjugated()))

        count_2 += 2
        edge_indices += [[i, j], [j, i]]
        second_indices += [[i - idx_1 - 1, j - idx_1 - 1], [j - idx_1 - 1, i - idx_1 - 1]]
        edge_attrs += [e, e]

    try:
        edge_index_1, edge_weight_1 = get_laplacian(
            torch.tensor(edge_indices[:count_1], dtype=torch.int64).T,
            normalization='sym',
            num_nodes=count_1
        )
    except IndexError:
        return None

    try:
        edge_index_2, edge_weight_2 = get_laplacian(
            torch.tensor(second_indices, dtype=torch.int64).T,
            normalization='sym',
            num_nodes=count_2
        )
    except IndexError:
        return None

    L1 = to_scipy_sparse_matrix(edge_index_1, edge_weight_1, count_1)
    L2 = to_scipy_sparse_matrix(edge_index_2, edge_weight_2, count_2)
    eig_vals, eig_vecs = eig(L1.todense())
    eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()])
    pe = torch.from_numpy(eig_vecs[:, 1:1 + 1])
    pe = pe[pe.nonzero()].squeeze(dim=-1)[..., 0].unsqueeze(-1)
    eig_vals, eig_vecs = eig(L2.todense())
    eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()])
    sec_pe = torch.from_numpy(eig_vecs[:, 1:1 + 1])
    sec_pe = sec_pe[sec_pe.nonzero()].squeeze(-1)[..., 0].unsqueeze(-1)
    pe = torch.cat((pe, sec_pe, torch.tensor([[0], [0]])), dim=0)
    sign = -1 + 2 * torch.randint(0, 2, (1,))
    pe *= sign

    common = rdFMCS.FindMCS([mol1, mol2],
                            atomCompare=rdFMCS.AtomCompare.CompareAnyHeavyAtom,
                            bondCompare=rdFMCS.BondCompare.CompareAny, timeout=10)
    common_struct = Chem.MolFromSmarts(common.smartsString)
    virtual_1 = len(xs) - 2
    virtual_2 = len(xs) - 1

    for i in mol1.GetSubstructMatch(common_struct):
        edge_indices += [[i, virtual_1], [virtual_1, i]]
        e = list(np.zeros(18))
        edge_attrs += [e, e]
    for i in mol2.GetSubstructMatch(common_struct):
        edge_indices += [[i + idx_1 + 1, virtual_2], [virtual_2, i + idx_1 + 1]]
        e = list(np.zeros(18))
        edge_attrs += [e, e]

    edge_indices += [[virtual_1, virtual_2], [virtual_2, virtual_1]]
    e = list(np.zeros(18))
    edge_attrs += [e, e]
    try:
        x = torch.cat((x, pe), dim=-1)
    except RuntimeError:
        return None
    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.double).view(2, -1)
    torch.tensor(edge_attrs, dtype=torch.double)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.double).view(-1, 18)

    if edge_index.numel() > 0:  # Sort indices.
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def apply_operation(vec1, vec2, operator):
    if operator in [';', '&']:
        return vec1 + vec2
    elif operator == ',':
        return (vec1 + vec2) / 2
    elif operator == '!':
        return -vec1
    return vec1  # В случае неизвестной операции возвращаем vec1


# Функция для разбора выражения
def parse_expression(expr):
    expr = expr.replace('[', '(').replace(']', ')')
    tokens = re.findall(r'(\d+|[HcCOSPFBrlI=\-&,();:X])', expr)
    stack = []
    current_vec = np.zeros(16)
    operators = []

    for i, token in enumerate(tokens):
        try:
            if token == 'C' and tokens[i + 1] == 'l':
                tokens[i] += 'l'
            elif token == 'B' and tokens[i + 1] == 'r':
                tokens[i] += 'r'
        except IndexError:
            pass

    for token in tokens:
        if token.isdigit():  # Если это число
            if stack and stack[-1] in ['H', 'X', 'R']:  # Да, это H или X
                atom = stack.pop()
                current_vec += one_hot[atom] * int(token)
                continue

        if token in one_hot:  # Если это атом
            stack.append(token)
            current_vec += one_hot[token]

        elif token in [';', ',', '&', '!']:  # Если это операция
            operators.append(token)

        elif token == '(':  # Начало подвыражения
            stack.append(token)

        elif token == ')':  # Конец подвыражения
            sub_vec = np.zeros(16)
            while stack and stack[-1] != '(':
                sub_vec += one_hot[stack.pop()]
            stack.pop()  # Удаляем '('
            current_vec += sub_vec  # Добавляем подвыражение к текущему
    return current_vec


def from_rule(rule: str) -> Data:
    sub, prod = [
        Chem.MolFromSmarts(mol)
        if not mol.startswith('(') else Chem.MolFromSmarts(mol[1:-1])
        for mol in rule.split('>>')
    ]
    xs = []
    for atom in sub.GetAtoms():
        if atom.GetSmarts().startswith('['):
            xs.append(parse_expression(atom.GetSmarts()[1:-1].split(':')[0]))
        else:
            xs.append(one_hot[atom.GetSmarts()])
    for atom in prod.GetAtoms():
        if atom.GetSmarts().startswith('['):
            xs.append(parse_expression(atom.GetSmarts()[1:-1].split(':')[0]))
        else:
            xs.append(one_hot[atom.GetSmarts()])
    x = torch.tensor(xs, dtype=torch.double).view(-1, 16)

    edge_indices, edge_attrs = [], []
    idxs = set()
    count_1 = 0
    for bond in sub.GetBonds():  # type: ignore
        i = bond.GetBeginAtomIdx()
        idxs.add(i)
        j = bond.GetEndAtomIdx()
        idxs.add(j)

        e = []
        e.extend(e_map['bond_type'][int(bond.GetBondType())])
        e.extend(e_map['stereo'][int(bond.GetStereo())])
        e.append(int(bond.GetIsConjugated()))

        count_1 += 2
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    try:
        idx_1 = max(list(idxs))
    except Exception:
        idx_1 = 0
    count_2 = 0
    second_indices = []
    for bond in prod.GetBonds():  # type: ignore
        i = bond.GetBeginAtomIdx() + idx_1 + 1
        j = bond.GetEndAtomIdx() + idx_1 + 1

        e = []
        e.extend(e_map['bond_type'][int(bond.GetBondType())])
        e.extend(e_map['stereo'][int(bond.GetStereo())])
        e.append(int(bond.GetIsConjugated()))

        count_2 += 2
        edge_indices += [[i, j], [j, i]]
        second_indices += [[i - idx_1 - 1, j - idx_1 - 1],
                           [j - idx_1 - 1, i - idx_1 - 1]]
        edge_attrs += [e, e]

    for atom in sub.GetAtoms():
        try:
            num = atom.GetProp('molAtomMapNumber')
        except KeyError:
            continue
        for sec_atom in prod.GetAtoms():
            try:
                if sec_atom.GetProp('molAtomMapNumber') == num:
                    a, b = atom.GetIdx(), sec_atom.GetIdx() + idx_1 + 1
                    edge_indices += [[a, b], [b, a]]
                    e = list(np.zeros(18))
                    edge_attrs += [e, e]
            except KeyError:
                continue

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.double).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.double).view(-1, 18)

    if edge_index.numel() > 0:  # Sort indices.
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)