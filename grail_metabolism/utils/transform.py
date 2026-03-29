from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem import rdFingerprintGenerator
from torch import Tensor
from torch_geometric.data import Data

SINGLE_NODE_DIM = 16
PAIR_NODE_DIM = 18
EDGE_DIM = 18
FINGERPRINT_DIM = 1024

_BOND_FEATURES = np.array(
    [
        [0.0] * 11,
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

_HYBRIDIZATION_INDEX: Dict[int, int] = {
    int(Chem.rdchem.HybridizationType.UNSPECIFIED): 0,
    int(Chem.rdchem.HybridizationType.S): 1,
    int(Chem.rdchem.HybridizationType.SP): 2,
    int(Chem.rdchem.HybridizationType.SP2): 3,
    int(Chem.rdchem.HybridizationType.SP3): 4,
    int(Chem.rdchem.HybridizationType.SP2D): 5,
    int(Chem.rdchem.HybridizationType.SP3D): 6,
    int(Chem.rdchem.HybridizationType.SP3D2): 7,
}

_ATOM_TOKENS = np.eye(16, dtype=np.float32)
_TOKEN_EMBEDDINGS: Dict[str, np.ndarray] = {
    "6": _ATOM_TOKENS[0],
    "7": _ATOM_TOKENS[1],
    "8": _ATOM_TOKENS[2],
    "9": _ATOM_TOKENS[3],
    "15": _ATOM_TOKENS[4],
    "16": _ATOM_TOKENS[5],
    "17": _ATOM_TOKENS[6],
    "35": _ATOM_TOKENS[7],
    "53": _ATOM_TOKENS[8],
    "C": _ATOM_TOKENS[0],
    "N": _ATOM_TOKENS[1],
    "O": _ATOM_TOKENS[2],
    "F": _ATOM_TOKENS[3],
    "P": _ATOM_TOKENS[4],
    "S": _ATOM_TOKENS[5],
    "Cl": _ATOM_TOKENS[6],
    "Br": _ATOM_TOKENS[7],
    "I": _ATOM_TOKENS[8],
    "c": _ATOM_TOKENS[9],
    "n": _ATOM_TOKENS[10],
    "o": _ATOM_TOKENS[2],
    "s": _ATOM_TOKENS[5],
    "H": _ATOM_TOKENS[14],
    "X": _ATOM_TOKENS[15],
    "R": _ATOM_TOKENS[15],
    "*": np.ones(16, dtype=np.float32),
    "-": _ATOM_TOKENS[13],
    "=": _ATOM_TOKENS[12] + _ATOM_TOKENS[13],
    ":": _ATOM_TOKENS[13] + 0.5 * _ATOM_TOKENS[12],
}

_MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    includeChirality=True,
    fpSize=FINGERPRINT_DIM,
    countSimulation=False,
)


def _safe_charge(atom: Chem.Atom) -> float:
    try:
        value = atom.GetProp("_GasteigerCharge")
        charge = float(value)
        if np.isfinite(charge):
            return charge
    except Exception:
        pass
    return 0.0


def _hybridization_vector(atom: Chem.Atom) -> np.ndarray:
    out = np.zeros(8, dtype=np.float32)
    index = _HYBRIDIZATION_INDEX.get(int(atom.GetHybridization()), 0)
    out[index] = 1.0
    return out


def _atom_features(atom: Chem.Atom) -> List[float]:
    features: List[float] = [
        atom.GetAtomicNum() / 118.0,
        atom.GetTotalDegree() / 8.0,
        float(atom.GetFormalCharge()),
        atom.GetTotalNumHs() / 4.0,
        atom.GetNumRadicalElectrons() / 4.0,
    ]
    features.extend(_hybridization_vector(atom).tolist())
    features.append(float(atom.GetIsAromatic()))
    features.append(float(atom.IsInRing()))
    features.append(_safe_charge(atom))
    return features


def _bond_features(bond: Chem.Bond) -> List[float]:
    bond_type_idx = min(int(bond.GetBondType()), len(_BOND_FEATURES) - 1)
    stereo = np.zeros(6, dtype=np.float32)
    stereo_idx = min(int(bond.GetStereo()), 5)
    stereo[stereo_idx] = 1.0
    return (
        _BOND_FEATURES[bond_type_idx].tolist()
        + stereo.tolist()
        + [float(bond.GetIsConjugated())]
    )


def _empty_edge_tensors(edge_dim: int = EDGE_DIM) -> tuple[Tensor, Tensor]:
    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.empty((0, edge_dim), dtype=torch.float32)
    return edge_index, edge_attr


def _sort_edges(edge_index: Tensor, edge_attr: Tensor, num_nodes: int) -> tuple[Tensor, Tensor]:
    if edge_index.numel() == 0:
        return edge_index, edge_attr
    perm = (edge_index[0] * max(num_nodes, 1) + edge_index[1]).argsort()
    return edge_index[:, perm], edge_attr[perm]


def _fingerprint(mol: Chem.Mol) -> Tensor:
    fp = np.asarray(_MORGAN_GENERATOR.GetFingerprint(mol), dtype=np.float32)
    return torch.tensor(fp, dtype=torch.float32).view(1, -1)


def _prepare_molecule(mol: Chem.Mol) -> Chem.Mol:
    prepared = Chem.Mol(mol)
    try:
        Chem.rdPartialCharges.ComputeGasteigerCharges(prepared)
    except Exception:
        pass
    return prepared


def _edge_list_from_mol(mol: Chem.Mol, offset: int = 0) -> tuple[List[List[int]], List[List[float]]]:
    edge_indices: List[List[int]] = []
    edge_attrs: List[List[float]] = []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx() + offset
        end = bond.GetEndAtomIdx() + offset
        features = _bond_features(bond)
        edge_indices.extend([[begin, end], [end, begin]])
        edge_attrs.extend([features, features])
    return edge_indices, edge_attrs


def from_rdmol(mol: Any) -> Optional[Data]:
    if not isinstance(mol, Chem.Mol):
        return None

    prepared = _prepare_molecule(mol)
    x_rows = [_atom_features(atom) for atom in prepared.GetAtoms()]
    if not x_rows:
        return None

    x = torch.tensor(x_rows, dtype=torch.float32).view(-1, SINGLE_NODE_DIM)
    edge_indices, edge_attrs = _edge_list_from_mol(prepared)
    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32).view(-1, EDGE_DIM)
        edge_index, edge_attr = _sort_edges(edge_index, edge_attr, x.size(0))
    else:
        edge_index, edge_attr = _empty_edge_tensors()

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    graph.fp = _fingerprint(prepared)
    return graph


def from_pair(mol1: Any, mol2: Any) -> Optional[Data]:
    if not isinstance(mol1, Chem.Mol) or not isinstance(mol2, Chem.Mol):
        return None

    prepared1 = _prepare_molecule(mol1)
    prepared2 = _prepare_molecule(mol2)
    graph1 = from_rdmol(prepared1)
    graph2 = from_rdmol(prepared2)
    if graph1 is None or graph2 is None:
        return None

    try:
        common = rdFMCS.FindMCS(
            [prepared1, prepared2],
            atomCompare=rdFMCS.AtomCompare.CompareAnyHeavyAtom,
            bondCompare=rdFMCS.BondCompare.CompareAny,
            timeout=5,
        )
        common_mol = Chem.MolFromSmarts(common.smartsString) if common.smartsString else None
    except Exception:
        common_mol = None

    common_sub = set(prepared1.GetSubstructMatch(common_mol)) if common_mol else set()
    common_prod = set(prepared2.GetSubstructMatch(common_mol)) if common_mol else set()

    x1 = torch.cat(
        (
            graph1.x,
            torch.zeros((graph1.x.size(0), 1), dtype=torch.float32),
            torch.tensor([[1.0 if idx in common_sub else 0.0] for idx in range(graph1.x.size(0))]),
        ),
        dim=1,
    )
    x2 = torch.cat(
        (
            graph2.x,
            torch.ones((graph2.x.size(0), 1), dtype=torch.float32),
            torch.tensor([[1.0 if idx in common_prod else 0.0] for idx in range(graph2.x.size(0))]),
        ),
        dim=1,
    )
    x = torch.cat((x1, x2), dim=0).view(-1, PAIR_NODE_DIM)

    edge_indices, edge_attrs = _edge_list_from_mol(prepared1)
    offset = graph1.x.size(0)
    edge_indices_2, edge_attrs_2 = _edge_list_from_mol(prepared2, offset=offset)
    edge_indices.extend(edge_indices_2)
    edge_attrs.extend(edge_attrs_2)

    if common_sub and common_prod:
        for left, right in zip(sorted(common_sub), sorted(common_prod)):
            features = [0.0] * EDGE_DIM
            edge_indices.extend([[left, offset + right], [offset + right, left]])
            edge_attrs.extend([features, features])

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32).view(-1, EDGE_DIM)
        edge_index, edge_attr = _sort_edges(edge_index, edge_attr, x.size(0))
    else:
        edge_index, edge_attr = _empty_edge_tensors()

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    graph.fp = torch.cat((_fingerprint(prepared1), _fingerprint(prepared2)), dim=1)
    return graph


def apply_operation(vec1: np.ndarray, vec2: np.ndarray, operator: str) -> np.ndarray:
    if operator in {";", "&"}:
        return vec1 + vec2
    if operator == ",":
        return (vec1 + vec2) / 2.0
    if operator == "!":
        return -vec1
    return vec1


def parse_expression(expr: str) -> np.ndarray:
    expr = expr.replace("#", "")
    expr = expr.replace("[", "(").replace("]", ")")
    tokens = list(_tokenize_expression(expr))
    stack: List[str] = []
    current_vec = np.zeros(16, dtype=np.float32)

    for token in tokens:
        if token.isdigit() and stack and stack[-1] in {"H", "X", "R"}:
            atom = stack.pop()
            current_vec += _TOKEN_EMBEDDINGS[atom] * int(token)
            continue

        if token in _TOKEN_EMBEDDINGS:
            stack.append(token)
            current_vec += _TOKEN_EMBEDDINGS[token]
            continue

        if token == "(":
            stack.append(token)
            continue

        if token == ")":
            sub_vec = np.zeros(16, dtype=np.float32)
            while stack and stack[-1] != "(":
                sub_vec += _TOKEN_EMBEDDINGS.get(stack.pop(), 0.0)
            if stack and stack[-1] == "(":
                stack.pop()
            current_vec += sub_vec

    return current_vec


def _tokenize_expression(expr: str) -> Iterable[str]:
    raw = Chem.MolFromSmarts(f"[{expr}]")
    if raw is None:
        for token in ("Cl", "Br"):
            expr = expr.replace(token, f" {token} ")
        for char in "()!;,&:-=XRHCONSPFIcnosp0123456789":
            expr = expr.replace(char, f" {char} ")
        return [token for token in expr.split() if token]

    tokens: List[str] = []
    current = ""
    for char in expr:
        if char.isalnum():
            current += char
            continue
        if current:
            tokens.append(current)
            current = ""
        if not char.isspace():
            tokens.append(char)
    if current:
        tokens.append(current)

    merged: List[str] = []
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if idx + 1 < len(tokens) and token == "C" and tokens[idx + 1] == "l":
            merged.append("Cl")
            idx += 2
            continue
        if idx + 1 < len(tokens) and token == "B" and tokens[idx + 1] == "r":
            merged.append("Br")
            idx += 2
            continue
        merged.append(token)
        idx += 1
    return merged


def _rule_atom_features(atom: Chem.Atom) -> np.ndarray:
    smarts = atom.GetSmarts()
    if smarts.startswith("["):
        return parse_expression(smarts[1:-1].split(":")[0])
    return _TOKEN_EMBEDDINGS.get(smarts, np.zeros(16, dtype=np.float32))


def from_rule(rule: str) -> Data:
    parts = rule.split(">>")
    if len(parts) != 2:
        raise ValueError(f"Invalid SMARTS rule: {rule}")

    sub_smarts, prod_smarts = parts
    sub = Chem.MolFromSmarts(sub_smarts[1:-1] if sub_smarts.startswith("(") and sub_smarts.endswith(")") else sub_smarts)
    prod = Chem.MolFromSmarts(prod_smarts[1:-1] if prod_smarts.startswith("(") and prod_smarts.endswith(")") else prod_smarts)
    if sub is None or prod is None:
        raise ValueError(f"Failed to parse SMARTS rule: {rule}")

    xs = [_rule_atom_features(atom) for atom in sub.GetAtoms()]
    xs.extend(_rule_atom_features(atom) for atom in prod.GetAtoms())
    x = torch.tensor(np.asarray(xs, dtype=np.float32), dtype=torch.float32).view(-1, SINGLE_NODE_DIM)

    edge_indices, edge_attrs = _edge_list_from_mol(sub)
    sub_offset = sub.GetNumAtoms()
    prod_indices, prod_attrs = _edge_list_from_mol(prod, offset=sub_offset)
    edge_indices.extend(prod_indices)
    edge_attrs.extend(prod_attrs)

    for atom in sub.GetAtoms():
        if not atom.HasProp("molAtomMapNumber"):
            continue
        atom_map = atom.GetProp("molAtomMapNumber")
        for prod_atom in prod.GetAtoms():
            if prod_atom.HasProp("molAtomMapNumber") and prod_atom.GetProp("molAtomMapNumber") == atom_map:
                features = [0.0] * EDGE_DIM
                left = atom.GetIdx()
                right = prod_atom.GetIdx() + sub_offset
                edge_indices.extend([[left, right], [right, left]])
                edge_attrs.extend([features, features])

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32).view(-1, EDGE_DIM)
        edge_index, edge_attr = _sort_edges(edge_index, edge_attr, x.size(0))
    else:
        edge_index, edge_attr = _empty_edge_tensors()

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def apply_feature_projection(
    graph: Data,
    node_projector: Optional[Any] = None,
    edge_projector: Optional[Any] = None,
) -> Data:
    projected = graph.clone()
    if node_projector is not None and projected.x.numel() > 0:
        projected.x = torch.tensor(node_projector.transform(projected.x.numpy()), dtype=torch.float32)
    if edge_projector is not None and projected.edge_attr.numel() > 0:
        projected.edge_attr = torch.tensor(edge_projector.transform(projected.edge_attr.numpy()), dtype=torch.float32)
    return projected
