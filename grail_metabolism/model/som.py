"""Site-of-metabolism (SoM) regioselectivity prior.

The generator scores *rules per substrate* but does not say *where* on the substrate a
rule fires, so two products of the same rule at different atoms rank identically. Much of
the recall gap is this regioselectivity miss (right reaction, wrong site -> InChIKey
mismatch). This module learns a per-atom "how likely is this the reaction site" signal
from the annotated pairs themselves and uses it as a soft multiplicative reweight at
ranking time -- never a hard gate (see ModelWrapper.generate).

Labels are derived from each (substrate, metabolite) pair by element-aware MCS: a
substrate atom is a site of metabolism if it is outside the common core OR its local
environment (bonding/H-count/charge/aromaticity) changes between substrate and product,
plus its 1-hop neighbors. The SAME `_reacting_atoms` routine localizes sites at inference,
so there is no train/inference skew.

See docs/superpowers/specs/2026-06-21-som-prior-design.md.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Set

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdFMCS
from torch import nn
from torch_geometric.data import Data

from ..utils.transform import EDGE_DIM, SINGLE_NODE_DIM, from_rdmol
from ._graph import GraphEncoder


def _atom_env(atom: Chem.Atom) -> tuple:
    """Local environment signature; changes here mark a reaction site even when the
    permissive (CompareAny) MCS keeps the atom in the common core (e.g. C-O -> C=O).
    Bond orders are summed and doubled so aromatic (1.5) stays an exact integer."""
    bond_order_x2 = int(round(sum(b.GetBondTypeAsDouble() for b in atom.GetBonds()) * 2))
    return (
        atom.GetDegree(),
        bond_order_x2,
        atom.GetTotalNumHs(),
        atom.GetFormalCharge(),
        int(atom.GetIsAromatic()),
    )


def _reacting_atoms(sub_mol: Chem.Mol, prod_mol: Chem.Mol) -> Set[int]:
    """Substrate atom indices that change between substrate and product, + 1-hop neighbors.

    Indices are into ``sub_mol`` (which must be the SMILES-parsed substrate, so they align
    with ``from_rdmol`` graph nodes; _prepare_molecule preserves atom order). Returns an
    empty set on MCS failure or when nothing changed (identity), which the caller treats as
    "no SoM information" (neutral)."""
    if sub_mol is None or prod_mol is None:
        return set()
    n = sub_mol.GetNumAtoms()
    try:
        res = rdFMCS.FindMCS(
            [sub_mol, prod_mol],
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareAny,
            timeout=5,
        )
    except Exception:
        return set()
    core = Chem.MolFromSmarts(res.smartsString) if res.smartsString else None
    if core is None:
        return set()
    match_sub = sub_mol.GetSubstructMatch(core)
    match_prod = prod_mol.GetSubstructMatch(core)
    if not match_sub or not match_prod or len(match_sub) != len(match_prod):
        return set()

    reacting: Set[int] = set(range(n)) - set(match_sub)  # atoms outside the conserved core
    for s_idx, p_idx in zip(match_sub, match_prod):       # mapped atoms whose environment changed
        if _atom_env(sub_mol.GetAtomWithIdx(s_idx)) != _atom_env(prod_mol.GetAtomWithIdx(p_idx)):
            reacting.add(s_idx)

    expanded = set(reacting)
    for idx in reacting:
        for nb in sub_mol.GetAtomWithIdx(idx).GetNeighbors():
            expanded.add(nb.GetIdx())
    return expanded


def derive_som_labels(sub_smiles: str, met_smiles) -> Set[int]:
    """Union of reacting substrate atoms over one or more annotated metabolites."""
    sub_mol = Chem.MolFromSmiles(sub_smiles)
    if sub_mol is None:
        return set()
    mets: Iterable[str] = [met_smiles] if isinstance(met_smiles, str) else list(met_smiles)
    out: Set[int] = set()
    for met in mets:
        prod = Chem.MolFromSmiles(met)
        if prod is not None:
            out |= _reacting_atoms(sub_mol, prod)
    return out


def product_som_score(som_atoms: np.ndarray, sub_mol: Chem.Mol, prod_smiles: str, aggregation: str = "max") -> float:
    """SoM plausibility of the site at which ``prod_smiles`` differs from the substrate.

    Returns a neutral 1.0 when the site can't be localized (parse/MCS failure or identity),
    so an un-localizable candidate is neither boosted nor penalized relative to its raw
    generator score."""
    if som_atoms is None or len(som_atoms) == 0:
        return 1.0
    prod = Chem.MolFromSmiles(prod_smiles)
    if prod is None:
        return 1.0
    reacting = _reacting_atoms(sub_mol, prod)
    vals = [float(som_atoms[i]) for i in reacting if 0 <= i < len(som_atoms)]
    if not vals:
        return 1.0
    return max(vals) if aggregation == "max" else float(np.mean(vals))


class SoMPredictor(nn.Module):
    """Per-atom site-of-metabolism classifier sharing GRAIL's GNN backbone.

    Trained standalone on MCS-derived node labels (BCE). The encoder/head split is kept so
    the same head can later be folded into joint generator training (joint-ready hook)."""

    def __init__(
        self,
        in_channels: int = SINGLE_NODE_DIM,
        edge_dim: int = EDGE_DIM,
        hidden_dims: Sequence[int] = (64, 64),
        out_dim: int = 64,
        conv_kind: str = "gatv2",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.arch = {
            "in_channels": int(in_channels),
            "edge_dim": int(edge_dim),
            "hidden_dims": list(hidden_dims),
            "out_dim": int(out_dim),
            "conv_kind": conv_kind,
            "dropout": float(dropout),
        }
        self.encoder = GraphEncoder(in_channels, edge_dim, list(hidden_dims), out_dim, conv_kind=conv_kind, dropout=dropout)
        self.head = nn.Linear(out_dim, 1)
        self._cache: Dict[str, np.ndarray] = {}

    def node_logits(self, data: Data) -> torch.Tensor:
        return self.head(self.encoder.forward_nodes(data)).squeeze(-1)

    @torch.no_grad()
    def score_atoms(self, smiles: str) -> np.ndarray:
        """Per-atom SoM probabilities (sigmoid) for a substrate, indexed like its
        ``from_rdmol`` graph / SMILES atom order. Cached; inference-only."""
        cached = self._cache.get(smiles)
        if cached is not None:
            return cached
        mol = Chem.MolFromSmiles(smiles)
        data = from_rdmol(mol) if mol is not None else None
        if data is None or data.x.size(0) == 0:
            arr = np.zeros(0, dtype=np.float32)
            self._cache[smiles] = arr
            return arr
        was_training = self.training
        self.eval()
        logits = self.node_logits(data)
        if was_training:
            self.train()
        arr = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
        self._cache[smiles] = arr
        return arr


def build_som_dataset(molframe) -> List[Data]:
    """Build node-labelled single-graphs from a MolFrame's substrate->metabolites map.
    Each returned Data carries ``.y`` (per-atom 0/1 SoM label) for node-level BCE."""
    samples: List[Data] = []
    for sub, mets in molframe.map.items():
        mol = Chem.MolFromSmiles(sub)
        data = from_rdmol(mol) if mol is not None else None
        if data is None or data.x.size(0) == 0:
            continue
        labels = derive_som_labels(sub, mets)
        y = torch.zeros(data.x.size(0), dtype=torch.float32)
        for idx in labels:
            if 0 <= idx < y.size(0):
                y[idx] = 1.0
        data.y = y
        samples.append(data)
    return samples
