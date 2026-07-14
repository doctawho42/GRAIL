"""Dense `(type, site)` training-label dataset builder.

Replaces GRAIL's 7,581-way PU rule label with a DENSE target over the coarse reaction-type
vocabulary built by `model/reaction_types.py`. Each train substrate becomes a graph (via
`utils/transform.from_rdmol`) carrying two labels:

  - `.y_type`: `FloatTensor[num_types]` multi-hot over reaction types that yield a true,
    annotated metabolite for this substrate.
  - `.y_site`: `FloatTensor[num_atoms]` per-atom reacting-atom label (from
    `model/som.py:derive_som_labels`), independent of `catalog`/`rule_to_type`.

Type labels come from the mining catalog's `source_pairs` (the exact train (substrate,
product) pairs each SMIRKS was mined from), NOT a full-bank RDKit re-apply -- reusing that
recorded provenance avoids re-running the ~90-minute ceiling pass.

See `.superpowers/sdd/task-2-brief.md`.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import torch
from rdkit import Chem
from torch_geometric.data import Data

from ..utils.preparation import MolFrame, _standardize_smiles_cached
from ..utils.transform import from_rdmol
from .som import derive_som_labels

__all__ = ["build_factorized_dataset"]

_DEFAULT_CATALOG_PATH = Path(__file__).resolve().parents[2] / "results" / "mined_rule_catalog_v2.json"


def _load_default_catalog() -> dict:
    with open(_DEFAULT_CATALOG_PATH, "r") as f:
        return json.load(f)


def _build_sub_to_types(catalog: dict, rule_to_type: Dict[str, int]) -> Dict[str, Set[int]]:
    """`{standardized_sub_smiles: {type_id, ...}}` from the catalog's `source_pairs`.

    Catalog `source_pairs` hold RAW SMILES from the clean train SDF; `MolFrame` standardizes
    its substrate keys via `_standardize_smiles_cached`. Standardizing here is what lets the
    keys line up with `molframe.map` -- without it every `y_type` would be silently all-zero.
    """
    sub_to_types: Dict[str, Set[int]] = defaultdict(set)
    for smirks, entry in catalog.items():
        t = rule_to_type.get(smirks, -1)
        if t < 0:
            continue
        for sub_smi, _prod_smi in entry.get("source_pairs", []):
            try:
                key = _standardize_smiles_cached(sub_smi)
            except Exception:
                continue
            sub_to_types[key].add(t)
    return sub_to_types


def build_factorized_dataset(
    molframe: MolFrame,
    rule_to_type: Dict[str, int],
    catalog: Optional[dict] = None,
) -> List[Data]:
    """Build the dense `(type, site)`-labelled dataset from a `MolFrame`.

    Args:
        molframe: source of substrate -> true-metabolite pairs (`.map`), with standardized
            substrate keys (the default `MolFrame(standartize=True)`).
        rule_to_type: `{smirks: type_id}` (Task 1's `coarse_type_vocab.json["rule_to_type"]`);
            `type_id == -1` (rare/unparseable) rules are excluded from type labels.
        catalog: `{smirks: {"count", "source_pairs": [[sub_smi, prod_smi], ...], ...}}`
            (Task's mining catalog). When `None`, loaded from
            `results/mined_rule_catalog_v2.json`.

    Returns:
        One `Data` (from `from_rdmol`) per substrate with `.map`-derivable atoms, each
        carrying `.y_type: FloatTensor[num_types]` and `.y_site: FloatTensor[num_atoms]`.
    """
    if catalog is None:
        catalog = _load_default_catalog()

    sub_to_types = _build_sub_to_types(catalog, rule_to_type)
    num_types = 1 + max(rule_to_type.values())

    dataset: List[Data] = []
    for sub, mets in molframe.map.items():
        mol = Chem.MolFromSmiles(sub)
        data = from_rdmol(mol) if mol is not None else None
        if data is None or data.x.size(0) == 0:
            continue

        y_type = torch.zeros(num_types, dtype=torch.float32)
        for t in sub_to_types.get(sub, ()):
            if 0 <= t < num_types:
                y_type[t] = 1.0

        site_labels = derive_som_labels(sub, mets)
        y_site = torch.zeros(data.x.size(0), dtype=torch.float32)
        for idx in site_labels:
            if 0 <= idx < y_site.size(0):
                y_site[idx] = 1.0

        data.y_type = y_type
        data.y_site = y_site
        dataset.append(data)

    return dataset
