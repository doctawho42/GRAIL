"""Diversity, coverage, and mode-discovery metrics for set-valued generation.

These metrics evaluate a *set generator* (e.g. the Stage-2b Set-GFlowNet) on
how well it covers the space of true annotated metabolites and how diverse/
calibrated its produced sets are, independent of the pair-wise precision/
recall/F1 in ``grail_metabolism/metrics.py``.

Kept dependency-light and pure (RDKit only, no torch) so it can be imported
from evaluation scripts and notebooks without pulling in the model stack.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, List, Sequence, Set

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold


def modes_discovered(sampled_sets: Iterable[frozenset], annotated_ik: Set[str]) -> int:
    """Count distinct annotated InChIKeys found across ALL sampled sets.

    Takes the union of every set in ``sampled_sets`` (each a ``frozenset[str]``
    of InChIKeys) and intersects it with ``annotated_ik``, the ground-truth
    annotated InChIKeys for the substrate. This is a coverage/"mode discovery"
    metric: how many of the true metabolites did the generator find in *any*
    of its sampled sets, not just its best/first one.
    """
    union: Set[str] = set()
    for s in sampled_sets:
        union.update(s)
    return len(union & set(annotated_ik))


def mean_pairwise_tanimoto(smiles_list: Sequence[str]) -> float:
    """Mean Tanimoto similarity (Morgan/ECFP4 fingerprints) over unordered pairs.

    Unparseable SMILES (``Chem.MolFromSmiles(s) is None``) are skipped rather
    than raising. With fewer than 2 parseable molecules there are no pairs to
    average over; by convention (matching the "identical molecules -> 1.0"
    intuition for a degenerate/singleton set) this returns 1.0.
    """
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    mols = [m for m in mols if m is not None]

    if len(mols) < 2:
        return 1.0

    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048) for m in mols]

    sims: List[float] = []
    for fp_a, fp_b in combinations(fps, 2):
        sims.append(DataStructs.TanimotoSimilarity(fp_a, fp_b))

    return sum(sims) / len(sims)


def n_unique_scaffolds(smiles_list: Sequence[str]) -> int:
    """Number of distinct Bemis-Murcko scaffolds among the given SMILES.

    Unparseable SMILES are skipped. An empty (or all-unparseable) input
    returns 0.
    """
    scaffolds: Set[str] = set()
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        scaffolds.add(scaffold)
    return len(scaffolds)


def set_size_calibration(sampled_sets: Iterable[frozenset], annotated_ik: Set[str]) -> float:
    """Mean sampled set size minus the true annotated-set size.

    Positive values mean the generator over-produces relative to the true
    number of annotated metabolites; negative values mean it under-produces.
    An empty ``sampled_sets`` iterable is treated as a mean size of 0.
    """
    sets = list(sampled_sets)
    mean_size = sum(len(s) for s in sets) / len(sets) if sets else 0.0
    return mean_size - len(annotated_ik)
